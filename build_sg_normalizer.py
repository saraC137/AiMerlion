"""
build_sg_normalizer.py
======================
✨ FAIRY CODEMOTHER'S ONE-TIME DATA PREP EXTRAVAGANZA! ✨

Reads the Graduate Employment Survey CSV (NTU, NUS, SIT, SMU, SUSS, SUTD)
and distils it into `sg_normalizer.db` — a polished SQLite lookup database
used at runtime by SGInstitutionNormalizer.

Tables built
------------
  institutions       — 6 SG public universities + external partner schools
  institution_aliases— lowercased alias strings for fast O(1) lookup
  degrees            — every degree variant found in the CSV, grouped by base_name

Usage
-----
  python build_sg_normalizer.py --csv GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv
  python build_sg_normalizer.py --csv path/to/file.csv --db sg_normalizer.db --verbose

Re-run safe: INSERT OR IGNORE keeps the script idempotent.
"""

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  FAIRY CODEMOTHER'S CANONICAL INSTITUTION REGISTRY  💅
# ══════════════════════════════════════════════════════════════════

# (canonical_name, short_code, is_partner, country)
CORE_SG_UNIVERSITIES = [
    ("Nanyang Technological University",                  "NTU",  False, "SG"),
    ("National University of Singapore",                  "NUS",  False, "SG"),
    ("Singapore Institute of Technology",                 "SIT",  False, "SG"),
    ("Singapore Management University",                   "SMU",  False, "SG"),
    ("Singapore University of Social Sciences",           "SUSS", False, "SG"),
    ("Singapore University of Technology and Design",     "SUTD", False, "SG"),
]

# External partner institutions found in the SIT programmes
PARTNER_INSTITUTIONS = [
    ("Culinary Institute of America",      "CIA",       True, "US"),
    ("DigiPen Institute of Technology",    "DigiPen",   True, "US"),
    ("Newcastle University",               "Newcastle", True, "UK"),
    ("Massey University",                  "Massey",    True, "NZ"),
    ("Trinity College Dublin",             "TCD",       True, "IE"),
    ("Technische Universität München",     "TUM",       True, "DE"),
    ("The Glasgow School of Art",          "GSA",       True, "UK"),
    ("University of Glasgow",              "UofG",      True, "UK"),
    ("University of Liverpool",            "UoL",       True, "UK"),
    ("University of Manchester",           "UoM",       True, "UK"),
    ("University of Nevada, Las Vegas",    "UNLV",      True, "US"),
    ("Wheelock College",                   "Wheelock",  True, "US"),
]

# ── Alias seeds per short_code ────────────────────────────────────
ALIAS_SEEDS: dict[str, list[str]] = {
    "NTU": [
        "ntu",
        "nanyang tech",
        "nanyang technological university",
        "nanyang technological u",
        "nanyang university",
    ],
    "NUS": [
        "nus",
        "national university of singapore",
        "national u",
        "national university",
        "national uni of singapore",
    ],
    "SIT": [
        "sit",
        "singapore institute of technology",
        "singapore inst of technology",
        "singapore inst. of technology",
    ],
    "SMU": [
        "smu",
        "singapore management university",
        "singapore management u",
        "singapore mgmt university",
    ],
    "SUSS": [
        "suss",
        "singapore university of social sciences",
        "singapore uni of social sciences",
        "unisim",                        # legacy name
    ],
    "SUTD": [
        "sutd",
        "singapore university of technology and design",
        "singapore uni of technology and design",
    ],
    "CIA": [
        "culinary institute of america",
        "the culinary institute of america",
        "cia",
    ],
    "DigiPen": [
        "digipen",
        "digipen institute of technology",
        "sit- digipen institute of technology",
        "sit-digipen institute of technology",
    ],
    "Newcastle": [
        "newcastle university",
        "newcastle uni",
        "sit- newcastle university",
        "sit-newcastle university",
    ],
    "Massey": [
        "massey university",
        "massey uni",
        "sit- massey university",
        "sit-massey university",
    ],
    "TCD": [
        "trinity college dublin",
        "tcd",
        "sit-trinity college dublin",
        "sit- trinity college dublin",
        "singapore institute of technology -trinity college dublin",
        "trinity college dublin / singapore institute of technology-trinity college dublin",
    ],
    "TUM": [
        "technische universität münchen",
        "technische universitat munchen",
        "tu münchen",
        "tum",
        "sit- technische universitat munchen",
    ],
    "GSA": [
        "the glasgow school of art",
        "glasgow school of art",
        "gsa",
    ],
    "UofG": [
        "university of glasgow",
        "glasgow university",
        "sit-university of glasgow",
    ],
    "UoL": [
        "university of liverpool",
        "liverpool university",
    ],
    "UoM": [
        "university of manchester",
        "manchester university",
    ],
    "UNLV": [
        "university of nevada, las vegas",
        "university of nevada las vegas",
        "unlv",
    ],
    "Wheelock": [
        "wheelock college",
        "wheelock",
    ],
}

# ── CSV school name → canonical institution short_code ────────────
# Handles SIT-prefixed composite names from the CSV
CSV_SCHOOL_TO_SHORTCODE: dict[str, str] = {
    # SIT partner variants (normalise to the partner institution)
    "SIT- DigiPen Institute of Technology":   "DigiPen",
    "SIT-DigiPen Institute of Technology":    "DigiPen",
    "SIT- Newcastle University":              "Newcastle",
    "SIT-Newcastle University":               "Newcastle",
    "SIT- Massey University":                 "Massey",
    "SIT-Massey University":                  "Massey",
    "SIT- Technische Universitat Munchen":    "TUM",
    "SIT-Trinity College Dublin":             "TCD",
    "SIT- Trinity College Dublin / Trinity College Dublin": "TCD",
    "SIT-Trinity College Dublin / Trinity College Dublin":  "TCD",
    "SIT / SIT-Trinity College Dublin / Trinity College Dublin": "TCD",
    "Singapore Institute of Technology -Trinity College Dublin": "TCD",
    "Singapore Institute of Technology -Trinity College Dublin / Trinity College Dublin": "TCD",
    "Trinity College Dublin / Singapore Institute of Technology-Trinity College Dublin": "TCD",
    "Culinary Institute of America":          "CIA",
    "The Culinary Institute of America":      "CIA",
    "DigiPen Institute of Technology":        "DigiPen",
    "Newcastle University":                   "Newcastle",
    "Technische Universität München":         "TUM",
    "The Glasgow School of Art":              "GSA",
    "University of Glasgow":                  "UofG",
    "University of Liverpool":                "UoL",
    "University of Manchester":               "UoM",
    "University of Nevada, Las Vegas":        "UNLV",
    "Wheelock College":                       "Wheelock",
    # SG university self-references in school column
    "Singapore Institute of Technology":      "SIT",
    "Singapore Institute of Technology (SIT)":"SIT",
    "Trinity College Dublin":                 "TCD",
}


# ══════════════════════════════════════════════════════════════════
#  DEGREE PARSING UTILITIES  🎓
# ══════════════════════════════════════════════════════════════════

def _extract_program_duration(degree: str) -> Optional[str]:
    """Pull '4-year' / '3-year' from any variant spelling."""
    m = re.search(r'(\d)[\s-]?yr[s]?\b', degree, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-year"
    m = re.search(r'(\d)[\s-]?year[s]?\b', degree, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-year"
    return None


def _has_honours(degree: str) -> bool:
    """True if degree name signals an honours / distinction level."""
    return bool(re.search(
        r'\b(?:honours?|hons|cum laude)\b',
        degree, re.IGNORECASE
    ))


def _compute_base_name(degree: str) -> str:
    """
    Strip all decorators to get the plain base programme name.

    'Accountancy (4-years programme) Cum Laude and above'  →  'Accountancy'
    'Bachelor of Arts with Honours in Interior Design'     →  'Bachelor of Arts'
    'Bachelor of Engineering with Honours in Aerospace Systems' → 'Bachelor of Engineering'
    'Computer Science ^'                                   →  'Computer Science'
    """
    s = degree.strip()

    # Remove trailing ^ / ^ marker (SIT disambiguation symbol)
    s = re.sub(r'\s*\^.*$', '', s).strip()

    # "Cum Laude and above" variants
    s = re.sub(
        r'\s*[\(\-]?\s*cum laude(?:\s+and\s+above)?\s*\)?',
        '', s, flags=re.IGNORECASE
    ).strip()

    # "with Honours in X" or "with Honours"
    s = re.sub(r'\s+with\s+honours.*$', '', s, flags=re.IGNORECASE).strip()

    # "(Honours Programme)" / "(Hons)"
    s = re.sub(
        r'\s*\(\s*(?:\d[\s-]?yr[s]?\s+direct\s+)?honours?\s*(?:programme)?\s*\)',
        '', s, flags=re.IGNORECASE
    ).strip()
    s = re.sub(r'\s*\(\s*hons\s*\)', '', s, flags=re.IGNORECASE).strip()

    # "(4-year programme)" / "(4-years programme)"
    s = re.sub(
        r'\s*\(\s*\d[\s-]?years?\s+programme\s*\)',
        '', s, flags=re.IGNORECASE
    ).strip()

    # "(3-yr direct Honours Programme)"
    s = re.sub(
        r'\s*\(\s*\d[\s-]?yr[s]?\s+direct\s+\w+\s+programme\s*\)',
        '', s, flags=re.IGNORECASE
    ).strip()

    # Trailing punctuation artifacts
    s = s.rstrip('*-').strip()

    return s or degree.strip()


# ══════════════════════════════════════════════════════════════════
#  DATABASE DDL
# ══════════════════════════════════════════════════════════════════

DDL = """
CREATE TABLE IF NOT EXISTS institutions (
    institution_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name       TEXT    NOT NULL UNIQUE,
    short_code           TEXT    NOT NULL,
    is_partner_institution INTEGER NOT NULL DEFAULT 0,
    country              TEXT    NOT NULL DEFAULT 'SG'
);

CREATE TABLE IF NOT EXISTS institution_aliases (
    alias_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    institution_id  INTEGER NOT NULL REFERENCES institutions(institution_id),
    alias_text      TEXT    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS degrees (
    degree_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name      TEXT    NOT NULL,
    base_name           TEXT    NOT NULL,
    institution_id      INTEGER NOT NULL REFERENCES institutions(institution_id),
    has_honours         INTEGER NOT NULL DEFAULT 0,
    program_duration    TEXT,
    full_original_name  TEXT    NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_aliases_text       ON institution_aliases(alias_text);
CREATE INDEX IF NOT EXISTS idx_degrees_base       ON degrees(base_name);
CREATE INDEX IF NOT EXISTS idx_degrees_inst       ON degrees(institution_id);
CREATE INDEX IF NOT EXISTS idx_degrees_canonical  ON degrees(canonical_name);
"""


# ══════════════════════════════════════════════════════════════════
#  BUILD HELPERS
# ══════════════════════════════════════════════════════════════════

def _box(msg: str, width: int = 66) -> str:
    inner = f"  {msg}  "
    pad = width - len(inner)
    inner = inner + " " * max(pad, 0)
    return (
        f"╔{'═' * width}╗\n"
        f"║{inner}║\n"
        f"╚{'═' * width}╝"
    )


def _seed_institutions(cur: sqlite3.Cursor, verbose: bool) -> dict[str, int]:
    """Insert all SG universities + partner institutions. Returns short_code → id."""
    all_insts = CORE_SG_UNIVERSITIES + PARTNER_INSTITUTIONS
    code_to_id: dict[str, int] = {}

    for canonical, code, is_partner, country in all_insts:
        cur.execute(
            "INSERT OR IGNORE INTO institutions "
            "(canonical_name, short_code, is_partner_institution, country) "
            "VALUES (?, ?, ?, ?)",
            (canonical, code, int(is_partner), country),
        )
        cur.execute(
            "SELECT institution_id FROM institutions WHERE short_code = ?", (code,)
        )
        row = cur.fetchone()
        code_to_id[code] = row[0]
        if verbose:
            print(f"    {'🤝' if is_partner else '🏛️ '} [{code:10s}] {canonical}")

    return code_to_id


def _seed_aliases(
    cur: sqlite3.Cursor,
    code_to_id: dict[str, int],
    verbose: bool,
) -> None:
    """Seed institution_aliases from ALIAS_SEEDS."""
    total = 0
    for code, aliases in ALIAS_SEEDS.items():
        inst_id = code_to_id.get(code)
        if inst_id is None:
            continue
        for alias in aliases:
            cur.execute(
                "INSERT OR IGNORE INTO institution_aliases (institution_id, alias_text) "
                "VALUES (?, ?)",
                (inst_id, alias.lower().strip()),
            )
            total += 1
    if verbose:
        print(f"    💬 {total} alias rows seeded")


def _infer_institution_id(
    university_name: str,
    school_name: str,
    code_to_id: dict[str, int],
) -> int:
    """Map a (university, school) CSV row to an institution_id."""
    # Try CSV_SCHOOL_TO_SHORTCODE first (handles SIT-partner variants)
    school_code = CSV_SCHOOL_TO_SHORTCODE.get(school_name)
    if school_code and school_code in code_to_id:
        return code_to_id[school_code]

    # Fall back to university column
    UNI_TO_CODE = {
        "Nanyang Technological University":              "NTU",
        "National University of Singapore":              "NUS",
        "Singapore Institute of Technology":             "SIT",
        "Singapore Management University":               "SMU",
        "Singapore University of Social Sciences":       "SUSS",
        "Singapore University of Technology and Design": "SUTD",
    }
    code = UNI_TO_CODE.get(university_name)
    if code and code in code_to_id:
        return code_to_id[code]

    # Last resort: NTU
    return code_to_id["NTU"]


def _seed_degrees(
    cur: sqlite3.Cursor,
    csv_path: Path,
    code_to_id: dict[str, int],
    verbose: bool,
) -> int:
    """Parse the CSV and insert every unique degree variant."""
    inserted = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            university = (row.get("university") or "").strip()
            school     = (row.get("school")     or "").strip()
            degree_raw = (row.get("degree")     or "").strip()

            if not degree_raw:
                continue

            inst_id      = _infer_institution_id(university, school, code_to_id)
            base         = _compute_base_name(degree_raw)
            honours      = int(_has_honours(degree_raw))
            duration     = _extract_program_duration(degree_raw)
            # canonical_name = cleaned-up base (title-cased normalisation skipped
            # to keep the exact programme name as extracted from the CSV)
            canonical    = base

            cur.execute(
                "INSERT OR IGNORE INTO degrees "
                "(canonical_name, base_name, institution_id, has_honours, "
                " program_duration, full_original_name) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (canonical, base, inst_id, honours, duration, degree_raw),
            )
            if cur.rowcount:
                inserted += 1
                if verbose:
                    print(f"    🎓 [{base[:45]:<45s}] honours={honours} dur={duration}")

    return inserted


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def build(csv_path: Path, db_path: Path, verbose: bool) -> None:
    print()
    print(_box("✨  SG NORMALIZER DB BUILD  ✨  by Fairy Codemother"))
    print()

    if not csv_path.exists():
        print(f"❌  CSV not found: {csv_path}")
        sys.exit(1)

    print(f"📂  CSV  : {csv_path}")
    print(f"🗄️   DB   : {db_path}")
    print()

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    cur = con.cursor()

    # ── Schema ──────────────────────────────────────────────────
    print("┌─ Step 1 ─ Schema ─────────────────────────────────────┐")
    for stmt in DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)
    print("│  ✅ Tables & indexes ready                            │")
    print("└───────────────────────────────────────────────────────┘")
    print()

    # ── Institutions ────────────────────────────────────────────
    print("┌─ Step 2 ─ Institutions ───────────────────────────────┐")
    code_to_id = _seed_institutions(cur, verbose)
    print(f"│  ✅ {len(code_to_id)} institutions seeded               "
          f"{'':>10}│")
    print("└───────────────────────────────────────────────────────┘")
    print()

    # ── Aliases ─────────────────────────────────────────────────
    print("┌─ Step 3 ─ Aliases ────────────────────────────────────┐")
    _seed_aliases(cur, code_to_id, verbose)
    cur.execute("SELECT COUNT(*) FROM institution_aliases")
    alias_count = cur.fetchone()[0]
    print(f"│  ✅ {alias_count} aliases in table                    "
          f"{'':>12}│")
    print("└───────────────────────────────────────────────────────┘")
    print()

    # ── Degrees ─────────────────────────────────────────────────
    print("┌─ Step 4 ─ Degrees ────────────────────────────────────┐")
    n_degrees = _seed_degrees(cur, csv_path, code_to_id, verbose)
    cur.execute("SELECT COUNT(*) FROM degrees")
    total_deg = cur.fetchone()[0]
    print(f"│  ✅ {n_degrees} new rows inserted  ({total_deg} total) "
          f"{'':>8}│")
    print("└───────────────────────────────────────────────────────┘")
    print()

    con.commit()
    con.close()

    print(_box(f"🎉  Build complete!  DB → {db_path.name}"))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build sg_normalizer.db from the Graduate Employment Survey CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv"),
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("sg_normalizer.db"),
        help="Output SQLite database path (default: sg_normalizer.db).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print every row as it is inserted.",
    )
    args = parser.parse_args()
    build(args.csv, args.db, args.verbose)
