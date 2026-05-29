"""
sg_normalizer.py
================
✨ FAIRY CODEMOTHER'S RUNTIME NORMALIZER — The Velvet Rope of Education Data! ✨

Given the messy, inconsistent institution / degree strings that come out of the
TGone Ollama extractor, this module maps them to canonical names drawn from
the Graduate Employment Survey reference dataset.

Quick start
-----------
    from sg_normalizer import SGInstitutionNormalizer

    norm = SGInstitutionNormalizer("sg_normalizer.db")

    result = norm.normalize_institution("NTU")
    # → {"canonical_name": "Nanyang Technological University",
    #    "short_code": "NTU", "is_partner": False,
    #    "confidence": 0.95, "matched_alias": "ntu"}

    result = norm.normalize_degree("Bachelor of Engineering (Hons) in CS", "NTU")
    # → {"canonical_name": "Bachelor of Engineering",
    #    "base_name": "Bachelor of Engineering",
    #    "has_honours": True, "confidence": 0.95, "institution": {...}}

Graceful degradation
--------------------
If `sg_normalizer.db` is missing the class logs a warning and every
`normalize_*` call returns ``None`` — the extraction pipeline continues
unaffected.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process as rf_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("⚠️  rapidfuzz not installed — fuzzy matching disabled.")


# ══════════════════════════════════════════════════════════════════
#  INTERNAL TYPE ALIASES
# ══════════════════════════════════════════════════════════════════

InstitutionResult = Optional[dict]   # {"canonical_name", "short_code", ...}
DegreeResult      = Optional[dict]   # {"canonical_name", "base_name", ...}

_FUZZY_THRESHOLD = 85   # minimum score (0–100) to accept a fuzzy match


# ══════════════════════════════════════════════════════════════════
#  DEGREE PARSING HELPERS  (duplicated from build_sg_normalizer so
#  that sg_normalizer.py is self-contained at runtime)
# ══════════════════════════════════════════════════════════════════

def _has_honours(text: str) -> bool:
    return bool(re.search(r'\b(?:honours?|hons|cum laude)\b', text, re.IGNORECASE))


def _extract_program_duration(text: str) -> Optional[str]:
    m = re.search(r'(\d)[\s-]?yr[s]?\b', text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-year"
    m = re.search(r'(\d)[\s-]?year[s]?\b', text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-year"
    return None


def _strip_honours_and_duration(text: str) -> str:
    """
    Return the base degree name after removing honours / duration decorators.
    Mirrors the logic in build_sg_normalizer._compute_base_name.
    """
    s = text.strip()
    s = re.sub(r'\s*\^.*$', '', s).strip()
    s = re.sub(
        r'\s*[\(\-]?\s*cum laude(?:\s+and\s+above)?\s*\)?',
        '', s, flags=re.IGNORECASE
    ).strip()
    s = re.sub(r'\s+with\s+honours.*$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(
        r'\s*\(\s*(?:\d[\s-]?yr[s]?\s+direct\s+)?honours?\s*(?:programme)?\s*\)',
        '', s, flags=re.IGNORECASE
    ).strip()
    s = re.sub(r'\s*\(\s*hons\s*\)', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(
        r'\s*\(\s*\d[\s-]?years?\s+programme\s*\)',
        '', s, flags=re.IGNORECASE
    ).strip()
    s = re.sub(
        r'\s*\(\s*\d[\s-]?yr[s]?\s+direct\s+\w+\s+programme\s*\)',
        '', s, flags=re.IGNORECASE
    ).strip()
    s = s.rstrip('*-').strip()
    return s or text.strip()


def _normalize_lookup_key(raw: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for lookup."""
    s = raw.lower().strip()
    s = re.sub(r'[.\-,;:()/]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ══════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ══════════════════════════════════════════════════════════════════

class SGInstitutionNormalizer:
    """
    ✨ The Velvet-Rope Bouncer of Singapore Education Data! ✨

    Normalises raw institution and degree strings extracted from resumes into
    canonical names backed by the Graduate Employment Survey reference DB.

    Attributes
    ----------
    db_path : Path
        Path to `sg_normalizer.db` built by `build_sg_normalizer.py`.
    available : bool
        False if the DB is missing — every method returns None gracefully.
    """

    def __init__(self, db_path: str | Path = "sg_normalizer.db") -> None:
        self.db_path   = Path(db_path)
        self.available = False

        # In-memory caches loaded on init
        self._alias_to_inst:  dict[str, dict] = {}   # alias_text → institution row
        self._inst_by_id:     dict[int, dict] = {}   # institution_id → row
        self._inst_canonicals: dict[str, dict] = {}  # lowercase canonical → row
        self._degrees:        list[dict]       = []  # all degree rows
        self._degree_base_index: dict[str, list[dict]] = {}  # base_name.lower() → rows

        if not self.db_path.exists():
            logger.warning(
                "⚠️  sg_normalizer.db not found at '%s'. "
                "Run build_sg_normalizer.py first. Normalisation disabled.",
                self.db_path,
            )
            return

        try:
            self._load_cache()
            self.available = True
            logger.info(
                "✅  SGInstitutionNormalizer ready — %d institutions, %d aliases, %d degrees",
                len(self._inst_by_id),
                len(self._alias_to_inst),
                len(self._degrees),
            )
        except Exception as exc:
            logger.error("❌  Failed to load sg_normalizer.db: %s", exc)

    # ── Cache loader ────────────────────────────────────────────

    def _load_cache(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row

        # Institutions
        for row in con.execute("SELECT * FROM institutions"):
            d = dict(row)
            self._inst_by_id[d["institution_id"]] = d
            self._inst_canonicals[d["canonical_name"].lower()] = d

        # Aliases  →  institution dict
        for row in con.execute(
            "SELECT a.alias_text, i.* "
            "FROM institution_aliases a "
            "JOIN institutions i USING (institution_id)"
        ):
            d = dict(row)
            self._alias_to_inst[d["alias_text"]] = d

        # Degrees
        for row in con.execute("SELECT * FROM degrees"):
            d = dict(row)
            self._degrees.append(d)
            key = d["base_name"].lower()
            self._degree_base_index.setdefault(key, []).append(d)

        con.close()

    # ── Institution helpers ─────────────────────────────────────

    def _inst_dict_to_result(
        self, inst: dict, confidence: float, matched_alias: str = ""
    ) -> InstitutionResult:
        return {
            "canonical_name": inst["canonical_name"],
            "short_code":     inst["short_code"],
            "is_partner":     bool(inst["is_partner_institution"]),
            "country":        inst.get("country", "SG"),
            "confidence":     round(confidence, 4),
            "matched_alias":  matched_alias,
        }

    # ── Public API ──────────────────────────────────────────────

    def normalize_institution(self, raw_text: str) -> InstitutionResult:
        """
        ✨ Map a raw institution string to its canonical form. ✨

        Strategy
        --------
        1. Exact canonical match              → confidence 1.00
        2. Exact alias match (case-insensitive stripped) → confidence 0.95
        3. Fuzzy match via rapidfuzz token_sort_ratio    → confidence score/100
        4. No match                           → None

        Parameters
        ----------
        raw_text : str
            Raw institution name from a resume (e.g. "NTU", "Nanyang Tech").

        Returns
        -------
        dict | None
            Keys: canonical_name, short_code, is_partner, country,
                  confidence, matched_alias.
        """
        if not self.available:
            return None
        if not raw_text or not raw_text.strip():
            return None

        raw_text = raw_text.strip()

        # 1. Exact canonical name match
        lower_raw = raw_text.lower()
        if lower_raw in self._inst_canonicals:
            inst = self._inst_canonicals[lower_raw]
            return self._inst_dict_to_result(inst, 1.0, raw_text)

        # 2. Exact alias match (normalised key)
        key = _normalize_lookup_key(raw_text)
        if key in self._alias_to_inst:
            inst = self._alias_to_inst[key]
            return self._inst_dict_to_result(inst, 0.95, key)

        # 2b. Direct alias lookup without extra normalisation
        if lower_raw in self._alias_to_inst:
            inst = self._alias_to_inst[lower_raw]
            return self._inst_dict_to_result(inst, 0.95, lower_raw)

        # 3. Fuzzy match
        if _RAPIDFUZZ_AVAILABLE:
            choices = list(self._alias_to_inst.keys()) + list(self._inst_canonicals.keys())
            match = rf_process.extractOne(
                key,
                choices,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=_FUZZY_THRESHOLD,
            )
            if match:
                matched_str, score, _ = match
                inst = (
                    self._alias_to_inst.get(matched_str)
                    or self._inst_canonicals.get(matched_str)
                )
                if inst:
                    return self._inst_dict_to_result(inst, score / 100, matched_str)

        return None

    def normalize_degree(
        self,
        raw_text: str,
        institution_hint: Optional[str] = None,
    ) -> DegreeResult:
        """
        ✨ Map a raw degree string to its canonical programme name. ✨

        Strategy
        --------
        1. Strip honours/duration decorators to get the base candidate.
        2. Exact match against degree base_name index.
        3. If institution_hint given, prefer rows that match that institution.
        4. Fuzzy match if no exact hit.

        Parameters
        ----------
        raw_text : str
            Raw degree string (e.g. "B.Eng. (Hons) Computer Engineering").
        institution_hint : str, optional
            Short code or partial canonical name (e.g. "NTU", "NUS")
            to narrow down matches when the same degree exists at multiple unis.

        Returns
        -------
        dict | None
            Keys: canonical_name, base_name, has_honours, program_duration,
                  full_original_name, confidence, institution.
        """
        if not self.available:
            return None
        if not raw_text or not raw_text.strip():
            return None

        raw_text    = raw_text.strip()
        has_hon     = _has_honours(raw_text)
        duration    = _extract_program_duration(raw_text)
        base_query  = _strip_honours_and_duration(raw_text)
        base_key    = base_query.lower()
        lookup_key  = _normalize_lookup_key(base_query)

        # ── Resolve institution hint to an id set ────────────────
        hint_ids: set[int] = set()
        if institution_hint:
            inst_result = self.normalize_institution(institution_hint)
            if inst_result:
                # find id
                for iid, inst in self._inst_by_id.items():
                    if inst["short_code"] == inst_result["short_code"]:
                        hint_ids.add(iid)
                        break

        # ── Helper to pick best row when multiple exist ──────────
        def _pick(rows: list[dict]) -> dict:
            if not rows:
                return {}
            if hint_ids:
                preferred = [r for r in rows if r["institution_id"] in hint_ids]
                if preferred:
                    return preferred[0]
            return rows[0]

        def _build_degree_result(row: dict, confidence: float) -> DegreeResult:
            inst = self._inst_by_id.get(row["institution_id"])
            inst_out = (
                {
                    "canonical_name": inst["canonical_name"],
                    "short_code":     inst["short_code"],
                    "is_partner":     bool(inst["is_partner_institution"]),
                }
                if inst else None
            )
            return {
                "canonical_name":    row["canonical_name"],
                "base_name":         row["base_name"],
                "has_honours":       has_hon,
                "program_duration":  duration,
                "full_original_name": row["full_original_name"],
                "confidence":        round(confidence, 4),
                "institution":       inst_out,
            }

        # 1. Exact base_name match
        if base_key in self._degree_base_index:
            row = _pick(self._degree_base_index[base_key])
            return _build_degree_result(row, 0.95)

        # 2. Normalised-key match
        for stored_key, rows in self._degree_base_index.items():
            if _normalize_lookup_key(stored_key) == lookup_key:
                row = _pick(rows)
                return _build_degree_result(row, 0.90)

        # 3. Fuzzy match
        if _RAPIDFUZZ_AVAILABLE:
            choices = list(self._degree_base_index.keys())
            match = rf_process.extractOne(
                lookup_key,
                choices,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=_FUZZY_THRESHOLD,
            )
            if match:
                matched_key, score, _ = match
                row = _pick(self._degree_base_index[matched_key])
                return _build_degree_result(row, score / 100)

        # 4. Partial match fallback — if raw base is a substring of a known degree
        for stored_key, rows in self._degree_base_index.items():
            if base_key in stored_key or stored_key in base_key:
                row = _pick(rows)
                return _build_degree_result(row, 0.70)

        return None

    def bulk_normalize_education(
        self, records: list[dict]
    ) -> list[dict]:
        """
        ✨ Batch-normalise a list of education dicts from the extractor. ✨

        Each input record must have at least one of:
          ``{"institution": str, "degree": str}``

        The method adds two keys in-place and returns the same list:
          ``institution_normalized`` — result of normalize_institution()
          ``degree_normalized``      — result of normalize_degree()

        Parameters
        ----------
        records : list[dict]
            Education entries as returned by _extract_education_regex.

        Returns
        -------
        list[dict]
            Same list with normalisation results attached.
        """
        if not self.available:
            for rec in records:
                rec["institution_normalized"] = None
                rec["degree_normalized"]      = None
            return records

        for rec in records:
            raw_inst   = rec.get("institution") or ""
            raw_degree = rec.get("degree")      or ""

            inst_result = self.normalize_institution(raw_inst)
            hint = inst_result["short_code"] if inst_result else None

            rec["institution_normalized"] = inst_result
            rec["degree_normalized"]      = self.normalize_degree(raw_degree, hint)

        return records
