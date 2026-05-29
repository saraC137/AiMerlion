"""
import_kaggle_ner.py

💅✨ FAIRY CODEMOTHER'S KAGGLE/HUGGINGFACE NER IMPORTER ✨💅

Converts external NER datasets into AiMerlion's SQLite database format
so train_ner.py can see them alongside your manual annotations.

Think of it like tailoring someone else's outfit to fit YOUR
measurements — same fabric, custom fit! 🎭👗

Supported sources:
  ┌──────────────────┬─────────────────────────────────────────────────┐
  │ --source          │ Dataset & Format                                │
  ├──────────────────┼─────────────────────────────────────────────────┤
  │ dataturks        │ DataTurks JSON (220 resumes, 10 entity types)   │
  │                  │ Also works for Mehyar IT CVs (same JSON format) │
  │ bio              │ yashpwrr BIO-tagged CSV (22K+ samples, 25 types)│
  │ classification   │ Bhawal CSV (2,484 resumes, 24 categories)       │
  └──────────────────┴─────────────────────────────────────────────────┘

Candidate ID Ranges (to avoid collision):
  Real data:       1 - 899,999
  DataTurks:       900,000 - 919,999
  Mehyar:          920,000 - 929,999  (use --source dataturks)
  BIO/yashpwrr:    930,000 - 949,999
  Majinuub:        950,000 - 959,999
  SkillSpan:       960,000+

Usage:
    # DataTurks (download traindata.json from GitHub first)
    python import_kaggle_ner.py --source dataturks --input traindata.json

    # Mehyar IT CVs (same span format as DataTurks)
    python import_kaggle_ner.py --source dataturks --input mehyar_annotated_cvs.json

    # yashpwrr BIO-tagged (download CSV from Kaggle first)
    python import_kaggle_ner.py --source bio --input resume_ner_dataset.csv

    # Classification (Bhawal resume categories)
    python import_kaggle_ner.py --source classification --input Resume.csv

    # Always dry-run first to preview!
    python import_kaggle_ner.py --source dataturks --input traindata.json --dry-run

    # Mehyar uses EXCLUSIVE end indices (unlike DataTurks INCLUSIVE)
    python import_kaggle_ner.py --source dataturks --input mehyar_annotated_cvs.json --exclusive-end

Dependencies:
    No extra dependencies! Uses only Python stdlib (json, csv, sqlite3).
"""

import os
import sys
import json
import csv
import sqlite3
import argparse
import logging
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# 🎨 CONSOLE — Pretty output matching AiMerlion's signature style
# =============================================================================

class Console:
    """Pretty console output matching the AiMerlion pipeline style."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def banner(text: str):
        width = 60
        print(f"\n{'=' * width}")
        print(f"  {Console.BOLD}{Console.HEADER}{text}{Console.RESET}")
        print(f"{'=' * width}")

    @staticmethod
    def success(msg: str):
        print(f"  {Console.GREEN}✅ {msg}{Console.RESET}")

    @staticmethod
    def warning(msg: str):
        print(f"  {Console.YELLOW}⚠️  {msg}{Console.RESET}")

    @staticmethod
    def error(msg: str):
        print(f"  {Console.RED}❌ {msg}{Console.RESET}")

    @staticmethod
    def info(msg: str):
        print(f"  {Console.CYAN}💡 {msg}{Console.RESET}")

    @staticmethod
    def stat(label: str, value: Any):
        print(f"    {Console.DIM}{label}:{Console.RESET} {Console.BOLD}{value}{Console.RESET}")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 💅 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 🗺️ ENTITY LABEL MAPPING
# External datasets use DIFFERENT label names than AiMerlion.
# This map translates their vocabulary to ours.
# Like a bilingual dictionary between two fashion houses! 👗📖
# =============================================================================

# DataTurks labels → AiMerlion entity types
# The DataTurks dataset uses these 10 label names:
DATATURKS_LABEL_MAP = {
    # ── Standard casing (as seen in traindata.json) ───────────────
    "Name":                "PERSON_NAME",
    "Email Address":       "EMAIL",
    "Phone":               "PHONE",
    "Location":            "LOCATION",
    "Designation":         "JOB_TITLE",
    "Companies worked at": "COMPANY",
    "Skills":              "SKILL",
    "College Name":        "INSTITUTION",
    "Degree":              "DEGREE",
    "Graduation Year":     "GRADUATION_YEAR",
    "Years of Experience": "YEARS_EXPERIENCE",
    # ── Upper-case variants (defensive — some files vary) ─────────
    "SKILLS":              "SKILL",
    "LOCATION":            "LOCATION",
    "NAME":                "PERSON_NAME",
    "EMAIL":               "EMAIL",
    "EMAIL ADDRESS":       "EMAIL",
    "DESIGNATION":         "JOB_TITLE",
    "COMPANIES WORKED AT": "COMPANY",
    "COLLEGE NAME":        "INSTITUTION",
    "DEGREE":              "DEGREE",
    "GRADUATION YEAR":     "GRADUATION_YEAR",
    "YEARS OF EXPERIENCE": "YEARS_EXPERIENCE",
    # ── Mehyar IT CVs use this label ──────────────────────────────
    "IT SKILL":            "SKILL",
    "IT_SKILL":            "SKILL",
    "it skill":            "SKILL",
    "SKILL":               "SKILL",
    "Skill":               "SKILL",
}

# yashpwrr BIO labels → AiMerlion entity types
# Their BIO tags look like: B-Skills, I-Skills, B-Name, etc.
# We strip the B-/I- prefix and map the base label.
BIO_LABEL_MAP = {
    "Name":                 "PERSON_NAME",
    "Email Address":        "EMAIL",
    "Phone":                "PHONE",
    "Location":             "LOCATION",
    "Designation":          "JOB_TITLE",
    "Companies worked at":  "COMPANY",
    "Companies Worked At":  "COMPANY",
    "Skills":               "SKILL",
    "College Name":         "INSTITUTION",
    "Degree":               "DEGREE",
    "Graduation Year":      "GRADUATION_YEAR",
    "Years of Experience":  "YEARS_EXPERIENCE",
    "CGPA":                 "CGPA",
    "Awards":               "AWARD",
    "Certifications":       "CERTIFICATION",
    "Projects":             "PROJECT",
    "Links":                "URL",
    "Languages":            "LANGUAGE",
    "Interests":            "INTEREST",
    "Summary":              "SUMMARY",
    "University":           "INSTITUTION",
    "Publications":         "PUBLICATION",
    "Courses":              "COURSE",
    "References":           "REFERENCE",
    "Address":              "LOCATION",
    "Objective":            "SUMMARY",
}

# Classification category mapping (Bhawal dataset → AiMerlion Function)
CLASSIFICATION_MAP = {
    "Information-Technology":  "IT",
    "HR":                      "HR",
    "Finance":                 "Finance",
    "Accountant":              "Finance",
    "Engineering":             "Engineering",
    "Teacher":                 "Education",
    "Healthcare":              "Healthcare",
    "Sales":                   "Sales",
    "Digital-Media":           "Marketing",
    "Business-Development":    "Business Development",
    "Aviation":                "Aviation",
    "Advocate":                "Legal",
    "Banking":                 "Banking",
    "Consultant":              "Consulting",
    "Public-Relations":        "Marketing",
    "Arts":                    "Arts",
    "Automation-Testing":      "IT",
    "Blockchain":              "IT",
    "BPO":                     "Operations",
    "Chef":                    "F&B",
    "Civil-Engineer":          "Engineering",
    "Construction":            "Engineering",
    "Database":                "IT",
    "Designer":                "Design",
    "DevOps-Engineer":         "IT",
    "DotNet-Developer":        "IT",
    "Electrical-Engineering":  "Engineering",
    "ETL-Developer":           "IT",
    "Fitness":                 "Others",
    "Hadoop":                  "IT",
    "Java-Developer":          "IT",
    "Mechanical-Engineer":     "Engineering",
    "Network-Security-Engineer": "IT",
    "Operations-Manager":      "Operations",
    "PMO":                     "Operations",
    "Python-Developer":        "IT",
    "React-Developer":         "IT",
    "SAP-Developer":           "IT",
    "Testing":                 "IT",
    "Web-Designing":           "IT",
}


# =============================================================================
# 🛠️ DATABASE HELPERS
# =============================================================================

def ensure_tables_exist(conn: sqlite3.Connection):
    """
    Make sure the required tables exist in the database.
    Uses IF NOT EXISTS so it won't break existing tables.
    Safety first, darling! 💅
    """
    # ── raw_extractions — stores the full resume text ─────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_extractions (
            candidate_id INTEGER PRIMARY KEY,
            raw_text TEXT,
            file_name TEXT DEFAULT '',
            folder_path TEXT DEFAULT '',
            filenames TEXT DEFAULT '',
            resume_language TEXT DEFAULT 'en',
            text_length INTEGER DEFAULT 0,
            extraction_timestamp TEXT
        )
    """)

    # ── ner_documents — one row per document ──────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ner_documents (
            doc_id TEXT PRIMARY KEY,
            candidate_id INTEGER,
            status TEXT DEFAULT 'completed',
            annotator TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            function TEXT DEFAULT '',
            industry TEXT DEFAULT ''
        )
    """)

    # ── ner_annotations — one row per entity span ─────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ner_annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            doc_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            text_content TEXT,
            layer INTEGER DEFAULT 0,
            confidence REAL DEFAULT 1.0,
            annotator TEXT DEFAULT 'kaggle_import',
            created_at TEXT,
            UNIQUE(doc_id, entity_type, char_start, char_end, layer)
        )
    """)
    conn.commit()


def get_next_candidate_id(conn: sqlite3.Connection, prefix: int = 900000) -> int:
    """
    Generate a unique candidate_id for imported data.

    ⚠️ CRITICAL: We use IDs starting at the given prefix to avoid
    collisions with your REAL candidates (which start from 1).
    Think of it as the VIP section — separate from general admission! 🎟️
    """
    row = conn.execute(
        "SELECT MAX(candidate_id) as max_id FROM ner_documents "
        "WHERE candidate_id >= ?",
        (prefix,)
    ).fetchone()

    if row and row[0] is not None:
        return row[0] + 1
    return prefix


# =============================================================================
# 📥 IMPORTER 1: DATATURKS JSON (+ Mehyar IT CVs)
#
# DataTurks format: Each line is a JSON object with:
#   {"content": "full resume text...",
#    "annotation": [{"label": ["Skills"], "points": [{"start": 100, "end": 120, "text": "Python"}]}]}
#
# Mehyar format: Similar but slightly different keys:
#   {"text": "full CV text...",
#    "entities": [{"start": 100, "end": 120, "label": "IT SKILL"}]}
#
# CRITICAL: DataTurks uses INCLUSIVE end indices (end char IS part of entity)
#           AiMerlion uses EXCLUSIVE end indices (end char is NOT part of entity)
#           So we add +1 to DataTurks end indices!
#           Mehyar uses EXCLUSIVE end indices already — use --exclusive-end flag!
# =============================================================================

def import_dataturks(
    input_path: str,
    db_path: str,
    dry_run: bool = False,
    exclusive_end: bool = False,
    candidate_id_prefix: int = 900000
) -> Dict[str, int]:
    """
    📥 Import DataTurks-format or Mehyar-format JSON into AiMerlion's DB.

    Handles TWO slightly different JSON structures:
      Format A (DataTurks): {"content": "...", "annotation": [{"label": [...], "points": [{"start":..., "end":..., "text":...}]}]}
      Format B (Mehyar):    {"text": "...", "entities": [{"start":..., "end":..., "label": "..."}]}

    Auto-detects which format based on the keys present in each record.

    Args:
        input_path:           Path to the downloaded JSON file
        db_path:              Path to resume_extractions.db
        dry_run:              If True, preview without writing to DB
        exclusive_end:        If True, end indices are already exclusive (Mehyar)
                              If False, end indices are inclusive (DataTurks, +1 needed)
        candidate_id_prefix:  Starting candidate_id range (default: 900,000)

    Returns:
        Dict with import statistics
    """
    Console.banner("📥 Importing DataTurks/Mehyar JSON")

    # ── Load and parse the JSON file ──────────────────────────────
    # DataTurks files are JSONL (one JSON object per line)
    # Some files might also be a JSON array
    records = []
    parse_errors = 0

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # ── Try JSON array first ──────────────────────────────────
        if content.startswith("["):
            try:
                records = json.loads(content)
                Console.info(f"Parsed as JSON array: {len(records)} records")
            except json.JSONDecodeError:
                pass

        # ── Fall back to JSONL (one object per line) ──────────────
        if not records:
            for line_num, line in enumerate(content.split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if parse_errors <= 5:
                        logger.warning(f"Line {line_num}: malformed JSON — {e}")
            Console.info(f"Parsed as JSONL: {len(records)} records")

    except FileNotFoundError:
        Console.error(f"File not found: {input_path}")
        return {"error": "File not found"}

    if parse_errors > 0:
        Console.warning(f"{parse_errors} malformed lines skipped")

    if not records:
        Console.error("No valid records found in file!")
        return {"imported": 0}

    # ── Connect to database ───────────────────────────────────────
    if not dry_run:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_tables_exist(conn)
    else:
        conn = None

    # ── Process each record ───────────────────────────────────────
    imported = 0
    skipped = 0
    total_annotations = 0
    entity_counter = Counter()
    now = datetime.now().isoformat()

    for idx, record in enumerate(records):
        # ── Auto-detect format ────────────────────────────────────
        # Format A (DataTurks): has "content" + "annotation"
        # Format B (Mehyar):    has "text" + "entities"
        if "content" in record:
            # ── FORMAT A: DataTurks ───────────────────────────────
            full_text = record.get("content", "")
            raw_annotations = record.get("annotation", [])

            if not full_text or not raw_annotations:
                skipped += 1
                continue

            # Parse DataTurks annotation structure
            spans = []
            for ann in raw_annotations:
                # DataTurks wraps label in a list: {"label": ["Skills"]}
                labels = ann.get("label", [])
                if isinstance(labels, list) and labels:
                    label = labels[0]
                elif isinstance(labels, str):
                    label = labels
                else:
                    continue

                points = ann.get("points", [])
                if not points:
                    continue

                for point in points:
                    start = point.get("start", 0)
                    end = point.get("end", 0)
                    text = point.get("text", "")

                    # ── Map label to AiMerlion entity type ────────
                    entity_type = DATATURKS_LABEL_MAP.get(label)
                    if not entity_type:
                        # Try case-insensitive lookup
                        entity_type = DATATURKS_LABEL_MAP.get(label.upper())
                    if not entity_type:
                        logger.debug(f"Unknown label '{label}' — skipping")
                        continue

                    # ── CRITICAL: DataTurks uses INCLUSIVE end! ───
                    # DataTurks: {"start": 0, "end": 12} = chars 0..12 (13 chars)
                    # AiMerlion: char_end is EXCLUSIVE = chars 0..12 needs end=13
                    if not exclusive_end:
                        end = end + 1

                    spans.append((entity_type, start, end, text))

        elif "text" in record:
            # ── FORMAT B: Mehyar ──────────────────────────────────
            full_text = record.get("text", "")
            raw_entities = record.get("entities", [])

            if not full_text or not raw_entities:
                skipped += 1
                continue

            spans = []
            for ent in raw_entities:
                start = ent.get("start", 0)
                end = ent.get("end", 0)
                label = ent.get("label", "")

                entity_type = DATATURKS_LABEL_MAP.get(label)
                if not entity_type:
                    entity_type = DATATURKS_LABEL_MAP.get(label.upper())
                if not entity_type:
                    entity_type = DATATURKS_LABEL_MAP.get(label.strip())
                if not entity_type:
                    logger.debug(f"Unknown label '{label}' — skipping")
                    continue

                # Mehyar uses EXCLUSIVE end already — but check flag
                if not exclusive_end:
                    end = end + 1

                # Extract text from the full content using offsets
                text = full_text[start:end].strip() if end <= len(full_text) else ""

                spans.append((entity_type, start, end, text))
        else:
            # Unknown format — skip
            skipped += 1
            continue

        if not spans:
            skipped += 1
            continue

        # ── Insert into database ──────────────────────────────────
        if dry_run:
            imported += 1
            for etype, _, _, _ in spans:
                entity_counter[etype] += 1
                total_annotations += 1
            if imported <= 3:
                Console.info(f"[DRY RUN] Record {idx}: {len(spans)} spans")
                for etype, s, e, t in spans[:5]:
                    Console.stat(f"  {etype}", f"[{s}:{e}] \"{t[:50]}\"")
            continue

        candidate_id = get_next_candidate_id(conn, candidate_id_prefix)
        doc_id = f"kaggle_dt_{idx:06d}"

        try:
            # Check for duplicate
            existing = conn.execute(
                "SELECT doc_id FROM ner_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing:
                skipped += 1
                continue

            # Insert raw text
            conn.execute(
                """INSERT OR IGNORE INTO raw_extractions
                   (candidate_id, raw_text, resume_language, text_length,
                    extraction_timestamp)
                   VALUES (?, ?, 'en', ?, ?)""",
                (candidate_id, full_text, len(full_text), now)
            )

            # Insert document record
            conn.execute(
                """INSERT INTO ner_documents
                   (doc_id, candidate_id, status, annotator, notes)
                   VALUES (?, ?, 'completed', 'kaggle_dataturks', ?)""",
                (doc_id, candidate_id,
                 json.dumps({"source": "dataturks", "file": os.path.basename(input_path)}))
            )

            # Insert annotation spans
            for entity_type, char_start, char_end, text in spans:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO ner_annotations
                           (candidate_id, doc_id, entity_type, char_start,
                            char_end, text_content, layer, confidence, annotator, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, 0, 0.9, 'kaggle_dataturks', ?)""",
                        (candidate_id, doc_id, entity_type,
                         char_start, char_end, text, now)
                    )
                    entity_counter[entity_type] += 1
                    total_annotations += 1
                except sqlite3.IntegrityError:
                    pass  # Duplicate span — skip silently

            imported += 1

            # Commit every 100 records for performance
            if imported % 100 == 0:
                conn.commit()
                Console.info(f"  Imported {imported} documents...")

        except sqlite3.Error as e:
            logger.warning(f"Error importing record {idx}: {e}")
            skipped += 1

    if conn:
        conn.commit()
        conn.close()

    # ── Print summary ─────────────────────────────────────────────
    Console.success(f"Imported: {imported} documents")
    Console.stat("Annotations", total_annotations)
    Console.stat("Skipped", skipped)

    if entity_counter:
        Console.info("Entity type distribution:")
        for etype, count in entity_counter.most_common():
            Console.stat(f"  {etype}", count)

    return {"imported": imported, "annotations": total_annotations, "skipped": skipped}


# =============================================================================
# 📥 IMPORTER 2: BIO-TAGGED CSV (yashpwrr)
#
# CSV format: token,tag
# Each row is one token with its BIO tag (B-Skills, I-Skills, O, etc.)
# Blank lines separate documents.
#
# We reconstruct full text by joining tokens with spaces, then
# convert BIO spans back to character offsets for AiMerlion.
# =============================================================================

def import_bio_tagged(
    input_path: str,
    db_path: str,
    dry_run: bool = False,
    candidate_id_prefix: int = 930000
) -> Dict[str, int]:
    """
    📥 Import BIO-tagged CSV data (yashpwrr format) into AiMerlion's DB.

    CSV structure:
        token,tag
        John,B-Name
        Doe,I-Name
        is,O
        a,O
        (blank line = new document)

    The importer:
    1. Reads token+tag pairs from CSV
    2. Groups by document (blank line boundaries)
    3. Reconstructs full text by joining tokens with spaces
    4. Converts BIO spans to character offsets
    5. Maps BIO labels → AiMerlion entity types
    6. Inserts into SQLite

    Args:
        input_path:           Path to the BIO-tagged CSV file
        db_path:              Path to resume_extractions.db
        dry_run:              If True, preview without writing to DB
        candidate_id_prefix:  Starting candidate_id range (default: 930,000)

    Returns:
        Dict with import statistics
    """
    Console.banner("📥 Importing BIO-tagged CSV")

    # ── Read and group into documents ─────────────────────────────
    documents = []  # List of (tokens_list, tags_list) tuples
    current_tokens = []
    current_tags = []

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)

            # Skip header if present
            header = next(reader, None)
            if header and len(header) >= 2:
                # Check if this is actually a data row (not header)
                if header[1] in ("O",) or header[1].startswith("B-") or header[1].startswith("I-"):
                    current_tokens.append(header[0])
                    current_tags.append(header[1])

            for row_num, row in enumerate(reader, 2):
                # ── Blank line = document boundary ────────────────
                if not row or (len(row) == 1 and not row[0].strip()):
                    if current_tokens:
                        documents.append((list(current_tokens), list(current_tags)))
                        current_tokens = []
                        current_tags = []
                    continue

                # ── Parse token + tag ─────────────────────────────
                if len(row) >= 2:
                    token = row[0].strip()
                    tag = row[1].strip()
                    if token:  # Skip empty tokens
                        current_tokens.append(token)
                        current_tags.append(tag)
                elif len(row) == 1 and row[0].strip():
                    # Single column — might be tab-separated
                    parts = row[0].split("\t")
                    if len(parts) >= 2:
                        current_tokens.append(parts[0].strip())
                        current_tags.append(parts[1].strip())

        # Don't forget the last document!
        if current_tokens:
            documents.append((current_tokens, current_tags))

    except FileNotFoundError:
        Console.error(f"File not found: {input_path}")
        return {"error": "File not found"}

    Console.success(f"Parsed {len(documents)} documents from CSV")

    if not documents:
        Console.error("No documents found! Check CSV format.")
        return {"imported": 0}

    # ── Connect to database ───────────────────────────────────────
    if not dry_run:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_tables_exist(conn)
    else:
        conn = None

    # ── Process each document ─────────────────────────────────────
    imported = 0
    skipped = 0
    total_annotations = 0
    entity_counter = Counter()
    now = datetime.now().isoformat()

    for doc_idx, (tokens, tags) in enumerate(documents):
        if len(tokens) < 3:  # Skip tiny documents
            skipped += 1
            continue

        # ── Reconstruct full text from tokens ─────────────────────
        full_text = " ".join(tokens)

        # ── Compute token character positions ─────────────────────
        # Since we joined with spaces, each token starts at
        # the sum of all previous (token_length + 1) positions
        token_positions = []
        pos = 0
        for token in tokens:
            token_positions.append((pos, pos + len(token)))
            pos += len(token) + 1  # +1 for the space

        # ── Extract entity spans from BIO tags ────────────────────
        spans = []
        span_start_idx = None
        span_label = None

        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                # Close any open span
                if span_start_idx is not None:
                    base_label = span_label
                    entity_type = BIO_LABEL_MAP.get(base_label)
                    if entity_type:
                        char_start = token_positions[span_start_idx][0]
                        char_end = token_positions[i - 1][1]
                        span_text = full_text[char_start:char_end]
                        spans.append((entity_type, char_start, char_end, span_text))

                # Start new span
                span_label = tag[2:]  # Remove "B-" prefix
                span_start_idx = i

            elif tag.startswith("I-"):
                # Continue span — handle orphaned I-tag
                if span_start_idx is None:
                    # Orphaned I-tag: treat as B-
                    span_label = tag[2:]
                    span_start_idx = i

            elif tag == "O":
                # Close any open span
                if span_start_idx is not None:
                    base_label = span_label
                    entity_type = BIO_LABEL_MAP.get(base_label)
                    if entity_type:
                        char_start = token_positions[span_start_idx][0]
                        char_end = token_positions[i - 1][1]
                        span_text = full_text[char_start:char_end]
                        spans.append((entity_type, char_start, char_end, span_text))
                    span_start_idx = None
                    span_label = None

        # Close final span if document ends mid-entity
        if span_start_idx is not None:
            base_label = span_label
            entity_type = BIO_LABEL_MAP.get(base_label)
            if entity_type:
                char_start = token_positions[span_start_idx][0]
                char_end = token_positions[-1][1]
                span_text = full_text[char_start:char_end]
                spans.append((entity_type, char_start, char_end, span_text))

        if not spans:
            skipped += 1
            continue

        # ── Insert into database ──────────────────────────────────
        if dry_run:
            imported += 1
            for etype, _, _, _ in spans:
                entity_counter[etype] += 1
                total_annotations += 1
            if imported <= 3:
                Console.info(f"[DRY RUN] Doc {doc_idx}: {len(spans)} spans, {len(tokens)} tokens")
                for etype, s, e, t in spans[:5]:
                    Console.stat(f"  {etype}", f"[{s}:{e}] \"{t[:50]}\"")
            continue

        candidate_id = get_next_candidate_id(conn, candidate_id_prefix)
        doc_id = f"kaggle_bio_{doc_idx:06d}"

        try:
            existing = conn.execute(
                "SELECT doc_id FROM ner_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing:
                skipped += 1
                continue

            # Insert raw text
            conn.execute(
                """INSERT OR IGNORE INTO raw_extractions
                   (candidate_id, raw_text, resume_language, text_length,
                    extraction_timestamp)
                   VALUES (?, ?, 'en', ?, ?)""",
                (candidate_id, full_text, len(full_text), now)
            )

            # Insert document
            conn.execute(
                """INSERT INTO ner_documents
                   (doc_id, candidate_id, status, annotator, notes)
                   VALUES (?, ?, 'completed', 'kaggle_bio', ?)""",
                (doc_id, candidate_id,
                 json.dumps({"source": "yashpwrr_bio", "tokens": len(tokens)}))
            )

            # Insert annotations
            for entity_type, char_start, char_end, text in spans:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO ner_annotations
                           (candidate_id, doc_id, entity_type, char_start,
                            char_end, text_content, layer, confidence,
                            annotator, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, 0, 0.85, 'kaggle_bio', ?)""",
                        (candidate_id, doc_id, entity_type,
                         char_start, char_end, text, now)
                    )
                    entity_counter[entity_type] += 1
                    total_annotations += 1
                except sqlite3.IntegrityError:
                    pass

            imported += 1

            if imported % 500 == 0:
                conn.commit()
                Console.info(f"  Imported {imported} documents...")

        except sqlite3.Error as e:
            logger.warning(f"Error importing doc {doc_idx}: {e}")
            skipped += 1

    if conn:
        conn.commit()
        conn.close()

    Console.success(f"Imported: {imported} documents")
    Console.stat("Annotations", total_annotations)
    Console.stat("Skipped", skipped)

    if entity_counter:
        Console.info("Entity type distribution:")
        for etype, count in entity_counter.most_common(15):
            Console.stat(f"  {etype}", count)

    return {"imported": imported, "annotations": total_annotations, "skipped": skipped}


# =============================================================================
# 📥 IMPORTER 3: CLASSIFICATION CSV (Bhawal / Snehaanbhawal)
#
# CSV format: ID, Resume_str, Resume_html, Category
# Maps Kaggle categories → AiMerlion Function labels
# =============================================================================

def import_classification_csv(
    input_path: str,
    db_path: str,
    dry_run: bool = False,
    candidate_id_prefix: int = 940000
) -> Dict[str, int]:
    """
    📥 Import classification-labeled resumes (Category + text).

    CSV structure:
        ID, Resume_str, Resume_html, Category

    Maps Kaggle categories to AiMerlion Function labels and inserts
    into ner_documents with the function column set.

    Args:
        input_path:           Path to the classification CSV file
        db_path:              Path to resume_extractions.db
        dry_run:              If True, preview without writing to DB
        candidate_id_prefix:  Starting candidate_id range (default: 940,000)

    Returns:
        Dict with import statistics
    """
    Console.banner("📥 Importing Classification CSV")

    # ── Read CSV ──────────────────────────────────────────────────
    records = []
    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column name variations
                text = (row.get("Resume_str") or row.get("Resume")
                        or row.get("resume") or row.get("text") or "")
                category = (row.get("Category") or row.get("category")
                            or row.get("label") or "Others")

                if text and len(text.strip()) > 50:
                    records.append({"text": text.strip(), "category": category.strip()})

    except FileNotFoundError:
        Console.error(f"File not found: {input_path}")
        return {"error": "File not found"}

    Console.success(f"Parsed {len(records)} records from CSV")

    if not records:
        Console.error("No valid records found!")
        return {"imported": 0}

    # ── Connect to database ───────────────────────────────────────
    if not dry_run:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_tables_exist(conn)
    else:
        conn = None

    imported = 0
    skipped = 0
    category_counter = Counter()
    now = datetime.now().isoformat()

    for idx, record in enumerate(records):
        # Map category to AiMerlion Function label
        function_label = CLASSIFICATION_MAP.get(record["category"], "Others")
        category_counter[function_label] += 1

        if dry_run:
            imported += 1
            if imported <= 3:
                Console.info(f"[DRY RUN] \"{record['category']}\" → \"{function_label}\"")
            continue

        candidate_id = get_next_candidate_id(conn, candidate_id_prefix)
        doc_id = f"kaggle_cls_{idx:06d}"

        try:
            existing = conn.execute(
                "SELECT doc_id FROM ner_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing:
                skipped += 1
                continue

            # Insert raw text
            conn.execute(
                """INSERT OR IGNORE INTO raw_extractions
                   (candidate_id, raw_text, resume_language, text_length,
                    extraction_timestamp)
                   VALUES (?, ?, 'en', ?, ?)""",
                (candidate_id, record["text"], len(record["text"]), now)
            )

            # Insert document with Function label
            conn.execute(
                """INSERT INTO ner_documents
                   (doc_id, candidate_id, status, annotator, function, notes)
                   VALUES (?, ?, 'completed', 'kaggle_classification', ?, ?)""",
                (doc_id, candidate_id, function_label,
                 json.dumps({"original_category": record["category"]}))
            )

            imported += 1

            if imported % 500 == 0:
                conn.commit()
                Console.info(f"  Imported {imported} records...")

        except sqlite3.Error as e:
            logger.warning(f"Error importing record {idx}: {e}")
            skipped += 1

    if conn:
        conn.commit()
        conn.close()

    Console.success(f"Imported: {imported} records")
    Console.stat("Skipped", skipped)

    if category_counter:
        Console.info("Category distribution:")
        for cat, count in category_counter.most_common(15):
            Console.stat(f"  {cat}", count)

    return {"imported": imported, "skipped": skipped}


# =============================================================================
# 🧹 CLEANUP — Remove previously imported data
# =============================================================================

def cleanup_source(db_path: str, annotator_pattern: str, doc_pattern: str):
    """
    🧹 Remove previously imported data matching the given patterns.

    Args:
        db_path:           Path to resume_extractions.db
        annotator_pattern: Annotator value to match (e.g., 'kaggle_dataturks')
        doc_pattern:       Doc_id LIKE pattern (e.g., 'kaggle_dt_%')
    """
    if not os.path.exists(db_path):
        Console.warning("Database not found, nothing to clean")
        return 0

    conn = sqlite3.connect(db_path)

    count = conn.execute(
        "SELECT COUNT(*) FROM ner_documents WHERE doc_id LIKE ?",
        (doc_pattern,)
    ).fetchone()[0]

    if count == 0:
        Console.info(f"No previous data matching '{doc_pattern}' — nothing to clean")
        conn.close()
        return 0

    # Remove annotations first, then documents, then raw text
    conn.execute("DELETE FROM ner_annotations WHERE doc_id LIKE ?", (doc_pattern,))
    conn.execute("DELETE FROM ner_documents WHERE doc_id LIKE ?", (doc_pattern,))
    conn.commit()
    conn.close()

    Console.success(f"Removed {count} previously imported documents")
    return count


# =============================================================================
# 🎬 MAIN — THE GRAND IMPORT SHOW!
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="💅✨ Import Kaggle/HuggingFace NER datasets into AiMerlion ✨💅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # DataTurks NER (220 resumes, 10 entity types)
    python import_kaggle_ner.py --source dataturks --input traindata.json

    # Mehyar IT CVs (5,029 CVs, SKILL entity — uses exclusive ends!)
    python import_kaggle_ner.py --source dataturks --input mehyar_annotated_cvs.json --exclusive-end

    # yashpwrr BIO-tagged (22K+ samples, 25 entity types)
    python import_kaggle_ner.py --source bio --input resume_ner_dataset.csv

    # Bhawal classification (2,484 resumes, 24 categories)
    python import_kaggle_ner.py --source classification --input Resume.csv

    # Dry run (preview without writing)
    python import_kaggle_ner.py --source dataturks --input traindata.json --dry-run

    # Clean previous import before re-importing
    python import_kaggle_ner.py --source dataturks --input traindata.json --clean
        """
    )

    parser.add_argument(
        "--source", required=True,
        choices=["dataturks", "bio", "classification"],
        help="Dataset format: dataturks (JSON spans, also Mehyar), "
             "bio (BIO-tagged CSV), classification (Category + text CSV)"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the downloaded dataset file"
    )
    parser.add_argument(
        "--db", default="resume_extractions.db",
        help="Database path (default: resume_extractions.db)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be imported without writing to DB"
    )
    parser.add_argument(
        "--exclusive-end", action="store_true",
        help="End indices are already exclusive (use for Mehyar). "
             "Default: inclusive (DataTurks style, +1 added automatically)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove previously imported data for this source before importing"
    )

    args = parser.parse_args()

    # ── Grand opening ─────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  💅✨ KAGGLE/HUGGINGFACE DATA IMPORTER ✨💅")
    print("═" * 60)
    print(f"  📥 Source:        {args.source}")
    print(f"  📂 Input:         {args.input}")
    print(f"  🗄️  Database:      {args.db}")
    print(f"  🔍 Dry run:       {'Yes' if args.dry_run else 'No'}")
    if args.source == "dataturks":
        print(f"  📏 End indices:   {'Exclusive (Mehyar)' if args.exclusive_end else 'Inclusive (DataTurks, +1 auto)'}")
    print("═" * 60)

    # ── Validate input file ───────────────────────────────────────
    if not os.path.exists(args.input):
        Console.error(f"Input file not found: {args.input}")
        Console.info(f"Download the dataset first, then save to: {args.input}")
        return

    # ── Clean previous import (optional) ──────────────────────────
    if args.clean and not args.dry_run:
        if args.source == "dataturks":
            cleanup_source(args.db, "kaggle_dataturks", "kaggle_dt_%")
        elif args.source == "bio":
            cleanup_source(args.db, "kaggle_bio", "kaggle_bio_%")
        elif args.source == "classification":
            cleanup_source(args.db, "kaggle_classification", "kaggle_cls_%")

    # ── Run the appropriate importer ──────────────────────────────
    if args.source == "dataturks":
        result = import_dataturks(
            args.input, args.db, args.dry_run,
            exclusive_end=args.exclusive_end
        )
    elif args.source == "bio":
        result = import_bio_tagged(args.input, args.db, args.dry_run)
    elif args.source == "classification":
        result = import_classification_csv(args.input, args.db, args.dry_run)

    # ── Final summary ─────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  🎯 WHAT'S NEXT")
    print("═" * 60)

    if args.dry_run:
        print("  ✅ Dry run complete! Remove --dry-run to actually import.")
    else:
        print("  1. Validate your combined data:")
        print(f"     python train_ner.py --db {args.db} --validate")
        print()
        print("  2. Train the spaCy NER model:")
        print(f"     python train_ner.py --db {args.db} --train-ner --min-per-entity 10 --epochs 50")

        if args.source == "classification":
            print()
            print("  3. Train the classifier:")
            print(f"     python train_ner.py --db {args.db} --train-classifier")

    print()
    print("═" * 60)
    print("  💅✨ Your Fairy Codemother is PROUD of you! ✨💅")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()