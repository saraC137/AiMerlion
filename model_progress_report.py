"""
model_progress_report.py

💅✨ FAIRY CODEMOTHER'S MODEL PROGRESS REPORT ✨💅

The STAKEHOLDER DASHBOARD — one URL, one screenshot, one story:
"The model is getting better, and here's the proof."

Features:
  📈 Historical Timeline  — Weekly F1/P/R snapshots → line chart over time
  ⚔️ A/B Comparison       — Regex vs Ollama head-to-head per field
  🏋️ Difficulty Scoring    — Easy/Medium/Hard resume accuracy breakdown
  ✅ Regression Suite      — Golden set before/after testing

Architecture:
  Reads from existing resume_extractions.db (same as all other tools).
  Creates ONE new table: metric_snapshots (for timeline tracking).
  Optionally reads golden_set.json for regression testing.
  Imports ml_quality_audit.py for Levenshtein + NER metric calculations.

Port Map:
  5001  → Classification Dashboard
  5050  → Review Dashboard
  5055  → NER Annotation Tool
  5070  → Accuracy Validator
  5075  → Run Comparator
  5080  → THIS FILE (Model Progress Report) ← NEW! ✨

Usage:
    python model_progress_report.py
    python model_progress_report.py --snapshot       # Take a metric snapshot NOW
    python model_progress_report.py --regression     # Run regression suite (CLI)

Dependencies:
    pip install flask --break-system-packages
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import datetime
import math
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from functools import wraps

from flask import (
    Flask, request, jsonify, g, abort, make_response, render_template_string
)

# =============================================================================
# 🔬 ML QUALITY AUDIT (optional — graceful degradation)
# =============================================================================
try:
    from ml_quality_audit import (
        levenshtein_distance, similarity_ratio, compute_ner_metrics
    )
    ML_AUDIT_AVAILABLE = True
except ImportError:
    ML_AUDIT_AVAILABLE = False
    # Inline Levenshtein fallback
    def levenshtein_distance(s1: str, s2: str) -> int:
        if s1 == s2: return 0
        if not s1: return len(s2)
        if not s2: return len(s1)
        if len(s1) > len(s2): s1, s2 = s2, s1
        prev = list(range(len(s1) + 1))
        curr = [0] * (len(s1) + 1)
        for j in range(1, len(s2) + 1):
            curr[0] = j
            for i in range(1, len(s1) + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                curr[i] = min(curr[i-1]+1, prev[i]+1, prev[i-1]+cost)
            prev, curr = curr, prev
        return prev[len(s1)]

    def similarity_ratio(s1: str, s2: str) -> float:
        if not s1 and not s2: return 1.0
        if not s1 or not s2: return 0.0
        mx = max(len(s1), len(s2))
        return round(1.0 - levenshtein_distance(s1, s2) / mx, 4) if mx else 1.0

# =============================================================================
# 🔧 LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - 📊 %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 🎛️ CONFIGURATION
# =============================================================================

DATABASE_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")
GOLDEN_SET_PATH = os.environ.get("GOLDEN_SET_PATH", "golden_set.json")
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", r"C:\Users\user\github\AiMerlion\exports")

HOST = "0.0.0.0"
PORT = 5080
DEBUG = True

# Fields we track (same as accuracy_validator)
TRACKED_FIELDS = {
    "name":          {"db_col": "name",            "label": "Full Name",      "critical": True},
    "email":         {"db_col": "email",           "label": "Email",          "critical": True},
    "phone":         {"db_col": "phone",           "label": "Phone",          "critical": True},
    "date_of_birth": {"db_col": "date_of_birth",   "label": "Date of Birth",  "critical": False},
    "location":      {"db_col": "location",        "label": "Location",       "critical": False},
    "nationality":   {"db_col": "nationality",     "label": "Nationality",    "critical": False},
    "summary":       {"db_col": "summary",         "label": "Summary",        "critical": False},
    "skills_hard":   {"db_col": "skills_raw",      "label": "Hard Skills",    "critical": True},
    "skills_soft":   {"db_col": "skills_raw",      "label": "Soft Skills",    "critical": False},
    "experience":    {"db_col": "experience_raw",  "label": "Experience",     "critical": True},
    "education":     {"db_col": "education_raw",   "label": "Education",      "critical": True},
    "languages":     {"db_col": "languages",       "label": "Languages",      "critical": False},
    "certifications":{"db_col": "certifications",  "label": "Certifications", "critical": False},
}

VALID_VERDICTS = {"correct", "wrong", "partial", "missing", "skip"}

# Difficulty scoring weights
DIFFICULTY_WEIGHTS = {
    "multi_page":       1,   # More than 1 page
    "long_text":        1,   # text_length > 3000 chars
    "short_text":       2,   # text_length < 500 chars (likely scan/OCR issue)
    "non_english":      1,   # resume_language != 'English'
    "many_empty":       1,   # More than 5 empty critical fields
}


# =============================================================================
# 🏗️ FLASK APP
# =============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fairy-progress-sparkle-2026")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# =============================================================================
# 🔌 DATABASE
# =============================================================================

def get_db() -> sqlite3.Connection:
    if "db" not in g:
        if not os.path.exists(DATABASE_PATH):
            abort(500, description=f"Database not found: {DATABASE_PATH}")
        g.db = sqlite3.connect(DATABASE_PATH, timeout=15)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA synchronous=NORMAL")
        g.db.execute("PRAGMA cache_size=-8000")
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def initialize_schema():
    """
    Create ALL required tables if they don't exist yet.

    🧚‍♀️ FAIRY CODEMOTHER SAYS: Think of this like setting up the runway
    BEFORE the models walk — you need the infrastructure FIRST, honey!

    Tables created:
      - metric_snapshots   → Historical timeline tracking (our own table)
      - accuracy_reviews   → Human verdicts per field (may be created by
                             accuracy_validator.py; we create it here as
                             a safety net so the dashboard never crashes
                             on a fresh DB)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")

        # ── Our primary table: snapshot history ──────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metric_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                label TEXT DEFAULT '',
                total_reviewed INTEGER DEFAULT 0,
                total_candidates INTEGER DEFAULT 0,
                weighted_f1 REAL DEFAULT 0.0,
                weighted_precision REAL DEFAULT 0.0,
                weighted_recall REAL DEFAULT 0.0,
                overall_accuracy REAL DEFAULT 0.0,
                critical_accuracy REAL DEFAULT 0.0,
                field_metrics_json TEXT DEFAULT '{}',
                error_taxonomy_json TEXT DEFAULT '{}',
                notes TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshot_date
            ON metric_snapshots(snapshot_date)
        """)

        # ── Safety-net table: accuracy_reviews ───────────────────────────
        # This table is the PRIMARY output of accuracy_validator.py.
        # We create it here (IF NOT EXISTS) so this dashboard never
        # crashes with "no such table" on a fresh database where the
        # validator hasn't been run yet.
        # Schema mirrors what accuracy_validator.py produces.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                field_key TEXT NOT NULL,
                verdict TEXT NOT NULL
                    CHECK(verdict IN ('correct','wrong','partial','missing','skip')),
                correction TEXT DEFAULT '',
                reviewed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reviewer TEXT DEFAULT 'human'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accuracy_reviews_candidate
            ON accuracy_reviews(candidate_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accuracy_reviews_field
            ON accuracy_reviews(field_key, verdict)
        """)

        conn.commit()
        conn.close()
        logger.info("✨ Schema initialized! metric_snapshots + accuracy_reviews ready!")
    except sqlite3.Error as e:
        logger.error(f"❌ Schema init failed: {e}")

def table_exists(db: sqlite3.Connection, table_name: str) -> bool:
    """
    🔍 Check if a table exists in the database before querying it.

    🧚‍♀️ FAIRY CODEMOTHER SAYS: This is like checking if the guest is
    actually ON the list before letting them through the velvet rope!
    Never assume, always verify — that's runway wisdom, darling! 💋

    Args:
        db:         Active SQLite connection
        table_name: Table name to check

    Returns:
        True if table exists, False otherwise
    """
    row = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    return row is not None

# =============================================================================
# 📈 FEATURE 1: HISTORICAL TIMELINE
# =============================================================================

def take_snapshot(db: sqlite3.Connection, label: str = "") -> Dict:
    """
    📸 Capture current accuracy metrics and store as a snapshot.

    Like taking a photo at each fitting — so you can see the
    before/during/after of the whole transformation! 📸✨
    """
    # Fetch all reviews
    reviews = [dict(r) for r in db.execute(
        "SELECT candidate_id, field_key, verdict, correction FROM accuracy_reviews"
    ).fetchall()]

    if not reviews:
        return {"error": "No reviews to snapshot", "saved": False}

    # Compute verdict counts
    verdict_counts = defaultdict(int)
    field_verdicts = defaultdict(lambda: defaultdict(int))
    for r in reviews:
        if r["verdict"] != "skip":
            verdict_counts[r["verdict"]] += 1
            field_verdicts[r["field_key"]][r["verdict"]] += 1

    total = sum(verdict_counts.values())
    correct = verdict_counts.get("correct", 0)
    partial = verdict_counts.get("partial", 0)
    wrong = verdict_counts.get("wrong", 0)
    missing = verdict_counts.get("missing", 0)

    accuracy = ((correct + partial * 0.5) / total * 100) if total > 0 else 0

    # Compute per-field metrics
    field_metrics = {}
    crit_correct = crit_total = noncrit_correct = noncrit_total = 0

    for fk, fv in field_verdicts.items():
        f_total = sum(fv.values())
        f_correct = fv.get("correct", 0) + fv.get("partial", 0) * 0.5
        f_acc = (f_correct / f_total * 100) if f_total > 0 else 0
        info = TRACKED_FIELDS.get(fk, {})
        field_metrics[fk] = {"accuracy": round(f_acc, 1), "total": f_total}
        if info.get("critical"):
            crit_correct += f_correct
            crit_total += f_total
        else:
            noncrit_correct += f_correct
            noncrit_total += f_total

    crit_acc = (crit_correct / crit_total * 100) if crit_total > 0 else 0

    # Compute NER metrics if available
    w_f1 = w_p = w_r = 0.0
    if ML_AUDIT_AVAILABLE:
        ner = compute_ner_metrics(reviews, TRACKED_FIELDS)
        w_f1 = ner["weighted"]["f1"]
        w_p = ner["weighted"]["precision"]
        w_r = ner["weighted"]["recall"]

    # Error taxonomy
    error_tax = {"correct": correct, "wrong": wrong, "partial": partial, "missing": missing}

    total_candidates = db.execute("SELECT COUNT(*) as c FROM structured_extractions").fetchone()["c"]
    reviewed_candidates = db.execute("SELECT COUNT(DISTINCT candidate_id) as c FROM accuracy_reviews").fetchone()["c"]

    # Save snapshot
    db.execute("""
        INSERT INTO metric_snapshots
            (label, total_reviewed, total_candidates,
             weighted_f1, weighted_precision, weighted_recall,
             overall_accuracy, critical_accuracy,
             field_metrics_json, error_taxonomy_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        label or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        reviewed_candidates, total_candidates,
        round(w_f1, 4), round(w_p, 4), round(w_r, 4),
        round(accuracy, 1), round(crit_acc, 1),
        json.dumps(field_metrics), json.dumps(error_tax),
        f"Auto-snapshot: {reviewed_candidates} reviewed / {total_candidates} total"
    ))
    db.commit()

    logger.info(f"📸 Snapshot saved! F1={w_f1:.2f}, Accuracy={accuracy:.1f}%, "
                f"Reviewed={reviewed_candidates}/{total_candidates}")

    return {
        "saved": True,
        "weighted_f1": round(w_f1, 4),
        "accuracy": round(accuracy, 1),
        "critical_accuracy": round(crit_acc, 1),
        "reviewed": reviewed_candidates,
        "total": total_candidates,
    }


# =============================================================================
# ⚔️ FEATURE 2: A/B COMPARISON
# =============================================================================

def compute_ab_comparison(db: sqlite3.Connection) -> Dict:
    """
    ⚔️ Compare Regex vs Ollama extraction quality head-to-head.

    Splits reviewed candidates by ai_assisted flag, computes
    accuracy metrics for each group per field. Like judging
    two contestants side by side — may the best method win! 👑
    """
    # Get reviewed candidates with their extraction method
    # ── Guard: return empty result if reviews table doesn't exist yet ────
    # Happens on a fresh DB before accuracy_validator.py has been run.
    if not table_exists(db, "accuracy_reviews"):
        logger.warning("⚠️ accuracy_reviews table not found — returning empty A/B data")
        return {
            "ai":    {"name": "Ollama (AI-Assisted)", "count": 0, "accuracy": 0, "field_accuracy": {}},
            "regex": {"name": "Regex Only",           "count": 0, "accuracy": 0, "field_accuracy": {}},
            "field_winners": {},
            "summary": {"ai_wins": 0, "regex_wins": 0, "ties": 0, "overall_winner": "tie"},
            "_warning": "No accuracy_reviews found. Run the Accuracy Validator first!",
        }

    rows = db.execute("""
        SELECT DISTINCT ar.candidate_id, s.ai_assisted, s.extraction_method
        FROM accuracy_reviews ar
        JOIN structured_extractions s ON ar.candidate_id = s.candidate_id
    """).fetchall()

    ai_cids = [r["candidate_id"] for r in rows if r["ai_assisted"]]
    regex_cids = [r["candidate_id"] for r in rows if not r["ai_assisted"]]

    def _compute_group_metrics(candidate_ids, group_name):
        if not candidate_ids:
            return {"name": group_name, "count": 0, "accuracy": 0, "field_accuracy": {}}
        placeholders = ",".join("?" * len(candidate_ids))
        reviews = [dict(r) for r in db.execute(f"""
            SELECT field_key, verdict FROM accuracy_reviews
            WHERE candidate_id IN ({placeholders}) AND verdict != 'skip'
        """, candidate_ids).fetchall()]

        total = len(reviews)
        correct = sum(1 for r in reviews if r["verdict"] == "correct")
        partial = sum(1 for r in reviews if r["verdict"] == "partial")
        accuracy = ((correct + partial * 0.5) / total * 100) if total > 0 else 0

        # Per-field accuracy
        field_acc = {}
        field_groups = defaultdict(list)
        for r in reviews:
            field_groups[r["field_key"]].append(r["verdict"])

        for fk, verdicts in field_groups.items():
            ft = len(verdicts)
            fc = sum(1 for v in verdicts if v == "correct") + sum(0.5 for v in verdicts if v == "partial")
            info = TRACKED_FIELDS.get(fk, {})
            field_acc[fk] = {
                "accuracy": round((fc / ft * 100) if ft > 0 else 0, 1),
                "total": ft,
                "label": info.get("label", fk),
                "critical": info.get("critical", False),
            }

        return {
            "name": group_name,
            "count": len(candidate_ids),
            "total_reviews": total,
            "accuracy": round(accuracy, 1),
            "field_accuracy": field_acc,
        }

    ai_metrics = _compute_group_metrics(ai_cids, "Ollama (AI-Assisted)")
    regex_metrics = _compute_group_metrics(regex_cids, "Regex Only")

    # Per-field winner
    field_winners = {}
    all_fields = set(list(ai_metrics["field_accuracy"].keys()) + list(regex_metrics["field_accuracy"].keys()))
    for fk in all_fields:
        ai_acc = ai_metrics["field_accuracy"].get(fk, {}).get("accuracy", 0)
        rx_acc = regex_metrics["field_accuracy"].get(fk, {}).get("accuracy", 0)
        info = TRACKED_FIELDS.get(fk, {})
        if ai_acc > rx_acc:
            winner = "ai"
        elif rx_acc > ai_acc:
            winner = "regex"
        else:
            winner = "tie"
        field_winners[fk] = {
            "winner": winner,
            "ai_accuracy": ai_acc,
            "regex_accuracy": rx_acc,
            "delta": round(ai_acc - rx_acc, 1),
            "label": info.get("label", fk),
        }

    ai_wins = sum(1 for fw in field_winners.values() if fw["winner"] == "ai")
    regex_wins = sum(1 for fw in field_winners.values() if fw["winner"] == "regex")

    return {
        "ai": ai_metrics,
        "regex": regex_metrics,
        "field_winners": field_winners,
        "summary": {
            "ai_wins": ai_wins,
            "regex_wins": regex_wins,
            "ties": len(field_winners) - ai_wins - regex_wins,
            "overall_winner": "ai" if ai_wins > regex_wins else "regex" if regex_wins > ai_wins else "tie",
        },
    }


# =============================================================================
# 🏋️ FEATURE 3: DIFFICULTY SCORING
# =============================================================================

def compute_difficulty_scores(db: sqlite3.Connection) -> Dict:
    """
    🏋️ Score each resume's extraction difficulty and cross-reference
    with accuracy, revealing how the model handles easy vs hard resumes.

    Difficulty factors:
      - Short text (<500 chars)     → +2 (likely OCR/scan issue)
      - Long text (>3000 chars)     → +1 (more content to parse)
      - Multi-page (>1 page)        → +1
      - Non-English content         → +1
      - Many empty critical fields  → +1

    Tiers: 0-1 = Easy 🟢, 2-3 = Medium 🟡, 4-5 = Hard 🔴
    """
    # Fetch candidate data with raw extraction metadata
    rows = db.execute("""
        SELECT
            s.candidate_id,
            s.name, s.email, s.phone, s.skills_raw,
            s.experience_raw, s.education_raw, s.summary,
            s.extraction_status,
            r.text_length, r.pdf_page_count, r.resume_language
        FROM structured_extractions s
        LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
    """).fetchall()

    # Fetch all reviews indexed by candidate
    # ── Guard: skip review cross-referencing if table doesn't exist yet ──
    if not table_exists(db, "accuracy_reviews"):
        logger.warning("⚠️ accuracy_reviews table not found — returning difficulty without accuracy data")
        review_rows = []
    else:
        review_rows = db.execute("""
            SELECT candidate_id, field_key, verdict FROM accuracy_reviews
            WHERE verdict != 'skip'
        """).fetchall()

    reviews_by_cid = defaultdict(list)
    for r in review_rows:
        reviews_by_cid[r["candidate_id"]].append(dict(r))

    candidates = []
    tier_stats = {"easy": {"count": 0, "correct": 0, "total": 0},
                  "medium": {"count": 0, "correct": 0, "total": 0},
                  "hard": {"count": 0, "correct": 0, "total": 0}}

    for row in rows:
        cid = row["candidate_id"]
        text_len = row["text_length"] or 0
        pages = row["pdf_page_count"] or 1
        lang = row["resume_language"] or "English"

        # Count empty critical fields
        empty_critical = 0
        for fk, info in TRACKED_FIELDS.items():
            if info.get("critical"):
                db_col = info["db_col"]
                val = row[db_col] if db_col in row.keys() else None
                if not val or (isinstance(val, str) and not val.strip()):
                    empty_critical += 1

        # Compute difficulty score
        score = 0
        factors = []
        if text_len < 500 and text_len > 0:
            score += DIFFICULTY_WEIGHTS["short_text"]
            factors.append("short_text")
        if text_len > 3000:
            score += DIFFICULTY_WEIGHTS["long_text"]
            factors.append("long_text")
        if pages > 1:
            score += DIFFICULTY_WEIGHTS["multi_page"]
            factors.append("multi_page")
        if lang.lower() not in ("english", "en", ""):
            score += DIFFICULTY_WEIGHTS["non_english"]
            factors.append("non_english")
        if empty_critical >= 4:
            score += DIFFICULTY_WEIGHTS["many_empty"]
            factors.append("many_empty_fields")

        # Classify tier
        if score <= 1:
            tier = "easy"
        elif score <= 3:
            tier = "medium"
        else:
            tier = "hard"

        # Compute accuracy for this candidate (if reviewed)
        cid_reviews = reviews_by_cid.get(cid, [])
        cid_total = len(cid_reviews)
        cid_correct = sum(1 for r in cid_reviews if r["verdict"] == "correct")
        cid_partial = sum(0.5 for r in cid_reviews if r["verdict"] == "partial")
        cid_acc = ((cid_correct + cid_partial) / cid_total * 100) if cid_total > 0 else None

        tier_stats[tier]["count"] += 1
        if cid_total > 0:
            tier_stats[tier]["correct"] += cid_correct + cid_partial
            tier_stats[tier]["total"] += cid_total

        candidates.append({
            "candidate_id": cid,
            "name": row["name"] or "",
            "difficulty_score": score,
            "tier": tier,
            "factors": factors,
            "text_length": text_len,
            "pages": pages,
            "language": lang,
            "accuracy": round(cid_acc, 1) if cid_acc is not None else None,
            "reviews": cid_total,
        })

    # Compute tier-level accuracy
    tier_accuracy = {}
    for tier, stats in tier_stats.items():
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else None
        tier_accuracy[tier] = {
            "count": stats["count"],
            "reviewed": stats["total"],
            "accuracy": round(acc, 1) if acc is not None else None,
        }

    candidates.sort(key=lambda c: c["difficulty_score"], reverse=True)

    return {
        "candidates": candidates,
        "tier_accuracy": tier_accuracy,
        "total_candidates": len(candidates),
        "reviewed_count": sum(1 for c in candidates if c["accuracy"] is not None),
    }


# =============================================================================
# ✅ FEATURE 4: REGRESSION SUITE
# =============================================================================

def run_regression_test(db: sqlite3.Connection) -> Dict:
    """
    ✅ Compare current DB extractions against a golden set.

    Reads golden_set.json (hand-verified correct extractions),
    finds matching candidates in the DB, and computes per-field
    similarity scores. Like a final exam against the answer key! 📝✨

    golden_set.json format:
    [
      {
        "candidate_id": 1,
        "name": "Alex Rivier",
        "email": "alex.rivier@email.com",
        "phone": "+1-555-0199",
        ...
      },
      ...
    ]
    """
    if not os.path.isfile(GOLDEN_SET_PATH):
        return {
            "available": False,
            "error": f"Golden set not found: {GOLDEN_SET_PATH}. "
                     f"Create a JSON file with hand-verified extractions.",
            "how_to_create": (
                "Export verified candidates from the accuracy validator, "
                "correct any errors, and save as golden_set.json."
            ),
        }

    try:
        with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
            golden_records = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"available": False, "error": f"Failed to read golden set: {e}"}

    if not isinstance(golden_records, list) or not golden_records:
        return {"available": False, "error": "Golden set is empty or not a JSON array"}

    # Map field keys to DB columns for golden set comparison
    golden_field_map = {
        "name": "name", "email": "email", "phone": "phone",
        "location": "location", "summary": "summary",
        "skills_raw": "skills_raw", "skills": "skills_raw",
        "experience_raw": "experience_raw", "experience": "experience_raw",
        "education_raw": "education_raw", "education": "education_raw",
        "languages": "languages", "certifications": "certifications",
    }

    results = []
    total_fields = 0
    total_pass = 0
    field_scores = defaultdict(list)

    for golden in golden_records:
        cid = golden.get("candidate_id") or golden.get("ID")
        if cid is None:
            continue

        # Fetch current extraction from DB
        row = db.execute("""
            SELECT * FROM structured_extractions WHERE candidate_id = ?
        """, (cid,)).fetchone()

        if not row:
            results.append({
                "candidate_id": cid, "status": "not_found",
                "fields": {}, "pass_rate": 0,
            })
            continue

        current = dict(row)
        field_results = {}

        for golden_key, db_col in golden_field_map.items():
            golden_val = str(golden.get(golden_key, "") or "").strip()
            current_val = str(current.get(db_col, "") or "").strip()

            if not golden_val:
                continue  # No golden value = not tested

            total_fields += 1
            sim = similarity_ratio(current_val.lower(), golden_val.lower())
            passed = sim >= 0.85  # 85% similarity = pass
            if passed:
                total_pass += 1

            field_results[golden_key] = {
                "golden": golden_val[:80],
                "current": current_val[:80],
                "similarity": round(sim * 100, 1),
                "passed": passed,
            }
            field_scores[golden_key].append(sim)

        candidate_fields = len(field_results)
        candidate_pass = sum(1 for f in field_results.values() if f["passed"])

        results.append({
            "candidate_id": cid,
            "name": golden.get("name", ""),
            "status": "tested",
            "fields": field_results,
            "pass_rate": round((candidate_pass / candidate_fields * 100) if candidate_fields > 0 else 0, 1),
            "passed": candidate_pass,
            "total": candidate_fields,
        })

    # Per-field average similarity
    field_summary = {}
    for fk, scores in field_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        field_summary[fk] = {
            "avg_similarity": round(avg * 100, 1),
            "tests": len(scores),
            "pass_rate": round(sum(1 for s in scores if s >= 0.85) / len(scores) * 100, 1) if scores else 0,
        }

    overall_pass_rate = round((total_pass / total_fields * 100) if total_fields > 0 else 0, 1)

    return {
        "available": True,
        "golden_set_size": len(golden_records),
        "tested": len([r for r in results if r["status"] == "tested"]),
        "not_found": len([r for r in results if r["status"] == "not_found"]),
        "overall_pass_rate": overall_pass_rate,
        "total_fields_tested": total_fields,
        "total_passed": total_pass,
        "total_failed": total_fields - total_pass,
        "results": results,
        "field_summary": field_summary,
        "status": "PASS" if overall_pass_rate >= 80 else "WARN" if overall_pass_rate >= 60 else "FAIL",
    }


# =============================================================================
# 🌐 API ENDPOINTS
# =============================================================================

@app.route("/api/timeline", methods=["GET"])
def api_timeline():
    """📈 Get all metric snapshots for the timeline chart."""
    db = get_db()
    rows = db.execute("""
        SELECT * FROM metric_snapshots ORDER BY snapshot_date ASC
    """).fetchall()

    snapshots = []
    for r in rows:
        d = dict(r)
        d["field_metrics"] = json.loads(d.get("field_metrics_json", "{}") or "{}")
        d["error_taxonomy"] = json.loads(d.get("error_taxonomy_json", "{}") or "{}")
        snapshots.append(d)

    return jsonify({"snapshots": snapshots, "total": len(snapshots)})


@app.route("/api/snapshot", methods=["POST"])
def api_take_snapshot():
    """📸 Take a new metric snapshot right now."""
    db = get_db()
    data = request.get_json() or {}
    label = data.get("label", "")
    result = take_snapshot(db, label)
    return jsonify(result)


@app.route("/api/ab-comparison", methods=["GET"])
def api_ab_comparison():
    """⚔️ Get Regex vs Ollama head-to-head comparison."""
    db = get_db()
    return jsonify(compute_ab_comparison(db))


@app.route("/api/difficulty", methods=["GET"])
def api_difficulty():
    """🏋️ Get difficulty scores and tier-level accuracy."""
    db = get_db()
    return jsonify(compute_difficulty_scores(db))


@app.route("/api/regression", methods=["GET"])
def api_regression():
    """✅ Run regression suite against golden set."""
    db = get_db()
    return jsonify(run_regression_test(db))


@app.route("/api/summary", methods=["GET"])
def api_executive_summary():
    """📊 Executive summary — the stakeholder headline numbers."""
    db = get_db()

    total_candidates = db.execute("SELECT COUNT(*) as c FROM structured_extractions").fetchone()["c"]
    # ── Guard: accuracy_reviews may not exist on fresh DB ───────────────
    if not table_exists(db, "accuracy_reviews"):
        logger.warning("⚠️ accuracy_reviews table not found — summary will show zeros")
        reviewed = 0
        current_acc = 0.0
    else:
        reviewed = db.execute(
            "SELECT COUNT(DISTINCT candidate_id) as c FROM accuracy_reviews"
        ).fetchone()["c"]

        verdict_row = db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN verdict='correct' THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN verdict='partial' THEN 0.5 ELSE 0 END) as partial_credit
            FROM accuracy_reviews WHERE verdict != 'skip'
        """).fetchone()

        current_total = verdict_row["total"] or 0
        current_acc = (
            ((verdict_row["correct"] or 0) + (verdict_row["partial_credit"] or 0))
            / current_total * 100
        ) if current_total > 0 else 0.0

    latest = db.execute("SELECT * FROM metric_snapshots ORDER BY snapshot_date DESC LIMIT 1").fetchone()
    first = db.execute("SELECT * FROM metric_snapshots ORDER BY snapshot_date ASC LIMIT 1").fetchone()

    # Improvement delta (latest vs first snapshot)
    f1_delta = None
    acc_delta = None
    if latest and first and latest["id"] != first["id"]:
        f1_delta = round((latest["weighted_f1"] or 0) - (first["weighted_f1"] or 0), 4)
        acc_delta = round((latest["overall_accuracy"] or 0) - (first["overall_accuracy"] or 0), 1)

    return jsonify({
        "total_candidates": total_candidates,
        "reviewed": reviewed,
        "review_progress": round((reviewed / total_candidates * 100) if total_candidates > 0 else 0, 1),
        "current_accuracy": round(current_acc, 1),
        "current_f1": round(latest["weighted_f1"] * 100, 1) if latest and latest["weighted_f1"] else None,
        "f1_delta": round(f1_delta * 100, 1) if f1_delta is not None else None,
        "accuracy_delta": acc_delta,
        "snapshots_count": db.execute("SELECT COUNT(*) as c FROM metric_snapshots").fetchone()["c"],
        "latest_snapshot_date": latest["snapshot_date"] if latest else None,
    })


# =============================================================================
# 🎨 DASHBOARD
# =============================================================================

@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "db": DATABASE_PATH, "golden_set": GOLDEN_SET_PATH})


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📊 Model Progress Report — AiMerlion</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0a14;--bg-card:#151221;--bg-panel:#110e1d;--bg-input:#1a1630;
  --border:#2a2545;--border-f:#a855f7;
  --text:#f0ecf9;--text-sec:#9b8fc4;--text-mut:#6b5f8a;
  --accent:#c084fc;--accent-dk:#7c3aed;--glow:rgba(168,85,247,0.15);
  --gold:#fbbf24;--pink:#ec4899;--teal:#2dd4bf;
  --ok:#22c55e;--bad:#ef4444;--warn:#f59e0b;--purple:#8b5cf6;
  --mono:'JetBrains Mono','Cascadia Code','Fira Code','Consolas',monospace;
  --display:'Playfair Display','Georgia',serif;
  --body:'Segoe UI',system-ui,-apple-system,sans-serif;
}
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Playfair+Display:wght@700;800&display=swap');
body{background:linear-gradient(180deg,var(--bg),#0f0b1a);color:var(--text);font-family:var(--body);min-height:100vh}
.header{background:linear-gradient(135deg,var(--bg-card),#1a1040);border-bottom:1px solid var(--border);padding:16px 28px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.header h1{font-family:var(--display);font-size:1.3rem;color:var(--accent);font-weight:700}
.header .sub{font-size:10px;color:var(--text-mut);font-family:var(--mono)}
.btn{border:none;border-radius:8px;padding:8px 18px;font-size:12px;cursor:pointer;font-weight:600;transition:all 0.2s;display:inline-flex;align-items:center;gap:6px}
.btn-primary{background:var(--accent-dk);color:#fff}.btn-primary:hover{background:var(--accent)}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text-sec)}.btn-outline:hover{border-color:var(--accent);color:var(--accent)}
.btn-sm{padding:5px 12px;font-size:11px}
.tab-bar{display:flex;gap:2px;padding:8px 28px 0;border-bottom:1px solid var(--border)}
.tab-btn{background:transparent;border:none;border-bottom:2px solid transparent;padding:8px 16px;color:var(--text-sec);font-size:12px;cursor:pointer;font-weight:500;margin-bottom:-1px;transition:all .15s}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent);font-weight:700}
.content{padding:24px 28px;max-width:1200px;margin:0 auto}
.view{display:none}.view.active{display:block}
.stats-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}
.stat-card{background:linear-gradient(135deg,var(--bg-card),var(--bg-panel));border:1px solid var(--border);border-radius:14px;padding:16px 20px;flex:1;min-width:130px}
.stat-card .label{font-size:10px;color:var(--text-mut);text-transform:uppercase;letter-spacing:0.08em;font-family:var(--mono)}
.stat-card .value{font-size:28px;font-weight:800;font-family:var(--display);line-height:1.2;margin-top:4px}
.stat-card .sub{font-size:10px;color:var(--text-sec);margin-top:2px}
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;margin-bottom:20px;overflow:hidden}
.card-header{padding:14px 20px;border-bottom:1px solid var(--border);font-size:14px;font-weight:700;color:var(--accent);display:flex;align-items:center;gap:8px}
.card-body{padding:16px 20px}
.card-body canvas{max-height:300px}
table{width:100%;border-collapse:collapse;font-size:11px}
th{text-align:left;padding:8px 14px;font-size:10px;color:var(--text-mut);text-transform:uppercase;letter-spacing:0.06em;font-family:var(--mono);border-bottom:1px solid var(--border);background:var(--bg-panel)}
td{padding:7px 14px;border-bottom:1px solid rgba(42,37,69,0.2);font-family:var(--mono)}
tr:hover{background:var(--glow)}
.empty{text-align:center;padding:50px 20px;color:var(--text-sec)}
.loading{text-align:center;padding:40px;color:var(--text-sec)}
.badge{font-size:10px;font-family:var(--mono);padding:2px 8px;border-radius:4px;font-weight:700}
.badge-pass{background:rgba(34,197,94,0.12);color:var(--ok)}
.badge-fail{background:rgba(239,68,68,0.12);color:var(--bad)}
.badge-warn{background:rgba(245,158,11,0.12);color:var(--warn)}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>📊 Model Progress Report</h1>
    <div class="sub">AiMerlion — Stakeholder Dashboard • Prove the glow-up ✨</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <button class="btn btn-primary btn-sm" onclick="takeSnapshot()">📸 Take Snapshot</button>
    <button class="btn btn-outline btn-sm" onclick="loadAll()">🔄 Refresh</button>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('summary')" id="tab-summary">📊 Summary</button>
  <button class="tab-btn" onclick="switchTab('timeline')" id="tab-timeline">📈 Timeline</button>
  <button class="tab-btn" onclick="switchTab('ab')" id="tab-ab">⚔️ A/B Test</button>
  <button class="tab-btn" onclick="switchTab('difficulty')" id="tab-difficulty">🏋️ Difficulty</button>
  <button class="tab-btn" onclick="switchTab('regression')" id="tab-regression">✅ Regression</button>
</div>

<div class="content">
  <div class="view active" id="view-summary"><div id="summaryContent"><div class="loading">Loading...</div></div></div>
  <div class="view" id="view-timeline"><div id="timelineContent"><div class="loading">Loading...</div></div></div>
  <div class="view" id="view-ab"><div id="abContent"><div class="loading">Loading...</div></div></div>
  <div class="view" id="view-difficulty"><div id="difficultyContent"><div class="loading">Loading...</div></div></div>
  <div class="view" id="view-regression"><div id="regressionContent"><div class="loading">Loading...</div></div></div>
</div>

<script>
const API = '';
let charts = {};
function destroyChart(id){if(charts[id]){charts[id].destroy();delete charts[id]}}
async function apiFetch(url,opts={}){
  const r=await fetch(API+url,{headers:{'Content-Type':'application/json'},...opts});
  if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.error||'HTTP '+r.status)}
  return r.json();
}
function esc(s){if(!s)return'';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML}
function statCard(label,value,color,sub){
  return '<div class="stat-card"><div class="label">'+label+'</div><div class="value" style="color:'+color+'">'+value+'</div>'+(sub?'<div class="sub">'+sub+'</div>':'')+'</div>';
}

function switchTab(tab){
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  document.getElementById('view-'+tab).classList.add('active');
}

async function takeSnapshot(){
  try{
    const r=await apiFetch('/api/snapshot',{method:'POST',body:JSON.stringify({label:''})});
    if(r.saved){alert('Snapshot saved! F1='+r.weighted_f1+' Acc='+r.accuracy+'%');loadAll();}
    else alert(r.error||'No reviews to snapshot');
  }catch(e){alert('Error: '+e.message)}
}

async function loadAll(){loadSummary();loadTimeline();loadAB();loadDifficulty();loadRegression()}

// ═══════ SUMMARY ═══════
async function loadSummary(){
  const w=document.getElementById('summaryContent');
  try{
    const s=await apiFetch('/api/summary');
    let h='<div class="stats-row">';
    h+=statCard('📊 Accuracy',s.current_accuracy+'%',s.current_accuracy>=80?'var(--ok)':s.current_accuracy>=50?'var(--warn)':'var(--bad)');
    h+=statCard('🎯 F1 Score',s.current_f1!=null?s.current_f1+'%':'—','var(--accent)',s.f1_delta!=null?(s.f1_delta>0?'+':'')+s.f1_delta+'% since first':'Take snapshots to track');
    h+=statCard('👥 Reviewed',s.reviewed,'var(--teal)',s.review_progress+'% of '+s.total_candidates);
    h+=statCard('📸 Snapshots',s.snapshots_count,'var(--purple)',s.latest_snapshot_date||'None yet');
    if(s.accuracy_delta!=null){
      const dc=s.accuracy_delta>0?'var(--ok)':'var(--bad)';
      h+=statCard('📈 Improvement',(s.accuracy_delta>0?'+':'')+s.accuracy_delta+'%',dc,'accuracy since first snapshot');
    }
    h+='</div>';
    h+='<div class="card"><div class="card-header">💡 Quick Start Guide</div><div class="card-body" style="font-size:12px;line-height:1.8;color:var(--text-sec)">';
    h+='<strong style="color:var(--accent)">1.</strong> Review candidates in the <a href="http://localhost:5070/dashboard" style="color:var(--accent)" target="_blank">Accuracy Validator</a> (port 5070)<br>';
    h+='<strong style="color:var(--accent)">2.</strong> Click 📸 Take Snapshot to record current metrics<br>';
    h+='<strong style="color:var(--accent)">3.</strong> Repeat after each model change to build a timeline<br>';
    h+='<strong style="color:var(--accent)">4.</strong> Check ⚔️ A/B to see if Ollama beats regex<br>';
    h+='<strong style="color:var(--accent)">5.</strong> Check 🏋️ Difficulty to ensure hard resumes improve too<br>';
    h+='<strong style="color:var(--accent)">6.</strong> Create golden_set.json for ✅ Regression testing';
    h+='</div></div>';
    w.innerHTML=h;
  }catch(e){w.innerHTML='<div class="empty">'+esc(e.message)+'</div>'}
}

// ═══════ TIMELINE ═══════
async function loadTimeline(){
  const w=document.getElementById('timelineContent');
  try{
    const d=await apiFetch('/api/timeline');
    if(d.total===0){w.innerHTML='<div class="empty">No snapshots yet. Click 📸 Take Snapshot to start tracking!</div>';return}
    let h='<div class="card"><div class="card-header">📈 Accuracy & F1 Over Time</div><div class="card-body"><canvas id="timeChart"></canvas></div></div>';
    h+='<div class="card"><div class="card-header">📋 Snapshot History</div>';
    h+='<table><thead><tr><th>Date</th><th>Accuracy</th><th>F1</th><th>Critical Acc</th><th>Reviewed</th><th>Label</th></tr></thead><tbody>';
    d.snapshots.forEach(function(s){
      h+='<tr><td>'+esc(s.snapshot_date)+'</td>';
      h+='<td style="color:var(--ok);font-weight:700">'+s.overall_accuracy+'%</td>';
      h+='<td style="color:var(--accent);font-weight:700">'+(s.weighted_f1*100).toFixed(1)+'%</td>';
      h+='<td>'+s.critical_accuracy+'%</td>';
      h+='<td>'+s.total_reviewed+'/'+s.total_candidates+'</td>';
      h+='<td style="color:var(--text-mut)">'+esc(s.label)+'</td></tr>';
    });
    h+='</tbody></table></div>';
    w.innerHTML=h;
    // Draw chart
    destroyChart('timeChart');
    const ctx=document.getElementById('timeChart');
    if(ctx){
      const labels=d.snapshots.map(function(s){return s.snapshot_date.split(' ')[0]||s.snapshot_date.substring(0,10)});
      charts['timeChart']=new Chart(ctx,{type:'line',data:{labels:labels,datasets:[
        {label:'Accuracy %',data:d.snapshots.map(function(s){return s.overall_accuracy}),borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,0.1)',borderWidth:2.5,tension:0.3,pointRadius:5,pointBackgroundColor:'#22c55e',fill:true},
        {label:'F1 %',data:d.snapshots.map(function(s){return(s.weighted_f1*100).toFixed(1)}),borderColor:'#c084fc',backgroundColor:'rgba(192,132,252,0.1)',borderWidth:2.5,tension:0.3,pointRadius:5,pointBackgroundColor:'#c084fc',fill:true},
        {label:'Critical Acc %',data:d.snapshots.map(function(s){return s.critical_accuracy}),borderColor:'#fbbf24',borderWidth:1.5,tension:0.3,pointRadius:3,borderDash:[5,3]}
      ]},options:{responsive:true,scales:{y:{beginAtZero:true,max:100,ticks:{color:'#6b5f8a'},grid:{color:'#2a2545'}},x:{ticks:{color:'#9b8fc4',font:{size:10}},grid:{display:false}}},plugins:{legend:{labels:{color:'#9b8fc4',font:{size:10}}}}}});
    }
  }catch(e){w.innerHTML='<div class="empty">'+esc(e.message)+'</div>'}
}

// ═══════ A/B COMPARISON ═══════
async function loadAB(){
  const w=document.getElementById('abContent');
  try{
    const d=await apiFetch('/api/ab-comparison');
    let h='<div class="stats-row">';
    h+=statCard('🤖 Ollama',d.ai.accuracy+'%',d.ai.accuracy>=d.regex.accuracy?'var(--ok)':'var(--bad)',d.ai.count+' candidates');
    h+=statCard('⚙️ Regex',d.regex.accuracy+'%',d.regex.accuracy>=d.ai.accuracy?'var(--ok)':'var(--bad)',d.regex.count+' candidates');
    const ws=d.summary;
    const wc=ws.overall_winner==='ai'?'var(--ok)':ws.overall_winner==='regex'?'var(--warn)':'var(--text-sec)';
    h+=statCard('🏆 Winner',ws.overall_winner==='ai'?'Ollama':ws.overall_winner==='regex'?'Regex':'Tie',wc,ws.ai_wins+' vs '+ws.regex_wins+' fields');
    h+='</div>';
    if(d.ai.count===0&&d.regex.count===0){h+='<div class="empty">No reviewed candidates with extraction method data. Review candidates in the Accuracy Validator first.</div>';w.innerHTML=h;return}
    h+='<div class="card"><div class="card-header">⚔️ Field-by-Field: Ollama vs Regex</div><div class="card-body"><canvas id="abChart"></canvas></div></div>';
    h+='<div class="card"><div class="card-header">📋 Field Winners</div><table><thead><tr><th>Field</th><th>Ollama</th><th>Regex</th><th>Delta</th><th>Winner</th></tr></thead><tbody>';
    for(const[fk,fw] of Object.entries(d.field_winners)){
      const wIcon=fw.winner==='ai'?'<span style="color:var(--ok)">🤖 Ollama</span>':fw.winner==='regex'?'<span style="color:var(--warn)">⚙️ Regex</span>':'<span style="color:var(--text-mut)">Tie</span>';
      h+='<tr><td style="font-weight:700">'+esc(fw.label)+'</td><td>'+fw.ai_accuracy+'%</td><td>'+fw.regex_accuracy+'%</td>';
      h+='<td style="color:'+(fw.delta>0?'var(--ok)':fw.delta<0?'var(--bad)':'var(--text-mut)')+'">'+(fw.delta>0?'+':'')+fw.delta+'%</td>';
      h+='<td>'+wIcon+'</td></tr>';
    }
    h+='</tbody></table></div>';
    w.innerHTML=h;
    // Chart
    destroyChart('abChart');
    const ctx=document.getElementById('abChart');
    if(ctx){
      const fields=Object.keys(d.field_winners);
      const labels=fields.map(function(f){return d.field_winners[f].label});
      charts['abChart']=new Chart(ctx,{type:'bar',data:{labels:labels,datasets:[
        {label:'Ollama',data:fields.map(function(f){return d.field_winners[f].ai_accuracy}),backgroundColor:'rgba(34,197,94,0.5)',borderColor:'rgba(34,197,94,0.8)',borderWidth:1,borderRadius:4},
        {label:'Regex',data:fields.map(function(f){return d.field_winners[f].regex_accuracy}),backgroundColor:'rgba(139,92,246,0.4)',borderColor:'rgba(139,92,246,0.7)',borderWidth:1,borderRadius:4}
      ]},options:{responsive:true,indexAxis:'y',scales:{x:{beginAtZero:true,max:100,ticks:{color:'#6b5f8a'},grid:{color:'#2a2545'}},y:{ticks:{color:'#9b8fc4',font:{size:10}},grid:{display:false}}},plugins:{legend:{labels:{color:'#9b8fc4'}}}}});
    }
  }catch(e){w.innerHTML='<div class="empty">'+esc(e.message)+'</div>'}
}

// ═══════ DIFFICULTY ═══════
async function loadDifficulty(){
  const w=document.getElementById('difficultyContent');
  try{
    const d=await apiFetch('/api/difficulty');
    const ta=d.tier_accuracy;
    let h='<div class="stats-row">';
    const tiers=[['🟢 Easy','easy','var(--ok)'],['🟡 Medium','medium','var(--warn)'],['🔴 Hard','hard','var(--bad)']];
    tiers.forEach(function(t){
      const s=ta[t[1]]||{};
      h+=statCard(t[0],s.accuracy!=null?s.accuracy+'%':'—',t[2],s.count+' resumes, '+s.reviewed+' reviews');
    });
    h+='</div>';
    h+='<div class="card"><div class="card-header">🏋️ Accuracy by Difficulty Tier</div><div class="card-body"><canvas id="diffChart"></canvas></div></div>';
    // Show top hardest candidates
    const hardCands=d.candidates.filter(function(c){return c.tier==='hard'&&c.accuracy!=null}).slice(0,15);
    if(hardCands.length>0){
      h+='<div class="card"><div class="card-header">🔴 Hardest Resumes (reviewed)</div><table><thead><tr><th>ID</th><th>Name</th><th>Score</th><th>Tier</th><th>Accuracy</th><th>Factors</th></tr></thead><tbody>';
      hardCands.forEach(function(c){
        h+='<tr><td>'+c.candidate_id+'</td><td>'+esc(c.name)+'</td>';
        h+='<td style="font-weight:700;color:var(--bad)">'+c.difficulty_score+'</td>';
        h+='<td><span class="badge badge-fail">hard</span></td>';
        h+='<td>'+(c.accuracy!=null?c.accuracy+'%':'—')+'</td>';
        h+='<td style="font-size:10px;color:var(--text-mut)">'+c.factors.join(', ')+'</td></tr>';
      });
      h+='</tbody></table></div>';
    }
    w.innerHTML=h;
    destroyChart('diffChart');
    const ctx=document.getElementById('diffChart');
    if(ctx){
      const tierLabels=['Easy','Medium','Hard'];
      const tierData=[ta.easy?.accuracy||0,ta.medium?.accuracy||0,ta.hard?.accuracy||0];
      const tierColors=['rgba(34,197,94,0.6)','rgba(245,158,11,0.6)','rgba(239,68,68,0.6)'];
      charts['diffChart']=new Chart(ctx,{type:'bar',data:{labels:tierLabels,datasets:[{label:'Accuracy %',data:tierData,backgroundColor:tierColors,borderRadius:8,barThickness:60}]},options:{responsive:true,scales:{y:{beginAtZero:true,max:100,ticks:{color:'#6b5f8a'},grid:{color:'#2a2545'}},x:{ticks:{color:'#9b8fc4'},grid:{display:false}}},plugins:{legend:{display:false}}}});
    }
  }catch(e){w.innerHTML='<div class="empty">'+esc(e.message)+'</div>'}
}

// ═══════ REGRESSION ═══════
async function loadRegression(){
  const w=document.getElementById('regressionContent');
  try{
    const d=await apiFetch('/api/regression');
    if(!d.available){w.innerHTML='<div class="empty"><div style="font-size:36px;margin-bottom:12px">📋</div><h3 style="margin-bottom:8px;color:var(--text-mut)">Golden Set Not Found</h3><p style="font-size:12px;color:var(--text-mut);max-width:400px;margin:0 auto">'+esc(d.error)+'</p></div>';return}
    const sc=d.status==='PASS'?'var(--ok)':d.status==='WARN'?'var(--warn)':'var(--bad)';
    let h='<div class="stats-row">';
    h+=statCard('✅ Status',d.status,sc);
    h+=statCard('📊 Pass Rate',d.overall_pass_rate+'%',sc,d.total_passed+'/'+d.total_fields_tested+' fields');
    h+=statCard('📋 Golden Set',d.golden_set_size+' candidates','var(--accent)',d.tested+' tested, '+d.not_found+' not found');
    h+=statCard('❌ Failures',d.total_failed,d.total_failed>0?'var(--bad)':'var(--ok)');
    h+='</div>';
    // Per-field summary
    if(Object.keys(d.field_summary).length>0){
      h+='<div class="card"><div class="card-header">📋 Per-Field Regression Results</div><table><thead><tr><th>Field</th><th>Avg Similarity</th><th>Pass Rate</th><th>Tests</th></tr></thead><tbody>';
      for(const[fk,fs] of Object.entries(d.field_summary)){
        const c=fs.pass_rate>=85?'var(--ok)':fs.pass_rate>=60?'var(--warn)':'var(--bad)';
        h+='<tr><td style="font-weight:700">'+esc(fk)+'</td>';
        h+='<td style="color:'+c+';font-weight:700">'+fs.avg_similarity+'%</td>';
        h+='<td>'+fs.pass_rate+'%</td><td>'+fs.tests+'</td></tr>';
      }
      h+='</tbody></table></div>';
    }
    // Per-candidate results
    h+='<div class="card"><div class="card-header">👥 Candidate Results</div><table><thead><tr><th>ID</th><th>Name</th><th>Status</th><th>Pass Rate</th><th>Fields</th></tr></thead><tbody>';
    d.results.forEach(function(r){
      const badge=r.status==='not_found'?'<span class="badge badge-warn">not found</span>':r.pass_rate>=85?'<span class="badge badge-pass">PASS</span>':'<span class="badge badge-fail">FAIL</span>';
      h+='<tr><td>'+r.candidate_id+'</td><td>'+esc(r.name)+'</td><td>'+badge+'</td>';
      h+='<td>'+(r.pass_rate||0)+'%</td><td>'+(r.passed||0)+'/'+(r.total||0)+'</td></tr>';
    });
    h+='</tbody></table></div>';
    w.innerHTML=h;
  }catch(e){w.innerHTML='<div class="empty">'+esc(e.message)+'</div>'}
}

loadAll();
</script>
</body>
</html>
"""


# =============================================================================
# 🚀 MAIN
# =============================================================================

def main():
    import model_progress_report as _mod

    parser = argparse.ArgumentParser(
        description="💅✨ Fairy Codemother's Model Progress Report ✨💅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--snapshot", action="store_true", help="Take a metric snapshot and exit")
    parser.add_argument("--regression", action="store_true", help="Run regression suite (CLI) and exit")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--db", default=_mod.DATABASE_PATH)
    parser.add_argument("--golden", default=_mod.GOLDEN_SET_PATH)
    args = parser.parse_args()

    _mod.DATABASE_PATH = args.db
    _mod.GOLDEN_SET_PATH = args.golden

    # Ensure schema exists
    if os.path.exists(DATABASE_PATH):
        initialize_schema()

    # ── CLI: Snapshot mode ─────────────────────────────────────────────
    if args.snapshot:
        if not os.path.exists(DATABASE_PATH):
            print(f"  ❌ Database not found: {DATABASE_PATH}")
            sys.exit(1)
        conn = sqlite3.connect(DATABASE_PATH, timeout=15)
        conn.row_factory = sqlite3.Row
        result = take_snapshot(conn, "CLI snapshot")
        conn.close()
        if result.get("saved"):
            print(f"  ✅ Snapshot saved! Accuracy={result['accuracy']}% F1={result['weighted_f1']}")
        else:
            print(f"  ⚠️ {result.get('error', 'Unknown error')}")
        sys.exit(0)

    # ── CLI: Regression mode ───────────────────────────────────────────
    if args.regression:
        if not os.path.exists(DATABASE_PATH):
            print(f"  ❌ Database not found: {DATABASE_PATH}")
            sys.exit(1)
        conn = sqlite3.connect(DATABASE_PATH, timeout=15)
        conn.row_factory = sqlite3.Row
        result = run_regression_test(conn)
        conn.close()
        if not result.get("available"):
            print(f"  ❌ {result.get('error', 'Golden set not found')}")
            sys.exit(1)
        print(f"\n  ✅ Regression Suite: {result['status']}")
        print(f"     Pass rate: {result['overall_pass_rate']}% ({result['total_passed']}/{result['total_fields_tested']})")
        print(f"     Golden set: {result['golden_set_size']} candidates")
        for fk, fs in result["field_summary"].items():
            icon = "✅" if fs["pass_rate"] >= 85 else "❌"
            print(f"     {icon} {fk:20} {fs['avg_similarity']}% sim, {fs['pass_rate']}% pass")
        sys.exit(0)

    # ── Dashboard mode ─────────────────────────────────────────────────
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " 🧚‍♀️✨ MODEL PROGRESS REPORT ✨🧚‍♀️ ".center(62) + "║")
    print("╠" + "═" * 62 + "╣")
    print("║" + f"  📂 Database:    {DATABASE_PATH}".ljust(62) + "║")
    print("║" + f"  📋 Golden Set:  {GOLDEN_SET_PATH}".ljust(62) + "║")
    print("║" + f"  🌐 Server:      http://localhost:{args.port}".ljust(62) + "║")
    print("║" + "".ljust(62) + "║")
    print("║" + f"  ✨ DASHBOARD:   http://localhost:{args.port}/dashboard".ljust(62) + "║")
    print("║" + "".ljust(62) + "║")
    print("║" + "  Port Map:".ljust(62) + "║")
    print("║" + "    5001 → Classification Dashboard".ljust(62) + "║")
    print("║" + "    5050 → Review Dashboard".ljust(62) + "║")
    print("║" + "    5055 → NER Annotation Tool".ljust(62) + "║")
    print("║" + "    5070 → Accuracy Validator".ljust(62) + "║")
    print("║" + "    5075 → Run Comparator".ljust(62) + "║")
    print("║" + "    5080 → Model Progress Report (this!) ✨".ljust(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

    app.run(host=HOST, port=args.port, debug=DEBUG)


if __name__ == "__main__":
    main()