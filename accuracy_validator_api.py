"""
accuracy_validator_api.py

💅✨ FAIRY CODEMOTHER'S AI EXTRACTION ACCURACY VALIDATOR API ✨💅

A lightweight Flask REST API that bridges the React-based Accuracy Validator
dashboard to the existing resume_extractions.db. This lets you review
the AI extractor's work field-by-field and track accuracy metrics over time.

Architecture:
  - Flask web server on port 5070 (separate from all other dashboards)
  - Reads from existing resume_extractions.db (raw + structured data)
  - Creates NEW table: accuracy_reviews (for storing review verdicts)
  - CORS-enabled so the React artifact can call it from claude.ai
  - Follows same patterns as review_dashboard.py and annotation_tool.py

Port Map (for reference):
  - 5050  → review_dashboard.py
  - 5055  → annotation_tool.py
  - 5001  → classification_dashboard.py
  - 5070  → THIS FILE (accuracy_validator_api.py) ← NEW! ✨
  NOTE: Port 5060/5061 are BLOCKED by Chrome (SIP protocol).

Usage:
    python accuracy_validator_api.py
    # API available at http://localhost:5070/api/

Endpoints:
    GET  /api/candidates              → Paginated candidate list
    GET  /api/candidates/<id>         → Full candidate data (raw + structured)
    GET  /api/candidates/<id>/raw     → Raw resume text only
    GET  /api/candidates/<id>/log     → Extraction log history
    POST /api/reviews/<id>            → Save review verdicts for a candidate
    GET  /api/reviews                 → All reviews (with optional candidate filter)
    GET  /api/reviews/<id>            → Reviews for a specific candidate
    DELETE /api/reviews               → Reset all reviews
    GET  /api/metrics                 → Aggregated accuracy metrics
    GET  /api/stats                   → Database overview stats

Dependencies:
    pip install flask --break-system-packages
"""

import sqlite3
import json
import os
import re
import datetime
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps

from flask import (
    Flask, request, jsonify, g, abort, make_response, render_template_string
)

# =============================================================================
# 🔍 VECTOR SEARCH ENGINE (optional — graceful degradation)
# =============================================================================
try:
    from vector_search import VectorSearchEngine
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

# Global engine instance (lazy-initialized)
_vector_engine = None

# =============================================================================
# 🔬 ML QUALITY AUDIT ENGINE (optional — graceful degradation)
# =============================================================================
try:
    from ml_quality_audit import run_full_audit
    ML_AUDIT_AVAILABLE = True
except ImportError:
    ML_AUDIT_AVAILABLE = False

def get_vector_engine():
    """
    Lazy-initialize the vector search engine.
    Only loads when first search request comes in.
    """
    global _vector_engine
    if _vector_engine is None:
        if not VECTOR_SEARCH_AVAILABLE:
            return None
        try:
            _vector_engine = VectorSearchEngine()
        except Exception as e:
            logging.getLogger(__name__).error(f"❌ Vector engine init failed: {e}")
            return None
    return _vector_engine

# =============================================================================
# 🔧 LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 🎯 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 🎛️ CONFIGURATION
# =============================================================================

# Same database as all other tools — we're a family! 👨‍👩‍👧‍👦
DATABASE_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")

# ── File serving configuration ─────────────────────────────────────
# Where do the original resume files live on disk?
# Set FILES_DIR to the folder containing your PDFs/DOCXs.
# The filenames column in raw_extractions stores the original filename(s).
#
# Think of this as the physical filing cabinet address — we need to
# know which drawer to open to pull the original document! 🗄️✨
#
# Example: FILES_DIR = "C:/Users/user/github/AiMerlion/resumes"
# Or set env var: set RESUME_FILES_DIR=C:\path\to\resumes
FILES_DIR = os.environ.get("RESUME_FILES_DIR", "merlion_resumes")

# File types we can render inline in the browser vs download-only
RENDERABLE_TYPES = {".pdf"}                              # Browser renders natively
DOWNLOADABLE_TYPES = {".docx", ".doc", ".odt", ".rtf"}  # Offered as download
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"} # Show as <img>

# Security: Maximum file size we'll serve (prevent accidental 500MB files)
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))

# Server settings
HOST = "0.0.0.0"
PORT = 5070  # ⚠️ NOT 5060! Chrome blocks 5060/5061 (SIP ports) with ERR_UNSAFE_PORT
DEBUG = True

# Pagination defaults
DEFAULT_PER_PAGE = 30
MAX_PER_PAGE = 100

# Fields tracked by the accuracy validator
# Maps the review field keys → DB column names in structured_extractions
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

# Valid verdict values
VALID_VERDICTS = {"correct", "wrong", "partial", "missing", "skip"}


# =============================================================================
# 🏗️ FLASK APP
# =============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fairy-accuracy-sparkle-2026")


# =============================================================================
# 🌐 CORS MIDDLEWARE
# =============================================================================
# The React dashboard runs on claude.ai, which is a different origin.
# We need to allow cross-origin requests so it can reach this API.
# Think of CORS as the VIP wristband — without it, the bouncer (browser)
# blocks you at the door! 🚪✨

@app.after_request
def add_cors_headers(response):
    """
    Add CORS headers to every response.

    Allows the React artifact on claude.ai (or localhost dev) to call
    this API without the browser blocking the request.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


@app.route("/api/<path:path>", methods=["OPTIONS"])
def handle_preflight(path):
    """Handle CORS preflight requests (OPTIONS)."""
    response = make_response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# =============================================================================
# 🔌 DATABASE CONNECTION MANAGEMENT
# =============================================================================

def get_db() -> sqlite3.Connection:
    """
    Get or create a request-scoped database connection.

    Uses Flask's g object so each HTTP request gets its own connection,
    reused across helper calls within the same request.
    Like sharing one backstage pass for the whole show! 🎭✨
    """
    if "db" not in g:
        if not os.path.exists(DATABASE_PATH):
            logger.error(f"❌ Database not found: {DATABASE_PATH}")
            abort(500, description=f"Database not found: {DATABASE_PATH}")

        try:
            g.db = sqlite3.connect(DATABASE_PATH, timeout=15)
            g.db.row_factory = sqlite3.Row
            # ⚡ Performance PRAGMAs (same as review_dashboard.py)
            g.db.execute("PRAGMA journal_mode=WAL")
            g.db.execute("PRAGMA synchronous=NORMAL")
            g.db.execute("PRAGMA foreign_keys=ON")
            g.db.execute("PRAGMA cache_size=-8000")         # 8MB cache
            g.db.execute("PRAGMA temp_store=MEMORY")
        except sqlite3.Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            abort(500, description=f"Database connection failed: {e}")

    return g.db


@app.teardown_appcontext
def close_db(exception):
    """Auto-close the database connection when the request ends."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


# =============================================================================
# 🏗️ SCHEMA INITIALIZATION — accuracy_reviews table
# =============================================================================

def initialize_accuracy_schema():
    """
    Create the accuracy_reviews table if it doesn't exist.

    This table stores human verdicts for each field of each candidate,
    so we can calculate how accurate the AI extractor really is.
    Think of it as the judge's scorecard at a beauty pageant! 👑📋

    Schema:
      - candidate_id  → links to structured_extractions
      - field_key     → which field (name, email, phone, etc.)
      - verdict       → correct, wrong, partial, missing, skip
      - correction    → the human-provided correct value (if wrong/partial)
      - reviewer      → who reviewed it (defaults to 'default')
      - reviewed_at   → when the review happened
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER NOT NULL,
                field_key TEXT NOT NULL,
                verdict TEXT NOT NULL CHECK(verdict IN ('correct','wrong','partial','missing','skip')),
                correction TEXT DEFAULT '',
                reviewer TEXT DEFAULT 'default',
                reviewed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT DEFAULT '',
                UNIQUE(candidate_id, field_key, reviewer)
            )
        """)

        # ⚡ Indexes for common query patterns
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accuracy_candidate
            ON accuracy_reviews(candidate_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accuracy_field
            ON accuracy_reviews(field_key)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accuracy_verdict
            ON accuracy_reviews(verdict)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accuracy_reviewer
            ON accuracy_reviews(reviewer)
        """)

        conn.commit()
        conn.close()
        logger.info("✨ accuracy_reviews table initialized!")

    except sqlite3.Error as e:
        logger.error(f"❌ Failed to initialize accuracy schema: {e}")
        raise


# =============================================================================
# 🛡️ INPUT VALIDATION HELPERS
# =============================================================================

def validate_candidate_id(candidate_id: Any) -> int:
    """
    Validate that candidate_id is a positive integer.
    The bouncer at the VIP door — no fakes allowed! 🚪
    """
    try:
        cid = int(candidate_id)
        if cid <= 0:
            raise ValueError("Must be positive")
        return cid
    except (TypeError, ValueError):
        abort(400, description=f"Invalid candidate ID: {candidate_id}")


def validate_verdict(verdict: str) -> str:
    """Validate that a verdict is one of the allowed values."""
    if verdict not in VALID_VERDICTS:
        abort(400, description=f"Invalid verdict '{verdict}'. Must be one of: {VALID_VERDICTS}")
    return verdict


def safe_json_parse(text: Optional[str]) -> Any:
    """
    Safely parse a JSON string, returning the original text if parsing fails.
    Some fields in the DB are stored as JSON strings (skills_json, experience_json).
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


# =============================================================================
# 🌐 API ROUTES — Candidates
# =============================================================================

@app.route("/api/candidates", methods=["GET"])
def api_list_candidates():
    """
    📋 List candidates with pagination and optional filters.

    Query params:
      page      → Page number (default 1)
      per_page  → Results per page (default 30, max 100)
      status    → Filter by extraction_status (e.g. 'Complete', 'Partial')
      search    → Search by name or email (case-insensitive)
      reviewed  → Filter: 'yes' = has reviews, 'no' = no reviews, 'all' = any

    Returns:
      JSON with candidates array, pagination info, and summary counts.
    """
    db = get_db()
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", DEFAULT_PER_PAGE, type=int), MAX_PER_PAGE)
    status_filter = request.args.get("status", "")
    search_query = request.args.get("search", "").strip()
    reviewed_filter = request.args.get("reviewed", "all")

    offset = (page - 1) * per_page

    # ── Build WHERE clause dynamically ─────────────────────────────────
    conditions = []
    params = []

    if status_filter:
        conditions.append("s.extraction_status = ?")
        params.append(status_filter)

    if search_query:
        conditions.append("(s.name LIKE ? OR s.email LIKE ? OR CAST(s.candidate_id AS TEXT) LIKE ?)")
        like_q = f"%{search_query}%"
        params.extend([like_q, like_q, like_q])

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    # ── Count totals ───────────────────────────────────────────────────
    total = db.execute(
        f"SELECT COUNT(*) as c FROM structured_extractions s {where_clause}",
        params
    ).fetchone()["c"]

    total_pages = max(1, math.ceil(total / per_page))

    # ── Fetch candidates with review counts ────────────────────────────
    rows = db.execute(f"""
        SELECT
            s.candidate_id,
            s.name,
            s.email,
            s.phone,
            s.extraction_status,
            s.extraction_method,
            s.ai_assisted,
            s.created_at,
            -- Count of accuracy reviews for this candidate
            (SELECT COUNT(*) FROM accuracy_reviews ar
             WHERE ar.candidate_id = s.candidate_id) as review_count,
            -- Count of 'correct' verdicts
            (SELECT COUNT(*) FROM accuracy_reviews ar
             WHERE ar.candidate_id = s.candidate_id AND ar.verdict = 'correct') as correct_count,
            -- Count of 'wrong' verdicts
            (SELECT COUNT(*) FROM accuracy_reviews ar
             WHERE ar.candidate_id = s.candidate_id AND ar.verdict = 'wrong') as wrong_count
        FROM structured_extractions s
        {where_clause}
        GROUP BY s.candidate_id
        ORDER BY s.candidate_id ASC
        LIMIT ? OFFSET ?
    """, params + [per_page, offset]).fetchall()

    # ── Apply reviewed filter post-query ───────────────────────────────
    # (Doing this post-query to keep the main query simple and indexable)
    candidates = []
    for row in rows:
        r = dict(row)
        r["ai_assisted"] = bool(r.get("ai_assisted"))
        if reviewed_filter == "yes" and r["review_count"] == 0:
            continue
        if reviewed_filter == "no" and r["review_count"] > 0:
            continue
        candidates.append(r)

    # ── Summary counts ─────────────────────────────────────────────────
    total_reviewed = db.execute("""
        SELECT COUNT(DISTINCT candidate_id) as c FROM accuracy_reviews
    """).fetchone()["c"]

    total_all = db.execute(
        "SELECT COUNT(*) as c FROM structured_extractions"
    ).fetchone()["c"]

    return jsonify({
        "candidates": candidates,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
        },
        "summary": {
            "total_candidates": total_all,
            "total_reviewed": total_reviewed,
            "total_pending": total_all - total_reviewed,
        }
    })


@app.route("/api/candidates/<candidate_id>", methods=["GET"])
def api_get_candidate(candidate_id):
    """
    🎭 Get FULL candidate data — raw text + structured extraction + reviews.

    This is the main payload for the review dashboard. It returns everything
    needed to display one candidate's extraction for validation.

    Returns:
      JSON with:
        - candidate_id
        - raw_text (full resume text)
        - structured (all extracted fields)
        - extraction_log (field-level extraction history)
        - reviews (existing accuracy verdicts, if any)
    """
    cid = validate_candidate_id(candidate_id)
    db = get_db()

    # ── Structured extraction ──────────────────────────────────────────
    structured_row = db.execute("""
        SELECT * FROM structured_extractions
        WHERE candidate_id = ?
        ORDER BY created_at DESC LIMIT 1
    """, (cid,)).fetchone()

    if not structured_row:
        abort(404, description=f"Candidate {cid} not found in structured_extractions")

    structured = dict(structured_row)
    # Convert ai_assisted from int to bool
    structured["ai_assisted"] = bool(structured.get("ai_assisted"))

    # Parse JSON fields for richer display
    structured["skills_parsed"] = safe_json_parse(structured.get("skills_json"))
    structured["experience_parsed"] = safe_json_parse(structured.get("experience_json"))
    structured["education_parsed"] = safe_json_parse(structured.get("education_json"))

    # ── Raw text ───────────────────────────────────────────────────────
    raw_row = db.execute("""
        SELECT raw_text, text_length, resume_language, filenames,
               extraction_timestamp, pdf_page_count
        FROM raw_extractions
        WHERE candidate_id = ?
        ORDER BY extraction_timestamp DESC LIMIT 1
    """, (cid,)).fetchone()

    raw_data = dict(raw_row) if raw_row else {
        "raw_text": "",
        "text_length": 0,
        "resume_language": "Unknown",
        "filenames": "",
        "extraction_timestamp": None,
        "pdf_page_count": None,
    }

    # ── Extraction log ─────────────────────────────────────────────────
    log_rows = db.execute("""
        SELECT field_name, extraction_method, extracted_value,
               was_successful, was_overridden, override_reason,
               error_message, timestamp
        FROM extraction_log
        WHERE candidate_id = ?
        ORDER BY timestamp DESC
        LIMIT 200
    """, (cid,)).fetchall()

    extraction_log = [dict(r) for r in log_rows]

    # ── Existing reviews ───────────────────────────────────────────────
    review_rows = db.execute("""
        SELECT field_key, verdict, correction, reviewer, reviewed_at, notes
        FROM accuracy_reviews
        WHERE candidate_id = ?
        ORDER BY reviewed_at DESC
    """, (cid,)).fetchall()

    # Group reviews by field_key for easy lookup
    reviews = {}
    for r in review_rows:
        rd = dict(r)
        reviews[rd["field_key"]] = rd

    return jsonify({
        "candidate_id": cid,
        "raw_text": raw_data.get("raw_text", ""),
        "raw_meta": {
            "text_length": raw_data.get("text_length"),
            "resume_language": raw_data.get("resume_language"),
            "filenames": raw_data.get("filenames"),
            "extraction_timestamp": raw_data.get("extraction_timestamp"),
            "pdf_page_count": raw_data.get("pdf_page_count"),
        },
        "structured": structured,
        "extraction_log": extraction_log,
        "reviews": reviews,
    })


@app.route("/api/candidates/<candidate_id>/raw", methods=["GET"])
def api_get_raw_text(candidate_id):
    """📄 Get ONLY the raw resume text for a candidate (lighter payload)."""
    cid = validate_candidate_id(candidate_id)
    db = get_db()

    row = db.execute("""
        SELECT raw_text, text_length FROM raw_extractions
        WHERE candidate_id = ?
        ORDER BY extraction_timestamp DESC LIMIT 1
    """, (cid,)).fetchone()

    if not row:
        abort(404, description=f"No raw text found for candidate {cid}")

    return jsonify({
        "candidate_id": cid,
        "raw_text": row["raw_text"],
        "text_length": row["text_length"],
    })


@app.route("/api/candidates/<candidate_id>/log", methods=["GET"])
def api_get_extraction_log(candidate_id):
    """🔎 Get extraction log history for a candidate."""
    cid = validate_candidate_id(candidate_id)
    db = get_db()

    rows = db.execute("""
        SELECT field_name, extraction_method, extracted_value,
               was_successful, was_overridden, override_reason,
               error_message, timestamp
        FROM extraction_log
        WHERE candidate_id = ?
        ORDER BY timestamp DESC
        LIMIT 500
    """, (cid,)).fetchall()

    return jsonify({
        "candidate_id": cid,
        "log_entries": [dict(r) for r in rows],
        "total": len(rows),
    })


# =============================================================================
# 🌐 API ROUTES — Reviews (CRUD)
# =============================================================================

@app.route("/api/reviews/<candidate_id>", methods=["POST"])
def api_save_reviews(candidate_id):
    """
    💾 Save review verdicts for a candidate.

    Expects JSON body:
    {
      "reviews": {
        "name":   { "verdict": "correct", "correction": "", "notes": "" },
        "email":  { "verdict": "wrong",   "correction": "real@email.com", "notes": "Typo" },
        ...
      },
      "reviewer": "soraya"   ← optional, defaults to "default"
    }

    Uses INSERT OR REPLACE so calling this endpoint multiple times
    for the same candidate + field + reviewer updates in place.
    """
    cid = validate_candidate_id(candidate_id)
    db = get_db()

    # Verify candidate exists
    exists = db.execute(
        "SELECT 1 FROM structured_extractions WHERE candidate_id = ?", (cid,)
    ).fetchone()
    if not exists:
        abort(404, description=f"Candidate {cid} not found")

    data = request.get_json()
    if not data or "reviews" not in data:
        abort(400, description="Request body must contain 'reviews' object")

    reviews = data["reviews"]
    reviewer = data.get("reviewer", "default")
    now = datetime.datetime.now().isoformat()

    saved_count = 0
    errors = []

    for field_key, review_data in reviews.items():
        # ── Validate field key ─────────────────────────────────────────
        if field_key not in TRACKED_FIELDS:
            errors.append(f"Unknown field key: {field_key}")
            continue

        # ── Validate verdict ───────────────────────────────────────────
        verdict = review_data.get("verdict", "").strip().lower()
        if verdict not in VALID_VERDICTS:
            errors.append(f"Invalid verdict for {field_key}: {verdict}")
            continue

        correction = review_data.get("correction", "").strip()
        notes = review_data.get("notes", "").strip()

        try:
            db.execute("""
                INSERT OR REPLACE INTO accuracy_reviews
                    (candidate_id, field_key, verdict, correction, reviewer, reviewed_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cid, field_key, verdict, correction, reviewer, now, notes))
            saved_count += 1
        except sqlite3.Error as e:
            errors.append(f"DB error saving {field_key}: {str(e)}")

    db.commit()

    logger.info(f"💾 Saved {saved_count} reviews for candidate {cid} (reviewer: {reviewer})")

    return jsonify({
        "status": "ok",
        "candidate_id": cid,
        "saved": saved_count,
        "errors": errors,
    })


@app.route("/api/reviews", methods=["GET"])
def api_get_all_reviews():
    """
    📋 Get all reviews, optionally filtered by candidate or field.

    Query params:
      candidate_id  → Filter by candidate
      field_key     → Filter by field (e.g. 'name', 'email')
      verdict       → Filter by verdict (e.g. 'wrong', 'correct')
      reviewer      → Filter by reviewer name
    """
    db = get_db()

    conditions = []
    params = []

    cid = request.args.get("candidate_id")
    if cid:
        conditions.append("candidate_id = ?")
        params.append(int(cid))

    field = request.args.get("field_key")
    if field:
        conditions.append("field_key = ?")
        params.append(field)

    verdict = request.args.get("verdict")
    if verdict:
        conditions.append("verdict = ?")
        params.append(verdict)

    reviewer = request.args.get("reviewer")
    if reviewer:
        conditions.append("reviewer = ?")
        params.append(reviewer)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    rows = db.execute(f"""
        SELECT ar.*, s.name as candidate_name
        FROM accuracy_reviews ar
        LEFT JOIN structured_extractions s ON ar.candidate_id = s.candidate_id
        {where}
        ORDER BY ar.reviewed_at DESC
        LIMIT 5000
    """, params).fetchall()

    return jsonify({
        "reviews": [dict(r) for r in rows],
        "total": len(rows),
    })


@app.route("/api/reviews/<candidate_id>", methods=["GET"])
def api_get_candidate_reviews(candidate_id):
    """🔍 Get all reviews for a specific candidate."""
    cid = validate_candidate_id(candidate_id)
    db = get_db()

    rows = db.execute("""
        SELECT field_key, verdict, correction, reviewer, reviewed_at, notes
        FROM accuracy_reviews
        WHERE candidate_id = ?
        ORDER BY reviewed_at DESC
    """, (cid,)).fetchall()

    # Group by field_key
    reviews = {}
    for r in rows:
        rd = dict(r)
        reviews[rd["field_key"]] = rd

    return jsonify({
        "candidate_id": cid,
        "reviews": reviews,
        "total": len(reviews),
    })


@app.route("/api/reviews", methods=["DELETE"])
def api_reset_reviews():
    """
    🗑️ Reset ALL reviews. Use with caution!

    Expects JSON body: { "confirm": true }
    Without confirmation, this returns a warning instead of deleting.
    """
    data = request.get_json() or {}

    if not data.get("confirm"):
        return jsonify({
            "status": "warning",
            "message": "Send { 'confirm': true } to delete all reviews. This cannot be undone!",
        }), 400

    db = get_db()

    count = db.execute("SELECT COUNT(*) as c FROM accuracy_reviews").fetchone()["c"]
    db.execute("DELETE FROM accuracy_reviews")
    db.commit()

    logger.warning(f"🗑️ All {count} accuracy reviews deleted!")

    return jsonify({
        "status": "ok",
        "deleted": count,
    })


# =============================================================================
# 🌐 API ROUTES — Metrics & Stats
# =============================================================================

@app.route("/api/metrics", methods=["GET"])
def api_get_metrics():
    """
    📊 Get aggregated accuracy metrics across all reviewed candidates.

    Returns:
      - overall accuracy (correct + partial*0.5 / reviewed)
      - counts by verdict
      - field-by-field breakdown
      - critical vs non-critical accuracy
      - top problem fields
      - review progress
    """
    db = get_db()

    # ── Overall counts ─────────────────────────────────────────────────
    verdict_counts = {}
    rows = db.execute("""
        SELECT verdict, COUNT(*) as cnt
        FROM accuracy_reviews
        GROUP BY verdict
    """).fetchall()
    for r in rows:
        verdict_counts[r["verdict"]] = r["cnt"]

    total_reviews = sum(verdict_counts.values())
    correct = verdict_counts.get("correct", 0)
    wrong = verdict_counts.get("wrong", 0)
    partial = verdict_counts.get("partial", 0)
    missing = verdict_counts.get("missing", 0)
    skipped = verdict_counts.get("skip", 0)

    reviewed = total_reviews - skipped
    accuracy = ((correct + partial * 0.5) / reviewed * 100) if reviewed > 0 else 0

    # ── Field-by-field breakdown ───────────────────────────────────────
    field_rows = db.execute("""
        SELECT
            field_key,
            COUNT(*) as total,
            SUM(CASE WHEN verdict = 'correct' THEN 1 ELSE 0 END) as correct,
            SUM(CASE WHEN verdict = 'wrong' THEN 1 ELSE 0 END) as wrong,
            SUM(CASE WHEN verdict = 'partial' THEN 1 ELSE 0 END) as partial,
            SUM(CASE WHEN verdict = 'missing' THEN 1 ELSE 0 END) as missing,
            SUM(CASE WHEN verdict = 'skip' THEN 1 ELSE 0 END) as skipped
        FROM accuracy_reviews
        GROUP BY field_key
        ORDER BY field_key
    """).fetchall()

    field_metrics = {}
    for r in field_rows:
        rd = dict(r)
        field_reviewed = rd["total"] - rd["skipped"]
        field_acc = ((rd["correct"] + rd["partial"] * 0.5) / field_reviewed * 100) if field_reviewed > 0 else 0
        field_info = TRACKED_FIELDS.get(rd["field_key"], {})
        field_metrics[rd["field_key"]] = {
            **rd,
            "accuracy": round(field_acc, 1),
            "label": field_info.get("label", rd["field_key"]),
            "critical": field_info.get("critical", False),
        }

    # ── Critical vs non-critical accuracy ──────────────────────────────
    critical_correct = 0
    critical_total = 0
    noncritical_correct = 0
    noncritical_total = 0

    for field_key, fm in field_metrics.items():
        field_reviewed = fm["total"] - fm["skipped"]
        field_score = fm["correct"] + fm["partial"] * 0.5
        if fm["critical"]:
            critical_correct += field_score
            critical_total += field_reviewed
        else:
            noncritical_correct += field_score
            noncritical_total += field_reviewed

    critical_accuracy = (critical_correct / critical_total * 100) if critical_total > 0 else 0
    noncritical_accuracy = (noncritical_correct / noncritical_total * 100) if noncritical_total > 0 else 0

    # ── Top problem fields (sorted by wrong + missing count) ───────────
    problem_fields = sorted(
        [
            {"field": k, "label": v.get("label", k), "wrong": v["wrong"], "missing": v["missing"],
             "error_rate": round(((v["wrong"] + v["missing"]) / max(1, v["total"] - v["skipped"])) * 100, 1)}
            for k, v in field_metrics.items()
            if (v["wrong"] + v["missing"]) > 0
        ],
        key=lambda x: x["error_rate"],
        reverse=True
    )

    # ── Review progress ────────────────────────────────────────────────
    total_candidates = db.execute(
        "SELECT COUNT(*) as c FROM structured_extractions"
    ).fetchone()["c"]

    reviewed_candidates = db.execute(
        "SELECT COUNT(DISTINCT candidate_id) as c FROM accuracy_reviews"
    ).fetchone()["c"]

    progress = (reviewed_candidates / total_candidates * 100) if total_candidates > 0 else 0

    # ── Corrections log (most recent wrong/partial with corrections) ───
    correction_rows = db.execute("""
        SELECT ar.candidate_id, ar.field_key, ar.verdict, ar.correction,
               ar.reviewed_at, s.name as candidate_name
        FROM accuracy_reviews ar
        LEFT JOIN structured_extractions s ON ar.candidate_id = s.candidate_id
        WHERE ar.verdict IN ('wrong', 'partial') AND ar.correction != ''
        ORDER BY ar.reviewed_at DESC
        LIMIT 50
    """).fetchall()

    return jsonify({
        "overall": {
            "accuracy": round(accuracy, 1),
            "total_reviews": total_reviews,
            "correct": correct,
            "wrong": wrong,
            "partial": partial,
            "missing": missing,
            "skipped": skipped,
            "reviewed": reviewed,
        },
        "field_metrics": field_metrics,
        "critical_accuracy": round(critical_accuracy, 1),
        "noncritical_accuracy": round(noncritical_accuracy, 1),
        "problem_fields": problem_fields[:10],
        "progress": {
            "total_candidates": total_candidates,
            "reviewed_candidates": reviewed_candidates,
            "percentage": round(progress, 1),
        },
        "recent_corrections": [dict(r) for r in correction_rows],
    })


@app.route("/api/stats", methods=["GET"])
def api_get_stats():
    """
    📊 Get a high-level overview of the database.

    Mirrors the db_manager.get_database_stats() method but as an API endpoint.
    This gives the dashboard context about the dataset size and health.
    """
    db = get_db()

    stats = {}

    # Total counts per table
    for table in ["raw_extractions", "structured_extractions", "extraction_log", "accuracy_reviews"]:
        try:
            row = db.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
            stats[f"total_{table}"] = row["c"]
        except sqlite3.OperationalError:
            stats[f"total_{table}"] = 0

    # Extraction status breakdown
    rows = db.execute("""
        SELECT extraction_status, COUNT(*) as cnt
        FROM structured_extractions
        GROUP BY extraction_status
        ORDER BY cnt DESC
    """).fetchall()
    stats["status_breakdown"] = {r["extraction_status"]: r["cnt"] for r in rows}

    # AI vs Regex breakdown
    row = db.execute("""
        SELECT
            SUM(CASE WHEN ai_assisted = 1 THEN 1 ELSE 0 END) as ai_count,
            SUM(CASE WHEN ai_assisted = 0 THEN 1 ELSE 0 END) as regex_count
        FROM structured_extractions
    """).fetchone()
    stats["ai_assisted_count"] = row["ai_count"] or 0
    stats["regex_only_count"] = row["regex_count"] or 0

    # Field completeness (what % of candidates have each field filled?)
    total = stats.get("total_structured_extractions", 0)
    fields_to_check = ["name", "email", "phone", "date_of_birth",
                       "skills_raw", "experience_raw", "education_raw",
                       "languages", "certifications", "summary"]

    completeness = {}
    for field in fields_to_check:
        if total > 0:
            filled = db.execute(f"""
                SELECT COUNT(*) as c FROM structured_extractions
                WHERE {field} IS NOT NULL AND {field} != ''
            """).fetchone()["c"]
            completeness[field] = round((filled / total) * 100, 1)
        else:
            completeness[field] = 0.0
    stats["field_completeness_pct"] = completeness

    return jsonify(stats)

# =============================================================================
# 🔬 API ROUTES — ML Quality Audit
# =============================================================================

@app.route("/api/ml-audit", methods=["GET"])
def api_ml_audit():
    """
    🔬 Run a comprehensive ML Quality Audit on all reviewed candidates.

    Performs 5 analyses:
      1. Core NER Metrics (P/R/F1 per field + weighted average)
      2. Error Taxonomy (Boundary / Type / Missing / Spurious)
      3. Field Sensitivity (weakest links + root cause)
      4. Textual Similarity (Levenshtein fuzzy matching)
      5. Improvement Roadmap (prioritized recommendations)

    Query params:
      reviewer  → optional filter by reviewer name

    Returns: Comprehensive audit JSON with all 5 analyses.
    """
    if not ML_AUDIT_AVAILABLE:
        return jsonify({
            "error": "ML audit module not found. "
                     "Ensure ml_quality_audit.py is in the same directory.",
            "available": False,
        }), 503

    db = get_db()
    reviewer_filter = request.args.get("reviewer", "")

    # ── Fetch all reviews ──────────────────────────────────────────────
    if reviewer_filter:
        review_rows = db.execute("""
            SELECT candidate_id, field_key, verdict, correction,
                   reviewer, reviewed_at, notes
            FROM accuracy_reviews
            WHERE reviewer = ?
            ORDER BY candidate_id, field_key
        """, (reviewer_filter,)).fetchall()
    else:
        review_rows = db.execute("""
            SELECT candidate_id, field_key, verdict, correction,
                   reviewer, reviewed_at, notes
            FROM accuracy_reviews
            ORDER BY candidate_id, field_key
        """).fetchall()

    reviews = [dict(r) for r in review_rows]

    if len(reviews) == 0:
        return jsonify({
            "error": "No reviews found. Review some candidates first!",
            "available": True,
            "total_reviews": 0,
        })

    # ── Fetch structured extractions for reviewed candidates ───────────
    reviewed_cids = list(set(r["candidate_id"] for r in reviews))
    placeholders = ",".join("?" * len(reviewed_cids))
    extraction_rows = db.execute(f"""
        SELECT * FROM structured_extractions
        WHERE candidate_id IN ({placeholders})
    """, reviewed_cids).fetchall()

    extractions = {row["candidate_id"]: dict(row) for row in extraction_rows}

    # ── Run the full audit! ────────────────────────────────────────────
    try:
        result = run_full_audit(reviews, extractions, TRACKED_FIELDS)
        result["available"] = True
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ ML audit failed: {e}")
        return jsonify({"error": f"Audit failed: {str(e)}", "available": True}), 500


# =============================================================================
# 🌐 API ROUTES — File Serving
# =============================================================================

def _resolve_candidate_file(candidate_id: int) -> Optional[Dict[str, Any]]:
    """
    Locate the original resume file for a candidate on disk.

    The `filenames` column in raw_extractions stores the original
    filename(s) as a string (sometimes semicolon-separated for
    multi-file candidates). We try MULTIPLE strategies to find the
    file, because resumes can move around like divas between dressing rooms! 🗄️🔍

    Search strategy (in order — first match wins):
      1. folder_path + filename  → The EXACT original location from extraction
      2. FILES_DIR + filename    → The configured flat resume folder
      3. FILES_DIR recursive     → Walk subdirectories looking for the filename
      4. Case-insensitive match  → Windows is case-insensitive, filenames drift!

    Returns:
        Dict with: {path, filename, ext, size_bytes} or None if not found.
    """
    db = get_db()

    # ── Fetch BOTH filenames AND folder_path from the database ─────
    # 💅 THE MISSING CLUE!
    # folder_path stores the EXACT directory where the file lived at
    # extraction time. The old code ignored this column entirely —
    # like having a GPS coordinate but only using the city name! 🗺️✨
    row = db.execute("""
        SELECT filenames, folder_path FROM raw_extractions
        WHERE candidate_id = ?
        ORDER BY extraction_timestamp DESC LIMIT 1
    """, (candidate_id,)).fetchone()

    if not row or not row["filenames"]:
        return None

    # filenames can be "resume.pdf" or "resume.pdf; cover.pdf"
    raw_filenames = row["filenames"]
    candidates_filenames = [
        f.strip() for f in re.split(r'[;,|]', raw_filenames)
        if f.strip()
    ]

    # ── Helper: build result dict from a valid path ────────────────
    def _make_result(full_path: str) -> Dict[str, Any]:
        """Package a found file into the standard result dict."""
        size = os.path.getsize(full_path)
        safe_filename = os.path.basename(full_path)
        _, ext = os.path.splitext(safe_filename)
        return {
            "path": full_path,
            "filename": safe_filename,
            "ext": ext.lower(),
            "size_bytes": size,
        }

    files_dir = os.path.abspath(FILES_DIR)
    folder_path = row["folder_path"] if row["folder_path"] else None

    for filename in candidates_filenames:
        # Security: strip any path traversal attempts — no "../../../etc/passwd"
        # drama on this stage, darling! 🚫
        safe_filename = os.path.basename(filename)

        # ── Strategy 1: folder_path + filename ─────────────────────
        # The ORIGINAL extraction path — most reliable if the file
        # hasn't been moved since extraction! Like checking the last
        # known address first! 🏠✨
        if folder_path:
            # folder_path might be the candidate folder (containing the file)
            # or the parent folder. Try both patterns:
            original_path = os.path.join(folder_path, safe_filename)
            if os.path.isfile(original_path):
                return _make_result(original_path)

            # Also try: folder_path might BE the file directory itself
            # (for individual file processing vs folder processing)
            parent_of_folder = os.path.dirname(folder_path)
            alt_path = os.path.join(parent_of_folder, safe_filename)
            if os.path.isfile(alt_path):
                return _make_result(alt_path)

        # ── Strategy 2: FILES_DIR + filename (flat lookup) ─────────
        # The configured resume directory — works when all resumes
        # are dumped in one flat folder! 📁
        flat_path = os.path.join(files_dir, safe_filename)
        if os.path.isfile(flat_path):
            return _make_result(flat_path)

        # ── Strategy 3: Case-insensitive search in FILES_DIR ───────
        # Windows filenames are case-insensitive, but Python's
        # os.path.isfile() might miss "Resume.PDF" vs "resume.pdf"
        # on some filesystem configs. Like a bouncer who doesn't
        # recognise you without your stage makeup! 💄
        try:
            lower_target = safe_filename.lower()
            for entry in os.scandir(files_dir):
                if entry.is_file() and entry.name.lower() == lower_target:
                    return _make_result(entry.path)
        except OSError:
            pass  # Directory might not exist — non-fatal

        # ── Strategy 4: Recursive subdirectory search ──────────────
        # If resumes are in subfolders (batch1/, 2024/, etc.),
        # walk the tree to find them. Like sending a search party
        # through every room in the mansion! 🏰🔍
        #
        # ⚡ PERFORMANCE: os.walk can be slow on huge directories.
        # We cap the depth to 3 levels and stop on first match.
        try:
            for root, dirs, files in os.walk(files_dir):
                # ── Depth limiter: max 3 levels deep ───────────────
                # Without this, a deeply nested folder structure could
                # make the search take forever. Three levels covers:
                # resumes/batch/file.pdf or resumes/year/month/file.pdf
                depth = root[len(files_dir):].count(os.sep)
                if depth >= 3:
                    dirs.clear()  # Stop descending deeper
                    continue

                for f in files:
                    if f.lower() == lower_target:
                        return _make_result(os.path.join(root, f))
        except OSError:
            pass  # Walk failure — non-fatal

    # ── All strategies exhausted — file truly not found ────────────
    # Log what we tried so debugging is easier! 📋
    logger.debug(
        f"📎 File not found for candidate {candidate_id}: "
        f"tried filenames={candidates_filenames}, "
        f"folder_path={folder_path}, files_dir={files_dir}"
    )
    return None


@app.route("/api/candidates/<candidate_id>/file", methods=["GET"])
def api_serve_candidate_file(candidate_id):
    """
    📎 Serve the original resume file for a candidate.

    Used by the preview modal to display PDFs inline or offer
    DOCX/DOC files as downloads. Think of it as the document
    concierge — she fetches the original from the filing room! 🗂️✨

    Query params:
      download=1  → Force download (Content-Disposition: attachment)

    Returns:
      The actual file bytes with appropriate Content-Type headers.
      404 if file not found on disk.
      413 if file exceeds MAX_FILE_SIZE_MB.
    """
    cid = validate_candidate_id(candidate_id)

    file_info = _resolve_candidate_file(cid)
    if not file_info:
        return jsonify({
            "error": "File not found on disk",
            "candidate_id": cid,
            "files_dir": os.path.abspath(FILES_DIR),
            "tip": f"Set RESUME_FILES_DIR env var to your resumes folder.",
        }), 404

    # ── Size guard ─────────────────────────────────────────────────
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_info["size_bytes"] > max_bytes:
        return jsonify({
            "error": f"File too large to serve ({file_info['size_bytes'] // 1024 // 1024}MB). "
                     f"Max: {MAX_FILE_SIZE_MB}MB",
        }), 413

    # ── Determine Content-Type ─────────────────────────────────────
    ext = file_info["ext"]
    mime_map = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc":  "application/msword",
        ".odt":  "application/vnd.oasis.opendocument.text",
        ".rtf":  "application/rtf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    content_type = mime_map.get(ext, "application/octet-stream")

    # ── Inline vs attachment ────────────────────────────────────────
    # PDFs and images can be displayed inline in the browser iframe.
    # Word docs must be downloaded — browsers can't render them natively.
    # Like VIP table vs takeaway — both get their food, different service! 🍽️
    force_download = request.args.get("download", "0") == "1"
    is_inline = ext in RENDERABLE_TYPES | IMAGE_TYPES and not force_download
    disposition = "inline" if is_inline else "attachment"

    logger.info(
        f"📎 Serving file for candidate {cid}: "
        f"{file_info['filename']} ({file_info['size_bytes']} bytes, {disposition})"
    )

    try:
        from flask import send_file
        return send_file(
            file_info["path"],
            mimetype=content_type,
            as_attachment=not is_inline,
            download_name=file_info["filename"],
        )
    except Exception as e:
        logger.error(f"❌ Failed to serve file for candidate {cid}: {e}")
        return jsonify({"error": f"Failed to read file: {e}"}), 500


@app.route("/api/candidates/<candidate_id>/file/info", methods=["GET"])
def api_candidate_file_info(candidate_id):
    """
    📋 Get metadata about a candidate's file without serving it.

    Used by the frontend to check if a file exists and what type it is
    BEFORE trying to render it — so we can show the right UI!
    Like calling ahead to check the restaurant is open! 📞✨
    """
    cid = validate_candidate_id(candidate_id)
    file_info = _resolve_candidate_file(cid)

    if not file_info:
        return jsonify({
            "found": False,
            "candidate_id": cid,
            "files_dir": os.path.abspath(FILES_DIR),
        })

    ext = file_info["ext"]
    return jsonify({
        "found": True,
        "candidate_id": cid,
        "filename": file_info["filename"],
        "ext": ext,
        "size_bytes": file_info["size_bytes"],
        "size_mb": round(file_info["size_bytes"] / 1024 / 1024, 2),
        "renderable": ext in RENDERABLE_TYPES,  # Can iframe show it?
        "is_image": ext in IMAGE_TYPES,
        "downloadable": ext in DOWNLOADABLE_TYPES,
        "serve_url": f"/api/candidates/{cid}/file",
        "download_url": f"/api/candidates/{cid}/file?download=1",
    })

# =============================================================================
# 🌐 API ROUTES — Vector Search
# =============================================================================

@app.route("/api/search", methods=["POST"])
def api_search_candidates():
    """
    🔍 Search for candidates matching a job description!

    Expects JSON body:
    {
      "job_description": "We are looking for a Senior Python Developer...",
      "top_k": 10,
      "skill_filter": ["Python", "AWS"],    ← optional
      "location_filter": "Singapore"         ← optional
    }

    Returns ranked list of matching candidates with similarity scores.
    """
    engine = get_vector_engine()
    if engine is None:
        return jsonify({
            "error": "Vector search not available. Install dependencies:\n"
                     "  pip install qdrant-client --break-system-packages\n"
                     "  pip install ollama --break-system-packages\n"
                     "  ollama pull nomic-embed-text",
            "vector_search_available": False,
        }), 503

    data = request.get_json()
    if not data or not data.get("job_description"):
        abort(400, description="Request must contain 'job_description' field")

    jd = data["job_description"]
    top_k = min(data.get("top_k", 10), 50)
    skill_filter = data.get("skill_filter")
    location_filter = data.get("location_filter")

    results = engine.search(
        job_description=jd,
        top_k=top_k,
        skill_filter=skill_filter,
        location_filter=location_filter,
    )

    return jsonify(results)


@app.route("/api/search/index", methods=["POST"])
def api_index_candidates():
    """
    📥 Index all candidates into the vector database.

    Call this once after initial setup, or after adding new candidates.
    Skips already-indexed candidates for efficiency.

    Optional JSON body:
    { "rebuild": true }  ← drops and rebuilds the entire index
    """
    engine = get_vector_engine()
    if engine is None:
        return jsonify({"error": "Vector search not available"}), 503

    data = request.get_json() or {}

    if data.get("rebuild"):
        stats = engine.rebuild_index()
    else:
        stats = engine.index_all_candidates()

    return jsonify({"status": "ok", **stats})


@app.route("/api/search/stats", methods=["GET"])
def api_search_stats():
    """📊 Get vector database stats."""
    engine = get_vector_engine()
    if engine is None:
        return jsonify({
            "vector_search_available": False,
            "message": "Install: pip install qdrant-client ollama; ollama pull nomic-embed-text"
        })

    stats = engine.get_stats()
    stats["vector_search_available"] = True
    return jsonify(stats)


# =============================================================================
# 🎨 DASHBOARD — Served directly by Flask (no React/npm needed!)
# =============================================================================
# Same pattern as annotation_tool.py and review_dashboard.py:
# Flask serves the HTML, which calls our own /api/ endpoints.
# No CORS issues because it's the SAME origin! 🎉

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧚‍♀️ AI Extraction Accuracy Validator</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  /* ── CSS RESET & VARIABLES ──────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:        #0c0a14; --bg-card:   #151221; --bg-panel:  #110e1d;
    --bg-input:  #1a1630; --border:    #2a2545; --border-f:  #a855f7;
    --text:      #f0ecf9; --text-sec:  #9b8fc4; --text-mut:  #6b5f8a;
    --accent:    #c084fc; --accent-dk: #7c3aed; --glow: rgba(168,85,247,0.15);
    --gold:      #fbbf24; --pink:      #ec4899; --teal: #2dd4bf;
    --ok:        #22c55e; --bad:       #ef4444; --warn: #f59e0b;
    --purple:    #8b5cf6;
    --mono: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    --display: 'Playfair Display', 'Georgia', serif;
    --body: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Playfair+Display:wght@700;800&display=swap');

  body {
    background: linear-gradient(180deg, var(--bg) 0%, #0f0b1a 100%);
    color: var(--text); font-family: var(--body);
    min-height: 100vh; overflow-x: hidden;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* ── HEADER ─────────────────────────────────────────────────────── */
  .header {
    background: linear-gradient(135deg, var(--bg-card) 0%, #1a1040 100%);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 10px;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .header-left .icon { font-size: 28px; }
  .header-left h1 {
    font-family: var(--display); font-size: 1.2rem; color: var(--accent);
    font-weight: 700; letter-spacing: -0.02em;
  }
  .header-left .sub {
    font-size: 10px; color: var(--text-mut); font-family: var(--mono);
  }
  .tab-bar {
    display: flex; gap: 3px; background: var(--bg-input);
    border-radius: 8px; padding: 2px;
  }
  .tab-btn {
    background: transparent; border: none; border-radius: 6px;
    padding: 6px 14px; color: var(--text-sec); font-size: 11px;
    cursor: pointer; font-weight: 500; transition: all 0.2s;
  }
  .tab-btn.active { background: var(--accent-dk); color: #fff; font-weight: 700; }

  /* ── SEARCH BAR ─────────────────────────────────────────────────── */
  .search-bar {
    padding: 16px 24px; display: flex; gap: 10px; align-items: center;
    border-bottom: 1px solid var(--border);
  }
  .search-bar input {
    flex: 1; max-width: 400px; background: var(--bg-input);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 16px; color: var(--text); font-size: 13px;
    font-family: var(--mono); outline: none;
  }
  .search-bar input:focus { border-color: var(--border-f); }
  .search-bar .stats { font-size: 11px; color: var(--text-mut); }

  /* ── BUTTONS ────────────────────────────────────────────────────── */
  .btn {
    border: none; border-radius: 8px; padding: 8px 18px;
    font-size: 12px; cursor: pointer; font-weight: 600;
    transition: all 0.2s; display: inline-flex; align-items: center; gap: 5px;
  }
  .btn-primary { background: var(--accent-dk); color: #fff; }
  .btn-primary:hover { background: var(--accent); }
  .btn-outline {
    background: transparent; border: 1px solid var(--border);
    color: var(--text-sec);
  }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent); }
  .btn-danger { background: transparent; border: 1px solid rgba(239,68,68,0.3); color: var(--bad); }
  .btn-sm { padding: 4px 10px; font-size: 11px; }

  /* ── CANDIDATE GRID ─────────────────────────────────────────────── */
  .candidates-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 10px; padding: 16px 24px;
  }
  .candidate-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; cursor: pointer;
    transition: all 0.15s; text-align: left;
  }
  .candidate-card:hover {
    border-color: var(--accent); background: var(--glow);
  }
  .candidate-card .cid { font-size: 14px; font-weight: 700; color: var(--text); }
  .candidate-card .name { font-size: 13px; color: var(--accent); font-weight: 600; margin: 4px 0; }
  .candidate-card .email { font-size: 11px; color: var(--text-sec); }
  .candidate-card .meta { display: flex; gap: 8px; margin-top: 8px; font-size: 10px; color: var(--text-mut); }
  .badge {
    font-size: 10px; font-family: var(--mono); padding: 2px 8px;
    border-radius: 4px; font-weight: 600;
  }
  .badge-ok { background: rgba(34,197,94,0.12); color: var(--ok); }
  .badge-pending { background: rgba(107,95,138,0.15); color: var(--text-mut); }

  /* ── PAGINATION ─────────────────────────────────────────────────── */
  .pagination {
    display: flex; justify-content: center; gap: 6px;
    padding: 16px 24px; font-size: 11px;
  }

  /* ── SPLIT PANELS (Review Detail) ───────────────────────────────── */
  .back-bar {
    background: var(--bg-panel); border-bottom: 1px solid var(--border);
    padding: 8px 24px; display: flex; align-items: center;
    justify-content: space-between;
  }
  .back-bar .info { display: flex; align-items: center; gap: 12px; }
  .back-bar .cname { font-size: 13px; font-weight: 700; color: var(--accent); }
  .back-bar .progress { font-size: 10px; color: var(--text-mut); font-family: var(--mono); }

  .split-panels { display: flex; height: calc(100vh - 150px); }

  .panel-left {
    flex: 1 1 42%; border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
  }
  .panel-right {
    flex: 1 1 58%; display: flex; flex-direction: column;
  }
  .panel-header {
    padding: 8px 16px; background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    font-size: 11px; font-weight: 700;
  }
  .panel-header.teal { color: var(--teal); }
  .panel-header.accent { color: var(--accent); }
  .panel-body {
    flex: 1; overflow: auto; padding: 14px 16px;
  }
  .raw-text {
    font-family: var(--mono); font-size: 11.5px; line-height: 1.7;
    color: var(--text-sec); white-space: pre-wrap; word-break: break-word;
    background: var(--bg);
  }
  .raw-text mark {
    background: rgba(168,85,247,0.3); color: var(--accent);
    border-radius: 3px; padding: 1px 3px;
    border: 1px solid rgba(168,85,247,0.25);
  }

  /* ── FIELD CARDS ────────────────────────────────────────────────── */
  .field-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
    cursor: pointer; transition: all 0.15s;
  }
  .field-card.active { background: var(--glow); border-color: rgba(168,85,247,0.4); }
  .field-card .field-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 4px;
  }
  .field-card .field-name {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 700;
  }
  .field-card .field-name .critical {
    font-size: 8px; color: var(--gold); margin-left: 5px;
    background: rgba(251,191,36,0.1); padding: 2px 5px; border-radius: 3px;
  }
  .field-card .field-value {
    background: var(--bg-input); border-radius: 5px; padding: 6px 10px;
    font-size: 11px; line-height: 1.5; font-family: var(--mono);
    color: var(--text); max-height: 80px; overflow: auto; word-break: break-word;
  }
  .field-card .field-value.empty { color: var(--text-mut); font-style: italic; }
  .field-card .field-value.long { font-family: var(--body); }

  .verdict-badge {
    font-size: 9px; font-weight: 700; padding: 2px 8px;
    border-radius: 5px; font-family: var(--mono);
  }

  /* ── VERDICT BUTTONS ────────────────────────────────────────────── */
  .verdict-row { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 8px; }
  .verdict-btn {
    background: transparent; border: 1.5px solid var(--border);
    border-radius: 8px; padding: 6px 14px; color: var(--text-mut);
    font-size: 12px; font-weight: 500; cursor: pointer;
    transition: all 0.2s; display: flex; align-items: center; gap: 4px;
    font-family: var(--mono);
  }
  .verdict-btn.active { font-weight: 700; }
  .verdict-btn.correct       { border-color: var(--ok);   color: var(--ok);   background: rgba(34,197,94,0.08); }
  .verdict-btn.wrong         { border-color: var(--bad);  color: var(--bad);  background: rgba(239,68,68,0.08); }
  .verdict-btn.partial       { border-color: var(--warn); color: var(--warn); background: rgba(245,158,11,0.08); }
  .verdict-btn.missing       { border-color: var(--purple);color: var(--purple);background: rgba(139,92,246,0.08); }
  .verdict-btn.skip          { border-color: var(--text-mut);color: var(--text-mut);background: rgba(100,116,139,0.08);}

  .correction-input {
    width: 100%; background: var(--bg-input); border: 1px solid var(--border-f);
    border-radius: 5px; padding: 6px 10px; color: var(--text);
    font-size: 11px; font-family: var(--mono); outline: none; margin-top: 6px;
  }
  .shortcuts { font-size: 9px; color: var(--text-mut); margin-top: 5px; }
  .shortcuts b { color: var(--accent); }

  /* ── METRICS ────────────────────────────────────────────────────── */
  .metrics-wrap { padding: 20px 24px; max-width: 960px; margin: 0 auto; }
  .stats-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-panel) 100%);
    border: 1px solid var(--border); border-radius: 14px; padding: 14px 18px;
    flex: 1; min-width: 120px;
  }
  .stat-card .label {
    font-size: 10px; color: var(--text-mut); text-transform: uppercase;
    letter-spacing: 0.08em; font-family: var(--mono);
  }
  .stat-card .value {
    font-size: 26px; font-weight: 800; font-family: var(--display);
    line-height: 1.2; margin-top: 3px;
  }
  .stat-card .sub { font-size: 10px; color: var(--text-sec); }

  .field-table {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; margin-bottom: 20px;
  }
  .field-table .table-header {
    padding: 12px 18px; border-bottom: 1px solid var(--border);
    font-size: 13px; font-weight: 700; color: var(--accent);
  }
  .field-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 18px; border-bottom: 1px solid rgba(42,37,69,0.15);
  }
  .field-row .fname { width: 120px; font-size: 11px; font-weight: 600; }
  .field-row .bar-wrap { flex: 1; height: 16px; background: var(--bg-input); border-radius: 4px; overflow: hidden; display: flex; }
  .field-row .bar-ok   { background: var(--ok); }
  .field-row .bar-warn { background: var(--warn); }
  .field-row .bar-bad  { background: var(--bad); }
  .field-row .bar-miss { background: var(--purple); }
  .field-row .pct {
    width: 45px; text-align: right; font-size: 11px; font-weight: 700;
    font-family: var(--mono);
  }
  .field-row .counts {
    display: flex; gap: 4px; font-size: 9px; font-family: var(--mono); min-width: 120px;
  }

  /* ── STATUS BAR ─────────────────────────────────────────────────── */
  .status-bar {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
    background: linear-gradient(90deg, var(--bg-card), #1a1040);
    border-top: 1px solid var(--border); padding: 6px 24px;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 10px; color: var(--text-mut); font-family: var(--mono);
  }
  .status-bar .dot {
    width: 6px; height: 6px; border-radius: 50%; display: inline-block;
    margin-right: 6px;
  }
  .dot-ok { background: var(--ok); }
  .dot-bad { background: var(--bad); }

  /* ── ML AUDIT TAB ────────────────────────────────────────────────── */
  .audit-wrap { padding: 20px 24px; max-width: 1100px; margin: 0 auto; }
  .audit-section {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; margin-bottom: 20px; overflow: hidden;
  }
  .audit-section-header {
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    font-size: 14px; font-weight: 700; color: var(--accent);
    display: flex; align-items: center; gap: 8px;
  }
  .audit-table {
    width: 100%; border-collapse: collapse; font-size: 11px;
  }
  .audit-table th {
    text-align: left; padding: 8px 14px; font-size: 10px;
    color: var(--text-mut); text-transform: uppercase;
    letter-spacing: 0.06em; font-family: var(--mono);
    border-bottom: 1px solid var(--border); background: var(--bg-panel);
  }
  .audit-table td {
    padding: 7px 14px; border-bottom: 1px solid rgba(42,37,69,0.2);
    font-family: var(--mono);
  }
  .audit-table tr:hover { background: var(--glow); }
  .severity-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
  }
  .sev-excellent { background: var(--ok); }
  .sev-good { background: var(--teal); }
  .sev-needs_improvement, .sev-at_risk { background: var(--warn); }
  .sev-poor, .sev-critical { background: var(--bad); }
  .sev-healthy { background: var(--ok); }
  .roadmap-card {
    padding: 14px 20px; border-bottom: 1px solid rgba(42,37,69,0.2);
  }
  .roadmap-card:last-child { border-bottom: none; }
  .roadmap-title {
    font-size: 13px; font-weight: 700; display: flex;
    align-items: center; gap: 8px; margin-bottom: 6px;
  }
  .roadmap-detail { font-size: 11px; color: var(--text-sec); line-height: 1.6; }
  .roadmap-meta {
    margin-top: 6px; font-size: 10px; font-family: var(--mono);
    color: var(--text-mut); display: flex; gap: 14px;
  }
  .chart-box { padding: 16px 20px; }
  .chart-box canvas { max-height: 280px; }

  /* ── LOADING / EMPTY ────────────────────────────────────────────── */
  .loading { text-align: center; padding: 40px; color: var(--text-sec); }
  .empty { text-align: center; padding: 60px; color: var(--text-sec); }
  .saving-indicator { color: var(--gold); font-size: 10px; }

  /* ── PREVIEW MODAL ──────────────────────────────────────────── */
  .modal-backdrop {
    position: fixed; inset: 0; z-index: 500;
    background: rgba(8,6,18,0.82);
    backdrop-filter: blur(4px);
    display: flex; align-items: center; justify-content: center;
    padding: 20px;
    animation: fadeIn 0.15s ease;
  }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  .modal-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    width: 100%; max-width: 780px;
    max-height: 88vh;
    display: flex; flex-direction: column;
    box-shadow: 0 24px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(168,85,247,0.15);
    animation: slideUp 0.18s ease;
    transition: max-width 0.25s ease, max-height 0.25s ease;
  }
  /* 💅 EXPANDED MODE — When showing PDFs/DOCX, the modal goes FULL DIVA!
     More width for the document, more height for scrolling. Like upgrading
     from economy to first class — the document DESERVES the legroom! ✈️✨ */
  .modal-box.modal-expanded {
    max-width: 1100px;
    max-height: 95vh;
  }
  @keyframes slideUp {
    from { transform: translateY(18px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
  }

  .modal-header {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    flex-shrink: 0;
  }
  .modal-header .modal-title {
    display: flex; flex-direction: column; gap: 3px;
  }
  .modal-header .modal-name {
    font-size: 15px; font-weight: 700; color: var(--accent);
  }
  .modal-header .modal-meta {
    font-size: 10px; color: var(--text-mut); font-family: var(--mono);
  }
  .modal-close {
    background: var(--bg-input); border: 1px solid var(--border);
    border-radius: 8px; width: 32px; height: 32px;
    color: var(--text-sec); font-size: 16px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s; flex-shrink: 0;
  }
  .modal-close:hover { border-color: var(--bad); color: var(--bad); }

  .modal-tabs {
    display: flex; gap: 2px; padding: 8px 14px 0;
    border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  .modal-tab {
    background: transparent; border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 14px; font-size: 11px; font-weight: 600;
    color: var(--text-mut); cursor: pointer; transition: all 0.15s;
    margin-bottom: -1px;
  }
  .modal-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .modal-body {
    flex: 1; overflow: auto; padding: 16px 20px;
  }
  .modal-raw {
    font-family: var(--mono); font-size: 11.5px; line-height: 1.75;
    color: var(--text-sec); white-space: pre-wrap; word-break: break-word;
  }
  .modal-structured {}
  .modal-field-row {
    display: flex; gap: 10px; padding: 7px 0;
    border-bottom: 1px solid rgba(42,37,69,0.4); font-size: 12px;
  }
  .modal-field-row:last-child { border-bottom: none; }
  .modal-field-label {
    width: 120px; flex-shrink: 0;
    font-size: 10px; font-weight: 700; color: var(--text-mut);
    text-transform: uppercase; letter-spacing: 0.06em; padding-top: 2px;
  }
  .modal-field-value {
    flex: 1; color: var(--text); line-height: 1.55; word-break: break-word;
  }
  .modal-field-value.empty { color: var(--text-mut); font-style: italic; font-size: 11px; }
  .modal-field-value.mono { font-family: var(--mono); font-size: 11px; }
  .modal-loading {
    text-align: center; padding: 40px; color: var(--text-sec); font-size: 13px;
  }
  .modal-footer {
    padding: 10px 20px; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    flex-shrink: 0;
  }
  .modal-footer .char-count { font-size: 10px; color: var(--text-mut); font-family: var(--mono); }

  /* ── DOCUMENT VIEWER ─────────────────────────────────────────── */
  .doc-viewer-wrap {
    display: flex; flex-direction: column;
    height: 100%; min-height: 480px;
  }
  .doc-iframe {
    flex: 1; width: 100%; border: none;
    border-radius: 8px; background: #fff;
    min-height: 480px;
  }
  .doc-image {
    max-width: 100%; border-radius: 8px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    display: block; margin: 0 auto;
  }
  .doc-download-card {
    background: var(--bg-input); border: 1px solid var(--border);
    border-radius: 12px; padding: 28px 24px; text-align: center;
    margin: 20px 0;
  }
  .doc-download-card .doc-icon { font-size: 52px; margin-bottom: 12px; }
  .doc-download-card .doc-name {
    font-size: 14px; font-weight: 700; color: var(--text);
    margin-bottom: 6px; word-break: break-all;
  }
  .doc-download-card .doc-size {
    font-size: 11px; color: var(--text-mut);
    font-family: var(--mono); margin-bottom: 18px;
  }
  .doc-not-found {
    text-align: center; padding: 40px 20px; color: var(--text-sec);
  }
  .doc-not-found .nf-icon { font-size: 48px; margin-bottom: 12px; }
  .doc-not-found .nf-title {
    font-size: 15px; font-weight: 700; color: var(--text-mut);
    margin-bottom: 8px;
  }
  .doc-not-found .nf-tip {
    font-size: 11px; color: var(--text-mut); font-family: var(--mono);
    background: var(--bg-input); border-radius: 6px; padding: 10px 14px;
    display: inline-block; margin-top: 10px; text-align: left;
    line-height: 1.7;
  }

  /* ── SCROLLBAR ──────────────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--accent-dk); }

  /* ── DOCX INLINE PREVIEW ─────────────────────────────────────── */
  /* 💅 mammoth.js converts DOCX → HTML in the browser!
     This container makes it look polished and readable,
     like reading the resume on crisp white paper! 📄✨        */
  .docx-preview-wrap {
    display: flex; flex-direction: column;
    height: 100%; min-height: 480px;
  }
  .docx-preview-toolbar {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 8px;
  }
  .docx-preview-toolbar span {
    font-size: 11px; color: var(--text-mut); font-family: var(--mono);
  }
  .docx-preview-toolbar a {
    font-size: 11px; color: var(--accent); text-decoration: none;
  }
  .docx-preview-toolbar a:hover { text-decoration: underline; }
  .docx-html-container {
    flex: 1; overflow-y: auto; overflow-x: hidden;
    background: #fff; color: #1a1a2e; border-radius: 8px;
    padding: 28px 32px; min-height: 480px;
    font-family: 'Segoe UI', 'Calibri', Arial, sans-serif;
    font-size: 13px; line-height: 1.7;
  }
  /* ── Tame mammoth.js HTML output ──────────────────────────────
     mammoth generates clean but unstyled HTML — we add just enough
     polish so it looks like a real Word document preview! 📝     */
  .docx-html-container h1 { font-size: 20px; margin: 16px 0 8px; color: #0d1117; }
  .docx-html-container h2 { font-size: 16px; margin: 14px 0 6px; color: #1a1a2e; }
  .docx-html-container h3 { font-size: 14px; margin: 12px 0 4px; color: #24292f; }
  .docx-html-container p  { margin: 4px 0 8px; }
  .docx-html-container ul,
  .docx-html-container ol { margin: 6px 0 10px 20px; padding: 0; }
  .docx-html-container li { margin-bottom: 3px; }
  .docx-html-container table {
    border-collapse: collapse; width: 100%; margin: 10px 0;
    font-size: 12px;
  }
  .docx-html-container table td,
  .docx-html-container table th {
    border: 1px solid #d0d7de; padding: 6px 10px; text-align: left;
  }
  .docx-html-container table th {
    background: #f0f3f6; font-weight: 700;
  }
  .docx-html-container strong,
  .docx-html-container b { font-weight: 700; }
  .docx-html-container a { color: #0969da; }
  /* ── Raw text fallback for .doc files ──────────────────────── */
  .doc-raw-fallback {
    flex: 1; overflow-y: auto; overflow-x: hidden;
    background: var(--bg-input); color: var(--text);
    border-radius: 8px; padding: 24px 28px;
    min-height: 480px; white-space: pre-wrap;
    word-wrap: break-word;
    font-family: var(--mono); font-size: 12px; line-height: 1.7;
  }
</style>
<!-- 📘 mammoth.js — converts DOCX binary to HTML in-browser!
     No server-side conversion needed. Think of it as a tiny
     Microsoft Word viewer living inside your browser tab! 🎉 -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mammoth/1.8.0/mammoth.browser.min.js"></script>
</head>
<body>

<!-- ── HEADER ────────────────────────────────────────────────────── -->
<div class="header">
  <div class="header-left">
    <span class="icon">🧚‍♀️</span>
    <div>
      <h1>AI Extraction Accuracy Validator</h1>
      <div class="sub">v2.0 • Flask-Served Dashboard • AiMerlion QA</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('review')" id="tab-review">📋 Review</button>
      <button class="tab-btn" onclick="switchTab('metrics')" id="tab-metrics">📊 Metrics</button>
      <button class="tab-btn" onclick="switchTab('search')" id="tab-search">🔍 JD Search</button>
      <button class="tab-btn" onclick="switchTab('audit')" id="tab-audit">🔬 ML Audit</button>
    </div>
    <button class="btn btn-danger btn-sm" onclick="resetAll()">🗑️ Reset</button>
  </div>
</div>

<!-- ── REVIEW: CANDIDATE LIST ───────────────────────────────────── -->
<div id="view-list">
  <div class="search-bar">
    <input type="text" id="searchInput" placeholder="Search by name, email, or ID..."
           onkeydown="if(event.key==='Enter')searchCandidates()">
    <button class="btn btn-primary" onclick="searchCandidates()">🔍 Search</button>
    <span class="stats" id="listStats"></span>
  </div>
  <div class="candidates-grid" id="candidatesGrid"></div>
  <div class="pagination" id="pagination"></div>
</div>

<!-- ── REVIEW: CANDIDATE DETAIL ─────────────────────────────────── -->
<div id="view-detail" style="display:none;">
  <div class="back-bar">
    <button class="btn btn-outline btn-sm" onclick="backToList()">← Back to List</button>
    <div class="info">
      <span class="cname" id="detailName"></span>
      <span class="progress" id="detailProgress"></span>
      <span class="saving-indicator" id="savingIndicator" style="display:none;">💾 Saving...</span>
    </div>
    <!-- 📎 View Document — opens the preview modal directly on the Document tab!
         Like having a "show me the receipt" button right next to the review form! 🧾✨ -->
    <button class="btn btn-primary btn-sm" id="detailDocBtn"
            onclick="openPreviewToDoc()"
            style="margin-left:auto;font-size:11px;padding:5px 14px;">
      📎 View Document
    </button>
  </div>
  <div class="split-panels">
    <div class="panel-left">
      <div class="panel-header teal">
        📄 RAW RESUME TEXT
        <span style="font-weight:400;color:var(--text-mut);font-size:10px;" id="rawMeta"></span>
      </div>
      <div class="panel-body raw-text" id="rawTextPanel"></div>
    </div>
    <div class="panel-right">
      <div class="panel-header accent">
        🤖 AI EXTRACTION — Field-by-Field Review
        <span style="font-weight:400;color:var(--text-mut);font-size:10px;" id="extractionMeta"></span>
      </div>
      <div class="panel-body" id="fieldsPanel" style="padding:10px 14px;"></div>
    </div>
  </div>
</div>

<!-- ── METRICS VIEW ─────────────────────────────────────────────── -->
<div id="view-metrics" style="display:none;">
  <div class="metrics-wrap" id="metricsContent">
    <div class="loading">⏳ Loading metrics...</div>
  </div>
</div>

<!-- ── ML AUDIT VIEW ────────────────────────────────────────────── -->
<div id="view-audit" style="display:none;">
  <div class="audit-wrap" id="auditContent">
    <div class="empty">🔬 Click the ML Audit tab to generate a comprehensive ML quality report</div>
  </div>
</div>

<!-- ── SEARCH VIEW ──────────────────────────────────────────────── -->
<div id="view-search" style="display:none;">
  <div style="padding:20px 24px;max-width:960px;margin:0 auto;">

    <!-- Index status banner -->
    <div id="indexBanner" style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:14px 18px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;">
      <div>
        <span style="font-size:13px;font-weight:700;color:var(--accent);">🧠 Vector Search Engine</span>
        <span id="indexStatus" style="font-size:11px;color:var(--text-mut);margin-left:12px;">Checking...</span>
      </div>
      <div style="display:flex;gap:8px;">
        <button class="btn btn-primary btn-sm" onclick="indexCandidates(false)">📥 Index New</button>
        <button class="btn btn-outline btn-sm" onclick="indexCandidates(true)">🔄 Rebuild All</button>
      </div>
    </div>

    <!-- Job Description Input -->
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:16px;">
      <div style="font-size:13px;font-weight:700;color:var(--accent);margin-bottom:10px;">📝 Paste Job Description</div>
      <textarea id="jdInput" rows="8" placeholder="Paste the full job description here...&#10;&#10;Example: We are looking for a Senior Software Engineer with 5+ years of experience in Python, AWS, and microservices..."
        style="width:100%;box-sizing:border-box;background:var(--bg-input);border:1px solid var(--border);border-radius:8px;padding:12px;color:var(--text);font-size:13px;line-height:1.6;resize:vertical;outline:none;font-family:var(--body);"></textarea>

      <!-- Filters row -->
      <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;align-items:center;">
        <div style="flex:1;min-width:200px;">
          <label style="font-size:10px;color:var(--text-mut);display:block;margin-bottom:4px;">🔧 Must-have Skills (comma-separated)</label>
          <input id="skillFilter" type="text" placeholder="e.g. Python, AWS, Docker"
            style="width:100%;box-sizing:border-box;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-size:12px;font-family:var(--mono);outline:none;">
        </div>
        <div style="min-width:150px;">
          <label style="font-size:10px;color:var(--text-mut);display:block;margin-bottom:4px;">📍 Location</label>
          <input id="locFilter" type="text" placeholder="e.g. Singapore"
            style="width:100%;box-sizing:border-box;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-size:12px;font-family:var(--mono);outline:none;">
        </div>
        <div style="min-width:80px;">
          <label style="font-size:10px;color:var(--text-mut);display:block;margin-bottom:4px;">📊 Results</label>
          <select id="topKSelect" style="width:100%;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;padding:8px;color:var(--text);font-size:12px;outline:none;">
            <option value="5">Top 5</option>
            <option value="10" selected>Top 10</option>
            <option value="20">Top 20</option>
            <option value="50">Top 50</option>
          </select>
        </div>
        <div style="padding-top:16px;">
          <button class="btn btn-primary" onclick="searchByJD()" style="padding:10px 24px;font-size:13px;">🔍 Search</button>
        </div>
      </div>
    </div>

    <!-- Search Results -->
    <div id="searchResults"></div>
  </div>
</div>

<!-- ── PREVIEW MODAL ───────────────────────────────────────────── -->
<!-- Hidden by default. openPreview(id) makes it visible. ✨        -->
<div id="previewModal" class="modal-backdrop" style="display:none;"
     onclick="if(event.target===this)closePreview()">
  <div class="modal-box">

    <div class="modal-header">
      <div class="modal-title">
        <span class="modal-name" id="previewName">Loading...</span>
        <span class="modal-meta" id="previewMeta"></span>
      </div>
      <button class="modal-close" onclick="closePreview()" title="Close (Esc)">✕</button>
    </div>

    <!-- Tab strip: Raw Text / Structured Fields / Document -->
    <div class="modal-tabs">
      <button class="modal-tab active" id="ptab-raw"
              onclick="switchPreviewTab('raw')">📄 Raw Text</button>
      <button class="modal-tab" id="ptab-structured"
              onclick="switchPreviewTab('structured')">🤖 Extracted Fields</button>
      <button class="modal-tab" id="ptab-document"
              onclick="switchPreviewTab('document')">📎 Document</button>
    </div>

    <!-- Tab bodies -->
    <div class="modal-body" id="previewBody">
      <div class="modal-loading">✨ Fetching candidate data...</div>
    </div>

    <div class="modal-footer">
      <span class="char-count" id="previewCharCount"></span>
      <button class="btn btn-primary btn-sm"
              onclick="openFromPreview()" id="previewOpenBtn"
              style="display:none;">
        📋 Open Full Review →
      </button>
    </div>

  </div>
</div>

<!-- ── STATUS BAR ───────────────────────────────────────────────── -->
<div class="status-bar">
  <div id="statusLeft"><span class="dot dot-ok"></span> Loading...</div>
  <div style="color:var(--pink);">💅 "Measure it to manage it, darling!" — Fairy Codemother ✨</div>
</div>

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- ── JAVASCRIPT ─────────────────────────────────────────────── -->
<!-- ═══════════════════════════════════════════════════════════════ -->
<script>
// ── CONFIGURATION ─────────────────────────────────────────────────
const API = '';  // Same origin — no need for full URL!

// ── FIELD DEFINITIONS ─────────────────────────────────────────────
const FIELD_DEFS = [
  { key:'name',         dbCol:'name',          label:'Full Name',      icon:'👤', critical:true  },
  { key:'email',        dbCol:'email',         label:'Email',          icon:'📧', critical:true  },
  { key:'phone',        dbCol:'phone',         label:'Phone',          icon:'📱', critical:true  },
  { key:'date_of_birth',dbCol:'date_of_birth', label:'Date of Birth',  icon:'🎂', critical:false },
  { key:'location',     dbCol:'location',      label:'Location',       icon:'📍', critical:false },
  { key:'nationality',  dbCol:'nationality',   label:'Nationality',    icon:'🏳️', critical:false },
  { key:'summary',      dbCol:'summary',       label:'Summary',        icon:'📝', critical:false },
  { key:'skills_hard',  dbCol:'skills_raw',    label:'Skills (Raw)',   icon:'🔧', critical:true  },
  { key:'languages',    dbCol:'languages',     label:'Languages',      icon:'🌐', critical:false },
  { key:'certifications',dbCol:'certifications',label:'Certifications', icon:'🏅', critical:false },
  { key:'experience',   dbCol:'experience_raw', label:'Work Experience',icon:'💼', critical:true  },
  { key:'education',    dbCol:'education_raw',  label:'Education',      icon:'🎓', critical:true  },
];

const VERDICTS = {
  correct: { label:'Correct', color:'var(--ok)',     icon:'✅', cls:'correct' },
  wrong:   { label:'Wrong',   color:'var(--bad)',    icon:'❌', cls:'wrong'   },
  partial: { label:'Partial', color:'var(--warn)',   icon:'⚠️', cls:'partial' },
  missing: { label:'Missing', color:'var(--purple)', icon:'🔮', cls:'missing' },
  skip:    { label:'Skip',    color:'var(--text-mut)',icon:'⏭️', cls:'skip'   },
};

// ── STATE ─────────────────────────────────────────────────────────
let currentView = 'review';
let candidates = [];
let paginationData = { page: 1, total_pages: 1, total: 0 };
let summaryData = {};
let activeCandidate = null;     // Full candidate data from API
let reviews = {};               // { fieldKey: { verdict, correction } }
let activeField = null;
let saveTimer = null;

// ── API HELPER ────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.description || err.message || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── TAB SWITCHING ─────────────────────────────────────────────────
function switchTab(tab) {
  currentView = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');

  document.getElementById('view-list').style.display    = (tab === 'review' && !activeCandidate) ? '' : 'none';
  document.getElementById('view-detail').style.display   = (tab === 'review' && activeCandidate) ? '' : 'none';
  document.getElementById('view-metrics').style.display  = tab === 'metrics' ? '' : 'none';
  document.getElementById('view-search').style.display   = tab === 'search' ? '' : 'none';
  document.getElementById('view-audit').style.display    = tab === 'audit' ? '' : 'none';

  if (tab === 'metrics') loadMetrics();
  if (tab === 'search') loadSearchStats();
  if (tab === 'audit') loadAudit();
}

// ── ML QUALITY AUDIT ─────────────────────────────────────────────────
let auditCharts = {};

function destroyAuditChart(id) {
  if (auditCharts[id]) { auditCharts[id].destroy(); delete auditCharts[id]; }
}

async function loadAudit() {
  const wrap = document.getElementById('auditContent');
  wrap.innerHTML = '<div class="loading">🔬 Running ML Quality Audit...</div>';

  try {
    const audit = await apiFetch('/api/ml-audit');

    if (audit.error && !audit.ner_metrics) {
      wrap.innerHTML = '<div class="empty">' + esc(audit.error) + '</div>';
      return;
    }

    const nm = audit.ner_metrics;
    const et = audit.error_taxonomy;
    const fs = audit.field_sensitivity;
    const fz = audit.fuzzy_matches;
    const rm = audit.roadmap;
    const sm = audit.summary;

    let html = '';

    // ═══════ SUMMARY STAT CARDS ═══════
    html += '<div class="stats-row">';
    html += statCard('🎯 Weighted F1', (sm.weighted_f1 * 100).toFixed(1) + '%',
      sm.weighted_f1 >= 0.8 ? 'var(--ok)' : sm.weighted_f1 >= 0.5 ? 'var(--warn)' : 'var(--bad)');
    html += statCard('📊 Micro P', (nm.micro.precision * 100).toFixed(1) + '%', 'var(--accent)');
    html += statCard('📡 Micro R', (nm.micro.recall * 100).toFixed(1) + '%', 'var(--teal)');
    html += statCard('🏷️ Total Errors', sm.total_errors, sm.total_errors > 0 ? 'var(--bad)' : 'var(--ok)');
    html += statCard('🚨 Critical Issues', sm.critical_issues,
      sm.critical_issues > 0 ? 'var(--bad)' : 'var(--ok)');
    html += statCard('📋 Reviews Used', sm.total_reviews, 'var(--text-sec)');
    html += '</div>';

    // ═══════ 1. NER METRICS TABLE ═══════
    html += '<div class="audit-section">';
    html += '<div class="audit-section-header">🎯 Core NER Metrics — Precision / Recall / F1 per Field</div>';
    html += '<table class="audit-table"><thead><tr>';
    html += '<th>Field</th><th>Critical</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th><th>Support</th>';
    html += '</tr></thead><tbody>';

    for (const [fk, fm] of Object.entries(nm.field_metrics)) {
      const f1Color = fm.f1 >= 0.8 ? 'var(--ok)' : fm.f1 >= 0.5 ? 'var(--warn)' : 'var(--bad)';
      html += '<tr>' +
        '<td style="font-weight:700;color:var(--text)">' + esc(fm.label) + '</td>' +
        '<td>' + (fm.critical ? '<span style="color:var(--gold)">★</span>' : '—') + '</td>' +
        '<td>' + (fm.precision * 100).toFixed(1) + '%</td>' +
        '<td>' + (fm.recall * 100).toFixed(1) + '%</td>' +
        '<td style="color:' + f1Color + ';font-weight:700">' + (fm.f1 * 100).toFixed(1) + '%</td>' +
        '<td style="color:var(--ok)">' + fm.tp + '</td>' +
        '<td style="color:var(--bad)">' + fm.fp + '</td>' +
        '<td style="color:var(--purple)">' + fm.fn + '</td>' +
        '<td>' + fm.support + '</td></tr>';
    }

    // Weighted average row
    const w = nm.weighted;
    html += '<tr style="background:var(--bg-panel);font-weight:700">' +
      '<td style="color:var(--accent)">WEIGHTED AVG</td><td></td>' +
      '<td>' + (w.precision * 100).toFixed(1) + '%</td>' +
      '<td>' + (w.recall * 100).toFixed(1) + '%</td>' +
      '<td style="color:var(--accent)">' + (w.f1 * 100).toFixed(1) + '%</td>' +
      '<td colspan="4" style="color:var(--text-mut);font-weight:400">Critical fields weighted 2×</td></tr>';
    html += '</tbody></table></div>';

    // ═══════ 2. ERROR TAXONOMY (chart + table) ═══════
    html += '<div class="audit-section">';
    html += '<div class="audit-section-header">🏷️ Error Taxonomy — Classification of All Discrepancies</div>';
    html += '<div style="display:grid;grid-template-columns:280px 1fr;gap:0">';

    // Donut chart placeholder
    html += '<div class="chart-box"><canvas id="auditTaxChart" width="250" height="250"></canvas></div>';

    // Taxonomy table
    html += '<div style="padding:16px 20px">';
    html += '<table class="audit-table"><thead><tr><th>Category</th><th>Count</th><th>%</th><th>Description</th></tr></thead><tbody>';
    const taxLabels = {
      boundary: {icon: '🔲', desc: 'Clipped/extended text (substring mismatch)'},
      type_error: {icon: '🔀', desc: 'Semantic mislabeling (wrong field category)'},
      missing: {icon: '👻', desc: 'Model failed to extract existing data'},
      spurious: {icon: '🎪', desc: 'Hallucinated or noise extraction'}
    };
    for (const [cat, count] of Object.entries(et.taxonomy_counts)) {
      const info = taxLabels[cat] || {icon: '❓', desc: cat};
      const pct = et.taxonomy_percentages[cat] || 0;
      html += '<tr>' +
        '<td style="font-weight:700;color:var(--text)">' + info.icon + ' ' + cat.replace('_',' ') + '</td>' +
        '<td>' + count + '</td><td>' + pct + '%</td>' +
        '<td style="font-family:var(--body);color:var(--text-sec)">' + info.desc + '</td></tr>';
    }
    html += '</tbody></table></div></div></div>';

    // ═══════ 3. FIELD SENSITIVITY ═══════
    html += '<div class="audit-section">';
    html += '<div class="audit-section-header">🎯 Field-Level Sensitivity — Weakest Links Analysis</div>';
    html += '<table class="audit-table"><thead><tr>';
    html += '<th>Rank</th><th>Field</th><th>F1</th><th>Severity</th><th>Errors</th><th>Root Cause</th>';
    html += '</tr></thead><tbody>';

    fs.forEach(function(f, i) {
      const f1Color = f.f1 >= 0.8 ? 'var(--ok)' : f.f1 >= 0.5 ? 'var(--warn)' : 'var(--bad)';
      const rootShort = f.root_cause_detail.length > 120 ? f.root_cause_detail.substring(0,120) + '...' : f.root_cause_detail;
      html += '<tr>' +
        '<td style="color:var(--text-mut)">#' + (i + 1) + '</td>' +
        '<td style="font-weight:700">' + esc(f.label) + (f.critical ? ' <span style="color:var(--gold)">★</span>' : '') + '</td>' +
        '<td style="color:' + f1Color + ';font-weight:700">' + (f.f1 * 100).toFixed(1) + '%</td>' +
        '<td><span class="severity-dot sev-' + f.severity + '"></span>' + f.severity.replace('_',' ') + '</td>' +
        '<td>' + f.total_errors + '</td>' +
        '<td style="font-family:var(--body);font-size:10px;color:var(--text-sec)">' + esc(rootShort) + '</td></tr>';
    });
    html += '</tbody></table></div>';

    // ═══════ 4. FUZZY MATCHES ═══════
    if (fz.length > 0) {
      html += '<div class="audit-section">';
      html += '<div class="audit-section-header">📏 Textual Similarity — Levenshtein Fuzzy Matching (' + fz.length + ' comparisons)</div>';
      html += '<table class="audit-table"><thead><tr>';
      html += '<th>ID</th><th>Field</th><th>Similarity</th><th>Lev Dist</th><th>Model Output</th><th>Ground Truth</th>';
      html += '</tr></thead><tbody>';

      fz.slice(0, 30).forEach(function(fm) {
        const simColor = fm.similarity_pct >= 70 ? 'var(--ok)' : fm.similarity_pct >= 40 ? 'var(--warn)' : 'var(--bad)';
        html += '<tr>' +
          '<td>' + fm.candidate_id + '</td>' +
          '<td style="font-weight:600">' + esc(fm.field_label) + '</td>' +
          '<td style="color:' + simColor + ';font-weight:700">' + fm.similarity_pct + '%</td>' +
          '<td>' + (fm.levenshtein_distance != null ? fm.levenshtein_distance : '—') + '</td>' +
          '<td style="font-size:10px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(fm.model_value) + '">' + (esc(fm.model_value) || '<em style="color:var(--text-mut)">empty</em>') + '</td>' +
          '<td style="font-size:10px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(fm.ground_truth) + '">' + (esc(fm.ground_truth) || '<em style="color:var(--text-mut)">empty</em>') + '</td></tr>';
      });
      html += '</tbody></table></div>';
    }

    // ═══════ 5. IMPROVEMENT ROADMAP ═══════
    html += '<div class="audit-section">';
    html += '<div class="audit-section-header">🗺️ Improvement Roadmap — ' + rm.length + ' Recommendations</div>';
    rm.forEach(function(r) {
      const prioColor = r.priority === 'critical' ? 'var(--bad)' :
                        r.priority === 'medium' ? 'var(--warn)' : 'var(--ok)';
      html += '<div class="roadmap-card">' +
        '<div class="roadmap-title">' +
          '<span>' + r.priority_icon + '</span>' +
          '<span style="color:' + prioColor + ';text-transform:uppercase;font-size:10px;font-family:var(--mono)">' + r.priority + '</span>' +
          '<span style="color:var(--text)">' + esc(r.title) + '</span>' +
        '</div>' +
        '<div class="roadmap-detail">' + esc(r.detail) + '</div>' +
        '<div class="roadmap-meta">' +
          '<span>📁 ' + r.category + '</span>' +
          '<span>💪 ' + r.effort + ' effort</span>' +
          '<span>🎯 ' + esc(r.expected_impact) + '</span>' +
        '</div></div>';
    });
    html += '</div>';

    // Refresh button
    html += '<div style="text-align:center;margin-top:16px">';
    html += '<button class="btn btn-primary" onclick="loadAudit()">🔬 Refresh Audit</button></div>';

    wrap.innerHTML = html;

    // Draw taxonomy donut chart
    drawTaxonomyChart(et);

  } catch (err) {
    wrap.innerHTML = '<div class="empty">❌ Audit error: ' + esc(err.message) + '</div>';
  }
}

function drawTaxonomyChart(et) {
  destroyAuditChart('auditTaxChart');
  const ctx = document.getElementById('auditTaxChart');
  if (!ctx || typeof Chart === 'undefined') return;

  const labels = ['Boundary', 'Type Error', 'Missing (FN)', 'Spurious (FP)'];
  const data = [
    et.taxonomy_counts.boundary || 0,
    et.taxonomy_counts.type_error || 0,
    et.taxonomy_counts.missing || 0,
    et.taxonomy_counts.spurious || 0
  ];
  const colors = ['#f59e0b', '#8b5cf6', '#6366f1', '#ef4444'];

  auditCharts['auditTaxChart'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{ data: data, backgroundColor: colors, borderColor: '#151221', borderWidth: 2 }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#9b8fc4', font: { size: 10 }, padding: 8 } }
      }
    }
  });
}

// ── VECTOR SEARCH FUNCTIONS ───────────────────────────────────────
async function loadSearchStats() {
  try {
    const data = await apiFetch('/api/search/stats');
    const el = document.getElementById('indexStatus');
    if (data.vector_search_available) {
      el.innerHTML = `<span style="color:var(--ok)">✅ ${data.total_documents} candidates indexed</span>`;
    } else {
      el.innerHTML = `<span style="color:var(--warn)">⚠️ Not available — ${data.message || 'install dependencies'}</span>`;
    }
  } catch (err) {
    document.getElementById('indexStatus').innerHTML =
      `<span style="color:var(--bad)">❌ ${err.message}</span>`;
  }
}

async function indexCandidates(rebuild) {
  const el = document.getElementById('indexStatus');
  el.innerHTML = '<span style="color:var(--gold)">⏳ Indexing... (this may take a minute)</span>';
  try {
    const data = await apiFetch('/api/search/index', {
      method: 'POST',
      body: JSON.stringify({ rebuild: rebuild }),
    });
    el.innerHTML = `<span style="color:var(--ok)">✅ Done! ${data.indexed} new + ${data.already_indexed || 0} existing = ${data.total_in_collection} total (${data.total_time}s)</span>`;
  } catch (err) {
    el.innerHTML = `<span style="color:var(--bad)">❌ ${err.message}</span>`;
  }
}

async function searchByJD() {
  const jd = document.getElementById('jdInput').value.trim();
  if (!jd || jd.length < 10) {
    alert('Please enter a job description (at least 10 characters)');
    return;
  }

  const skillRaw = document.getElementById('skillFilter').value.trim();
  const skillFilter = skillRaw ? skillRaw.split(',').map(s => s.trim()).filter(s => s) : null;
  const locFilter = document.getElementById('locFilter').value.trim() || null;
  const topK = parseInt(document.getElementById('topKSelect').value) || 10;

  const wrap = document.getElementById('searchResults');
  wrap.innerHTML = '<div class="loading">🔍 Searching candidates... (embedding + similarity)</div>';

  try {
    const data = await apiFetch('/api/search', {
      method: 'POST',
      body: JSON.stringify({
        job_description: jd,
        top_k: topK,
        skill_filter: skillFilter,
        location_filter: locFilter,
      }),
    });

    if (data.error) {
      wrap.innerHTML = `<div class="empty" style="color:var(--bad)">❌ ${esc(data.error)}</div>`;
      return;
    }

    const results = data.results || [];
    const qi = data.query_info || {};

    let html = `<div style="font-size:11px;color:var(--text-mut);margin-bottom:12px;font-family:var(--mono);">
      Found ${results.length} matches in ${qi.search_time_seconds}s • ${data.total_candidates} candidates in vector DB
      ${skillFilter ? ' • Skills filter: ' + skillFilter.join(', ') : ''}
      ${locFilter ? ' • Location: ' + locFilter : ''}
    </div>`;

    if (results.length === 0) {
      html += '<div class="empty">No matching candidates found. Try broadening your search or indexing more candidates.</div>';
    } else {
      results.forEach(r => {
        const scoreColor = r.similarity_score >= 70 ? 'var(--ok)' : r.similarity_score >= 50 ? 'var(--warn)' : 'var(--bad)';
        const barWidth = Math.max(5, r.similarity_score);

        html += `
          <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:14px 18px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
              <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:20px;font-weight:800;color:${scoreColor};font-family:var(--display);min-width:35px;">#${r.rank}</span>
                <div>
                  <div style="font-size:14px;font-weight:700;color:var(--text);">${esc(r.name)}</div>
                  <div style="font-size:11px;color:var(--text-sec);">${esc(r.email)} ${r.phone ? '• ' + esc(r.phone) : ''} ${r.location ? '• 📍 ' + esc(r.location) : ''}</div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:22px;font-weight:800;color:${scoreColor};font-family:var(--display);">${r.similarity_score}%</div>
                <div style="font-size:9px;color:var(--text-mut);">match score</div>
              </div>
            </div>
            <!-- Score bar -->
            <div style="height:6px;background:var(--bg-input);border-radius:3px;overflow:hidden;margin-bottom:8px;">
              <div style="height:100%;width:${barWidth}%;background:${scoreColor};border-radius:3px;transition:width 0.3s;"></div>
            </div>
            <!-- Skills preview -->
            <div style="font-size:11px;color:var(--text-sec);line-height:1.6;">
              <span style="color:var(--accent);font-weight:600;">Skills:</span>
              ${esc(truncate(r.skills, 200))}
            </div>
            <!-- Preview button — opens the full modal! ✨ -->
            <div style="margin-top:10px;display:flex;gap:8px;align-items:center;">
              <button class="btn btn-outline btn-sm"
                      onclick="openPreview(${r.candidate_id}, event)"
                      style="font-size:11px;">
                👁️ Preview Resume
              </button>
              <span style="font-size:10px;color:var(--text-mut);font-family:var(--mono);">
                ID #${r.candidate_id}
              </span>
            </div>
          </div>`;
      });
    }

    wrap.innerHTML = html;
  } catch (err) {
    wrap.innerHTML = `<div class="empty" style="color:var(--bad)">❌ Search error: ${esc(err.message)}</div>`;
  }
}

// ── LOAD CANDIDATES ───────────────────────────────────────────────
async function loadCandidates(page = 1) {
  const search = document.getElementById('searchInput').value.trim();
  const qs = search ? `&search=${encodeURIComponent(search)}` : '';
  const grid = document.getElementById('candidatesGrid');

  // ── Show loading state while fetching ─────────────────────────
  grid.innerHTML = '<div class="modal-loading">✨ Loading candidates...</div>';

  try {
    const data = await apiFetch(`/api/candidates?page=${page}&per_page=20${qs}`);
    candidates = data.candidates || [];
    paginationData = data.pagination || { page:1, total_pages:1, total:0 };
    summaryData = data.summary || {};
    renderCandidateList();
    updateStatus();
  } catch (err) {
    // 💅 INLINE ERROR — Show the error RIGHT ON THE PAGE instead of
    // a dismissible alert() that vanishes and leaves you wondering
    // what went wrong. Like posting the rejection letter on the
    // door instead of slipping it under the mat! 🚪✨
    console.error('❌ loadCandidates failed:', err);
    grid.innerHTML = `
      <div style="text-align:center;padding:40px 20px;">
        <div style="font-size:48px;margin-bottom:12px;">🚨</div>
        <div style="font-size:15px;font-weight:700;color:var(--bad);margin-bottom:8px;">
          Failed to load candidates
        </div>
        <div style="font-size:12px;color:var(--text-sec);margin-bottom:16px;max-width:500px;margin-left:auto;margin-right:auto;">
          ${esc(err.message)}
        </div>
        <div style="background:var(--bg-input);border:1px solid var(--border);border-radius:8px;
                    padding:14px 18px;display:inline-block;text-align:left;font-size:11px;
                    color:var(--text-mut);font-family:var(--mono);line-height:1.8;">
          <b style="color:var(--text);">🔍 Troubleshooting checklist:</b><br>
          1️⃣ Does the database file exist?<br>
          &nbsp;&nbsp;&nbsp;Check: <b>resume_extractions.db</b> in the project folder<br>
          2️⃣ Have you run the extraction pipeline?<br>
          &nbsp;&nbsp;&nbsp;Run: <b>python main.py</b> to process resumes first<br>
          3️⃣ Is the DB path correct?<br>
          &nbsp;&nbsp;&nbsp;Set: <b>set RESUME_DB_PATH=C:\\\\path\\\\to\\\\resume_extractions.db</b><br>
          4️⃣ Check the terminal for server-side errors 👀
        </div>
        <div style="margin-top:16px;">
          <button class="btn btn-primary btn-sm" onclick="loadCandidates(1)">🔄 Retry</button>
        </div>
      </div>`;
    document.getElementById('pagination').innerHTML = '';
    document.getElementById('statusLeft').innerHTML =
      '<span class="dot" style="background:var(--bad)"></span> Error — check troubleshooting guide above';
  }
}
function searchCandidates() { loadCandidates(1); }

function renderCandidateList() {
  const grid = document.getElementById('candidatesGrid');
  document.getElementById('listStats').textContent =
    `${summaryData.total_candidates || 0} candidates • ${summaryData.total_reviewed || 0} reviewed`;

  if (candidates.length === 0) {
    grid.innerHTML = '<div class="empty">No candidates found. Run the extraction pipeline first!</div>';
    document.getElementById('pagination').innerHTML = '';
    return;
  }

  grid.innerHTML = candidates.map(c => `
    <div class="candidate-card" onclick="openCandidate(${c.candidate_id})">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="cid">#${c.candidate_id}</span>
        <span class="badge ${c.review_count > 0 ? 'badge-ok' : 'badge-pending'}">
          ${c.review_count > 0 ? '✅ ' + c.review_count + ' reviewed' : '⏳ Pending'}
        </span>
      </div>
      <div class="name">${esc(c.name || 'Unknown')}</div>
      <div class="email">${esc(c.email || '')}</div>
      <div style="margin: 6px 0 2px;">
        <button class="btn btn-outline btn-sm"
                onclick="openPreview(${c.candidate_id}, event)"
                style="font-size:10px;padding:3px 10px;">
          👁️ Preview
        </button>
      </div>
      <div class="meta">
        <span>${esc(c.extraction_status || '')}</span>
        <span>${c.ai_assisted ? '🤖 AI' : '📏 Regex'}</span>
        ${c.wrong_count > 0 ? '<span style="color:var(--bad)">❌' + c.wrong_count + '</span>' : ''}
        ${c.correct_count > 0 ? '<span style="color:var(--ok)">✅' + c.correct_count + '</span>' : ''}
      </div>
    </div>
  `).join('');

  // Pagination
  const p = paginationData;
  let phtml = '';
  if (p.page > 1) phtml += `<button class="btn btn-outline btn-sm" onclick="loadCandidates(${p.page-1})">← Prev</button>`;
  phtml += `<span style="padding:6px 12px;font-size:11px;color:var(--text-mut)">Page ${p.page} of ${p.total_pages} (${p.total} total)</span>`;
  if (p.page < p.total_pages) phtml += `<button class="btn btn-outline btn-sm" onclick="loadCandidates(${p.page+1})">Next →</button>`;
  document.getElementById('pagination').innerHTML = phtml;
}

// ── OPEN CANDIDATE ────────────────────────────────────────────────
async function openCandidate(candidateId) {
  try {
    activeCandidate = await apiFetch(`/api/candidates/${candidateId}`);
    reviews = activeCandidate.reviews || {};
    activeField = null;

    document.getElementById('view-list').style.display = 'none';
    document.getElementById('view-detail').style.display = '';

    renderDetail();
    updateStatus();
  } catch (err) {
    alert('Error loading candidate: ' + err.message);
  }
}

function backToList() {
  activeCandidate = null;
  reviews = {};
  activeField = null;
  document.getElementById('view-detail').style.display = 'none';
  document.getElementById('view-list').style.display = '';
  loadCandidates(paginationData.page);  // Refresh list to show updated review counts
}

// ── RENDER DETAIL ─────────────────────────────────────────────────
function renderDetail() {
  if (!activeCandidate) return;
  const s = activeCandidate.structured || {};
  const m = activeCandidate.raw_meta || {};

  document.getElementById('detailName').textContent =
    `#${activeCandidate.candidate_id} ${s.name || 'Unknown'}`;
  updateProgressCounter();

  // Raw text
  document.getElementById('rawMeta').textContent =
    `${(m.text_length || 0).toLocaleString()} chars • ${m.resume_language || ''}`;
  document.getElementById('rawTextPanel').textContent = activeCandidate.raw_text || '(No raw text)';

  // Extraction meta
  document.getElementById('extractionMeta').textContent =
    `${s.extraction_method || ''} ${s.ai_assisted ? '• AI-assisted' : ''}`;

  // Fields
  renderFields();
}

function renderFields() {
  const s = activeCandidate.structured || {};
  const panel = document.getElementById('fieldsPanel');
  let html = '';

  FIELD_DEFS.forEach(f => {
    const value = s[f.dbCol] || '';
    const review = reviews[f.key] || {};
    const isActive = activeField === f.key;
    const isEmpty = !value || value.trim() === '';
    const verdict = review.verdict;
    const vInfo = verdict ? VERDICTS[verdict] : null;

    const isLong = ['summary','skills_raw','experience_raw','education_raw'].includes(f.dbCol);

    html += `
      <div class="field-card ${isActive ? 'active' : ''}"
           onclick="selectField('${f.key}')"
           style="${verdict ? 'border-color:' + vInfo.color.replace('var(','').replace(')','') + '33;' : ''}">
        <div class="field-header">
          <div class="field-name">
            <span style="font-size:14px">${f.icon}</span>
            <span style="color:${f.critical ? 'var(--gold)' : 'var(--text)'}">${f.label}</span>
            ${f.critical ? '<span class="critical">CRITICAL</span>' : ''}
          </div>
          ${verdict ? `<span class="verdict-badge" style="color:${vInfo.color};background:${vInfo.color.replace(')',',0.1)').replace('var(','rgba(')}">${vInfo.icon} ${vInfo.label}</span>` : ''}
        </div>
        <div class="field-value ${isEmpty ? 'empty' : ''} ${isLong ? 'long' : ''}">
          ${isEmpty ? '⊘ Not extracted' : esc(truncate(value, isLong ? 250 : 100))}
        </div>
        ${isActive ? renderVerdictButtons(f.key, review) : ''}
      </div>`;
  });

  panel.innerHTML = html + '<div style="height:40px"></div>';
}

function renderVerdictButtons(fieldKey, review) {
  const verdict = review.verdict || '';
  let html = '<div class="verdict-row">';
  for (const [v, info] of Object.entries(VERDICTS)) {
    html += `<button class="verdict-btn ${v} ${verdict === v ? 'active' : ''}"
              onclick="event.stopPropagation(); setVerdict('${fieldKey}','${v}')">
              ${info.icon} ${info.label}</button>`;
  }
  html += '</div>';

  // Correction input for wrong/partial
  if (verdict === 'wrong' || verdict === 'partial') {
    html += `<input class="correction-input" type="text"
              value="${esc(review.correction || '')}"
              placeholder="Type the correct value..."
              onclick="event.stopPropagation()"
              oninput="setCorrection('${fieldKey}', this.value)">`;
  }
  html += '<div class="shortcuts">⌨️ <b>C</b>=Correct <b>W</b>=Wrong <b>P</b>=Partial <b>M</b>=Missing <b>S</b>=Skip</div>';
  return html;
}

// ── FIELD SELECTION ───────────────────────────────────────────────
function selectField(fieldKey) {
  activeField = fieldKey;
  renderFields();
  highlightInRawText(fieldKey);
}

// ── SET VERDICT ───────────────────────────────────────────────────
function setVerdict(fieldKey, verdict) {
  reviews[fieldKey] = {
    ...(reviews[fieldKey] || {}),
    verdict: verdict,
    correction: (verdict === 'wrong' || verdict === 'partial') ? (reviews[fieldKey]?.correction || '') : '',
    reviewed_at: new Date().toISOString(),
  };

  // Auto-advance to next field
  const idx = FIELD_DEFS.findIndex(f => f.key === fieldKey);
  if (idx < FIELD_DEFS.length - 1) activeField = FIELD_DEFS[idx + 1].key;

  renderFields();
  updateProgressCounter();
  scheduleSave();
}

function setCorrection(fieldKey, text) {
  if (!reviews[fieldKey]) reviews[fieldKey] = { verdict: 'wrong' };
  reviews[fieldKey].correction = text;
  scheduleSave();
}

// ── AUTO-SAVE ─────────────────────────────────────────────────────
function scheduleSave() {
  clearTimeout(saveTimer);
  document.getElementById('savingIndicator').style.display = '';
  saveTimer = setTimeout(saveReviews, 1500);
}

async function saveReviews() {
  if (!activeCandidate) return;
  try {
    await apiFetch(`/api/reviews/${activeCandidate.candidate_id}`, {
      method: 'POST',
      body: JSON.stringify({ reviews: reviews, reviewer: 'default' }),
    });
  } catch (err) {
    console.error('Save error:', err);
  } finally {
    document.getElementById('savingIndicator').style.display = 'none';
  }
}

// ── HIGHLIGHT IN RAW TEXT ─────────────────────────────────────────
function highlightInRawText(fieldKey) {
  const panel = document.getElementById('rawTextPanel');
  const s = activeCandidate.structured || {};
  const def = FIELD_DEFS.find(f => f.key === fieldKey);
  const value = s[def?.dbCol] || '';

  // Reset to plain text first
  panel.textContent = activeCandidate.raw_text || '';

  if (!value || value.length < 3) return;

  // Get search terms (split skills/experience by delimiters)
  let terms = [];
  if (value.includes(',') || value.includes('||')) {
    terms = value.split(/[,|]+/).slice(0, 5).map(t => t.trim()).filter(t => t.length > 2);
  } else {
    terms = [value.slice(0, 60).trim()];
  }

  // Build highlighted HTML
  let html = esc(activeCandidate.raw_text || '');
  terms.forEach(term => {
    const escaped = term.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
    const re = new RegExp('(' + escaped + ')', 'gi');
    html = html.replace(re, '<mark>$1</mark>');
  });
  panel.innerHTML = html;

  // Scroll to first match
  const firstMark = panel.querySelector('mark');
  if (firstMark) firstMark.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── KEYBOARD SHORTCUTS ────────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  if (!activeField || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const map = { C:'correct', W:'wrong', P:'partial', M:'missing', S:'skip' };
  const v = map[e.key.toUpperCase()];
  if (v) { e.preventDefault(); setVerdict(activeField, v); }
});

// ── METRICS ───────────────────────────────────────────────────────
async function loadMetrics() {
  const wrap = document.getElementById('metricsContent');
  try {
    const m = await apiFetch('/api/metrics');
    const o = m.overall;

    let html = '<div class="stats-row">';
    html += statCard('🎯 Accuracy', o.accuracy + '%', o.accuracy >= 80 ? 'var(--ok)' : o.accuracy >= 50 ? 'var(--warn)' : 'var(--bad)', o.reviewed + ' fields');
    html += statCard('⚡ Critical', m.critical_accuracy + '%', m.critical_accuracy >= 80 ? 'var(--ok)' : 'var(--warn)');
    html += statCard('✅ Correct', o.correct, 'var(--ok)');
    html += statCard('❌ Wrong', o.wrong, 'var(--bad)');
    html += statCard('⚠️ Partial', o.partial, 'var(--warn)');
    html += statCard('📈 Progress', m.progress.percentage + '%', 'var(--teal)', m.progress.reviewed_candidates + '/' + m.progress.total_candidates);
    html += '</div>';

    // Field table
    const fm = m.field_metrics;
    if (Object.keys(fm).length > 0) {
      html += '<div class="field-table"><div class="table-header">📊 Field-by-Field Accuracy</div>';
      for (const [k, f] of Object.entries(fm)) {
        const pctOk = f.total > 0 ? (f.correct/f.total*100) : 0;
        const pctWarn = f.total > 0 ? (f.partial/f.total*100) : 0;
        const pctBad = f.total > 0 ? (f.wrong/f.total*100) : 0;
        const pctMiss = f.total > 0 ? (f.missing/f.total*100) : 0;
        const accColor = f.accuracy >= 80 ? 'var(--ok)' : f.accuracy >= 50 ? 'var(--warn)' : f.total > 0 ? 'var(--bad)' : 'var(--text-mut)';

        html += `<div class="field-row">
          <span class="fname" style="color:${f.critical ? 'var(--gold)' : 'var(--text)'}">${f.label}</span>
          <div class="bar-wrap">
            ${f.total > 0 ? `
              <div class="bar-ok"   style="width:${pctOk}%"></div>
              <div class="bar-warn" style="width:${pctWarn}%"></div>
              <div class="bar-bad"  style="width:${pctBad}%"></div>
              <div class="bar-miss" style="width:${pctMiss}%"></div>
            ` : '<div style="width:100%;display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--text-mut)">—</div>'}
          </div>
          <span class="pct" style="color:${accColor}">${f.total > 0 ? f.accuracy + '%' : '—'}</span>
          <div class="counts">
            <span style="color:var(--ok)">✅${f.correct}</span>
            <span style="color:var(--warn)">⚠️${f.partial}</span>
            <span style="color:var(--bad)">❌${f.wrong}</span>
            <span style="color:var(--purple)">🔮${f.missing}</span>
          </div>
        </div>`;
      }
      html += '</div>';
    }

    // Problem fields
    if (m.problem_fields && m.problem_fields.length > 0) {
      html += '<div class="field-table" style="border-color:rgba(239,68,68,0.2)"><div class="table-header" style="color:var(--bad)">🚨 Top Problem Fields</div>';
      m.problem_fields.slice(0,5).forEach(pf => {
        html += `<div style="display:flex;gap:10px;padding:8px 18px;font-size:12px;align-items:center">
          <span style="font-weight:700;color:var(--bad);width:40px">${pf.error_rate}%</span>
          <span>${pf.label}</span>
          <span style="font-size:10px;color:var(--text-mut)">(${pf.wrong} wrong + ${pf.missing} missing)</span>
        </div>`;
      });
      html += '</div>';
    }

    html += '<div style="text-align:center;margin-top:16px"><button class="btn btn-primary" onclick="loadMetrics()">🔄 Refresh Metrics</button></div>';
    wrap.innerHTML = html;
  } catch (err) {
    wrap.innerHTML = `<div class="empty">Error loading metrics: ${esc(err.message)}</div>`;
  }
}

function statCard(label, value, color, sub) {
  return `<div class="stat-card">
    <div class="label">${label}</div>
    <div class="value" style="color:${color}">${value}</div>
    ${sub ? '<div class="sub">' + sub + '</div>' : ''}
  </div>`;
}

// ── RESET ─────────────────────────────────────────────────────────
async function resetAll() {
  if (!confirm('Delete ALL review data? This cannot be undone!')) return;
  try {
    await apiFetch('/api/reviews', { method: 'DELETE', body: JSON.stringify({ confirm: true }) });
    reviews = {};
    if (activeCandidate) { renderFields(); updateProgressCounter(); }
    loadCandidates(1);
    alert('All reviews deleted!');
  } catch (err) { alert('Error: ' + err.message); }
}

// ── HELPERS ───────────────────────────────────────────────────────
function esc(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function truncate(s, n) {
  return (s && s.length > n) ? s.slice(0, n) + '…' : (s || '');
}

function updateProgressCounter() {
  const count = Object.keys(reviews).filter(k => reviews[k]?.verdict).length;
  const el = document.getElementById('detailProgress');
  if (el) el.textContent = `${count}/${FIELD_DEFS.length} fields reviewed`;
}

function updateStatus() {
  const el = document.getElementById('statusLeft');
  if (activeCandidate) {
    const count = Object.keys(reviews).filter(k => reviews[k]?.verdict).length;
    el.innerHTML = `<span class="dot dot-ok"></span> Reviewing #${activeCandidate.candidate_id} • ${count}/${FIELD_DEFS.length} fields`;
  } else {
    el.innerHTML = `<span class="dot dot-ok"></span> ${summaryData.total_candidates || '?'} candidates • ${summaryData.total_reviewed || 0} reviewed`;
  }
}

// ── PREVIEW MODAL ─────────────────────────────────────────────────
// Stores the currently previewed candidate data so the
// "Open Full Review" button can navigate to it directly. 💅
let previewData = null;
let previewTab = 'raw';

/**
 * Open the preview modal for a given candidate ID.
 * Fetches full candidate data from /api/candidates/<id>
 * and renders both raw text + structured fields.
 *
 * Think of it as peeking behind the curtain before the big show! 🎭
 *
 * @param {number} candidateId  - The candidate's DB ID
 * @param {Event}  [evt]        - Optional click event (to stop propagation
 *                                so clicking the card doesn't also open review)
 */
async function openPreview(candidateId, evt) {
  // Stop the click from bubbling up to the parent card's onclick,
  // which would navigate away before the modal even opens! 🚫
  if (evt) evt.stopPropagation();

  // Reset & show modal with loading state
  previewData = null;
  previewTab = 'raw';
  document.getElementById('previewName').textContent = `#${candidateId} — Loading...`;
  document.getElementById('previewMeta').textContent = '';
  document.getElementById('previewCharCount').textContent = '';
  document.getElementById('previewOpenBtn').style.display = 'none';
  document.getElementById('previewBody').innerHTML =
    '<div class="modal-loading">✨ Fetching candidate data...</div>';

  // Activate raw tab visually
  document.getElementById('ptab-raw').classList.add('active');
  document.getElementById('ptab-structured').classList.remove('active');

  // Show the modal backdrop
  document.getElementById('previewModal').style.display = 'flex';
  document.body.style.overflow = 'hidden'; // Prevent background scroll

  try {
    // Fetch full candidate record (raw text + structured + reviews)
    const data = await apiFetch(`/api/candidates/${candidateId}`);
    previewData = data;
    const s = data.structured || {};
    const m = data.raw_meta || {};

    // ── Update header ────────────────────────────────────────────
    document.getElementById('previewName').textContent =
      `#${candidateId} ${s.name || 'Unknown Candidate'}`;
    document.getElementById('previewMeta').textContent = [
      s.email,
      s.phone,
      m.resume_language ? `🌐 ${m.resume_language}` : null,
      m.pdf_page_count   ? `📄 ${m.pdf_page_count}p` : null,
      s.extraction_method,
      s.ai_assisted ? '🤖 AI-assisted' : '📏 Regex',
    ].filter(Boolean).join('  •  ');

    // ── Show char count ──────────────────────────────────────────
    const chars = (data.raw_text || '').length;
    document.getElementById('previewCharCount').textContent =
      `${chars.toLocaleString()} chars`;

    // ── Show "Open Full Review" button only on Review tab ────────
    if (currentView === 'review') {
      document.getElementById('previewOpenBtn').style.display = '';
    }

    // ── Render the active tab ────────────────────────────────────
    renderPreviewTab('raw');

  } catch (err) {
    document.getElementById('previewBody').innerHTML =
      `<div class="modal-loading" style="color:var(--bad)">
         ❌ Failed to load candidate: ${esc(err.message)}
       </div>`;
  }
}

/** Close the preview modal and restore body scroll. */
function closePreview() {
  document.getElementById('previewModal').style.display = 'none';
  document.body.style.overflow = '';
  previewData = null;

  // ── Reset expanded state so next open starts normal-sized ────
  // Without this, opening a new preview after viewing a document
  // would start expanded even on the Raw Text tab! 🐛
  const modalBox = document.querySelector('#previewModal .modal-box');
  if (modalBox) modalBox.classList.remove('modal-expanded');
}

/** Switch between Raw / Structured / Document tabs inside the modal.
 *  💅 Also handles the DIVA EXPANSION — when Document tab is active,
 *  the modal goes wide to give the PDF/DOCX room to breathe!
 *  When switching back to Raw/Structured, she returns to normal size.
 *  Like a stage that expands for the grand finale! 🎭✨              */
function switchPreviewTab(tab) {
  previewTab = tab;

  // ── Toggle active state on ALL three tabs ───────────────────
  // 🐛 FIX: The old code forgot ptab-document, so clicking
  // Document left the tab visually unselected. Not anymore! 💅
  document.getElementById('ptab-raw').classList.toggle('active', tab === 'raw');
  document.getElementById('ptab-structured').classList.toggle('active', tab === 'structured');
  document.getElementById('ptab-document').classList.toggle('active', tab === 'document');

  // ── Expand/collapse modal for Document tab ──────────────────
  // Documents need more screen real estate — PDFs and DOCX
  // look ridiculous crammed into 780px. Time to go WIDE! 💃
  const modalBox = document.querySelector('#previewModal .modal-box');
  if (modalBox) {
    modalBox.classList.toggle('modal-expanded', tab === 'document');
  }

  renderPreviewTab(tab);
}

/**
 * Render the modal body for the given tab.
 * raw        → plain pre-formatted resume text
 * structured → field-by-field table of extracted values
 * document   → actual file rendered inline (PDF/image) or download card (DOCX)
 */
async function renderPreviewTab(tab) {
  if (!previewData) return;
  const body = document.getElementById('previewBody');
  const s = previewData.structured || {};

  if (tab === 'raw') {
    // ── Raw resume text ────────────────────────────────────────
    const raw = previewData.raw_text || '';
    body.innerHTML = raw
      ? `<pre class="modal-raw">${esc(raw)}</pre>`
      : `<div class="modal-loading" style="color:var(--text-mut)">
           📭 No raw text available for this candidate.
         </div>`;

  } else if (tab === 'structured') {
    // ── Structured field table ─────────────────────────────────
    const fields = [
      { label: '👤 Name',           value: s.name },
      { label: '📧 Email',          value: s.email },
      { label: '📱 Phone',          value: s.phone },
      { label: '🎂 Date of Birth',  value: s.date_of_birth },
      { label: '📍 Location',       value: s.location },
      { label: '🏳️ Nationality',   value: s.nationality },
      { label: '📝 Summary',        value: s.summary },
      { label: '🔧 Skills',         value: s.skills_raw,      mono: true },
      { label: '🌐 Languages',      value: s.languages },
      { label: '🏅 Certifications', value: s.certifications },
      { label: '💼 Experience',     value: s.experience_raw,  mono: true },
      { label: '🎓 Education',      value: s.education_raw,   mono: true },
    ];
    const rows = fields.map(f => {
      const isEmpty = !f.value || String(f.value).trim() === '';
      const cls = ['modal-field-value', isEmpty ? 'empty' : '', f.mono ? 'mono' : '']
        .filter(Boolean).join(' ');
      return `<div class="modal-field-row">
        <span class="modal-field-label">${f.label}</span>
        <span class="${cls}">${isEmpty ? '⊘ Not extracted' : esc(truncate(String(f.value), 400))}</span>
      </div>`;
    }).join('');
    body.innerHTML = `<div class="modal-structured">${rows}</div>`;

  } else if (tab === 'document') {
    // ── Document viewer ────────────────────────────────────────
    // First show a loading state, then fetch file info from API.
    // We check /file/info first (lightweight) before loading the
    // actual binary. Like calling ahead before driving to the archive! 📞
    body.innerHTML = '<div class="modal-loading">📎 Locating document file...</div>';

    const cid = previewData.candidate_id;

    try {
      const info = await apiFetch(`/api/candidates/${cid}/file/info`);

      if (!info.found) {
        // ── File not found on disk ─────────────────────────────
        body.innerHTML = `
          <div class="doc-not-found">
            <div class="nf-icon">🗄️</div>
            <div class="nf-title">Original file not found on disk</div>
            <p style="font-size:12px;color:var(--text-mut);margin:8px 0;">
              The raw text was extracted, but the source file isn't in the configured folder.
            </p>
            <div class="nf-tip">
              📂 Files directory: <b>${esc(info.files_dir)}</b><br>
              💡 Set env var: <b>set RESUME_FILES_DIR=C:\\path\\to\\resumes</b><br>
              🔄 Then restart the server, darling!
            </div>
          </div>`;
        return;
      }

      // ── File found! Render based on type ───────────────────────
      const fileUrl = `/api/candidates/${cid}/file`;
      const downloadUrl = `/api/candidates/${cid}/file?download=1`;

      // Update footer with file info
      document.getElementById('previewCharCount').textContent =
        `${esc(info.filename)} • ${info.size_mb} MB`;

      if (info.renderable) {
        // ── PDF: render inline in iframe ─────────────────────────
        // The iframe lets the browser's native PDF renderer do the
        // heavy lifting — no libraries needed! Like hiring a
        // professional projectionist instead of DIY! 🎬✨
        body.innerHTML = `
          <div class="doc-viewer-wrap">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:8px;">
              <span style="font-size:11px;color:var(--text-mut);font-family:var(--mono);">
                📄 ${esc(info.filename)} (${info.size_mb} MB)
              </span>
              <a href="${downloadUrl}" target="_blank"
                 style="font-size:11px;color:var(--accent);">
                ⬇️ Download
              </a>
            </div>
            <iframe
              class="doc-iframe"
              src="${fileUrl}"
              title="Resume: ${esc(info.filename)}"
              loading="lazy">
              <p style="padding:20px;color:#666;">
                Your browser can't display this PDF inline.
                <a href="${fileUrl}">Click here to open it.</a>
              </p>
            </iframe>
          </div>`;

      } else if (info.is_image) {
        // ── Image: render as <img> ────────────────────────────────
        body.innerHTML = `
          <div style="text-align:center;padding:10px 0;">
            <div style="font-size:11px;color:var(--text-mut);
                        margin-bottom:12px;font-family:var(--mono);">
              🖼️ ${esc(info.filename)} (${info.size_mb} MB)
            </div>
            <img class="doc-image"
                 src="${fileUrl}"
                 alt="Resume: ${esc(info.filename)}"
                 onerror="this.parentNode.innerHTML='<p style=color:var(--bad)>❌ Failed to load image</p>'">
            <div style="margin-top:12px;">
              <a href="${downloadUrl}"
                 class="btn btn-outline btn-sm"
                 style="font-size:11px;">
                ⬇️ Download Original
              </a>
            </div>
          </div>`;

      } else if (info.ext === '.docx' && typeof mammoth !== 'undefined') {
        // ── DOCX: Convert to HTML inline using mammoth.js! ────────
        // 💅 THE GLOW UP! Instead of "download to view", we convert
        // the DOCX to beautiful HTML right in the browser! mammoth.js
        // reads the DOCX binary (which is actually a ZIP of XML files,
        // like a resume wearing a trenchcoat 🕵️), extracts the content,
        // and renders it as clean HTML. No server round-trip needed!
        body.innerHTML = `
          <div class="docx-preview-wrap">
            <div class="docx-preview-toolbar">
              <span>📘 ${esc(info.filename)} (${info.size_mb} MB)</span>
              <a href="${downloadUrl}" target="_blank">⬇️ Download Original</a>
            </div>
            <div class="docx-html-container" id="docxPreviewContent">
              <div class="modal-loading">📘 Rendering Word document...</div>
            </div>
          </div>`;

        // ── Fetch the DOCX binary and convert with mammoth ────────
        // mammoth.convertToHtml() does all the magic: it reads the
        // ArrayBuffer, unpacks the DOCX ZIP, parses the XML, and
        // spits out clean HTML. Like a backstage crew transforming
        // a raw script into a polished performance! 🎭✨
        try {
          const fileResp = await fetch(fileUrl);
          if (!fileResp.ok) throw new Error(`HTTP ${fileResp.status}`);

          const arrayBuffer = await fileResp.arrayBuffer();
          const result = await mammoth.convertToHtml(
            { arrayBuffer: arrayBuffer },
            {
              // ── Style mapping: upgrade mammoth's default rendering ─
              // By default mammoth ignores some Word styles. These
              // mappings ensure headings, bold, italic come through! 📝
              styleMap: [
                "p[style-name='Heading 1'] => h1:fresh",
                "p[style-name='Heading 2'] => h2:fresh",
                "p[style-name='Heading 3'] => h3:fresh",
                "p[style-name='Title'] => h1.doc-title:fresh",
              ]
            }
          );

          const container = document.getElementById('docxPreviewContent');
          if (container) {
            if (result.value && result.value.trim().length > 0) {
              container.innerHTML = result.value;
            } else {
              // ── Empty conversion: DOCX might be image-only or weird format ─
              // Some resumes are basically a scanned image pasted into Word.
              // mammoth can't extract images as content, so we fall back
              // to the already-extracted raw text! 📄
              const rawText = previewData ? (previewData.raw_text || '') : '';
              if (rawText.trim()) {
                container.innerHTML =
                  '<div style="color:var(--warn);font-size:11px;margin-bottom:12px;">' +
                  '⚠️ Document has no extractable text content (may be image-based). ' +
                  'Showing extracted raw text instead:</div>' +
                  '<pre style="white-space:pre-wrap;word-wrap:break-word;' +
                  'font-family:inherit;margin:0;color:#1a1a2e;">' +
                  esc(rawText) + '</pre>';
              } else {
                container.innerHTML =
                  '<div style="text-align:center;padding:40px;color:#666;">' +
                  '📭 No text content could be extracted from this document.</div>';
              }
            }

            // ── Log conversion warnings for debugging ─────────────
            if (result.messages && result.messages.length > 0) {
              console.log('📘 mammoth.js conversion messages:', result.messages);
            }
          }
        } catch (docErr) {
          // ── Conversion failed: graceful fallback to raw text ─────
          // If mammoth chokes (corrupted DOCX, network error, etc.),
          // we still show SOMETHING useful — the raw extracted text! 💪
          console.error('📘 DOCX preview error:', docErr);
          const container = document.getElementById('docxPreviewContent');
          if (container) {
            const rawText = previewData ? (previewData.raw_text || '') : '';
            container.innerHTML =
              '<div style="color:var(--warn);font-size:11px;margin-bottom:12px;">' +
              '⚠️ Could not render DOCX preview (' + esc(docErr.message) + '). ' +
              'Showing extracted raw text:</div>' +
              '<pre style="white-space:pre-wrap;word-wrap:break-word;' +
              'font-family:inherit;margin:0;color:#1a1a2e;">' +
              esc(rawText) + '</pre>';
          }
        }

      } else {
        // ── DOC/ODT/RTF: Render raw text with download option ─────
        // 💅 Legacy .doc (pre-2007 binary) and other formats can't be
        // converted in-browser like .docx (which is XML-based).
        // Instead we show the ALREADY-EXTRACTED raw text beautifully
        // formatted — because we DID extract it during the pipeline!
        // It's like playing the audio recording when the singer can't
        // make it to the live show! 🎤✨
        const iconMap = {
          '.docx': '📘', '.doc': '📘',
          '.odt':  '📗', '.rtf': '📙',
        };
        const icon = iconMap[info.ext] || '📄';
        const rawText = previewData ? (previewData.raw_text || '') : '';

        body.innerHTML = `
          <div class="docx-preview-wrap">
            <div class="docx-preview-toolbar">
              <span>${icon} ${esc(info.filename)} (${info.size_mb} MB) •
                ${info.ext.toUpperCase()} format — showing extracted text</span>
              <a href="${downloadUrl}" target="_blank">⬇️ Download Original</a>
            </div>
            ${rawText.trim() ? `
              <div class="doc-raw-fallback">${esc(rawText)}</div>
            ` : `
              <div class="doc-download-card">
                <div class="doc-icon">${icon}</div>
                <div class="doc-name">${esc(info.filename)}</div>
                <div class="doc-size">${info.size_mb} MB • ${info.ext.toUpperCase()} document</div>
                <a href="${downloadUrl}"
                   class="btn btn-primary"
                   style="font-size:13px;padding:10px 28px;text-decoration:none;">
                  ⬇️ Download to View
                </a>
                <div style="font-size:11px;color:var(--text-mut);margin-top:14px;">
                  This format can't be previewed inline — open in Word or LibreOffice 💅
                </div>
              </div>
            `}
          </div>`;
      }

    } catch (err) {
      body.innerHTML = `
        <div class="doc-not-found">
          <div class="nf-icon">❌</div>
          <div class="nf-title">Error loading file info</div>
          <p style="font-size:12px;color:var(--bad);">${esc(err.message)}</p>
        </div>`;
    }
  }
}

/**
 * Open the preview modal directly on the Document tab!
 * 📎 Called from the detail view's "View Document" button.
 *
 * 💅 THE SHORTCUT! Instead of opening the modal → clicking Raw Text →
 * then clicking Document, this goes straight to the good stuff!
 * Like a VIP entrance that skips the queue! 🎟️✨
 *
 * Uses activeCandidate (already loaded by openCandidate) so we
 * don't need to re-fetch the candidate data — it's already backstage! 🎭
 */
async function openPreviewToDoc() {
  if (!activeCandidate) return;
  const candidateId = activeCandidate.candidate_id;

  // Reset & show modal with loading state
  previewData = null;
  previewTab = 'document';
  document.getElementById('previewName').textContent = `#${candidateId} — Loading...`;
  document.getElementById('previewMeta').textContent = '';
  document.getElementById('previewCharCount').textContent = '';
  document.getElementById('previewOpenBtn').style.display = 'none';
  document.getElementById('previewBody').innerHTML =
    '<div class="modal-loading">📎 Locating document file...</div>';

  // ── Activate Document tab visually (not Raw!) ─────────────────
  document.getElementById('ptab-raw').classList.remove('active');
  document.getElementById('ptab-structured').classList.remove('active');
  document.getElementById('ptab-document').classList.add('active');

  // ── Expand modal immediately for the document view ────────────
  const modalBox = document.querySelector('#previewModal .modal-box');
  if (modalBox) modalBox.classList.add('modal-expanded');

  // Show the modal
  document.getElementById('previewModal').style.display = 'flex';
  document.body.style.overflow = 'hidden';

  try {
    // ── Reuse activeCandidate data if available ─────────────────
    // No need to re-fetch — the detail view already loaded it!
    // Like using the photos you already have instead of scheduling
    // another photoshoot! 📸✨
    previewData = activeCandidate;
    const s = (activeCandidate.structured || {});
    const m = (activeCandidate.raw_meta || {});

    // Update header
    document.getElementById('previewName').textContent =
      `#${candidateId} ${s.name || 'Unknown Candidate'}`;
    document.getElementById('previewMeta').textContent = [
      s.email,
      s.phone,
      m.resume_language ? `🌐 ${m.resume_language}` : null,
      s.extraction_method,
      s.ai_assisted ? '🤖 AI-assisted' : '📏 Regex',
    ].filter(Boolean).join('  •  ');

    // ── Render Document tab directly! ─────────────────────────────
    renderPreviewTab('document');

  } catch (err) {
    document.getElementById('previewBody').innerHTML =
      `<div class="modal-loading" style="color:var(--bad)">
         ❌ Failed to load: ${esc(err.message)}
       </div>`;
  }
}

/**
 * Navigate from the modal directly into the full review view.
 * Only shown when user is on the Review tab — clicking this
 * closes the modal and opens the candidate for field-by-field review.
 */
function openFromPreview() {
  if (!previewData) return;
  const candidateId = previewData.candidate_id;
  closePreview();
  openCandidate(candidateId);
}

// Close modal on Escape key — keyboard UX, darling! ⌨️✨
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('previewModal').style.display !== 'none') {
    closePreview();
  }
});

// ── INIT ──────────────────────────────────────────────────────────
loadCandidates(1);
</script>
</body>
</html>
"""


@app.route("/dashboard")
def dashboard():
    """
    🎨 Serve the Accuracy Validator dashboard!

    Open http://localhost:PORT/dashboard in your browser.
    Everything runs from this single Flask server — no React, no npm,
    no Node.js needed. Just like annotation_tool.py! 💅
    """
    return render_template_string(DASHBOARD_HTML)


# =============================================================================
# 🏠 ROOT / HEALTH CHECK
# =============================================================================

@app.route("/")
def index():
    """
    🏠 Root endpoint — serves as a health check and API docs summary.
    """
    return jsonify({
        "service": "AI Extraction Accuracy Validator API",
        "version": "1.0.0",
        "author": "Fairy Codemother 🧚‍♀️✨",
        "status": "running",
        "database": DATABASE_PATH,
        "database_exists": os.path.exists(DATABASE_PATH),
        "endpoints": {
            "GET  /api/candidates":            "List candidates (paginated)",
            "GET  /api/candidates/<id>":        "Full candidate data",
            "GET  /api/candidates/<id>/raw":    "Raw resume text only",
            "GET  /api/candidates/<id>/log":    "Extraction log history",
            "POST /api/reviews/<id>":           "Save review verdicts",
            "GET  /api/reviews":                "All reviews (filterable)",
            "GET  /api/reviews/<id>":           "Reviews for one candidate",
            "DELETE /api/reviews":              "Reset all reviews",
            "GET  /api/metrics":                "Accuracy metrics",
            "GET  /api/stats":                  "Database overview stats",
        }
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """
    💓 Health check endpoint — THE FULL BODY SCAN! 🏥

    Hit http://localhost:5070/api/health in your browser to see
    exactly what's working and what's broken. This is your
    single source of truth when the dashboard is empty! 🔍✨
    """
    db_abs = os.path.abspath(DATABASE_PATH)
    db_exists = os.path.exists(db_abs)

    result = {
        "status": "unknown",
        "database_path_configured": DATABASE_PATH,
        "database_path_resolved": db_abs,
        "database_exists": db_exists,
        "files_dir": os.path.abspath(FILES_DIR),
        "files_dir_exists": os.path.isdir(os.path.abspath(FILES_DIR)),
        "cwd": os.getcwd(),
        "tables": {},
        "diagnosis": [],
    }

    if not db_exists:
        result["status"] = "BROKEN"
        result["diagnosis"].append(
            f"DATABASE NOT FOUND at {db_abs}! "
            f"Either copy your resume_extractions.db here, or "
            f"set RESUME_DB_PATH env var to the correct path."
        )
        # ── List .db files nearby so user can find the right one ───
        try:
            cwd_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.db')]
            result["db_files_in_cwd"] = cwd_files
        except OSError:
            pass
        return jsonify(result)

    # ── Database exists — check tables and row counts ──────────────
    try:
        conn = sqlite3.connect(db_abs, timeout=5)
        conn.row_factory = sqlite3.Row

        # List all tables
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        result["tables_found"] = table_names

        # Check critical tables
        for tbl in ["raw_extractions", "structured_extractions", "accuracy_reviews"]:
            if tbl in table_names:
                count = conn.execute(f"SELECT COUNT(*) as c FROM {tbl}").fetchone()["c"]
                result["tables"][tbl] = {"exists": True, "row_count": count}
            else:
                result["tables"][tbl] = {"exists": False, "row_count": 0}

        conn.close()

        # ── Build diagnosis ────────────────────────────────────────
        se = result["tables"].get("structured_extractions", {})
        re_tbl = result["tables"].get("raw_extractions", {})

        if not se.get("exists"):
            result["status"] = "BROKEN"
            result["diagnosis"].append(
                "structured_extractions table MISSING! "
                "This database might not be an AiMerlion database. "
                "Check RESUME_DB_PATH is pointing to the right file."
            )
        elif se.get("row_count", 0) == 0:
            result["status"] = "EMPTY"
            result["diagnosis"].append(
                "structured_extractions table exists but has 0 rows! "
                "Run the extraction pipeline first: python main.py"
            )
        else:
            result["status"] = "HEALTHY"
            result["diagnosis"].append(
                f"All good! {se['row_count']} candidates in "
                f"structured_extractions, {re_tbl.get('row_count', '?')} "
                f"in raw_extractions."
            )

    except Exception as e:
        result["status"] = "ERROR"
        result["diagnosis"].append(f"Database read error: {str(e)}")

    return jsonify(result)

@app.route("/api/test-grid")
def test_grid():
    """🧪 Minimal test page — bypasses all dashboard JS to prove the API works."""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>API Test</title>
    <style>
      body { background:#1a1a2e; color:#e0e0e0; font-family:monospace; padding:20px; }
      table { border-collapse:collapse; width:100%; margin-top:12px; }
      td,th { border:1px solid #333; padding:6px 10px; font-size:12px; text-align:left; }
      th { background:#2a2550; color:#a855f7; }
      .ok { color:#4ade80; } .bad { color:#f87171; }
    </style>
    </head>
    <body>
      <h2>🧪 API Direct Test</h2>
      <div id="out">Loading...</div>
      <script>
        (async () => {
          try {
            const res = await fetch('/api/candidates?page=1&per_page=10');
            const data = await res.json();
            if (!res.ok) {
              document.getElementById('out').innerHTML =
                '<p class="bad">API returned HTTP ' + res.status + '</p>' +
                '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
              return;
            }
            const c = data.candidates || [];
            let html = '<p class="ok">✅ API works! Got ' + c.length +
                       ' candidates (total: ' + (data.pagination||{}).total + ')</p>';
            html += '<table><tr><th>ID</th><th>Name</th><th>Email</th><th>Status</th></tr>';
            c.forEach(r => {
              html += '<tr><td>' + r.candidate_id + '</td><td>' + (r.name||'') +
                      '</td><td>' + (r.email||'') + '</td><td>' + (r.extraction_status||'') + '</td></tr>';
            });
            html += '</table>';
            document.getElementById('out').innerHTML = html;
          } catch (e) {
            document.getElementById('out').innerHTML =
              '<p class="bad">❌ Fetch failed: ' + e.message + '</p>';
          }
        })();
      </script>
    </body>
    </html>
    """)

# =============================================================================
# 🚀 MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " 🧚‍♀️✨ AI EXTRACTION ACCURACY VALIDATOR ✨🧚‍♀️ ".center(62) + "║")
    print("╠" + "═" * 62 + "╣")
    print("║" + f"  📂 Database:  {DATABASE_PATH}".ljust(62) + "║")
    print("║" + f"  🌐 Server:    http://localhost:{PORT}".ljust(62) + "║")
    print("║" + "".ljust(62) + "║")
    print("║" + f"  ✨ DASHBOARD: http://localhost:{PORT}/dashboard".ljust(62) + "║")
    print("║" + f"     ↑ Open this in your browser, darling! 💅".ljust(62) + "║")
    print("║" + "".ljust(62) + "║")
    print("║" + f"  📋 API Docs:  http://localhost:{PORT}/".ljust(62) + "║")
    print("║" + f"  📊 Metrics:   http://localhost:{PORT}/api/metrics".ljust(62) + "║")
    print("║" + "".ljust(62) + "║")
    print("║" + "  Port Map:".ljust(62) + "║")
    print("║" + "    5050 → Review Dashboard".ljust(62) + "║")
    print("║" + "    5055 → NER Annotation Tool".ljust(62) + "║")
    print("║" + "    5001 → Classification Dashboard".ljust(62) + "║")
    print("║" + "    5070 → Accuracy Validator (this!) ✨".ljust(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

        # ── Pre-flight database health check ───────────────────────────
    # 💅 THE BACKSTAGE INSPECTION! Before opening the curtains,
    # we check that the database exists and has data. This catches
    # the #1 issue (wrong DB path) BEFORE the user stares at a
    # blank screen wondering what went wrong! 🎭✨
    db_path = os.path.abspath(DATABASE_PATH)
    print()
    if not os.path.exists(db_path):
        print("  ❌ DATABASE NOT FOUND!")
        print(f"     Expected: {db_path}")
        print()
        print("  💡 Fix options:")
        print("     a) Copy your resume_extractions.db to this folder")
        print("     b) Set env var: set RESUME_DB_PATH=C:\\path\\to\\resume_extractions.db")
        print("     c) Run the extraction pipeline first: python main.py")
        print()
    else:
        # Quick table check — does structured_extractions have data?
        try:
            import sqlite3 as _sq
            _conn = _sq.connect(db_path, timeout=5)
            _count = _conn.execute(
                "SELECT COUNT(*) FROM structured_extractions"
            ).fetchone()[0]
            _conn.close()
            if _count == 0:
                print(f"  ⚠️  Database found but EMPTY (0 candidates)!")
                print(f"     Run the extraction pipeline: python main.py")
            else:
                print(f"  ✅ Database OK: {_count} candidates in structured_extractions")
        except Exception as _e:
            print(f"  ⚠️  Database found but error reading it: {_e}")
    print()

    # Initialize the accuracy_reviews table
    initialize_accuracy_schema()

    # 🚀 Launch!
    app.run(host=HOST, port=PORT, debug=DEBUG)