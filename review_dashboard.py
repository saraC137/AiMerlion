"""
review_dashboard.py

💅✨ FAIRY CODEMOTHER'S RESUME REVIEW DASHBOARD ✨💅

A web-based review interface that lets HR users:
  1. Browse all extracted candidates in a searchable list
  2. View raw resume text SIDE-BY-SIDE with structured extraction
  3. Edit any extracted field inline and save corrections
  4. Mark candidates as reviewed with notes
  5. Highlight text in raw view to cross-reference extraction accuracy

Think of this as the backstage mirror where you check the glam job
BEFORE the candidate data walks the runway! 🎭👑

Architecture:
  - Flask web server (lightweight, no heavy dependencies)
  - Connects to existing resume_extractions.db (read/write)
  - Single-file app with embedded templates (Jinja2)
  - HTMX for snappy inline edits without full page reloads
  - Responsive layout for desktop and tablet

Usage:
    python review_dashboard.py
    # Then open http://localhost:5050 in your browser

Dependencies:
    pip install flask --break-system-packages
"""

import sqlite3
import json
import os
import re
import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
from flask import (
    Flask, render_template_string, request, jsonify,
    redirect, url_for, flash, abort, g
)
from markupsafe import escape

# =============================================================================
# 🔧 APP CONFIGURATION
# =============================================================================

# Database path — same one your extraction pipeline writes to!
# Change this if your DB lives somewhere else, darling 💅
DATABASE_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")

# Server settings
HOST = "0.0.0.0"
PORT = 5050
DEBUG = True  # Set to False in production, sweetie!

# Pagination — how many candidates per page in the list view
CANDIDATES_PER_PAGE = 25

# Fields that are editable in the review interface
# Maps display_name -> db_column_name
EDITABLE_FIELDS = {
    "Name": "name",
    "Email": "email",
    "Phone": "phone",
    "Date of Birth": "date_of_birth",
    "Location": "location",
    "Nationality": "nationality",
    "Skills": "skills_raw",
    "Work Experience": "experience_raw",
    "Education": "education_raw",
    "Summary": "summary",
    "Certifications": "certifications",
    "Languages": "languages",
    "Projects": "projects",
    "Achievements": "achievements",
    "References": "references_info",
    "Hobbies": "hobbies",
}

# Fields considered "critical" — shown with warning badges if empty
CRITICAL_FIELDS = {"name", "email", "phone", "skills_raw", "experience_raw", "education_raw"}

# =============================================================================
# 🏗️ FLASK APP INITIALIZATION
# =============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fairy-codemother-sparkle-2026")

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - 🎭 %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 🔌 DATABASE CONNECTION MANAGEMENT
# Uses Flask's 'g' object for per-request connection lifecycle.
# Think of 'g' as the dressing room — each request gets its own! 🎬
# =============================================================================

def get_db() -> sqlite3.Connection:
    """
    Get or create a database connection for the current request.
    
    Connections are stored in Flask's 'g' object which is unique per request,
    ensuring thread safety without global connection sharing.
    
    Returns:
        sqlite3.Connection with Row factory enabled for dict-like access.
        
    Raises:
        sqlite3.Error: If the database file doesn't exist or is corrupted.
    """
    if "db" not in g:
        if not os.path.exists(DATABASE_PATH):
            logger.error(f"❌ Database not found at: {DATABASE_PATH}")
            abort(500, description=f"Database not found: {DATABASE_PATH}")
        
        try:
            g.db = sqlite3.connect(DATABASE_PATH, timeout=15)
            g.db.row_factory = sqlite3.Row
            # Enable WAL for concurrent read performance during reviews
            g.db.execute("PRAGMA journal_mode=WAL")
            g.db.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            abort(500, description=f"Database connection failed: {e}")
    
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """
    Auto-close the database connection when the request ends.
    Like wiping off your makeup after the show — clean up! 🧹
    """
    db = g.pop("db", None)
    if db is not None:
        db.close()


# =============================================================================
# 🛡️ INPUT VALIDATION HELPERS
# These guard against SQL injection and bad data. Safety first, darling! 💪
# =============================================================================

def validate_candidate_id(candidate_id: Any) -> int:
    """
    Validate and sanitize candidate_id input.
    
    Ensures the ID is a positive integer to prevent injection attacks.
    Think of this as the bouncer at the VIP door — no fakes allowed! 🚪
    
    Args:
        candidate_id: Raw input from URL or form data.
        
    Returns:
        Validated integer candidate_id.
        
    Raises:
        ValueError: If the ID is not a valid positive integer.
    """
    try:
        cid = int(candidate_id)
        if cid <= 0:
            raise ValueError("Candidate ID must be positive")
        return cid
    except (TypeError, ValueError) as e:
        logger.warning(f"⚠️ Invalid candidate ID received: {candidate_id}")
        abort(400, description=f"Invalid candidate ID: {candidate_id}")


def validate_field_name(field_name: str) -> str:
    """
    Validate that the field name maps to an allowed database column.
    
    CRITICAL for preventing SQL injection — we NEVER interpolate raw
    user input into SQL. Only whitelisted column names get through! 🛡️
    
    Args:
        field_name: The display name of the field to validate.
        
    Returns:
        The corresponding database column name.
        
    Raises:
        400 error if the field is not in the allowed list.
    """
    col = EDITABLE_FIELDS.get(field_name)
    if not col:
        # Also try by column name directly (for API calls)
        if field_name in EDITABLE_FIELDS.values():
            return field_name
        logger.warning(f"⚠️ Attempted edit on non-whitelisted field: {field_name}")
        abort(400, description=f"Field not editable: {field_name}")
    return col


def sanitize_text_input(text: str, max_length: int = 50000) -> str:
    """
    Sanitize user text input for safe database storage.
    
    Strips dangerous patterns while preserving legitimate resume content
    like special characters in names (O'Brien, García, etc.) and
    technical symbols in skills (C++, C#, etc.).
    
    Args:
        text: Raw text input from the user.
        max_length: Maximum allowed character count (default 50K for experience fields).
        
    Returns:
        Sanitized text string, truncated if over max_length.
    """
    if not text:
        return ""
    
    # Truncate to prevent absurdly large inputs
    text = text[:max_length]
    
    # Strip null bytes (can cause DB issues)
    text = text.replace("\x00", "")
    
    return text.strip()


# =============================================================================
# 📊 DATA RETRIEVAL FUNCTIONS
# The backstage crew that fetches data from the database! 🎬
# =============================================================================

def get_candidates_list(
    page: int = 1,
    per_page: int = CANDIDATES_PER_PAGE,
    search_query: str = "",
    status_filter: str = "",
    review_filter: str = "",
    sort_by: str = "candidate_id",
    sort_order: str = "ASC"
) -> Tuple[List[Dict], int]:
    """
    Retrieve a paginated, filterable list of candidates.
    
    This is the CASTING CALL list — all candidates lined up with their
    basic stats so the reviewer can pick who to examine first! 📋
    
    Args:
        page:           Current page number (1-indexed).
        per_page:       Number of results per page.
        search_query:   Filter by name, email, or candidate ID.
        status_filter:  Filter by extraction status (e.g., 'Complete', 'Partial').
        review_filter:  Filter by review state ('reviewed', 'unreviewed', '').
        sort_by:        Column to sort by (whitelisted options only).
        sort_order:     'ASC' or 'DESC'.
        
    Returns:
        Tuple of (list_of_candidate_dicts, total_count).
    """
    db = get_db()
    
    # Whitelist sortable columns to prevent SQL injection
    allowed_sorts = {
        "candidate_id": "s.candidate_id",
        "name": "s.name",
        "extraction_status": "s.extraction_status",
        "updated_at": "s.updated_at",
        "reviewed": "s.reviewed",
    }
    sort_col = allowed_sorts.get(sort_by, "s.candidate_id")
    sort_dir = "DESC" if sort_order.upper() == "DESC" else "ASC"
    
    # Build dynamic WHERE clause
    conditions = []
    params = []
    
    if search_query:
        # Search across name, email, and candidate_id
        conditions.append(
            "(s.name LIKE ? OR s.email LIKE ? OR CAST(s.candidate_id AS TEXT) LIKE ?)"
        )
        like_param = f"%{search_query}%"
        params.extend([like_param, like_param, like_param])
    
    if status_filter:
        conditions.append("s.extraction_status = ?")
        params.append(status_filter)
    
    if review_filter == "reviewed":
        conditions.append("s.reviewed = 1")
    elif review_filter == "unreviewed":
        conditions.append("(s.reviewed = 0 OR s.reviewed IS NULL)")
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    # Count total matching results for pagination
    count_sql = f"""
        SELECT COUNT(*) as total 
        FROM structured_extractions s
        {where_clause}
    """
    total = db.execute(count_sql, params).fetchone()["total"]
    
    # Fetch the current page of results
    offset = (page - 1) * per_page
    data_sql = f"""
        SELECT 
            s.candidate_id,
            s.name,
            s.email,
            s.phone,
            s.extraction_status,
            s.extraction_method,
            s.ai_assisted,
            s.reviewed,
            s.updated_at,
            s.created_at,
            -- Calculate completeness score (how many critical fields are filled)
            (CASE WHEN s.name IS NOT NULL AND s.name != '' THEN 1 ELSE 0 END +
             CASE WHEN s.email IS NOT NULL AND s.email != '' THEN 1 ELSE 0 END +
             CASE WHEN s.phone IS NOT NULL AND s.phone != '' THEN 1 ELSE 0 END +
             CASE WHEN s.skills_raw IS NOT NULL AND s.skills_raw != '' THEN 1 ELSE 0 END +
             CASE WHEN s.experience_raw IS NOT NULL AND s.experience_raw != '' THEN 1 ELSE 0 END +
             CASE WHEN s.education_raw IS NOT NULL AND s.education_raw != '' THEN 1 ELSE 0 END
            ) as completeness_score,
            r.text_length
        FROM structured_extractions s
        LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
        {where_clause}
        ORDER BY {sort_col} {sort_dir}
        LIMIT ? OFFSET ?
    """
    params.extend([per_page, offset])
    
    rows = db.execute(data_sql, params).fetchall()
    candidates = [dict(row) for row in rows]
    
    return candidates, total


def get_candidate_detail(candidate_id: int) -> Optional[Dict]:
    """
    Retrieve the FULL extraction data for a single candidate.
    
    This is the dressing room close-up — every detail under the spotlight! 💡
    
    Args:
        candidate_id: Validated candidate ID.
        
    Returns:
        Dict with all structured_extraction fields, or None if not found.
    """
    db = get_db()
    
    row = db.execute("""
        SELECT s.*, r.text_length, r.resume_language, r.filenames,
               r.extraction_timestamp as raw_extracted_at
        FROM structured_extractions s
        LEFT JOIN raw_extractions r ON s.candidate_id = r.candidate_id
        WHERE s.candidate_id = ?
        ORDER BY s.created_at DESC
        LIMIT 1
    """, (candidate_id,)).fetchone()
    
    return dict(row) if row else None


def get_raw_text(candidate_id: int) -> Optional[str]:
    """
    Retrieve the original raw text for a candidate's resume.
    
    This is the "before" photo — the unedited, unfiltered truth! 📸
    
    Args:
        candidate_id: Validated candidate ID.
        
    Returns:
        Raw text string, or None if no raw text exists.
    """
    db = get_db()
    
    row = db.execute("""
        SELECT raw_text FROM raw_extractions
        WHERE candidate_id = ?
        ORDER BY extraction_timestamp DESC
        LIMIT 1
    """, (candidate_id,)).fetchone()
    
    return row["raw_text"] if row else None


def get_extraction_history(candidate_id: int) -> List[Dict]:
    """
    Retrieve the extraction log history for a candidate.
    
    Every attempt, every method, every override — the full drama timeline! 🎬
    
    Args:
        candidate_id: Validated candidate ID.
        
    Returns:
        List of extraction log entries, ordered by timestamp.
    """
    db = get_db()
    
    rows = db.execute("""
        SELECT field_name, extraction_method, extracted_value,
               was_successful, was_overridden, override_reason,
               error_message, timestamp
        FROM extraction_log
        WHERE candidate_id = ?
        ORDER BY timestamp DESC
        LIMIT 100
    """, (candidate_id,)).fetchall()
    
    return [dict(row) for row in rows]


def get_status_options() -> List[str]:
    """
    Get all unique extraction status values from the database.
    Used to populate filter dropdowns dynamically.
    """
    db = get_db()
    rows = db.execute("""
        SELECT DISTINCT extraction_status 
        FROM structured_extractions 
        WHERE extraction_status IS NOT NULL
        ORDER BY extraction_status
    """).fetchall()
    return [row["extraction_status"] for row in rows]


def get_dashboard_stats() -> Dict:
    """
    Calculate summary statistics for the dashboard header.
    
    Returns:
        Dict with total counts, review progress, and quality metrics.
    """
    db = get_db()
    stats = {}
    
    try:
        # Total candidates
        stats["total"] = db.execute(
            "SELECT COUNT(*) as c FROM structured_extractions"
        ).fetchone()["c"]
        
        # Reviewed count
        stats["reviewed"] = db.execute(
            "SELECT COUNT(*) as c FROM structured_extractions WHERE reviewed = 1"
        ).fetchone()["c"]
        
        # Unreviewed count
        stats["unreviewed"] = stats["total"] - stats["reviewed"]
        
        # Review progress percentage
        stats["review_pct"] = (
            round((stats["reviewed"] / stats["total"]) * 100, 1)
            if stats["total"] > 0 else 0
        )
        
        # Candidates with missing critical fields
        stats["incomplete"] = db.execute("""
            SELECT COUNT(*) as c FROM structured_extractions
            WHERE name IS NULL OR name = '' 
               OR email IS NULL OR email = ''
               OR phone IS NULL OR phone = ''
        """).fetchone()["c"]
        
        # AI-assisted count
        stats["ai_assisted"] = db.execute(
            "SELECT COUNT(*) as c FROM structured_extractions WHERE ai_assisted = 1"
        ).fetchone()["c"]
        
    except sqlite3.Error as e:
        logger.error(f"❌ Stats query error: {e}")
        stats = {"total": 0, "reviewed": 0, "unreviewed": 0,
                 "review_pct": 0, "incomplete": 0, "ai_assisted": 0}
    
    return stats


# =============================================================================
# ✏️ DATA UPDATE FUNCTIONS
# The glam squad that fixes extraction mistakes! 💄
# =============================================================================

def update_extraction_field(
    candidate_id: int,
    column_name: str,
    new_value: str,
    editor_note: str = ""
) -> bool:
    """
    Update a single field in the structured_extractions table.
    
    Also logs the change in extraction_log for full audit trail.
    Think of this as the tailor making last-minute alterations
    before the gown hits the runway! ✂️👗
    
    Args:
        candidate_id: Validated candidate ID.
        column_name:  Whitelisted database column name.
        new_value:    The corrected value (already sanitized).
        editor_note:  Optional note explaining why the change was made.
        
    Returns:
        True if update succeeded, False otherwise.
    """
    db = get_db()
    
    try:
        # Fetch old value for the audit log
        old_row = db.execute(f"""
            SELECT {column_name} FROM structured_extractions
            WHERE candidate_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (candidate_id,)).fetchone()
        
        old_value = old_row[column_name] if old_row else None
        
        # Skip if the value hasn't actually changed (prevent ghost edits)
        if old_value == new_value:
            logger.info(f"ℹ️ No change detected for {column_name} on candidate {candidate_id}")
            return True
        
        now = datetime.datetime.now().isoformat()
        
        # Update the field
        # NOTE: column_name is WHITELISTED via validate_field_name(), safe for interpolation
        db.execute(f"""
            UPDATE structured_extractions
            SET {column_name} = ?,
                updated_at = ?,
                review_notes = COALESCE(review_notes, '') || ?
            WHERE candidate_id = ?
        """, (
            new_value,
            now,
            f"\n[DASHBOARD EDIT {now}] {column_name}: changed | {editor_note}",
            candidate_id
        ))
        
        # Log the correction in extraction_log for audit trail
        db.execute("""
            INSERT INTO extraction_log (
                candidate_id, field_name, extraction_method,
                extracted_value, was_successful, was_overridden,
                override_reason, timestamp
            ) VALUES (?, ?, 'dashboard_manual_edit', ?, 1, 1, ?, ?)
        """, (
            candidate_id,
            column_name,
            new_value[:500] if new_value else None,
            f"Dashboard edit. Old: '{str(old_value)[:200]}' | Note: {editor_note}",
            now
        ))
        
        db.commit()
        logger.info(f"✏️ Updated {column_name} for candidate {candidate_id}")
        return True
        
    except sqlite3.Error as e:
        db.rollback()
        logger.error(f"❌ Update failed for candidate {candidate_id}, field {column_name}: {e}")
        return False


def mark_candidate_reviewed(candidate_id: int, notes: str = "") -> bool:
    """
    Mark a candidate's extraction as reviewed by a human.
    
    This is the FINAL STAMP OF APPROVAL — the data quality seal! ✅👑
    
    Args:
        candidate_id: Validated candidate ID.
        notes:        Optional reviewer notes.
        
    Returns:
        True if marking succeeded, False otherwise.
    """
    db = get_db()
    
    try:
        now = datetime.datetime.now().isoformat()
        db.execute("""
            UPDATE structured_extractions
            SET reviewed = 1,
                review_notes = COALESCE(review_notes, '') || ?,
                updated_at = ?
            WHERE candidate_id = ?
        """, (
            f"\n[REVIEWED {now}] {notes}" if notes else f"\n[REVIEWED {now}]",
            now,
            candidate_id
        ))
        db.commit()
        logger.info(f"✅ Candidate {candidate_id} marked as reviewed")
        return True
        
    except sqlite3.Error as e:
        db.rollback()
        logger.error(f"❌ Failed to mark candidate {candidate_id} as reviewed: {e}")
        return False


def unmark_candidate_reviewed(candidate_id: int) -> bool:
    """
    Remove the 'reviewed' status from a candidate (for re-review).
    
    Sometimes you need a second look, darling! 🔄
    """
    db = get_db()
    
    try:
        now = datetime.datetime.now().isoformat()
        db.execute("""
            UPDATE structured_extractions
            SET reviewed = 0,
                review_notes = COALESCE(review_notes, '') || ?,
                updated_at = ?
            WHERE candidate_id = ?
        """, (
            f"\n[UNREVIEW {now}] Marked for re-review",
            now,
            candidate_id
        ))
        db.commit()
        return True
        
    except sqlite3.Error as e:
        db.rollback()
        logger.error(f"❌ Failed to unmark candidate {candidate_id}: {e}")
        return False


def bulk_save_fields(candidate_id: int, field_updates: Dict[str, str]) -> Tuple[int, int]:
    """
    Save multiple field edits at once (from the full edit form).
    
    The full wardrobe change — everything gets updated in one go! 👗👠💄
    
    Args:
        candidate_id: Validated candidate ID.
        field_updates: Dict mapping display_name -> new_value.
        
    Returns:
        Tuple of (success_count, failure_count).
    """
    success = 0
    failed = 0
    
    for display_name, new_value in field_updates.items():
        col = EDITABLE_FIELDS.get(display_name)
        if not col:
            # Try column name directly
            if display_name in EDITABLE_FIELDS.values():
                col = display_name
            else:
                logger.warning(f"⚠️ Skipping unknown field: {display_name}")
                failed += 1
                continue
        
        sanitized_value = sanitize_text_input(new_value)
        if update_extraction_field(candidate_id, col, sanitized_value, "bulk_save"):
            success += 1
        else:
            failed += 1
    
    return success, failed


# =============================================================================
# 🛣️ FLASK ROUTES — The Red Carpet Walkway! 🎬
# =============================================================================

@app.route("/")
def index():
    """
    🏠 Dashboard Home — The Main Stage!
    
    Shows summary stats and a paginated, searchable candidate list.
    """
    # Parse query parameters
    page = request.args.get("page", 1, type=int)
    search = request.args.get("search", "", type=str).strip()
    status = request.args.get("status", "", type=str).strip()
    review = request.args.get("review", "", type=str).strip()
    sort_by = request.args.get("sort", "candidate_id", type=str)
    sort_order = request.args.get("order", "ASC", type=str)
    
    # Clamp page to valid range
    page = max(1, page)
    
    # Fetch data
    stats = get_dashboard_stats()
    candidates, total = get_candidates_list(
        page=page, search_query=search, status_filter=status,
        review_filter=review, sort_by=sort_by, sort_order=sort_order
    )
    status_options = get_status_options()
    
    # Calculate pagination metadata
    total_pages = max(1, (total + CANDIDATES_PER_PAGE - 1) // CANDIDATES_PER_PAGE)
    page = min(page, total_pages)
    
    return render_template_string(
        INDEX_TEMPLATE,
        stats=stats,
        candidates=candidates,
        total=total,
        page=page,
        total_pages=total_pages,
        search=search,
        status_filter=status,
        review_filter=review,
        sort_by=sort_by,
        sort_order=sort_order,
        status_options=status_options,
        editable_fields=EDITABLE_FIELDS,
        critical_fields=CRITICAL_FIELDS,
    )


@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id):
    """
    🔍 Candidate Detail View — The Side-by-Side Spotlight!
    
    Left panel: Raw resume text (searchable, highlightable)
    Right panel: Structured extraction with inline edit capability
    """
    cid = validate_candidate_id(candidate_id)
    
    detail = get_candidate_detail(cid)
    if not detail:
        abort(404, description=f"Candidate {cid} not found")
    
    raw_text = get_raw_text(cid)
    history = get_extraction_history(cid)
    
    return render_template_string(
        DETAIL_TEMPLATE,
        candidate=detail,
        raw_text=raw_text or "(No raw text available)",
        history=history,
        editable_fields=EDITABLE_FIELDS,
        critical_fields=CRITICAL_FIELDS,
    )


@app.route("/api/save-field", methods=["POST"])
def api_save_field():
    """
    ✏️ API: Save a single field edit (called via HTMX/fetch).
    
    Expects JSON: {candidate_id, field_name, new_value, note}
    Returns JSON: {success, message, old_value, new_value}
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "message": "Invalid JSON payload"}), 400
    
    cid = validate_candidate_id(data.get("candidate_id"))
    field_name = data.get("field_name", "")
    new_value = sanitize_text_input(data.get("new_value", ""))
    note = sanitize_text_input(data.get("note", ""), max_length=500)
    
    # Validate field name against whitelist
    col = validate_field_name(field_name)
    
    success = update_extraction_field(cid, col, new_value, note)
    
    if success:
        return jsonify({
            "success": True,
            "message": f"✅ {field_name} updated successfully!",
            "new_value": new_value
        })
    else:
        return jsonify({
            "success": False,
            "message": f"❌ Failed to update {field_name}"
        }), 500


@app.route("/api/save-all", methods=["POST"])
def api_save_all():
    """
    💾 API: Bulk save all edited fields at once.
    
    Expects JSON: {candidate_id, fields: {field_name: value, ...}}
    Returns JSON: {success, saved_count, failed_count}
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "message": "Invalid JSON payload"}), 400
    
    cid = validate_candidate_id(data.get("candidate_id"))
    fields = data.get("fields", {})
    
    if not fields:
        return jsonify({"success": False, "message": "No fields to save"}), 400
    
    saved, failed = bulk_save_fields(cid, fields)
    
    return jsonify({
        "success": failed == 0,
        "message": f"✅ Saved {saved} fields" + (f", ❌ {failed} failed" if failed else ""),
        "saved_count": saved,
        "failed_count": failed
    })


@app.route("/api/mark-reviewed", methods=["POST"])
def api_mark_reviewed():
    """
    ✅ API: Mark a candidate as reviewed.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "message": "Invalid JSON"}), 400
    
    cid = validate_candidate_id(data.get("candidate_id"))
    notes = sanitize_text_input(data.get("notes", ""), max_length=1000)
    
    success = mark_candidate_reviewed(cid, notes)
    return jsonify({
        "success": success,
        "message": "✅ Marked as reviewed!" if success else "❌ Failed to mark"
    })


@app.route("/api/unmark-reviewed", methods=["POST"])
def api_unmark_reviewed():
    """
    🔄 API: Remove reviewed status (send back for re-review).
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "message": "Invalid JSON"}), 400
    
    cid = validate_candidate_id(data.get("candidate_id"))
    
    success = unmark_candidate_reviewed(cid)
    return jsonify({
        "success": success,
        "message": "🔄 Sent back for re-review" if success else "❌ Failed"
    })


@app.route("/api/search-raw-text", methods=["GET"])
def api_search_raw_text():
    """
    🔍 API: Search within a candidate's raw text.
    Returns match positions for highlighting.
    """
    cid = validate_candidate_id(request.args.get("candidate_id"))
    query = request.args.get("q", "").strip()
    
    if not query or len(query) < 2:
        return jsonify({"matches": 0, "positions": []})
    
    raw = get_raw_text(cid)
    if not raw:
        return jsonify({"matches": 0, "positions": []})
    
    # Find all occurrences (case-insensitive)
    positions = []
    raw_lower = raw.lower()
    query_lower = query.lower()
    start = 0
    while True:
        idx = raw_lower.find(query_lower, start)
        if idx == -1:
            break
        positions.append({"start": idx, "end": idx + len(query)})
        start = idx + 1
        # Safety limit — don't return thousands of matches
        if len(positions) >= 200:
            break
    
    return jsonify({"matches": len(positions), "positions": positions})


# =============================================================================
# 🎨 HTML TEMPLATES — The Stage Design! 🎭
# =============================================================================

# ---- Shared CSS that both templates use ----
SHARED_CSS = """
/* ============================================================
   🎨 FAIRY CODEMOTHER'S RESUME REVIEW DASHBOARD — STYLESHEET
   Aesthetic: Editorial/Magazine meets Data Dashboard
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    /* === Core Palette === */
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --bg-elevated: #252c35;
    --bg-input: #1a1f27;
    
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --text-link: #7ee8fa;
    
    --accent-primary: #7ee8fa;
    --accent-secondary: #eeb8ff;
    --accent-success: #56d364;
    --accent-warning: #e3b341;
    --accent-danger: #f85149;
    --accent-info: #79c0ff;
    
    --border-default: #30363d;
    --border-muted: #21262d;
    
    /* === Gradients === */
    --gradient-header: linear-gradient(135deg, #1a1044 0%, #0e1117 50%, #0a1628 100%);
    --gradient-accent: linear-gradient(135deg, #7ee8fa, #eeb8ff);
    --gradient-card-hover: linear-gradient(135deg, rgba(126,232,250,0.05), rgba(238,184,255,0.05));
    
    /* === Spacing & Sizing === */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --sidebar-width: 50%;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'DM Sans', system-ui, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
}

/* === SCROLLBAR STYLING === */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { 
    background: var(--border-default); 
    border-radius: 4px; 
}
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* === HEADER === */
.dashboard-header {
    background: var(--gradient-header);
    border-bottom: 1px solid var(--border-default);
    padding: 20px 32px;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
}

.header-inner {
    max-width: 1600px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.header-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 400;
    background: var(--gradient-accent);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    white-space: nowrap;
}

.header-title span {
    font-style: italic;
}

/* === STAT CARDS === */
.stats-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: 12px 18px;
    min-width: 120px;
    text-align: center;
    transition: border-color 0.2s;
}

.stat-card:hover { border-color: var(--accent-primary); }

.stat-number {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.2;
}

.stat-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

/* === FILTERS BAR === */
.filters-bar {
    background: var(--bg-secondary);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin: 20px 32px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    max-width: 1600px;
    margin-left: auto;
    margin-right: auto;
}

.filter-input, .filter-select {
    background: var(--bg-input);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    padding: 8px 14px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    outline: none;
    transition: border-color 0.2s;
}

.filter-input:focus, .filter-select:focus {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 2px rgba(126,232,250,0.15);
}

.filter-input { flex: 1; min-width: 200px; }
.filter-select { min-width: 140px; }

.filter-btn {
    background: var(--gradient-accent);
    color: var(--bg-primary);
    border: none;
    border-radius: var(--radius-sm);
    padding: 8px 20px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
}

.filter-btn:hover { opacity: 0.85; }

.filter-btn.secondary {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border-default);
}

/* === CANDIDATE TABLE === */
.table-container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 0 32px 32px;
    overflow-x: auto;
}

.candidate-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    overflow: hidden;
}

.candidate-table th {
    background: var(--bg-elevated);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-default);
    position: sticky;
    top: 0;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
}

.candidate-table th:hover { color: var(--accent-primary); }

.candidate-table th a {
    color: inherit;
    text-decoration: none;
}

.candidate-table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-muted);
    font-size: 0.9rem;
    vertical-align: middle;
}

.candidate-table tr:last-child td { border-bottom: none; }

.candidate-table tr:hover td {
    background: var(--gradient-card-hover);
}

.candidate-name-link {
    color: var(--accent-primary);
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s;
}

.candidate-name-link:hover {
    color: var(--accent-secondary);
    text-decoration: underline;
}

/* === BADGES === */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    white-space: nowrap;
}

.badge-success { background: rgba(86,211,100,0.15); color: var(--accent-success); }
.badge-warning { background: rgba(227,179,65,0.15); color: var(--accent-warning); }
.badge-danger  { background: rgba(248,81,73,0.15); color: var(--accent-danger); }
.badge-info    { background: rgba(121,192,255,0.15); color: var(--accent-info); }
.badge-muted   { background: rgba(110,118,129,0.15); color: var(--text-muted); }
.badge-reviewed { background: rgba(86,211,100,0.15); color: var(--accent-success); }
.badge-ai      { background: rgba(238,184,255,0.15); color: var(--accent-secondary); }

/* === COMPLETENESS BAR === */
.completeness-bar {
    width: 80px;
    height: 6px;
    background: var(--bg-primary);
    border-radius: 3px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
}

.completeness-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}

/* === PAGINATION === */
.pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 20px;
}

.pagination a, .pagination span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    height: 36px;
    padding: 0 10px;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    text-decoration: none;
    transition: all 0.15s;
}

.pagination a {
    background: var(--bg-card);
    color: var(--text-secondary);
    border: 1px solid var(--border-default);
}

.pagination a:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border-color: var(--accent-primary);
}

.pagination .current {
    background: var(--gradient-accent);
    color: var(--bg-primary);
    font-weight: 700;
    border: none;
}

.pagination .disabled {
    opacity: 0.35;
    pointer-events: none;
}

/* === DETAIL PAGE LAYOUT === */
.detail-container {
    display: flex;
    height: calc(100vh - 70px);
    overflow: hidden;
}

.raw-panel {
    width: var(--sidebar-width);
    border-right: 1px solid var(--border-default);
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary);
}

.extraction-panel {
    width: calc(100% - var(--sidebar-width));
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
}

.panel-header {
    padding: 14px 20px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border-default);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-shrink: 0;
}

.panel-title {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 1.1rem;
    color: var(--text-primary);
}

.panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

/* === RAW TEXT DISPLAY === */
.raw-text-content {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-secondary);
    padding: 4px;
}

.raw-text-content mark {
    background: rgba(126,232,250,0.3);
    color: var(--text-primary);
    border-radius: 2px;
    padding: 1px 2px;
}

/* === RAW TEXT SEARCH BOX === */
.raw-search-box {
    display: flex;
    align-items: center;
    gap: 8px;
}

.raw-search-input {
    background: var(--bg-input);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    padding: 6px 12px;
    font-size: 0.82rem;
    outline: none;
    width: 180px;
    font-family: 'DM Sans', sans-serif;
}

.raw-search-input:focus {
    border-color: var(--accent-primary);
}

.search-count {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
}

/* === EXTRACTION FIELDS === */
.field-group {
    margin-bottom: 16px;
    border: 1px solid var(--border-muted);
    border-radius: var(--radius-md);
    overflow: hidden;
    transition: border-color 0.2s;
}

.field-group:hover { border-color: var(--border-default); }
.field-group.field-empty { border-left: 3px solid var(--accent-warning); }
.field-group.field-critical-empty { border-left: 3px solid var(--accent-danger); }

.field-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    background: var(--bg-elevated);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
}

.field-label:hover { color: var(--accent-primary); }

.field-value {
    padding: 10px 14px;
    background: var(--bg-card);
    min-height: 36px;
}

.field-value-text {
    font-size: 0.9rem;
    color: var(--text-primary);
    white-space: pre-wrap;
    word-break: break-word;
}

.field-value-text.empty-field {
    color: var(--text-muted);
    font-style: italic;
}

/* === EDIT MODE === */
.field-edit-area {
    display: none;
    padding: 10px 14px;
    background: var(--bg-secondary);
}

.field-edit-area.active { display: block; }

.field-textarea {
    width: 100%;
    min-height: 60px;
    max-height: 400px;
    background: var(--bg-input);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    padding: 10px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.5;
    resize: vertical;
    outline: none;
}

.field-textarea:focus {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 2px rgba(126,232,250,0.15);
}

.field-edit-actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    align-items: center;
}

.btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: var(--radius-sm);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.15s;
}

.btn-save {
    background: var(--accent-success);
    color: var(--bg-primary);
}
.btn-save:hover { opacity: 0.85; }

.btn-cancel {
    background: transparent;
    color: var(--text-secondary);
    border-color: var(--border-default);
}
.btn-cancel:hover { color: var(--text-primary); }

.btn-review {
    background: var(--gradient-accent);
    color: var(--bg-primary);
    padding: 10px 28px;
    font-size: 0.9rem;
    border-radius: var(--radius-md);
}
.btn-review:hover { opacity: 0.85; }

.btn-unreview {
    background: var(--bg-elevated);
    color: var(--accent-warning);
    border-color: var(--accent-warning);
    padding: 10px 28px;
    font-size: 0.9rem;
    border-radius: var(--radius-md);
}

/* === TOAST NOTIFICATIONS === */
.toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.toast {
    background: var(--bg-card);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: 12px 20px;
    font-size: 0.88rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    animation: slideIn 0.3s ease, fadeOut 0.5s ease 2.5s;
    max-width: 360px;
}

.toast.success { border-left: 3px solid var(--accent-success); }
.toast.error { border-left: 3px solid var(--accent-danger); }

@keyframes slideIn {
    from { opacity: 0; transform: translateX(40px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes fadeOut {
    from { opacity: 1; }
    to { opacity: 0; }
}

/* === EXTRACTION HISTORY === */
.history-toggle {
    padding: 12px 20px;
    background: var(--bg-elevated);
    border-top: 1px solid var(--border-default);
    cursor: pointer;
    font-size: 0.82rem;
    color: var(--text-secondary);
    text-align: center;
    flex-shrink: 0;
}
.history-toggle:hover { color: var(--accent-primary); }

.history-panel {
    display: none;
    max-height: 300px;
    overflow-y: auto;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-default);
    padding: 12px 20px;
}

.history-panel.open { display: block; }

.history-entry {
    font-size: 0.78rem;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-muted);
    color: var(--text-secondary);
    font-family: 'DM Mono', monospace;
}

.history-entry:last-child { border-bottom: none; }
.history-field { color: var(--accent-primary); font-weight: 600; }
.history-method { color: var(--accent-secondary); }
.history-success { color: var(--accent-success); }
.history-fail { color: var(--accent-danger); }

/* === BACK LINK === */
.back-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: color 0.15s;
}
.back-link:hover { color: var(--accent-primary); }

/* === RESIZE HANDLE (between panels) === */
.resize-handle {
    width: 5px;
    cursor: col-resize;
    background: var(--border-default);
    transition: background 0.2s;
    flex-shrink: 0;
}
.resize-handle:hover { background: var(--accent-primary); }

/* === LINE NUMBERS IN RAW TEXT === */
.line-numbers {
    counter-reset: line;
}
.line-numbers .line::before {
    counter-increment: line;
    content: counter(line);
    display: inline-block;
    width: 3em;
    text-align: right;
    margin-right: 1em;
    color: var(--text-muted);
    font-size: 0.72rem;
    user-select: none;
    opacity: 0.5;
}

/* === RESPONSIVE ADJUSTMENTS === */
@media (max-width: 1200px) {
    .detail-container { flex-direction: column; height: auto; }
    .raw-panel, .extraction-panel { width: 100%; }
    .raw-panel { max-height: 50vh; border-right: none; border-bottom: 1px solid var(--border-default); }
    .resize-handle { display: none; }
}

@media (max-width: 768px) {
    .dashboard-header { padding: 14px 16px; }
    .header-inner { flex-direction: column; align-items: flex-start; }
    .filters-bar { margin: 12px 16px; padding: 12px; flex-direction: column; }
    .table-container { padding: 0 16px 16px; }
    .candidate-table { font-size: 0.82rem; }
}
"""

# ---- Index (List) Template ----
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Review Dashboard ✨</title>
    <style>""" + SHARED_CSS + """</style>
</head>
<body>

<!-- === HEADER === -->
<header class="dashboard-header">
    <div class="header-inner">
        <h1 class="header-title">✨ Resume Review <span>Dashboard</span></h1>
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-number" style="color: var(--accent-primary)">{{ stats.total }}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color: var(--accent-success)">{{ stats.reviewed }}</div>
                <div class="stat-label">Reviewed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color: var(--accent-warning)">{{ stats.unreviewed }}</div>
                <div class="stat-label">Pending</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color: var(--accent-danger)">{{ stats.incomplete }}</div>
                <div class="stat-label">Incomplete</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" style="color: var(--accent-secondary)">{{ stats.review_pct }}%</div>
                <div class="stat-label">Progress</div>
            </div>
        </div>
    </div>
</header>

<!-- === FILTERS === -->
<form class="filters-bar" method="GET" action="/">
    <input class="filter-input" type="text" name="search"
           placeholder="🔍 Search name, email, or ID..."
           value="{{ search }}" autocomplete="off">
    
    <select class="filter-select" name="status">
        <option value="">All Statuses</option>
        {% for opt in status_options %}
        <option value="{{ opt }}" {{ 'selected' if status_filter == opt }}>{{ opt }}</option>
        {% endfor %}
    </select>
    
    <select class="filter-select" name="review">
        <option value="" {{ 'selected' if review_filter == '' }}>All Reviews</option>
        <option value="reviewed" {{ 'selected' if review_filter == 'reviewed' }}>✅ Reviewed</option>
        <option value="unreviewed" {{ 'selected' if review_filter == 'unreviewed' }}>⏳ Unreviewed</option>
    </select>
    
    <button class="filter-btn" type="submit">Filter</button>
    <a class="filter-btn secondary" href="/" style="text-decoration:none">Clear</a>
</form>

<!-- === CANDIDATE TABLE === -->
<div class="table-container">
    {% if candidates %}
    <table class="candidate-table">
        <thead>
            <tr>
                {% set next_order = 'DESC' if sort_order == 'ASC' else 'ASC' %}
                <th><a href="?sort=candidate_id&order={{ next_order }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}">
                    ID {{ '▲' if sort_by == 'candidate_id' and sort_order == 'ASC' else ('▼' if sort_by == 'candidate_id' else '') }}
                </a></th>
                <th><a href="?sort=name&order={{ next_order }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}">
                    Name {{ '▲' if sort_by == 'name' and sort_order == 'ASC' else ('▼' if sort_by == 'name' else '') }}
                </a></th>
                <th>Email</th>
                <th>Phone</th>
                <th><a href="?sort=extraction_status&order={{ next_order }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}">
                    Status
                </a></th>
                <th>Completeness</th>
                <th>Method</th>
                <th><a href="?sort=reviewed&order={{ next_order }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}">
                    Review
                </a></th>
                <th><a href="?sort=updated_at&order={{ next_order }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}">
                    Updated
                </a></th>
            </tr>
        </thead>
        <tbody>
            {% for c in candidates %}
            <tr>
                <td style="font-family:'DM Mono',monospace; font-size:0.82rem; color:var(--text-muted)">
                    {{ c.candidate_id }}
                </td>
                <td>
                    <a class="candidate-name-link" href="/candidate/{{ c.candidate_id }}">
                        {{ c.name or '—' }}
                    </a>
                </td>
                <td style="font-size:0.82rem; color:var(--text-secondary)">{{ c.email or '—' }}</td>
                <td style="font-size:0.82rem; color:var(--text-secondary)">{{ c.phone or '—' }}</td>
                <td>
                    {% if c.extraction_status == 'Complete' or c.extraction_status == 'Success' %}
                        <span class="badge badge-success">{{ c.extraction_status }}</span>
                    {% elif c.extraction_status == 'Partial' %}
                        <span class="badge badge-warning">{{ c.extraction_status }}</span>
                    {% elif c.extraction_status == 'Failed' %}
                        <span class="badge badge-danger">{{ c.extraction_status }}</span>
                    {% else %}
                        <span class="badge badge-muted">{{ c.extraction_status or 'Unknown' }}</span>
                    {% endif %}
                </td>
                <td>
                    {% set pct = (c.completeness_score / 6 * 100)|int %}
                    {% set color = 'var(--accent-success)' if pct >= 80 else ('var(--accent-warning)' if pct >= 50 else 'var(--accent-danger)') %}
                    <span class="completeness-bar">
                        <span class="completeness-fill" style="width:{{ pct }}%; background:{{ color }}"></span>
                    </span>
                    <span style="font-size:0.75rem; color:var(--text-muted); font-family:'DM Mono',monospace">{{ c.completeness_score }}/6</span>
                </td>
                <td>
                    {% if c.ai_assisted %}
                        <span class="badge badge-ai">AI</span>
                    {% else %}
                        <span class="badge badge-muted">Regex</span>
                    {% endif %}
                </td>
                <td>
                    {% if c.reviewed %}
                        <span class="badge badge-reviewed">✅ Done</span>
                    {% else %}
                        <span class="badge badge-muted">⏳</span>
                    {% endif %}
                </td>
                <td style="font-size:0.75rem; color:var(--text-muted); font-family:'DM Mono',monospace">
                    {{ c.updated_at[:16] if c.updated_at else '—' }}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <!-- Pagination -->
    {% if total_pages > 1 %}
    <nav class="pagination">
        {% if page > 1 %}
            <a href="?page=1&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}">«</a>
            <a href="?page={{ page - 1 }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}">‹</a>
        {% else %}
            <span class="disabled">«</span>
            <span class="disabled">‹</span>
        {% endif %}
        
        {% for p in range(1, total_pages + 1) %}
            {% if p == page %}
                <span class="current">{{ p }}</span>
            {% elif p <= 3 or p >= total_pages - 2 or (p >= page - 2 and p <= page + 2) %}
                <a href="?page={{ p }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}">{{ p }}</a>
            {% elif p == 4 or p == total_pages - 3 %}
                <span style="color:var(--text-muted)">…</span>
            {% endif %}
        {% endfor %}
        
        {% if page < total_pages %}
            <a href="?page={{ page + 1 }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}">›</a>
            <a href="?page={{ total_pages }}&search={{ search }}&status={{ status_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}">»</a>
        {% else %}
            <span class="disabled">›</span>
            <span class="disabled">»</span>
        {% endif %}
    </nav>
    {% endif %}
    
    <div style="text-align:center; padding:8px; color:var(--text-muted); font-size:0.78rem;">
        Showing {{ ((page-1) * 25) + 1 }}–{{ [page * 25, total]|min }} of {{ total }} candidates
    </div>
    
    {% else %}
    <div style="text-align:center; padding:60px 20px; color:var(--text-secondary);">
        <div style="font-size:2rem; margin-bottom:12px;">🔍</div>
        <div style="font-size:1.1rem;">No candidates found</div>
        <div style="font-size:0.88rem; color:var(--text-muted); margin-top:6px;">
            {% if search %}Try adjusting your search filters{% else %}Run the extraction pipeline first to populate the database{% endif %}
        </div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

# ---- Detail (Side-by-Side Review) Template ----
DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review: {{ candidate.name or 'Candidate' }} #{{ candidate.candidate_id }} ✨</title>
    <style>""" + SHARED_CSS + """</style>
</head>
<body>

<!-- Toast notification container -->
<div class="toast-container" id="toastContainer"></div>

<!-- === COMPACT HEADER === -->
<header class="dashboard-header" style="padding:10px 24px;">
    <div class="header-inner">
        <div style="display:flex; align-items:center; gap:16px;">
            <a href="/" class="back-link">← Back to List</a>
            <h1 class="header-title" style="font-size:1.2rem;">
                {{ candidate.name or 'Unknown' }}
                <span style="font-size:0.82rem; opacity:0.6;">#{{ candidate.candidate_id }}</span>
            </h1>
            {% if candidate.reviewed %}
                <span class="badge badge-reviewed">✅ Reviewed</span>
            {% endif %}
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
            {% if candidate.reviewed %}
                <button class="btn btn-unreview" onclick="toggleReview(false)">🔄 Un-Review</button>
            {% else %}
                <button class="btn btn-review" onclick="toggleReview(true)">✅ Mark Reviewed</button>
            {% endif %}
        </div>
    </div>
</header>

<!-- === SIDE-BY-SIDE LAYOUT === -->
<div class="detail-container">
    
    <!-- LEFT PANEL: Raw Text -->
    <div class="raw-panel" id="rawPanel">
        <div class="panel-header">
            <span class="panel-title">📄 Raw Resume Text</span>
            <div class="raw-search-box">
                <input class="raw-search-input" type="text" 
                       id="rawSearchInput" placeholder="Find in text..."
                       autocomplete="off">
                <span class="search-count" id="searchCount"></span>
            </div>
        </div>
        <div class="panel-body" id="rawTextBody">
            <div class="raw-text-content line-numbers" id="rawTextContent">{% for line in raw_text.split('\n') %}<span class="line">{{ line }}</span>
{% endfor %}</div>
        </div>
    </div>
    
    <!-- RESIZE HANDLE -->
    <div class="resize-handle" id="resizeHandle"></div>
    
    <!-- RIGHT PANEL: Extracted Data -->
    <div class="extraction-panel" id="extractionPanel">
        <div class="panel-header">
            <span class="panel-title">📋 Extracted Data</span>
            <div style="display:flex; gap:8px; align-items:center;">
                <span class="badge {{ 'badge-ai' if candidate.ai_assisted else 'badge-muted' }}">
                    {{ candidate.extraction_method or 'Unknown' }}
                </span>
                <span class="badge {{ 'badge-success' if candidate.extraction_status in ['Complete','Success'] else 'badge-warning' }}">
                    {{ candidate.extraction_status or '?' }}
                </span>
            </div>
        </div>
        <div class="panel-body" id="fieldsContainer">
            {% for display_name, col_name in editable_fields.items() %}
            {% set value = candidate[col_name] if candidate[col_name] else '' %}
            {% set is_empty = not value or value|string|trim == '' %}
            {% set is_critical = col_name in critical_fields %}
            
            <div class="field-group {{ 'field-critical-empty' if is_empty and is_critical else ('field-empty' if is_empty else '') }}"
                 id="fieldGroup_{{ col_name }}">
                
                <!-- Field Label (click to toggle edit) -->
                <div class="field-label" onclick="toggleEdit('{{ col_name }}')">
                    <span>
                        {{ display_name }}
                        {% if is_empty and is_critical %}
                            <span style="color:var(--accent-danger); font-size:0.7rem;">● MISSING</span>
                        {% elif is_empty %}
                            <span style="color:var(--accent-warning); font-size:0.7rem;">● empty</span>
                        {% endif %}
                    </span>
                    <span style="font-size:0.72rem; color:var(--text-muted);">click to edit ✏️</span>
                </div>
                
                <!-- Display Value -->
                <div class="field-value" id="display_{{ col_name }}">
                    <div class="field-value-text {{ 'empty-field' if is_empty }}">{{ value if value else '(not extracted)' }}</div>
                </div>
                
                <!-- Edit Area (hidden by default) -->
                <div class="field-edit-area" id="edit_{{ col_name }}">
                    <textarea class="field-textarea" id="textarea_{{ col_name }}"
                              rows="{{ 1 if col_name in ['name','email','phone','date_of_birth','location','nationality'] else 6 }}"
                              data-field="{{ col_name }}"
                              data-display-name="{{ display_name }}">{{ value }}</textarea>
                    <div class="field-edit-actions">
                        <button class="btn btn-save" onclick="saveField('{{ col_name }}', '{{ display_name }}')">
                            💾 Save
                        </button>
                        <button class="btn btn-cancel" onclick="cancelEdit('{{ col_name }}')">
                            Cancel
                        </button>
                        <span id="status_{{ col_name }}" style="font-size:0.78rem; color:var(--text-muted);"></span>
                    </div>
                </div>
            </div>
            {% endfor %}
            
            <!-- Metadata section (read-only) -->
            <div style="margin-top:24px; padding:16px; background:var(--bg-secondary); border-radius:var(--radius-md); font-size:0.8rem; color:var(--text-muted);">
                <div style="font-weight:600; margin-bottom:8px; color:var(--text-secondary);">📊 Extraction Metadata</div>
                <div>Created: {{ candidate.created_at or '—' }}</div>
                <div>Updated: {{ candidate.updated_at or '—' }}</div>
                <div>Source: {{ candidate.filenames or '—' }}</div>
                <div>Language: {{ candidate.resume_language or '—' }}</div>
                <div>Raw text: {{ candidate.text_length or 0 }} chars</div>
                {% if candidate.review_notes %}
                <div style="margin-top:8px; padding:8px; background:var(--bg-card); border-radius:var(--radius-sm); white-space:pre-wrap;">
                    <strong>Review Notes:</strong>
                    {{ candidate.review_notes }}
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Extraction History Toggle -->
        <div class="history-toggle" onclick="toggleHistory()">
            📜 Extraction History ({{ history|length }} entries) — click to expand
        </div>
        <div class="history-panel" id="historyPanel">
            {% if history %}
                {% for h in history %}
                <div class="history-entry">
                    <span class="history-field">{{ h.field_name }}</span> 
                    via <span class="history-method">{{ h.extraction_method }}</span>
                    {% if h.was_successful %}
                        <span class="history-success">✓</span>
                    {% else %}
                        <span class="history-fail">✗</span>
                    {% endif %}
                    {% if h.was_overridden %}<span style="color:var(--accent-warning);">[overridden]</span>{% endif %}
                    <span style="float:right; opacity:0.5;">{{ h.timestamp[:16] if h.timestamp else '' }}</span>
                    {% if h.extracted_value %}
                    <div style="margin-left:20px; opacity:0.6; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:500px;">
                        → {{ h.extracted_value[:120] }}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            {% else %}
                <div style="color:var(--text-muted); text-align:center; padding:20px;">No extraction history found</div>
            {% endif %}
        </div>
    </div>
</div>

<!-- === JAVASCRIPT === -->
<script>
const CANDIDATE_ID = {{ candidate.candidate_id }};

// =============================================
// 🔔 TOAST NOTIFICATIONS
// Ephemeral pop-up messages — like a stage whisper from the wings! 🎭
// =============================================
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 3000);
}

// =============================================
// ✏️ FIELD EDITING — Toggle, Save, Cancel
// =============================================
function toggleEdit(colName) {
    const editArea = document.getElementById(`edit_${colName}`);
    const display = document.getElementById(`display_${colName}`);
    
    // Toggle visibility
    const isOpen = editArea.classList.contains('active');
    
    if (isOpen) {
        editArea.classList.remove('active');
        display.style.display = 'block';
    } else {
        editArea.classList.add('active');
        display.style.display = 'none';
        // Focus the textarea and move cursor to end
        const ta = document.getElementById(`textarea_${colName}`);
        ta.focus();
        ta.setSelectionRange(ta.value.length, ta.value.length);
    }
}

function cancelEdit(colName) {
    const editArea = document.getElementById(`edit_${colName}`);
    const display = document.getElementById(`display_${colName}`);
    editArea.classList.remove('active');
    display.style.display = 'block';
    
    // Reset textarea to original displayed value
    const displayText = display.querySelector('.field-value-text');
    const ta = document.getElementById(`textarea_${colName}`);
    const originalVal = displayText.classList.contains('empty-field') ? '' : displayText.textContent;
    ta.value = originalVal;
}

async function saveField(colName, displayName) {
    const ta = document.getElementById(`textarea_${colName}`);
    const statusEl = document.getElementById(`status_${colName}`);
    const newValue = ta.value;
    
    statusEl.textContent = 'Saving...';
    statusEl.style.color = 'var(--accent-info)';
    
    try {
        const resp = await fetch('/api/save-field', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                candidate_id: CANDIDATE_ID,
                field_name: colName,
                new_value: newValue,
                note: ''
            })
        });
        
        const data = await resp.json();
        
        if (data.success) {
            // Update the display value
            const display = document.getElementById(`display_${colName}`);
            const valueText = display.querySelector('.field-value-text');
            
            if (newValue.trim()) {
                valueText.textContent = newValue;
                valueText.classList.remove('empty-field');
            } else {
                valueText.textContent = '(not extracted)';
                valueText.classList.add('empty-field');
            }
            
            // Update field group styling
            const group = document.getElementById(`fieldGroup_${colName}`);
            group.classList.remove('field-empty', 'field-critical-empty');
            if (!newValue.trim()) {
                const criticalFields = {{ critical_fields|list|tojson }};
                if (criticalFields.includes(colName)) {
                    group.classList.add('field-critical-empty');
                } else {
                    group.classList.add('field-empty');
                }
            }
            
            // Close the edit area
            toggleEdit(colName);
            showToast(data.message, 'success');
            statusEl.textContent = '';
        } else {
            statusEl.textContent = data.message;
            statusEl.style.color = 'var(--accent-danger)';
            showToast(data.message, 'error');
        }
    } catch (err) {
        statusEl.textContent = 'Network error';
        statusEl.style.color = 'var(--accent-danger)';
        showToast('Network error — please try again', 'error');
        console.error('Save error:', err);
    }
}

// =============================================
// ✅ REVIEW TOGGLE
// =============================================
async function toggleReview(markAsReviewed) {
    const endpoint = markAsReviewed ? '/api/mark-reviewed' : '/api/unmark-reviewed';
    const notes = markAsReviewed 
        ? prompt('Optional review notes (or leave blank):') || ''
        : '';
    
    // User cancelled the prompt dialog
    if (markAsReviewed && notes === null) return;
    
    try {
        const resp = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ candidate_id: CANDIDATE_ID, notes })
        });
        
        const data = await resp.json();
        
        if (data.success) {
            showToast(data.message, 'success');
            // Reload to reflect the updated review status in the header
            setTimeout(() => location.reload(), 800);
        } else {
            showToast(data.message, 'error');
        }
    } catch (err) {
        showToast('Network error', 'error');
    }
}

// =============================================
// 🔍 RAW TEXT SEARCH & HIGHLIGHT
// Find-as-you-type within the raw resume text
// =============================================
let searchTimeout = null;

document.getElementById('rawSearchInput').addEventListener('input', function(e) {
    // Debounce: wait 250ms after user stops typing before searching
    // This prevents performance issues from highlighting on every keystroke
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => highlightRawText(e.target.value), 250);
});

function highlightRawText(query) {
    const container = document.getElementById('rawTextContent');
    const countEl = document.getElementById('searchCount');
    
    // Get original text from line spans
    const lines = container.querySelectorAll('.line');
    
    if (!query || query.length < 2) {
        // Reset — remove all highlights
        lines.forEach(line => {
            // Restore original text content (strip <mark> tags)
            line.innerHTML = escapeHtml(line.textContent);
        });
        countEl.textContent = '';
        return;
    }
    
    let totalMatches = 0;
    const escapedQuery = escapeRegex(query);
    const regex = new RegExp(`(${escapedQuery})`, 'gi');
    
    lines.forEach(line => {
        const originalText = line.textContent;
        const matches = originalText.match(regex);
        
        if (matches) {
            totalMatches += matches.length;
            // Replace matches with highlighted version
            line.innerHTML = escapeHtml(originalText).replace(
                new RegExp(`(${escapeRegex(escapeHtml(query))})`, 'gi'),
                '<mark>$1</mark>'
            );
        } else {
            line.innerHTML = escapeHtml(originalText);
        }
    });
    
    countEl.textContent = totalMatches > 0 ? `${totalMatches} found` : 'no matches';
    countEl.style.color = totalMatches > 0 ? 'var(--accent-success)' : 'var(--accent-danger)';
    
    // Auto-scroll to first match
    const firstMark = container.querySelector('mark');
    if (firstMark) {
        firstMark.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
}

// =============================================
// 📜 EXTRACTION HISTORY TOGGLE
// =============================================
function toggleHistory() {
    document.getElementById('historyPanel').classList.toggle('open');
}

// =============================================
// ↔️ RESIZABLE PANEL SPLITTER
// Drag the handle between raw text and extracted data to resize
// =============================================
(function initResize() {
    const handle = document.getElementById('resizeHandle');
    const rawPanel = document.getElementById('rawPanel');
    const extractionPanel = document.getElementById('extractionPanel');
    const container = document.querySelector('.detail-container');
    
    if (!handle || !rawPanel || !container) return;
    
    let isResizing = false;
    
    handle.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const containerRect = container.getBoundingClientRect();
        const newWidth = e.clientX - containerRect.left;
        const totalWidth = containerRect.width;
        
        // Clamp between 20% and 80% of total width
        const pct = Math.max(0.2, Math.min(0.8, newWidth / totalWidth));
        
        rawPanel.style.width = `${pct * 100}%`;
        extractionPanel.style.width = `${(1 - pct) * 100}%`;
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
})();

// =============================================
// ⌨️ KEYBOARD SHORTCUTS
// Ctrl+S to save the currently focused field
// =============================================
document.addEventListener('keydown', (e) => {
    // Ctrl+S / Cmd+S: Save focused field
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        const focused = document.activeElement;
        if (focused && focused.classList.contains('field-textarea')) {
            const colName = focused.dataset.field;
            const displayName = focused.dataset.displayName;
            if (colName && displayName) {
                saveField(colName, displayName);
            }
        }
    }
    
    // Escape: Close edit mode
    if (e.key === 'Escape') {
        const activeEdits = document.querySelectorAll('.field-edit-area.active');
        activeEdits.forEach(edit => {
            const colName = edit.id.replace('edit_', '');
            cancelEdit(colName);
        });
    }
});
</script>

</body>
</html>
"""


# =============================================================================
# 🚀 APP STARTUP
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ✨💅 FAIRY CODEMOTHER'S RESUME REVIEW DASHBOARD 💅✨")
    print("=" * 60)
    print(f"  📂 Database: {DATABASE_PATH}")
    print(f"  🌐 URL: http://localhost:{PORT}")
    print(f"  🔧 Debug: {DEBUG}")
    print("=" * 60)
    
    # Verify database exists before starting
    if not os.path.exists(DATABASE_PATH):
        print(f"\n  ❌ Database not found: {DATABASE_PATH}")
        print("  💡 Run your extraction pipeline first, or set RESUME_DB_PATH")
        print("     environment variable to point to your database.\n")
        exit(1)
    
    # Quick DB health check
    try:
        test_conn = sqlite3.connect(DATABASE_PATH)
        test_conn.row_factory = sqlite3.Row
        count = test_conn.execute("SELECT COUNT(*) as c FROM structured_extractions").fetchone()["c"]
        test_conn.close()
        print(f"  ✅ Database OK — {count} candidates found\n")
    except sqlite3.Error as e:
        print(f"\n  ❌ Database health check failed: {e}")
        print("  💡 Make sure the database has the expected schema.\n")
        exit(1)
    
    app.run(host=HOST, port=PORT, debug=DEBUG)