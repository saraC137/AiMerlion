"""Candidate profile export (JSON/CSV) and file download."""
import datetime
import os
import sqlite3

from flask import abort, jsonify, request
from typing import Dict, List

from annotation.app import app, exporter, get_db, storage


# =============================================================================
# 📋 CANDIDATE PROFILE EXPORT — JSON & CSV
# =============================================================================

def _load_structured_rows(candidate_ids: List[int]) -> Dict[int, Dict]:
    """
    Load structured_extractions rows for a list of candidate IDs.

    This is the PRIMARY data source for profile exports (JSON / CSV).
    structured_extractions holds the latest human-reviewed values written
    back on every annotation save, so it is always more accurate than
    re-deriving fields from raw ner_annotations spans.

    All columns that build_candidate_profile() can consume are fetched here:
      Scalar   — name, email, phone, location, summary, languages,
                 function, industry
      JSON     — skills_json, experience_json, education_json
      Raw text — projects  (Project Experience fallback)

    Args:
        candidate_ids: List of candidate IDs to look up.

    Returns:
        Dict of {candidate_id: row_dict} for fast lookup.
    """
    if not candidate_ids:
        return {}

    conn = get_db()
    try:
        placeholders = ",".join("?" * len(candidate_ids))
        rows = conn.execute(
            f"""SELECT
                    candidate_id,
                    name, email, phone, location,
                    summary, languages,
                    function, industry,
                    skills_json,
                    experience_json,
                    education_json,
                    projects
                FROM structured_extractions
                WHERE candidate_id IN ({placeholders})""",
            candidate_ids
        ).fetchall()
        return {row["candidate_id"]: dict(row) for row in rows}
    except sqlite3.OperationalError:
        # Table may not exist in all environments — degrade gracefully
        return {}
def _load_annotation_dates(doc_ids: List[str]) -> Dict[str, str]:
    """
    Load the creation timestamp for each document from ner_documents.

    Returns:
        Dict of {doc_id: created_at_string}
    """
    if not doc_ids:
        return {}

    conn = get_db()
    try:
        placeholders = ",".join("?" * len(doc_ids))
        rows = conn.execute(
            f"SELECT doc_id, created_at FROM ner_documents "
            f"WHERE doc_id IN ({placeholders})",
            doc_ids
        ).fetchall()
        return {row["doc_id"]: row["created_at"] for row in rows}
    except sqlite3.OperationalError:
        return {}
@app.route("/api/export-profiles", methods=["POST"])
def api_export_profiles():
    """
    📋 Export candidate profiles to JSON or CSV (talent database format).

    This is the NEW export endpoint — different from /api/export which
    produces NER training data. This one produces recruiter-friendly
    profiles in the structured talent database schema.

    Request body:
        {
            "format": "json" | "csv",     // Required
            "status": "completed" | "in_progress" | "all"  // Optional, default "completed"
        }

    Response:
        {
            "success": true,
            "path": "ner_exports/profiles_20260224_120000.json",
            "download_url": "/api/download/profiles_20260224_120000.json",
            "documents": 42,
            "format": "json"
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No request body provided"}), 400

    fmt = data.get("format", "json").lower()
    if fmt not in ("json", "csv"):
        return jsonify({"error": f"Unknown format '{fmt}'. Use 'json' or 'csv'."}), 400

    status_filter = data.get("status", "completed")

    # ── Load document IDs matching the status filter ─────────────────────
    conn = get_db()
    if status_filter == "all":
        rows = conn.execute(
            "SELECT doc_id, candidate_id FROM ner_documents"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT doc_id, candidate_id FROM ner_documents WHERE status = ?",
            (status_filter,)
        ).fetchall()
    if not rows:
        return jsonify({
            "error": f"No documents with status '{status_filter}' found."
        }), 404

    doc_ids        = [r["doc_id"]        for r in rows]
    candidate_ids  = [r["candidate_id"]  for r in rows]

    # ── Load full AnnotatedDocument objects ──────────────────────────────
    documents = []
    for doc_id in doc_ids:
        doc = storage.load_document(doc_id)
        if doc:
            documents.append(doc)

    if not documents:
        return jsonify({"error": "Could not load any documents."}), 404

    # ── Attach creation dates to each document's metadata ─────────────────
    # (AnnotatedDocument.metadata is a plain dict we can write to)
    date_map = _load_annotation_dates(doc_ids)
    for doc in documents:
        doc.metadata["created_at"] = date_map.get(doc.doc_id, "")

    # ── Load fallback structured data ─────────────────────────────────────
    structured_rows = _load_structured_rows(candidate_ids)

    # ── Write the export file ─────────────────────────────────────────────
    os.makedirs("ner_exports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        filename = f"profiles_{timestamp}.json"
        path = exporter.to_candidate_json(
            documents,
            f"ner_exports/{filename}",
            structured_rows=structured_rows
        )
    else:
        filename = f"profiles_{timestamp}.csv"
        path = exporter.to_candidate_csv(
            documents,
            f"ner_exports/{filename}",
            structured_rows=structured_rows
        )

    return jsonify({
        "success":      True,
        "path":         path,
        "download_url": f"/api/download/{filename}",
        "documents":    len(documents),
        "format":       fmt,
    })


@app.route("/api/download/<filename>")
def api_download(filename: str):
    """
    📥 Serve an export file for browser download.

    Security: only files inside the ner_exports/ directory are served.
    Path traversal attempts (e.g. '../../etc/passwd') are rejected.

    Args:
        filename: The filename inside ner_exports/ to serve.
    """
    from flask import send_file

    # ── Security: resolve path and confirm it stays inside ner_exports/ ──
    exports_dir = os.path.abspath("ner_exports")
    safe_path   = os.path.abspath(os.path.join(exports_dir, filename))

    # If the resolved path escapes the exports directory, reject it
    if not safe_path.startswith(exports_dir + os.sep):
        abort(403)

    if not os.path.exists(safe_path):
        abort(404)

    # Determine MIME type for the Content-Type header
    if filename.endswith(".json"):
        mimetype = "application/json"
    elif filename.endswith(".csv"):
        mimetype = "text/csv"
    else:
        mimetype = "application/octet-stream"

    return send_file(
        safe_path,
        mimetype=mimetype,
        as_attachment=True,   # Triggers browser "Save As" dialog
        download_name=filename
    )
