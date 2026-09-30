"""Registered annotators: list/add/delete and the progress tracker."""
import sqlite3

from flask import jsonify, request

from annotation.app import app, get_db, logger


# =============================================================================
# 👥 ANNOTATOR MANAGEMENT CRUD
# Add, list, delete registered annotators.
# Like a VIP guest list — only the approved names get in! 💅👑
# =============================================================================
 
@app.route("/api/annotators", methods=["GET"])
def api_annotators_list():
    #👥 List all registered annotators.
    conn = get_db()
    try:
        # ── Ensure table exists (safe for first run) ──────────────
        conn.execute('''
            CREATE TABLE IF NOT EXISTS annotators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                role TEXT DEFAULT 'both',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
 
        rows = conn.execute('''
            SELECT a.id, a.name, a.role, a.created_at,
                   -- Count how many docs this person annotated as primary
                   (SELECT COUNT(*) FROM ner_documents nd
                    WHERE nd.annotator = a.name) as primary_docs,
                   -- Count how many IAA docs
                   (SELECT COUNT(DISTINCT doc_id) FROM iaa_annotations ia
                    WHERE ia.annotator_name = a.name) as iaa_docs
            FROM annotators a
            ORDER BY a.name COLLATE NOCASE
        ''').fetchall()
 
        annotators = []
        for r in rows:
            annotators.append({
                "id": r["id"],
                "name": r["name"],
                "role": r["role"],
                "created_at": r["created_at"],
                "primary_docs": r["primary_docs"],
                "iaa_docs": r["iaa_docs"],
            })
 
        return jsonify({"annotators": annotators})
    except Exception as e:
        logger.error(f"❌ Annotators list failed: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/annotators", methods=["POST"])
def api_annotators_add():
    #👥 Add a new annotator.
    data = request.get_json()
    name = (data.get("name") or "").strip()
    role = (data.get("role") or "both").strip()
 
    if not name or len(name) < 2:
        return jsonify({"error": "Name must be at least 2 characters!"}), 400
 
    if role not in ("primary", "iaa", "both"):
        role = "both"
 
    conn = get_db()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS annotators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                role TEXT DEFAULT 'both',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute(
            "INSERT INTO annotators (name, role) VALUES (?, ?)",
            (name, role)
        )
        conn.commit()
        logger.info(f"👥 Annotator added: {name} (role={role})")
        return jsonify({"success": True, "name": name, "role": role})
    except sqlite3.IntegrityError:
        return jsonify({"error": f"'{name}' already exists!"}), 409
    except Exception as e:
        logger.error(f"❌ Add annotator failed: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/annotators/<int:annotator_id>", methods=["DELETE"])
def api_annotators_delete(annotator_id: int):
    #👥 Delete an annotator (does NOT delete their annotations).
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT name FROM annotators WHERE id = ?", (annotator_id,)
        ).fetchone()
        if not row:
            return jsonify({"error": "Annotator not found"}), 404
 
        name = row["name"]
        conn.execute("DELETE FROM annotators WHERE id = ?", (annotator_id,))
        conn.commit()
        logger.info(f"👥 Annotator deleted: {name} (id={annotator_id})")
        return jsonify({"success": True, "deleted": name})
    except Exception as e:
        logger.error(f"❌ Delete annotator failed: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/annotator-tracker")
def api_annotator_tracker():
    """
    👥 Annotator Tracker — who annotated what, when, and how much.

    Returns per-annotator summary, per-document detail, and IAA sessions.
    Think of it as the production manager's clipboard! 🎭📋
    """
    conn = get_db()
    try:
        # ── Per-annotator summary ─────────────────────────────────────
        annotator_summary = []
        try:
            rows = conn.execute("""
                SELECT
                    annotator,
                    COUNT(*) as total_docs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
                    MIN(created_at) as first_activity,
                    MAX(updated_at) as last_activity
                FROM ner_documents
                WHERE annotator != '' AND annotator IS NOT NULL
                GROUP BY annotator
                ORDER BY total_docs DESC
            """).fetchall()
            annotator_summary = [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass

        # ── Span counts per annotator ─────────────────────────────────
        span_counts = {}
        try:
            span_rows = conn.execute("""
                SELECT nd.annotator, COUNT(na.id) as span_count
                FROM ner_annotations na
                JOIN ner_documents nd ON na.doc_id = nd.doc_id
                WHERE nd.annotator != '' AND nd.annotator IS NOT NULL
                GROUP BY nd.annotator
            """).fetchall()
            span_counts = {r["annotator"]: r["span_count"] for r in span_rows}
        except sqlite3.OperationalError:
            pass

        for item in annotator_summary:
            item["total_spans"] = span_counts.get(item["annotator"], 0)

        # ── Recent documents with annotator info ──────────────────────
        doc_detail = []
        try:
            doc_rows = conn.execute("""
                SELECT
                    nd.doc_id, nd.candidate_id, nd.annotator, nd.status,
                    nd.function, nd.industry, nd.created_at, nd.updated_at,
                    s.name as candidate_name,
                    COUNT(na.id) as span_count
                FROM ner_documents nd
                LEFT JOIN structured_extractions s ON nd.candidate_id = s.candidate_id
                LEFT JOIN ner_annotations na ON nd.doc_id = na.doc_id
                GROUP BY nd.doc_id
                ORDER BY nd.updated_at DESC
                LIMIT 50
            """).fetchall()
            doc_detail = [dict(r) for r in doc_rows]
        except sqlite3.OperationalError:
            pass

        # ── IAA sessions ──────────────────────────────────────────────
        iaa_sessions = []
        try:
            iaa_rows = conn.execute("""
                SELECT
                    ia.doc_id, ia.candidate_id,
                    ia.annotator_name as annotator_b,
                    nd.annotator as annotator_a,
                    COUNT(*) as span_count,
                    MAX(ia.created_at) as last_activity
                FROM iaa_annotations ia
                LEFT JOIN ner_documents nd ON ia.doc_id = nd.doc_id
                GROUP BY ia.doc_id, ia.annotator_name
                ORDER BY ia.created_at DESC
            """).fetchall()
            iaa_sessions = [dict(r) for r in iaa_rows]
        except sqlite3.OperationalError:
            pass

        return jsonify({
            "annotators": annotator_summary,
            "documents": doc_detail,
            "iaa_sessions": iaa_sessions,
            "total_annotators": len(annotator_summary),
            "total_docs": len(doc_detail),
        })
    except Exception as e:
        logger.error(f"❌ Annotator tracker failed: {e}")
        return jsonify({"annotators": [], "documents": [], "iaa_sessions": [], "error": str(e)})
