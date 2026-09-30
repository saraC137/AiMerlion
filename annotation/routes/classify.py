"""Function/industry classification: auto-classify and manual save."""
import json
import sqlite3

from flask import jsonify, request
from extraction.ner_schema import ResumeClassifier, SpanAnnotation

from annotation.app import app, get_db, storage
from annotation.queries import get_raw_text_for_candidate


# =============================================================================
# 🎯 NEW ROUTES FOR FUNCTION/INDUSTRY CLASSIFICATION
# =============================================================================

@app.route("/api/classify/<int:candidate_id>", methods=["GET", "POST"])
def api_classify(candidate_id: int):
    """
    Predict Function, Industry, Hard Skills, Soft Skills, and Tags.

    🐛 BUG FIX: Previously a GET that loaded annotations from the DATABASE
    (the last SAVED state). If the user added new SKILL annotations but
    hadn't saved yet, re-run would classify from stale data — like reading
    yesterday's script instead of today's! 📜

    Now accepts POST with current in-memory annotations from the frontend,
    so re-run always classifies from the LIVE working state. Falls back
    to DB annotations for GET requests (backward compat).
    """
    raw_text = get_raw_text_for_candidate(candidate_id)
    if not raw_text:
        return jsonify({"error": "No text found"}), 404

    entities = []

    if request.method == "POST":
        # 🆕 Use CURRENT annotations from frontend (not stale DB data!)
        data = request.get_json() or {}
        annotations_list = data.get("annotations", [])
        entities = [
            SpanAnnotation(
                entity_type=a.get("entity_type", ""),
                char_start=a.get("char_start", 0),
                char_end=a.get("char_end", 0),
                text=a.get("text", ""),
                layer=a.get("layer", 0),
            )
            for a in annotations_list
            if a.get("entity_type") and a.get("text", "").strip()
        ]
    else:
        # Fallback: load from DB (backward compat for GET requests)
        doc_id = f"doc_{candidate_id}"
        doc = storage.load_document(doc_id)
        entities = doc.annotations if doc else []

    pred = ResumeClassifier.classify(raw_text, entities)
    return jsonify(pred)


@app.route("/api/save-classification", methods=["POST"])
def api_save_classification():
    """Save corrected Function/Industry/Skills/Tags for a candidate."""
    data = request.get_json()
    candidate_id = data.get("candidate_id")
    function = data.get("function", "")
    industry = data.get("industry", "")
    hard_skills = data.get("hard_skills", [])
    soft_skills = data.get("soft_skills", [])
    tags = data.get("tags", [])

    if not candidate_id:
        return jsonify({"error": "Missing candidate_id"}), 400

    doc_id = f"doc_{candidate_id}"
    conn = get_db()
    try:
        # Ensure hard_skills/soft_skills/tags columns exist (migration safety)
        for col in ['hard_skills', 'soft_skills', 'tags']:
            try:
                conn.execute(f"ALTER TABLE ner_documents ADD COLUMN {col} TEXT DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass  # column already exists

        # First, try to UPDATE an existing row
        cursor = conn.execute("""
            UPDATE ner_documents
            SET function = ?, industry = ?,
                hard_skills = ?, soft_skills = ?, tags = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = ?
        """, (function, industry,
              json.dumps(hard_skills, ensure_ascii=False),
              json.dumps(soft_skills, ensure_ascii=False),
              json.dumps(tags, ensure_ascii=False),
              doc_id))

        # If no row existed (first visit, annotations not saved yet),
        # INSERT a new row so the classification isn't lost! 💅
        # Like reserving a seat before the show starts — always be prepared! 🎭
        if cursor.rowcount == 0:
            conn.execute("""
                INSERT INTO ner_documents
                    (doc_id, candidate_id, status, function, industry,
                     hard_skills, soft_skills, tags)
                VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)
            """, (doc_id, candidate_id, function, industry,
                  json.dumps(hard_skills, ensure_ascii=False),
                  json.dumps(soft_skills, ensure_ascii=False),
                  json.dumps(tags, ensure_ascii=False)))

        conn.commit()
    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"success": True})
