"""Inter-annotator agreement: save second annotations, compute metrics, dashboard."""
import sqlite3

from flask import jsonify, request

from annotation.app import app, get_db, iaa_engine, logger, schema
from annotation.queries import get_raw_text_for_candidate


# =============================================================================
# 📏 INTER-ANNOTATOR AGREEMENT (IAA) ROUTES
# =============================================================================

@app.route("/api/iaa/save", methods=["POST"])
def api_iaa_save():
    """
    📏 Save Annotator B's annotations for IAA comparison.

    When a second annotator works on a document that already has primary
    annotations, their labels go into iaa_annotations (not ner_annotations).
    Think of it as a parallel universe annotation — same resume, different
    pair of eyes! 👀✨

    Expected JSON payload:
        {
            "doc_id": "doc_42",
            "annotator_name": "Soraya",
            "annotations": [ { entity_type, char_start, char_end, text, layer }, ... ]
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    doc_id = data.get("doc_id")
    annotator_name = data.get("annotator_name", "").strip()
    annotations_list = data.get("annotations", [])

    # ── Validation: the three non-negotiables ─────────────────────────
    if not doc_id:
        return jsonify({"error": "Missing doc_id — who are we saving for?"}), 400
    if not annotator_name:
        return jsonify({
            "error": "Missing annotator_name — we need to know who's "
                     "holding the second pen! 🖊️"
        }), 400
    if not annotations_list:
        return jsonify({
            "error": "No annotations to save — the canvas is blank, darling! 🎨"
        }), 400

    # ── Guard: annotator B must NOT be the same as annotator A ────────
    # Otherwise we'd be comparing someone to themselves — that's just
    # narcissism, not inter-annotator agreement! 💅
    conn = get_db()
    try:
        primary_row = conn.execute(
            "SELECT annotator FROM ner_documents WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
        if primary_row and primary_row["annotator"] == annotator_name:
            return jsonify({
                "error": f"'{annotator_name}' is already the primary annotator "
                         f"for this document. IAA requires a DIFFERENT annotator! "
                         f"Like having two judges, not one judge twice! 👯"
            }), 409
    except sqlite3.OperationalError:
        pass  # Table may not exist yet — let save_iaa_annotations handle it
    try:
        iaa_engine.save_iaa_annotations(
            doc_id=doc_id,
            annotator_name=annotator_name,
            annotations_list=annotations_list
        )
        logger.info(f"📏 IAA annotations saved: {doc_id} by {annotator_name} "
                     f"({len(annotations_list)} spans)")
        return jsonify({
            "success": True,
            "message": f"IAA annotations saved for {annotator_name}",
            "spans": len(annotations_list),
        })
    except Exception as e:
        logger.error(f"❌ IAA save failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/iaa/compute/<doc_id>")
def api_iaa_compute(doc_id: str):
    """
    📏 Compute IAA metrics for a SINGLE document.

    Returns span-level agreement (exact + partial match F1) and
    token-level Cohen's Kappa between the primary annotator and
    annotator B.

    Like getting two fashion critics to rate the same outfit and
    seeing how much they agree! 👗📊
    """
    # Find who annotated this doc as annotator B
    conn = get_db()
    try:
        iaa_row = conn.execute(
            "SELECT DISTINCT annotator_name FROM iaa_annotations WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return jsonify({"error": "IAA tables not yet initialized"}), 404
    if not iaa_row:
        return jsonify({
            "status": "no_data",
            "message": "No IAA annotations found for this document."
        })

    annotator_b = iaa_row["annotator_name"]

    # Span-level agreement
    span_result = iaa_engine.compute_span_agreement(doc_id, annotator_b)

    # Token-level kappa (needs raw text)
    raw_text = get_raw_text_for_candidate(
        int(doc_id.replace("doc_", "")) if doc_id.startswith("doc_") else 0
    )
    kappa_result = None
    if raw_text:
        try:
            kappa_result = iaa_engine.compute_token_kappa(doc_id, annotator_b, raw_text)
        except Exception as e:
            logger.warning(f"⚠️ Kappa computation failed for {doc_id}: {e}")

    return jsonify({
        "status": "ok",
        "doc_id": doc_id,
        "annotator_b": annotator_b,
        "span_agreement": span_result,
        "token_kappa": kappa_result,
    })


@app.route("/api/iaa/dashboard")
def api_iaa_dashboard():
    """
    📏📊 IAA Dashboard — aggregate metrics across ALL dual-annotated docs.

    This is the BIG picture, darling! Shows:
      - Overall Cohen's Kappa (how much do our annotators agree?)
      - Exact & partial match F1 (are they finding the same spans?)
      - Per-entity heatmap data (which entity types cause the most drama?)
      - Per-document breakdown (which resumes need a recount?)

    Returns the full compute_all() result from InterAnnotatorAgreement,
    enriched with raw_texts for kappa computation.

    Optional query params:
        ?annotator=<name>   Filter to a specific Annotator B's work only.
                            When omitted, all annotators are aggregated.

    Performance note: This scans ALL iaa_annotations rows — for large
    datasets (500+ docs), consider adding pagination. For now, the
    typical IAA workflow involves 20-50 dual-annotated docs, so this
    is perfectly performant. 🏎️💨
    """
    try:
        # Optional: filter dashboard to a single annotator's work 🎯
        annotator_filter = request.args.get("annotator", "").strip() or None

        # Gather raw texts for kappa — only for the relevant docs
        iaa_docs = iaa_engine.get_iaa_docs()
        if annotator_filter:
            iaa_docs = [d for d in iaa_docs if d["annotator_name"] == annotator_filter]

        raw_texts = {}
        for doc_info in iaa_docs:
            doc_id = doc_info["doc_id"]
            cid = doc_info.get("candidate_id", 0)
            if cid:
                text = get_raw_text_for_candidate(cid)
                if text:
                    raw_texts[doc_id] = text

        result = iaa_engine.compute_all(
            raw_texts=raw_texts if raw_texts else None,
            annotator_filter=annotator_filter
        )

        # Enrich with entity color map for the heatmap visualization
        result["entity_colors"] = schema.get_color_map()

        # Add entity labels for display
        entity_labels = {}
        for name, etype in schema.entities.items():
            entity_labels[name] = etype.label
        result["entity_labels"] = entity_labels

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ IAA dashboard failed: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "IAA dashboard computation failed. Check server logs for details."
        }), 500


@app.route("/api/iaa/annotations/<doc_id>")
def api_iaa_annotations(doc_id: str):
    """
    📏 Load existing IAA annotations for a specific doc + annotator.

    Called by enterIAAMode() on page load so Annotator B sees their
    PREVIOUS work restored — not a blank slate every time they return.

    Query params:
        ?annotator=<name>   The annotator whose IAA annotations to load.

    Returns:
        {
          "found": true/false,
          "annotations": [ { entity_type, char_start, char_end, text, layer }, ... ],
          "span_count": N
        }
    """
    annotator = request.args.get("annotator", "").strip()
    if not annotator:
        return jsonify({"found": False, "annotations": [], "span_count": 0,
                        "error": "Missing ?annotator= query param"}), 400

    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT entity_type, char_start, char_end, text_content, layer
               FROM iaa_annotations
               WHERE doc_id = ? AND annotator_name = ?
               ORDER BY char_start""",
            (doc_id, annotator)
        ).fetchall()
    except sqlite3.OperationalError:
        # Table doesn't exist yet — totally fine, just means no saved IAA yet
        return jsonify({"found": False, "annotations": [], "span_count": 0})
    if not rows:
        return jsonify({"found": False, "annotations": [], "span_count": 0})

    # Reshape to match the frontend annotation object format
    annotations_out = [
        {
            "entity_type": r["entity_type"],
            "char_start":  r["char_start"],
            "char_end":    r["char_end"],
            "text":        r["text_content"] or "",
            "layer":       r["layer"] or 0,
        }
        for r in rows
        if r["char_start"] >= 0  # filter out placeholder negatives
    ]

    logger.info(f"📏 Loaded {len(annotations_out)} IAA annotations: {doc_id} by {annotator}")
    return jsonify({
        "found": True,
        "annotations": annotations_out,
        "span_count": len(annotations_out)
    })


@app.route("/api/iaa/annotators")
def api_iaa_annotators():
    """
    👥 Return all unique Annotator B names that have saved IAA annotations.

    Powers the annotator selector in the IAA dashboard so users can:
    - See WHO has done IAA work (with doc count + last annotated date)
    - Filter the entire dashboard to a specific annotator's work
    - Add a new annotator name to the system

    Returns:
        {
          "annotators": [
            { "annotator_name": "Soraya", "doc_count": 5, "last_annotated": "..." },
            ...
          ],
          "total": N
        }
    """
    try:
        annotators = iaa_engine.get_iaa_annotators()
        return jsonify({"annotators": annotators, "total": len(annotators)})
    except Exception as e:
        logger.error(f"❌ IAA annotators list failed: {e}", exc_info=True)
        return jsonify({"annotators": [], "total": 0, "error": str(e)})


@app.route("/api/iaa/docs")
def api_iaa_docs():
    """
    📏 List all documents that have IAA annotations.

    Quick lookup so the frontend can show which docs are ready
    for comparison — like checking the RSVP list! 📋✨
    """
    try:
        docs = iaa_engine.get_iaa_docs()
        return jsonify({"docs": docs, "total": len(docs)})
    except Exception as e:
        logger.error(f"❌ IAA docs list failed: {e}")
        return jsonify({"docs": [], "total": 0, "error": str(e)})
