"""Annotation queue and editor: pages, save, pre-annotate, BIO/preview, training export, stats, edge cases."""
import datetime
import json
import os
import re
import sqlite3

from flask import flash, jsonify, redirect, render_template, request, url_for
from extraction.ner_schema import EdgeCaseHandler, ResumeClassifier, SpanAnnotation

from annotation.app import DATABASE_PATH, app, exporter, get_db, logger, pre_annotator, schema, storage, tagger, tokenizer
from annotation.queries import get_candidates_for_annotation, get_raw_text_for_candidate, get_structured_data


# =============================================================================
# 🌐 ROUTES
# =============================================================================

@app.route("/")
def index():
    """📋 Annotation queue — list of candidates to annotate."""
    page = request.args.get("page", 1, type=int)
    status = request.args.get("status", "all")
    data = get_candidates_for_annotation(page=page, status_filter=status)
    return render_template("index.html", **data, status_filter=status)


@app.route("/annotate/<int:candidate_id>")
def annotate(candidate_id: int):
    """🎨 Main annotation interface for a single candidate."""
    # Capture the queue page/status the user came from so "Back" returns there
    back_page = request.args.get("page", 1, type=int)
    back_status = request.args.get("status", "all")

    raw_text = get_raw_text_for_candidate(candidate_id)
    if not raw_text:
        flash(f"No raw text found for candidate {candidate_id}")
        return redirect(url_for("index"))

    structured = get_structured_data(candidate_id)
    doc_id = f"doc_{candidate_id}"

    # Load existing annotations or pre-annotate
    existing_doc = storage.load_document(doc_id)
    if existing_doc and existing_doc.annotations:
        annotations = existing_doc.annotations
        ann_status = existing_doc.status
        stored_function = existing_doc.metadata.get("function", "")
        stored_industry = existing_doc.metadata.get("industry", "")
        stored_hard_skills = existing_doc.metadata.get("hard_skills", [])
        stored_soft_skills = existing_doc.metadata.get("soft_skills", [])
        stored_tags = existing_doc.metadata.get("tags", [])
    else:
        # Auto pre-annotate for first visit
        annotations = pre_annotator.pre_annotate(raw_text, confidence_threshold=0.6)
        ann_status = "pending"
        # Run classification on the fly
        pred = ResumeClassifier.classify(raw_text, annotations)
        stored_function = pred["Function"]
        stored_industry = pred["Industry"]
        stored_hard_skills = pred.get("HardSkills", [])
        stored_soft_skills = pred.get("SoftSkills", [])
        stored_tags = pred.get("Tags", [])

        # 🆕 ML-enhanced Function prediction — overrides keyword result
        # if the ML model is confident. Graceful degradation if .pkl files
        # are missing — keyword classifier still works fine! 💅
        try:
            from ml.job_role_predictor import JobRolePredictor
            _role_predictor = JobRolePredictor()
            if _role_predictor.available and raw_text:
                ml_pred = _role_predictor.predict(raw_text)
                ml_function = ml_pred.get("function", "Others")
                ml_conf = ml_pred.get("confidence", 0.0)
                # Only override if ML is confident AND has a real category
                if ml_conf >= 0.15 and ml_function != "Others":
                    stored_function = ml_function
                    logger.info(
                        f"🤖 ML predicted role: {ml_pred['predicted_role']} "
                        f"({ml_conf:.1%}) → Function: {ml_function}"
                    )
        except ImportError:
            pass  # job_role_predictor.py not installed — that's fine!
        except Exception as e:
            logger.debug(f"ML prediction skipped: {e}")

    # Serialize annotations for JavaScript
    annotations_json = json.dumps([
        {
            "entity_type": a.entity_type,
            "char_start": a.char_start,
            "char_end": a.char_end,
            "text": a.text,
            "layer": a.layer,
            "confidence": a.confidence,
            "annotator": a.annotator,
        }
        for a in annotations
    ])

    # Serialize entity schema for JavaScript
    schema_json = json.dumps(schema.export_schema())

    # Build color map
    color_map_json = json.dumps(schema.get_color_map())

    return render_template(
        "annotate.html",
        candidate_id=candidate_id,
        raw_text=raw_text,
        structured=structured,
        annotations_json=annotations_json,
        schema_json=schema_json,
        color_map_json=color_map_json,
        ann_status=ann_status,
        doc_id=doc_id,
        stored_function=stored_function,
        stored_industry=stored_industry,
        stored_hard_skills=stored_hard_skills,
        stored_soft_skills=stored_soft_skills,
        stored_tags=stored_tags,
        back_page=back_page,
        back_status=back_status,
    )


@app.route('/api/save_annotations', methods=['POST'])
def save_annotations():
    try:
        data = request.json
        doc_id = data.get('doc_id')
        annotations_list = data.get('annotations', [])
        status = data.get('status', 'in_progress')
        func = data.get('function', '')
        ind = data.get('industry', '')
        hard_skills = data.get('hard_skills', [])
        soft_skills = data.get('soft_skills', [])
        tags = data.get('tags', [])
        # 👤 Annotator name — tracks WHO saved this document
        annotator_name = data.get('annotator_name', '')

        if not doc_id:
            return jsonify({"success": False, "error": "No doc_id"}), 400

        # ── 📊 Diagnostic logging: what data actually arrived? ────────
        logger.info(f"💾 SAVE REQUEST for {doc_id} | status={status} | "
                     f"annotator={annotator_name} | "
                     f"func={func} | ind={ind} | "
                     f"hard_skills={len(hard_skills)} | soft_skills={len(soft_skills)} | "
                     f"tags={len(tags)} | annotations={len(annotations_list)}")

        # Log a summary of what entity types + texts we received
        type_summary = {}
        for ann in annotations_list:
            etype = ann.get("entity_type", "?")
            text = ann.get("text", "")[:60]  # Truncate for readability
            cs = ann.get("char_start", "?")
            if etype not in type_summary:
                type_summary[etype] = []
            type_summary[etype].append(f"'{text}' (pos:{cs})")

        for etype, texts in type_summary.items():
            logger.info(f"   📋 {etype}: {texts}")

        # Save classification, status, and annotator name
        storage.update_document_metadata(doc_id, status, func, ind,
                                         hard_skills=hard_skills,
                                         soft_skills=soft_skills,
                                         tags=tags,
                                         annotator=annotator_name)

        # Save the actual entity spans
        storage.save_annotations(doc_id, annotations_list)

        # ── 💃 Update structured_extractions with corrected data ────────
        # Every time the annotator saves, we flatten the annotations back
        # into the structured_extractions table — updating the ORIGINAL
        # row with human-corrected values. No separate table needed!
        #
        # Think of it like the stylist fixing the mannequin's outfit
        # DIRECTLY instead of pinning notes on a clipboard! 👗✨
        storage.update_structured_extraction(
            doc_id=doc_id,
            annotations_list=annotations_list,
            function=func,
            industry=ind,
            status=status,
        )

        return jsonify({"success": True, "message": f"Saved as {status}"})
    except Exception as e:
        logger.error(f"❌ Save Error for {data.get('doc_id', '?')}: {e}",
                     exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pre-annotate/<int:candidate_id>")
def api_pre_annotate(candidate_id: int):
    """🤖 Run pre-annotation on a candidate's text."""
    threshold = request.args.get("threshold", 0.6, type=float)
    raw_text = get_raw_text_for_candidate(candidate_id)
    if not raw_text:
        return jsonify({"error": "No text found"}), 404

    annotations = pre_annotator.pre_annotate(raw_text, confidence_threshold=threshold)
    return jsonify({
        "annotations": [
            {
                "entity_type": a.entity_type,
                "char_start": a.char_start,
                "char_end": a.char_end,
                "text": a.text,
                "layer": a.layer,
                "confidence": a.confidence,
                "annotator": a.annotator,
            }
            for a in annotations
        ]
    })


@app.route("/api/bio-preview/<int:candidate_id>", methods=["POST"])
def api_bio_preview(candidate_id: int):
    """🏷️ Generate BIO tag preview from current annotations."""
    data = request.get_json()
    annotations_data = data.get("annotations", [])

    raw_text = get_raw_text_for_candidate(candidate_id)
    tokens = tokenizer.tokenize(raw_text)

    annotations = [
        SpanAnnotation(
            entity_type=a["entity_type"],
            char_start=a["char_start"],
            char_end=a["char_end"],
            text=a.get("text", ""),
            layer=a.get("layer", 0),
        )
        for a in annotations_data
    ]

    token_texts, bio_tags = tagger.tag_text(raw_text, annotations, tokens)

    # Return first 200 tokens for preview
    preview = [
        {"token": t, "tag": g}
        for t, g in zip(token_texts[:200], bio_tags[:200])
    ]

    tag_counts = {}
    for t in bio_tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1

    return jsonify({
        "preview": preview,
        "total_tokens": len(token_texts),
        "tag_counts": tag_counts,
    })


@app.route("/api/preview-before-save/<int:candidate_id>", methods=["POST"])
def api_preview_before_save(candidate_id: int):
    """
    🪞 SAVE PREVIEW — Validate annotations before committing to database!
    
    Think of this as the dress rehearsal before opening night, darling! 🎭
    We check EVERYTHING — missing required entities, exceeded limits,
    invalid patterns — so the annotator can fix issues BEFORE saving.
    
    Returns:
        JSON with validation results:
        - summary: entity counts grouped by category
        - warnings: list of issues found (level: critical/warning/info)
        - stats: overall annotation stats (total, human, auto, issues)
        - can_complete: boolean — whether "Mark Complete" is allowed
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    annotations_data = data.get("annotations", [])
    target_status = data.get("status", "in_progress")

    # ── Build entity type counts & collect annotated texts ────────
    entity_counts = {}        # {entity_type: count}
    entity_texts = {}         # {entity_type: [list of annotated texts]}
    human_count = 0
    auto_count = 0

    for idx, a in enumerate(annotations_data):
        etype = a.get("entity_type", "UNKNOWN")
        text = a.get("text", "")
        annotator = a.get("annotator", "human")

        entity_counts[etype] = entity_counts.get(etype, 0) + 1

        if etype not in entity_texts:
            entity_texts[etype] = []
        # Store FULL text + original annotation index for editable preview
        # No truncation! The frontend handles display — we need the real data
        # for editing to work correctly. Truncating here = data corruption! 💀
        entity_texts[etype].append({
            "text": text if text else "(empty)",
            "idx": idx
        })

        if annotator and annotator.startswith("auto"):
            auto_count += 1
        else:
            human_count += 1

    # ── Run validation against the EntitySchema ───────────────────
    warnings = []

    for etype_name, etype_def in schema.entities.items():
        count = entity_counts.get(etype_name, 0)

        # 1) Required entity missing entirely.
        #
        #    ⚠️  SOFT CRITICAL — shown in red so the annotator notices it,
        #    but it does NOT block "Mark Complete".  Some resumes genuinely
        #    omit fields (e.g. a resume with no email, no phone, etc.).
        #    The annotator sees the warning, acknowledges it, and can still
        #    save as completed.  Only a total absence of annotations (the
        #    hard-block below) prevents completion outright.
        if etype_def.is_required and count == 0:
            warnings.append({
                "level": "critical",
                "entity_type": etype_name,
                "label": etype_def.label,
                "message": (
                    f"'{etype_def.label}' is missing — most resumes include "
                    f"this field. If this resume genuinely doesn't have one, "
                    f"you can still mark it complete. ✅"
                ),
                "icon": "🚨"
            })

        # 2) Exceeds max_per_doc limit
        if etype_def.max_per_doc and count > etype_def.max_per_doc:
            warnings.append({
                "level": "warning",
                "entity_type": etype_name,
                "label": etype_def.label,
                "message": (
                    f"'{etype_def.label}' has {count} annotations "
                    f"but max expected is {etype_def.max_per_doc}. "
                    f"Check for duplicates or mis-labels."
                ),
                "icon": "⚠️"
            })

        # 3) Regex validation on annotated texts
        if etype_def.validation_regex and etype_name in entity_texts:
            try:
                pattern = re.compile(etype_def.validation_regex)
                for item in entity_texts[etype_name]:
                    text = item["text"] if isinstance(item, dict) else item
                    if not text or text == "(empty)":
                        continue
                    clean_text = text.strip()
                    if not pattern.match(clean_text):
                        warnings.append({
                            "level": "warning",
                            "entity_type": etype_name,
                            "label": etype_def.label,
                            "message": (
                                f"'{clean_text[:50]}' doesn't match expected "
                                f"pattern for {etype_def.label}. "
                                f"Verify it's labeled correctly."
                            ),
                            "icon": "🔍"
                        })
            except re.error:
                logger.warning(
                    f"Invalid validation_regex for {etype_name}: "
                    f"{etype_def.validation_regex}"
                )

    # ── Check for zero annotations edge case ──────────────────────
    if len(annotations_data) == 0:
        warnings.append({
            "level": "critical",
            "entity_type": None,
            "label": "No Annotations",
            "message": "No annotations to save! Add some labels first, darling! 💅",
            "icon": "🚨"
        })

    # ── Check for suspiciously short single-char annotations ──────
    for a in annotations_data:
        text = a.get("text", "").strip()
        etype = a.get("entity_type", "")
        # Skip META entities and single-char entities that are expected
        if etype.startswith("SECTION_") or etype == "BULLET_MARKER":
            continue
        if text and len(text) == 1 and etype not in ("GENDER",):
            etype_def = schema.entities.get(etype)
            label = etype_def.label if etype_def else etype
            warnings.append({
                "level": "info",
                "entity_type": etype,
                "label": label,
                "message": (
                    f"Single character '{text}' labeled as {label}. "
                    f"Might be too short — double-check the span boundary."
                ),
                "icon": "💡"
            })

    # ── Check for overlapping annotations on same layer ───────────
    layer_0 = sorted(
        [a for a in annotations_data if a.get("layer", 0) == 0],
        key=lambda x: x.get("char_start", 0)
    )
    for i in range(len(layer_0) - 1):
        curr = layer_0[i]
        nxt = layer_0[i + 1]
        if curr.get("char_end", 0) > nxt.get("char_start", 0):
            warnings.append({
                "level": "warning",
                "entity_type": curr.get("entity_type"),
                "label": "Overlap",
                "message": (
                    f"Overlapping spans: "
                    f"'{curr.get('text', '')[:30]}' ({curr.get('entity_type')}) "
                    f"overlaps with '{nxt.get('text', '')[:30]}' ({nxt.get('entity_type')}). "
                    f"Use different layers for nested entities."
                ),
                "icon": "🔀"
            })

    # ── Determine if "Mark Complete" is safe ──────────────────────
    critical_count = sum(1 for w in warnings if w["level"] == "critical")
    warning_count  = sum(1 for w in warnings if w["level"] == "warning")
    info_count     = sum(1 for w in warnings if w["level"] == "info")

    # ── Hard-block vs soft-critical separation ─────────────────────
    #
    #  HARD BLOCK  → zero annotations entirely.  There is nothing to
    #                save so completing makes no sense.
    #
    #  SOFT CRITICAL → required field missing (e.g. no PERSON_NAME).
    #                  Shown in red so the annotator notices, but some
    #                  resumes genuinely lack certain fields.  We warn
    #                  loudly and let the annotator make the call. 💅
    #
    hard_block_count = sum(
        1 for w in warnings
        if w["level"] == "critical" and w.get("entity_type") is None
        # entity_type is None only for the "No Annotations" check below
    )

    # "Mark Complete" is allowed as long as there is at least ONE annotation.
    # Soft criticals (missing required fields) show as red warnings but never
    # disable the button — the annotator decides whether to proceed.
    can_complete = (hard_block_count == 0) if target_status == "completed" else True

    # ── Build summary grouped by entity category ──────────────────
    summary_by_category = {}
    for etype_name, count in entity_counts.items():
        etype_def = schema.entities.get(etype_name)
        if etype_def:
            cat = etype_def.category.name
            color = etype_def.color
            label = etype_def.label
            max_doc = etype_def.max_per_doc
        else:
            cat = "UNKNOWN"
            color = "#888"
            label = etype_name
            max_doc = None

        if cat not in summary_by_category:
            summary_by_category[cat] = []

        summary_by_category[cat].append({
            "entity_type": etype_name,
            "label": label,
            "count": count,
            "color": color,
            "texts": entity_texts.get(etype_name, []),
            "max_per_doc": max_doc,
        })

    # ── Load current Function / Industry classification ─────────
    # Check ner_documents for a saved value first; if none, classify on the fly
    doc_id = f"doc_{candidate_id}"
    classification = {"function": "", "industry": "",
                      "hard_skills": [], "soft_skills": [], "tags": []}
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT function, industry, hard_skills, soft_skills, tags FROM ner_documents WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
        if row and (row["function"] or row["industry"]):
            classification["function"] = row["function"] or ""
            classification["industry"] = row["industry"] or ""
            # 🆕 Load skills & tags (JSON arrays, with graceful fallback)
            try:
                classification["hard_skills"] = json.loads(row["hard_skills"] or "[]")
            except (KeyError, json.JSONDecodeError):
                classification["hard_skills"] = []
            try:
                classification["soft_skills"] = json.loads(row["soft_skills"] or "[]")
            except (KeyError, json.JSONDecodeError):
                classification["soft_skills"] = []
            try:
                classification["tags"] = json.loads(row["tags"] or "[]")
            except (KeyError, json.JSONDecodeError):
                classification["tags"] = []
        else:
            # No saved classification — predict from raw text + current annotations
            raw_text = get_raw_text_for_candidate(candidate_id)
            if raw_text:
                span_anns = [
                    SpanAnnotation(
                        entity_type=a.get("entity_type", ""),
                        char_start=a.get("char_start", 0),
                        char_end=a.get("char_end", 0),
                        text=a.get("text", ""),
                        layer=a.get("layer", 0),
                    )
                    for a in annotations_data
                ]
                pred = ResumeClassifier.classify(raw_text, span_anns)
                classification["function"] = pred.get("Function", "")
                classification["industry"] = pred.get("Industry", "")
                classification["hard_skills"] = pred.get("HardSkills", [])
                classification["soft_skills"] = pred.get("SoftSkills", [])
                classification["tags"] = pred.get("Tags", [])
    except sqlite3.OperationalError:
        pass  # Table might not exist yet — degrade gracefully
    # ── Warn if classification is missing when marking complete ───
    if target_status == "completed":
        if not classification["function"] or classification["function"].lower() == "others":
            warnings.append({
                "level": "info",
                "entity_type": None,
                "label": "Function",
                "message": (
                    "Function classification is "
                    + ("'others'" if classification["function"].lower() == "others" else "empty")
                    + ". Consider setting it in the Classify tab."
                ),
                "icon": "🧠"
            })
        if not classification["industry"] or classification["industry"].lower() == "others":
            warnings.append({
                "level": "info",
                "entity_type": None,
                "label": "Industry",
                "message": (
                    "Industry classification is "
                    + ("'Others'" if classification["industry"].lower() == "others" else "empty")
                    + ". Consider setting it in the Classify tab."
                ),
                "icon": "🧠"
            })

    return jsonify({
        "summary": summary_by_category,
        "warnings": warnings,
        "stats": {
            "total_annotations": len(annotations_data),
            "human_count": human_count,
            "auto_count": auto_count,
            "entity_type_count": len(entity_counts),
            "critical_errors": critical_count,
            "warnings": warning_count,
            "info_hints": info_count,
        },
        "classification": classification,
        "can_complete": can_complete,
        "target_status": target_status,
        "entity_colors": schema.get_color_map(),
    })


@app.route("/api/export", methods=["POST"])
def api_export():
    """📤 Export annotated data to training format."""
    data = request.get_json()
    fmt = data.get("format", "conll")
    status_filter = data.get("status", "completed")

    # Load all completed documents
    conn = get_db()
    rows = conn.execute("""
        SELECT doc_id FROM ner_documents
        WHERE status = ?
    """, (status_filter,)).fetchall()
    documents = []
    for row in rows:
        doc = storage.load_document(row["doc_id"])
        if doc:
            documents.append(doc)

    if not documents:
        return jsonify({"error": "No completed annotations to export"}), 404

    os.makedirs("ner_exports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "conll":
        path = exporter.to_conll(documents, f"ner_exports/train_{timestamp}.conll")
    elif fmt == "spacy":
        path = exporter.to_spacy_json(documents, f"ner_exports/train_{timestamp}.json")
    elif fmt == "huggingface":
        path = exporter.to_huggingface(documents, f"ner_exports/train_{timestamp}.jsonl")
    elif fmt == "classification":
        # ── 🧠 Classification JSONL — Function & Industry training ────
        # This is a DIFFERENT beast from NER exports! Instead of
        # token-level BIO tags, it exports the WHOLE resume text
        # paired with document-level Function/Industry labels.
        # Think: NER = "what is each WORD?" vs Classification = "what
        # category is this whole RESUME?" 🏷️ vs 🧠
        output_path = f"ner_exports/classify_{timestamp}.jsonl"
        try:
            stats = exporter.to_classification_jsonl(
                db_path=DATABASE_PATH,
                output_path=output_path,
                min_text_length=50,
                include_features=True
            )
            return jsonify({
                "success": True,
                "path": output_path,
                "documents": stats.get("total_exported", 0),
                "format": fmt,
                "skipped": stats.get("skipped", 0),
                "function_classes": len(stats.get("function_distribution", {})),
                "industry_classes": len(stats.get("industry_distribution", {})),
            })
        except Exception as e:
            logger.error(f"❌ Classification export failed: {e}")
            return jsonify({
                "error": f"Classification export failed: {str(e)}. "
                         f"Make sure Function & Industry are set in the Classify tab!"
            }), 500
    else:
        path = exporter.to_custom_json(documents, f"ner_exports/train_{timestamp}.json")

    return jsonify({
        "success": True,
        "path": path,
        "documents": len(documents),
        "format": fmt,
    })


@app.route("/api/stats")
def api_stats():
    """
    📊 Get rich annotation statistics for the dashboard modal.

    Extends the base AnnotationStorage.get_annotation_stats() with:
      - Entity colour map (so bars match the annotation tool palette)
      - Queue summary (total/annotated/pending from structured_extractions)
      - Annotator leaderboard (top 5 most productive annotators)
      - Recent activity (last 7 days of annotation saves)
      - Human vs auto percentage breakdown
    """
    stats = storage.get_annotation_stats()

    # Attach entity colour map from the schema so the frontend can
    # colour each entity bar to match its annotation highlight colour
    stats["entity_colors"] = schema.get_color_map()

    conn = get_db()
    try:
        # Queue summary from structured_extractions
        # ⚡ Single query instead of 3 separate round-trips!
        # Like ordering your whole meal at once instead of
        # calling the waiter back three times! 🍽️✨
        stats_row = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM structured_extractions) as queue_total,
                (SELECT COUNT(*) FROM ner_documents WHERE status = 'completed') as queue_completed,
                (SELECT COUNT(*) FROM ner_documents WHERE status = 'in_progress') as queue_in_progress
        """).fetchone()
        stats["queue_total"] = stats_row["queue_total"] if stats_row else 0
        stats["queue_completed"] = stats_row["queue_completed"] if stats_row else 0
        stats["queue_in_progress"] = stats_row["queue_in_progress"] if stats_row else 0

        stats["queue_pending"] = max(
            0, stats["queue_total"]
            - stats["queue_completed"]
            - stats["queue_in_progress"]
        )

        # Annotator leaderboard: top 5 by human annotation count
        try:
            leader_rows = conn.execute("""
                SELECT nd.annotator, COUNT(na.id) as ann_count,
                       COUNT(DISTINCT na.doc_id) as doc_count
                FROM ner_annotations na
                JOIN ner_documents nd ON na.doc_id = nd.doc_id
                WHERE na.annotator = 'human'
                  AND nd.annotator != ''
                GROUP BY nd.annotator
                ORDER BY ann_count DESC
                LIMIT 5
            """).fetchall()
            stats["leaderboard"] = [
                {"annotator": r["annotator"],
                 "annotations": r["ann_count"],
                 "docs": r["doc_count"]}
                for r in leader_rows
            ]
        except Exception:
            stats["leaderboard"] = []

        # Recent daily activity: last 7 days
        try:
            activity_rows = conn.execute("""
                SELECT DATE(updated_at) as day, COUNT(*) as cnt
                FROM ner_annotations
                WHERE updated_at >= DATE('now', '-6 days')
                GROUP BY DATE(updated_at)
                ORDER BY day ASC
            """).fetchall()
            stats["recent_activity"] = [
                {"day": r["day"], "count": r["cnt"]}
                for r in activity_rows
            ]
        except Exception:
            stats["recent_activity"] = []

    except Exception:
        stats.setdefault("queue_total",       0)
        stats.setdefault("queue_completed",   0)
        stats.setdefault("queue_in_progress", 0)
        stats.setdefault("queue_pending",     0)
        stats.setdefault("leaderboard",       [])
        stats.setdefault("recent_activity",   [])
    return jsonify(stats)


@app.route("/api/edge-case-check", methods=["POST"])
def api_edge_case_check():
    """🧩 Check text for edge cases and suggest handling."""
    data = request.get_json()
    text = data.get("text", "")

    suggestions = []

    # Check for "Present" / "Current" in dates
    for keyword in EdgeCaseHandler.PRESENT_KEYWORDS:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        for m in pattern.finditer(text):
            suggestions.append({
                "type": "present_date",
                "char_start": m.start(),
                "char_end": m.end(),
                "text": m.group(),
                "advice": f"'{m.group()}' should be INCLUDED in the WORK_DATE span, not excluded.",
            })

    # Check for "Org, Location" patterns
    org_loc_pattern = re.compile(
        r'([A-Z][A-Za-z\s&.]+(?:Pte\.?\s*Ltd\.?|Sdn\.?\s*Bhd\.?|Inc\.?|Corp\.?)?)\s*,\s*([A-Z][A-Za-z\s]+)',
        re.MULTILINE
    )
    for m in org_loc_pattern.finditer(text):
        org_part, loc_part = EdgeCaseHandler.split_org_location(m.group())
        if loc_part:
            suggestions.append({
                "type": "org_location_split",
                "char_start": m.start(),
                "char_end": m.end(),
                "text": m.group(),
                "advice": f"Split into ORGANIZATION '{org_part}' and LOCATION '{loc_part}'.",
                "org": org_part,
                "loc": loc_part,
            })

    return jsonify({"suggestions": suggestions})
