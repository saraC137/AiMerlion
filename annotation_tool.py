"""
annotation_tool.py

💅✨ FAIRY CODEMOTHER'S NER ANNOTATION TOOL ✨💅

A web-based annotation interface for labeling resume entities!
Think of it as our own Label Studio / UBIAI — custom-built for
Singapore/Malaysia resume formats, and integrated directly
with our extraction database! 🎭📋

Features:
  🎨 Visual span-based annotation (click + drag to label text)
  🤖 Pre-annotation using dictionaries (auto-labels skills, cities, etc.)
  📑 Side-by-side: PDF-like text on left, entity labels on right
  🏷️ BIO tag preview for training data verification
  🎹 Keyboard shortcuts for fast annotation (1-9, H, Delete)
  🧬 Multi-layer support for nested entities
  📤 Export to CoNLL / spaCy / HuggingFace formats
  🗄️ SQLite storage alongside resume_extractions.db
  📊 Annotation progress dashboard
  🧠 Function/Industry classification tab with manual correction

Architecture:
  - Flask web server on port 5055 (separate from review dashboard)
  - Reads candidates from resume_extractions.db
  - Stores annotations in ner_annotations / ner_documents tables
  - Uses ner_schema.py for entity types, tokenizer, BIO tagger

Usage:
    pip install flask --break-system-packages
    python annotation_tool.py
    # Open http://localhost:5055

Dependencies:
    flask, ner_schema.py (in same directory)
"""

import sqlite3
import json
import os
import re
import datetime
import logging
from typing import Dict, List, Optional, Any

from flask import (
    Flask, render_template_string, request, jsonify,
    redirect, url_for, flash, abort
)
from markupsafe import escape

# Import NER schema components
from ner_schema import (
    EntitySchema, EntityCategory, ResumeTokenizer, BIOTagger,
    SpanAnnotation, AnnotatedDocument, NestedEntityLayer,
    EdgeCaseHandler, PreAnnotator, TrainingExporter,
    AnnotationStorage, ResumeClassifier,
    InterAnnotatorAgreement  # 📏 IAA computation engine
)

# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

DATABASE_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")
HOST = "0.0.0.0"
PORT = 5055
DEBUG = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - 🏷️ %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 🏗️ FLASK APP + NER COMPONENTS
# =============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fairy-ner-sparkle-2026")

# Initialize NER components
schema = EntitySchema()
tokenizer = ResumeTokenizer()
tagger = BIOTagger(schema)
pre_annotator = PreAnnotator(schema)
nested_handler = NestedEntityLayer(schema)
edge_handler = EdgeCaseHandler()
exporter = TrainingExporter(schema)
storage = AnnotationStorage(DATABASE_PATH)

iaa_engine = InterAnnotatorAgreement(DATABASE_PATH, schema)  # 📏 IAA metrics calculator


# =============================================================================
# 🛠️ HELPER FUNCTIONS
# =============================================================================

def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_candidates_for_annotation(
    page: int = 1,
    per_page: int = 30,
    status_filter: str = "all"
) -> Dict:
    """
    Fetch candidates available for annotation.
    Joins with ner_documents to show annotation status.
    """
    conn = get_db()
    try:
        # Count totals
        total = conn.execute(
            "SELECT COUNT(*) as c FROM structured_extractions"
        ).fetchone()["c"]

        # Count annotation statuses
        annotated = 0
        try:
            annotated = conn.execute(
                "SELECT COUNT(DISTINCT candidate_id) as c FROM ner_documents"
            ).fetchone()["c"]
        except sqlite3.OperationalError:
            pass  # Table might not exist yet

        # Build query with optional status filter
        offset = (page - 1) * per_page

        if status_filter == "annotated":
            rows = conn.execute("""
                SELECT s.candidate_id, s.name, s.email,
                       s.extraction_status, nd.status as ann_status,
                       nd.annotator as ann_by
                FROM structured_extractions s
                INNER JOIN ner_documents nd
                    ON 'doc_' || s.candidate_id = nd.doc_id
                ORDER BY s.candidate_id
                LIMIT ? OFFSET ?
            """, (per_page, offset)).fetchall()
        elif status_filter == "pending":
            rows = conn.execute("""
                SELECT s.candidate_id, s.name, s.email,
                       s.extraction_status, NULL as ann_status,
                       NULL as ann_by
                FROM structured_extractions s
                WHERE NOT EXISTS (
                    SELECT 1 FROM ner_documents nd
                    WHERE nd.doc_id = 'doc_' || s.candidate_id
                )
                ORDER BY s.candidate_id
                LIMIT ? OFFSET ?
            """, (per_page, offset)).fetchall()
        else:
            rows = conn.execute("""
                SELECT s.candidate_id, s.name, s.email,
                       s.extraction_status, nd.status as ann_status,
                       nd.annotator as ann_by
                FROM structured_extractions s
                LEFT JOIN ner_documents nd
                    ON 'doc_' || s.candidate_id = nd.doc_id
                ORDER BY s.candidate_id
                LIMIT ? OFFSET ?
            """, (per_page, offset)).fetchall()

        total_pages = max(1, (total + per_page - 1) // per_page)

        # ── Smart pagination window ───────────────────────────────────────
        # Instead of rendering all 126 page links (drama! 😱), we build a
        # compact window like:  « 1 … 11 [12] 13 … 126 »
        #
        # Rules:
        #   - Always show page 1 and the last page (anchors)
        #   - Show WINDOW pages either side of the current page
        #   - Insert None as a sentinel for the "…" ellipsis gap
        #   - Always show Prev / Next navigation arrows
        WINDOW = 2   # pages to show either side of current page
        pages_to_show = set()
        pages_to_show.add(1)
        pages_to_show.add(total_pages)
        for p in range(max(1, page - WINDOW), min(total_pages, page + WINDOW) + 1):
            pages_to_show.add(p)

        # Build an ordered list with None inserted wherever there is a gap > 1
        sorted_pages = sorted(pages_to_show)
        page_window = []   # Final list: integers for real pages, None for "…"
        for i, p in enumerate(sorted_pages):
            if i > 0 and p - sorted_pages[i - 1] > 1:
                page_window.append(None)   # Gap sentinel
            page_window.append(p)

        return {
            "candidates": [dict(r) for r in rows],
            "total": total,
            "annotated": annotated,
            "pending": total - annotated,
            "page": page,
            "total_pages": total_pages,
            "page_window": page_window,
        }
    finally:
        conn.close()


def get_raw_text_for_candidate(candidate_id: int) -> str:
    """Fetch raw resume text from the database."""
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT raw_text FROM raw_extractions
            WHERE candidate_id = ?
            ORDER BY extraction_timestamp DESC LIMIT 1
        """, (candidate_id,)).fetchone()
        return row["raw_text"] if row else ""
    finally:
        conn.close()


def get_structured_data(candidate_id: int) -> Dict:
    """Get structured extraction data for reference."""
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT * FROM structured_extractions
            WHERE candidate_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (candidate_id,)).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


# =============================================================================
# 🌐 ROUTES
# =============================================================================

@app.route("/")
def index():
    """📋 Annotation queue — list of candidates to annotate."""
    page = request.args.get("page", 1, type=int)
    status = request.args.get("status", "all")
    data = get_candidates_for_annotation(page=page, status_filter=status)
    return render_template_string(INDEX_TEMPLATE, **data, status_filter=status)


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

    return render_template_string(
        ANNOTATE_TEMPLATE,
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
    finally:
        conn.close()

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
    try:
        rows = conn.execute("""
            SELECT doc_id FROM ner_documents
            WHERE status = ?
        """, (status_filter,)).fetchall()
    finally:
        conn.close()

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
        total_row = conn.execute(
            "SELECT COUNT(*) as c FROM structured_extractions"
        ).fetchone()
        stats["queue_total"] = total_row["c"] if total_row else 0

        completed_row = conn.execute(
            "SELECT COUNT(DISTINCT candidate_id) as c FROM ner_documents "
            "WHERE status = 'completed'"
        ).fetchone()
        stats["queue_completed"] = completed_row["c"] if completed_row else 0

        in_progress_row = conn.execute(
            "SELECT COUNT(DISTINCT candidate_id) as c FROM ner_documents "
            "WHERE status = 'in_progress'"
        ).fetchone()
        stats["queue_in_progress"] = in_progress_row["c"] if in_progress_row else 0

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
    finally:
        conn.close()

    return jsonify(stats)


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
    finally:
        conn.close()


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
    finally:
        conn.close()


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
    finally:
        conn.close()


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
    try:
        if status_filter == "all":
            rows = conn.execute(
                "SELECT doc_id, candidate_id FROM ner_documents"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT doc_id, candidate_id FROM ner_documents WHERE status = ?",
                (status_filter,)
            ).fetchall()
    finally:
        conn.close()

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
    finally:
        conn.close()
    return jsonify({"success": True})

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
    finally:
        conn.close()

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
    finally:
        conn.close()

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

    Performance note: This scans ALL iaa_annotations rows — for large
    datasets (500+ docs), consider adding pagination. For now, the
    typical IAA workflow involves 20-50 dual-annotated docs, so this
    is perfectly performant. 🏎️💨
    """
    try:
        # Gather raw texts for all IAA-annotated docs (needed for kappa)
        iaa_docs = iaa_engine.get_iaa_docs()
        raw_texts = {}
        for doc_info in iaa_docs:
            doc_id = doc_info["doc_id"]
            cid = doc_info.get("candidate_id", 0)
            if cid:
                text = get_raw_text_for_candidate(cid)
                if text:
                    raw_texts[doc_id] = text

        result = iaa_engine.compute_all(raw_texts=raw_texts if raw_texts else None)

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
    finally:
        conn.close()

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


# =============================================================================
# 🎨 SHARED CSS
# =============================================================================

SHARED_CSS = """
:root {
    --bg-deep: #080b12;
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-elevated: #1c2333;
    --bg-hover: #252d3a;
    --border: #2a3140;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #545d68;
    --accent-cyan: #7ee8fa;
    --accent-purple: #eeb8ff;
    --accent-green: #56d364;
    --accent-orange: #f0883e;
    --accent-red: #da3633;
    --accent-yellow: #e3b341;
    --accent-pink: #f778ba;
    --accent-blue: #a371f7;
    --font-body: 'Outfit', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --font-display: 'Syne', system-ui, sans-serif;
    --radius: 8px;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 15px;
    line-height: 1.6;
}
a { color: var(--accent-cyan); text-decoration: none; }
a:hover { text-decoration: underline; }

.header {
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border);
    padding: 16px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.header h1 {
    font-family: var(--font-display);
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.header h1 span { color: var(--accent-cyan); }

.btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-size: 0.82rem;
    font-family: var(--font-body);
    cursor: pointer;
    transition: all 0.15s;
}
.btn:hover { background: var(--bg-hover); border-color: var(--accent-cyan); }
.btn-primary { background: var(--accent-cyan); color: var(--bg-deep); font-weight: 600; border-color: var(--accent-cyan); }
.btn-primary:hover { background: #5dd8ed; }
.btn-success { background: var(--accent-green); color: var(--bg-deep); border-color: var(--accent-green); }
.btn-danger { background: var(--accent-red); color: #fff; border-color: var(--accent-red); }
.btn-sm { padding: 4px 10px; font-size: 0.75rem; }

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-pending { background: rgba(227,179,65,0.15); color: var(--accent-yellow); }
.badge-progress { background: rgba(126,232,250,0.15); color: var(--accent-cyan); }
.badge-complete { background: rgba(86,211,100,0.15); color: var(--accent-green); }
.badge-auto { background: rgba(238,184,255,0.15); color: var(--accent-purple); }

.card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
}
.stat-row {
    display: flex;
    gap: 16px;
    padding: 12px 28px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border);
}
.stat-item {
    text-align: center;
    padding: 8px 20px;
}
.stat-number {
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 700;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

table {
    width: 100%;
    border-collapse: collapse;
}
th {
    text-align: left;
    padding: 10px 14px;
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
}
td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(42,49,64,0.5);
    font-size: 0.88rem;
}
tr:hover { background: var(--bg-hover); }

.pagination {
    display: flex;
    gap: 4px;
    justify-content: center;
    align-items: center;
    padding: 20px;
    flex-wrap: wrap;   /* Won't overflow even on narrow screens */
}
/* Standard numbered page link */
.pagination a {
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-secondary);
    font-size: 0.82rem;
    min-width: 36px;
    text-align: center;
    transition: all 0.15s;
}
.pagination a:hover, .pagination a.active {
    background: var(--accent-cyan);
    color: var(--bg-deep);
    border-color: var(--accent-cyan);
    text-decoration: none;
}
/* Prev / Next arrow buttons — slightly wider */
.pagination a.nav-arrow {
    padding: 6px 14px;
    color: var(--accent-cyan);
    font-weight: 600;
}
.pagination a.nav-arrow.disabled {
    opacity: 0.3;
    pointer-events: none;   /* Can't click when on first/last page */
}
/* Ellipsis gap — not a link, just visual spacing */
.pagination .ellipsis {
    padding: 6px 4px;
    color: var(--text-muted);
    font-size: 0.82rem;
    user-select: none;
}
/* Page info label on the right */
.pagination-info {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    padding-bottom: 8px;
}

/* =============================================================
 * 📊 STATS MODAL — Beautiful dashboard overlay
 * ============================================================= */

.stats-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    z-index: 600;
    align-items: flex-start;
    justify-content: center;
    padding: 32px 20px;
    overflow-y: auto;
    backdrop-filter: blur(4px);
}
.stats-overlay.open { display: flex; }

.stats-modal {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 16px;
    width: 100%;
    max-width: 900px;
    box-shadow: 0 24px 80px rgba(0,0,0,0.6);
    overflow: hidden;
    animation: modalSlideIn 0.25s ease;
}
@keyframes modalSlideIn {
    from { transform: translateY(-20px); opacity: 0; }
    to   { transform: translateY(0);     opacity: 1; }
}

/* Modal header with gradient accent */
.stats-header {
    background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    border-bottom: 1px solid var(--border);
    padding: 22px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
/* Decorative glow behind the header title */
.stats-header::before {
    content: '';
    position: absolute;
    top: -40px; left: -20px;
    width: 240px; height: 120px;
    background: radial-gradient(circle, rgba(126,232,250,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.stats-header h2 {
    font-family: var(--font-display);
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    position: relative;
}
.stats-header h2 span { color: var(--accent-cyan); }
.stats-header .subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 2px;
    position: relative;
}
.stats-close {
    width: 34px; height: 34px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-elevated);
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1rem;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
    position: relative;
    flex-shrink: 0;
}
.stats-close:hover { border-color: var(--accent-red); color: var(--accent-red); }

.stats-body {
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}

/* ── KPI Cards row ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}
.kpi-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 16px;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s, border-color 0.15s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: var(--kpi-color, var(--accent-cyan));
}
/* Subtle background glow using CSS custom property set per-card */
.kpi-card::before {
    content: '';
    position: absolute;
    bottom: -20px; right: -20px;
    width: 80px; height: 80px;
    background: radial-gradient(circle, color-mix(in srgb, var(--kpi-color, var(--accent-cyan)) 20%, transparent) 0%, transparent 70%);
    pointer-events: none;
}
.kpi-icon {
    font-size: 1.4rem;
    margin-bottom: 8px;
    display: block;
}
.kpi-value {
    font-family: var(--font-mono);
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
    color: var(--kpi-color, var(--accent-cyan));
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}
.kpi-sub {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Two-column grid for the charts section ── */
.stats-charts-row {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 16px;
    align-items: start;
}

/* ── Progress donut ring ── */
.donut-wrap {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}
.donut-wrap h4 {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
    text-align: center;
}
.donut-container {
    position: relative;
    width: 130px; height: 130px;
}
.donut-svg { transform: rotate(-90deg); }
.donut-track {
    fill: none;
    stroke: var(--bg-elevated);
    stroke-width: 14;
}
.donut-fill {
    fill: none;
    stroke: var(--accent-cyan);
    stroke-width: 14;
    stroke-linecap: round;
    /* stroke-dasharray and stroke-dashoffset set by JS */
    transition: stroke-dashoffset 1s ease;
}
.donut-fill.in-progress { stroke: var(--accent-yellow); }
.donut-center {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.donut-pct {
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent-cyan);
    line-height: 1;
}
.donut-pct-label {
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
/* Donut legend items */
.donut-legend {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.donut-legend-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.72rem;
}
.donut-legend-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.donut-legend-label {
    color: var(--text-secondary);
    flex: 1;
    margin-left: 6px;
}
.donut-legend-val {
    font-family: var(--font-mono);
    color: var(--text-primary);
    font-weight: 600;
}

/* ── Entity distribution bars ── */
.entity-chart-wrap {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
}
.entity-chart-wrap h4 {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
    margin-bottom: 14px;
}
.entity-bar-row {
    display: grid;
    grid-template-columns: 140px 1fr 52px;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.entity-bar-label {
    font-size: 0.76rem;
    color: var(--text-secondary);
    text-align: right;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.entity-bar-track {
    height: 8px;
    background: var(--bg-elevated);
    border-radius: 4px;
    overflow: hidden;
}
.entity-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
    min-width: 2px;
}
.entity-bar-count {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    text-align: right;
}

/* ── Bottom row: Leaderboard + Activity ── */
.stats-bottom-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.stats-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
}
.stats-panel h4 {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Leaderboard rows */
.leader-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(42,49,64,0.4);
    font-size: 0.82rem;
}
.leader-row:last-child { border-bottom: none; }
.leader-rank {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    width: 20px;
    text-align: center;
    flex-shrink: 0;
}
.leader-name {
    flex: 1;
    color: var(--text-primary);
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.leader-stat {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    text-align: right;
    flex-shrink: 0;
}
.leader-stat strong {
    color: var(--accent-cyan);
    font-size: 0.82rem;
}

/* Activity sparkline bars */
.activity-chart {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    height: 80px;
    padding-top: 8px;
}
.activity-bar-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    height: 100%;
    justify-content: flex-end;
}
.activity-bar {
    width: 100%;
    background: var(--accent-purple);
    border-radius: 3px 3px 0 0;
    min-height: 2px;
    transition: height 0.6s ease;
    opacity: 0.8;
}
.activity-bar:hover { opacity: 1; }
.activity-day {
    font-size: 0.58rem;
    color: var(--text-muted);
    text-align: center;
    white-space: nowrap;
}

/* Human vs Auto ratio bar */
.ratio-bar-wrap {
    margin-top: 8px;
}
.ratio-bar-track {
    height: 12px;
    background: var(--bg-elevated);
    border-radius: 6px;
    overflow: hidden;
    display: flex;
}
.ratio-bar-human {
    height: 100%;
    background: var(--accent-green);
    transition: width 0.8s ease;
}
.ratio-bar-auto {
    height: 100%;
    background: var(--accent-purple);
    flex: 1;
}
.ratio-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 0.7rem;
}
.ratio-label { display: flex; align-items: center; gap: 4px; color: var(--text-secondary); }
.ratio-dot { width: 8px; height: 8px; border-radius: 50%; }

/* Loading state */
.stats-loading {
    text-align: center;
    padding: 60px;
    color: var(--text-muted);
    font-size: 0.9rem;
}
.stats-loading .spin {
    display: inline-block;
    animation: spin 1s linear infinite;
    font-size: 1.5rem;
    margin-bottom: 12px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Error state */
.stats-error {
    text-align: center;
    padding: 40px;
    color: var(--accent-red);
    font-size: 0.88rem;
}

/* =============================================================
 * 📦 EXPORT MODAL — FULL GLAM EDITION ✨
 * Bigger, cleaner, luxurious, no more mess!
 * ============================================================= */

.export-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.85);
    z-index: 700;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    backdrop-filter: blur(8px);
}
.export-overlay.open { display: flex; }

.export-modal {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 20px;
    width: 100%;
    max-width: 560px;
    box-shadow: 0 30px 90px rgba(0,0,0,0.8);
    overflow: hidden;
    animation: modalSlideIn 0.35s cubic-bezier(0.34,1.56,0.64,1);
}

.export-modal-header {
    background: linear-gradient(135deg, #161b22, #1c2333);
    padding: 22px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
}
.export-modal-header h3 {
    font-family: var(--font-display);
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    color: var(--accent-cyan);
    margin: 0;
}

.export-close {
    width: 38px; height: 38px;
    border: 1px solid var(--border);
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border-radius: 10px;
    font-size: 1.1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}
.export-close:hover {
    background: var(--accent-red);
    color: white;
    border-color: var(--accent-red);
    transform: rotate(90deg);
}

.export-modal-body {
    padding: 32px 28px;
    background: var(--bg-primary);
}

.export-field {
    margin-bottom: 24px;
}
.export-field label {
    display: block;
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
    margin-bottom: 8px;
}
.export-field select {
    width: 100%;
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text-primary);
    padding: 12px 16px;
    font-size: 0.95rem;
    outline: none;
    transition: all 0.25s;
}
.export-field select:focus {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 0 4px rgba(126,232,250,0.15);
}

.export-hint {
    font-size: 0.78rem;
    line-height: 1.5;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 14px 18px;
    border-radius: 10px;
    border-left: 5px solid var(--accent-cyan);
}

.export-modal-footer {
    padding: 20px 28px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
    justify-content: flex-end;
}

"""


# =============================================================================
# 📄 INDEX TEMPLATE — Annotation Queue
# =============================================================================

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NER Annotation Tool ✨</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>""" + SHARED_CSS + """</style>
</head>
<body>

<!-- ── Toast notification container ────────────────────────────────────
     All toast messages are injected here by showToast().
     Lives outside all panels so it's always on top. 💅 -->
<div id="toast-container" role="status" aria-live="polite"></div>

<div class="header">
    <h1>🏷️ NER <span>Annotation Tool</span></h1>
    <div style="display:flex; gap:8px;">
        <button class="btn" onclick="openStats()" id="statsBtn" title="View annotation statistics dashboard">
            📊 Stats
        </button>
        <button class="btn" onclick="openAnnotatorTracker()" title="See who annotated what"
                style="border-color:var(--accent-purple); color:var(--accent-purple);">
            👥 Annotators
        </button>
        <!-- Profile exports: recruiter-friendly talent DB format -->
        <button class="btn btn-success" onclick="exportProfiles('csv')" title="Export as Excel-compatible CSV (talent database format)">
            📊 Export CSV
        </button>
        <button class="btn btn-primary" onclick="exportProfiles('json')" title="Export as JSON (talent database format)">
            📋 Export JSON
        </button>
        <!-- NER training data exports (existing formats) -->
        <button class="btn" onclick="exportTrainingData()" title="Export NER training data (CoNLL/spaCy/HuggingFace)">
            🤖 Export Training Data
        </button>
        <button class="btn" id="iaaToggleBtn" onclick="toggleIAAGlobal()"
                title="Toggle IAA mode — second annotator activates this before reviewing"
                style="border-color:var(--accent-pink); color:var(--accent-pink);">
            📏 IAA Mode
        </button>
    </div>
</div>

<div class="stat-row">
    <div class="stat-item">
        <div class="stat-number" style="color:var(--accent-cyan)">{{ total }}</div>
        <div class="stat-label">Total Candidates</div>
    </div>
    <div class="stat-item">
        <div class="stat-number" style="color:var(--accent-green)">{{ annotated }}</div>
        <div class="stat-label">Annotated</div>
    </div>
    <div class="stat-item">
        <div class="stat-number" style="color:var(--accent-yellow)">{{ pending }}</div>
        <div class="stat-label">Pending</div>
    </div>
    <div class="stat-item">
        <div class="stat-number" style="color:var(--accent-purple)">
            {{ ((annotated / total * 100) if total > 0 else 0)|int }}%
        </div>
        <div class="stat-label">Progress</div>
    </div>
</div>

<!-- 📏 IAA Mode banner on queue page -->
<div id="iaaQueueBanner" style="display:none; background:linear-gradient(90deg, rgba(247,120,186,0.15), rgba(238,184,255,0.08), rgba(247,120,186,0.15));
     border-bottom:2px solid rgba(247,120,186,0.5); padding:12px 28px;
     font-size:0.82rem; color:#f778ba; align-items:center; gap:12px;">
    <span style="background:#f778ba; color:#0e1117; font-size:0.65rem; font-weight:800;
                 padding:3px 10px; border-radius:4px; text-transform:uppercase; letter-spacing:0.08em;">
        IAA MODE ACTIVE
    </span>
    <span>Annotating as <strong id="iaaQueueName">—</strong>.
          Every candidate will open with a clean slate for independent annotation.</span>
    <button class="btn" onclick="toggleIAAGlobal()"
            style="margin-left:auto; font-size:0.7rem; padding:5px 14px;
                   border-color:#f778ba; color:#f778ba;">
        ✕ Deactivate
    </button>
</div>

<!-- Filter tabs -->
<div style="display:flex; gap:8px; padding:16px 28px; background:var(--bg-secondary); border-bottom:1px solid var(--border);">
    <a href="/?status=all" class="btn {{ 'btn-primary' if status_filter == 'all' else '' }}">All</a>
    <a href="/?status=pending" class="btn {{ 'btn-primary' if status_filter == 'pending' else '' }}">⏳ Pending</a>
    <a href="/?status=annotated" class="btn {{ 'btn-primary' if status_filter == 'annotated' else '' }}">✅ Annotated</a>
</div>

<!-- Candidate table -->
<div style="padding: 0 28px 28px;">
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Extraction</th>
                <th>Annotation</th>
                <th>Annotator</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
            {% for c in candidates %}
            <tr id="row-{{ c.candidate_id }}">
                <td style="font-family:var(--font-mono); color:var(--text-muted)">{{ c.candidate_id }}</td>
                <td><strong>{{ c.name or '—' }}</strong></td>
                <td style="color:var(--text-secondary)">{{ c.email or '—' }}</td>
                <td>
                    {% if c.extraction_status == 'Complete' %}
                        <span class="badge badge-complete">✅ Complete</span>
                    {% elif c.extraction_status == 'Partial' %}
                        <span class="badge badge-pending">🟡 Partial</span>
                    {% else %}
                        <span class="badge" style="background:rgba(218,54,51,0.15); color:var(--accent-red)">{{ c.extraction_status or '?' }}</span>
                    {% endif %}
                </td>
                <td>
                    {% if c.ann_status == 'completed' %}
                        <span class="badge badge-complete">✅ Done</span>
                    {% elif c.ann_status == 'in_progress' %}
                        <span class="badge badge-progress">🔵 In Progress</span>
                    {% elif c.ann_status %}
                        <span class="badge badge-pending">⏳ {{ c.ann_status }}</span>
                    {% else %}
                        <span class="badge badge-pending">⏳ Not Started</span>
                    {% endif %}
                </td>
                <td>
                    {% if c.ann_by %}
                        <span style="font-size:0.75rem; color:var(--accent-purple);">👤 {{ c.ann_by }}</span>
                    {% else %}
                        <span style="font-size:0.72rem; color:var(--text-muted);">—</span>
                    {% endif %}
                </td>
                <td>
                    <a href="/annotate/{{ c.candidate_id }}?page={{ page }}&status={{ status_filter }}" class="btn btn-sm btn-primary">
                        🏷️ Annotate
                    </a>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    {% if total_pages > 1 %}
    <div class="pagination-info">
        Page {{ page }} of {{ total_pages }} &nbsp;·&nbsp; {{ total }} total candidates
    </div>
    <div class="pagination">

        {# ── « Prev arrow ── #}
        <a href="/?page={{ page - 1 }}&status={{ status_filter }}"
           class="nav-arrow {{ 'disabled' if page <= 1 else '' }}"
           title="Previous page">« Prev</a>

        {# ── Windowed page numbers with … ellipsis ── #}
        {% for p in page_window %}
            {% if p is none %}
                {# Gap between page clusters → render an ellipsis #}
                <span class="ellipsis">…</span>
            {% else %}
                <a href="/?page={{ p }}&status={{ status_filter }}"
                   class="{{ 'active' if p == page else '' }}">{{ p }}</a>
            {% endif %}
        {% endfor %}

        {# ── Next » arrow ── #}
        <a href="/?page={{ page + 1 }}&status={{ status_filter }}"
           class="nav-arrow {{ 'disabled' if page >= total_pages else '' }}"
           title="Next page">Next »</a>

    </div>
    {% endif %}
</div>

<!-- ================================================================
     📊 STATS MODAL — Beautiful annotation dashboard
     ================================================================ -->
<div class="stats-overlay" id="statsOverlay" onclick="closeStatsOnBackdrop(event)">
  <div class="stats-modal" role="dialog" aria-modal="true" aria-labelledby="statsTitle">

    <div class="stats-header">
      <div>
        <h2 id="statsTitle">📊 Annotation <span>Dashboard</span></h2>
        <div class="subtitle" id="statsSubtitle">Loading…</div>
      </div>
      <button class="stats-close" onclick="closeStats()" title="Close">✕</button>
    </div>

    <div class="stats-body" id="statsBody">
      <!-- Content injected by renderStats() -->
      <div class="stats-loading">
        <div class="spin">⚙️</div>
        <div>Crunching the numbers, darling…</div>
      </div>
    </div>

  </div>
</div>

<!-- ================================================================
     📦 EXPORT MODAL — Now serving PURE GLAM ✨
     ================================================================ -->
    <div class="export-overlay" id="exportOverlay" onclick="if(event.target===this)closeExportModal()">
    <div class="export-modal">

        <!-- Header -->
        <div class="export-modal-header">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.6rem;">📦</span>
            <h3 id="exportModalTitle" style="margin:0; font-size:1.25rem; color:var(--accent-cyan);">Export Options</h3>
        </div>
        <button class="export-close" onclick="closeExportModal()" title="Cancel">✕</button>
        </div>

        <!-- Body -->
        <div class="export-modal-body" id="exportModalBody">
        <!-- JS will fill this -->
        </div>

        <!-- Footer -->
        <div class="export-modal-footer">
        <button class="btn" onclick="closeExportModal()">Cancel</button>
        <button class="btn btn-primary" id="exportConfirmBtn" onclick="confirmExport()">
            📤 Export Now
        </button>
        </div>

    </div>
    </div>

<!-- ================================================================
     👥 ANNOTATOR TRACKER MODAL
     ================================================================ -->
<div class="stats-overlay" id="trackerOverlay" onclick="closeTrackerOnBackdrop(event)">
  <div class="stats-modal" role="dialog" aria-modal="true" style="max-width:900px;">
    <div class="stats-header">
      <div>
        <h2>👥 Annotator <span>Tracker</span></h2>
        <div class="subtitle" id="trackerSubtitle">Loading…</div>
      </div>
      <button class="stats-close" onclick="closeTracker()" title="Close">✕</button>
    </div>
    <div class="stats-body" id="trackerBody" style="max-height:75vh; overflow-y:auto;">
      <div class="stats-loading">
        <div class="spin">⚙️</div>
        <div>Loading annotator data…</div>
      </div>
    </div>
  </div>
</div>

<script>
// ── Scroll to & highlight the row the user came from ──
(function() {
    const hash = window.location.hash;
    if (hash && hash.startsWith('#row-')) {
        // Remove hash immediately to prevent browser's native anchor jump
        history.replaceState(null, '', window.location.pathname + window.location.search);
        // Wait for layout, then scroll manually with header offset
        requestAnimationFrame(() => {
            const row = document.querySelector(hash);
            if (!row) return;
            const header = document.querySelector('.header');
            const headerH = header ? header.offsetHeight : 0;
            const rowTop = row.getBoundingClientRect().top + window.scrollY;
            window.scrollTo({ top: rowTop - headerH - 16, behavior: 'smooth' });
            row.style.transition = 'background 0.4s';
            row.style.background = 'rgba(99, 102, 241, 0.15)';
            setTimeout(() => { row.style.background = ''; }, 2000);
        });
    }
})();

// ================================================================
// 📦 EXPORT OPTIONS MODAL — Replaces prompt() calls
// ================================================================

/** Tracks what kind of export the user is performing */
let exportMode = null;  // 'csv', 'json', or 'training'

/**
 * openExportModal(mode)
 * 
 * Opens the export options modal with the right fields for each mode.
 * No more prompt() — this is a PROPER UI, darling! 💅✨
 * 
 * prompt() is like showing up to a gala in sweatpants —
 * it WORKS, but the bouncers (browsers) might not let you in! 🚫
 * 
 * @param {string} mode - 'csv', 'json', or 'training'
 */
function openExportModal(mode) {
    exportMode = mode;
    const overlay = document.getElementById('exportOverlay');
    const title = document.getElementById('exportModalTitle');
    const body = document.getElementById('exportModalBody');
    const confirmBtn = document.getElementById('exportConfirmBtn');

    // Reset confirm button
    confirmBtn.disabled = false;
    confirmBtn.textContent = '📤 Export';

    if (mode === 'csv' || mode === 'json') {
        // Profile exports — need status filter
        title.textContent = mode === 'csv' ? '📊 Export CSV Profiles' : '📋 Export JSON Profiles';
        body.innerHTML = `
            <div class="export-field">
                <label>Which records to export?</label>
                <select id="exportStatusSelect">
                    <option value="completed" selected>✅ Completed — fully annotated only</option>
                    <option value="in_progress">🔵 In Progress — include drafts too</option>
                    <option value="all">📋 All — everything in the queue</option>
                </select>
            </div>
            <div class="export-hint">
                💡 <strong>${mode.toUpperCase()}</strong> format — 
                ${mode === 'csv' 
                    ? 'Excel-compatible spreadsheet. Open with Data → Text to Columns for nested fields.'
                    : 'Structured JSON array. Ready for CRM/ATS import.'}
            </div>
        `;
    } else if (mode === 'training') {
        // NER training data export — need format picker
        title.textContent = '🤖 Export Training Data';
        body.innerHTML = `
            <div class="export-field">
                <label>Export format</label>
                <select id="exportFormatSelect" onchange="updateExportHint()">
                    <optgroup label="🏷️ NER Training (token-level)">
                        <option value="conll" selected>📄 CoNLL-2003 — most compatible</option>
                        <option value="spacy">🐍 spaCy v3 JSON</option>
                        <option value="huggingface">🤗 HuggingFace JSONL</option>
                        <option value="custom">📋 Full metadata JSON</option>
                    </optgroup>
                    <optgroup label="🧠 Classification Training (document-level)">
                        <option value="classification">🧠 Classification JSONL — Function &amp; Industry</option>
                    </optgroup>
                </select>
            </div>
            <div class="export-field">
                <label>Status filter</label>
                <select id="exportTrainStatusSelect">
                    <option value="completed" selected>✅ Completed only (recommended)</option>
                    <option value="in_progress">🔵 Include in-progress</option>
                    <option value="all">📋 All documents</option>
                </select>
            </div>
            <div class="export-hint" id="exportHintText">
                💡 NER model training formats — teaches the model to identify entities word-by-word.
            </div>
        `;
    }

    overlay.classList.add('open');

    // ── Auto-scroll the confirm button into view ────────────────────────
    // After a short paint delay, smoothly scroll the confirm button into
    // the visible area so the user never has to hunt for it. 💅
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            const btn = document.getElementById('exportConfirmBtn');
            if (btn) {
                btn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    });

    // Keyboard: Escape to close, Enter to confirm
    document.addEventListener('keydown', exportKeyHandler);
}

/**
 * confirmExport()
 * 
 * Reads the selected options from the modal and triggers the actual export.
 * The grand finale — lights, camera, DOWNLOAD! 🎬💾
 */
function confirmExport() {
    const confirmBtn = document.getElementById('exportConfirmBtn');
    confirmBtn.disabled = true;
    confirmBtn.textContent = '⏳ Exporting...';

    if (exportMode === 'csv' || exportMode === 'json') {
        // ── Profile export (CSV or JSON) ─────────────────────────────
        const status = document.getElementById('exportStatusSelect').value;

        fetch('/api/export-profiles', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ format: exportMode, status: status })
        })
        .then(r => r.json())
        .then(d => {
            closeExportModal();
            if (d.error) {
                showToast(`❌ Export failed: ${d.error}`, 'error');
                return;
            }
            // Trigger browser file download via hidden <a> tag
            const a = document.createElement('a');
            a.href = d.download_url;
            a.download = d.download_url.split('/').pop();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            showToast(
                `✅ <strong>${d.documents} profile${d.documents !== 1 ? 's' : ''} exported!</strong> Your ${d.format.toUpperCase()} is downloading — go slay that talent pipeline! 💼🎉`,
                'success',
                5000
            );
        })
        .catch(err => {
            closeExportModal();
            showToast(`❌ Network error: ${err}`, 'error');
        });

    } else if (exportMode === 'training') {
        // ── NER training data export ─────────────────────────────────
        const fmt = document.getElementById('exportFormatSelect').value;
        const status = document.getElementById('exportTrainStatusSelect').value;

        fetch('/api/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ format: fmt, status: status })
        })
        .then(r => r.json())
        .then(d => {
            closeExportModal();
            if (d.error) {
                showToast(`❌ ${d.error}`, 'error');
            } else {
                showToast(`🤖 Exported ${d.documents} docs → ${d.path} (${d.format})`, 'success');
            }
        })
        .catch(err => {
            closeExportModal();
            showToast(`❌ Network error: ${err}`, 'error');
        });
    }
}

/**
 * closeExportModal()
 * Hides the export modal and cleans up.
 */
function closeExportModal() {
    document.getElementById('exportOverlay').classList.remove('open');
    document.removeEventListener('keydown', exportKeyHandler);
    exportMode = null;
    // Reset confirm button state
    const btn = document.getElementById('exportConfirmBtn');
    if (btn) { btn.disabled = false; btn.textContent = '📤 Export'; }
}

/**
 * exportKeyHandler(event)
 * Escape → close, Enter → confirm
 */
function exportKeyHandler(event) {
    if (event.key === 'Escape') {
        event.preventDefault();
        closeExportModal();
    }
    if (event.key === 'Enter') {
        event.preventDefault();
        confirmExport();
    }
}

/**
 * exportProfiles(format)
 *
 * Wrapper called by toolbar buttons — now opens the modal
 * instead of using the cursed prompt()! 💅
 *
 * @param {string} format - "json" or "csv"
 */
function exportProfiles(format) {
    openExportModal(format);
}

/**
 * exportTrainingData()
 *
 * Wrapper called by toolbar button — opens modal for NER training export.
 */
function exportTrainingData() {
    openExportModal('training');
}

/**
 * updateExportHint()
 *
 * Dynamically updates the hint text in the export modal based on
 * the selected format. Because NER and Classification are VERY
 * different beasts, and we want the user to know which is which! 🧠🏷️
 */
function updateExportHint() {
    const fmt = document.getElementById('exportFormatSelect').value;
    const hint = document.getElementById('exportHintText');
    if (!hint) return;

    if (fmt === 'classification') {
        hint.innerHTML = '🧠 <strong>Classification training</strong> — exports whole resume text + Function/Industry labels. ' +
            'Teaches the model to predict which <em>department</em> and <em>industry</em> a candidate belongs to. ' +
            'Requires Function &amp; Industry set in the Classify tab.';
    } else {
        hint.innerHTML = '💡 <strong>NER training formats</strong> — exports token-level BIO tags. ' +
            'Teaches the model to identify entities (Name, Phone, Skills, etc.) word-by-word.';
    }
}


// ================================================================
// 📊 STATS MODAL — Open / Close / Render
// ================================================================

/**
 * openStats()
 * 
 * Fetches fresh stats from /api/stats and opens the modal.
 * Now with proper error handling and visual feedback! 💅
 * 
 * Think of it as opening the backstage door — you might find 
 * a perfectly organized dressing room, or you might find chaos.
 * Either way, we handle it gracefully! 🎭
 */
function openStats() {
    const overlay = document.getElementById('statsOverlay');
    const statsBtn = document.getElementById('statsBtn');

    // Visual feedback on button — so user knows it's working
    if (statsBtn) {
        statsBtn.textContent = '⏳ Loading...';
        statsBtn.disabled = true;
    }

    // Open the overlay immediately with loading state
    overlay.classList.add('open');
    document.addEventListener('keydown', closeStatsOnEscape);

    // Reset to loading state
    document.getElementById('statsBody').innerHTML = `
        <div class="stats-loading">
            <div class="spin">⚙️</div>
            <div>Crunching the numbers, darling…</div>
        </div>`;
    document.getElementById('statsSubtitle').textContent = 'Loading…';

    // Fetch fresh data every time the modal opens
    fetch('/api/stats')
        .then(r => {
            if (!r.ok) throw new Error(`Server returned ${r.status} ${r.statusText}`);
            return r.json();
        })
        .then(d => {
            renderStats(d);
            // Restore button state
            if (statsBtn) { statsBtn.textContent = '📊 Stats'; statsBtn.disabled = false; }
        })
        .catch(err => {
            console.error('❌ Stats fetch failed:', err);
            document.getElementById('statsBody').innerHTML = `
                <div class="stats-error">
                    ❌ Failed to load stats.<br>
                    <small style="color:var(--text-muted);">${escapeHtml(String(err))}</small>
                    <br><br>
                    <div style="font-size:0.75rem; color:var(--text-muted);">
                        💡 This usually means the database hasn't been set up yet,
                        or the server isn't running. Try saving at least one annotation first!
                    </div>
                </div>`;
            // Restore button state
            if (statsBtn) { statsBtn.textContent = '📊 Stats'; statsBtn.disabled = false; }
        });
}

function closeStats() {
    document.getElementById('statsOverlay').classList.remove('open');
    document.removeEventListener('keydown', closeStatsOnEscape);
    // Restore stats button text in case it was left in loading state
    const statsBtn = document.getElementById('statsBtn');
    if (statsBtn) { statsBtn.textContent = '📊 Stats'; statsBtn.disabled = false; }
}
function closeStatsOnBackdrop(e) {
    if (e.target === document.getElementById('statsOverlay')) closeStats();
}
function closeStatsOnEscape(e) {
    if (e.key === 'Escape') closeStats();
}

/**
 * renderStats(d)
 *
 * Takes the rich stats JSON from /api/stats and builds the full
 * modal content — KPI cards, donut ring, entity bars, leaderboard,
 * activity sparkline, and human/auto ratio bar.
 *
 * Everything is pure HTML/CSS — no charting library required! ✨
 *
 * @param {Object} d - Stats data from /api/stats
 */
function renderStats(d) {
    // ── Derived values ────────────────────────────────────────────────
    const total       = d.queue_total       || 0;
    const completed   = d.queue_completed   || 0;
    const inProgress  = d.queue_in_progress || 0;
    const pending     = d.queue_pending     || 0;
    const totalAnns   = d.total_annotations || 0;
    const humanAnns   = d.human_annotations || 0;
    const autoAnns    = d.auto_annotations  || 0;

    const completePct    = total > 0 ? (completed / total * 100)   : 0;
    const inProgressPct  = total > 0 ? (inProgress / total * 100)  : 0;
    const humanPct       = totalAnns > 0 ? (humanAnns / totalAnns * 100) : 0;

    // Update subtitle
    document.getElementById('statsSubtitle').textContent =
        `${total.toLocaleString()} candidates · ${totalAnns.toLocaleString()} annotations`;

    // ── KPI Cards ─────────────────────────────────────────────────────
    const kpiHTML = `
    <div class="kpi-grid">
        <div class="kpi-card" style="--kpi-color: var(--accent-cyan)">
            <span class="kpi-icon">👥</span>
            <div class="kpi-value">${total.toLocaleString()}</div>
            <div class="kpi-label">Total Candidates</div>
        </div>
        <div class="kpi-card" style="--kpi-color: var(--accent-green)">
            <span class="kpi-icon">✅</span>
            <div class="kpi-value">${completed.toLocaleString()}</div>
            <div class="kpi-label">Completed</div>
            <div class="kpi-sub">${completePct.toFixed(1)}% of queue</div>
        </div>
        <div class="kpi-card" style="--kpi-color: var(--accent-yellow)">
            <span class="kpi-icon">🔵</span>
            <div class="kpi-value">${inProgress.toLocaleString()}</div>
            <div class="kpi-label">In Progress</div>
            <div class="kpi-sub">${inProgressPct.toFixed(1)}% of queue</div>
        </div>
        <div class="kpi-card" style="--kpi-color: var(--accent-purple)">
            <span class="kpi-icon">🏷️</span>
            <div class="kpi-value">${totalAnns.toLocaleString()}</div>
            <div class="kpi-label">Total Annotations</div>
            <div class="kpi-sub">${humanAnns.toLocaleString()} human · ${autoAnns.toLocaleString()} auto</div>
        </div>
    </div>`;

    // ── Donut ring — completion progress ─────────────────────────────
    // SVG donut: circumference = 2π × r  (r = 54)
    const R = 54;
    const CIRC = 2 * Math.PI * R;
    const completeDash  = (completePct  / 100) * CIRC;
    const inProgDash    = (inProgressPct / 100) * CIRC;
    // Completed arc offset = 0 (starts at top after rotate(-90deg))
    // InProgress arc starts after the completed section
    const inProgOffset  = CIRC - completeDash;

    const donutHTML = `
    <div class="donut-wrap">
        <h4>📋 Queue Progress</h4>
        <div class="donut-container">
            <svg class="donut-svg" width="130" height="130" viewBox="0 0 130 130">
                <!-- Track (background ring) -->
                <circle class="donut-track" cx="65" cy="65" r="${R}"/>
                <!-- In-Progress arc (yellow, behind completed) -->
                <circle class="donut-fill in-progress" cx="65" cy="65" r="${R}"
                    stroke-dasharray="${inProgDash} ${CIRC}"
                    stroke-dashoffset="${-completeDash}"/>
                <!-- Completed arc (cyan, on top) -->
                <circle class="donut-fill" cx="65" cy="65" r="${R}"
                    stroke-dasharray="${completeDash} ${CIRC}"
                    stroke-dashoffset="0"/>
            </svg>
            <div class="donut-center">
                <div class="donut-pct">${completePct.toFixed(0)}%</div>
                <div class="donut-pct-label">Done</div>
            </div>
        </div>
        <div class="donut-legend">
            <div class="donut-legend-item">
                <div class="donut-legend-dot" style="background:var(--accent-green)"></div>
                <span class="donut-legend-label">Completed</span>
                <span class="donut-legend-val">${completed.toLocaleString()}</span>
            </div>
            <div class="donut-legend-item">
                <div class="donut-legend-dot" style="background:var(--accent-yellow)"></div>
                <span class="donut-legend-label">In Progress</span>
                <span class="donut-legend-val">${inProgress.toLocaleString()}</span>
            </div>
            <div class="donut-legend-item">
                <div class="donut-legend-dot" style="background:var(--bg-elevated); border:1px solid var(--border)"></div>
                <span class="donut-legend-label">Pending</span>
                <span class="donut-legend-val">${pending.toLocaleString()}</span>
            </div>
        </div>
    </div>`;

    // ── Entity distribution bars ──────────────────────────────────────
    const entityCounts = d.entity_counts || {};
    const entityColors = d.entity_colors || {};
    const maxCount = Math.max(1, ...Object.values(entityCounts));

    // Sort by count descending, take top 15
    const topEntities = Object.entries(entityCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);

    const entityBarsHTML = topEntities.map(([name, count]) => {
        const pct   = (count / maxCount * 100).toFixed(1);
        const color = entityColors[name] || '#7ee8fa';
        // Convert ENTITY_NAME → Entity Name for display
        const label = name.replace(/_/g, ' ')
                          .toLowerCase()
                          .replace(/\\b\\w/g, c => c.toUpperCase());
        return `
        <div class="entity-bar-row">
            <div class="entity-bar-label" title="${name}">${label}</div>
            <div class="entity-bar-track">
                <div class="entity-bar-fill"
                     style="width:${pct}%; background:${color};"
                     title="${count.toLocaleString()} annotations"></div>
            </div>
            <div class="entity-bar-count">${count.toLocaleString()}</div>
        </div>`;
    }).join('');

    const entityChartHTML = `
    <div class="entity-chart-wrap">
        <h4>🏷️ Entity Distribution <span style="color:var(--text-muted); font-weight:400">(top 15 by count)</span></h4>
        ${topEntities.length > 0 ? entityBarsHTML :
          '<div style="color:var(--text-muted); font-size:0.82rem; padding:20px 0; text-align:center;">No annotations yet — get labelling! 💅</div>'}
    </div>`;

    // ── Leaderboard ───────────────────────────────────────────────────
    const medals = ['🥇','🥈','🥉','4️⃣','5️⃣'];
    const leaderHTML = (d.leaderboard || []).length > 0
        ? (d.leaderboard || []).map((item, i) => `
            <div class="leader-row">
                <div class="leader-rank" style="color:${i < 3 ? 'var(--accent-yellow)' : 'var(--text-muted)'}">${medals[i] || (i+1)}</div>
                <div class="leader-name">${escapeHtml(item.annotator)}</div>
                <div class="leader-stat">
                    <strong>${item.annotations.toLocaleString()}</strong>
                    <span> anns · ${item.docs} docs</span>
                </div>
            </div>`).join('')
        : '<div style="color:var(--text-muted); font-size:0.8rem; padding:16px 0; text-align:center;">No annotator data yet.</div>';

    const leaderboardHTML = `
    <div class="stats-panel">
        <h4>🏆 Annotator Leaderboard</h4>
        ${leaderHTML}
    </div>`;

    // ── Activity sparkline ─────────────────────────────────────────────
    // Fill in any missing days in the last 7 days so we always show 7 bars
    const activity = d.recent_activity || [];
    const today = new Date();
    const days = [];
    for (let i = 6; i >= 0; i--) {
        const d2 = new Date(today);
        d2.setDate(today.getDate() - i);
        const key = d2.toISOString().slice(0, 10);
        const label = d2.toLocaleDateString('en-GB', {weekday:'short'});
        const found = activity.find(a => a.day === key);
        days.push({ label, count: found ? found.count : 0 });
    }
    const maxAct = Math.max(1, ...days.map(d => d.count));

    const activityBarsHTML = days.map(day => {
        const barPct = (day.count / maxAct * 100);
        const barHeight = Math.max(2, Math.round(barPct * 0.7)); // max 70px
        return `
        <div class="activity-bar-wrap" title="${day.label}: ${day.count} annotations">
            <div class="activity-bar" style="height:${barHeight}px"></div>
            <div class="activity-day">${day.label}</div>
        </div>`;
    }).join('');

    // ── Human vs Auto ratio ────────────────────────────────────────────
    const ratioHTML = `
    <div style="margin-top:16px;">
        <div style="font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.6px; font-weight:700; margin-bottom:8px;">
            ✍️ Human vs 🤖 Auto
        </div>
        <div class="ratio-bar-wrap">
            <div class="ratio-bar-track">
                <div class="ratio-bar-human" style="width:${humanPct}%"></div>
                <div class="ratio-bar-auto"></div>
            </div>
            <div class="ratio-labels">
                <div class="ratio-label">
                    <div class="ratio-dot" style="background:var(--accent-green)"></div>
                    Human: <strong style="color:var(--accent-green); margin-left:3px">${humanAnns.toLocaleString()}</strong>
                    &nbsp;<span style="color:var(--text-muted)">(${humanPct.toFixed(1)}%)</span>
                </div>
                <div class="ratio-label">
                    <div class="ratio-dot" style="background:var(--accent-purple)"></div>
                    Auto: <strong style="color:var(--accent-purple); margin-left:3px">${autoAnns.toLocaleString()}</strong>
                    &nbsp;<span style="color:var(--text-muted)">(${(100 - humanPct).toFixed(1)}%)</span>
                </div>
            </div>
        </div>
    </div>`;

    const activityHTML = `
    <div class="stats-panel">
        <h4>📅 Last 7 Days Activity</h4>
        <div class="activity-chart">${activityBarsHTML}</div>
        ${ratioHTML}
    </div>`;

    // ── Assemble everything into the modal body ────────────────────────
    document.getElementById('statsBody').innerHTML = `
        ${kpiHTML}
        <div class="stats-charts-row">
            ${donutHTML}
            ${entityChartHTML}
        </div>
        <div class="stats-bottom-row">
            ${leaderboardHTML}
            ${activityHTML}
        </div>`;
}

// Reuse escapeHtml if available (defined in annotate template),
// or define a local fallback here for the index page
function escapeHtml(text) {
    if (text == null) return '';
    const str = String(text);
    const div = document.createElement('div');
    div.textContent = str;
    // innerHTML escapes <, >, & — but NOT quotes!
    // We add quote escaping for safe attribute injection.
    return div.innerHTML
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * decodeHtmlEntities(text)
 *
 * 🐛 BUG FIX HELPER: Decodes HTML entities back to their original characters.
 * Used by textarea oninput handlers to compare current value against the
 * original (which was HTML-escaped for safe storage in data-original).
 *
 * @param {string} text - HTML-encoded text
 * @returns {string}    - Decoded original text
 */
function decodeHtmlEntities(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.innerHTML = text;
    return div.textContent || div.innerText || '';
}

// ================================================================
// 🍞 TOAST — Index page
// ================================================================
function showToast(msg, type = 'success', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.style.cssText = `
        padding:12px 20px; border-radius:8px; margin-bottom:8px; font-size:0.82rem;
        color:#e6edf3; max-width:480px; box-shadow:0 4px 24px rgba(0,0,0,0.4);
        background:${type === 'error' ? 'rgba(218,54,51,0.9)' : 'rgba(86,211,100,0.9)'};
    `;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ================================================================
// 📏 IAA GLOBAL TOGGLE — Queue page on/off switch
// ================================================================
// ================================================================
// 👥 ANNOTATOR TRACKER — Open / Close / Render
// ================================================================
function openAnnotatorTracker() {
    const overlay = document.getElementById('trackerOverlay');
    overlay.classList.add('open');
    document.addEventListener('keydown', closeTrackerOnEscape);
    document.getElementById('trackerBody').innerHTML = `
        <div class="stats-loading"><div class="spin">⚙️</div><div>Loading annotator data…</div></div>`;
    document.getElementById('trackerSubtitle').textContent = 'Loading…';
    fetch('/api/annotator-tracker')
        .then(r => r.json())
        .then(d => renderAnnotatorTracker(d))
        .catch(err => {
            document.getElementById('trackerBody').innerHTML =
                `<div style="text-align:center;padding:40px;color:#da3633;">Failed to load: ${err}</div>`;
        });
}
function closeTracker() {
    document.getElementById('trackerOverlay').classList.remove('open');
    document.removeEventListener('keydown', closeTrackerOnEscape);
}
function closeTrackerOnEscape(e) { if (e.key === 'Escape') closeTracker(); }
function closeTrackerOnBackdrop(e) {
    if (e.target === document.getElementById('trackerOverlay')) closeTracker();
}

function renderAnnotatorTracker(d) {
    const body = document.getElementById('trackerBody');
    document.getElementById('trackerSubtitle').textContent =
        `${d.total_annotators || 0} annotators · ${d.total_docs || 0} documents tracked`;

    // ── Annotator Summary Cards ──────────────────────────────
    let annoHTML = '';
    if ((d.annotators || []).length > 0) {
        const medals = ['🥇','🥈','🥉','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣'];
        annoHTML = `<div class="stats-panel"><h4>👥 Annotator Summary</h4>
            ${d.annotators.map((a, i) => {
                const pct = a.total_docs > 0 ? Math.round(a.completed / a.total_docs * 100) : 0;
                const barColor = pct >= 80 ? '#56d364' : pct >= 50 ? '#e3b341' : '#f0883e';
                return `<div style="padding:10px 0; border-bottom:1px solid rgba(42,49,64,0.5);">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                        <span style="font-size:1.1rem;">${medals[i] || ''}</span>
                        <strong style="color:#e6edf3;">${escapeHtml(a.annotator)}</strong>
                        <span style="font-size:0.65rem; color:#545d68; margin-left:auto;">
                            Last: ${a.last_activity ? a.last_activity.slice(0,16).replace('T',' ') : '—'}</span>
                    </div>
                    <div style="display:flex; gap:16px; font-size:0.72rem; color:#8b949e; margin-bottom:6px;">
                        <span>✅ ${a.completed||0} done</span>
                        <span>🔵 ${a.in_progress||0} wip</span>
                        <span>🏷️ ${(a.total_spans||0).toLocaleString()} spans</span>
                        <span>📄 ${a.total_docs||0} docs</span>
                    </div>
                    <div style="background:rgba(42,49,64,0.5); border-radius:4px; height:6px; overflow:hidden;">
                        <div style="background:${barColor}; height:100%; width:${pct}%; border-radius:4px;"></div>
                    </div>
                    <div style="font-size:0.6rem; color:#545d68; margin-top:2px; text-align:right;">${pct}% complete</div>
                </div>`;
            }).join('')}
        </div>`;
    } else {
        annoHTML = `<div class="stats-panel"><h4>👥 Annotator Summary</h4>
            <div style="color:#545d68; font-size:0.82rem; padding:20px 0; text-align:center;">
                No annotators recorded yet. Names are saved when you click 💾 Save or ✅ Complete.
            </div></div>`;
    }

    // ── IAA Sessions ─────────────────────────────────────────
    let iaaHTML = '';
    if ((d.iaa_sessions || []).length > 0) {
        iaaHTML = `<div class="stats-panel" style="margin-top:14px;"><h4>📏 IAA Sessions</h4>
            <div style="max-height:200px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:0.72rem;">
                <thead><tr style="color:#545d68; text-align:left;">
                    <th style="padding:6px 8px;">Doc</th><th style="padding:6px 8px;">Annotator A</th>
                    <th style="padding:6px 8px;">Annotator B</th><th style="padding:6px 8px;">Spans</th>
                    <th style="padding:6px 8px;">Date</th>
                </tr></thead><tbody>
                ${d.iaa_sessions.map(s => `<tr style="border-bottom:1px solid rgba(42,49,64,0.3);">
                    <td style="padding:6px 8px;"><a href="/annotate/${s.candidate_id||0}" style="color:#7ee8fa;">${escapeHtml(s.doc_id||'—')}</a></td>
                    <td style="padding:6px 8px; color:#8b949e;">${escapeHtml(s.annotator_a||'—')}</td>
                    <td style="padding:6px 8px; color:#f778ba; font-weight:600;">${escapeHtml(s.annotator_b||'—')}</td>
                    <td style="padding:6px 8px;">${s.span_count||0}</td>
                    <td style="padding:6px 8px; color:#545d68;">${s.last_activity ? s.last_activity.slice(0,10) : '—'}</td>
                </tr>`).join('')}
                </tbody></table></div></div>`;
    }

    // ── Recent Docs ──────────────────────────────────────────
    let docsHTML = '';
    if ((d.documents || []).length > 0) {
        docsHTML = `<div class="stats-panel" style="margin-top:14px;"><h4>📋 Recent Documents</h4>
            <div style="max-height:300px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse; font-size:0.72rem;">
                <thead><tr style="color:#545d68; text-align:left;">
                    <th style="padding:6px 8px;">ID</th><th style="padding:6px 8px;">Candidate</th>
                    <th style="padding:6px 8px;">Annotator</th><th style="padding:6px 8px;">Status</th>
                    <th style="padding:6px 8px;">Spans</th><th style="padding:6px 8px;">Last Saved</th>
                </tr></thead><tbody>
                ${d.documents.map(doc => {
                    const sc = doc.status==='completed' ? '#56d364' : doc.status==='in_progress' ? '#7ee8fa' : '#e3b341';
                    return `<tr style="border-bottom:1px solid rgba(42,49,64,0.3);">
                        <td style="padding:6px 8px;"><a href="/annotate/${doc.candidate_id||0}" style="color:#7ee8fa;">#${doc.candidate_id||'?'}</a></td>
                        <td style="padding:6px 8px; color:#8b949e; max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(doc.candidate_name||'—')}</td>
                        <td style="padding:6px 8px; color:#eeb8ff; font-weight:600;">${escapeHtml(doc.annotator||'—')}</td>
                        <td style="padding:6px 8px;"><span style="color:${sc};">${doc.status||'pending'}</span></td>
                        <td style="padding:6px 8px;">${doc.span_count||0}</td>
                        <td style="padding:6px 8px; color:#545d68;">${doc.updated_at ? doc.updated_at.slice(0,16).replace('T',' ') : '—'}</td>
                    </tr>`;
                }).join('')}
                </tbody></table></div></div>`;
    }

    body.innerHTML = `${annoHTML}${iaaHTML}${docsHTML}`;
}

function toggleIAAGlobal() {
    const isActive = localStorage.getItem('iaa_mode_active') === 'true';
    if (isActive) {
        localStorage.removeItem('iaa_mode_active');
        updateIAAQueueUI(false);
        showToast('📏 IAA Mode deactivated. Normal annotation restored.', 'success', 4000);
    } else {
        let name = localStorage.getItem('annotator_name') || '';
        if (!name) {
            name = prompt('Enter your annotator name (e.g. Soraya, Ahmad):');
            if (!name || name.trim().length < 2) {
                showToast('Need a name (2+ chars) to activate IAA!', 'error', 4000);
                return;
            }
            localStorage.setItem('annotator_name', name.trim());
        }
        localStorage.setItem('iaa_mode_active', 'true');
        updateIAAQueueUI(true);
        showToast(`📏 IAA ON! Annotating as "${name.trim()}". Open any candidate for a clean slate!`, 'success', 5000);
    }
}

function updateIAAQueueUI(active) {
    const btn = document.getElementById('iaaToggleBtn');
    const banner = document.getElementById('iaaQueueBanner');
    const nameEl = document.getElementById('iaaQueueName');
    if (active) {
        const name = localStorage.getItem('annotator_name') || '?';
        if (btn) { btn.textContent = '📏 IAA ON'; btn.style.background = 'rgba(247,120,186,0.2)'; btn.style.fontWeight = '700'; }
        if (banner) banner.style.display = 'flex';
        if (nameEl) nameEl.textContent = name;
    } else {
        if (btn) { btn.textContent = '📏 IAA Mode'; btn.style.background = ''; btn.style.fontWeight = ''; }
        if (banner) banner.style.display = 'none';
    }
}

// Restore IAA state on page load
(function() {
    if (localStorage.getItem('iaa_mode_active') === 'true') updateIAAQueueUI(true);
})();

</script>
</body>
</html>
"""


# =============================================================================
# 🎨 ANNOTATION TEMPLATE — The Main Event!
# =============================================================================

ANNOTATE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Annotate #{{ candidate_id }} | NER Tool</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    """ + SHARED_CSS + """

    /* ============ ANNOTATION LAYOUT ============ */

    /*
     * --side-width drives the right panel column.
     * The JS resize handler updates this variable on drag so the
     * grid reflows instantly without any layout thrashing. 💅
     */
    :root { --side-width: 320px; }

    .anno-layout {
        display: grid;
        /*
         * 🎭 THE GRAND STAGE:
         *   col 1 → text panel (flexible, takes all leftover space)
         *   col 2 → 5px drag handle (the velvet rope between our two rooms 🚪)
         *   col 3 → side panel (width driven by --side-width CSS variable)
         */
        grid-template-columns: 1fr 5px var(--side-width);
        height: calc(100vh - 88px);
        overflow: hidden;
    }

    /* LEFT: Text area with inline annotations */
    .text-panel {
        overflow-y: auto;
        padding: 24px 28px;
        line-height: 1.9;
        position: relative;
        /* Guard against grid blowout if text is very wide */
        min-width: 0;
    }

    /* ─── DRAG HANDLE ──────────────────────────────────────────────
     * Lives between the text panel and side panel.
     * Thin by default, widens on hover/drag for easy grabbing.
     * Think of it as the bouncer deciding how much room each section gets! 🚪
     * ────────────────────────────────────────────────────────────── */
    .resize-handle {
        background: var(--border);
        cursor: col-resize;
        position: relative;
        transition: background 0.15s;
        user-select: none;
    }
    /* Wider invisible hit-area so the user doesn't need to be pixel-perfect */
    .resize-handle::after {
        content: '';
        position: absolute;
        inset: 0 -6px;
        cursor: col-resize;
    }
    .resize-handle:hover,
    .resize-handle.dragging {
        background: var(--accent-cyan);
    }
    /* Three-dot grip icon that appears on hover */
    .resize-handle::before {
        content: '⋮';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1;
        opacity: 0;
        transition: opacity 0.15s;
        pointer-events: none;
    }
    .resize-handle:hover::before,
    .resize-handle.dragging::before { opacity: 1; }
    .text-panel .text-content {
        font-family: var(--font-mono);
        font-size: 0.88rem;
        white-space: pre-wrap;
        word-break: break-word;
        cursor: text;
        user-select: text;
        position: relative;
    }

    /* Highlighted annotation spans */
    .anno-span {
        border-radius: 3px;
        padding: 1px 2px;
        cursor: pointer;
        position: relative;
        transition: opacity 0.15s;
        border-bottom: 2px solid;
    }
    .anno-span:hover { opacity: 0.85; }
    .anno-span .anno-label {
        position: absolute;
        top: -18px;
        left: 0;
        font-size: 0.58rem;
        font-family: var(--font-body);
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        white-space: nowrap;
        padding: 1px 4px;
        border-radius: 3px;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.15s;
    }
    .anno-span:hover .anno-label { opacity: 1; }

    /* RIGHT: Entity palette + controls */
    .side-panel {
        background: var(--bg-primary);
        border-left: 1px solid var(--border);
        display: flex;
        flex-direction: column;
        /*
         * overflow:hidden keeps panel width rock-solid at --side-width.
         * No panel-level scrollbar = no 15px width-theft on children.
         * Each child that needs scrolling (palette) does it internally. 💅
         */
        overflow: hidden;
        box-sizing: border-box;
        /* Clamp so the user can't drag to an unusable size */
        min-width: 220px;
        max-width: 600px;
    }
    .side-section {
        padding: 14px 16px;
        border-bottom: 1px solid var(--border);
    }
    .side-section h3 {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }

    /* Entity buttons in palette */
    .entity-btn {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        margin: 2px;
        border: 2px solid transparent;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: var(--font-body);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s;
        opacity: 0.7;
    }
    .entity-btn:hover { opacity: 1; transform: scale(1.03); }
    .entity-btn.active {
        opacity: 1;
        border-color: #fff;
        box-shadow: 0 0 12px rgba(255,255,255,0.15);
    }
    .entity-btn .key-hint {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        opacity: 0.5;
        padding: 1px 4px;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 3px;
    }

    /* Annotation list in sidebar */
    .anno-list-item {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(42,49,64,0.3);
        font-size: 0.78rem;
    }
    .anno-list-item .anno-type {
        font-weight: 700;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        flex-shrink: 0;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .anno-list-item .anno-text {
        color: var(--text-secondary);
        word-break: break-word;
        flex: 1;
    }
    .anno-list-item .delete-btn {
        flex-shrink: 0;
        width: 18px;
        height: 18px;
        border: none;
        background: rgba(218,54,51,0.2);
        color: var(--accent-red);
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.65rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .anno-list-item .delete-btn:hover { background: rgba(218,54,51,0.5); }

    /* BIO preview panel */
    .bio-preview {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        max-height: 260px;
        overflow-y: auto;
        background: var(--bg-deep);
        padding: 10px;
        border-radius: var(--radius);
    }
    .bio-row { display: flex; gap: 12px; padding: 1px 0; }
    .bio-token { color: var(--text-secondary); min-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bio-tag { font-weight: 600; }
    .bio-tag.tag-o { color: var(--text-muted); }
    .bio-tag.tag-b { color: var(--accent-green); }
    .bio-tag.tag-i { color: var(--accent-cyan); }

    /* Toast notification */
    .toast {
        position: fixed;
        bottom: 24px;
        right: 24px;
        padding: 12px 20px;
        background: var(--bg-elevated);
        border: 1px solid var(--accent-green);
        border-radius: var(--radius);
        color: var(--accent-green);
        font-size: 0.85rem;
        z-index: 999;
        animation: slideUp 0.3s ease;
    }
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* ===========================================
     * 🪞 SAVE PREVIEW MODAL — QA Gate Before Save
     * "Check your reflection before the runway!" 💅
     * =========================================== */

    .preview-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.78);
        z-index: 600;
        align-items: flex-start;
        justify-content: center;
        padding: 30px 20px;
        overflow-y: auto;
    }
    .preview-overlay.open { display: flex; }

    .preview-modal {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 12px;
        width: 100%;
        max-width: 960px;
        box-shadow: var(--shadow);
        overflow: hidden;
        position: relative;
        animation: previewSlideIn 0.25s ease;
    }
    @keyframes previewSlideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to   { transform: translateY(0);     opacity: 1; }
    }

    .preview-header {
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border);
        padding: 18px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .preview-header h2 {
        font-family: var(--font-display);
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .preview-close {
        width: 32px; height: 32px;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--bg-elevated);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: 1rem;
        display: flex; align-items: center; justify-content: center;
        transition: all 0.15s;
    }
    .preview-close:hover {
        background: var(--accent-red);
        color: #fff;
        border-color: var(--accent-red);
    }

    .preview-body {
        padding: 24px 28px;
        max-height: 72vh;
        overflow-y: auto;
    }

    /* Stats banner cards */
    .preview-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    .preview-stat-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .preview-stat-card .stat-number {
        font-size: 1.5rem;
        font-weight: 800;
        font-family: var(--font-display);
        line-height: 1;
    }
    .preview-stat-card .stat-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Warning items */
    .preview-warnings { margin-bottom: 18px; }
    .preview-warning-item {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 0.78rem;
        display: flex;
        align-items: flex-start;
        gap: 8px;
        line-height: 1.4;
    }
    .preview-warning-item.critical {
        background: rgba(218, 54, 51, 0.12);
        border-left: 3px solid var(--accent-red);
        color: var(--accent-red);
    }
    .preview-warning-item.warning {
        background: rgba(240, 136, 62, 0.12);
        border-left: 3px solid var(--accent-orange);
        color: var(--accent-orange);
    }
    .preview-warning-item.info {
        background: rgba(86, 211, 100, 0.10);
        border-left: 3px solid var(--accent-green);
        color: var(--accent-green);
    }
    .preview-warning-icon {
        flex-shrink: 0;
        font-size: 0.9rem;
    }

    /* Entity summary rows */
    .preview-entity-group { margin-bottom: 14px; }
    .preview-entity-group h4 {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        padding-bottom: 4px;
        border-bottom: 1px solid var(--border);
    }
    .preview-entity-row {
        display: flex;
        align-items: center;
        padding: 5px 8px;
        border-radius: 4px;
        margin: 2px 0;
        font-size: 0.78rem;
        transition: background 0.1s;
    }
    .preview-entity-row:hover { background: var(--bg-hover); }
    .preview-entity-color {
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        flex-shrink: 0;
    }
    .preview-entity-label { font-weight: 600; flex: 1; }
    .preview-entity-count {
        font-weight: 700;
        font-family: var(--font-display);
        margin-left: 8px;
        min-width: 24px;
        text-align: right;
    }
    .preview-entity-limit {
        font-size: 0.65rem;
        color: var(--text-muted);
        margin-left: 4px;
    }
    .preview-entity-texts {
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-left: 18px;
        padding: 2px 0 4px 0;
        word-break: break-word;
    }

    /* ── Editable preview fields ─────────────────────────────────── */
    .preview-edit-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 3px 0 3px 18px;
    }
    .preview-edit-input {
        flex: 1;
        background: var(--bg-deep);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        padding: 4px 8px;
        outline: none;
        transition: border-color 0.15s;
    }
    .preview-edit-input:focus {
        border-color: var(--accent-cyan);
    }
    .preview-edit-input.edited {
        border-color: var(--accent-yellow);
        background: rgba(240,166,17,0.06);
    }
    .preview-edit-delete {
        background: none;
        border: 1px solid transparent;
        color: var(--text-muted);
        cursor: pointer;
        padding: 2px 5px;
        border-radius: 4px;
        font-size: 0.68rem;
        transition: all 0.15s;
    }
    .preview-edit-delete:hover {
        color: var(--accent-red);
        border-color: var(--accent-red);
        background: rgba(218,54,51,0.1);
    }
    .preview-edit-row.deleted {
        opacity: 0.35;
        text-decoration: line-through;
        pointer-events: none;
    }
    .preview-classify-select {
        width: 100%;
        background: var(--bg-deep);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-size: 0.8rem;
        padding: 6px 10px;
        outline: none;
        cursor: pointer;
    }
    .preview-classify-select:focus {
        border-color: var(--accent-cyan);
    }

    /* Footer with confirm/cancel */
    .preview-footer {
        background: var(--bg-secondary);
        border-top: 1px solid var(--border);
        padding: 16px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }
    .preview-footer-info {
        font-size: 0.72rem;
        color: var(--text-muted);
    }
    .preview-footer-actions { display: flex; gap: 8px; }

    /* Loading spinner */
    .preview-loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        color: var(--text-muted);
        font-size: 0.85rem;
    }
    .preview-loading .spinner {
        width: 32px; height: 32px;
        border: 3px solid var(--border);
        border-top-color: var(--accent-cyan);
        border-radius: 50%;
        animation: spin 0.6s linear infinite;
        margin-bottom: 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Success banner */
    .preview-all-clear {
        text-align: center;
        padding: 16px;
        background: rgba(86, 211, 100, 0.08);
        border: 1px solid rgba(86, 211, 100, 0.2);
        border-radius: 8px;
        margin-bottom: 18px;
        font-size: 0.85rem;
        color: var(--accent-green);
    }

    /* ================================================================
     * 🗂️ REDESIGNED SAVE PREVIEW FORM — Clean structured layout
     * Every field is visible. Every field is editable. PERIOD. 💅
     * ================================================================ */

    /* The main form container — wraps all section blocks */
    .preview-form {
        display: flex;
        flex-direction: column;
        gap: 0;
    }

    /* One category block (Personal / Professional / Education etc.) */
    .pf-section {
        margin-bottom: 18px;
    }

    /* Category heading bar */
    .pf-section-header {
        font-size: 0.63rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-muted);
        padding: 6px 0 5px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 4px;
    }

    /* One field row: label column | inputs column */
    .pf-field {
        display: grid;
        grid-template-columns: 175px 1fr;
        gap: 10px;
        padding: 4px 0;
        border-bottom: 1px solid rgba(42,49,64,0.25);
        align-items: start;
        min-height: 32px;
    }
    .pf-field:last-child { border-bottom: none; }

    /* Label column — coloured dot + name + required marker + missing badge */
    .pf-label {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 5px;
        padding-top: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--text-secondary);
        cursor: default;
        user-select: none;
        line-height: 1.3;
    }
    .pf-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .pf-required {
        color: var(--accent-red);
        font-size: 0.7rem;
        font-weight: 700;
        line-height: 1;
    }
    .pf-missing-badge {
        font-size: 0.56rem;
        font-weight: 700;
        color: var(--accent-orange);
        background: rgba(240,136,62,0.12);
        border: 1px solid rgba(240,136,62,0.35);
        padding: 1px 5px;
        border-radius: 10px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    /* Inputs column — stacks multiple rows vertically */
    .pf-inputs {
        display: flex;
        flex-direction: column;
        gap: 3px;
        padding: 2px 0;
    }

    /* Single input row: text input + optional delete button */
    .pf-input-row {
        display: flex;
        align-items: center;
        gap: 4px;
        transition: opacity 0.15s;
    }
    /* Deleted rows fade out and strike through */
    /* 🐛 BUG FIX: Previously used pointer-events:none which blocked
     * ALL clicks on the entire row — including the ↩ undo button!
     * Now we keep the row interactive but disable only the INPUT via
     * a child selector. The delete/undo button stays clickable.
     * Like dimming the lights but leaving the exit sign on! 🚪💡 */
    .pf-input-row.deleted {
        opacity: 0.28;
        text-decoration: line-through;
    }
    .pf-input-row.deleted .pf-input {
        pointer-events: none;
    }

    /* The actual text input */
    .pf-input {
        flex: 1;
        min-width: 0;
        background: var(--bg-deep);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 0.74rem;
        padding: 5px 9px;
        outline: none;
        transition: border-color 0.15s, background 0.15s;
    }
    .pf-input:focus {
        border-color: var(--accent-cyan);
        background: rgba(126,232,250,0.03);
    }
    /* Yellow border when user edits an existing value */
    .pf-input.edited {
        border-color: var(--accent-yellow);
        background: rgba(227,179,65,0.05);
    }

    /* ══════════════════════════════════════════════════════════════════
       📝 TEXTAREA STYLES — For long-form fields like JOB_DESCRIPTION
       
       🐛 BUG FIX: Regular <input type="text"> can't handle multi-line
       content! Newlines break the value="" attribute and browsers
       truncate the text. Textareas save the day! 💅
       ══════════════════════════════════════════════════════════════════ */
    .pf-textarea {
        flex: 1;
        min-width: 0;
        min-height: 60px;
        max-height: 200px;
        resize: vertical;
        background: var(--bg-deep);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 0.74rem;
        padding: 8px 9px;
        outline: none;
        transition: border-color 0.15s, background 0.15s;
        line-height: 1.5;
        overflow-y: auto;
    }
    .pf-textarea:focus {
        border-color: var(--accent-cyan);
        background: rgba(126,232,250,0.03);
    }
    .pf-textarea.edited {
        border-color: var(--accent-yellow);
        background: rgba(227,179,65,0.05);
    }
    .pf-textarea.pf-add-input {
        border-style: dashed;
        color: var(--text-muted);
        font-style: italic;
        min-height: 40px;
        background: transparent;
    }
    .pf-textarea.pf-add-input:focus,
    .pf-textarea.pf-add-input.edited {
        border-style: solid;
        color: var(--text-primary);
        font-style: normal;
        background: var(--bg-deep);
    }
    /* Align delete button to top for textarea rows */
    .pf-input-row:has(.pf-textarea) {
        align-items: flex-start;
    }
    .pf-input-row:has(.pf-textarea) .pf-del-btn {
        margin-top: 6px;
    }

    /* "Add…" input — dashed, italic, greyed — turns solid on focus/type */
    .pf-add-input {
        border-style: dashed;
        color: var(--text-muted);
        font-style: italic;
        font-size: 0.72rem;
        background: transparent;
    }
    .pf-add-input:focus,
    .pf-add-input.edited {
        border-style: solid;
        color: var(--text-primary);
        font-style: normal;
        background: var(--bg-deep);
    }

    /* Delete / undo button beside each existing annotation row */
    .pf-del-btn {
        flex-shrink: 0;
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid transparent;
        border-radius: 4px;
        background: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 0.68rem;
        line-height: 1;
        transition: all 0.12s;
    }
    .pf-del-btn:hover {
        color: var(--accent-red);
        border-color: var(--accent-red);
        background: rgba(218,54,51,0.1);
    }
    .pf-del-btn.undo {
        color: var(--accent-green);
        border-color: var(--accent-green);
        background: rgba(86,211,100,0.1);
    }

    /* Classification row (Function + Industry) */
    .pf-classify {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 18px;
        padding: 12px 14px;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    .pf-classify-field label {
        display: block;
        font-size: 0.61rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-muted);
        margin-bottom: 5px;
    }
    .pf-select {
        width: 100%;
        background: var(--bg-deep);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
        font-size: 0.78rem;
        padding: 6px 9px;
        outline: none;
        cursor: pointer;
        transition: border-color 0.15s;
    }
    .pf-select:focus { border-color: var(--accent-cyan); }

    /* ===========================================
     * 📦 EXPORT OPTIONS MODAL
     * Because prompt() is so last season! 💅
     * =========================================== */

    /* ── Export overlay backdrop ── */
    .export-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.78);
        z-index: 700;
        align-items: center;
        justify-content: center;
        padding: 20px;
        /* Allows the overlay itself to scroll on very small screens */
        overflow-y: auto;
    }
    .export-overlay.open { display: flex; }

    /* ── Export modal card ──
       Uses flex-column so header + body + footer stack properly.
       max-height ensures the footer (confirm button) is ALWAYS visible,
       even on short viewports — no more diva falling off the runway! 💅 */
    .export-modal {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 16px;
        width: 100%;
        max-width: 460px;
        /* Flex column layout keeps header/footer pinned, body scrolls */
        display: flex;
        flex-direction: column;
        max-height: calc(100vh - 60px); /* never taller than the viewport */
        box-shadow: 0 24px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05);
        overflow: hidden;
        animation: previewSlideIn 0.2s ease;
        /* Ensure whole card is visible in viewport */
        margin: auto;
    }

    /* ── Header: always pinned to top ── */
    .export-modal-header {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(99,102,241,0.08) 100%);
        border-bottom: 1px solid var(--border);
        padding: 18px 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-shrink: 0; /* never squish the header */
    }
    .export-modal-header h3 {
        font-family: var(--font-display);
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    /* ── Body: scrollable if content overflows ── */
    .export-modal-body {
        padding: 22px;
        overflow-y: auto; /* scroll when content is tall */
        flex: 1;           /* take remaining space between header and footer */
    }

    /* ── Field rows inside the modal body ── */
    .export-field {
        margin-bottom: 18px;
    }
    .export-field:last-child { margin-bottom: 0; }
    .export-field label {
        display: block;
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 7px;
        font-weight: 600;
    }
    .export-field select {
        width: 100%;
        padding: 10px 14px;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--bg-secondary);
        color: var(--text-primary);
        font-family: var(--font-body);
        font-size: 0.86rem;
        cursor: pointer;
        appearance: auto;
        transition: border-color 0.15s;
    }
    .export-field select:hover { border-color: var(--accent-cyan); }
    .export-field select:focus {
        outline: none;
        border-color: var(--accent-cyan);
        box-shadow: 0 0 0 3px rgba(6,182,212,0.15);
    }

    /* ── Footer: always pinned to bottom with confirm button ── */
    .export-modal-footer {
        background: var(--bg-secondary);
        border-top: 1px solid var(--border);
        padding: 14px 22px;
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        flex-shrink: 0; /* never squish the footer — it MUST stay visible */
    }
    /* Make the confirm button extra satisfying to press 💅 */
    .export-modal-footer .btn-primary {
        min-width: 120px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    /* ── Export info hint box (replaces inline style) ── */
    .export-hint {
        font-size: 0.75rem;
        color: var(--text-muted);
        padding: 10px 14px;
        background: var(--bg-secondary);
        border-radius: var(--radius);
        border-left: 3px solid var(--accent-cyan);
        line-height: 1.5;
        margin-top: 4px;
    }

    /* ═══════════════════════════════════════════════════
       TOAST NOTIFICATION SYSTEM 🍞✨
       Replaces the plain browser alert() with gorgeous
       animated toasts — because we deserve nice things!
       ═══════════════════════════════════════════════════ */
    #toast-container {
        position: fixed;
        bottom: 28px;
        right: 28px;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        gap: 10px;
        pointer-events: none; /* clicks pass through the gap */
    }
    .toast {
        pointer-events: all;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 14px 18px;
        border-radius: 12px;
        background: var(--bg-primary);
        border: 1px solid var(--border);
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.04);
        min-width: 280px;
        max-width: 380px;
        animation: toastIn 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        position: relative;
        overflow: hidden;
    }
    /* Coloured left-accent bar per toast type */
    .toast::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 12px 0 0 12px;
    }
    .toast.toast-success::before { background: var(--accent-green); }
    .toast.toast-error::before   { background: #f87171; }
    .toast.toast-info::before    { background: var(--accent-cyan); }
    .toast.toast-warning::before { background: var(--accent-yellow); }

    .toast-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 1px; }
    .toast-content { flex: 1; }
    .toast-title {
        font-family: var(--font-display);
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .toast-msg {
        font-size: 0.78rem;
        color: var(--text-muted);
        line-height: 1.5;
    }
    .toast-close {
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 0.85rem;
        padding: 0;
        flex-shrink: 0;
        opacity: 0.6;
        transition: opacity 0.15s;
    }
    .toast-close:hover { opacity: 1; }
    /* Progress bar showing auto-dismiss countdown */
    .toast-progress {
        position: absolute;
        bottom: 0; left: 0;
        height: 3px;
        border-radius: 0 0 12px 12px;
        animation: toastProgress linear forwards;
    }
    .toast.toast-success .toast-progress { background: var(--accent-green); }
    .toast.toast-error   .toast-progress { background: #f87171; }
    .toast.toast-info    .toast-progress { background: var(--accent-cyan); }
    .toast.toast-warning .toast-progress { background: var(--accent-yellow); }

    /* Slide-out animation */
    .toast.toast-hiding {
        animation: toastOut 0.25s ease forwards;
    }

    @keyframes toastIn {
        from { opacity: 0; transform: translateX(60px) scale(0.92); }
        to   { opacity: 1; transform: translateX(0) scale(1); }
    }
    @keyframes toastOut {
        from { opacity: 1; transform: translateX(0) scale(1); }
        to   { opacity: 0; transform: translateX(60px) scale(0.92); }
    }
    @keyframes toastProgress {
        from { width: 100%; }
        to   { width: 0%; }
    }

    /* Edge case suggestion */
    .edge-hint {
        padding: 6px 10px;
        margin: 4px 0;
        background: rgba(240,136,62,0.1);
        border-left: 3px solid var(--accent-orange);
        border-radius: 0 var(--radius) var(--radius) 0;
        font-size: 0.75rem;
        color: var(--accent-orange);
    }

    /* Toolbar */
    .toolbar {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        align-items: center;
    }

    /* ===========================================
     * 🔍 INSPECTOR MODALS — Labels / BIO / Edge
     * Like popup dressing rooms for your data! 💄
     * =========================================== */

    .inspector-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.72);
        z-index: 550;
        align-items: flex-start;
        justify-content: center;
        padding: 40px 20px;
        overflow-y: auto;
        backdrop-filter: blur(3px);
    }
    .inspector-overlay.open { display: flex; }

    .inspector-modal {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 12px;
        width: 100%;
        max-width: 780px;
        max-height: 80vh;
        box-shadow: var(--shadow);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        animation: previewSlideIn 0.2s ease;
    }
    .inspector-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border);
        flex-shrink: 0;
    }
    .inspector-header h3 {
        font-family: var(--font-display);
        font-size: 1rem;
        font-weight: 700;
    }
    .inspector-body {
        padding: 16px 20px;
        overflow-y: auto;
        flex: 1;
    }

    /* 🆕 Wider modal variant for Classification & Skills popup */
    .classify-modal-wide {
        width: 620px;
        max-width: 90vw;
        max-height: 85vh;
    }

    /* ============================================================
     * 📏 IAA MODAL STYLES — Agreement Dashboard
     * ============================================================ */

    /* Wider modal for heatmap + metrics layout */
    .iaa-modal-wide {
        width: 820px;
        max-width: 95vw;
        max-height: 90vh;
    }

    /* IAA mode banner — shown when annotator B is working */
    .iaa-mode-banner {
        background: linear-gradient(90deg, rgba(247,120,186,0.18), rgba(238,184,255,0.10), rgba(247,120,186,0.18));
        border-top: 2px solid rgba(247,120,186,0.5);
        border-bottom: 2px solid rgba(247,120,186,0.5);
        padding: 12px 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.82rem;
        color: var(--accent-pink);
        animation: iaaBannerPulse 3s ease infinite;
        position: relative;
        overflow: hidden;
    }
    /* Animated shimmer across the banner */
    .iaa-mode-banner::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(247,120,186,0.08) 40%,
            rgba(247,120,186,0.15) 50%,
            rgba(247,120,186,0.08) 60%,
            transparent 100%);
        animation: iaaShimmer 4s ease infinite;
    }
    @keyframes iaaShimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    @keyframes iaaBannerPulse {
        0%, 100% { border-color: rgba(247,120,186,0.4); }
        50% { border-color: rgba(247,120,186,0.8); }
    }
    .iaa-mode-banner > * { position: relative; z-index: 1; }
    .iaa-mode-banner .iaa-badge {
        background: var(--accent-pink);
        color: var(--bg-primary);
        font-size: 0.68rem;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        animation: iaaBadgePulse 2s ease infinite;
    }
    @keyframes iaaBadgePulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(247,120,186,0.4); }
        50% { box-shadow: 0 0 12px 3px rgba(247,120,186,0.3); }
    }

    /* ============================================================
     * 📏🔥 IAA ACTIVE — body.iaa-active theme overrides
     *
     * When active, the ENTIRE page gets a pink accent treatment
     * so Annotator B can NEVER forget they're in IAA mode.
     * ============================================================ */
    body.iaa-active {
        outline: 3px solid rgba(247,120,186,0.5);
        outline-offset: -3px;
        animation: iaaPagePulse 3s ease infinite;
    }
    @keyframes iaaPagePulse {
        0%, 100% { outline-color: rgba(247,120,186,0.35); }
        50% { outline-color: rgba(247,120,186,0.7); }
    }
    body.iaa-active .header {
        background: linear-gradient(180deg, var(--bg-primary) 0%, rgba(247,120,186,0.06) 100%);
        border-bottom: 2px solid rgba(247,120,186,0.5);
    }
    body.iaa-active .text-panel {
        background: linear-gradient(135deg, rgba(247,120,186,0.03) 0%, transparent 60%);
    }
    body.iaa-active .side-panel {
        border-left: 2px solid rgba(247,120,186,0.4);
    }
    body.iaa-active .palette-section h3 {
        color: var(--accent-pink);
    }
    body.iaa-active .resize-handle:hover,
    body.iaa-active .resize-handle.dragging {
        background: var(--accent-pink);
    }
    body.iaa-active .iaa-dimmed-btn {
        opacity: 0.3;
        pointer-events: none;
        filter: grayscale(0.8);
    }
    body.iaa-active .anno-span {
        box-shadow: 0 0 0 1px rgba(247,120,186,0.2);
    }
    /* Shrink layout to accommodate the IAA banner above it */
    body.iaa-active .anno-layout {
        height: calc(100vh - 140px);
    }

    /* Annotator name prompt modal */
    .annotator-prompt-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.85);
        z-index: 9999;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(6px);
    }
    .annotator-prompt-overlay.open { display: flex; }
    .annotator-prompt-card {
        background: var(--bg-primary);
        border: 1px solid var(--accent-pink);
        border-radius: 16px;
        padding: 36px 40px;
        width: 380px;
        max-width: 90vw;
        text-align: center;
        box-shadow: 0 0 60px rgba(247,120,186,0.15);
        animation: previewSlideIn 0.3s ease;
    }
    .annotator-prompt-card h3 {
        font-family: var(--font-display);
        font-size: 1.2rem;
        color: var(--accent-pink);
        margin-bottom: 8px;
    }
    .annotator-prompt-card p {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-bottom: 18px;
        line-height: 1.6;
    }
    .annotator-prompt-card input {
        width: 100%;
        padding: 10px 14px;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text-primary);
        font-family: var(--font-body);
        font-size: 0.88rem;
        margin-bottom: 14px;
        outline: none;
        transition: border-color 0.2s;
    }
    .annotator-prompt-card input:focus {
        border-color: var(--accent-pink);
    }

    /* IAA KPI cards row */
    .iaa-kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 18px;
    }
    .iaa-kpi-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 12px;
        text-align: center;
    }
    .iaa-kpi-value {
        font-family: var(--font-display);
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .iaa-kpi-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Heatmap grid */
    .iaa-heatmap {
        display: grid;
        gap: 2px;
        margin-bottom: 18px;
    }
    .iaa-heatmap-cell {
        border-radius: 4px;
        padding: 6px 4px;
        text-align: center;
        font-size: 0.62rem;
        font-weight: 600;
        color: var(--text-primary);
        transition: transform 0.15s, box-shadow 0.15s;
        cursor: default;
        position: relative;
    }
    .iaa-heatmap-cell:hover {
        transform: scale(1.08);
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
        z-index: 2;
    }
    .iaa-heatmap-header {
        font-size: 0.58rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 4px;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .iaa-heatmap-row-label {
        font-size: 0.62rem;
        color: var(--text-secondary);
        text-align: right;
        padding-right: 6px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Per-document results table */
    .iaa-doc-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.72rem;
    }
    .iaa-doc-table th {
        background: var(--bg-secondary);
        padding: 8px 10px;
        text-align: left;
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid var(--border);
    }
    .iaa-doc-table td {
        padding: 8px 10px;
        border-bottom: 1px solid rgba(42,49,64,0.5);
        color: var(--text-secondary);
    }
    .iaa-doc-table tr:hover td {
        background: var(--bg-hover);
    }

    /* Kappa interpretation badge */
    .kappa-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kappa-almost-perfect { background: rgba(86,211,100,0.2); color: #56d364; }
    .kappa-substantial    { background: rgba(126,232,250,0.2); color: #7ee8fa; }
    .kappa-moderate       { background: rgba(227,179,65,0.2);  color: #e3b341; }
    .kappa-fair           { background: rgba(240,136,62,0.2);  color: #f0883e; }
    .kappa-poor           { background: rgba(218,54,51,0.2);   color: #da3633; }

    /* Empty state for IAA dashboard */
    .iaa-empty-state {
        text-align: center;
        padding: 50px 30px;
        color: var(--text-muted);
    }
    .iaa-empty-state .iaa-empty-icon {
        font-size: 3rem;
        margin-bottom: 12px;
    }
    .iaa-empty-state h4 {
        font-family: var(--font-display);
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .iaa-empty-state p {
        font-size: 0.78rem;
        line-height: 1.7;
        max-width: 380px;
        margin: 0 auto;
    }

    /* Badge on buttons to show annotation count */
    .tab-badge {
        display: inline-block;
        background: var(--accent-cyan);
        color: var(--bg-deep);
        font-size: 0.58rem;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 8px;
        margin-left: 3px;
        vertical-align: middle;
    }

    /* The entity palette section — always visible! */
    .palette-section {
        padding: 12px 14px;
        border-bottom: 1px solid var(--border);
        /*
         * flex:1 — palette fills ALL remaining height after Classification
         * is pinned at the bottom. No arbitrary max-height needed anymore.
         * min-height:0 is required: without it, flex children ignore
         * overflow and the panel can still blow out. 🎨
         */
        flex: 1;
        overflow-y: auto;
        min-height: 0;
    }
    .palette-section h3 {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    /*
     * Shared style for the Function / Industry <select> dropdowns.
     * display:block + width:100% + box-sizing:border-box = selects fill
     * their grid cell exactly, no overflow, no browser quirks. 🎯
     * (Replaces class="btn" which injects inline-flex and breaks width.)
     */
    .classify-select {
        display: block;
        width: 100%;
        box-sizing: border-box;
        font-size: 0.72rem;
        padding: 5px 6px;
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        font-family: var(--font-body);
        cursor: pointer;
        transition: border-color 0.15s;
    }
    .classify-select:hover { border-color: var(--accent-cyan); }
    .classify-select:focus { outline: none; border-color: var(--accent-cyan); }

    /* 🆕 Skill/Tag chip containers and chips */
    .skill-tags-container {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        min-height: 24px;
        padding: 4px;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
    }
    .skill-tags-container:empty::before {
        content: 'No items yet — use Re-run or add manually';
        color: var(--text-muted);
        font-size: 0.58rem;
        font-style: italic;
        padding: 2px 4px;
    }
    .skill-tag-chip {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        font-size: 0.62rem;
        padding: 2px 7px;
        border-radius: 12px;
        background: rgba(126, 232, 250, 0.12);
        color: var(--accent-cyan);
        border: 1px solid rgba(126, 232, 250, 0.25);
        line-height: 1.4;
        max-width: 100%;
        word-break: break-word;
    }
    .skill-tag-chip.soft {
        background: rgba(238, 184, 255, 0.12);
        color: var(--accent-purple);
        border-color: rgba(238, 184, 255, 0.25);
    }
    .skill-tag-chip.ai-tag {
        background: rgba(86, 211, 100, 0.12);
        color: var(--accent-green);
        border-color: rgba(86, 211, 100, 0.25);
    }
    .skill-tag-chip .tag-remove {
        cursor: pointer;
        font-size: 0.55rem;
        opacity: 0.6;
        transition: opacity 0.15s;
        margin-left: 2px;
    }
    .skill-tag-chip .tag-remove:hover {
        opacity: 1;
        color: var(--accent-red);
    }
    .skill-tag-input {
        flex: 1;
        min-width: 80px;
        font-size: 0.62rem;
        padding: 3px 6px;
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px dashed var(--border);
        border-radius: var(--radius);
        font-family: var(--font-body);
    }
    .skill-tag-input:focus {
        outline: none;
        border-color: var(--accent-cyan);
        border-style: solid;
    }

    /* ── Quick-pick palette ─────────────────────────────────────── */
    .palette-group-label {
        font-size: 0.58rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin: 8px 0 4px;
    }
    .palette-group-label:first-child { margin-top: 0; }
    .palette-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 2px; }
    .palette-chip {
        font-size: 0.6rem;
        padding: 2px 8px;
        border-radius: 20px;
        cursor: pointer;
        border: 1px solid rgba(86,211,100,0.35);
        color: var(--text-muted);
        background: transparent;
        transition: all 0.15s ease;
        user-select: none;
    }
    .palette-chip:hover { border-color: var(--accent-green); color: var(--accent-green); }
    .palette-chip.active {
        background: rgba(86,211,100,0.18);
        border-color: var(--accent-green);
        color: var(--accent-green);
        font-weight: 600;
    }

    /* Annotation list items */
    .anno-list-empty {
        color: var(--text-muted);
        font-size: 0.78rem;
        text-align: center;
        padding: 20px 0;
    }

    /*
     * 🔦 FOCUSED ANNOTATION ROW
     * Lights up when the user clicks a highlighted span in the text.
     * Two-beat pulse: bright cyan flash → fades to a subtle standing highlight
     * so you always know WHERE you are in the list. Like a spotlight sweeping
     * the crowd and landing on YOUR table! 🎤✨
     */
    @keyframes annoFocusPulse {
        0%   { background: rgba(126,232,250,0.35); }
        60%  { background: rgba(126,232,250,0.20); }
        100% { background: rgba(126,232,250,0.12); }
    }
    .anno-list-item--focused {
        animation: annoFocusPulse 0.6s ease forwards;
        border-radius: 6px;
        /* Left accent bar — a little "you are here" flag 🚩 */
        border-left: 3px solid var(--accent-cyan) !important;
        padding-left: 8px !important;
    }
    /* ===========================================
     * ❓ HELP MODAL — The Fairy Codemother's Manual
     * =========================================== */

    /* Dark overlay behind the modal */
    .help-overlay {
        display: none;                   /* Hidden until openHelp() is called */
        position: fixed;
        inset: 0;                        /* Cover the entire viewport */
        background: rgba(0, 0, 0, 0.72);
        z-index: 500;
        align-items: flex-start;         /* Modal sits near top, not dead centre */
        justify-content: center;
        padding: 40px 20px;
        overflow-y: auto;
    }
    /* When open, switch display to flex so it centres the modal card */
    .help-overlay.open { display: flex; }

    /* The modal card itself */
    .help-modal {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 12px;
        width: 100%;
        max-width: 760px;
        box-shadow: var(--shadow);
        overflow: hidden;               /* Clip children to rounded corners */
        position: relative;
    }

    /* Sticky header inside the modal */
    .help-header {
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border);
        padding: 20px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .help-header h2 {
        font-family: var(--font-display);
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--accent-purple);
        letter-spacing: -0.5px;
    }
    .help-close {
        width: 32px; height: 32px;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--bg-elevated);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: 1rem;
        display: flex; align-items: center; justify-content: center;
        transition: all 0.15s;
    }
    .help-close:hover { border-color: var(--accent-red); color: var(--accent-red); }

    /* Scrollable body inside the modal */
    .help-body {
        padding: 24px 28px;
        overflow-y: auto;
        max-height: calc(90vh - 80px);  /* Don't let it grow taller than the screen */
    }

    /* Section headers inside the manual */
    .help-section {
        margin-bottom: 28px;
    }
    .help-section h3 {
        font-size: 0.78rem;
        color: var(--accent-cyan);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }
    .help-section p {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.7;
        margin-bottom: 8px;
    }

    /* Step list — numbered workflow steps */
    .help-steps {
        counter-reset: step-counter;
        list-style: none;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .help-steps li {
        counter-increment: step-counter;
        display: flex;
        gap: 14px;
        align-items: flex-start;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 12px 14px;
        font-size: 0.86rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    /* The big numbered circle before each step */
    .help-steps li::before {
        content: counter(step-counter);
        flex-shrink: 0;
        width: 24px; height: 24px;
        background: var(--accent-cyan);
        color: var(--bg-deep);
        border-radius: 50%;
        font-size: 0.72rem;
        font-weight: 800;
        display: flex; align-items: center; justify-content: center;
        margin-top: 1px;
    }
    .help-steps li strong { color: var(--text-primary); }

    /* Keyboard shortcut grid */
    .shortcut-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 8px;
    }
    .shortcut-item {
        display: flex;
        align-items: center;
        gap: 10px;
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 8px 12px;
        font-size: 0.82rem;
        color: var(--text-secondary);
    }
    /* Rendered keyboard key badge */
    .kbd {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 28px; height: 22px;
        padding: 0 6px;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-bottom-width: 2px;         /* Classic keycap depth effect */
        border-radius: 4px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--accent-yellow);
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* Tips / warning cards inside the manual */
    .help-tip {
        display: flex;
        gap: 10px;
        padding: 10px 14px;
        border-radius: var(--radius);
        font-size: 0.84rem;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .help-tip.tip-info    { background: rgba(126,232,250,0.08); border-left: 3px solid var(--accent-cyan); color: var(--accent-cyan); }
    .help-tip.tip-warn    { background: rgba(240,136,62,0.08);  border-left: 3px solid var(--accent-orange); color: var(--accent-orange); }
    .help-tip.tip-success { background: rgba(86,211,100,0.08);  border-left: 3px solid var(--accent-green); color: var(--accent-green); }

    /* Entity category colour pill (used in the manual's entity reference) */
    .entity-ref-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 600;
        margin: 2px;
    }

   <!-- ================================================================
     📦 EXPORT MODAL — FULL GLAM EDITION ✨
     ================================================================ -->
<div class="export-overlay" id="exportOverlay" onclick="if(event.target===this)closeExportModal()">
  <div class="export-modal">

    <!-- Header -->
    <div class="export-modal-header">
      <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.6rem;">📦</span>
        <h3 id="exportModalTitle" style="margin:0; font-size:1.25rem; color:var(--accent-cyan);">Export Options</h3>
      </div>
      <button class="export-close" onclick="closeExportModal()" title="Cancel">✕</button>
    </div>

    <!-- Body -->
    <div class="export-modal-body" id="exportModalBody">
      <!-- JS will fill this -->
    </div>

    <!-- Footer -->
    <div class="export-modal-footer">
      <button class="btn" onclick="closeExportModal()">Cancel</button>
      <button class="btn btn-primary" id="exportConfirmBtn" onclick="confirmExport()">
        📤 Export Now
      </button>
    </div>

  </div>
</div>

    </style>
</head>
<body>

<!-- ================================================================
     HEADER — Two-row layout: Title row + Tool bar
     Like a marquee above the stage, darling! 🎭
     ================================================================ -->
<div class="header" style="flex-direction:column; align-items:stretch; gap:0; padding:0;">

    <!-- Row 1: Navigation + Title + Save Actions -->
    <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 24px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <a href="/?page={{ back_page }}&status={{ back_status }}#row-{{ candidate_id }}" class="btn btn-sm" style="font-size:0.72rem; padding:4px 10px;">← Back</a>
            <h1 style="font-size:1.1rem;">🏷️ <span>#{{ candidate_id }}</span>
                {% if structured.name %}<span style="font-weight:400; color:var(--text-secondary); font-size:0.82rem; margin-left:4px;">{{ structured.name }}</span>{% endif %}
            </h1>
            <!-- id="statusBadge" lets JS update this live after every save
                 without needing a full page reload! 💅 -->
            <span id="statusBadge"
                  class="badge {{ 'badge-complete' if ann_status == 'completed' else 'badge-progress' if ann_status == 'in_progress' else 'badge-pending' }}">
                {{ ann_status }}
            </span>
            <!-- 👤 Current annotator — shows who's working, click to change -->
            <span id="annotatorBadge"
                  style="font-size:0.65rem; color:var(--text-muted); cursor:pointer;
                         padding:2px 8px; border:1px dashed var(--border); border-radius:4px;"
                  onclick="changeAnnotatorName()"
                  title="Click to change annotator name">
                👤 <span id="annotatorBadgeName">—</span>
            </span>
        </div>
        <div style="display:flex; gap:6px;" id="normalSaveButtons">
            <button class="btn" onclick="saveAnnotations('in_progress')" title="Save draft (Ctrl+S)">💾 Save</button>
            <button class="btn btn-success" onclick="showSavePreview('completed')" title="Validate & mark complete">✅ Complete</button>
        </div>
    </div>

    <!-- Row 2: Tool buttons — grouped by function -->
    <div style="display:flex; align-items:center; justify-content:space-between; padding:6px 24px; background:var(--bg-secondary); border-top:1px solid var(--border);">
        <!-- Left: Annotation tools -->
        <div style="display:flex; gap:6px;">
            <button class="btn btn-sm" onclick="runPreAnnotate()" title="Re-run auto-labeling">🤖 Pre-Annotate</button>
            <button class="btn btn-sm" onclick="clearAll()" title="Remove all annotations">🗑️ Clear</button>
        </div>
        <!-- Center: Inspector popups -->
        <div style="display:flex; gap:6px;">
            <button class="btn btn-sm" onclick="openLabelsModal()" title="View all labels">
                📋 Labels <span class="tab-badge" id="headerLabelsBadge">0</span>
            </button>
            <button class="btn btn-sm" onclick="openBIOModal()" title="Preview BIO tags">🏷️ BIO</button>
            <button class="btn btn-sm" onclick="openEdgeModal()" title="Edge case scan">🧩 Edge</button>
            <button class="btn btn-sm" onclick="openIAAModal()" title="Inter-Annotator Agreement dashboard"
                    style="border-color:var(--accent-pink); color:var(--accent-pink);">
                📏 IAA
            </button>
            <button class="btn btn-sm" onclick="openClassifyModal()" title="Classification & Skills"
                    style="border-color:var(--accent-purple); color:var(--accent-purple);">
                🧠 Classify
            </button>
            <!-- 📋 Annotations popup button — clicks open the full Labels inspector modal 
                 (the same rich popup that openLabelsModal() builds). The badge counter
                 is updated live by updateAnnoList() whenever annotations change. 🎭 -->
            <button class="btn btn-sm" onclick="openLabelsModal()" title="View all annotations in popup">
                📋 Annotations
                <span class="tab-badge" id="headerAnnotationsBadge">0</span>
            </button>
        </div>
        <!-- Right: Help -->
        <div>
            <button class="btn btn-sm" onclick="openHelp()" title="User manual" style="border-color:var(--accent-purple); color:var(--accent-purple);">❓ Help</button>
        </div>
    </div>
</div>

<!-- ================================================================
     📏 IAA MODE BANNER — Between toolbar and main layout.
     THIS POSITION IS CRITICAL! Must be ABOVE the anno-layout grid
     or the banner renders off-screen below the viewport! 🎭
     ================================================================ -->
<div id="iaaBanner" class="iaa-mode-banner" style="display:none;">
    <span class="iaa-badge">IAA MODE</span>
    <span>Annotating as <strong id="iaaBannerName">—</strong> (Annotator B).
          Primary annotations hidden · Label from scratch.</span>
    <!-- Save status indicator — shows ✅ after successful save, ⚠️ if unsaved changes -->
    <span id="iaaSaveStatus" style="font-size:0.72rem; color:var(--text-muted);
          margin-left:8px; transition: color 0.3s;">
        <!-- Populated dynamically by JS -->
    </span>
    <button class="btn btn-sm" onclick="saveIAAAnnotations()"
            id="iaaSaveBtn"
            style="margin-left:auto; font-size:0.78rem; padding:6px 18px;
                   background:rgba(247,120,186,0.15);
                   border: 2px solid var(--accent-pink); color:var(--accent-pink); font-weight:700;">
        💾 Save IAA
    </button>
    <button class="btn btn-sm" onclick="exitIAAMode()"
            style="font-size:0.68rem; padding:4px 12px;">
        ✕ Exit
    </button>
</div>

    <!-- MAIN LAYOUT — 3 columns: text | drag-handle | side panel -->
<div class="anno-layout" id="annoLayout">

    <!-- LEFT: Annotated Text -->
    <div class="text-panel" id="textPanel">
        <div class="text-content" id="textContent"></div>
    </div>

    <!-- DRAG HANDLE — grab and drag left/right to resize the side panel.
         Think of it as the velvet rope between VIP and the dance floor! 🚪💅 -->
    <div class="resize-handle" id="resizeHandle" title="Drag to resize panel"></div>

    <!-- RIGHT: Side Panel — Entity Palette + Classify + Annotation List -->
    <div class="side-panel">

        <!-- 
            🎨 ENTITY PALETTE — Always visible at top.
            Your makeup station is ALWAYS open, honey! 💄
        -->
        <div class="palette-section">
            <h3>🎨 Entity Palette <small style="opacity:0.5; font-size:0.58rem;">(select then highlight)</small></h3>
            <div id="entityPalette"></div>
        </div>

        <!-- 
            🧠 CLASSIFICATION — Moved to popup modal!
            Just a quick-access button here to open it. 💅
        -->
        <div style="padding:8px 14px; border-top:1px solid var(--border); flex-shrink:0;">
            <button class="btn btn-sm" onclick="openClassifyModal()" 
                    style="width:100%; font-size:0.68rem; border-color:var(--accent-purple); color:var(--accent-purple);">
                🧠 Classification & Skills
            </button>
        </div>

    </div><!-- end side-panel -->
</div>

<!-- ================================================================
     ❓ HELP MODAL — Fairy Codemother's Annotation Manual
     Click the dark backdrop or press Escape to close.
     ================================================================ -->
<div class="help-overlay" id="helpOverlay" onclick="closeHelpOnBackdrop(event)">
  <div class="help-modal" role="dialog" aria-modal="true" aria-labelledby="helpTitle">

    <!-- Modal Header -->
    <div class="help-header">
      <h2 id="helpTitle">✨ Annotation Tool — User Manual</h2>
      <button class="help-close" onclick="closeHelp()" title="Close manual">✕</button>
    </div>

    <!-- Modal Body — all the good stuff! -->
    <div class="help-body">

      <!-- ── SECTION 1: Quick Start ── -->
      <div class="help-section">
        <h3>🚀 Quick Start — The 3-Step Routine</h3>
        <p>Annotating a resume is a 3-step dance. Master these and you'll be flying through the queue!</p>
        <ol class="help-steps">
          <li><div><strong>Pick your entity label</strong> from the palette on the right panel — click a coloured button (e.g. <em>Person Name</em>, <em>Email Address</em>) OR press its keyboard shortcut number. The active label will glow with a white border.</div></li>
          <li><div><strong>Highlight the text</strong> in the resume on the left — click and drag across the words you want to label. Release the mouse and the annotation appears as a coloured highlight instantly.</div></li>
          <li><div><strong>Save your work</strong> — press <span class="kbd">Ctrl S</span> to save a draft anytime, or click <em>✅ Mark Complete</em> when the resume is fully annotated and ready for export.</div></li>
        </ol>
      </div>

      <!-- ── SECTION 2: Highlighting Tips ── -->
      <div class="help-section">
        <h3>🖱️ Highlighting — Tips for Accurate Selection</h3>
        <div class="help-tip tip-info">
          💡 <span>Always <strong>select an entity label first</strong>, THEN drag to highlight. Dragging without an active label does nothing — the text stays unselected.</span>
        </div>
        <div class="help-tip tip-warn">
          ⚠️ <span><strong>Avoid dragging into coloured label chips</strong> — the tiny floating tags above highlighted text (like "PERSON", "EMAIL"). If your selection clips one, a warning toast appears. Just try again, starting your drag slightly lower or to the side of the chip.</span>
        </div>
        <div class="help-tip tip-warn">
          ⚠️ <span><strong>Overlapping annotations</strong> on the same layer are not allowed. If you try to highlight text that's already labelled, you'll get a warning. Delete the existing annotation first, then re-label.</span>
        </div>
        <div class="help-tip tip-success">
          ✅ <span>The tool <strong>automatically trims</strong> leading and trailing whitespace from your selection — no need to be pixel-perfect at the word edges.</span>
        </div>
        <div class="help-tip tip-success">
          ✅ <span>Clicking any coloured highlight in the text panel will <strong>jump to that annotation</strong> in the Labels list tab on the right.</span>
        </div>
      </div>

      <!-- ── SECTION 3: Keyboard Shortcuts ── -->
      <div class="help-section">
        <h3>⌨️ Keyboard Shortcuts</h3>
        <p>Speed up your workflow — no mouse needed for most actions!</p>
        <div class="shortcut-grid" id="shortcutGrid">
          <!-- Static shortcuts always available -->
          <div class="shortcut-item"><span class="kbd">Ctrl S</span> Save draft</div>
          <div class="shortcut-item"><span class="kbd">Del</span> Delete last annotation</div>
          <div class="shortcut-item"><span class="kbd">Esc</span> Deselect active label</div>
          <!-- Entity shortcut keys are injected dynamically by buildShortcutGrid() on init -->
        </div>
      </div>

      <!-- ── SECTION 4: The Side Panel Tabs ── -->
      <div class="help-section">
        <h3>🗂️ Side Panel — What Each Section Does</h3>
        <ol class="help-steps">
          <li><div><strong>🎨 Entity Palette</strong> (top, always visible) — Your label buttons, organised by category: Personal, Professional, Education, Skills. Buttons with a keyboard shortcut show the key hint on the right side of the button.</div></li>
          <li><div><strong>📋 Labels tab</strong> — Lists every annotation you've added. Each item shows the entity type, a preview of the labelled text, and a <em>✕</em> delete button. The badge on the tab shows the total count.</div></li>
          <li><div><strong>🏷️ BIO tab</strong> — Previews the BIO-encoded training format (B-ENTITY = beginning of entity, I-ENTITY = inside entity, O = outside / no label). Hit <em>🔄 Refresh</em> after adding annotations. This is exactly what your NER model sees during training!</div></li>
          <li><div><strong>🧩 Edge tab</strong> — Scans the resume for tricky annotation patterns like date ranges containing "Present" or "Organisation, Location" combos that need splitting. Hit <em>🔍 Scan</em> to run the check and see hints.</div></li>
          <li><div><strong>🧠 Classify tab</strong> — Shows automatically predicted Function and Industry based on the resume text. You can correct them using the dropdowns and save. These values will appear in the exported talent database profiles.</div></li>
        </ol>
      </div>

      <!-- ── SECTION 5: Toolbar Buttons ── -->
      <div class="help-section">
        <h3>🔧 Toolbar Buttons Explained</h3>
        <ol class="help-steps">
          <li><div><strong>🤖 Pre-Annotate</strong> — Runs the auto-labeller with a confidence threshold you choose (0.0–1.0, default 0.6). It adds suggested labels without overwriting your existing human annotations.</div></li>
          <li><div><strong>🧩 Edge Cases</strong> — Jumps to the Edge tab and scans the text for commonly mis-annotated patterns.</div></li>
          <li><div><strong>🏷️ BIO Preview</strong> — Jumps to the BIO tab and loads the current token-tag breakdown.</div></li>
          <li><div><strong>🗑️ Clear</strong> — Removes ALL annotations after a confirmation prompt. This cannot be undone!</div></li>
          <li><div><strong>💾 Save Draft</strong> — Saves with status <em>in_progress</em>. Come back and continue later.</div></li>
          <li><div><strong>✅ Mark Complete</strong> — Saves with status <em>completed</em>. Only completed resumes are included in training data exports.</div></li>
        </ol>
      </div>

      <!-- ── SECTION 6: Edge Cases Guide ── -->
      <div class="help-section">
        <h3>🧩 Common Edge Cases & How to Handle Them</h3>
        <div class="help-tip tip-warn">
          ⚠️ <span><strong>Date ranges with "Present" or "Current"</strong> — e.g. <em>"Jan 2020 – Present"</em>. The word "Present" MUST be included inside the <em>WORK_DATE</em> span. Do NOT stop your selection before it.</span>
        </div>
        <div class="help-tip tip-warn">
          ⚠️ <span><strong>Organisation + Location on the same line</strong> — e.g. <em>"DBS Bank, Singapore"</em>. Create TWO separate annotations: <em>ORGANIZATION</em> for "DBS Bank" and <em>LOCATION</em> for "Singapore". Do NOT label the full line as one entity.</span>
        </div>
        <div class="help-tip tip-warn">
          ⚠️ <span><strong>Combined Gender/Age field</strong> — e.g. <em>"Female/22"</em> or <em>"Male / 35"</em>. Label the <strong>entire combined string</strong> as <em>GENDER</em> — do NOT split gender and age into separate spans. The age is considered part of that single field value.</span>
        </div>
        <div class="help-tip tip-info">
          💡 <span><strong>Marital Status field</strong> — e.g. <em>"Single"</em>, <em>"Married"</em>, <em>"Divorced"</em>. Label only the status VALUE as <em>MARITAL_STATUS</em>, not the field label ("Status:") itself.</span>
        </div>
        <div class="help-tip tip-info">
          💡 <span><strong>Qualifications with honours</strong> — e.g. <em>"B.Eng (Hons)"</em>. Include the full string including the honours notation inside the <em>DEGREE</em> span.</span>
        </div>
        <div class="help-tip tip-info">
          💡 <span><strong>Multiple phone numbers</strong> — label each number as a separate <em>PHONE</em> annotation, not both in one span.</span>
        </div>
        <div class="help-tip tip-info">
          💡 <span><strong>NRIC/Passport numbers</strong> — always label the full alphanumeric string including any trailing letter (e.g. "S9243711F").</span>
        </div>
      </div>

      <!-- ── SECTION 7: Export ── -->
      <div class="help-section">
        <h3>📤 Exporting Data</h3>
        <p>Two types of exports are available from the main queue page (← Back):</p>
        <ol class="help-steps">
          <li><div><strong>📊 Export CSV</strong> — Talent database profile in Excel-compatible CSV. One row per candidate. Nested fields (Work Experience, Education, tags/skills) are stored as JSON inside their cells. Open in Excel and use <em>Data → Text to Columns</em> or Power Query to expand them.</div></li>
          <li><div><strong>📋 Export JSON</strong> — Same talent database profile but as a clean JSON array. One object per candidate with all structured fields. Ready to import into your CRM or ATS system.</div></li>
          <li><div><strong>🤖 Export Training Data</strong> — NER model training formats: CoNLL-2003, spaCy v3 JSON, HuggingFace JSONL, or full metadata JSON. Only for ML/NLP use — not for the talent database.</div></li>
        </ol>
        <div class="help-tip tip-info">
          💡 <span>Profile exports (CSV/JSON) include all annotated fields: Name, Phone, Email, Current Company, Current Title, Current Location, <strong>Expected Location</strong>, Gender, <strong>Function</strong>, <strong>Industry</strong>, Summary, Language Skills, Work Experience, Education, <strong>Project Experience</strong>, and skill <strong>tags</strong>. Only <em>Team</em>, <em>Last Contact</em>, and <em>Created By</em> require manual input in your talent database.</span>
        </div>
        <div class="help-tip tip-info">
          💡 <span>Only resumes with status <em>completed</em> are included by default. You can choose <em>in_progress</em> or <em>all</em> when prompted.</span>
        </div>
      </div>

    </div><!-- /help-body -->
  </div><!-- /help-modal -->
</div><!-- /help-overlay -->

<!-- ================================================================
     🪞 SAVE PREVIEW MODAL — Quality check before committing!
     "A queen inspects her outfit before the ball!" 👑
     ================================================================ -->
<div class="preview-overlay" id="previewOverlay" onclick="closePreviewOnBackdrop(event)">
  <div class="preview-modal" role="dialog" aria-modal="true" aria-labelledby="previewTitle">

    <!-- Modal Header -->
    <div class="preview-header">
      <h2 id="previewTitle" style="color: var(--accent-cyan);">
        🪞 Save Preview — <span id="previewStatusLabel">Draft</span>
      </h2>
      <button class="preview-close" onclick="closePreview()" title="Cancel (Escape)">✕</button>
    </div>

    <!-- Modal Body — populated by JS -->
    <div class="preview-body" id="previewBody">
      <div class="preview-loading">
        <div class="spinner"></div>
        Validating annotations... ✨
      </div>
    </div>

    <!-- Modal Footer — Confirm / Cancel -->
    <div class="preview-footer">
      <div class="preview-footer-info" id="previewFooterInfo">
        Press <strong>Enter</strong> to confirm, <strong>Escape</strong> to cancel
      </div>
      <div class="preview-footer-actions">
        <button class="btn" onclick="closePreview()">← Go Back</button>
        <button class="btn btn-primary" id="previewConfirmBtn" onclick="confirmSave()">
          💾 Confirm Save
        </button>
      </div>
    </div>

  </div>
</div>

<!-- ================================================================
     📋 LABELS INSPECTOR MODAL — Full annotation list popup
     ================================================================ -->
<div class="inspector-overlay" id="labelsOverlay" onclick="if(event.target===this)closeInspector('labels')">
  <div class="inspector-modal" role="dialog" aria-modal="true">
    <div class="inspector-header">
      <h3>📋 All Annotations <span class="tab-badge" id="labelsModalBadge">0</span></h3>
      <button class="preview-close" onclick="closeInspector('labels')" title="Close (Escape)">✕</button>
    </div>
    <div class="inspector-body" id="labelsModalBody">
      <div class="anno-list-empty">Loading…</div>
    </div>
  </div>
</div>

<!-- ================================================================
     🏷️ BIO PREVIEW INSPECTOR MODAL
     ================================================================ -->
<div class="inspector-overlay" id="bioOverlay" onclick="if(event.target===this)closeInspector('bio')">
  <div class="inspector-modal" role="dialog" aria-modal="true">
    <div class="inspector-header">
      <h3>🏷️ BIO Tag Preview</h3>
      <button class="preview-close" onclick="closeInspector('bio')" title="Close (Escape)">✕</button>
    </div>
    <div class="inspector-body">
      <button class="btn btn-sm" onclick="loadBIOPreview()" style="margin-bottom:10px; width:100%;">
          🔄 Refresh BIO Preview
      </button>
      <div style="margin-bottom:8px; font-size:0.72rem; color:var(--text-muted);" id="bioStats">
          Click Refresh to generate BIO tag preview.
      </div>
      <div class="bio-preview" id="bioPreview">
          <div style="color:var(--text-muted); font-size:0.78rem; text-align:center; padding:20px 0;">
              Press Refresh to load tags. ✨
          </div>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================
     🧩 EDGE CASES INSPECTOR MODAL
     ================================================================ -->
<div class="inspector-overlay" id="edgeOverlay" onclick="if(event.target===this)closeInspector('edge')">
  <div class="inspector-modal" role="dialog" aria-modal="true">
    <div class="inspector-header">
      <h3>🧩 Edge Case Analysis</h3>
      <button class="preview-close" onclick="closeInspector('edge')" title="Close (Escape)">✕</button>
    </div>
    <div class="inspector-body">
      <button class="btn btn-sm" onclick="loadEdgeCases()" style="margin-bottom:10px; width:100%;">
          🔍 Scan for Edge Cases
      </button>
      <div id="edgeHints">
          <div style="color:var(--text-muted); font-size:0.78rem; text-align:center; padding:20px 0;">
              Click Scan to detect tricky annotation patterns. 🧩
          </div>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================
     🧠 CLASSIFICATION & SKILLS MODAL
     
     The grand Classification popup! Contains Function, Industry,
     Hard Skills, Soft Skills, and AI-generated Tags. 
     Opens via the 🧠 Classify toolbar button or sidebar button.
     
     Uses the inspector-overlay pattern (same as Labels/BIO/Edge)
     but wider to fit the skills chip layout. Think of it as 
     opening a full talent dossier — the whole story on one page! 💅✨
     ================================================================ -->
<div class="inspector-overlay" id="classifyOverlay" onclick="if(event.target===this)closeInspector('classify')">
  <div class="inspector-modal classify-modal-wide" role="dialog" aria-modal="true">
    <div class="inspector-header" style="border-bottom-color:var(--accent-purple);">
      <h3>🧠 Classification & Skills</h3>
      <button class="preview-close" onclick="closeInspector('classify')" title="Close (Escape)">✕</button>
    </div>
    <div class="inspector-body" style="padding:20px 24px;">

      <!-- Function + Industry dropdowns — two-column -->
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px;">
        <div>
          <label style="font-size:0.7rem; color:var(--text-muted); display:block; margin-bottom:4px; font-weight:600;">
            🧠 Function / Department
          </label>
          <select id="functionSelect" class="classify-select">
            <!-- options populated by populateClassification() in JS -->
          </select>
        </div>
        <div>
          <label style="font-size:0.7rem; color:var(--text-muted); display:block; margin-bottom:4px; font-weight:600;">
            🏢 Industry Sector
          </label>
          <select id="industrySelect" class="classify-select">
            <!-- options populated by populateClassification() in JS -->
          </select>
        </div>
      </div>

      <!-- Hard Skills -->
      <div style="margin-bottom:14px;">
        <label style="font-size:0.7rem; color:var(--accent-cyan); display:block; margin-bottom:4px; font-weight:600;">
          🔧 Hard Skills <small style="opacity:0.5; color:var(--text-muted);">(technical skills from resume)</small>
        </label>
        <div id="hardSkillsContainer" class="skill-tags-container"></div>
        <div style="display:flex; gap:4px; margin-top:5px;">
          <input id="hardSkillInput" type="text" class="skill-tag-input"
                 placeholder="Add hard skill…" 
                 onkeydown="if(event.key==='Enter'){addSkillTag('hard');event.preventDefault();}">
          <button class="btn btn-sm" onclick="addSkillTag('hard')" 
                  style="font-size:0.65rem; padding:3px 10px; flex-shrink:0;">+ Add</button>
        </div>
      </div>

      <!-- Soft Skills -->
      <div style="margin-bottom:14px;">
        <label style="font-size:0.7rem; color:var(--accent-purple); display:block; margin-bottom:4px; font-weight:600;">
          💬 Soft Skills <small style="opacity:0.5; color:var(--text-muted);">(interpersonal skills from resume)</small>
        </label>
        <div id="softSkillsContainer" class="skill-tags-container"></div>
        <div style="display:flex; gap:4px; margin-top:5px;">
          <input id="softSkillInput" type="text" class="skill-tag-input"
                 placeholder="Add soft skill…"
                 onkeydown="if(event.key==='Enter'){addSkillTag('soft');event.preventDefault();}">
          <button class="btn btn-sm" onclick="addSkillTag('soft')" 
                  style="font-size:0.65rem; padding:3px 10px; flex-shrink:0;">+ Add</button>
        </div>
      </div>

      <!-- Tags -->
      <div style="margin-bottom:16px;">
        <label style="font-size:0.7rem; color:var(--accent-green); display:block; margin-bottom:4px; font-weight:600;">
          🏷️ Tags <small style="opacity:0.5; color:var(--text-muted);">(customizable keywords for organizing, filtering &amp; searching)</small>
        </label>

        <!-- Active tag chips -->
        <div id="tagsContainer" class="skill-tags-container tags-ai"></div>

        <!-- Manual input row -->
        <div style="display:flex; gap:4px; margin-top:5px;">
          <input id="tagInput" type="text" class="skill-tag-input"
                 placeholder="Type a custom tag…"
                 onkeydown="if(event.key==='Enter'){addSkillTag('tag');event.preventDefault();}">
          <button class="btn btn-sm" onclick="addSkillTag('tag')"
                  style="font-size:0.65rem; padding:3px 10px; flex-shrink:0;">+ Add</button>
          <button class="btn btn-sm" id="paletteToggleBtn"
                  onclick="toggleTagPalette()"
                  style="font-size:0.65rem; padding:3px 10px; flex-shrink:0; opacity:0.75;"
                  title="Quick-pick common tags">⚡ Quick Pick</button>
        </div>

        <!-- ── Quick-pick palette ──────────────────────────────── -->
        <div id="tagPalette" style="display:none; margin-top:10px;
             border:1px solid rgba(86,211,100,0.2); border-radius:8px;
             padding:10px 12px; background:rgba(86,211,100,0.04);">

          <div style="font-size:0.62rem; color:var(--text-muted);
               margin-bottom:8px; letter-spacing:0.04em; text-transform:uppercase;">
            Click to add · Click again to remove
          </div>

          <div id="tagPaletteGroups"></div>
        </div>
      </div>

      <!-- Action row + Status -->
      <div style="display:flex; gap:8px; align-items:center;">
        <button class="btn btn-sm" onclick="reclassify()" style="font-size:0.72rem; padding:6px 16px;">
          🔄 Re‑run AI Classification
        </button>
        <button class="btn btn-sm btn-primary" onclick="saveClassification()" style="font-size:0.72rem; padding:6px 16px;">
          💾 Save Classification
        </button>
        <div id="classifyStatus" style="font-size:0.65rem; color:var(--text-muted); margin-left:8px;"></div>
      </div>

    </div>
  </div>
</div>

<!-- ================================================================
     📏 IAA DASHBOARD MODAL
     
     The grand agreement showdown! Shows Cohen's Kappa, F1 scores,
     and a gorgeous heatmap of per-entity agreement. Like a talent
     show scoreboard where every judge's vote is visible! 🏆👩‍⚖️👨‍⚖️
     ================================================================ -->
<div class="inspector-overlay" id="iaaOverlay" onclick="if(event.target===this)closeInspector('iaa')">
  <div class="inspector-modal iaa-modal-wide" role="dialog" aria-modal="true">
    <div class="inspector-header" style="border-bottom-color:var(--accent-pink);">
      <h3>📏 Inter-Annotator Agreement</h3>
      <div style="display:flex; gap:8px; align-items:center;">
        <!-- Toggle: show this doc only vs. all docs -->
        <button class="btn btn-sm" id="iaaToggleScope" onclick="toggleIAAScope()"
                style="font-size:0.65rem; padding:3px 10px;">
            🔍 This Doc
        </button>
        <button class="preview-close" onclick="closeInspector('iaa')" title="Close (Escape)">✕</button>
      </div>
    </div>
    <div class="inspector-body" id="iaaModalBody" style="padding:20px 24px;">
      <!-- Content injected by renderIAADashboard() -->
      <div style="text-align:center; padding:40px; color:var(--text-muted);">
        Loading IAA data... ⏳
      </div>
    </div>
  </div>
</div>

<!-- ================================================================
     👤 ANNOTATOR NAME PROMPT
     
     Shows on first visit (if no annotator name in localStorage).
     Like the sign-in sheet at the stage door — you can't go on
     without putting your name down, darling! 🎭✍️
     ================================================================ -->
<div class="annotator-prompt-overlay" id="annotatorPromptOverlay">
  <div class="annotator-prompt-card">
    <div style="font-size:2.2rem; margin-bottom:8px;">🎭</div>
    <h3>Welcome, Annotator!</h3>
    <p>
      Enter your name so we can track who annotated what.
      This enables <strong>Inter-Annotator Agreement</strong> (IAA)
      — comparing annotations across different people! 📏
    </p>
    <input type="text" id="annotatorNameInput" placeholder="Your name (e.g. Soraya, Ahmad, etc.)"
           maxlength="50" autocomplete="off"
           onkeydown="if(event.key==='Enter')confirmAnnotatorName()">
    <div style="display:flex; gap:8px; justify-content:center;">
      <button class="btn btn-sm btn-primary" onclick="confirmAnnotatorName()"
              style="padding:8px 24px; font-size:0.8rem;">
        ✨ Let's Go!
      </button>
    </div>
    <div id="annotatorNameError" style="color:var(--accent-red); font-size:0.7rem; margin-top:8px; display:none;"></div>
  </div>
</div>

<script>

const DOC_ID = {{ doc_id | tojson }};

// =================================================================
// 🗂️ TAB SWITCHING — Navigate between side panel sections
// =================================================================
/**
 * switchTab(tabId, clickedBtn)
 * 
 * LEGACY — kept for backward compat but now mostly a no-op.
 * The old tab system is replaced by inspector modals! 🎭
 */
function switchTab(tabId, clickedBtn) {
    // Redirect to the new modal system
    if (tabId === 'bio') openBIOModal();
    else if (tabId === 'edge') openEdgeModal();
    else if (tabId === 'annotations') openLabelsModal();
}

// =================================================================
// 🔍 INSPECTOR MODALS — Labels / BIO / Edge popups
// =================================================================

/** Currently open inspector (for Escape key routing) */
let activeInspector = null;

/**
 * openLabelsModal(focusIdx)
 * 
 * Opens a popup showing ALL annotations in a scrollable list.
 * Like unrolling the FULL guest list at the gala! 📜✨
 *
 * @param {number|null} focusIdx - Optional index of the annotation to
 *   scroll to and highlight. Pass null (or omit) to just open the list.
 */
function openLabelsModal(focusIdx = null) {
    const body = document.getElementById('labelsModalBody');
    const badge = document.getElementById('labelsModalBadge');
    if (badge) badge.textContent = annotations.length;

    if (!annotations.length) {
        body.innerHTML = '<div class="anno-list-empty">No annotations yet. Select a label above, then highlight text! ✨</div>';
    } else {
        // Render every annotation as a list row.
        // data-idx stamps the array index on the DOM so focusAnnotation()
        // can find the right row with a single querySelector. 🎯
        body.innerHTML = annotations.map((a, i) => {
            const color = COLOR_MAP[a.entity_type] || '#888';
            const label = SCHEMA.entities[a.entity_type]?.label || a.entity_type;
            const text = (a.text || '').substring(0, 80);
            const autoIcon = a.annotator?.startsWith('auto') ? ' 🤖' : ' 👤';
            return `<div class="anno-list-item" data-idx="${i}">
                <span class="anno-type" style="background:${color}20; color:${color};">${label}${autoIcon}</span>
                <span class="anno-text">"${escapeHtml(text)}"</span>
                <button class="delete-btn" onclick="deleteAnnotation(${i}); openLabelsModal();" title="Delete">✕</button>
            </div>`;
        }).join('');
    }

    // Open the inspector overlay first so the DOM is visible
    openInspector('labels');

    // If a specific annotation was requested, scroll to it and flash it.
    // requestAnimationFrame defers until after the browser has painted the
    // newly opened modal — without this, scrollIntoView is a no-op because
    // the element isn't visible yet. Timing is everything, darling! ⏱️✨
    if (focusIdx !== null) {
        requestAnimationFrame(() => {
            const target = body.querySelector(`[data-idx="${focusIdx}"]`);
            if (!target) return;

            // Remove any previous focus highlight before adding the new one
            // (handles the case where modal was already open) 💅
            body.querySelectorAll('.anno-list-item--focused')
                .forEach(el => el.classList.remove('anno-list-item--focused'));

            // Scroll the row into the centre of the modal body
            target.scrollIntoView({ behavior: 'smooth', block: 'center' });

            // Add the pulse class — CSS animation fires automatically
            target.classList.add('anno-list-item--focused');
        });
    }
}

/**
 * openBIOModal()
 * 
 * Opens the BIO tag preview popup and auto-loads data.
 * Behind the curtain at the stage machinery! ⚙️
 */
function openBIOModal() {
    openInspector('bio');
    // Auto-load if there are annotations
    if (annotations.length > 0) {
        loadBIOPreview();
    }
}

/**
 * openEdgeModal()
 * 
 * Opens the edge case inspector popup and auto-scans.
 * The divas of annotation need SPECIAL attention! 💅
 */
function openEdgeModal() {
    openInspector('edge');
    loadEdgeCases();
}

/**
 * openInspector(name)
 * Generic helper — opens the named inspector overlay.
 */
function openInspector(name) {
    // Close any already-open inspector first
    if (activeInspector) closeInspector(activeInspector);

    const overlay = document.getElementById(name + 'Overlay');
    if (overlay) overlay.classList.add('open');
    activeInspector = name;
    document.addEventListener('keydown', inspectorEscapeHandler);
}

/**
 * closeInspector(name)
 * Generic helper — closes the named inspector overlay.
 */
function closeInspector(name) {
    const overlay = document.getElementById(name + 'Overlay');
    if (overlay) overlay.classList.remove('open');
    activeInspector = null;
    document.removeEventListener('keydown', inspectorEscapeHandler);
}

/**
 * inspectorEscapeHandler(event)
 * Escape key closes the active inspector modal.
 */
function inspectorEscapeHandler(event) {
    if (event.key === 'Escape' && activeInspector) {
        event.preventDefault();
        closeInspector(activeInspector);
    }
}

// =================================================================
// 🧠 STATE
// =================================================================
const CANDIDATE_ID = {{ candidate_id }};
const RAW_TEXT = {{ raw_text | tojson }};
const SCHEMA = {{ schema_json | safe }};
const COLOR_MAP = {{ color_map_json | safe }};
let annotations = {{ annotations_json | safe }};
let activeEntityType = null;

// =================================================================
// 🎨 ENTITY PALETTE — Build the button grid
// =================================================================
function buildPalette() {
    const palette = document.getElementById('entityPalette');
    palette.innerHTML = '';

    const categories = {};
    for (const [name, info] of Object.entries(SCHEMA.entities)) {
        const cat = info.category;
        if (!categories[cat]) categories[cat] = [];
        categories[cat].push({name, ...info});
    }

    for (const [cat, entities] of Object.entries(categories)) {
        const catLabel = document.createElement('div');
        catLabel.style.cssText = 'font-size:0.62rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; margin-top:8px; margin-bottom:4px;';
        catLabel.textContent = cat;
        palette.appendChild(catLabel);

        for (const e of entities) {
            const btn = document.createElement('button');
            btn.className = 'entity-btn';
            btn.dataset.entity = e.name;
            btn.style.background = e.color + '20';
            btn.style.color = e.color;
            btn.innerHTML = e.label + (e.shortcut_key ?
                ` <span class="key-hint">${e.shortcut_key}</span>` : '');
            btn.title = e.description.substring(0, 120) + '...';
            btn.onclick = () => selectEntity(e.name);
            palette.appendChild(btn);
        }
    }
}

function selectEntity(name) {
    activeEntityType = (activeEntityType === name) ? null : name;
    document.querySelectorAll('.entity-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.entity === activeEntityType);
    });
}

// =================================================================
// 📝 TEXT RENDERING — Display text with inline annotations
// =================================================================
function renderText() {
    const container = document.getElementById('textContent');

    // ── Separate positioned vs. manual/edited annotations ────────
    // Positioned: have valid char_start/char_end → render inline in RAW_TEXT
    // Manual/Edited: char_start < 0 → can't map to RAW_TEXT, show separately
    //
    // 🐛 BUG FIX: Previously ALL annotations went through
    // RAW_TEXT.substring(char_start, char_end), which meant:
    //   - Manual entries (char_start=-1) produced empty text
    //   - Edited entries still showed OLD text from the original position
    // Now manual/edited annotations render using their .text property! ✨
    const positioned = [];
    const manualEdited = [];

    for (const ann of annotations) {
        if (ann.char_start >= 0 && ann.char_end > ann.char_start) {
            positioned.push(ann);
        } else {
            manualEdited.push(ann);
        }
    }

    // Sort positioned annotations by start position
    const sorted = [...positioned].sort((a, b) => a.char_start - b.char_start);

    let html = '';
    let pos = 0;

    for (const ann of sorted) {
        // Text before this annotation
        if (ann.char_start > pos) {
            html += escapeHtml(RAW_TEXT.substring(pos, ann.char_start));
        }

        // The annotation span — uses RAW_TEXT slice for positioned annotations
        const color = COLOR_MAP[ann.entity_type] || '#888';
        const spanText = RAW_TEXT.substring(ann.char_start, ann.char_end);
        const label = SCHEMA.entities[ann.entity_type]?.label || ann.entity_type;
        const autoTag = ann.annotator?.startsWith('auto') ? ' 🤖' : '';

        html += `<span class="anno-span" `
            + `style="background:${color}20; border-color:${color}; color:${color};" `
            + `data-idx="${annotations.indexOf(ann)}" `
            + `onclick="focusAnnotation(${annotations.indexOf(ann)})" `
            + `title="${label}: ${escapeHtml(spanText)}">`
            + `<span class="anno-label" style="background:${color}; color:#000;">${label}${autoTag}</span>`
            + escapeHtml(spanText)
            + `</span>`;

        pos = ann.char_end;
    }

    // Remaining text
    if (pos < RAW_TEXT.length) {
        html += escapeHtml(RAW_TEXT.substring(pos));
    }

    // ── Manual / Edited annotations section ──────────────────────────
    // These are annotations created or edited via Save Preview that
    // don't have valid char positions in RAW_TEXT. We show them in a
    // dedicated block below the main text so nothing is invisible! 💅
    if (manualEdited.length > 0) {
        html += `<div style="margin-top:16px; padding-top:12px; border-top:1px dashed var(--border-color);">`;
        html += `<div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:8px;">`;
        html += `✏️ Manually added/edited entries (${manualEdited.length}):`;
        html += `</div>`;
        for (const ann of manualEdited) {
            const color = COLOR_MAP[ann.entity_type] || '#888';
            const label = SCHEMA.entities[ann.entity_type]?.label || ann.entity_type;
            const displayText = ann.text || '(empty)';
            html += `<span class="anno-span" `
                + `style="background:${color}20; border-color:${color}; color:${color}; margin:2px 4px 2px 0; display:inline-block;" `
                + `data-idx="${annotations.indexOf(ann)}" `
                + `onclick="focusAnnotation(${annotations.indexOf(ann)})" `
                + `title="${label}: ${escapeHtml(displayText)}">`
                + `<span class="anno-label" style="background:${color}; color:#000;">${label} ✏️</span>`
                + escapeHtml(displayText)
                + `</span> `;
        }
        html += `</div>`;
    }

    container.innerHTML = html;
    updateAnnoList();
}

/**
 * escapeHtml(text)
 *
 * Escapes HTML special characters so text can be safely injected
 * into BOTH innerHTML AND attribute values (like value="...").
 *
 * 🐛 BUG FIX: The old version used textContent→innerHTML which
 * only escapes <, >, and &. It did NOT escape double quotes!
 * When annotation text contained " characters (e.g. "Big Corp"),
 * the preview form's value="..." attributes would BREAK, causing:
 *   - Input values getting truncated at the first quote
 *   - confirmSave() reading corrupted data
 *   - Silent JS errors preventing saveAnnotations() from running
 *
 * Think of it like writing your name on a form but the pen
 * explodes at the quotation marks — the rest is just a smear! 🖊️💥
 *
 * Now we also escape " → &quot; and ' → &#39; so the text is
 * safe EVERYWHERE in HTML — tags, attributes, you name it! 💅
 *
 * @param {string} text - Raw text to escape
 * @returns {string}    - HTML-safe escaped string
 */
function escapeHtml(text) {
    if (text == null) return '';
    const str = String(text);
    const div = document.createElement('div');
    div.textContent = str;
    // innerHTML escapes <, >, & — but NOT quotes!
    // We add quote escaping for safe attribute injection.
    return div.innerHTML
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// =================================================================
// ✏️ ANNOTATION CREATION — Select text to create annotation
// =================================================================
document.getElementById('textPanel').addEventListener('mouseup', (e) => {
    if (!activeEntityType) return;

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const range = selection.getRangeAt(0);
    const container = document.getElementById('textContent');

    // ── Guard: if the selection boundary is inside an .anno-label chip
    //    (the floating coloured label above a span), the user accidentally
    //    dragged into a UI element that isn't raw text. Abort gracefully.
    const startEl = range.startContainer.parentElement;
    const endEl   = range.endContainer.parentElement;
    if (
        startEl?.classList.contains('anno-label') ||
        endEl?.classList.contains('anno-label')
    ) {
        selection.removeAllRanges();
        showToast('⚠️ Selection clipped a label chip — try again!', 'warning');
        return;
    }

    // ── Guard: selection must be fully inside the text content container
    if (!container.contains(range.commonAncestorContainer)) {
        selection.removeAllRanges();
        return;
    }

    // Calculate character offsets in the raw text
    // We need to map DOM positions back to raw text positions
    const charStart = getTextOffset(container, range.startContainer, range.startOffset);
    const charEnd   = getTextOffset(container, range.endContainer,   range.endOffset);

    

    if (charStart === null || charEnd === null || charStart >= charEnd) {
        selection.removeAllRanges();
        return;
    }

    const selectedText = RAW_TEXT.substring(charStart, charEnd).trim();
    if (!selectedText) {
        selection.removeAllRanges();
        return;
    }

    // Adjust charStart/charEnd inward to strip any leading/trailing whitespace
    // so the highlight hugs the actual word, not trailing spaces
    const rawSlice  = RAW_TEXT.substring(charStart, charEnd);
    const trimStart = charStart + (rawSlice.length - rawSlice.trimStart().length);
    const trimEnd   = charEnd   - (rawSlice.length - rawSlice.trimEnd().length);

    // ══════════════════════════════════════════════════════════════════
    // 🧬 NESTED ENTITY HANDLING — The Fabulous Layer Cake! 🎂
    // ══════════════════════════════════════════════════════════════════
    // Check if this span overlaps an existing annotation.
    // If the parent allows nesting for this entity type, use layer 1!
    // Otherwise, block the overlap as before.
    
    let targetLayer = 0;
    const overlappingParent = annotations.find(a =>
        a.layer === 0 &&
        !(trimEnd <= a.char_start || trimStart >= a.char_end)
    );
    
    if (overlappingParent) {
        // Check if parent entity allows this child to nest inside
        const parentDef = SCHEMA.entities[overlappingParent.entity_type];
        const allowsNesting = parentDef?.allows_nesting === true;
        const allowedChildren = parentDef?.nested_children || [];
        
        if (allowsNesting && allowedChildren.includes(activeEntityType)) {
            // ✅ Nesting allowed! Use layer 1 for the child annotation
            targetLayer = 1;
            console.log(`🧬 Nesting ${activeEntityType} inside ${overlappingParent.entity_type} on layer 1`);
        } else {
            // ❌ Nesting NOT allowed for this combination
            showToast(
                `⚠️ Cannot nest ${activeEntityType} inside ${overlappingParent.entity_type}. ` +
                `Delete the parent annotation first, or this entity type is not allowed as a child.`,
                'warning'
            );
            selection.removeAllRanges();
            return;
        }
    }

    // Create the new annotation object (now with smart layer!)
    annotations.push({
        entity_type: activeEntityType,
        char_start:  trimStart,
        char_end:    trimEnd,
        text:        RAW_TEXT.substring(trimStart, trimEnd),
        layer:       targetLayer,  // 🆕 Dynamic layer based on nesting!
        confidence:  1.0,
        annotator:   'human'
    });

    // Mark IAA annotations as unsaved whenever a new label is added in IAA mode
    if (iaaMode) setIAASaveStatus('unsaved');

    selection.removeAllRanges();
    renderText();
    showToast(`✅ Added ${activeEntityType}: "${selectedText.substring(0, 40)}${selectedText.length > 40 ? '...' : ''}"`);
});

/**
 * getTextOffset(root, node, offset)
 *
 * 🐛 BUG FIX — THE GREAT OFFSET DRAMA OF 2026 🎭
 *
 * The original version used a TreeWalker that walked ALL text nodes,
 * including the text inside floating .anno-label chips (e.g. "PERSON",
 * "EMAIL"). Those labels are pure UI decoration injected by renderText()
 * — they are NOT part of the raw resume text at all!
 *
 * So every existing annotation label was silently adding extra characters
 * to the offset count, causing subsequent selections to land in the WRONG
 * position. Like a choreographer counting the stagehands' steps instead
 * of just the dancers'! 💃🕺
 *
 * THE FIX: we pass a custom NodeFilter that tells the TreeWalker to SKIP
 * any text node whose parent is an .anno-label element.
 *
 * @param {Element}  root   - The #textContent container element
 * @param {Node}     node   - The DOM node where the selection boundary lives
 * @param {number}   offset - Character offset within that node
 * @returns {number|null}   - Character offset in RAW_TEXT, or null if not found
 */
function getTextOffset(root, node, offset) {

    // ── Custom filter: skip .anno-label text so we don't count
    //    UI labels (e.g. "PERSON", "EMAIL") as raw text characters.
    const skipLabels = {
        acceptNode(n) {
            // If the text node's parent is an .anno-label chip, skip it entirely
            if (n.parentElement && n.parentElement.classList.contains('anno-label')) {
                return NodeFilter.FILTER_SKIP;
            }
            return NodeFilter.FILTER_ACCEPT;
        }
    };

    // Walk only REAL text nodes (skipping label nodes via the filter above)
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, skipLabels);
    let charCount = 0;

    while (walker.nextNode()) {
        if (walker.currentNode === node) {
            // Found the exact text node — return accumulated count + local offset
            return charCount + offset;
        }
        charCount += walker.currentNode.textContent.length;
    }

    // ── Fallback: handles the rare case where the selection boundary lands
    //    on an ELEMENT node rather than a text node (e.g. user clicks at the
    //    very end of an .anno-span). We walk sibling elements and sum only
    //    the "real" text (via innerText of non-label children).
    if (node.nodeType !== Node.TEXT_NODE) {
        const walker2 = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, skipLabels);
        let count = 0;

        while (walker2.nextNode()) {
            // Check if this text node is inside the target element node
            if (node.contains && node.contains(walker2.currentNode)) {
                return count + offset;
            }
            count += walker2.currentNode.textContent.length;
        }
        // Last-resort: return accumulated count (best guess)
        return count;
    }

    return null;  // Could not resolve — selection will be ignored gracefully
}

// =================================================================
// 📋 ANNOTATION LIST — Sidebar list with delete buttons
// =================================================================
function updateAnnoList() {
    // annoList may be absent if the sidebar section has been removed —
    // null-guard so we never crash when writing to it. 💅
    const list = document.getElementById('annoList');

    // Update badge counts everywhere they appear
    const badge = document.getElementById('annoCountBadge');
    if (badge) badge.textContent = annotations.length;

    // Labels badge in toolbar (mirrors annotation count)
    const headerBadge = document.getElementById('headerLabelsBadge');
    if (headerBadge) headerBadge.textContent = annotations.length;

    // 📋 Annotations badge next to Edge button in toolbar
    const headerAnnotationsBadge = document.getElementById('headerAnnotationsBadge');
    if (headerAnnotationsBadge) headerAnnotationsBadge.textContent = annotations.length;

    // If the sidebar list element doesn't exist, nothing more to do here.
    if (!list) return;

    if (!annotations.length) {
        // Empty state message — friendly and helpful
        list.innerHTML = '<div class="anno-list-empty">No annotations yet.<br>Select a label above, then drag to highlight text! ✨</div>';
        return;
    }

    // Render each annotation as a list item with delete button
    list.innerHTML = annotations.map((a, i) => {
        const color = COLOR_MAP[a.entity_type] || '#888';
        const label = SCHEMA.entities[a.entity_type]?.label || a.entity_type;
        // Truncate long text for display — 50 chars max
        const text = (a.text || '').substring(0, 50);
        const autoIcon = a.annotator?.startsWith('auto') ? ' 🤖' : '';

        return `<div class="anno-list-item">
            <span class="anno-type" style="background:${color}20; color:${color};">${label}${autoIcon}</span>
            <span class="anno-text">"${escapeHtml(text)}"</span>
            <button class="delete-btn" onclick="deleteAnnotation(${i})" title="Delete">✕</button>
        </div>`;
    }).join('');
}

function deleteAnnotation(idx) {
    annotations.splice(idx, 1);
    // Mark IAA annotations as unsaved whenever a label is removed in IAA mode
    if (iaaMode) setIAASaveStatus('unsaved');
    renderText();
}

function focusAnnotation(idx) {
    /*
     * 🔦 FOCUS ANNOTATION
     *
     * Called when the user clicks a coloured span in the resume text.
     * We want to show them WHERE that annotation lives in the Labels list —
     * like tapping someone on the shoulder and pointing to their name on
     * the guest list! 🎉
     *
     * Flow:
     *   1. Open the Labels inspector modal (re-renders the full list)
     *   2. openLabelsModal(idx) handles the scroll + pulse highlight
     *      via requestAnimationFrame so the DOM is painted first.
     */
    openLabelsModal(idx);
}

function clearAll() {
    if (!confirm('Remove ALL annotations? This cannot be undone.')) return;
    annotations = [];
    renderText();
    showToast('🗑️ All annotations cleared');
}

// =================================================================
// 🪞 SAVE PREVIEW — "Check your look before the runway!" 💅
// =================================================================

/** Tracks which status the user intends to save as */
let pendingSaveStatus = null;

/**
 * showSavePreview(status)
 * 
 * THE MAIN ENTRY POINT — replaces direct saveAnnotations() calls.
 * Opens a preview modal, fetches validation from the backend,
 * and lets the annotator review everything before committing.
 * 
 * Like a dress rehearsal before the big show! 🎭✨
 * 
 * @param {string} status - 'in_progress' or 'completed'
 */
function showSavePreview(status) {
    pendingSaveStatus = status;

    // Open the modal with loading state
    const overlay = document.getElementById('previewOverlay');
    const body = document.getElementById('previewBody');
    const statusLabel = document.getElementById('previewStatusLabel');
    const confirmBtn = document.getElementById('previewConfirmBtn');

    // Set header label based on target status
    statusLabel.textContent = status === 'completed' ? 'Mark Complete' : 'Draft';
    statusLabel.style.color = status === 'completed' 
        ? 'var(--accent-green)' : 'var(--accent-cyan)';

    // Reset confirm button
    confirmBtn.disabled = false;
    confirmBtn.textContent = status === 'completed' 
        ? '✅ Confirm Complete' : '💾 Confirm Save';
    confirmBtn.className = status === 'completed' 
        ? 'btn btn-success' : 'btn btn-primary';

    // Show loading spinner
    body.innerHTML = `
        <div class="preview-loading">
            <div class="spinner"></div>
            Validating ${annotations.length} annotation${annotations.length !== 1 ? 's' : ''}... ✨
        </div>
    `;

    overlay.classList.add('open');

    // Listen for keyboard shortcuts inside modal
    document.addEventListener('keydown', previewKeyHandler);

    // Fetch validation from backend
    fetch(`/api/preview-before-save/${CANDIDATE_ID}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            annotations: annotations,
            status: status
        })
    })
    .then(r => {
        if (!r.ok) throw new Error(`Server returned ${r.status}`);
        return r.json();
    })
    .then(data => {
        renderPreviewContent(data, status);
    })
    .catch(err => {
        body.innerHTML = `
            <div class="preview-loading" style="color: var(--accent-red);">
                <div style="font-size: 2rem; margin-bottom: 12px;">❌</div>
                <div>Failed to validate: ${escapeHtml(err.message)}</div>
                <div style="font-size: 0.72rem; margin-top: 8px; color: var(--text-muted);">
                    You can still save directly — close this and use Ctrl+Shift+S to bypass preview.
                </div>
            </div>
        `;
    });
}

/**
 * renderPreviewContent(data, status)
 *
 * 🪞 THE GRAND REVEAL — Completely rewritten!
 *
 * NOW shows EVERY entity type from the schema as an editable field,
 * whether or not it has been annotated. Empty fields show a dashed
 * "Add..." input so the annotator can fill them in right here.
 *
 * Layout:
 *   1. Stats banner (compact)
 *   2. Warnings (if any)
 *   3. Classification row (Function / Industry dropdowns)
 *   4. Full entity form — ALL fields, ALL categories, ALL editable
 *
 * @param {Object} data   - Response from /api/preview-before-save
 * @param {string} status - Target save status ('in_progress' | 'completed')
 */
function renderPreviewContent(data, status) {
    const body       = document.getElementById('previewBody');
    const confirmBtn = document.getElementById('previewConfirmBtn');
    const footerInfo = document.getElementById('previewFooterInfo');
    const s          = data.stats;
    let html         = '';

    // ── 1. STATS BANNER ──────────────────────────────────────────────
    // Compact row of KPI cards — IDs on total so pfMarkDelete() can update it live
    html += `<div class="preview-stats">
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-cyan);" id="prevStatTotal">${s.total_annotations}</div>
            <div class="stat-label">Total Labels</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-green);">${s.entity_type_count}</div>
            <div class="stat-label">Types Used</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-purple);">${s.human_count}</div>
            <div class="stat-label">Human 👤</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number" style="color:var(--accent-orange);">${s.auto_count}</div>
            <div class="stat-label">Auto 🤖</div>
        </div>
        <div class="preview-stat-card">
            <div class="stat-number"
                 style="color:${(s.critical_errors + s.warnings) > 0 ? 'var(--accent-red)' : 'var(--accent-green)'};">
                ${s.critical_errors + s.warnings}
            </div>
            <div class="stat-label">Issues</div>
        </div>
    </div>`;

    // ── 2. WARNINGS ──────────────────────────────────────────────────
    // Sort: critical first, then warning, then info
    if (data.warnings && data.warnings.length > 0) {
        html += `<div class="preview-warnings">`;
        const sortOrder = { critical: 0, warning: 1, info: 2 };
        const sortedWarnings = [...data.warnings].sort((a, b) =>
            (sortOrder[a.level] || 9) - (sortOrder[b.level] || 9)
        );
        for (const w of sortedWarnings) {
            html += `<div class="preview-warning-item ${escapeHtml(w.level)}">
                <span class="preview-warning-icon">${w.icon || '⚠️'}</span>
                <span>${escapeHtml(w.message)}</span>
            </div>`;
        }
        html += `</div>`;
    } else {
        html += `<div class="preview-all-clear">
            ✅ All checks passed — you're ready to save! 🎉
        </div>`;
    }

    // ── 3. BUILD EXISTING-ANNOTATION LOOKUP ──────────────────────────
    // Map: entity_type → [{text, idx}, ...]  from the backend summary.
    // This tells us which types already have annotations vs. which are empty.
    //
    // Like the guest list for the gala — some seats already filled,
    // some still open for walk-ins! 🎟️
    const annoByType = {};
    if (data.summary) {
        for (const entityList of Object.values(data.summary)) {
            for (const entity of entityList) {
                annoByType[entity.entity_type] = (entity.texts || []).map(item =>
                    typeof item === 'object'
                        ? { text: item.text, idx: item.idx }
                        : { text: item,      idx: -1 }
                );
            }
        }
    }

    // ── 4. CLASSIFICATION (Function / Industry / Skills / Tags) ─────────
    // These two get dropdown selects, not plain text inputs,
    // because they map to a controlled vocabulary list. 💅
    // 🆕 Skills & Tags shown as read-only chip displays in preview.
    const cls      = data.classification || {};
    const funcVal  = cls.function || '';
    const indVal   = cls.industry  || '';

    const funcOptions = FUNCTION_OPTIONS.map(f =>
        `<option value="${escapeHtml(f)}" ${f === funcVal ? 'selected' : ''}>${escapeHtml(f)}</option>`
    ).join('');
    const indOptions = INDUSTRY_OPTIONS.map(i =>
        `<option value="${escapeHtml(i)}" ${i === indVal ? 'selected' : ''}>${escapeHtml(i)}</option>`
    ).join('');

    // 🆕 Build chip HTML for skills & tags in preview
    const previewHardSkills = cls.hard_skills || currentHardSkills || [];
    const previewSoftSkills = cls.soft_skills || currentSoftSkills || [];
    const previewTags = cls.tags || currentTags || [];

    const hardChips = previewHardSkills.map(s =>
        `<span class="skill-tag-chip">${escapeHtml(s)}</span>`
    ).join('') || '<span style="color:var(--text-muted);font-size:0.6rem;font-style:italic;">None</span>';

    const softChips = previewSoftSkills.map(s =>
        `<span class="skill-tag-chip soft">${escapeHtml(s)}</span>`
    ).join('') || '<span style="color:var(--text-muted);font-size:0.6rem;font-style:italic;">None</span>';

    const tagChips = previewTags.map(s =>
        `<span class="skill-tag-chip ai-tag">${escapeHtml(s)}</span>`
    ).join('') || '<span style="color:var(--text-muted);font-size:0.6rem;font-style:italic;">None</span>';

    html += `<div class="pf-classify">
        <div class="pf-classify-field">
            <label>🧠 Function / Department</label>
            <select id="previewFunctionSelect" class="pf-select">
                <option value="">— Select Function —</option>
                ${funcOptions}
            </select>
        </div>
        <div class="pf-classify-field">
            <label>🏢 Industry Sector</label>
            <select id="previewIndustrySelect" class="pf-select">
                <option value="">— Select Industry —</option>
                ${indOptions}
            </select>
        </div>
    </div>
    <div style="margin-top:8px;">
        <label style="font-size:0.62rem;color:var(--text-muted);display:block;margin-bottom:3px;">🔧 Hard Skills</label>
        <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">${hardChips}</div>
        <label style="font-size:0.62rem;color:var(--text-muted);display:block;margin-bottom:3px;">💬 Soft Skills</label>
        <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">${softChips}</div>
        <label style="font-size:0.62rem;color:var(--text-muted);display:block;margin-bottom:3px;">🏷️ Tags (AI)</label>
        <div style="display:flex;flex-wrap:wrap;gap:3px;">${tagChips}</div>
    </div>`;

    // ── 5. THE COMPLETE ENTITY FORM ───────────────────────────────────
    //
    // EVERY entity type from the schema is shown here — including empty ones.
    // Each field shows:
    //   • Existing annotation(s) → editable input + ✕ delete button
    //   • Always one dashed "Add…" input at the bottom for new values
    //
    // We skip:
    //   - FUNCTION / INDUSTRY  →  handled above in Classification section
    //   - SECTION_HEADER       →  structural text marker, not a profile field
    //   - BULLET_MARKER        →  structural text marker, not a profile field
    //
    // SUMMARY_TEXT lives in META in the schema but IS a profile field,
    // so we include it explicitly in the PROFESSIONAL section. 💅

    const SKIP_ENTITIES = new Set([
        'FUNCTION', 'INDUSTRY',       // handled in Classification above
        'SECTION_HEADER', 'BULLET_MARKER'  // structural META markers
    ]);

    // ── MULTI_VALUE_TYPES ─────────────────────────────────────────────
    //
    // Entity types where the user is allowed (and encouraged!) to enter
    // MORE THAN ONE value in the Save Preview form.
    //
    // Rules of thumb for what goes here:
    //   ✅ ADD if max_per_doc > 1 in the schema AND it makes real-world sense
    //      for a candidate to have multiple (e.g. two phone numbers, work email
    //      + personal email, past jobs, multiple skills)
    //   ❌ SKIP if it should only ever have one entry per person (name, DOB,
    //      NRIC, gender — there is only ONE you, darling! 💅)
    //
    // When a user fills the "Add…" input for a multi-value type, a brand-new
    // dashed "Add…" input instantly appears below it — no page refresh needed.
    // Think of it like handing someone a clipboard and they keep asking for
    // more lines to fill in! 📋✨
    // ══════════════════════════════════════════════════════════════════
    // 📝 LONG_TEXT_TYPES — Entities that need <textarea> instead of <input>
    //
    // 🐛 BUG FIX: Regular <input type="text"> cannot handle multi-line
    // content! When a user highlights a long job description paragraph
    // with newlines, the text would DISAPPEAR on save because:
    //   1. Newlines break the HTML value="..." attribute
    //   2. <input> elements literally can't store multi-line text
    //
    // These entity types commonly contain paragraph text, bullet points,
    // or multi-line descriptions — they get <textarea> elements instead!
    // Like giving a diva a proper stage instead of a broom closet! 🎭✨
    // ══════════════════════════════════════════════════════════════════
    const LONG_TEXT_TYPES = new Set([
        'JOB_DESCRIPTION',       // Multi-line job responsibilities
        'PROJECT_DESCRIPTION',   // Project details and achievements
        'SUMMARY_TEXT',          // Career summary / objective
        'METRIC',                // Can be detailed achievement text
    ]);

    const MULTI_VALUE_TYPES = new Set([
        // PERSONAL — multiple contact points are common
        'EMAIL',              // work + personal email
        'PHONE',              // mobile + office + home
        'LOCATION',           // home address + district area
        'EXPECTED_LOCATION',  // open to multiple cities

        // PROFESSIONAL — candidates have many roles
        'JOB_TITLE',          // current + previous titles
        'ORGANIZATION',       // current + all past employers
        'WORK_DATE',          // one range per role
        'WORK_LOCATION',      // different office locations
        'JOB_DESCRIPTION',    // many bullet points per role
        'METRIC',             // many quantitative achievements
        'PROJECT_TITLE',      // multiple projects
        'PROJECT_DESCRIPTION',// multiple project bullets
        'INDUSTRY',           // cross-sector experience

        // EDUCATION — multiple degrees are common
        'DEGREE',
        'INSTITUTION',
        'FIELD_OF_STUDY',
        'EDU_DATE',
        'GPA',

        // SKILLS — always multiple!
        'SKILL',
        'SOFT_SKILL',
        'CERTIFICATION',
        'LANGUAGE_SKILL',
        'PROFICIENCY',
        'CERT_ISSUER',
        'CERT_DATE',
    ]);

    // Category sections — controls grouping, icon, label, and field ORDER.
    // Order matters: most important fields appear at the top of each group.
    const CATEGORY_SECTIONS = [
        {
            label: '👤 Personal Info',
            // Required fields first, then demographics, then optional IDs
            types: [
                'PERSON_NAME', 'EMAIL', 'PHONE',
                'DATE_OF_BIRTH', 'GENDER', 'MARITAL_STATUS',
                'NATIONALITY', 'NRIC_ID',
                'LOCATION', 'EXPECTED_LOCATION'
            ]
        },
        {
            label: '💼 Professional Experience',
            // Current role first, then work history blocks, then summary
            types: [
                'JOB_TITLE', 'ORGANIZATION', 'WORK_DATE', 'WORK_LOCATION',
                'JOB_DESCRIPTION', 'METRIC',
                'PROJECT_TITLE', 'PROJECT_DESCRIPTION',
                'SUMMARY_TEXT'
            ]
        },
        {
            label: '🎓 Education',
            types: [
                'DEGREE', 'INSTITUTION', 'FIELD_OF_STUDY', 'EDU_DATE', 'GPA'
            ]
        },
        {
            label: '🛠️ Skills & Certifications',
            types: [
                'LANGUAGE_SKILL', 'PROFICIENCY',
                'SKILL', 'SOFT_SKILL', 'SKILL_CATEGORY',
                'CERTIFICATION', 'CERT_ISSUER', 'CERT_DATE'
            ]
        }
    ];

    html += `<div class="preview-form">`;

    for (const section of CATEGORY_SECTIONS) {
        // Filter to entity types that actually exist in this schema build
        // (future-proofs against schema additions/removals)
        const validTypes = section.types.filter(t => SCHEMA.entities[t] && !SKIP_ENTITIES.has(t));
        if (validTypes.length === 0) continue;

        html += `<div class="pf-section">
            <div class="pf-section-header">${escapeHtml(section.label)}</div>`;

        for (const entityType of validTypes) {
            const def        = SCHEMA.entities[entityType];
            const existing   = annoByType[entityType] || [];
            const color      = def.color || '#7ee8fa';
            const isRequired = !!def.is_required;
            const hasMissing = isRequired && existing.length === 0;

            html += `<div class="pf-field" data-entity-type="${entityType}">
                <!-- Label column: coloured dot + name + required marker + "Missing" badge -->
                <div class="pf-label">
                    <span class="pf-dot" style="background:${color};"></span>
                    <span>${escapeHtml(def.label)}</span>
                    ${isRequired
                        ? `<span class="pf-required" title="Required — must have at least one annotation">*</span>`
                        : ''}
                    ${hasMissing
                        ? `<span class="pf-missing-badge">Missing</span>`
                        : ''}
                </div>

                <!-- Inputs column: existing rows + always-visible "Add…" row -->
                <div class="pf-inputs" id="pf_inputs_${entityType}">`;

            // ── Render each existing annotation as an editable row ─────────
            // 🐛 BUG FIX: Use <textarea> for LONG_TEXT_TYPES to handle
            // multi-line content (job descriptions, summaries, etc.)
            // Regular <input> elements truncate text at newlines! 💀
            for (const ann of existing) {
                const isLongText = LONG_TEXT_TYPES.has(entityType);
                const safeText = escapeHtml(ann.text);
                
                if (isLongText) {
                    // 📝 TEXTAREA for long-form content
                    html += `<div class="pf-input-row"
                                  data-anno-idx="${ann.idx}"
                                  data-entity-type="${entityType}">
                        <textarea class="pf-input pf-textarea preview-edit-input"
                               data-anno-idx="${ann.idx}"
                               data-entity-type="${entityType}"
                               data-original="${safeText}"
                               oninput="this.classList.toggle('edited', this.value.trim() !== decodeHtmlEntities(this.dataset.original).trim())"
                               placeholder="${escapeHtml(def.label)}…"
                               title="Edit ${escapeHtml(def.label)}">${safeText}</textarea>
                        <button class="pf-del-btn"
                                onclick="pfMarkDelete(this, ${ann.idx}, '${entityType}')"
                                title="Mark for deletion">✕</button>
                    </div>`;
                } else {
                    // 📝 Regular INPUT for single-line content
                    html += `<div class="pf-input-row"
                                  data-anno-idx="${ann.idx}"
                                  data-entity-type="${entityType}">
                        <input type="text"
                               class="pf-input preview-edit-input"
                               value="${safeText}"
                               data-anno-idx="${ann.idx}"
                               data-entity-type="${entityType}"
                               data-original="${safeText}"
                               oninput="this.classList.toggle('edited', this.value !== this.dataset.original)"
                               placeholder="${escapeHtml(def.label)}…"
                               title="Edit ${escapeHtml(def.label)}"
                        />
                        <button class="pf-del-btn"
                                onclick="pfMarkDelete(this, ${ann.idx}, '${entityType}')"
                                title="Mark for deletion">✕</button>
                    </div>`;
                }
            }

            // ── Always render one dashed "Add…" input row ──────────────────
            //
            // For MULTI_VALUE_TYPES: when the user fills this input, a NEW
            // dashed row automatically appears below via pfHandleAddInput().
            // That spawned row ALSO gets the same handler — it's recursive!
            //
            // For single-value types: fills once and stops. No new row spawned.
            //
            // Like a pad of paper — tear off a sheet, another appears! 📋✨
            const isMulti = MULTI_VALUE_TYPES.has(entityType);
            const isLongText = LONG_TEXT_TYPES.has(entityType);

            // 🐛 BUG FIX: Use <textarea> for LONG_TEXT_TYPES "Add..." rows too!
            if (isLongText) {
                html += `<div class="pf-input-row pf-add-row"
                              data-anno-idx="-1"
                              data-entity-type="${entityType}"
                              data-is-new="true">
                    <textarea class="pf-input pf-textarea pf-add-input"
                           data-anno-idx="-1"
                           data-entity-type="${entityType}"
                           data-original=""
                           data-is-new="true"
                           data-is-multi="${isMulti}"
                           data-is-long-text="true"
                           oninput="pfHandleAddInput(this)"
                           placeholder="Add ${escapeHtml(def.label)}…"
                           title="Type to add a new ${escapeHtml(def.label)}"></textarea>
                </div>`;
            } else {
                html += `<div class="pf-input-row pf-add-row"
                              data-anno-idx="-1"
                              data-entity-type="${entityType}"
                              data-is-new="true">
                    <input type="text"
                           class="pf-input pf-add-input"
                           value=""
                           data-anno-idx="-1"
                           data-entity-type="${entityType}"
                           data-original=""
                           data-is-new="true"
                           data-is-multi="${isMulti}"
                           oninput="pfHandleAddInput(this)"
                           placeholder="Add ${escapeHtml(def.label)}…"
                           title="Type to add a new ${escapeHtml(def.label)}"
                    />
                </div>`;
            }

            html += `    </div><!-- /pf-inputs -->
            </div><!-- /pf-field -->`;
        }

        html += `</div><!-- /pf-section -->`;
    }

    html += `</div><!-- /preview-form -->`;

    // ── Inject into modal ──────────────────────────────────────────────
    body.innerHTML = html;

    // ── 6. CONFIRM BUTTON STATE ───────────────────────────────────────
    // Hard block: zero annotations → can't save as complete
    // Soft criticals: missing fields but still allowed with a warning
    // All clear: green light! 🟢
    if (!data.can_complete && status === 'completed') {
        confirmBtn.disabled    = true;
        confirmBtn.textContent = '🚫 Add Annotations First';
        confirmBtn.className   = 'btn';
        confirmBtn.style.opacity = '0.5';
        confirmBtn.style.cursor  = 'not-allowed';
        footerInfo.innerHTML = `
            <span style="color:var(--accent-red);">
                🚨 No annotations found — add some labels before marking complete.
            </span>`;

    } else if (status === 'completed' && s.critical_errors > 0) {
        confirmBtn.disabled    = false;
        confirmBtn.style.opacity = '1';
        confirmBtn.style.cursor  = 'pointer';
        footerInfo.innerHTML = `
            <span style="color:var(--accent-orange);">
                ⚠️ ${s.critical_errors} field${s.critical_errors !== 1 ? 's' : ''} flagged —
                resume may genuinely be missing them.
                You can still mark complete if you're sure. 👀
            </span>`;

    } else {
        confirmBtn.disabled    = false;
        confirmBtn.style.opacity = '1';
        confirmBtn.style.cursor  = 'pointer';
        footerInfo.innerHTML = `Press <strong>Enter</strong> to confirm, <strong>Escape</strong> to cancel`;
    }
}

/**
 * pfMarkDelete(btn, annoIdx, entityType)
 *
 * Toggles an existing annotation row as "marked for deletion" in the preview.
 * Updates the Total Labels live counter in the stats banner.
 * The ✕ button toggles to ↩ (undo) so nothing is irrevocable! 💅
 *
 * @param {HTMLElement} btn       - The delete button that was clicked
 * @param {number}      annoIdx   - Index in the annotations[] array
 * @param {string}      entityType - Entity type string (for logging)
 */
function pfMarkDelete(btn, annoIdx, entityType) {
    // Walk up to the parent .pf-input-row
    const row = btn.closest('.pf-input-row');
    if (!row) return;

    // Toggle deleted state
    const isDeleted = row.classList.toggle('deleted');
    btn.textContent = isDeleted ? '↩' : '✕';
    btn.title       = isDeleted ? 'Undo deletion' : 'Mark for deletion';
    btn.classList.toggle('undo', isDeleted);   // Green style when showing undo

    // ── Live-update Total Labels counter ─────────────────────────────
    // Count only real annotation rows (not .pf-add-row) that are not deleted
    const totalActive = document.querySelectorAll(
        '.pf-input-row:not(.deleted):not(.pf-add-row)'
    ).length;
    const totalEl = document.getElementById('prevStatTotal');
    if (totalEl) totalEl.textContent = totalActive;

    console.log(`${isDeleted ? '🗑️ Mark delete' : '↩ Undo delete'}: [${annoIdx}] ${entityType}`);
}

/**
 * pfHandleAddInput(input)
 *
 * Called by oninput on every "Add…" dashed field in the preview form.
 *
 * Does two things:
 *   1. Toggles CSS classes so the input turns from dashed/italic (empty)
 *      to solid/normal (has text) — pure visual feedback.
 *
 *   2. For MULTI_VALUE_TYPES only: when THIS input gets text AND it's the
 *      last "Add…" row for its entity type, instantly injects a new empty
 *      "Add…" row below it. Like summoning a fresh line on a form! 📋
 *
 * We never spawn a new row if the user hasn't typed anything — that would
 * give us an infinite cascade of empty rows. Very un-glamorous. 🙅
 *
 * @param {HTMLInputElement} input - The input that fired the event
 */
function pfHandleAddInput(input) {
    const hasText    = input.value.trim() !== '';
    const entityType = input.dataset.entityType;
    const isMulti    = input.dataset.isMulti === 'true';

    // ── Step 1: Toggle visual state ──────────────────────────────────
    // .edited       → yellow border when the user has changed the value
    // .pf-add-input → dashed/italic style when empty; removed when filled
    input.classList.toggle('edited',       hasText);
    input.classList.toggle('pf-add-input', !hasText);

    // ── Step 2: Spawn a new "Add…" row for multi-value types ─────────
    // Guard conditions — ALL must be true to spawn:
    //   a) Entity type allows multiple values
    //   b) User has actually typed something (not just focused / cleared)
    //   c) This input has NOT already triggered a spawn (data-spawned flag)
    //      ← THIS is the key fix for the "jumps every keypress" bug! 🐛
    //   d) This input IS the current last "Add…" row for its entity type
    //
    // Without guard (c), every single keypress would check "am I last?" —
    // and after the first spawn the new row IS last, so nothing happens,
    // but without the flag the OLD row still thinks it CAN spawn again
    // if the user clears and re-types. The flag makes it one-shot. 💅
    if (!isMulti || !hasText) return;

    // ── Guard (c): only spawn once per input ─────────────────────────
    // We stamp a custom data attribute the first time we spawn.
    // Subsequent keystrokes in the same input skip the spawn block. 🎯
    if (input.dataset.spawned === 'true') return;

    const inputsContainer = input.closest('.pf-inputs');
    if (!inputsContainer) return;

    // ── Guard (d): only spawn from the LAST add-row ──────────────────
    const allAddRows = inputsContainer.querySelectorAll('.pf-add-row');
    const lastAddRow = allAddRows[allAddRows.length - 1];
    if (input.closest('.pf-add-row') !== lastAddRow) return;

    // ── Mark this input as "already spawned" before we do anything ───
    // Must set BEFORE DOM manipulation so re-entrant oninput calls
    // (which can fire during insertAdjacentElement in some browsers) skip. 🔒
    input.dataset.spawned = 'true';

    // Retrieve entity label from schema for placeholder text
    const def   = SCHEMA.entities[entityType];
    const label = def ? def.label : entityType;

    // ── Build the new empty "Add…" row ───────────────────────────────
    // We create DOM elements directly (not innerHTML concat) so the event
    // handler binds correctly without needing eval(). Safer! 💅
    const newRow = document.createElement('div');
    newRow.className          = 'pf-input-row pf-add-row';
    newRow.dataset.annoIdx    = '-1';
    newRow.dataset.entityType = entityType;
    newRow.dataset.isNew      = 'true';

    // 🐛 BUG FIX: Create textarea for LONG_TEXT_TYPES, input for others
    const isLongText = input.dataset.isLongText === 'true' || LONG_TEXT_TYPES.has(entityType);
    const newInput = document.createElement(isLongText ? 'textarea' : 'input');
    
    if (!isLongText) {
        newInput.type = 'text';
        newInput.className = 'pf-input pf-add-input';
    } else {
        newInput.className = 'pf-input pf-textarea pf-add-input';
    }
    newInput.value       = '';
    newInput.placeholder = `Add another ${label}…`;
    newInput.title       = `Type to add another ${label}`;

    // Stamp all data attributes so confirmSave() picks this row up correctly
    newInput.dataset.annoIdx    = '-1';
    newInput.dataset.entityType = entityType;
    newInput.dataset.original   = '';
    newInput.dataset.isNew      = 'true';
    newInput.dataset.isMulti    = 'true';   // propagate multi flag to children
    newInput.dataset.spawned    = 'false';  // new row starts un-spawned

    // Attach the SAME handler — enables recursive multi-row addition 🔄
    newInput.addEventListener('input', () => pfHandleAddInput(newInput));

    newRow.appendChild(newInput);

    // Insert AFTER the current last add-row
    lastAddRow.insertAdjacentElement('afterend', newRow);

    // ── NO auto-focus ─────────────────────────────────────────────────
    // 🐛 BUG FIXED: auto-focus was here before.
    // It stole the cursor to the new blank row after the FIRST character,
    // mid-word. User now stays in their current input and can Tab or click
    // into the new row when they are done. 💅
}

/**
 * confirmSave(overrideStatus, shouldClose)
 * Collects data from the preview window and prepares it for the server.
 */
function confirmSave(overrideStatus = null, shouldClose = true) {
    const statusToSave = overrideStatus || pendingSaveStatus;

    // ── 🛡️ SAFETY NET — wrap everything in try-catch ─────────────
    // 🐛 BUG FIX: Previously, ANY error in data collection (e.g. an
    // out-of-bounds annotation index, a corrupted input value, or a
    // DOM element that vanished) would throw an uncaught TypeError.
    // That silently halted execution BEFORE saveAnnotations() ran,
    // so the user saw the preview close but nothing actually saved.
    // No toast, no alert, no error — just... silence. 😱
    //
    // Now we catch errors, show the user a clear message, and still
    // attempt the save with whatever data we managed to collect.
    // Like a stage manager calling "SHOW MUST GO ON!" even when
    // someone trips backstage! 🎭💪
    try {
        // Collect indices of annotations marked for deletion
        const deletedIndices = new Set();
        document.querySelectorAll('.pf-input-row.deleted').forEach(row => {
            const idx = row.dataset.annoIdx;
            if (idx && idx !== '-1') {
                deletedIndices.add(parseInt(idx, 10));
            }
        });

        // 🐛 BUG FIX: Include both input AND textarea elements!
        const allPreviewInputs = document.querySelectorAll('.pf-input, .pf-textarea');;
        allPreviewInputs.forEach(input => {
            const idx = parseInt(input.dataset.annoIdx, 10);
            const entityType = input.dataset.entityType;
            const isNew = input.dataset.isNew === 'true';
            const newText = input.value.trim();
            const row = input.closest('.pf-input-row');
            if (row && row.classList.contains('deleted')) return; // skip deleted

            if (!newText) return;

            if (!isNew && idx >= 0) {
                // 🛡️ Guard: verify the annotation at this index still exists
                // before updating. If the array was modified while the preview
                // was open (e.g. Delete key pressed outside input focus),
                // the index could be stale. Skip gracefully instead of crashing.
                if (idx < annotations.length && annotations[idx]) {
                    const oldText = annotations[idx].text;
                    
                    // 🐛 BUG FIX: Compare TRIMMED versions to avoid false positives
                    // from whitespace differences. The original highlight might have
                    // captured trailing spaces, but input.value.trim() removes them.
                    // Without this, unchanged annotations lose their char positions! 💅
                    const textActuallyChanged = (oldText || '').trim() !== newText;
                    
                    // Only update text if it actually changed
                    if (textActuallyChanged) {
                        annotations[idx].text = newText;

                        // 🐛 BUG FIX: When the user EDITS the text of an existing
                        // annotation, the char_start/char_end positions become STALE
                        // because they still point to the ORIGINAL text in RAW_TEXT.
                        //
                        // Before this fix, renderText() would re-slice RAW_TEXT at
                        // the old positions and show the ORIGINAL text — making it
                        // look like the edit was lost! 😱
                        //
                        // Now we mark edited annotations with char_start/char_end = -1
                        // to signal: "this is user-edited text — don't re-slice!"
                        // Think of it like crossing out a handwritten note and writing
                        // a new one — you can't pretend the old ink is still there! 🖊️✨
                        annotations[idx].char_start = -1;
                        annotations[idx].char_end   = -1;
                        annotations[idx].annotator  = 'human';  // Mark as human-edited
                        console.log(`✏️ Edited [${idx}] ${entityType}: "${oldText}" → "${newText}" (positions reset)`);
                    }
                } else {
                    console.warn(`⚠️ confirmSave: annotations[${idx}] is undefined — skipping update for ${entityType}`);
                }
            } else if (isNew && newText) {
                // Push new annotation (manual entry, no char positions)
                annotations.push({
                    entity_type: entityType,
                    char_start: -1,
                    char_end: -1,
                    text: newText,
                    layer: 0,
                    annotator: 'human',
                    confidence: 1.0
                });
            }
        });

        // Remove deleted annotations
        if (deletedIndices.size > 0) {
            annotations = annotations.filter((_, idx) => !deletedIndices.has(idx));
        }

        // Sync classification from preview to main selects
        const previewFunc = document.getElementById('previewFunctionSelect');
        const previewInd = document.getElementById('previewIndustrySelect');
        if (previewFunc && previewInd) {
            const funcSelect = document.getElementById('functionSelect');
            const indSelect = document.getElementById('industrySelect');
            if (funcSelect) funcSelect.value = previewFunc.value;
            if (indSelect) indSelect.value = previewInd.value;
        }
    } catch (err) {
        // ── Error during data collection — warn but don't abort ──────
        console.error('❌ confirmSave data collection error:', err);
        showToast(`⚠️ Some edits may not have been captured: ${err.message}`, 'warning');
        // Fall through to save whatever we DID manage to collect
    }

    // Close or refresh UI
    if (shouldClose) {
        closePreview();
    } else {
        showSavePreview(statusToSave); // re-render with updated data
    }

    // Actually save to server — always attempt this, even if collection had issues
    saveAnnotations(statusToSave);

    // Refresh text highlights and annotation list
    renderText();
    if (activeInspector === 'labels') openLabelsModal();
}

/**
 * markPreviewDelete(btn, annoIdx)
 * 
 * Toggles an annotation row as "marked for deletion" in the preview.
 * ALSO live-updates:
 *   - Per-entity-type count badges
 *   - Total Labels stat in the banner
 * Everything stays in sync — like a perfectly choreographed number! 🎭💅
 */
function markPreviewDelete(btn, annoIdx) {
    const row = btn.closest('.preview-edit-row');
    if (!row) return;

    const isDeleted = row.classList.toggle('deleted');
    btn.textContent = isDeleted ? '↩' : '✕';
    btn.title = isDeleted ? 'Undo deletion' : 'Mark for deletion';

    // ── Update per-entity-type count ─────────────────────────────
    const entityType = row.dataset.entityType;
    if (entityType) {
        const activeRows = document.querySelectorAll(
            `.preview-edit-row[data-entity-type="${entityType}"]:not(.deleted)`
        );
        const countEl = document.getElementById('prevCount_' + entityType);
        if (countEl) countEl.textContent = activeRows.length;
    }

    // ── Update Total Labels stat in banner ───────────────────────
    const totalActive = document.querySelectorAll('.preview-edit-row:not(.deleted)').length;
    const totalEl = document.getElementById('prevStatTotal');
    if (totalEl) totalEl.textContent = totalActive;
}

/**
 * closePreview()
 * Hides the preview modal and cleans up event listeners.
 */
function closePreview() {
    document.getElementById('previewOverlay').classList.remove('open');
    document.removeEventListener('keydown', previewKeyHandler);
    pendingSaveStatus = null;
}

/**
 * closePreviewOnBackdrop(event)
 * Closes modal only when clicking the dark backdrop, not the modal itself.
 */
function closePreviewOnBackdrop(event) {
    if (event.target === document.getElementById('previewOverlay')) {
        closePreview();
    }
}

/**
 * previewKeyHandler(event)
 * Keyboard shortcuts while preview modal is open.
 */
/**
 * previewKeyHandler(event)
 * Keyboard shortcuts while preview modal is open.
 */
function previewKeyHandler(event) {
    if (event.key === 'Escape') {
        event.preventDefault();
        closePreview();
    }

    // 🐛 BUG FIX: Previously, pressing Enter while editing an input
    // forced the save status to 'in_progress' — even if the user opened
    // the preview via "✅ Mark Complete" (status = 'completed').
    // The user would see their status silently downgraded and think
    // the "completed" save didn't work. Drama! 🎭
    //
    // NOW: Enter in an input → save as Draft (in_progress) WITHOUT closing,
    // so the user can keep editing. This is a "quick save" convenience.
    // Enter outside an input → save with the INTENDED status and close.
    // Like hitting "Save" vs "Save & Close" in any good editor! 💾
    if (event.key === 'Enter') {
        const inInput = event.target.closest('.pf-input, .preview-edit-input, select');
        
        if (inInput) {
            event.preventDefault();
            // Quick-save as draft while staying in the preview
            confirmSave('in_progress', false); 
        } else {
            event.preventDefault();
            // Full save with the originally intended status, then close
            confirmSave(pendingSaveStatus, true); 
        }
    }
}

// =================================================================
// 💾 SAVE / LOAD
// =================================================================
/**
 * saveAnnotations(status)
 * Sends the final data payload to Python.
 */
function saveAnnotations(status = 'in_progress') {
    // 🛡️ Guard: ensure status is always a valid string
    // If somehow null/undefined leaks through (e.g. from pendingSaveStatus
    // being cleared by closePreview), default to 'in_progress'.
    if (!status) status = 'in_progress';

    const payload = {
        doc_id: DOC_ID,
        status: status,
        function: document.getElementById('functionSelect').value,
        industry: document.getElementById('industrySelect').value,
        hard_skills: currentHardSkills,
        soft_skills: currentSoftSkills,
        tags: currentTags,
        annotations: annotations,
        // 👤 Track who saved — stored in ner_documents.annotator
        annotator_name: getAnnotatorName() || ''
    };

    fetch('/api/save_annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
    if (data.success) {
        if (status === 'completed') {
            showToast(
                '🏆 Marked Complete! Candidate is officially runway-ready! ✨',
                'success',
                6000
            );
            showCompletionCelebration();   // ← Big glamorous finale only for Complete
        } else {
            showToast(
                '💾 Draft Saved! Progress secured — come back anytime! ✨',
                'success',
                4000
            );
            showDraftSavedCelebration();   // ← Tiny cute party ONLY for drafts
        }
        updateStatusBadge(status);
    } else {
        showToast('❌ Save failed: ' + (data.error || 'Unknown error'), 'error', 6000);
    }
})
    .catch(err => {
        showToast('❌ Network error: ' + err, 'error', 6000);
    });
}

/**
 * updateStatusBadge(status)
 *
 * Updates the header status badge text + CSS class to reflect the
 * latest save status — live, no page reload needed.
 *
 * Status → badge mapping (mirrors Jinja2 logic in the template):
 *   completed   → badge-complete  ✅ completed
 *   in_progress → badge-progress  🔵 in_progress
 *   anything else → badge-pending ⏳ <status>
 *
 * Drama analogy: It's like the stage manager flipping the sign from
 * "REHEARSAL" to "PERFORMANCE" the moment the show goes live! 🎭✨
 *
 * @param {string} status - 'completed' | 'in_progress' | 'pending'
 */
function updateStatusBadge(status) {
    const badge = document.getElementById('statusBadge');
    if (!badge) return;  // Guard — badge must exist in DOM

    // Remove all existing badge-* colour classes before adding the new one
    badge.classList.remove('badge-complete', 'badge-progress', 'badge-pending');

    if (status === 'completed') {
        badge.classList.add('badge-complete');
        badge.textContent = 'completed';
    } else if (status === 'in_progress') {
        badge.classList.add('badge-progress');
        badge.textContent = 'in_progress';
    } else {
        badge.classList.add('badge-pending');
        badge.textContent = status || 'pending';
    }
}



// =================================================================
// 🤖 PRE-ANNOTATE
// =================================================================
function runPreAnnotate() {
    const threshold = parseFloat(prompt('Confidence threshold (0.0 - 1.0):', '0.6'));
    if (isNaN(threshold)) return;

    fetch(`/api/pre-annotate/${CANDIDATE_ID}?threshold=${threshold}`)
    .then(r => r.json())
    .then(d => {
        if (d.annotations) {
            // Merge: keep human annotations, add new auto ones that don't overlap
            const humanAnns = annotations.filter(a => a.annotator === 'human');
            const autoAnns = d.annotations.filter(newA => {
                return !humanAnns.some(h =>
                    !(newA.char_end <= h.char_start || newA.char_start >= h.char_end)
                );
            });
            annotations = [...humanAnns, ...autoAnns];
            renderText();
            showToast(`🤖 Added ${autoAnns.length} auto-annotations (kept ${humanAnns.length} human)`);
        }
    });
}

// =================================================================
// 🏷️ BIO PREVIEW — Now lives in its own glorious tab! ✨
// =================================================================

/**
 * loadBIOPreview()
 * 
 * Fetches BIO tags from the server and renders them in the BIO tab.
 * Previously this was a toggle — now it's a proper tab citizen.
 * Like upgrading from a flip phone to a smartphone, darling! 📱✨
 */
function loadBIOPreview() {
    const preview = document.getElementById('bioPreview');
    const statsEl = document.getElementById('bioStats');

    // Show loading state
    preview.innerHTML = '<div style="color:var(--text-muted); font-size:0.78rem; text-align:center; padding:20px 0;">⏳ Loading BIO tags...</div>';

    fetch(`/api/bio-preview/${CANDIDATE_ID}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({annotations})  // Send current annotations state
    })
    .then(r => r.json())
    .then(d => {
        const total = d.total_tokens;
        const oCount = d.tag_counts['O'] || 0;
        const labeled = total - oCount;
        const coverage = total > 0 ? ((labeled / total) * 100).toFixed(1) : 0;

        // Update the stats line above the preview box
        if (statsEl) {
            statsEl.innerHTML = `
                <strong style="color:var(--accent-cyan)">${total}</strong> tokens | 
                <strong style="color:var(--accent-green)">${labeled}</strong> labeled 
                (<strong style="color:var(--accent-yellow)">${coverage}%</strong> coverage) | 
                <span style="color:var(--text-muted)">Showing first 200</span>
            `;
        }

        // Render each token-tag pair as a row
        preview.innerHTML = d.preview.map(p => {
            // Color code: O = muted, B- = green (start), I- = cyan (inside)
            const tagClass = p.tag === 'O' ? 'tag-o' :
                             p.tag.startsWith('B-') ? 'tag-b' : 'tag-i';
            return `<div class="bio-row">
                <span class="bio-token">${escapeHtml(p.token)}</span>
                <span class="bio-tag ${tagClass}">${p.tag}</span>
            </div>`;
        }).join('');
    })
    .catch(err => {
        // Graceful error state — no crashes allowed on this runway! 🛑
        preview.innerHTML = `<div style="color:var(--accent-red); font-size:0.78rem; padding:10px 0;">
            ❌ Failed to load BIO preview. Check console for details.
        </div>`;
        console.error('BIO preview error:', err);
    });
}

// Keep old function name as alias so any existing calls still work
function showBIOPreview() {
    openBIOModal();
}

// =================================================================
// 🧩 EDGE CASE CHECK — Now in its own tab! 
// =================================================================

/**
 * loadEdgeCases()
 * 
 * Scans the raw text for potential annotation edge cases and 
 * renders hints in the Edge Cases tab. 
 * Think of edge cases as the divas of the annotation world —
 * they need SPECIAL attention! 💅
 */
function loadEdgeCases() {
    const container = document.getElementById('edgeHints');

    // Show loading state
    container.innerHTML = '<div style="color:var(--text-muted); font-size:0.78rem; text-align:center; padding:20px 0;">⏳ Scanning for edge cases...</div>';

    fetch('/api/edge-case-check', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: RAW_TEXT})
    })
    .then(r => r.json())
    .then(d => {
        if (!d.suggestions?.length) {
            // All clear — no drama today! 🎉
            container.innerHTML = '<div style="color:var(--accent-green); font-size:0.78rem; text-align:center; padding:20px 0;">✅ No edge cases detected! Clean and fabulous! ✨</div>';
            return;
        }
        // Render each edge case hint as a styled card
        container.innerHTML = d.suggestions.map(s =>
            `<div class="edge-hint">
                <strong>${s.type}:</strong> "${escapeHtml(s.text)}"<br>
                💡 ${s.advice}
            </div>`
        ).join('');
    })
    .catch(err => {
        container.innerHTML = `<div style="color:var(--accent-red); font-size:0.78rem;">❌ Scan failed. Check console.</div>`;
        console.error('Edge case check error:', err);
    });
}

// Keep old name as alias for backward compat
function checkEdgeCases() {
    openEdgeModal();
}

// =================================================================
// 🧠 FUNCTION/INDUSTRY/SKILLS/TAGS CLASSIFICATION
// =================================================================

// Hardcoded options from the schema
const FUNCTION_OPTIONS = [
    "Administration", "Accounting & Finance", "Customer Service",
    "Engineering", "Facilities Management", "Fresh Graduates (Deg/Dip)",
    "Fresh Graduates (ITE)", "Human Resources", "IT", "Legal",
    "Management", "Marketing", "Procurement", "Production",
    "Sales", "Shipping & Logistics", "others"
];
const INDUSTRY_OPTIONS = [
    "Automotive", "Banking & Finance", "Building/Construction",
    "Chemicals", "Education", "Engineering - Aerospace",
    "Engineering - Precision", "Energy (Oil & Gas)", "Environment & Water",
    "Healthcare", "Hospitality & Tourism", "F&B", "FMCG",
    "Infocomm", "Medical Technology", "Marine & Offshore",
    "Retail", "Pharmaceutical/Biotech", "Robotics", "Semiconductor",
    "Supply Chain Mgt & Logistics", "Telco", "Trading", "Others"
];

// 🆕 In-memory arrays for skills & tags (synced with chip UI)
let currentHardSkills = [];
let currentSoftSkills = [];
let currentTags = [];

/**
 * renderSkillChips(containerId, items, chipClass, arrayRef)
 *
 * Renders an array of skill/tag strings as clickable chip elements
 * inside the given container. Each chip has a ✕ button to remove it.
 *
 * Think of it like pinning nametags on a board — each one removable
 * with a single click! 📌✨
 *
 * @param {string} containerId - DOM id of the container div
 * @param {string[]} items - Array of skill/tag strings
 * @param {string} chipClass - CSS class for chip styling ('hard'|'soft'|'ai-tag')
 * @param {string} arrayName - Which array to modify ('hard'|'soft'|'tag')
 */
function renderSkillChips(containerId, items, chipClass, arrayName) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    items.forEach((item, idx) => {
        const chip = document.createElement('span');
        chip.className = `skill-tag-chip ${chipClass}`;
        chip.innerHTML = `${escapeHtml(item)}<span class="tag-remove" onclick="removeSkillTag('${arrayName}', ${idx})">✕</span>`;
        container.appendChild(chip);
    });
}

/**
 * addSkillTag(type)
 *
 * Reads the corresponding input field, adds the value to the
 * in-memory array, re-renders chips, and clears the input.
 * Deduplicates automatically — no double outfits, darling! 👗
 *
 * @param {string} type - 'hard' | 'soft' | 'tag'
 */
function addSkillTag(type) {
    let input, arr, containerId, chipClass;

    if (type === 'hard') {
        input = document.getElementById('hardSkillInput');
        arr = currentHardSkills;
        containerId = 'hardSkillsContainer';
        chipClass = '';
    } else if (type === 'soft') {
        input = document.getElementById('softSkillInput');
        arr = currentSoftSkills;
        containerId = 'softSkillsContainer';
        chipClass = 'soft';
    } else {
        input = document.getElementById('tagInput');
        arr = currentTags;
        containerId = 'tagsContainer';
        chipClass = 'ai-tag';
    }

    if (!input) return;
    const val = input.value.trim();
    if (!val) return;

    // Deduplicate (case-insensitive check)
    if (!arr.some(s => s.toLowerCase() === val.toLowerCase())) {
        arr.push(val);
    }

    renderSkillChips(containerId, arr, chipClass, type);
    input.value = '';
    input.focus();
}

/**
 * removeSkillTag(type, index)
 *
 * Removes a skill/tag from the in-memory array by index and re-renders.
 * Like plucking a bad sequin off a gown — quick and painless! ✂️✨
 *
 * @param {string} type - 'hard' | 'soft' | 'tag'
 * @param {number} index - Array index to remove
 */
function removeSkillTag(type, index) {
    if (type === 'hard') {
        currentHardSkills.splice(index, 1);
        renderSkillChips('hardSkillsContainer', currentHardSkills, '', 'hard');
    } else if (type === 'soft') {
        currentSoftSkills.splice(index, 1);
        renderSkillChips('softSkillsContainer', currentSoftSkills, 'soft', 'soft');
    } else {
        currentTags.splice(index, 1);
        renderSkillChips('tagsContainer', currentTags, 'ai-tag', 'tag');
        syncPaletteState();
    }
}

// =================================================================
// ⚡ QUICK-PICK TAG PALETTE
// =================================================================

// Core tags grouped by category — mirrors ResumeClassifier.CORE_TAGS
const CORE_TAGS = {
    "Availability":  ["Available Now", "Immediate", "1-Month Notice", "2-Month Notice", "3-Month Notice"],
    "Work Mode":     ["Remote Only", "Hybrid OK", "On-site Only", "Open to Relocation", "Willing to Travel"],
    "Work Type":     ["Full-time", "Part-time", "Contract", "Freelance", "Temp"],
    "Seniority":     ["Fresh Grad", "Junior", "Mid-level", "Senior", "Lead", "Manager", "Director", "C-Suite"],
    "People":        ["Team Lead", "People Manager", "Individual Contributor", "Career Switch", "Return to Work"],
    "Languages":     ["English", "Mandarin", "Malay", "Tamil", "Bilingual", "Trilingual"],
    "Status":        ["Citizen (SG)", "PR (SG)", "EP Holder", "Citizen (MY)", "PR (MY)", "Visa Required"],
    "Tech":          ["Full Stack", "Frontend", "Backend", "Mobile Dev", "DevOps", "Data Science", "AI/ML", "Cybersecurity", "Cloud", "QA / Testing"],
    "Finance":       ["ACCA", "CPA", "CFA", "Big 4", "Audit", "Tax", "Payroll", "Financial Reporting", "SAP User"],
    "HR":            ["Generalist", "Talent Acquisition", "L&D", "Payroll", "HRIS", "Business Partner"],
    "Sales & Mktg":  ["B2B", "B2C", "SaaS Sales", "Key Account Mgmt", "Digital Mktg", "SEO/SEM", "CRM"],
};

const GROUP_COLORS = {
    "Availability": "#e3b341", "Work Mode": "#7ee8fa", "Work Type": "#7ee8fa",
    "Seniority": "#eeb8ff",    "People": "#eeb8ff",    "Languages": "#f778ba",
    "Status": "#f778ba",       "Tech": "#56d364",      "Finance": "#7ee8fa",
    "HR": "#eeb8ff",           "Sales & Mktg": "#e3b341",
};

/**
 * buildTagPalette()
 * Renders the quick-pick palette groups into #tagPaletteGroups.
 * Called once on classify modal open.
 */
function buildTagPalette() {
    const container = document.getElementById('tagPaletteGroups');
    if (!container || container.dataset.built) return;
    container.dataset.built = '1';

    for (const [group, chips] of Object.entries(CORE_TAGS)) {
        const color = GROUP_COLORS[group] || 'var(--accent-green)';

        const label = document.createElement('div');
        label.className = 'palette-group-label';
        label.textContent = group;
        label.style.color = color;
        container.appendChild(label);

        const row = document.createElement('div');
        row.className = 'palette-chips';

        chips.forEach(tag => {
            const btn = document.createElement('span');
            btn.className = 'palette-chip';
            btn.textContent = tag;
            btn.dataset.tag = tag;
            btn.style.borderColor = color + '55';
            btn.addEventListener('click', () => togglePaletteTag(tag, btn, color));
            row.appendChild(btn);
        });

        container.appendChild(row);
    }
}

/**
 * togglePaletteTag(tag, btn, color)
 * Adds or removes a tag when its palette chip is clicked.
 * Syncs the chip's visual state with currentTags.
 */
function togglePaletteTag(tag, btn, color) {
    const idx = currentTags.findIndex(t => t.toLowerCase() === tag.toLowerCase());
    if (idx === -1) {
        // Add
        currentTags.push(tag);
        btn.classList.add('active');
        btn.style.borderColor = color;
        btn.style.color = color;
    } else {
        // Remove
        currentTags.splice(idx, 1);
        btn.classList.remove('active');
        btn.style.borderColor = color + '55';
        btn.style.color = '';
    }
    renderSkillChips('tagsContainer', currentTags, 'ai-tag', 'tag');
    syncPaletteState();  // Keep all chips in sync
}

/**
 * syncPaletteState()
 * Marks palette chips as active/inactive based on currentTags.
 * Called after external changes (populate, remove chip).
 */
function syncPaletteState() {
    const palette = document.getElementById('tagPaletteGroups');
    if (!palette) return;
    palette.querySelectorAll('.palette-chip').forEach(btn => {
        const tag = btn.dataset.tag;
        const color = btn.style.borderColor.replace('55', '') || 'var(--accent-green)';
        const active = currentTags.some(t => t.toLowerCase() === tag.toLowerCase());
        btn.classList.toggle('active', active);
    });
}

/**
 * toggleTagPalette()
 * Shows/hides the quick-pick palette and builds it on first open.
 */
function toggleTagPalette() {
    const palette = document.getElementById('tagPalette');
    const btn = document.getElementById('paletteToggleBtn');
    if (!palette) return;
    const isOpen = palette.style.display !== 'none';
    palette.style.display = isOpen ? 'none' : 'block';
    btn.style.opacity = isOpen ? '0.75' : '1';
    btn.style.borderColor = isOpen ? '' : 'var(--accent-green)';
    if (!isOpen) {
        buildTagPalette();   // Build on first open
        syncPaletteState();  // Reflect any already-added tags
    }
}

/**
 * populateClassification(functionVal, industryVal, hardSkills, softSkills, tags)
 *
 * Populates the Function & Industry dropdowns AND the skill/tag chip
 * containers with the given values. Called on page load and after
 * re-classification.
 *
 * 🆕 Now handles the full classification suite: Function, Industry,
 *     Hard Skills, Soft Skills, and AI-generated Tags!
 */
function populateClassification(functionVal, industryVal, hardSkills, softSkills, tags) {
    const funcSelect = document.getElementById('functionSelect');
    const indSelect = document.getElementById('industrySelect');

    // Guard: if the Classify tab elements aren't in the DOM yet, bail out
    if (!funcSelect || !indSelect) {
        console.warn('⚠️ Classification selects not found in DOM');
        return;
    }

    // Normalize empty/null/undefined to empty string for safe comparison
    functionVal = (functionVal || '').trim();
    industryVal = (industryVal || '').trim();

    console.log('🧠 Populating classification:', {functionVal, industryVal,
        hardSkills: (hardSkills || []).length,
        softSkills: (softSkills || []).length,
        tags: (tags || []).length
    });

    funcSelect.innerHTML = FUNCTION_OPTIONS.map(f =>
        `<option value="${f}" ${f === functionVal ? 'selected' : ''}>${f}</option>`
    ).join('');
    indSelect.innerHTML = INDUSTRY_OPTIONS.map(i =>
        `<option value="${i}" ${i === industryVal ? 'selected' : ''}>${i}</option>`
    ).join('');

    // 🆕 Populate skill/tag chips
    currentHardSkills = Array.isArray(hardSkills) ? [...hardSkills] : [];
    currentSoftSkills = Array.isArray(softSkills) ? [...softSkills] : [];
    currentTags = Array.isArray(tags) ? [...tags] : [];

    renderSkillChips('hardSkillsContainer', currentHardSkills, '', 'hard');
    renderSkillChips('softSkillsContainer', currentSoftSkills, 'soft', 'soft');
    renderSkillChips('tagsContainer', currentTags, 'ai-tag', 'tag');
}

/**
 * openClassifyModal()
 *
 * Opens the Classification & Skills popup modal.
 * Uses the same inspector-overlay pattern as Labels/BIO/Edge.
 * Think of it as opening the talent dossier! 📋✨
 */
function openClassifyModal() {
    openInspector('classify');
}

function reclassify() {
    document.getElementById('classifyStatus').textContent = '⏳ Classifying...';

    // 🐛 BUG FIX: Send CURRENT in-memory annotations via POST!
    // Previously this was a GET that loaded from the DATABASE (stale data).
    // Now we send the live working annotations so the classifier sees
    // the latest SKILL and SOFT_SKILL entities — even if unsaved! 💅
    fetch(`/api/classify/${CANDIDATE_ID}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ annotations: annotations })
    })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                document.getElementById('classifyStatus').textContent = '❌ ' + data.error;
                return;
            }
            populateClassification(
                data.Function, data.Industry,
                data.HardSkills || [], data.SoftSkills || [], data.Tags || []
            );
            document.getElementById('classifyStatus').textContent =
                `✅ Updated! ${(data.HardSkills||[]).length} hard, ${(data.SoftSkills||[]).length} soft, ${(data.Tags||[]).length} tags`;
        })
        .catch(err => {
            document.getElementById('classifyStatus').textContent = '❌ Error: ' + err;
        });
}

function saveClassification() {
    const functionVal = document.getElementById('functionSelect').value;
    const industryVal = document.getElementById('industrySelect').value;
    fetch('/api/save-classification', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            candidate_id: CANDIDATE_ID,
            function: functionVal,
            industry: industryVal,
            hard_skills: currentHardSkills,
            soft_skills: currentSoftSkills,
            tags: currentTags
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.success) {
            document.getElementById('classifyStatus').textContent = '✅ Saved!';
            showToast(
                '🧠 Classification Saved! Function, Industry, Skills & Tags locked in! 💼',
                'success',
                3500
            );
            // Auto-clear inline status after 3s so it does not go stale
            setTimeout(function() {
                var el = document.getElementById('classifyStatus');
                if (el && el.textContent === '✅ Saved!') { el.textContent = ''; }
            }, 3000);
        } else {
            document.getElementById('classifyStatus').textContent = '❌ Save failed';
            showToast('❌ Classification save failed — try again, darling!', 'error');
        }
    })
    .catch(function(err) {
        document.getElementById('classifyStatus').textContent = '❌ Network error';
        showToast('❌ Network error: ' + err, 'error');
    });
}

// =================================================================
// 🚪 PANEL RESIZE — Drag handle to resize the side panel
// =================================================================
/**
 * How it works, darling! 💅
 *
 * The .anno-layout grid uses `grid-template-columns: 1fr 5px var(--side-width)`.
 * When the user drags the resize handle, we calculate how far the mouse is
 * from the RIGHT edge of the viewport — that distance becomes the new
 * --side-width value, which the grid picks up instantly via CSS variables.
 *
 * Think of it as sliding a partition wall in a nightclub: the dance floor
 * (text panel) gets bigger or smaller as the VIP room (side panel) moves! 🪩
 *
 * Min/max are enforced in CSS (.side-panel min-width/max-width) but we also
 * clamp here in JS so the CSS variable never goes to a nonsensical value.
 */
(function initPanelResize() {
    const handle    = document.getElementById('resizeHandle');
    const layout    = document.getElementById('annoLayout');
    const MIN_WIDTH = 220;   /* px — narrowest usable panel */
    const MAX_WIDTH = 600;   /* px — widest panel before it hogs the screen */

    if (!handle || !layout) return;  // Guard: elements must exist

    let dragging = false;

    handle.addEventListener('mousedown', (e) => {
        e.preventDefault();          // Prevent text selection during drag
        dragging = true;
        handle.classList.add('dragging');
        // Use document listeners so drag works even if mouse leaves the handle
        document.addEventListener('mousemove', onDrag);
        document.addEventListener('mouseup', stopDrag);
    });

    function onDrag(e) {
        if (!dragging) return;

        // Distance from the mouse to the right edge = desired panel width
        const newWidth = Math.min(
            MAX_WIDTH,
            Math.max(MIN_WIDTH, window.innerWidth - e.clientX)
        );

        // Update the CSS variable — the grid reflows automatically 🎉
        layout.style.setProperty('--side-width', newWidth + 'px');
    }

    function stopDrag() {
        dragging = false;
        handle.classList.remove('dragging');
        document.removeEventListener('mousemove', onDrag);
        document.removeEventListener('mouseup', stopDrag);
    }
})();

// =================================================================
// ⌨️ KEYBOARD SHORTCUTS
// =================================================================
document.addEventListener('keydown', (e) => {
    // Ignore if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    // Number keys 1-9, H → select entity type
    const entities = Object.entries(SCHEMA.entities);
    for (const [name, info] of entities) {
        if (info.shortcut_key && e.key === info.shortcut_key) {
            e.preventDefault();
            selectEntity(name);
            return;
        }
    }

    // Delete/Backspace → delete last annotation
    if (e.key === 'Delete' || e.key === 'Backspace') {
        if (annotations.length > 0 && !e.target.closest('.anno-span')) {
            e.preventDefault();
            annotations.pop();
            renderText();
            showToast('🗑️ Deleted last annotation');
        }
    }

    // Ctrl+S → direct save as draft (no preview needed)
    if (e.ctrlKey && !e.shiftKey && e.key === 's') {
        e.preventDefault();
        saveAnnotations('in_progress');
    }

    // Ctrl+Shift+S → save AND validate via preview (for Mark Complete)
    if (e.ctrlKey && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        showSavePreview('completed');
    }

    // Escape → deselect entity type
    if (e.key === 'Escape') {
        activeEntityType = null;
        document.querySelectorAll('.entity-btn').forEach(b => b.classList.remove('active'));
    }
});

// =================================================================
// 🍞 TOAST NOTIFICATION
// =================================================================
/**
 * showToast(msg, type, duration)
 *
 * 💅 The glow-up that plain browser alert() always deserved!
 * Displays a beautiful animated toast notification in the bottom-right
 * corner. Uses the #toast-container div + CSS from the stylesheet.
 *
 * @param {string} msg      - Message text to display
 * @param {string} type     - 'success' | 'warning' | 'error' | 'info'
 * @param {number} duration - Auto-dismiss in ms (default: 4000)
 */
function showToast(msg, type = 'success', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    // Map type to an appropriate icon
    const icons = { success: '✅', warning: '⚠️', error: '❌', info: 'ℹ️' };
    const icon = icons[type] || '✅';

    // Build the toast element using the CSS classes from our stylesheet
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <div class="toast-content">
            <div class="toast-msg" style="color:var(--text-primary);font-size:0.83rem;line-height:1.5;">
                ${msg.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
            </div>
        </div>
        <button class="toast-close" onclick="this.closest('.toast')._dismiss()" title="Dismiss">✕</button>
        <div class="toast-progress" style="animation-duration:${duration}ms;"></div>
    `;

    // Attach dismiss helper directly to element for clean self-reference
    toast._dismiss = () => {
        if (toast._dismissed) return;
        toast._dismissed = true;
        clearTimeout(toast._timer);
        toast.classList.add('toast-hiding');
        toast.addEventListener('animationend', () => toast.remove(), { once: true });
    };

    container.appendChild(toast);

    // Auto-dismiss after duration
    toast._timer = setTimeout(() => toast._dismiss(), duration);
}

/**
 * showDraftSavedCelebration()
 * 
 * The cute little sister to the big completion party!
 * Quick, fabulous, and gone before you blink — perfect for draft saves.
 * Like the makeup artist giving you a mirror check backstage: "Slay, queen!"
 */
function showDraftSavedCelebration() {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position:fixed; inset:0; pointer-events:none; z-index:9999;
        display:flex; align-items:center; justify-content:center;
        animation: draftFade 2s ease forwards;
    `;

    // Tiny confetti (just 8 pieces, soft pastel colors)
    let confetti = '';
    const colors = ['#7ee8fa', '#eeb8ff', '#56d364', '#f778ba', '#e3b341'];
    for (let i = 0; i < 8; i++) {
        const c = colors[i % colors.length];
        const left = 30 + Math.random() * 40;
        confetti += `<div style="position:absolute; width:8px; height:8px; border-radius:50%; 
            background:${c}; left:${left}%; top:-20px; 
            animation: miniConfetti ${1.2 + Math.random()}s linear forwards;"></div>`;
    }

    overlay.innerHTML = `
        <div style="background:rgba(14,17,23,0.95); border:2px solid #7ee8fa; border-radius:9999px;
            padding:14px 32px; display:flex; align-items:center; gap:12px; box-shadow:0 0 30px #7ee8fa80;">
            ${confetti}
            <span style="font-size:2rem;">💾</span>
            <div>
                <div style="color:#7ee8fa; font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;">
                    DRAFT SAVED, BITCH! ✨
                </div>
                <div style="color:#8b949e; font-size:0.82rem;">Your work is locked & loaded, honey</div>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);

    // Auto-dismiss + click anywhere to dismiss
    setTimeout(() => overlay.remove(), 2200);
    overlay.addEventListener('click', () => overlay.remove(), {once: true});
}

// Add these keyframes once (put this inside your <style> or right after the other @keyframes)
const draftStyle = document.createElement('style');
draftStyle.textContent = `
    @keyframes draftFade { 0% {opacity:1; transform:scale(0.9)} 80% {opacity:1} 100% {opacity:0; transform:scale(1.05)} }
    @keyframes miniConfetti { to { transform:translateY(180px) rotate(720deg); opacity:0; } }
`;
document.head.appendChild(draftStyle);

/**
 * showCompletionCelebration()
 *
 * 🎊 THE GRAND FINALE — Full-screen celebration when marking a candidate Complete!
 * Auto-dismisses in 3.5s OR on any click/Escape keypress.
 * Like the final curtain call — lights, confetti, APPLAUSE! 🎭✨
 */
function showCompletionCelebration() {
    // Remove any stale overlay first (defensive coding, darling! 💅)
    const existing = document.getElementById('completionCelebration');
    if (existing) existing.remove();

    // Inject CSS keyframes just once
    if (!document.getElementById('celebStyles')) {
        const style = document.createElement('style');
        style.id = 'celebStyles';
        style.textContent = `
            @keyframes celebFadeIn  { from{opacity:0} to{opacity:1} }
            @keyframes celebFadeOut { from{opacity:1;transform:scale(1)} to{opacity:0;transform:scale(1.04)} }
            @keyframes celebCardIn  { from{transform:translateY(-30px) scale(0.9);opacity:0} to{transform:translateY(0) scale(1);opacity:1} }
            @keyframes celebPulse   { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
            @keyframes confettiFall { 0%{transform:translateY(-20px) rotate(0deg);opacity:1} 100%{transform:translateY(200px) rotate(720deg);opacity:0} }
        `;
        document.head.appendChild(style);
    }

    // Build the backdrop overlay
    const overlay = document.createElement('div');
    overlay.id = 'completionCelebration';
    overlay.style.cssText = `
        position:fixed; inset:0; background:rgba(0,0,0,0.82);
        z-index:9998; display:flex; align-items:center; justify-content:center;
        animation:celebFadeIn 0.3s ease forwards; cursor:pointer;
        backdrop-filter:blur(6px);
    `;

    // Generate confetti dots
    const colors = ['#56d364','#7ee8fa','#eeb8ff','#e3b341','#f778ba'];
    let confettiHTML = '';
    for (let i = 0; i < 14; i++) {
        const c = colors[i % colors.length];
        const sz = 6 + Math.floor(Math.random() * 8);
        const left = 4 + Math.floor(Math.random() * 92);
        const delay = (Math.random() * 1.8).toFixed(2);
        const dur = (1.4 + Math.random() * 1.2).toFixed(2);
        confettiHTML += `<div style="position:absolute;width:${sz}px;height:${sz}px;
            border-radius:50%;background:${c};left:${left}%;top:-12px;
            animation:confettiFall ${dur}s ${delay}s ease-in forwards;"></div>`;
    }

    // Build the card
    const card = document.createElement('div');
    card.style.cssText = `
        background: linear-gradient(135deg, #0e1117 0%, #1a2332 50%, #0e1117 100%);
        border: 1px solid rgba(86,211,100,0.45);
        border-radius: 20px; padding: 48px 56px; text-align: center;
        max-width: 440px; position: relative; overflow: hidden;
        box-shadow: 0 0 60px rgba(86,211,100,0.2), 0 24px 80px rgba(0,0,0,0.6);
        animation: celebCardIn 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards;
    `;
    card.innerHTML = `
        ${confettiHTML}
        <div style="font-size:3.8rem;margin-bottom:14px;animation:celebPulse 0.9s ease infinite;">🏆</div>
        <div style="font-family:'Syne',system-ui;font-size:1.6rem;font-weight:800;
            color:#56d364;letter-spacing:-0.5px;margin-bottom:10px;">
            Annotation Complete!
        </div>
        <div style="font-size:0.95rem;color:#8b949e;line-height:1.7;margin-bottom:22px;">
            This candidate is officially<br>
            <strong style="color:#e6edf3;">runway-ready</strong>
            and queued for export! ✨
        </div>
        <div style="display:flex;gap:10px;justify-content:center;
            font-size:0.72rem;color:#545d68;flex-wrap:wrap;">
            <span>✅ Saved to database</span>
            <span>·</span>
            <span>🎯 Status: Completed</span>
            <span>·</span>
            <span>📤 Ready to export</span>
        </div>
        <div style="margin-top:18px;font-size:0.65rem;color:#3a4250;">
            Click anywhere or press Escape to close
        </div>
    `;

    overlay.appendChild(card);
    document.body.appendChild(overlay);

    // Dismiss logic — click, Escape, or 3.5s auto-dismiss
    const dismiss = () => {
        if (!overlay.parentNode) return;
        overlay.style.animation = 'celebFadeOut 0.25s ease forwards';
        overlay.addEventListener('animationend', () => overlay.remove(), { once: true });
        document.removeEventListener('keydown', escHandler);
    };
    const escHandler = (e) => { if (e.key === 'Escape') dismiss(); };
    overlay.addEventListener('click', dismiss);
    document.addEventListener('keydown', escHandler);
    setTimeout(dismiss, 3500);
}

// =================================================================
// ❓ HELP MODAL — Open / Close + Shortcut Grid Builder
// =================================================================

/**
 * openHelp()
 * Opens the help manual modal by adding the 'open' class to the overlay.
 * Also traps Escape key to close it, like a civilised exit! 💅
 */
function openHelp() {
    document.getElementById('helpOverlay').classList.add('open');
    // Listen for Escape key to close the modal
    document.addEventListener('keydown', closeHelpOnEscape);
}

/**
 * closeHelp()
 * Hides the help modal and removes the Escape key listener.
 */
function closeHelp() {
    document.getElementById('helpOverlay').classList.remove('open');
    document.removeEventListener('keydown', closeHelpOnEscape);
}

/**
 * closeHelpOnBackdrop(event)
 * Closes the modal only if the user clicked the dark backdrop itself,
 * NOT if they clicked inside the white modal card.
 * Like leaving the party only when you WANT to — not when someone bumps you! 🎉
 *
 * @param {MouseEvent} event - The click event on the overlay div
 */
function closeHelpOnBackdrop(event) {
    // event.target is the element that was actually clicked.
    // If it's the overlay div (not the modal card inside), close.
    if (event.target === document.getElementById('helpOverlay')) {
        closeHelp();
    }
}

/**
 * closeHelpOnEscape(event)
 * Keyboard handler: closes the modal when Escape is pressed.
 * Added/removed dynamically so it doesn't interfere with annotation shortcuts
 * when the modal is closed.
 *
 * @param {KeyboardEvent} event
 */
function closeHelpOnEscape(event) {
    if (event.key === 'Escape') {
        closeHelp();
    }
}

/**
 * buildShortcutGrid()
 *
 * Reads the entity schema and injects each entity's keyboard shortcut
 * into the Help Modal shortcut grid — so the manual is always in sync
 * with the actual schema, no matter how many entities get added! 🎹
 *
 * Think of it as a self-updating setlist for our show. ✨
 */
function buildShortcutGrid() {
    const grid = document.getElementById('shortcutGrid');
    if (!grid) return;  // Modal might not be in the DOM yet — guard gracefully

    // Loop through the schema and add one grid item per entity with a shortcut
    for (const [name, info] of Object.entries(SCHEMA.entities)) {
        if (!info.shortcut_key) continue;  // Skip entities without shortcuts

        const item = document.createElement('div');
        item.className = 'shortcut-item';
        // Color the label to match the entity's colour in the palette
        item.innerHTML = `
            <span class="kbd">${info.shortcut_key}</span>
            <span style="color:${info.color}; font-weight:600;">${info.label}</span>
        `;
        grid.appendChild(item);
    }
}

// =================================================================
// 📏 IAA — INTER-ANNOTATOR AGREEMENT SYSTEM
// =================================================================
//
// This section handles:
//   1. Annotator name management (localStorage-backed)
//   2. IAA mode toggle (annotator B can label the same doc)
//   3. IAA dashboard with heatmap + metrics
//
// Architecture analogy: Think of IAA like having TWO fashion critics
// review the same runway show independently, then comparing their
// scorecards to see how often they agree. The heatmap shows which
// "categories" (entity types) cause the most disagreement! 👗📊
// =================================================================

/** @type {boolean} Whether we're currently in IAA annotation mode */
let iaaMode = false;

/** @type {boolean} Tracks whether IAA annotations have unsaved changes */
let iaaHasUnsavedChanges = false;

/** @type {string} Scope for the IAA dashboard: 'doc' or 'all' */
let iaaDashboardScope = 'all';

/**
 * setIAASaveStatus(state, spanCount?)
 *
 * Updates the save status indicator in the IAA banner.
 * Like a little scoreboard that always tells Annotator B
 * whether their work is safe! 📋✨
 *
 * States:
 *   'loading'  → ⏳ fetching previous annotations on mode enter
 *   'saved'    → ✅ N spans saved  (shown after successful save or on restore)
 *   'unsaved'  → ● Unsaved changes  (shown when annotations are modified)
 *   'saving'   → 💾 Saving...  (shown while the fetch is in-flight)
 *   'error'    → ⚠️ Save failed  (shown if save request errors out)
 */
function setIAASaveStatus(state, spanCount) {
    const el = document.getElementById('iaaSaveStatus');
    const btn = document.getElementById('iaaSaveBtn');
    if (!el) return;

    switch (state) {
        case 'loading':
            el.textContent = '⏳ Loading previous work...';
            el.style.color = 'var(--text-muted)';
            iaaHasUnsavedChanges = false;
            break;
        case 'saved':
            el.textContent = `✅ ${spanCount != null ? spanCount + ' span' + (spanCount !== 1 ? 's' : '') + ' ' : ''}saved`;
            el.style.color = 'var(--accent-green)';
            iaaHasUnsavedChanges = false;
            // Flash the save button green briefly as confirmation
            if (btn) {
                btn.style.borderColor = 'var(--accent-green)';
                btn.style.color = 'var(--accent-green)';
                setTimeout(() => {
                    btn.style.borderColor = 'var(--accent-pink)';
                    btn.style.color = 'var(--accent-pink)';
                }, 2500);
            }
            break;
        case 'unsaved':
            el.textContent = '● Unsaved changes';
            el.style.color = 'var(--accent-yellow)';
            iaaHasUnsavedChanges = true;
            break;
        case 'saving':
            el.textContent = '💾 Saving...';
            el.style.color = 'var(--text-muted)';
            break;
        case 'error':
            el.textContent = '⚠️ Save failed — try again';
            el.style.color = 'var(--accent-red)';
            iaaHasUnsavedChanges = true;
            break;
        default:
            el.textContent = '';
    }
}

// -----------------------------------------------------------------
// 👤 ANNOTATOR NAME — localStorage-backed identity
// -----------------------------------------------------------------

/**
 * getAnnotatorName()
 *
 * Returns the stored annotator name from localStorage, or empty string.
 * Used to populate the annotator field on saves and IAA sessions.
 *
 * @returns {string} The annotator's name
 */
function getAnnotatorName() {
    return localStorage.getItem('annotator_name') || '';
}

/**
 * setAnnotatorName(name)
 *
 * Stores the annotator name in localStorage for persistence across sessions.
 *
 * @param {string} name - The annotator's display name
 */
function setAnnotatorName(name) {
    localStorage.setItem('annotator_name', name.trim());
}

/**
 * initAnnotatorName()
 *
 * Called on page load. If no annotator name is stored, shows the
 * welcome prompt. Otherwise, silently proceeds.
 *
 * Like checking IDs at the door — if you've been here before,
 * you waltz right in. First-timers need to sign the guest book! 🎭
 */
function initAnnotatorName() {
    const name = getAnnotatorName();
    if (!name) {
        // Show the name prompt overlay
        document.getElementById('annotatorPromptOverlay').classList.add('open');
        // Auto-focus the input after a tiny delay (for animation)
        setTimeout(() => {
            document.getElementById('annotatorNameInput').focus();
        }, 200);
    }
}

/**
 * confirmAnnotatorName()
 *
 * Validates and saves the name from the prompt input.
 * Closes the overlay on success, shows error on empty.
 */
function confirmAnnotatorName() {
    const input = document.getElementById('annotatorNameInput');
    const errorEl = document.getElementById('annotatorNameError');
    const name = input.value.trim();

    if (!name) {
        errorEl.textContent = "Honey, we need a name! Even stage names count! 💅";
        errorEl.style.display = 'block';
        input.focus();
        return;
    }

    // Basic sanitization: alphanumeric + spaces + common chars
    if (name.length < 2) {
        errorEl.textContent = "That's too short, darling — at least 2 characters!";
        errorEl.style.display = 'block';
        input.focus();
        return;
    }

    if (name.length > 50) {
        errorEl.textContent = "Whoa, keep it under 50 characters, superstar! ✨";
        errorEl.style.display = 'block';
        input.focus();
        return;
    }

    setAnnotatorName(name);
    errorEl.style.display = 'none';
    document.getElementById('annotatorPromptOverlay').classList.remove('open');
    updateAnnotatorBadge();
    showToast(`👋 Welcome, ${name}! Your annotations will be tracked. ✨`, 'success', 4000);
}

/**
 * updateAnnotatorBadge()
 * Updates the 👤 badge in the header with the current annotator name.
 */
function updateAnnotatorBadge() {
    const badge = document.getElementById('annotatorBadgeName');
    if (!badge) return;
    const name = getAnnotatorName();
    badge.textContent = name || '(set name)';
    const container = document.getElementById('annotatorBadge');
    if (container) {
        container.style.borderStyle = name ? 'solid' : 'dashed';
        container.style.color = name ? 'var(--accent-cyan)' : 'var(--text-muted)';
    }
}

/**
 * changeAnnotatorName()
 * Quick inline name change via prompt — click the 👤 badge to trigger.
 */
function changeAnnotatorName() {
    const current = getAnnotatorName();
    const newName = prompt('Annotator name:', current || '');
    if (newName === null) return;
    if (newName.trim().length < 2) {
        showToast('Name needs at least 2 characters!', 'error', 3000);
        return;
    }
    setAnnotatorName(newName.trim());
    updateAnnotatorBadge();
    showToast(`👤 Annotator changed to "${newName.trim()}" ✨`, 'success', 3000);
}

// -----------------------------------------------------------------
// 📏 IAA MODE — Toggle between primary and IAA annotation
// -----------------------------------------------------------------

/** Stash of Annotator A's annotations — restored on exit */
let primaryAnnotationsBackup = null;

/**
 * enterIAAMode()
 *
 * THE FULL TRANSFORMATION:
 *   1. Backs up & HIDES Annotator A's annotations (clean slate, no bias)
 *   2. Adds body.iaa-active (pink theme overrides kick in)
 *   3. Dims normal Save/Complete buttons (use Save IAA instead)
 *   4. Shows the IAA banner (between toolbar and layout — VISIBLE!)
 *   5. Stores state in localStorage (persists across candidates)
 */
function enterIAAMode() {
    const annotatorName = getAnnotatorName();
    if (!annotatorName) {
        document.getElementById('annotatorPromptOverlay').classList.add('open');
        setTimeout(() => document.getElementById('annotatorNameInput').focus(), 200);
        return;
    }

    iaaMode = true;

    // ── Backup & clear primary annotations ───────────────────────────
    // We always back up Annotator A's work so we can restore it on exit.
    primaryAnnotationsBackup = JSON.parse(JSON.stringify(annotations));
    annotations = [];
    renderText();
    updateAnnoList();

    // ── Visual transformation ─────────────────────────────────────────
    document.body.classList.add('iaa-active');
    const normalBtns = document.getElementById('normalSaveButtons');
    if (normalBtns) normalBtns.classList.add('iaa-dimmed-btn');

    // ── Show banner ───────────────────────────────────────────────────
    const banner = document.getElementById('iaaBanner');
    banner.style.display = 'flex';
    document.getElementById('iaaBannerName').textContent = annotatorName;

    // Persist IAA mode across page loads / candidate navigation
    localStorage.setItem('iaa_mode_active', 'true');

    // ── THE FIX: Load previously saved IAA annotations for this doc ───
    // Without this, every time Annotator B re-enters IAA mode their
    // previous labels are gone — like re-painting a canvas you already
    // finished! This fetches their saved work from the DB and restores it.
    setIAASaveStatus('loading');  // Show a loading indicator immediately
    fetch(`/api/iaa/annotations/${DOC_ID}?annotator=${encodeURIComponent(annotatorName)}`)
        .then(res => res.json())
        .then(data => {
            if (data.found && data.annotations && data.annotations.length > 0) {
                // ✅ Previous IAA work found — restore it!
                annotations = data.annotations;
                renderText();
                updateAnnoList();
                setIAASaveStatus('saved', data.span_count);
                showToast(
                    `📏 IAA Mode ON — restored ${data.span_count} saved annotation` +
                    `${data.span_count !== 1 ? 's' : ''} by "${annotatorName}". ` +
                    `Continue labeling or save when done! ✨`,
                    'success', 5000
                );
            } else {
                // 🆕 No previous work — clean slate as expected
                setIAASaveStatus('unsaved');
                showToast(
                    `📏 IAA Mode ON — annotating as "${annotatorName}". ` +
                    `Primary annotations hidden. Label from scratch! ✨`,
                    'success', 5000
                );
            }
        })
        .catch(err => {
            // Network error loading previous annotations — warn but don't block
            console.warn('Could not load previous IAA annotations:', err);
            setIAASaveStatus('unsaved');
            showToast(
                `📏 IAA Mode ON — annotating as "${annotatorName}". ` +
                `(Could not load previous work: ${err}) ⚠️`,
                'warning', 5000
            );
        });
}

/**
 * exitIAAMode()
 *
 * Restores everything: annotations, theme, buttons, banner.
 */
function exitIAAMode() {
    iaaMode = false;

    // Restore primary annotations
    if (primaryAnnotationsBackup) {
        annotations = primaryAnnotationsBackup;
        primaryAnnotationsBackup = null;
        renderText();
        updateAnnoList();
    }

    // Reverse visual transformation
    document.body.classList.remove('iaa-active');
    const normalBtns = document.getElementById('normalSaveButtons');
    if (normalBtns) normalBtns.classList.remove('iaa-dimmed-btn');
    document.getElementById('iaaBanner').style.display = 'none';

    localStorage.removeItem('iaa_mode_active');

    showToast('📏 IAA Mode OFF — primary annotations restored! ✨', 'success', 3000);
}

/**
 * autoDetectIAAMode()
 *
 * Called on page load. If IAA was activated from the queue page,
 * auto-enters IAA mode so Annotator B gets a clean slate
 * on every candidate without clicking anything extra.
 */
function autoDetectIAAMode() {
    if (localStorage.getItem('iaa_mode_active') === 'true') {
        setTimeout(() => enterIAAMode(), 300);
    }
}

/**
 * saveIAAAnnotations()
 *
 * Saves the current annotation state as Annotator B's IAA set.
 * Routes to /api/iaa/save with the annotator name attached.
 *
 * Edge case: If annotations array is empty, warns and bails.
 */
function saveIAAAnnotations() {
    const annotatorName = getAnnotatorName();
    if (!annotatorName) {
        showToast('❌ No annotator name set — please enter your name first!', 'error', 4000);
        return;
    }

    if (!annotations || annotations.length === 0) {
        showToast('❌ No annotations to save — label some entities first! 🎨', 'error', 4000);
        return;
    }

    // Show saving state on the banner button
    setIAASaveStatus('saving');

    const payload = {
        doc_id: DOC_ID,
        annotator_name: annotatorName,
        annotations: annotations
    };

    fetch('/api/iaa/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => {
        if (!res.ok) {
            return res.json().catch(() => {
                throw new Error(`Server error ${res.status}`);
            }).then(errData => {
                throw new Error(errData.error || `HTTP ${res.status}`);
            });
        }
        return res.json();
    })
    .then(data => {
        if (data.success) {
            // ✅ Update the banner status to "saved" with span count
            setIAASaveStatus('saved', data.spans);
            showToast(
                `✅ IAA annotations saved! ${data.spans} span${data.spans !== 1 ? 's' : ''} ` +
                `by "${annotatorName}" — your labels are locked in, darling! 🏆`,
                'success', 5000
            );
        } else {
            setIAASaveStatus('error');
            showToast('❌ IAA save failed: ' + (data.error || 'Unknown'), 'error', 6000);
        }
    })
    .catch(err => {
        setIAASaveStatus('error');
        showToast('❌ Network error saving IAA: ' + err, 'error', 6000);
    });
}


// -----------------------------------------------------------------
// 📏📊 IAA DASHBOARD — Modal with heatmap + metrics
// -----------------------------------------------------------------

/**
 * openIAAModal()
 *
 * Opens the IAA dashboard modal and fetches data based on current scope.
 * Default scope is 'all' (aggregate across all dual-annotated docs).
 *
 * Like opening the judges' scorecards at the end of the competition! 🏆
 */
function openIAAModal() {
    openInspector('iaa');
    loadIAADashboard();
}

/**
 * toggleIAAScope()
 *
 * Switches between 'This Doc' and 'All Docs' views in the dashboard.
 */
function toggleIAAScope() {
    const btn = document.getElementById('iaaToggleScope');
    if (iaaDashboardScope === 'all') {
        iaaDashboardScope = 'doc';
        btn.textContent = '📊 All Docs';
        btn.title = 'Show aggregate metrics across all documents';
    } else {
        iaaDashboardScope = 'all';
        btn.textContent = '🔍 This Doc';
        btn.title = 'Show metrics for this document only';
    }
    loadIAADashboard();
}

/**
 * loadIAADashboard()
 *
 * Fetches IAA data from the appropriate endpoint and renders it.
 */
function loadIAADashboard() {
    const body = document.getElementById('iaaModalBody');
    body.innerHTML = '<div style="text-align:center; padding:40px; color:var(--text-muted);">Loading IAA metrics... ⏳</div>';

    const url = iaaDashboardScope === 'doc'
        ? `/api/iaa/compute/${DOC_ID}`
        : '/api/iaa/dashboard';

    fetch(url)
        .then(res => {
            if (!res.ok) {
                return res.json().catch(() => {
                    throw new Error(`Server error ${res.status} (${res.statusText})`);
                }).then(errData => {
                    throw new Error(errData.error || errData.message || `HTTP ${res.status}`);
                });
            }
            return res.json();
        })
        .then(data => {
            if (data.status === 'error') {
                body.innerHTML = `<div style="text-align:center; padding:40px; color:var(--accent-red);">
                    ❌ IAA Error: ${escapeHtml(data.error || data.message || 'Unknown error')}<br>
                    <small style="color:var(--text-muted); margin-top:8px; display:block;">
                        Check the server logs for the full traceback.
                    </small></div>`;
            } else if (data.status === 'no_data') {
                renderIAAEmptyState(body, data.message);
            } else if (iaaDashboardScope === 'doc') {
                renderIAASingleDoc(body, data);
            } else {
                renderIAAAllDocs(body, data);
            }
        })
        .catch(err => {
            body.innerHTML = `<div style="text-align:center; padding:40px; color:var(--accent-red);">
                ❌ Failed to load IAA data: ${escapeHtml(String(err))}</div>`;
        });
}

/**
 * renderIAAEmptyState(container, message)
 *
 * Shows a friendly empty state when no IAA data exists yet.
 */
function renderIAAEmptyState(container, message) {
    container.innerHTML = `
        <div class="iaa-empty-state">
            <div class="iaa-empty-icon">📏</div>
            <h4>No IAA Data Yet</h4>
            <p>${escapeHtml(message || 'No documents have been dual-annotated yet.')}</p>
            <div style="margin-top:18px; font-size:0.72rem; color:var(--text-secondary); line-height:1.8;">
                <strong>How to start IAA:</strong><br>
                1. Have Annotator A complete annotations normally<br>
                2. A different annotator enters <strong>IAA Mode</strong> (📏 button)<br>
                3. Annotator B labels the same document independently<br>
                4. Click <strong>💾 Save IAA</strong> to store their labels<br>
                5. Come back here to see agreement metrics! 🏆
            </div>
            <div style="margin-top:16px;">
                <button class="btn btn-sm" onclick="closeInspector('iaa'); enterIAAMode();"
                        style="border-color:var(--accent-pink); color:var(--accent-pink); padding:6px 18px;">
                    📏 Enter IAA Mode
                </button>
            </div>
        </div>
    `;
}

/**
 * kappaInterpretation(k)
 *
 * Returns a human-readable label + CSS class for a Kappa value.
 * Uses Landis & Koch (1977) interpretation scale.
 *
 * Think of it as the Michelin star rating for agreement:
 *   ⭐⭐⭐ Almost Perfect  (0.81 - 1.00)
 *   ⭐⭐   Substantial     (0.61 - 0.80)
 *   ⭐     Moderate        (0.41 - 0.60)
 *   (meh)  Fair            (0.21 - 0.40)
 *   (yikes) Slight/Poor    (≤ 0.20)
 *
 * @param {number} k - Cohen's Kappa value (-1 to 1)
 * @returns {Object} label (string), cls (string)
 */
function kappaInterpretation(k) {
    if (k >= 0.81) return { label: 'Almost Perfect', cls: 'kappa-almost-perfect' };
    if (k >= 0.61) return { label: 'Substantial',    cls: 'kappa-substantial' };
    if (k >= 0.41) return { label: 'Moderate',       cls: 'kappa-moderate' };
    if (k >= 0.21) return { label: 'Fair',           cls: 'kappa-fair' };
    return { label: 'Poor/Slight', cls: 'kappa-poor' };
}

/**
 * f1Color(f1)
 *
 * Returns a CSS color for an F1 value — green for high, red for low.
 * Used for both heatmap cells and inline score displays.
 *
 * @param {number} f1 - F1 score (0.0 to 1.0)
 * @returns {string} CSS color string
 */
function f1Color(f1) {
    if (f1 >= 0.8)  return 'rgba(86,211,100,0.85)';   // Green — great!
    if (f1 >= 0.6)  return 'rgba(126,232,250,0.75)';   // Cyan — decent
    if (f1 >= 0.4)  return 'rgba(227,179,65,0.75)';    // Yellow — meh
    if (f1 >= 0.2)  return 'rgba(240,136,62,0.75)';    // Orange — concerning
    return 'rgba(218,54,51,0.7)';                       // Red — yikes
}

/**
 * f1Background(f1)
 *
 * Returns a lighter background color for heatmap cells.
 * Lower opacity than f1Color so text remains readable.
 *
 * @param {number} f1 - F1 score (0.0 to 1.0)
 * @returns {string} CSS background color
 */
function f1Background(f1) {
    if (f1 >= 0.8)  return 'rgba(86,211,100,0.2)';
    if (f1 >= 0.6)  return 'rgba(126,232,250,0.15)';
    if (f1 >= 0.4)  return 'rgba(227,179,65,0.15)';
    if (f1 >= 0.2)  return 'rgba(240,136,62,0.15)';
    return 'rgba(218,54,51,0.15)';
}

/**
 * renderIAASingleDoc(container, data)
 *
 * Renders IAA metrics for a single document.
 * Shows span-level and token-level agreement with per-entity breakdown.
 */
function renderIAASingleDoc(container, data) {
    const span = data.span_agreement || {};
    const kappa = data.token_kappa || {};
    const exact = span.exact || {};
    const partial = span.partial || {};
    const perEntity = span.per_entity || {};

    // KPI cards
    const kappaVal = kappa.kappa != null ? kappa.kappa : null;
    const kappaInterp = kappaVal != null ? kappaInterpretation(kappaVal) : null;

    let html = `
        <div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:12px;">
            Comparing primary annotations vs <strong style="color:var(--accent-pink);">${escapeHtml(data.annotator_b || '?')}</strong>
            on document <strong>${escapeHtml(data.doc_id || '')}</strong>
        </div>

        <div class="iaa-kpi-row">
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${kappaVal != null ? f1Color(kappaVal) : 'var(--text-muted)'}">
                    ${kappaVal != null ? kappaVal.toFixed(3) : '—'}
                </div>
                <div class="iaa-kpi-label">Cohen's Kappa</div>
                ${kappaInterp ? `<div class="kappa-badge ${kappaInterp.cls}" style="margin-top:4px;">${kappaInterp.label}</div>` : ''}
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(exact.f1 || 0)}">
                    ${(exact.f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Exact F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(partial.f1 || 0)}">
                    ${(partial.f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Partial F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:var(--text-primary)">
                    ${kappa.tokens || 0}
                </div>
                <div class="iaa-kpi-label">Tokens Compared</div>
            </div>
        </div>
    `;

    // Per-entity breakdown as mini heatmap
    const entityNames = Object.keys(perEntity).sort();
    if (entityNames.length > 0) {
        html += buildEntityHeatmapSection(perEntity, entityNames, data);
    }

    container.innerHTML = html;
}

/**
 * renderIAAAllDocs(container, data)
 *
 * Renders aggregate IAA metrics across all dual-annotated documents.
 * Shows overall KPIs, per-entity heatmap, and per-document table.
 */
function renderIAAAllDocs(container, data) {
    const agg = data.aggregate || {};
    const docs = data.docs || [];
    const perEntity = agg.per_entity || {};
    const entityColors = data.entity_colors || {};
    const entityLabels = data.entity_labels || {};

    // Overall KPIs
    const kappaVal = agg.avg_kappa != null ? agg.avg_kappa : null;
    const kappaInterp = kappaVal != null ? kappaInterpretation(kappaVal) : null;

    let html = `
        <div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:12px;">
            Aggregate across <strong style="color:var(--accent-cyan);">${agg.total_docs || 0}</strong> dual-annotated documents
        </div>

        <div class="iaa-kpi-row">
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${kappaVal != null ? f1Color(kappaVal) : 'var(--text-muted)'}">
                    ${kappaVal != null ? kappaVal.toFixed(3) : '—'}
                </div>
                <div class="iaa-kpi-label">Avg Cohen's Kappa</div>
                ${kappaInterp ? `<div class="kappa-badge ${kappaInterp.cls}" style="margin-top:4px;">${kappaInterp.label}</div>` : ''}
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(agg.avg_exact_f1 || 0)}">
                    ${(agg.avg_exact_f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Avg Exact F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:${f1Color(agg.avg_partial_f1 || 0)}">
                    ${(agg.avg_partial_f1 || 0).toFixed(3)}
                </div>
                <div class="iaa-kpi-label">Avg Partial F1</div>
            </div>
            <div class="iaa-kpi-card">
                <div class="iaa-kpi-value" style="color:var(--text-primary)">
                    ${agg.total_docs || 0}
                </div>
                <div class="iaa-kpi-label">Documents</div>
            </div>
        </div>
    `;

    // ── Per-Entity Heatmap ────────────────────────────────────────
    // The STAR of the show! 🌟 A color-coded grid showing F1, Precision,
    // and Recall for each entity type. Red = disagreement, Green = harmony.
    const entityNames = Object.keys(perEntity).sort();
    if (entityNames.length > 0) {
        const metrics = ['avg_f1', 'avg_precision', 'avg_recall'];
        const metricLabels = { avg_f1: 'F1', avg_precision: 'Prec', avg_recall: 'Rec' };

        html += `
            <div style="margin-bottom:6px;">
                <h4 style="font-family:var(--font-display); font-size:0.85rem; color:var(--text-primary); margin-bottom:4px;">
                    🔥 Agreement Heatmap by Entity Type
                </h4>
                <div style="font-size:0.62rem; color:var(--text-muted);">
                    Hover for details · 🟢 High agreement · 🔴 Low agreement
                </div>
            </div>
            <div class="iaa-heatmap" style="grid-template-columns: 140px repeat(${metrics.length}, 1fr);">
                <!-- Header row -->
                <div class="iaa-heatmap-header"></div>
                ${metrics.map(m => `<div class="iaa-heatmap-header">${metricLabels[m]}</div>`).join('')}

                <!-- Data rows -->
                ${entityNames.map(eName => {
                    const eData = perEntity[eName];
                    const label = entityLabels[eName] || eName;
                    const color = entityColors[eName] || '#8b949e';
                    return `
                        <div class="iaa-heatmap-row-label" title="${eName}">
                            <span style="display:inline-block; width:8px; height:8px; border-radius:2px;
                                         background:${color}; margin-right:5px; flex-shrink:0;"></span>
                            ${escapeHtml(label)}
                        </div>
                        ${metrics.map(m => {
                            const val = eData[m] || 0;
                            return `<div class="iaa-heatmap-cell"
                                         style="background:${f1Background(val)}; color:${f1Color(val)};"
                                         title="${label} — ${metricLabels[m]}: ${val.toFixed(3)} (${eData.doc_count || 0} docs)">
                                        ${val.toFixed(2)}
                                    </div>`;
                        }).join('')}
                    `;
                }).join('')}
            </div>
        `;
    }

    // ── Per-Document Table ─────────────────────────────────────────
    if (docs.length > 0) {
        html += `
            <h4 style="font-family:var(--font-display); font-size:0.85rem; color:var(--text-primary);
                        margin-top:20px; margin-bottom:8px;">
                📋 Per-Document Breakdown
            </h4>
            <div style="max-height:250px; overflow-y:auto; border:1px solid var(--border); border-radius:8px;">
                <table class="iaa-doc-table">
                    <thead>
                        <tr>
                            <th>Doc ID</th>
                            <th>Annotator B</th>
                            <th>Exact F1</th>
                            <th>Partial F1</th>
                            <th>Kappa</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${docs.map(doc => {
                            const sa = doc.span_agreement || {};
                            const ex = sa.exact || {};
                            const pa = sa.partial || {};
                            const tk = doc.token_kappa || {};
                            const kVal = tk.kappa != null ? tk.kappa : null;
                            const kInterp = kVal != null ? kappaInterpretation(kVal) : null;
                            return `
                                <tr>
                                    <td><a href="/annotate/${doc.candidate_id || 0}" style="color:var(--accent-cyan);">
                                        ${escapeHtml(doc.doc_id)}</a></td>
                                    <td>${escapeHtml(doc.annotator_b || '—')}</td>
                                    <td style="color:${f1Color(ex.f1 || 0)}">${(ex.f1 || 0).toFixed(3)}</td>
                                    <td style="color:${f1Color(pa.f1 || 0)}">${(pa.f1 || 0).toFixed(3)}</td>
                                    <td>
                                        ${kVal != null ? `<span style="color:${f1Color(kVal)}">${kVal.toFixed(3)}</span>` : '—'}
                                        ${kInterp ? `<span class="kappa-badge ${kInterp.cls}" style="margin-left:4px;">${kInterp.label}</span>` : ''}
                                    </td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    // ── IAA Mode CTA ──────────────────────────────────────────────
    html += `
        <div style="margin-top:18px; padding-top:14px; border-top:1px solid var(--border);
                     display:flex; gap:8px; align-items:center;">
            <button class="btn btn-sm" onclick="enterIAAMode(); closeInspector('iaa');"
                    style="border-color:var(--accent-pink); color:var(--accent-pink); padding:6px 16px;">
                📏 Enter IAA Mode
            </button>
            <span style="font-size:0.65rem; color:var(--text-muted);">
                Annotate this document as a second reviewer
            </span>
        </div>
    `;

    container.innerHTML = html;
}

/**
 * buildEntityHeatmapSection(perEntity, entityNames, data)
 *
 * Builds the per-entity heatmap section for a SINGLE document view.
 * Shows F1, Precision, and Recall for each entity type.
 *
 * @param {Object} perEntity - { ENTITY_NAME: { f1, precision, recall, count_a, count_b } }
 * @param {string[]} entityNames - Sorted entity type names
 * @param {Object} data - Full API response (for colors/labels)
 * @returns {string} HTML string
 */
function buildEntityHeatmapSection(perEntity, entityNames, data) {
    const entityColors = (data && data.entity_colors) || COLOR_MAP || {};
    const metrics = ['f1', 'precision', 'recall'];
    const metricLabels = { f1: 'F1', precision: 'Prec', recall: 'Rec' };

    let html = `
        <div style="margin-bottom:6px;">
            <h4 style="font-family:var(--font-display); font-size:0.85rem; color:var(--text-primary); margin-bottom:4px;">
                🔥 Agreement by Entity Type
            </h4>
            <div style="font-size:0.62rem; color:var(--text-muted);">
                Hover for span counts · 🟢 High · 🔴 Low
            </div>
        </div>
        <div class="iaa-heatmap" style="grid-template-columns: 140px repeat(${metrics.length}, 1fr);">
            <div class="iaa-heatmap-header"></div>
            ${metrics.map(m => `<div class="iaa-heatmap-header">${metricLabels[m]}</div>`).join('')}
            ${entityNames.map(eName => {
                const eData = perEntity[eName];
                const label = (SCHEMA.entities[eName] || {}).label || eName;
                const color = entityColors[eName] || '#8b949e';
                return `
                    <div class="iaa-heatmap-row-label" title="${eName}">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:2px;
                                     background:${color}; margin-right:5px; flex-shrink:0;"></span>
                        ${escapeHtml(label)}
                    </div>
                    ${metrics.map(m => {
                        const val = eData[m] || 0;
                        const tooltip = `${label} ${metricLabels[m]}: ${val.toFixed(3)} (A:${eData.count_a||0} / B:${eData.count_b||0} spans)`;
                        return `<div class="iaa-heatmap-cell"
                                     style="background:${f1Background(val)}; color:${f1Color(val)};"
                                     title="${tooltip}">
                                    ${val.toFixed(2)}
                                </div>`;
                    }).join('')}
                `;
            }).join('')}
        </div>
    `;
    return html;
}

// =================================================================
// 🚀 INIT
// =================================================================
buildPalette();
buildShortcutGrid();
initAnnotatorName();   // 👤 Prompt for annotator name on first visit
updateAnnotatorBadge(); // 👤 Show current annotator in header badge
renderText();
autoDetectIAAMode();   // 📏 Auto-enter IAA mode if activated from queue page

// Populate classification dropdowns with stored values from server
// NOTE: Using tojson filter to prevent Jinja2 HTML-escaping '&' chars
// in values like "Accounting & Finance" → would become "&amp;" and break matching!
populateClassification(
    {{ stored_function | tojson }},
    {{ stored_industry | tojson }},
    {{ stored_hard_skills | tojson }},
    {{ stored_soft_skills | tojson }},
    {{ stored_tags | tojson }}
);
</script>
</body>
</html>
"""


# =============================================================================
# 🚀 APP STARTUP
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🏷️✨ FAIRY CODEMOTHER'S NER ANNOTATION TOOL ✨🏷️")
    print("=" * 60)
    print(f"  📂 Database: {DATABASE_PATH}")
    print(f"  🌐 URL: http://localhost:{PORT}")
    print(f"  📋 Entity types: {len(schema.entities)}")
    print(f"  🏷️ BIO tags: {len(schema.get_bio_tags())}")
    print("=" * 60)

    if not os.path.exists(DATABASE_PATH):
        print(f"\n  ❌ Database not found: {DATABASE_PATH}")
        print("  💡 Run your extraction pipeline first!")
        exit(1)

    app.run(host=HOST, port=PORT, debug=DEBUG)