"""Read-only DB helpers: annotation queue, raw text, structured extraction rows."""
import sqlite3

from typing import Dict

from annotation.app import get_db


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
                   nd.annotator as ann_by,
                   iaa_sub.iaa_annotators,
                   iaa_sub.iaa_spans
            FROM structured_extractions s
            INNER JOIN ner_documents nd
                ON s.candidate_id = nd.candidate_id
            LEFT JOIN (
                SELECT candidate_id,
                       GROUP_CONCAT(DISTINCT annotator_name) as iaa_annotators,
                       COUNT(*) as iaa_spans
                FROM iaa_annotations
                GROUP BY candidate_id
            ) iaa_sub ON iaa_sub.candidate_id = s.candidate_id
            ORDER BY s.candidate_id
            LIMIT ? OFFSET ?
        """, (per_page, offset)).fetchall()
    elif status_filter == "pending":
        rows = conn.execute("""
            SELECT s.candidate_id, s.name, s.email,
                   s.extraction_status, NULL as ann_status,
                   NULL as ann_by,
                   NULL as iaa_annotators,
                   NULL as iaa_spans
            FROM structured_extractions s
            WHERE NOT EXISTS (
                SELECT 1 FROM ner_documents nd
                WHERE nd.candidate_id = s.candidate_id
            )
            ORDER BY s.candidate_id
            LIMIT ? OFFSET ?
        """, (per_page, offset)).fetchall()
    else:
        rows = conn.execute("""
            SELECT s.candidate_id, s.name, s.email,
                   s.extraction_status, nd.status as ann_status,
                   nd.annotator as ann_by,
                   iaa_sub.iaa_annotators,
                   iaa_sub.iaa_spans
            FROM structured_extractions s
            LEFT JOIN ner_documents nd
                ON s.candidate_id = nd.candidate_id
            LEFT JOIN (
                SELECT candidate_id,
                       GROUP_CONCAT(DISTINCT annotator_name) as iaa_annotators,
                       COUNT(*) as iaa_spans
                FROM iaa_annotations
                GROUP BY candidate_id
            ) iaa_sub ON iaa_sub.candidate_id = s.candidate_id
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
def get_raw_text_for_candidate(candidate_id: int) -> str:
    """Fetch raw resume text from the database."""
    conn = get_db()
    row = conn.execute("""
        SELECT raw_text FROM raw_extractions
        WHERE candidate_id = ?
        ORDER BY extraction_timestamp DESC LIMIT 1
    """, (candidate_id,)).fetchone()
    return row["raw_text"] if row else ""
def get_structured_data(candidate_id: int) -> Dict:
    """Get structured extraction data for reference."""
    conn = get_db()
    row = conn.execute("""
        SELECT * FROM structured_extractions
        WHERE candidate_id = ?
        ORDER BY created_at DESC LIMIT 1
    """, (candidate_id,)).fetchone()
    return dict(row) if row else {}
