"""
ml_quality_audit.py

💅✨ FAIRY CODEMOTHER'S ML QUALITY AUDIT ENGINE ✨💅

A Senior ML QA Engineer in a Python module — performs comprehensive
performance audits on your LLM resume extraction pipeline by comparing
model output against human-reviewed ground truth.

Think of this as the HEAD JUDGE at the AI Beauty Pageant — she doesn't
just say "nice dress", she scores the walk, the talk, the poise, and
writes a 10-page improvement plan with receipts! 👠📋🔬

Features:
  1. Core NER Metrics    — Precision, Recall, F1 per field + weighted avg
  2. Error Taxonomy      — Boundary / Type / Missing / Spurious classification
  3. Field Sensitivity   — Weakest link identification with root cause
  4. Textual Similarity  — Levenshtein distance + ratio for fuzzy matches
  5. Improvement Roadmap — Specific remedies based on error patterns

Architecture:
  This is a PURE ANALYSIS MODULE — no Flask, no database connections.
  It takes pre-fetched data dicts and returns structured results.
  The accuracy_validator_api.py imports this and wires it to endpoints.

  ┌─────────────────────────────┐
  │  accuracy_validator_api.py  │
  │  (Flask + DB + Dashboard)   │
  │         │                   │
  │         ▼                   │
  │  ml_quality_audit.py        │  ← THIS FILE
  │  (Pure analysis engine)     │
  └─────────────────────────────┘

Usage (standalone for testing):
    from ml_quality_audit import run_full_audit
    result = run_full_audit(reviews, extractions, tracked_fields)

Dependencies:
    None! Pure Python — no external packages needed. 💅
    (Levenshtein is implemented from scratch because we're THAT independent.)
"""

import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# 📏 TEXTUAL SIMILARITY — Levenshtein Distance (Pure Python)
# =============================================================================
# Why build our own? Because installing python-Levenshtein just for one
# function is like hiring a full orchestra when you only need a triangle! 🔺
# Our implementation handles the common cases efficiently enough for
# resume-scale texts (typically < 1000 chars per field).

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    📏 Compute the Levenshtein (edit) distance between two strings.

    The minimum number of single-character edits (insertions, deletions,
    substitutions) needed to transform s1 into s2.

    Like counting how many alterations a tailor needs to make to turn
    one dress into another — fewer edits = more similar! 👗✂️

    Uses the optimized two-row Wagner-Fischer algorithm:
    Time: O(m*n), Space: O(min(m,n)) where m,n = string lengths.

    Args:
        s1: First string (model output)
        s2: Second string (ground truth)

    Returns:
        Integer edit distance (0 = identical)
    """
    # Edge cases — the warm-up act before the main show! 🎭
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    # Ensure s1 is the shorter string for space optimization
    # Like putting the shorter dancer in front — efficiency! 💃
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)

    # Two-row approach: only keep current and previous row
    prev_row = list(range(m + 1))
    curr_row = [0] * (m + 1)

    for j in range(1, n + 1):
        curr_row[0] = j
        for i in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                curr_row[i - 1] + 1,       # Insertion
                prev_row[i] + 1,            # Deletion
                prev_row[i - 1] + cost,     # Substitution
            )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[m]


def similarity_ratio(s1: str, s2: str) -> float:
    """
    📊 Compute a 0.0–1.0 similarity ratio between two strings.

    Uses: 1.0 - (levenshtein_distance / max_length)

    1.0 = identical, 0.0 = completely different.
    Like a compatibility score on a dating app — but for text! 💕📊

    Args:
        s1: First string
        s2: Second string

    Returns:
        Float between 0.0 and 1.0
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    dist = levenshtein_distance(s1, s2)
    return round(1.0 - (dist / max_len), 4)


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    📏 Levenshtein distance normalized to 0.0–1.0 range.

    0.0 = identical, 1.0 = completely different.
    Inverse of similarity_ratio — some people prefer their
    scores like golf (lower is better)! ⛳

    Args:
        s1: First string
        s2: Second string

    Returns:
        Float between 0.0 and 1.0
    """
    return round(1.0 - similarity_ratio(s1, s2), 4)


# =============================================================================
# 🧬 HELPER: Value normalization for comparison
# =============================================================================

def _normalize_for_comparison(value: Any) -> str:
    """
    Normalize a field value into a comparable string.

    Handles None, lists, dicts, and strings uniformly.
    Like getting everyone into the same costume before
    the group photo — standardization! 📸
    """
    if value is None:
        return ""
    if isinstance(value, list):
        # Sort for order-independent comparison
        return ", ".join(sorted(str(v) for v in value if v))
    if isinstance(value, dict):
        # Flatten dict to key=value pairs
        return ", ".join(f"{k}={v}" for k, v in sorted(value.items()) if v)
    return str(value).strip()


def _is_boundary_error(model_val: str, truth_val: str) -> bool:
    """
    🔍 Detect if the error is a boundary/truncation issue.

    A boundary error means the model extracted a SUBSTRING of the truth
    (clipped text) or the truth is a substring of the model output
    (extra text grabbed). Like cutting the hem too short or too long! ✂️

    Examples:
      - "Alex Riv" vs "Alex Rivier" → True (model clipped the name)
      - "Computer Science" vs "Bachelor of Science in Computer Science" → True
      - "555-0199" vs "+1-555-0199" → True (country code boundary)

    Args:
        model_val: What the model extracted
        truth_val: The ground truth value

    Returns:
        True if one is a meaningful substring of the other
    """
    if not model_val or not truth_val:
        return False

    m = model_val.lower().strip()
    t = truth_val.lower().strip()

    # One must be a substring of the other
    if m in t or t in m:
        # But not an exact match (that would be correct, not a boundary error)
        if m != t:
            # The substring must be at least 40% of the longer string
            # to count as boundary (not just a random word overlap)
            shorter = min(len(m), len(t))
            longer = max(len(m), len(t))
            if shorter / longer >= 0.4:
                return True

    return False


def _is_type_error(field_key: str, model_val: str, truth_val: str) -> bool:
    """
    🔀 Detect if the error is a type/mislabeling issue.

    A type error means the model confused the semantic category:
    e.g., put a company name in the school field, or a job title
    in the skill field. Like wearing a ball gown to a beach party! 🏖️👗

    This is harder to detect automatically, so we use heuristics:
    - Very low similarity + both non-empty = likely type confusion
    - Field contains content that belongs to a different field category
    """
    if not model_val or not truth_val:
        return False

    sim = similarity_ratio(model_val.lower(), truth_val.lower())

    # If similarity is very low (< 0.3) but both are non-empty,
    # the model likely extracted something from the wrong field
    if sim < 0.3 and len(model_val) > 3 and len(truth_val) > 3:
        return True

    return False


# =============================================================================
# 🔬 CORE ANALYSIS ENGINE
# =============================================================================

def compute_ner_metrics(
    reviews: List[Dict],
    tracked_fields: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    🎯 Compute Precision, Recall, and F1-Score for each field.

    This is the SCORECARD — the numbers that tell you exactly how
    your AI model is performing on each extraction task! 📊👑

    Verdict → Metric Mapping:
    ┌───────────┬────┬────┬────┐
    │ Verdict   │ TP │ FP │ FN │
    ├───────────┼────┼────┼────┤
    │ correct   │ 1  │ 0  │ 0  │  Model got it right! ✅
    │ partial   │0.5 │0.5 │0.5 │  Half credit — close but not perfect ⚠️
    │ wrong     │ 0  │ 1  │ 0  │  Model extracted garbage 🗑️
    │ missing   │ 0  │ 0  │ 1  │  Model missed it entirely 👻
    │ skip      │ —  │ —  │ —  │  Excluded from metrics ⏭️
    └───────────┴────┴────┴────┘

    Partial gets HALF credit for both TP and FP+FN because the model
    DID extract something (partially right = partial true positive),
    but it also partially failed (partial false positive + partial miss).

    Args:
        reviews: List of review dicts with field_key, verdict, correction, etc.
        tracked_fields: Field configuration dict from TRACKED_FIELDS

    Returns:
        Dict with per-field and overall metrics
    """
    # ── Accumulate counts per field ────────────────────────────────────
    field_counts = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "total": 0})

    for review in reviews:
        fk = review.get("field_key", "")
        verdict = review.get("verdict", "")

        if verdict == "skip":
            continue

        field_counts[fk]["total"] += 1

        if verdict == "correct":
            field_counts[fk]["tp"] += 1.0
        elif verdict == "partial":
            field_counts[fk]["tp"] += 0.5
            field_counts[fk]["fp"] += 0.5
            field_counts[fk]["fn"] += 0.5
        elif verdict == "wrong":
            field_counts[fk]["fp"] += 1.0
        elif verdict == "missing":
            field_counts[fk]["fn"] += 1.0

    # ── Compute P/R/F1 per field ──────────────────────────────────────
    field_metrics = {}
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_weight = 0.0

    for field_key, counts in field_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        field_info = tracked_fields.get(field_key, {})
        is_critical = field_info.get("critical", False)

        # Weight: critical fields count double in the weighted average
        # Because getting the name wrong is worse than getting hobbies
        # wrong — priorities, darling! 👑
        weight = 2.0 if is_critical else 1.0
        total_weight += weight

        field_metrics[field_key] = {
            "label": field_info.get("label", field_key),
            "critical": is_critical,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": round(tp, 1),
            "fp": round(fp, 1),
            "fn": round(fn, 1),
            "support": counts["total"],
            "weight": weight,
        }

    # ── Micro-averaged overall metrics ─────────────────────────────────
    # Micro = pool all TP/FP/FN across fields, then compute
    # This gives more weight to fields with more reviews
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    # ── Weighted-average (by field weight) ─────────────────────────────
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f1 = 0.0

    for fk, fm in field_metrics.items():
        w = fm["weight"]
        weighted_p += fm["precision"] * w
        weighted_r += fm["recall"] * w
        weighted_f1 += fm["f1"] * w

    if total_weight > 0:
        weighted_p /= total_weight
        weighted_r /= total_weight
        weighted_f1 /= total_weight

    return {
        "field_metrics": field_metrics,
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
            "total_tp": round(total_tp, 1),
            "total_fp": round(total_fp, 1),
            "total_fn": round(total_fn, 1),
        },
        "weighted": {
            "precision": round(weighted_p, 4),
            "recall": round(weighted_r, 4),
            "f1": round(weighted_f1, 4),
        },
        "total_reviewed": sum(c["total"] for c in field_counts.values()),
    }


def classify_errors(
    reviews: List[Dict],
    extractions: Dict[int, Dict],
    tracked_fields: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    🏷️ Classify every discrepancy into the Error Taxonomy.

    Categories:
      🔲 Boundary Error  — Clipped/extended text (substring relationship)
      🔀 Type Error       — Semantic mislabeling (wrong field category)
      👻 Missing (FN)     — Model failed to extract an existing field
      🎪 Spurious (FP)    — Model hallucinated or extracted noise

    Each error gets a detailed entry with the model value, ground truth,
    fuzzy score, and classification reasoning. Like a forensic report
    for each wardrobe malfunction! 🔍👗

    Args:
        reviews: List of review dicts (must include candidate_id, field_key,
                 verdict, correction)
        extractions: Dict mapping candidate_id → structured extraction dict
        tracked_fields: Field configuration from TRACKED_FIELDS

    Returns:
        Dict with error list, counts by category, and per-field breakdown
    """
    errors = []
    taxonomy_counts = {
        "boundary": 0,
        "type_error": 0,
        "missing": 0,
        "spurious": 0,
    }
    field_taxonomy = defaultdict(lambda: {"boundary": 0, "type_error": 0, "missing": 0, "spurious": 0})

    for review in reviews:
        verdict = review.get("verdict", "")
        if verdict in ("correct", "skip"):
            continue

        fk = review.get("field_key", "")
        cid = review.get("candidate_id")
        correction = review.get("correction", "").strip()

        # Get the model's extracted value for this candidate + field
        field_info = tracked_fields.get(fk, {})
        db_col = field_info.get("db_col", fk)

        extraction = extractions.get(cid, {})
        model_value = str(extraction.get(db_col, "") or "").strip()

        # ── Classify the error type ────────────────────────────────────
        error_type = "spurious"  # Default to spurious
        reason = ""

        if verdict == "missing":
            # Model missed it entirely — classic False Negative
            error_type = "missing"
            reason = "Model failed to extract this field from the resume"

        elif verdict == "partial":
            # Partial match — is it a boundary issue or something else?
            truth = correction if correction else model_value
            if _is_boundary_error(model_value, truth):
                error_type = "boundary"
                reason = f"Text boundary clipped: extracted '{_truncate(model_value, 40)}' vs truth '{_truncate(truth, 40)}'"
            else:
                error_type = "boundary"  # Partial is usually boundary
                reason = f"Partial extraction — incomplete or imprecise content"

        elif verdict == "wrong":
            truth = correction if correction else ""
            if _is_boundary_error(model_value, truth):
                error_type = "boundary"
                reason = f"Severe boundary error: '{_truncate(model_value, 40)}' vs '{_truncate(truth, 40)}'"
            elif _is_type_error(fk, model_value, truth):
                error_type = "type_error"
                reason = f"Semantic confusion — model value doesn't match expected field type"
            else:
                error_type = "spurious"
                reason = f"Incorrect extraction or hallucinated content"

        # ── Compute fuzzy similarity ───────────────────────────────────
        truth_for_sim = correction if correction else ""
        fuzzy_score = similarity_ratio(
            model_value.lower(), truth_for_sim.lower()
        ) if (model_value and truth_for_sim) else 0.0

        taxonomy_counts[error_type] += 1
        field_taxonomy[fk][error_type] += 1

        errors.append({
            "candidate_id": cid,
            "field_key": fk,
            "field_label": field_info.get("label", fk),
            "critical": field_info.get("critical", False),
            "verdict": verdict,
            "error_type": error_type,
            "reason": reason,
            "model_value": _truncate(model_value, 100),
            "ground_truth": _truncate(correction, 100),
            "fuzzy_score": round(fuzzy_score, 4),
            "levenshtein_distance": levenshtein_distance(
                model_value[:200].lower(),
                truth_for_sim[:200].lower()
            ) if (model_value and truth_for_sim) else None,
        })

    total_errors = sum(taxonomy_counts.values())

    return {
        "errors": errors,
        "taxonomy_counts": taxonomy_counts,
        "taxonomy_percentages": {
            k: round(v / total_errors * 100, 1) if total_errors > 0 else 0.0
            for k, v in taxonomy_counts.items()
        },
        "total_errors": total_errors,
        "field_taxonomy": dict(field_taxonomy),
        "top_error_fields": _rank_fields_by_errors(field_taxonomy),
    }


def analyze_field_sensitivity(
    ner_metrics: Dict,
    error_taxonomy: Dict,
    tracked_fields: Dict[str, Dict]
) -> List[Dict]:
    """
    🎯 Identify which fields are the "weakest links" and WHY.

    Combines F1 scores with error taxonomy to produce a ranked
    sensitivity report. Like a reality TV elimination ranking —
    who's in the danger zone? 📉🚨

    Root cause categories:
      - "format_sensitivity": Errors likely from layout/format issues
        (boundary errors dominate → parsing/OCR problem)
      - "semantic_confusion": Model confuses field types
        (type errors dominate → needs better training examples)
      - "extraction_gap": Model simply can't find the data
        (missing errors dominate → needs prompt/few-shot work)
      - "hallucination": Model invents data
        (spurious errors dominate → needs output validation)

    Returns:
        Sorted list of field analysis dicts (worst F1 first)
    """
    field_metrics = ner_metrics.get("field_metrics", {})
    field_tax = error_taxonomy.get("field_taxonomy", {})

    sensitivity = []

    for fk, fm in field_metrics.items():
        tax = field_tax.get(fk, {"boundary": 0, "type_error": 0, "missing": 0, "spurious": 0})
        total_errors = sum(tax.values())

        # Determine dominant error type (root cause)
        if total_errors == 0:
            root_cause = "healthy"
            root_cause_detail = "No errors detected — this field is performing well! ✨"
        else:
            dominant = max(tax, key=tax.get)
            dominant_pct = round(tax[dominant] / total_errors * 100, 1) if total_errors > 0 else 0

            root_cause_map = {
                "boundary": "format_sensitivity",
                "type_error": "semantic_confusion",
                "missing": "extraction_gap",
                "spurious": "hallucination",
            }
            root_cause = root_cause_map.get(dominant, "unknown")

            detail_map = {
                "format_sensitivity": (
                    f"Boundary errors dominate ({dominant_pct}%). "
                    f"The model clips or extends text — likely a parsing, "
                    f"OCR, or section-boundary detection issue."
                ),
                "semantic_confusion": (
                    f"Type errors dominate ({dominant_pct}%). "
                    f"The model confuses this field with other fields — "
                    f"needs clearer training examples or schema constraints."
                ),
                "extraction_gap": (
                    f"Missing errors dominate ({dominant_pct}%). "
                    f"The model can't find this data — may need prompt "
                    f"engineering, OCR preprocessing, or more training data."
                ),
                "hallucination": (
                    f"Spurious errors dominate ({dominant_pct}%). "
                    f"The model invents data for this field — needs output "
                    f"validation, constrained decoding, or post-processing."
                ),
            }
            root_cause_detail = detail_map.get(root_cause, "Unknown error pattern")

        field_info = tracked_fields.get(fk, {})

        sensitivity.append({
            "field_key": fk,
            "label": field_info.get("label", fk),
            "critical": field_info.get("critical", False),
            "f1": fm["f1"],
            "precision": fm["precision"],
            "recall": fm["recall"],
            "support": fm["support"],
            "total_errors": total_errors,
            "error_breakdown": tax,
            "root_cause": root_cause,
            "root_cause_detail": root_cause_detail,
            "severity": _classify_severity(fm["f1"], fm.get("critical", False)),
        })

    # Sort: worst F1 first (the ones needing the most help!)
    sensitivity.sort(key=lambda x: x["f1"])

    return sensitivity


def compute_fuzzy_matches(
    reviews: List[Dict],
    extractions: Dict[int, Dict],
    tracked_fields: Dict[str, Dict]
) -> List[Dict]:
    """
    📏 Compute fuzzy match scores for all non-binary comparisons.

    For every review that has both a model output and a correction,
    calculates the Levenshtein distance and similarity ratio.

    Useful for understanding HOW CLOSE the model was — sometimes
    "wrong" is just one character off! Like almost nailing the
    runway walk but tripping on the last step 👠💥

    Returns:
        List of fuzzy match result dicts, sorted by similarity (worst first)
    """
    fuzzy_results = []

    for review in reviews:
        verdict = review.get("verdict", "")
        if verdict in ("correct", "skip"):
            continue

        fk = review.get("field_key", "")
        cid = review.get("candidate_id")
        correction = review.get("correction", "").strip()

        field_info = tracked_fields.get(fk, {})
        db_col = field_info.get("db_col", fk)

        extraction = extractions.get(cid, {})
        model_value = str(extraction.get(db_col, "") or "").strip()

        # Only compute fuzzy if both values exist
        if not model_value and not correction:
            continue

        model_norm = model_value[:500].lower()
        truth_norm = correction[:500].lower() if correction else ""

        lev_dist = levenshtein_distance(model_norm, truth_norm) if truth_norm else None
        sim = similarity_ratio(model_norm, truth_norm) if truth_norm else 0.0

        fuzzy_results.append({
            "candidate_id": cid,
            "field_key": fk,
            "field_label": field_info.get("label", fk),
            "verdict": verdict,
            "model_value": _truncate(model_value, 80),
            "ground_truth": _truncate(correction, 80),
            "levenshtein_distance": lev_dist,
            "similarity_ratio": round(sim, 4),
            "similarity_pct": round(sim * 100, 1),
        })

    # Sort: lowest similarity first (worst matches = most interesting!)
    fuzzy_results.sort(key=lambda x: x["similarity_ratio"])

    return fuzzy_results


def generate_improvement_roadmap(
    ner_metrics: Dict,
    error_taxonomy: Dict,
    field_sensitivity: List[Dict],
    total_reviewed: int
) -> List[Dict]:
    """
    🗺️ Generate a prioritized improvement roadmap based on all findings.

    This is where the Fairy Codemother puts on her strategy hat and
    tells you EXACTLY what to fix, in what order, and how! 📋✨

    Each recommendation has:
      - priority: 🔴 Critical / 🟡 Medium / 🟢 Low
      - category: What type of fix (data, prompt, model, pipeline)
      - title: Short description
      - detail: Specific actionable steps
      - expected_impact: What improvement to expect

    Returns:
        Sorted list of recommendation dicts (highest priority first)
    """
    recommendations = []

    # ── Rule 1: Insufficient review data ───────────────────────────────
    if total_reviewed < 20:
        recommendations.append({
            "priority": "critical",
            "priority_icon": "🔴",
            "category": "data",
            "title": "Increase review sample size",
            "detail": (
                f"Only {total_reviewed} field reviews completed. For statistically "
                f"reliable metrics, aim for at least 50–100 reviewed candidates. "
                f"Current metrics may be misleading due to small sample bias."
            ),
            "expected_impact": "More reliable metrics and error pattern detection",
            "effort": "medium",
        })

    # ── Rule 2: Low overall F1 ─────────────────────────────────────────
    weighted_f1 = ner_metrics.get("weighted", {}).get("f1", 0)
    if weighted_f1 < 0.5:
        recommendations.append({
            "priority": "critical",
            "priority_icon": "🔴",
            "category": "model",
            "title": "Overall model performance critically low",
            "detail": (
                f"Weighted F1 = {weighted_f1:.2f} (target: ≥0.80). "
                f"Consider: (a) reviewing prompt templates for clarity, "
                f"(b) adding more diverse training examples, "
                f"(c) checking if OCR/text extraction introduces noise."
            ),
            "expected_impact": "Broad improvement across all fields",
            "effort": "high",
        })

    # ── Rule 3: High boundary error rate ───────────────────────────────
    tax_counts = error_taxonomy.get("taxonomy_counts", {})
    total_errors = error_taxonomy.get("total_errors", 0)
    boundary_rate = (tax_counts.get("boundary", 0) / total_errors * 100) if total_errors > 0 else 0

    if boundary_rate > 30:
        recommendations.append({
            "priority": "critical",
            "priority_icon": "🔴",
            "category": "pipeline",
            "title": f"Boundary errors are {boundary_rate:.0f}% of all errors",
            "detail": (
                "The model frequently clips or extends extracted text. "
                "Remedies: (a) improve text normalization / encoding cleanup, "
                "(b) add explicit section boundary markers in prompts, "
                "(c) for SG/MY resumes — handle date-first formats and "
                "bin/binti naming patterns explicitly."
            ),
            "expected_impact": "10–25% F1 improvement on affected fields",
            "effort": "medium",
        })

    # ── Rule 4: High hallucination rate ────────────────────────────────
    spurious_rate = (tax_counts.get("spurious", 0) / total_errors * 100) if total_errors > 0 else 0

    if spurious_rate > 25:
        recommendations.append({
            "priority": "critical",
            "priority_icon": "🔴",
            "category": "model",
            "title": f"Hallucination rate high ({spurious_rate:.0f}%)",
            "detail": (
                "The model frequently invents or misassigns data. "
                "Remedies: (a) add output validation / schema constraints, "
                "(b) use constrained decoding or regex post-filters, "
                "(c) add negative examples in training data showing what "
                "NOT to extract, (d) lower temperature if using Ollama."
            ),
            "expected_impact": "15–30% precision improvement",
            "effort": "medium",
        })

    # ── Rule 5: High missing rate ──────────────────────────────────────
    missing_rate = (tax_counts.get("missing", 0) / total_errors * 100) if total_errors > 0 else 0

    if missing_rate > 30:
        recommendations.append({
            "priority": "medium",
            "priority_icon": "🟡",
            "category": "prompt",
            "title": f"Extraction gaps: {missing_rate:.0f}% of errors are missing fields",
            "detail": (
                "The model fails to find data that exists in the resume. "
                "Remedies: (a) add explicit field prompts with examples, "
                "(b) use few-shot examples showing extraction from similar "
                "resume layouts, (c) check if OCR is losing content from "
                "multi-column or graphical sections."
            ),
            "expected_impact": "15–20% recall improvement",
            "effort": "medium",
        })

    # ── Rule 6: Per-field critical weaknesses ──────────────────────────
    for fs in field_sensitivity:
        if fs["f1"] < 0.5 and fs["critical"]:
            recommendations.append({
                "priority": "critical",
                "priority_icon": "🔴",
                "category": "training",
                "title": f"Critical field '{fs['label']}' has F1 = {fs['f1']:.2f}",
                "detail": (
                    f"Root cause: {fs['root_cause_detail']} "
                    f"This is a CRITICAL field — prioritize fixing it. "
                    f"Consider augmenting training data with {fs['support'] * 3}+ "
                    f"annotated examples specifically for this field."
                ),
                "expected_impact": f"Direct improvement to {fs['label']} extraction quality",
                "effort": "medium",
            })
        elif fs["f1"] < 0.6 and not fs["critical"]:
            recommendations.append({
                "priority": "medium",
                "priority_icon": "🟡",
                "category": "training",
                "title": f"Field '{fs['label']}' has F1 = {fs['f1']:.2f}",
                "detail": (
                    f"Root cause: {fs['root_cause_detail']} "
                    f"Add targeted training examples for this field."
                ),
                "expected_impact": f"Improvement to {fs['label']} extraction",
                "effort": "low",
            })

    # ── Rule 7: Type errors suggest schema confusion ───────────────────
    type_error_rate = (tax_counts.get("type_error", 0) / total_errors * 100) if total_errors > 0 else 0

    if type_error_rate > 15:
        recommendations.append({
            "priority": "medium",
            "priority_icon": "🟡",
            "category": "prompt",
            "title": f"Semantic confusion ({type_error_rate:.0f}% type errors)",
            "detail": (
                "The model mislabels field types (e.g., company as school). "
                "Remedies: (a) add field-specific descriptions in prompt, "
                "(b) add contrastive examples (this IS a company, this is NOT), "
                "(c) validate output against known entity lists."
            ),
            "expected_impact": "5–15% precision improvement",
            "effort": "low",
        })

    # ── Rule 8: SG/MY specific recommendations ─────────────────────────
    # Always include these for AiMerlion context
    recommendations.append({
        "priority": "low",
        "priority_icon": "🟢",
        "category": "domain",
        "title": "SG/MY resume format handling",
        "detail": (
            "Ensure prompt templates handle: (a) NRIC-based name formats, "
            "(b) bin/binti naming conventions, (c) date-first experience "
            "entries (e.g., '2020-2023 | Company | Role'), (d) local "
            "certifications (CMFAS, BizSafe, WSQ), (e) +65/+60 phone "
            "formats, (f) Pte Ltd / Sdn Bhd company suffixes."
        ),
        "expected_impact": "Better SG/MY resume handling overall",
        "effort": "low",
    })

    # ── Sort by priority ───────────────────────────────────────────────
    priority_order = {"critical": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

    return recommendations


# =============================================================================
# 🎬 MAIN ORCHESTRATOR — Run the Full Audit
# =============================================================================

def run_full_audit(
    reviews: List[Dict],
    extractions: Dict[int, Dict],
    tracked_fields: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    🎬 Run the complete ML Quality Audit — all 5 analyses in one call.

    This is the GRAND FINALE — every analysis combined into one
    comprehensive report. Like the final episode where ALL the judges
    give their scores simultaneously! 🏆✨

    Args:
        reviews: List of accuracy_reviews rows (as dicts)
        extractions: Dict of candidate_id → structured_extractions row (as dict)
        tracked_fields: TRACKED_FIELDS config dict

    Returns:
        Comprehensive audit result dict with all 5 analyses
    """
    logger.info(f"🔬 Running full ML audit on {len(reviews)} reviews, "
                f"{len(extractions)} candidates...")

    # 1. Core NER Metrics
    ner_metrics = compute_ner_metrics(reviews, tracked_fields)

    # 2. Error Taxonomy
    error_taxonomy = classify_errors(reviews, extractions, tracked_fields)

    # 3. Field Sensitivity
    field_sensitivity = analyze_field_sensitivity(
        ner_metrics, error_taxonomy, tracked_fields
    )

    # 4. Fuzzy Matches
    fuzzy_matches = compute_fuzzy_matches(reviews, extractions, tracked_fields)

    # 5. Improvement Roadmap
    roadmap = generate_improvement_roadmap(
        ner_metrics, error_taxonomy, field_sensitivity,
        ner_metrics["total_reviewed"]
    )

    return {
        "ner_metrics": ner_metrics,
        "error_taxonomy": error_taxonomy,
        "field_sensitivity": field_sensitivity,
        "fuzzy_matches": fuzzy_matches,
        "roadmap": roadmap,
        "summary": {
            "total_reviews": len(reviews),
            "total_candidates": len(extractions),
            "weighted_f1": ner_metrics["weighted"]["f1"],
            "total_errors": error_taxonomy["total_errors"],
            "recommendations": len(roadmap),
            "critical_issues": sum(1 for r in roadmap if r["priority"] == "critical"),
        },
    }


# =============================================================================
# 🧰 INTERNAL HELPERS
# =============================================================================

def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis if too long."""
    if not text:
        return ""
    return text[:max_len - 3] + "..." if len(text) > max_len else text


def _classify_severity(f1: float, is_critical: bool) -> str:
    """Classify field health based on F1 and criticality."""
    if f1 >= 0.9:
        return "excellent"
    elif f1 >= 0.7:
        return "good"
    elif f1 >= 0.5:
        return "needs_improvement" if not is_critical else "at_risk"
    else:
        return "critical" if is_critical else "poor"


def _rank_fields_by_errors(field_taxonomy: Dict) -> List[Dict]:
    """Rank fields by total error count (worst first)."""
    ranked = []
    for fk, tax in field_taxonomy.items():
        total = sum(tax.values())
        ranked.append({"field_key": fk, "total_errors": total, **tax})
    ranked.sort(key=lambda x: x["total_errors"], reverse=True)
    return ranked[:10]