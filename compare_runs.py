"""
compare_runs.py

💅✨ FAIRY CODEMOTHER'S EXTRACTION RUN COMPARATOR ✨💅

Compare multiple extracted_resume_data_*.json exports side-by-side
to track how your AI extraction pipeline improves (or regresses!)
over time. Think of it as the before/after runway walk — we need
to SEE the glow-up, darling! 🎭📊

Features:
  - Auto-discovers JSON files by glob pattern in a configurable directory
  - Matches candidates by ID across 3+ extraction runs
  - Computes per-field diffs: changed / added / removed / unchanged
  - Aggregate stats: completeness %, field coverage, trend lines
  - CLI mode: rich terminal summary tables
  - Flask dashboard (port 5075): Chart.js visualizations + drill-down

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  JSON Files  →  Comparison Engine  →  CLI Report            │
  │  (exports/)     (pure Python)         (terminal tables)     │
  │                       │                                     │
  │                       └──────────→  Flask Dashboard          │
  │                                     (port 5075, Chart.js)   │
  └─────────────────────────────────────────────────────────────┘

Port Map (for reference):
  5001  → classification_dashboard.py
  5050  → review_dashboard.py
  5055  → annotation_tool.py
  5070  → accuracy_validator_api.py
  5075  → THIS FILE (compare_runs.py) ← NEW! ✨
  NOTE: 5060/5061 are BLOCKED by Chrome (SIP protocol).

Usage:
    # CLI mode — quick terminal summary
    python compare_runs.py --cli

    # Dashboard mode — visual browser experience
    python compare_runs.py

    # Custom directory
    python compare_runs.py --dir C:/Users/user/github/AiMerlion/exports

    # Specific files
    python compare_runs.py --files run1.json run2.json run3.json

Dependencies:
    pip install flask --break-system-packages
"""

import os
import sys
import json
import glob
import re
import math
import logging
import argparse
import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# =============================================================================
# 🔬 ML QUALITY AUDIT (optional — graceful degradation)
# =============================================================================
try:
    from ml_quality_audit import (
        levenshtein_distance, similarity_ratio, normalized_levenshtein
    )
    ML_AUDIT_AVAILABLE = True
except ImportError:
    ML_AUDIT_AVAILABLE = False
    # Inline fallback — basic Levenshtein so comparator works standalone
    def levenshtein_distance(s1: str, s2: str) -> int:
        if s1 == s2: return 0
        if not s1: return len(s2)
        if not s2: return len(s1)
        if len(s1) > len(s2): s1, s2 = s2, s1
        prev = list(range(len(s1) + 1))
        curr = [0] * (len(s1) + 1)
        for j in range(1, len(s2) + 1):
            curr[0] = j
            for i in range(1, len(s1) + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                curr[i] = min(curr[i-1]+1, prev[i]+1, prev[i-1]+cost)
            prev, curr = curr, prev
        return prev[len(s1)]

    def similarity_ratio(s1: str, s2: str) -> float:
        if not s1 and not s2: return 1.0
        if not s1 or not s2: return 0.0
        mx = max(len(s1), len(s2))
        return round(1.0 - levenshtein_distance(s1, s2) / mx, 4) if mx else 1.0

# =============================================================================
# 🔧 LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 🔄 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 🎛️ CONFIGURATION
# =============================================================================

# Where to look for JSON export files
# Supports both the resume_exporter.py output and custom naming
DEFAULT_EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "exports")

# Glob patterns to discover JSON files (tried in order)
# The user's files: extracted_resume_data_20260401_052057.json
# The exporter's files: candidates_20260401_052057.json
JSON_GLOB_PATTERNS = [
    "extracted_resume_data_*.json",
    "candidates_*.json",
    "*.json",  # Fallback — any JSON (we'll validate structure)
]

# Regex to extract timestamp from filenames
# Matches: _YYYYMMDD_HHMMSS or _YYYYMMDD patterns
TIMESTAMP_REGEX = re.compile(r"_(\d{8})_?(\d{6})?")

# Fields we compare — grouped by importance
# Think of these as the categories in a pageant:
# 👑 CRITICAL = The crown jewels — must be right
# 💎 IMPORTANT = The accessories — really should be right
# 🎀 NICE_TO_HAVE = The finishing touches
COMPARISON_FIELDS = {
    "critical": {
        "Name":     {"type": "string", "label": "Full Name"},
        "Email":    {"type": "string", "label": "Email"},
        "Phone":    {"type": "string", "label": "Phone"},
    },
    "important": {
        "Current Company":  {"type": "string", "label": "Current Company"},
        "Current Title":    {"type": "string", "label": "Current Title"},
        "Current Location": {"type": "string", "label": "Location"},
        "Function":         {"type": "string", "label": "Function"},
        "Industry":         {"type": "string", "label": "Industry"},
        "Summary":          {"type": "string", "label": "Summary"},
    },
    "nice_to_have": {
        "Language Skills":    {"type": "string",  "label": "Languages"},
        "Project Experience": {"type": "string",  "label": "Projects"},
    },
    "complex": {
        "Work Experience": {"type": "array", "label": "Experience"},
        "Education":       {"type": "array", "label": "Education"},
        "tags":            {"type": "array", "label": "Skills/Tags"},
    },
}

# Flatten all field names for quick lookup
ALL_FIELD_NAMES = {}
for group_fields in COMPARISON_FIELDS.values():
    ALL_FIELD_NAMES.update(group_fields)

# Flask settings
HOST = "0.0.0.0"
PORT = 5075
DEBUG = True


# =============================================================================
# 📂 JSON FILE DISCOVERY & LOADING
# =============================================================================

def discover_json_files(directory: str) -> List[Dict[str, Any]]:
    """
    🔍 Auto-discover JSON export files in a directory.

    Tries multiple glob patterns and validates each file contains
    a list of candidate records with an "ID" field.

    Returns a list of dicts:
        [{"path": ..., "filename": ..., "timestamp": ..., "record_count": ...}, ...]

    Sorted by timestamp (oldest → newest) so you see the chronological
    glow-up progression! 📈✨
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        logger.warning(f"⚠️ Directory not found: {directory}")
        return []

    found_files = set()

    for pattern in JSON_GLOB_PATTERNS:
        matches = glob.glob(os.path.join(directory, pattern))
        for match in matches:
            found_files.add(os.path.abspath(match))

    # Validate and collect metadata for each file
    valid_files = []
    for filepath in sorted(found_files):
        meta = _validate_json_file(filepath)
        if meta:
            valid_files.append(meta)

    # Sort by timestamp (oldest first for chronological comparison)
    valid_files.sort(key=lambda f: f["timestamp"] or "0000")

    logger.info(f"📂 Discovered {len(valid_files)} valid JSON files in {directory}")
    return valid_files


def _validate_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """
    ✅ Validate a JSON file contains an array of candidate records.

    Checks:
    1. File is valid JSON
    2. Top-level is a list
    3. At least one record has an "ID" field

    Like checking a contestant's ID at registration — no ID, no entry! 🪪
    """
    filename = os.path.basename(filepath)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"  ⏭️ Skipping {filename}: invalid JSON ({e})")
        return None
    except OSError as e:
        logger.debug(f"  ⏭️ Skipping {filename}: read error ({e})")
        return None

    # Must be a list of records
    if not isinstance(data, list):
        logger.debug(f"  ⏭️ Skipping {filename}: not a JSON array")
        return None

    if len(data) == 0:
        logger.debug(f"  ⏭️ Skipping {filename}: empty array")
        return None

    # At least one record must have an "ID" field
    has_id = any(isinstance(r, dict) and "ID" in r for r in data)
    if not has_id:
        logger.debug(f"  ⏭️ Skipping {filename}: no records with 'ID' field")
        return None

    # Extract timestamp from filename
    timestamp = _extract_timestamp(filename)

    # File size for display
    size_bytes = os.path.getsize(filepath)

    return {
        "path": filepath,
        "filename": filename,
        "timestamp": timestamp,
        "timestamp_display": _format_timestamp(timestamp),
        "record_count": len(data),
        "size_bytes": size_bytes,
        "size_display": _format_size(size_bytes),
    }


def _extract_timestamp(filename: str) -> Optional[str]:
    """
    🕐 Extract a sortable timestamp string from a filename.

    Supports patterns:
    - extracted_resume_data_20260401_052057.json → "20260401_052057"
    - candidates_20260401.json → "20260401_000000"

    Returns ISO-sortable string or None if no timestamp found.
    """
    match = TIMESTAMP_REGEX.search(filename)
    if match:
        date_part = match.group(1)
        time_part = match.group(2) or "000000"
        return f"{date_part}_{time_part}"
    return None


def _format_timestamp(ts: Optional[str]) -> str:
    """Format a timestamp string for human display."""
    if not ts:
        return "Unknown date"
    try:
        # Parse YYYYMMDD_HHMMSS
        dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.strftime("%d %b %Y, %H:%M")
    except ValueError:
        return ts


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 / 1024:.1f} MB"


def load_json_records(filepath: str) -> List[Dict]:
    """
    📋 Load candidate records from a JSON file.

    Returns a list of dicts, each representing one candidate.
    Records without an "ID" field are skipped (the bouncer says no).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"❌ Failed to load {filepath}: {e}")
        return []

    if not isinstance(data, list):
        return []

    # Filter to only records with an ID
    valid = [r for r in data if isinstance(r, dict) and "ID" in r]
    return valid


# =============================================================================
# 🔬 COMPARISON ENGINE — The Brains of the Operation! 🧠
# =============================================================================

def compare_runs(file_metas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    🔬 Compare 2+ extraction runs and produce a comprehensive diff.

    This is the MAIN comparison function — the grand choreographer
    who coordinates all the dancers on stage! 💃🕺

    Args:
        file_metas: List of file metadata dicts from discover_json_files()

    Returns:
        A rich comparison result dict with:
        - run_summaries: per-run aggregate stats
        - field_trends: field completeness across runs
        - candidate_diffs: per-candidate field-level changes
        - improvement_summary: what got better/worse
    """
    if len(file_metas) < 2:
        return {"error": "Need at least 2 JSON files to compare", "runs": len(file_metas)}

    # ── Step 1: Load all runs ──────────────────────────────────────────
    runs = []
    for meta in file_metas:
        records = load_json_records(meta["path"])
        # Index by candidate ID for O(1) lookup
        indexed = {r["ID"]: r for r in records}
        runs.append({
            "meta": meta,
            "records": records,
            "indexed": indexed,
            "ids": set(indexed.keys()),
        })

    # ── Step 2: Find the universe of all candidate IDs ─────────────────
    all_ids = set()
    for run in runs:
        all_ids.update(run["ids"])
    all_ids = sorted(all_ids)

    # ── Step 3: Per-run aggregate stats ────────────────────────────────
    run_summaries = []
    for i, run in enumerate(runs):
        summary = _compute_run_summary(run, i)
        run_summaries.append(summary)

    # ── Step 4: Field completeness trends across runs ──────────────────
    field_trends = _compute_field_trends(runs)

    # ── Step 5: Per-candidate diffs (sequential run pairs) ─────────────
    candidate_diffs = _compute_candidate_diffs(runs, all_ids)

    # ── Step 6: Improvement summary ────────────────────────────────────
    improvement = _compute_improvement_summary(runs, candidate_diffs)

    # ── Step 7: ML Quality Audit (similarity-based) ────────────────────
    quality_audit = _compute_quality_audit(runs, all_ids)

    return {
        "run_count": len(runs),
        "total_unique_candidates": len(all_ids),
        "run_summaries": run_summaries,
        "field_trends": field_trends,
        "candidate_diffs": candidate_diffs,
        "improvement": improvement,
        "quality_audit": quality_audit,
        "candidate_ids": all_ids,
    }


def _compute_run_summary(run: Dict, index: int) -> Dict:
    """
    📊 Compute aggregate stats for a single run.

    Counts how many candidates have each field filled,
    like taking attendance at a beauty school class! 💄📋
    """
    records = run["records"]
    total = len(records)
    meta = run["meta"]

    # Count filled fields
    field_fill = {}
    for field_name in ALL_FIELD_NAMES:
        filled = 0
        for r in records:
            val = r.get(field_name)
            if _is_filled(val):
                filled += 1
        pct = round((filled / total) * 100, 1) if total > 0 else 0.0
        field_fill[field_name] = {"filled": filled, "total": total, "pct": pct}

    # Overall completeness: avg fill rate across ALL tracked fields
    all_pcts = [v["pct"] for v in field_fill.values()]
    overall_completeness = round(sum(all_pcts) / len(all_pcts), 1) if all_pcts else 0.0

    return {
        "index": index,
        "filename": meta["filename"],
        "timestamp": meta["timestamp"],
        "timestamp_display": meta["timestamp_display"],
        "record_count": total,
        "field_fill": field_fill,
        "overall_completeness": overall_completeness,
    }


def _is_filled(value: Any) -> bool:
    """
    Check if a field value is meaningfully filled (not empty/null/blank).

    An empty string, None, empty list, or whitespace-only string
    all count as NOT filled. Like an empty dress form — technically
    there but nothing to show! 👗
    """
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, list):
        return len(value) > 0
    return True


def _compute_field_trends(runs: List[Dict]) -> Dict[str, List[Dict]]:
    """
    📈 Compute how each field's completeness changes across runs.

    Returns a dict of field_name → list of {run_index, pct, filled, total}.
    Perfect for line charts showing the glow-up trajectory! 🚀
    """
    trends = {}
    for field_name in ALL_FIELD_NAMES:
        trend_points = []
        for i, run in enumerate(runs):
            total = len(run["records"])
            filled = 0
            for r in run["records"]:
                if _is_filled(r.get(field_name)):
                    filled += 1
            pct = round((filled / total) * 100, 1) if total > 0 else 0.0
            trend_points.append({
                "run_index": i,
                "filename": run["meta"]["filename"],
                "timestamp_display": run["meta"]["timestamp_display"],
                "filled": filled,
                "total": total,
                "pct": pct,
            })
        trends[field_name] = trend_points
    return trends


def _compute_candidate_diffs(
    runs: List[Dict], all_ids: List[Any]
) -> List[Dict]:
    """
    🔍 Compute per-candidate field-level diffs between consecutive runs.

    For each candidate, tracks what changed from run N to run N+1:
    - "improved": field went from empty → filled, or value changed
    - "regressed": field went from filled → empty
    - "changed": field value changed (both non-empty)
    - "unchanged": same value
    - "new": candidate appeared in this run
    - "removed": candidate disappeared from this run

    Like a before/after photo comparison at a makeover show! 💇‍♀️→💃
    """
    candidate_diffs = []

    for cid in all_ids:
        candidate_entry = {
            "id": cid,
            "name": "",  # Will be filled from latest run
            "presence": [],  # Which runs contain this candidate
            "field_changes": {},  # field_name → list of changes across run pairs
        }

        # Track presence across runs
        for i, run in enumerate(runs):
            present = cid in run["ids"]
            candidate_entry["presence"].append(present)
            if present and not candidate_entry["name"]:
                candidate_entry["name"] = run["indexed"][cid].get("Name", "")

        # Compare consecutive run pairs
        for field_name in ALL_FIELD_NAMES:
            changes = []
            for i in range(len(runs)):
                rec = runs[i]["indexed"].get(cid)
                val = rec.get(field_name) if rec else None

                if i == 0:
                    changes.append({
                        "run_index": i,
                        "value": _summarize_value(val),
                        "filled": _is_filled(val),
                        "change": "baseline",
                    })
                else:
                    prev_rec = runs[i - 1]["indexed"].get(cid)
                    prev_val = prev_rec.get(field_name) if prev_rec else None

                    change_type = _classify_change(prev_val, val)
                    changes.append({
                        "run_index": i,
                        "value": _summarize_value(val),
                        "filled": _is_filled(val),
                        "change": change_type,
                    })

            candidate_entry["field_changes"][field_name] = changes

        candidate_diffs.append(candidate_entry)

    return candidate_diffs


def _classify_change(old_val: Any, new_val: Any) -> str:
    """
    Classify the change between two field values.

    Returns one of: improved, regressed, changed, unchanged, new, removed
    """
    old_filled = _is_filled(old_val)
    new_filled = _is_filled(new_val)

    if not old_filled and not new_filled:
        return "unchanged"  # Both empty
    if not old_filled and new_filled:
        return "improved"   # Empty → Filled ✨
    if old_filled and not new_filled:
        return "regressed"  # Filled → Empty 😢
    # Both filled — check if value actually changed
    if _values_equal(old_val, new_val):
        return "unchanged"
    return "changed"  # Value modified


def _values_equal(a: Any, b: Any) -> bool:
    """
    Compare two field values for equality.

    Handles strings (case-insensitive, trimmed), lists (sorted comparison),
    and dicts (key-by-key). Like comparing two outfits — we check every
    accessory, not just the dress! 👗🔍
    """
    # Normalize strings
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()

    # Normalize lists (compare sorted JSON representations)
    if isinstance(a, list) and isinstance(b, list):
        try:
            return json.dumps(sorted(a, key=str), sort_keys=True) == \
                   json.dumps(sorted(b, key=str), sort_keys=True)
        except (TypeError, ValueError):
            return str(a) == str(b)

    # Fallback to string comparison
    return str(a).strip() == str(b).strip()


def _summarize_value(val: Any) -> str:
    """
    Produce a short display string for a field value.

    Arrays show count + preview, long strings get truncated.
    Like a movie trailer — give them the highlights! 🎬
    """
    if val is None:
        return ""
    if isinstance(val, list):
        if len(val) == 0:
            return ""
        # Show count + first item preview for arrays
        first = val[0]
        if isinstance(first, dict):
            # Work Experience / Education — show company/school
            preview = first.get("company") or first.get("school") or \
                      first.get("title") or str(first)[:40]
            return f"[{len(val)} items] {preview}..."
        else:
            preview = ", ".join(str(v) for v in val[:3])
            if len(val) > 3:
                preview += f" +{len(val) - 3} more"
            return preview
    text = str(val).strip()
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _compute_improvement_summary(
    runs: List[Dict], candidate_diffs: List[Dict]
) -> Dict:
    """
    📈 Compute a high-level improvement summary across all run transitions.

    Counts how many fields improved, regressed, or changed across
    each consecutive run pair. The scoreboard that matters! 🏆
    """
    if len(runs) < 2:
        return {"transitions": []}

    transitions = []
    for i in range(1, len(runs)):
        counts = {"improved": 0, "regressed": 0, "changed": 0, "unchanged": 0}
        field_counts = defaultdict(lambda: {"improved": 0, "regressed": 0, "changed": 0, "unchanged": 0})

        for cd in candidate_diffs:
            for field_name, changes in cd["field_changes"].items():
                if i < len(changes):
                    change = changes[i]["change"]
                    if change in counts:
                        counts[change] += 1
                        field_counts[field_name][change] += 1

        total_comparisons = sum(counts.values())
        transitions.append({
            "from_run": i - 1,
            "to_run": i,
            "from_filename": runs[i - 1]["meta"]["filename"],
            "to_filename": runs[i]["meta"]["filename"],
            "counts": counts,
            "total_comparisons": total_comparisons,
            "improvement_rate": round(
                (counts["improved"] / total_comparisons * 100)
                if total_comparisons > 0 else 0, 1
            ),
            "regression_rate": round(
                (counts["regressed"] / total_comparisons * 100)
                if total_comparisons > 0 else 0, 1
            ),
            "field_counts": dict(field_counts),
        })

    return {"transitions": transitions}


def _compute_quality_audit(runs: List[Dict], all_ids: List[Any]) -> Dict:
    """
    🔬 ML QUALITY AUDIT — Compute per-run quality metrics using the
    LATEST run as the reference (pseudo ground-truth).

    Since we don't have human-reviewed ground truth here (that lives
    in accuracy_validator_api.py), we treat the LATEST extraction run
    as the "best available" reference and measure how much earlier runs
    deviate from it. Like measuring how far each rehearsal was from
    the FINAL performance! 🎭📏

    Computes:
      - Per-field average similarity across runs (Levenshtein ratio)
      - Per-run overall quality score (avg similarity to latest)
      - Error taxonomy: boundary / value_change / missing / spurious
      - Field-level weakness ranking
      - Quality trend across runs

    Returns:
        Dict with quality_by_run, field_quality, error_taxonomy, weakest_fields
    """
    if len(runs) < 2:
        return {"available": False, "reason": "Need 2+ runs"}

    ref_run = runs[-1]  # Latest run = reference
    ref_indexed = ref_run["indexed"]
    tracked_fields = list(ALL_FIELD_NAMES.keys())

    # ── Per-run quality scores ─────────────────────────────────────────
    quality_by_run = []

    for run_idx, run in enumerate(runs):
        field_similarities = defaultdict(list)
        error_counts = {"boundary": 0, "value_change": 0, "missing": 0, "spurious": 0, "match": 0}
        total_comparisons = 0

        for cid in all_ids:
            run_rec = run["indexed"].get(cid)
            ref_rec = ref_indexed.get(cid)

            for field_name in tracked_fields:
                run_val = _normalize_val(run_rec.get(field_name) if run_rec else None)
                ref_val = _normalize_val(ref_rec.get(field_name) if ref_rec else None)

                # Skip if both empty — nothing to compare
                if not run_val and not ref_val:
                    continue

                total_comparisons += 1
                sim = similarity_ratio(run_val.lower(), ref_val.lower()) if (run_val and ref_val) else 0.0
                field_similarities[field_name].append(sim)

                # ── Classify the difference ────────────────────────
                if run_val == ref_val or sim >= 0.95:
                    error_counts["match"] += 1
                elif not run_val and ref_val:
                    error_counts["missing"] += 1
                elif run_val and not ref_val:
                    error_counts["spurious"] += 1
                elif (run_val in ref_val or ref_val in run_val) and sim >= 0.4:
                    error_counts["boundary"] += 1
                else:
                    error_counts["value_change"] += 1

        # Average similarity across all fields
        all_sims = []
        field_avg = {}
        for fname, sims in field_similarities.items():
            avg = round(sum(sims) / len(sims), 4) if sims else 0.0
            field_avg[fname] = {"avg_similarity": avg, "comparisons": len(sims)}
            all_sims.extend(sims)

        overall_sim = round(sum(all_sims) / len(all_sims), 4) if all_sims else 0.0

        quality_by_run.append({
            "run_index": run_idx,
            "filename": run["meta"]["filename"],
            "timestamp_display": run["meta"]["timestamp_display"],
            "overall_similarity": overall_sim,
            "overall_similarity_pct": round(overall_sim * 100, 1),
            "total_comparisons": total_comparisons,
            "error_counts": dict(error_counts),
            "field_avg": field_avg,
            "is_reference": run_idx == len(runs) - 1,
        })

    # ── Field-level weakness ranking (worst avg similarity first) ──────
    # Aggregate across ALL non-reference runs
    field_totals = defaultdict(lambda: {"sum_sim": 0.0, "count": 0, "errors": 0})
    for qr in quality_by_run[:-1]:  # Exclude reference run
        for fname, fdata in qr["field_avg"].items():
            field_totals[fname]["sum_sim"] += fdata["avg_similarity"] * fdata["comparisons"]
            field_totals[fname]["count"] += fdata["comparisons"]
        for err_type in ("boundary", "value_change", "missing", "spurious"):
            # We can't attribute errors to specific fields from the run-level
            # counts, so we compute field-level from the raw similarities
            pass

    weakest_fields = []
    for fname, totals in field_totals.items():
        avg = round(totals["sum_sim"] / totals["count"], 4) if totals["count"] > 0 else 0.0
        info = ALL_FIELD_NAMES.get(fname, {})
        weakest_fields.append({
            "field": fname,
            "label": info.get("label", fname),
            "avg_similarity": avg,
            "avg_similarity_pct": round(avg * 100, 1),
            "comparisons": totals["count"],
            "severity": "critical" if avg < 0.5 else "needs_improvement" if avg < 0.7 else "good" if avg < 0.9 else "excellent",
        })
    weakest_fields.sort(key=lambda x: x["avg_similarity"])

    # ── Quality trend (how does overall similarity grow per run?) ───────
    quality_trend = [
        {
            "run_index": qr["run_index"],
            "filename": qr["filename"],
            "similarity_pct": qr["overall_similarity_pct"],
        }
        for qr in quality_by_run
    ]

    # ── Aggregate error taxonomy across all non-reference runs ─────────
    total_tax = {"boundary": 0, "value_change": 0, "missing": 0, "spurious": 0, "match": 0}
    for qr in quality_by_run[:-1]:
        for k, v in qr["error_counts"].items():
            total_tax[k] += v
    total_errors = total_tax["boundary"] + total_tax["value_change"] + total_tax["missing"] + total_tax["spurious"]
    tax_pct = {
        k: round(v / total_errors * 100, 1) if total_errors > 0 else 0.0
        for k, v in total_tax.items() if k != "match"
    }

    return {
        "available": True,
        "reference_run": quality_by_run[-1]["filename"],
        "quality_by_run": quality_by_run,
        "quality_trend": quality_trend,
        "weakest_fields": weakest_fields,
        "error_taxonomy": {
            "counts": {k: v for k, v in total_tax.items() if k != "match"},
            "percentages": tax_pct,
            "total_errors": total_errors,
            "total_matches": total_tax["match"],
        },
    }


def _normalize_val(val: Any) -> str:
    """Normalize a field value to a comparable string."""
    if val is None:
        return ""
    if isinstance(val, list):
        if len(val) == 0:
            return ""
        return ", ".join(sorted(str(v) for v in val if v))
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True, default=str)
    return str(val).strip()


# =============================================================================
# 🖥️ CLI REPORTER — Terminal Summary
# =============================================================================

def print_cli_report(result: Dict):
    """
    🖥️ Print a rich terminal summary of the comparison results.

    Uses box-drawing characters for that premium CLI aesthetic.
    Because even terminal output deserves to look FABULOUS! 💅
    """
    if "error" in result:
        print(f"\n  ❌ {result['error']}")
        return

    W = 70  # Box width

    print()
    print("╔" + "═" * W + "╗")
    print("║" + " 🧚‍♀️✨ EXTRACTION RUN COMPARISON REPORT ✨🧚‍♀️ ".center(W) + "║")
    print("╠" + "═" * W + "╣")
    print("║" + f"  📊 Runs compared: {result['run_count']}".ljust(W) + "║")
    print("║" + f"  👥 Unique candidates: {result['total_unique_candidates']}".ljust(W) + "║")
    print("╠" + "═" * W + "╣")

    # ── Per-run summary ────────────────────────────────────────────────
    print("║" + " 📋 RUN SUMMARIES".ljust(W) + "║")
    print("╠" + "─" * W + "╣")

    for rs in result["run_summaries"]:
        label = f"  Run {rs['index']}: {rs['filename']}"
        if len(label) > W - 2:
            label = label[:W - 5] + "..."
        print("║" + label.ljust(W) + "║")
        print("║" + f"    📅 {rs['timestamp_display']}  |  "
                     f"👥 {rs['record_count']} candidates  |  "
                     f"📊 {rs['overall_completeness']}% complete".ljust(W) + "║")
        print("║" + ("─" * W) + "║")

    # ── Improvement summary ────────────────────────────────────────────
    transitions = result["improvement"]["transitions"]
    if transitions:
        print("╠" + "═" * W + "╣")
        print("║" + " 📈 IMPROVEMENT TRACKING".ljust(W) + "║")
        print("╠" + "─" * W + "╣")

        for t in transitions:
            from_name = t["from_filename"][:25]
            to_name = t["to_filename"][:25]
            print("║" + f"  {from_name} → {to_name}".ljust(W) + "║")
            c = t["counts"]
            print("║" + f"    ✅ Improved: {c['improved']:>5}  |  "
                         f"❌ Regressed: {c['regressed']:>5}  |  "
                         f"🔄 Changed: {c['changed']:>5}  |  "
                         f"➖ Same: {c['unchanged']:>5}".ljust(W) + "║")
            print("║" + f"    📈 Improvement rate: {t['improvement_rate']}%  |  "
                         f"📉 Regression rate: {t['regression_rate']}%".ljust(W) + "║")
            print("║" + ("─" * W) + "║")

    # ── Top improved / regressed fields ────────────────────────────────
    if transitions:
        last_t = transitions[-1]
        fc = last_t.get("field_counts", {})

        # Sort fields by improvement count
        top_improved = sorted(
            fc.items(), key=lambda x: x[1]["improved"], reverse=True
        )[:5]
        top_regressed = sorted(
            fc.items(), key=lambda x: x[1]["regressed"], reverse=True
        )[:5]

        if any(v["improved"] > 0 for _, v in top_improved):
            print("╠" + "═" * W + "╣")
            print("║" + " 🌟 TOP IMPROVED FIELDS (latest transition)".ljust(W) + "║")
            print("╠" + "─" * W + "╣")
            for field, counts in top_improved:
                if counts["improved"] > 0:
                    label = ALL_FIELD_NAMES.get(field, {}).get("label", field)
                    print("║" + f"    ✅ {label:<25} +{counts['improved']} improved".ljust(W) + "║")

        if any(v["regressed"] > 0 for _, v in top_regressed):
            print("╠" + "═" * W + "╣")
            print("║" + " 🚨 TOP REGRESSED FIELDS (latest transition)".ljust(W) + "║")
            print("╠" + "─" * W + "╣")
            for field, counts in top_regressed:
                if counts["regressed"] > 0:
                    label = ALL_FIELD_NAMES.get(field, {}).get("label", field)
                    print("║" + f"    ❌ {label:<25} -{counts['regressed']} regressed".ljust(W) + "║")

    # ── Field completeness trend (latest vs first) ─────────────────────
    if len(result["run_summaries"]) >= 2:
        print("╠" + "═" * W + "╣")
        print("║" + " 📊 FIELD COMPLETENESS: FIRST RUN → LATEST RUN".ljust(W) + "║")
        print("╠" + "─" * W + "╣")

        first_rs = result["run_summaries"][0]
        last_rs = result["run_summaries"][-1]

        for field_name, info in ALL_FIELD_NAMES.items():
            first_pct = first_rs["field_fill"].get(field_name, {}).get("pct", 0)
            last_pct = last_rs["field_fill"].get(field_name, {}).get("pct", 0)
            delta = last_pct - first_pct
            arrow = "📈" if delta > 0 else "📉" if delta < 0 else "➖"
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            label = info.get("label", field_name)
            print("║" + f"    {label:<22} {first_pct:>6.1f}% → {last_pct:>6.1f}%  {arrow} {delta_str}".ljust(W) + "║")

    # ── Quality Audit Summary ──────────────────────────────────────────
    qa = result.get("quality_audit", {})
    if qa.get("available"):
        print("╠" + "═" * W + "╣")
        print("║" + " 🔬 ML QUALITY AUDIT (vs latest run as reference)".ljust(W) + "║")
        print("╠" + "─" * W + "╣")
        print("║" + f"  📏 Reference: {qa['reference_run']}".ljust(W) + "║")
        print("║" + ("─" * W) + "║")

        for qr in qa["quality_by_run"]:
            tag = " ← REFERENCE" if qr["is_reference"] else ""
            sim_arrow = "🟢" if qr["overall_similarity_pct"] >= 90 else "🟡" if qr["overall_similarity_pct"] >= 70 else "🔴"
            print("║" + f"  Run {qr['run_index']}: {qr['filename'][:35]}{tag}".ljust(W) + "║")
            print("║" + f"    {sim_arrow} Similarity to ref: {qr['overall_similarity_pct']}%  "
                         f"({qr['total_comparisons']} fields compared)".ljust(W) + "║")

        # Error taxonomy
        et = qa.get("error_taxonomy", {})
        if et.get("total_errors", 0) > 0:
            print("║" + ("─" * W) + "║")
            print("║" + "  Error Taxonomy (non-reference runs):".ljust(W) + "║")
            for cat in ("boundary", "value_change", "missing", "spurious"):
                cnt = et["counts"].get(cat, 0)
                pct = et["percentages"].get(cat, 0)
                if cnt > 0:
                    print("║" + f"    {cat:<16} {cnt:>5} ({pct}%)".ljust(W) + "║")

        # Weakest fields
        wf = [f for f in qa.get("weakest_fields", []) if f["avg_similarity"] < 0.9][:5]
        if wf:
            print("║" + ("─" * W) + "║")
            print("║" + "  Weakest Fields (avg similarity to reference):".ljust(W) + "║")
            for f in wf:
                icon = "🔴" if f["severity"] == "critical" else "🟡"
                print("║" + f"    {icon} {f['label']:<22} {f['avg_similarity_pct']}% similarity".ljust(W) + "║")

    print("╚" + "═" * W + "╝")
    print()


# =============================================================================
# 🌐 FLASK DASHBOARD — The Visual Extravaganza! 🎭
# =============================================================================

try:
    from flask import Flask, request, jsonify, g, make_response, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# We only create the Flask app if Flask is available
# (CLI mode works without Flask)
if FLASK_AVAILABLE:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "fairy-compare-sparkle-2026")

    # ── CORS ──────────────────────────────────────────────────────────
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    # ── State: which directory are we scanning? ───────────────────────
    # Set by CLI args or env var, used by all endpoints
    _exports_dir = DEFAULT_EXPORTS_DIR

    def set_exports_dir(d: str):
        global _exports_dir
        _exports_dir = os.path.abspath(d)

    # ── API ENDPOINTS ─────────────────────────────────────────────────

    @app.route("/api/files", methods=["GET"])
    def api_list_files():
        """📂 List available JSON files in the exports directory."""
        files = discover_json_files(_exports_dir)
        return jsonify({
            "files": files,
            "directory": _exports_dir,
            "total": len(files),
        })

    @app.route("/api/compare", methods=["POST"])
    def api_compare():
        """
        🔬 Run comparison on selected files.

        Expects JSON body:
        { "files": ["file1.json", "file2.json", ...] }

        Or with no body → compares ALL discovered files.
        """
        data = request.get_json() or {}
        selected_filenames = data.get("files", [])

        all_files = discover_json_files(_exports_dir)

        if selected_filenames:
            # Filter to selected files only
            selected_set = set(selected_filenames)
            file_metas = [f for f in all_files if f["filename"] in selected_set]
        else:
            file_metas = all_files

        if len(file_metas) < 2:
            return jsonify({
                "error": f"Need at least 2 files to compare. Found {len(file_metas)}.",
                "available_files": [f["filename"] for f in all_files],
            }), 400

        result = compare_runs(file_metas)
        return jsonify(result)

    @app.route("/api/candidate/<candidate_id>", methods=["POST"])
    def api_candidate_detail(candidate_id):
        """
        🔍 Get detailed field-level comparison for a single candidate.

        Loads the raw values from each selected run for side-by-side display.
        """
        data = request.get_json() or {}
        selected_filenames = data.get("files", [])

        all_files = discover_json_files(_exports_dir)

        if selected_filenames:
            selected_set = set(selected_filenames)
            file_metas = [f for f in all_files if f["filename"] in selected_set]
        else:
            file_metas = all_files

        # Load each run and extract this candidate's record
        try:
            cid = int(candidate_id)
        except (ValueError, TypeError):
            cid = candidate_id

        run_records = []
        for meta in file_metas:
            records = load_json_records(meta["path"])
            indexed = {r["ID"]: r for r in records}
            candidate_rec = indexed.get(cid, None)
            run_records.append({
                "filename": meta["filename"],
                "timestamp_display": meta["timestamp_display"],
                "present": candidate_rec is not None,
                "record": candidate_rec,
            })

        return jsonify({
            "candidate_id": cid,
            "runs": run_records,
        })

    # ── DASHBOARD HTML ────────────────────────────────────────────────

    @app.route("/")
    @app.route("/dashboard")
    def dashboard():
        """🎨 Serve the comparison dashboard."""
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "exports_dir": _exports_dir,
            "exports_dir_exists": os.path.isdir(_exports_dir),
        })


# =============================================================================
# 🎨 DASHBOARD HTML — The Main Event! 🎭
# =============================================================================

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔄 Extraction Run Comparator</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
/* ── CSS RESET & VARIABLES ───────────────────────────────────────── */
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

/* ── HEADER ─────────────────────────────────────────────────────── */
.header {
  background: linear-gradient(135deg, var(--bg-card) 0%, #1a1040 100%);
  border-bottom: 1px solid var(--border);
  padding: 16px 28px; display: flex; align-items: center;
  justify-content: space-between; flex-wrap: wrap; gap: 12px;
}
.header-left { display: flex; align-items: center; gap: 14px; }
.header-left .icon { font-size: 30px; }
.header-left h1 {
  font-family: var(--display); font-size: 1.3rem; color: var(--accent);
  font-weight: 700; letter-spacing: -0.02em;
}
.header-left .sub {
  font-size: 10px; color: var(--text-mut); font-family: var(--mono);
}
.header-right { display: flex; gap: 8px; align-items: center; }

/* ── BUTTONS ────────────────────────────────────────────────────── */
.btn {
  border: none; border-radius: 8px; padding: 8px 18px;
  font-size: 12px; cursor: pointer; font-weight: 600;
  transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px;
}
.btn-primary { background: var(--accent-dk); color: #fff; }
.btn-primary:hover { background: var(--accent); }
.btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text-sec); }
.btn-outline:hover { border-color: var(--accent); color: var(--accent); }
.btn-sm { padding: 5px 12px; font-size: 11px; }

/* ── FILE SELECTOR ──────────────────────────────────────────────── */
.file-selector {
  padding: 20px 28px; border-bottom: 1px solid var(--border);
  background: var(--bg-panel);
}
.file-selector h2 {
  font-size: 14px; font-weight: 700; color: var(--accent); margin-bottom: 10px;
}
.file-grid {
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;
}
.file-chip {
  background: var(--bg-input); border: 1.5px solid var(--border);
  border-radius: 8px; padding: 8px 14px; cursor: pointer;
  transition: all 0.15s; font-size: 11px; user-select: none;
}
.file-chip:hover { border-color: var(--accent); }
.file-chip.selected {
  background: rgba(168,85,247,0.12); border-color: var(--accent);
  color: var(--accent); font-weight: 700;
}
.file-chip .fname { font-family: var(--mono); font-weight: 600; }
.file-chip .fmeta { font-size: 9px; color: var(--text-mut); margin-top: 3px; }
.selector-actions { display: flex; gap: 8px; align-items: center; }
.selector-hint {
  font-size: 10px; color: var(--text-mut); font-family: var(--mono);
  margin-left: 10px;
}

/* ── TAB BAR ────────────────────────────────────────────────────── */
.tab-bar {
  display: flex; gap: 2px; padding: 8px 28px 0;
  border-bottom: 1px solid var(--border);
}
.tab-btn {
  background: transparent; border: none;
  border-bottom: 2px solid transparent;
  padding: 8px 18px; color: var(--text-sec); font-size: 12px;
  cursor: pointer; font-weight: 500; transition: all 0.15s;
  margin-bottom: -1px;
}
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 700; }

/* ── MAIN CONTENT ───────────────────────────────────────────────── */
.content { padding: 24px 28px; max-width: 1200px; margin: 0 auto; }
.view-panel { display: none; }
.view-panel.active { display: block; }

/* ── STAT CARDS ──────────────────────────────────────────────────── */
.stats-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
.stat-card {
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-panel) 100%);
  border: 1px solid var(--border); border-radius: 14px; padding: 16px 20px;
  flex: 1; min-width: 140px;
}
.stat-card .label {
  font-size: 10px; color: var(--text-mut); text-transform: uppercase;
  letter-spacing: 0.08em; font-family: var(--mono);
}
.stat-card .value {
  font-size: 28px; font-weight: 800; font-family: var(--display);
  line-height: 1.2; margin-top: 4px;
}
.stat-card .sub { font-size: 10px; color: var(--text-sec); margin-top: 2px; }

/* ── CHART CONTAINERS ────────────────────────────────────────────── */
.chart-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; padding: 20px; margin-bottom: 20px;
}
.chart-card h3 {
  font-size: 14px; font-weight: 700; color: var(--accent);
  margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.chart-wrap { position: relative; width: 100%; }
.chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }

/* ── TRANSITION CARDS ────────────────────────────────────────────── */
.transition-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
}
.transition-header {
  display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
  font-size: 13px; font-weight: 700; color: var(--teal);
}
.transition-header .arrow { color: var(--accent); }
.change-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
  font-size: 12px; font-weight: 700; font-family: var(--mono);
  padding: 4px 12px; border-radius: 6px;
}
.pill-ok { background: rgba(34,197,94,0.1); color: var(--ok); }
.pill-bad { background: rgba(239,68,68,0.1); color: var(--bad); }
.pill-warn { background: rgba(245,158,11,0.1); color: var(--warn); }
.pill-mute { background: rgba(107,95,138,0.1); color: var(--text-mut); }

/* ── CANDIDATE TABLE ─────────────────────────────────────────────── */
.table-wrap {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 14px; overflow: hidden; margin-bottom: 20px;
}
.table-header {
  padding: 12px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.table-header h3 { font-size: 14px; font-weight: 700; color: var(--accent); }
.table-filter {
  background: var(--bg-input); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 12px; color: var(--text);
  font-size: 11px; font-family: var(--mono); outline: none;
}
.table-filter:focus { border-color: var(--border-f); }
table { width: 100%; border-collapse: collapse; }
th {
  text-align: left; padding: 10px 16px; font-size: 10px;
  color: var(--text-mut); text-transform: uppercase;
  letter-spacing: 0.06em; font-family: var(--mono);
  border-bottom: 1px solid var(--border);
  background: var(--bg-panel);
}
td {
  padding: 8px 16px; font-size: 12px; border-bottom: 1px solid rgba(42,37,69,0.3);
}
tr { cursor: pointer; transition: background 0.1s; }
tr:hover { background: var(--glow); }
.change-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 4px;
}
.dot-ok { background: var(--ok); }
.dot-bad { background: var(--bad); }
.dot-warn { background: var(--warn); }
.dot-mute { background: var(--text-mut); }

/* ── CANDIDATE DETAIL MODAL ──────────────────────────────────────── */
.modal-backdrop {
  position: fixed; inset: 0; z-index: 500;
  background: rgba(8,6,18,0.85); backdrop-filter: blur(4px);
  display: flex; align-items: center; justify-content: center;
  padding: 20px; animation: fadeIn 0.15s ease;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.modal-box {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 16px; width: 100%; max-width: 1000px;
  max-height: 90vh; display: flex; flex-direction: column;
  box-shadow: 0 24px 80px rgba(0,0,0,0.6);
  animation: slideUp 0.18s ease;
}
@keyframes slideUp {
  from { transform: translateY(18px); opacity: 0; }
  to   { transform: translateY(0); opacity: 1; }
}
.modal-head {
  padding: 16px 24px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.modal-head h2 { font-size: 16px; color: var(--accent); }
.modal-close {
  background: var(--bg-input); border: 1px solid var(--border);
  border-radius: 8px; width: 32px; height: 32px; color: var(--text-sec);
  font-size: 16px; cursor: pointer; display: flex;
  align-items: center; justify-content: center;
}
.modal-close:hover { border-color: var(--bad); color: var(--bad); }
.modal-body { flex: 1; overflow-y: auto; padding: 20px 24px; }

.diff-field {
  margin-bottom: 14px; background: var(--bg-panel);
  border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px;
}
.diff-field-name {
  font-size: 11px; font-weight: 700; color: var(--text-mut);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
  font-family: var(--mono);
}
.diff-values { display: flex; gap: 8px; flex-wrap: wrap; }
.diff-val {
  flex: 1; min-width: 200px; background: var(--bg-input);
  border-radius: 6px; padding: 8px 12px; font-size: 11px;
  font-family: var(--mono); line-height: 1.5; word-break: break-word;
  border: 1.5px solid transparent;
}
.diff-val.improved { border-color: var(--ok); }
.diff-val.regressed { border-color: var(--bad); }
.diff-val.changed { border-color: var(--warn); }
.diff-val-label {
  font-size: 9px; color: var(--text-mut); margin-bottom: 4px;
  display: flex; align-items: center; gap: 5px;
}

/* ── EMPTY STATE ─────────────────────────────────────────────────── */
.empty-state {
  text-align: center; padding: 60px 20px; color: var(--text-sec);
}

/* ── ML AUDIT ────────────────────────────────────────────────────── */
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
.sev-dot {
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: 6px;
}
.sev-excellent { background: var(--ok); }
.sev-good { background: var(--teal); }
.sev-needs_improvement { background: var(--warn); }
.sev-critical { background: var(--bad); }
.chart-box { padding: 16px 20px; }
.chart-box canvas { max-height: 280px; }
.empty-state .icon { font-size: 48px; margin-bottom: 12px; }
.empty-state h3 { font-size: 16px; color: var(--text-mut); margin-bottom: 8px; }
.empty-state p { font-size: 12px; color: var(--text-mut); max-width: 400px; margin: 0 auto; }

/* ── LOADING ─────────────────────────────────────────────────────── */
.loading { text-align: center; padding: 40px; color: var(--text-sec); }
.loading::after {
  content: ''; display: inline-block; width: 20px; height: 20px;
  border: 2px solid var(--accent); border-top-color: transparent;
  border-radius: 50%; animation: spin 0.8s linear infinite;
  margin-left: 10px; vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── SCROLLBAR ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<!-- ── HEADER ──────────────────────────────────────────────────────── -->
<div class="header">
  <div class="header-left">
    <span class="icon">🔄</span>
    <div>
      <h1>Extraction Run Comparator</h1>
      <span class="sub">Track your AI's glow-up across runs ✨</span>
    </div>
  </div>
  <div class="header-right">
    <span id="dirLabel" style="font-size:10px;color:var(--text-mut);font-family:var(--mono)"></span>
    <button class="btn btn-outline btn-sm" onclick="loadFiles()">🔄 Refresh</button>
  </div>
</div>

<!-- ── FILE SELECTOR ──────────────────────────────────────────────── -->
<div class="file-selector">
  <h2>📂 Select Runs to Compare <span class="selector-hint">(click to toggle, min 2)</span></h2>
  <div class="file-grid" id="fileGrid"></div>
  <div class="selector-actions">
    <button class="btn btn-outline btn-sm" onclick="selectAll()">Select All</button>
    <button class="btn btn-outline btn-sm" onclick="deselectAll()">Deselect All</button>
    <button class="btn btn-primary" onclick="runComparison()" id="compareBtn" disabled>
      🔬 Compare Selected
    </button>
    <span id="selCount" class="selector-hint"></span>
  </div>
</div>

<!-- ── TAB BAR ─────────────────────────────────────────────────────── -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview')" id="tab-overview">📊 Overview</button>
  <button class="tab-btn" onclick="switchTab('trends')" id="tab-trends">📈 Trends</button>
  <button class="tab-btn" onclick="switchTab('candidates')" id="tab-candidates">👥 Candidates</button>
  <button class="tab-btn" onclick="switchTab('quality')" id="tab-quality">🔬 ML Audit</button>
</div>

<!-- ── CONTENT PANELS ──────────────────────────────────────────────── -->
<div class="content">

  <!-- OVERVIEW -->
  <div class="view-panel active" id="view-overview">
    <div id="overviewContent">
      <div class="empty-state">
        <div class="icon">📂</div>
        <h3>Select files &amp; compare</h3>
        <p>Pick at least 2 JSON export files above, then hit Compare to see the magic! ✨</p>
      </div>
    </div>
  </div>

  <!-- TRENDS -->
  <div class="view-panel" id="view-trends">
    <div id="trendsContent">
      <div class="empty-state">
        <div class="icon">📈</div>
        <h3>Run a comparison first</h3>
        <p>Trend charts will appear here after you compare runs.</p>
      </div>
    </div>
  </div>

  <!-- CANDIDATES -->
  <div class="view-panel" id="view-candidates">
    <div id="candidatesContent">
      <div class="empty-state">
        <div class="icon">👥</div>
        <h3>Run a comparison first</h3>
        <p>Candidate-level diffs will appear here after you compare runs.</p>
      </div>
    </div>
  </div>

  <!-- ML QUALITY AUDIT -->
  <div class="view-panel" id="view-quality">
    <div id="qualityContent">
      <div class="empty-state">
        <div class="icon">🔬</div>
        <h3>Run a comparison first</h3>
        <p>ML Quality Audit will appear here — similarity metrics, error taxonomy, and weakest fields.</p>
      </div>
    </div>
  </div>

</div>

<!-- ── CANDIDATE DETAIL MODAL ──────────────────────────────────────── -->
<div class="modal-backdrop" id="modalBackdrop" style="display:none" onclick="closeModal(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-head">
      <h2 id="modalTitle">Candidate Detail</h2>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

<script>
// =================================================================
// 🧠 STATE
// =================================================================
let allFiles = [];
let selectedFiles = new Set();
let comparisonResult = null;
let activeTab = 'overview';

// Chart instances — must destroy before recreating!
let chartInstances = {};

// =================================================================
// 🌐 API HELPERS
// =================================================================
const API = '';  // Same origin

async function apiFetch(url, opts = {}) {
  const res = await fetch(API + url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

// =================================================================
// 📂 FILE LOADING
// =================================================================
async function loadFiles() {
  try {
    const data = await apiFetch('/api/files');
    allFiles = data.files || [];
    document.getElementById('dirLabel').textContent = data.directory || '';
    renderFileGrid();
  } catch (err) {
    document.getElementById('fileGrid').innerHTML =
      `<div style="color:var(--bad);font-size:12px">Error loading files: ${esc(err.message)}</div>`;
  }
}

function renderFileGrid() {
  const grid = document.getElementById('fileGrid');
  if (allFiles.length === 0) {
    grid.innerHTML = '<div style="color:var(--text-mut);font-size:12px">No JSON files found in the exports directory.</div>';
    return;
  }
  grid.innerHTML = allFiles.map(f => `
    <div class="file-chip ${selectedFiles.has(f.filename) ? 'selected' : ''}"
         onclick="toggleFile('${esc(f.filename)}')">
      <div class="fname">${esc(f.filename)}</div>
      <div class="fmeta">${esc(f.timestamp_display)} · ${f.record_count} candidates · ${esc(f.size_display)}</div>
    </div>
  `).join('');
  updateSelCount();
}

function toggleFile(filename) {
  if (selectedFiles.has(filename)) {
    selectedFiles.delete(filename);
  } else {
    selectedFiles.add(filename);
  }
  renderFileGrid();
}

function selectAll() {
  allFiles.forEach(f => selectedFiles.add(f.filename));
  renderFileGrid();
}

function deselectAll() {
  selectedFiles.clear();
  renderFileGrid();
}

function updateSelCount() {
  const n = selectedFiles.size;
  document.getElementById('selCount').textContent = `${n} selected`;
  document.getElementById('compareBtn').disabled = n < 2;
}

// =================================================================
// 🔬 COMPARISON
// =================================================================
async function runComparison() {
  const btn = document.getElementById('compareBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Comparing...';

  try {
    const result = await apiFetch('/api/compare', {
      method: 'POST',
      body: JSON.stringify({ files: Array.from(selectedFiles) }),
    });
    comparisonResult = result;
    renderOverview();
    renderTrends();
    renderCandidates();
    renderQuality();
  } catch (err) {
    document.getElementById('overviewContent').innerHTML =
      `<div class="empty-state"><div class="icon">❌</div><h3>${esc(err.message)}</h3></div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔬 Compare Selected';
  }
}

// =================================================================
// 📊 OVERVIEW TAB
// =================================================================
function renderOverview() {
  const r = comparisonResult;
  if (!r) return;
  const wrap = document.getElementById('overviewContent');
  let html = '';

  // ── Stat cards ──
  html += '<div class="stats-row">';
  html += statCard('🔄 Runs', r.run_count, 'var(--accent)');
  html += statCard('👥 Candidates', r.total_unique_candidates, 'var(--teal)');

  const last = r.run_summaries[r.run_summaries.length - 1];
  const first = r.run_summaries[0];
  const delta = (last.overall_completeness - first.overall_completeness).toFixed(1);
  const deltaColor = delta > 0 ? 'var(--ok)' : delta < 0 ? 'var(--bad)' : 'var(--text-mut)';
  html += statCard('📊 Latest Completeness', last.overall_completeness + '%', deltaColor,
    `${delta > 0 ? '+' : ''}${delta}% vs first run`);

  if (r.improvement && r.improvement.transitions.length > 0) {
    const lt = r.improvement.transitions[r.improvement.transitions.length - 1];
    html += statCard('📈 Improvements', lt.counts.improved, 'var(--ok)', `${lt.improvement_rate}% rate`);
    html += statCard('📉 Regressions', lt.counts.regressed, 'var(--bad)', `${lt.regression_rate}% rate`);
  }
  html += '</div>';

  // ── Transition cards ──
  if (r.improvement && r.improvement.transitions.length > 0) {
    r.improvement.transitions.forEach(t => {
      html += `<div class="transition-card">
        <div class="transition-header">
          <span>${shortName(t.from_filename)}</span>
          <span class="arrow">→</span>
          <span>${shortName(t.to_filename)}</span>
        </div>
        <div class="change-pills">
          <span class="pill pill-ok">✅ ${t.counts.improved} improved</span>
          <span class="pill pill-bad">❌ ${t.counts.regressed} regressed</span>
          <span class="pill pill-warn">🔄 ${t.counts.changed} changed</span>
          <span class="pill pill-mute">➖ ${t.counts.unchanged} same</span>
        </div>
      </div>`;
    });
  }

  // ── Completeness comparison chart ──
  html += `<div class="chart-card">
    <h3>📊 Overall Completeness per Run</h3>
    <div class="chart-wrap"><canvas id="chartCompleteness"></canvas></div>
  </div>`;

  // ── Run comparison bar chart ──
  html += `<div class="chart-card">
    <h3>📋 Field Completeness — Latest vs First Run</h3>
    <div class="chart-wrap"><canvas id="chartFieldCompare"></canvas></div>
  </div>`;

  wrap.innerHTML = html;

  // ── Draw charts ──
  drawCompletenessChart(r);
  drawFieldCompareChart(r);
}

function drawCompletenessChart(r) {
  destroyChart('chartCompleteness');
  const ctx = document.getElementById('chartCompleteness');
  if (!ctx) return;

  const labels = r.run_summaries.map(s => shortName(s.filename));
  const data = r.run_summaries.map(s => s.overall_completeness);

  chartInstances['chartCompleteness'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Completeness %',
        data,
        backgroundColor: data.map((v, i) =>
          i === data.length - 1 ? 'rgba(192,132,252,0.7)' : 'rgba(139,92,246,0.35)'
        ),
        borderColor: 'rgba(192,132,252,0.9)',
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: '#6b5f8a' }, grid: { color: '#2a2545' } },
        x: { ticks: { color: '#9b8fc4', font: { size: 10 } }, grid: { display: false } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function drawFieldCompareChart(r) {
  destroyChart('chartFieldCompare');
  const ctx = document.getElementById('chartFieldCompare');
  if (!ctx) return;

  const first = r.run_summaries[0];
  const last = r.run_summaries[r.run_summaries.length - 1];
  const fields = Object.keys(first.field_fill);
  const labels = fields.map(f => {
    const info = FIELD_LABELS[f];
    return info || f;
  });

  chartInstances['chartFieldCompare'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: shortName(first.filename),
          data: fields.map(f => first.field_fill[f]?.pct || 0),
          backgroundColor: 'rgba(139,92,246,0.3)',
          borderColor: 'rgba(139,92,246,0.6)',
          borderWidth: 1, borderRadius: 4,
        },
        {
          label: shortName(last.filename),
          data: fields.map(f => last.field_fill[f]?.pct || 0),
          backgroundColor: 'rgba(45,212,191,0.4)',
          borderColor: 'rgba(45,212,191,0.7)',
          borderWidth: 1, borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true, indexAxis: 'y',
      scales: {
        x: { beginAtZero: true, max: 100, ticks: { color: '#6b5f8a' }, grid: { color: '#2a2545' } },
        y: { ticks: { color: '#9b8fc4', font: { size: 10, family: 'JetBrains Mono' } }, grid: { display: false } }
      },
      plugins: { legend: { labels: { color: '#9b8fc4', font: { size: 10 } } } }
    }
  });
}

// =================================================================
// 📈 TRENDS TAB
// =================================================================
function renderTrends() {
  const r = comparisonResult;
  if (!r) return;
  const wrap = document.getElementById('trendsContent');
  let html = '';

  // ── Line chart: field completeness over runs ──
  html += `<div class="chart-card">
    <h3>📈 Field Completeness Trend Across Runs</h3>
    <div class="chart-wrap"><canvas id="chartTrend"></canvas></div>
  </div>`;

  // ── Improvement/regression stacked bar per transition ──
  if (r.improvement.transitions.length > 0) {
    html += `<div class="chart-card">
      <h3>🔄 Changes per Transition</h3>
      <div class="chart-wrap"><canvas id="chartChanges"></canvas></div>
    </div>`;
  }

  wrap.innerHTML = html;
  drawTrendChart(r);
  drawChangesChart(r);
}

function drawTrendChart(r) {
  destroyChart('chartTrend');
  const ctx = document.getElementById('chartTrend');
  if (!ctx) return;

  const labels = r.run_summaries.map(s => shortName(s.filename));
  const colors = [
    '#c084fc', '#2dd4bf', '#fbbf24', '#ec4899', '#22c55e',
    '#ef4444', '#8b5cf6', '#f59e0b', '#06b6d4', '#a855f7',
    '#fb923c', '#84cc16', '#e879f9',
  ];

  // Pick key fields for the trend chart (not ALL — too noisy)
  const keyFields = ['Name','Email','Phone','Current Company','Current Title',
                     'Summary','Work Experience','Education','tags'];
  const datasets = keyFields.map((f, i) => {
    const trend = r.field_trends[f];
    if (!trend) return null;
    return {
      label: FIELD_LABELS[f] || f,
      data: trend.map(t => t.pct),
      borderColor: colors[i % colors.length],
      backgroundColor: 'transparent',
      borderWidth: 2,
      tension: 0.3,
      pointRadius: 4,
      pointBackgroundColor: colors[i % colors.length],
    };
  }).filter(Boolean);

  chartInstances['chartTrend'] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: '#6b5f8a' }, grid: { color: '#2a2545' } },
        x: { ticks: { color: '#9b8fc4', font: { size: 10 } }, grid: { color: '#1a1630' } }
      },
      plugins: {
        legend: {
          labels: { color: '#9b8fc4', font: { size: 10 }, usePointStyle: true, pointStyle: 'circle' }
        }
      }
    }
  });
}

function drawChangesChart(r) {
  destroyChart('chartChanges');
  const ctx = document.getElementById('chartChanges');
  if (!ctx) return;
  if (!r.improvement.transitions.length) return;

  const labels = r.improvement.transitions.map(t =>
    shortName(t.from_filename) + ' → ' + shortName(t.to_filename)
  );

  chartInstances['chartChanges'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Improved', data: r.improvement.transitions.map(t => t.counts.improved),
          backgroundColor: 'rgba(34,197,94,0.6)', borderRadius: 4,
        },
        {
          label: 'Regressed', data: r.improvement.transitions.map(t => t.counts.regressed),
          backgroundColor: 'rgba(239,68,68,0.6)', borderRadius: 4,
        },
        {
          label: 'Changed', data: r.improvement.transitions.map(t => t.counts.changed),
          backgroundColor: 'rgba(245,158,11,0.5)', borderRadius: 4,
        },
      ]
    },
    options: {
      responsive: true,
      scales: {
        y: { stacked: true, ticks: { color: '#6b5f8a' }, grid: { color: '#2a2545' } },
        x: { stacked: true, ticks: { color: '#9b8fc4', font: { size: 10 } }, grid: { display: false } }
      },
      plugins: { legend: { labels: { color: '#9b8fc4', font: { size: 10 } } } }
    }
  });
}

// =================================================================
// 👥 CANDIDATES TAB
// =================================================================
let candidateFilter = '';

function renderCandidates() {
  const r = comparisonResult;
  if (!r || !r.candidate_diffs) return;
  const wrap = document.getElementById('candidatesContent');

  // Compute per-candidate change summary
  const summaries = r.candidate_diffs.map(cd => {
    let improved = 0, regressed = 0, changed = 0;
    for (const [, changes] of Object.entries(cd.field_changes)) {
      for (const ch of changes) {
        if (ch.change === 'improved') improved++;
        else if (ch.change === 'regressed') regressed++;
        else if (ch.change === 'changed') changed++;
      }
    }
    return { ...cd, improved, regressed, changed };
  });

  // Sort: most changes first
  summaries.sort((a, b) => (b.improved + b.regressed + b.changed) - (a.improved + a.regressed + a.changed));

  let html = `<div class="table-wrap">
    <div class="table-header">
      <h3>👥 Candidate-Level Changes</h3>
      <input class="table-filter" placeholder="🔍 Filter by name or ID..."
             oninput="filterCandidates(this.value)" id="candFilter">
    </div>
    <table>
      <thead><tr>
        <th>ID</th><th>Name</th>
        <th>✅ Improved</th><th>❌ Regressed</th><th>🔄 Changed</th>
        <th>Runs Present</th>
      </tr></thead>
      <tbody id="candTableBody">`;

  summaries.forEach(cd => {
    const presenceStr = cd.presence.map((p, i) => p ? `R${i}` : '').filter(Boolean).join(', ');
    html += `<tr onclick="openCandidateModal(${cd.id})" data-name="${esc(cd.name)}" data-id="${cd.id}">
      <td style="font-family:var(--mono);font-weight:600">${cd.id}</td>
      <td>${esc(cd.name) || '<span style="color:var(--text-mut)">—</span>'}</td>
      <td>${cd.improved > 0 ? `<span class="change-dot dot-ok"></span>${cd.improved}` : '<span style="color:var(--text-mut)">0</span>'}</td>
      <td>${cd.regressed > 0 ? `<span class="change-dot dot-bad"></span>${cd.regressed}` : '<span style="color:var(--text-mut)">0</span>'}</td>
      <td>${cd.changed > 0 ? `<span class="change-dot dot-warn"></span>${cd.changed}` : '<span style="color:var(--text-mut)">0</span>'}</td>
      <td style="font-size:10px;color:var(--text-mut)">${presenceStr}</td>
    </tr>`;
  });

  html += '</tbody></table></div>';
  wrap.innerHTML = html;
}

function filterCandidates(query) {
  const q = query.toLowerCase();
  document.querySelectorAll('#candTableBody tr').forEach(tr => {
    const name = (tr.dataset.name || '').toLowerCase();
    const id = (tr.dataset.id || '').toLowerCase();
    tr.style.display = (name.includes(q) || id.includes(q)) ? '' : 'none';
  });
}

// =================================================================
// 🔍 CANDIDATE DETAIL MODAL
// =================================================================
async function openCandidateModal(candidateId) {
  const backdrop = document.getElementById('modalBackdrop');
  const body = document.getElementById('modalBody');
  const title = document.getElementById('modalTitle');

  backdrop.style.display = 'flex';
  title.textContent = `Candidate #${candidateId}`;
  body.innerHTML = '<div class="loading">Loading detail</div>';

  try {
    const data = await apiFetch(`/api/candidate/${candidateId}`, {
      method: 'POST',
      body: JSON.stringify({ files: Array.from(selectedFiles) }),
    });

    title.textContent = `#${candidateId} — ${data.runs.find(r => r.present)?.record?.Name || 'Unknown'}`;

    let html = '';
    // Show each tracked field across runs
    const fields = Object.keys(FIELD_LABELS);
    fields.forEach(fieldName => {
      html += `<div class="diff-field">
        <div class="diff-field-name">${FIELD_LABELS[fieldName] || fieldName}</div>
        <div class="diff-values">`;

      data.runs.forEach((run, i) => {
        const val = run.present ? (run.record[fieldName] ?? '') : null;
        const displayVal = val === null ? '(not in this run)' :
                           (typeof val === 'object' ? JSON.stringify(val, null, 1) : String(val || '—'));

        // Determine change class
        let cls = '';
        if (i > 0 && run.present && data.runs[i-1].present) {
          const prev = data.runs[i-1].record[fieldName];
          const cur = run.record[fieldName];
          const prevFilled = prev !== null && prev !== undefined && String(prev).trim() !== '';
          const curFilled = cur !== null && cur !== undefined && String(cur).trim() !== '';
          if (!prevFilled && curFilled) cls = 'improved';
          else if (prevFilled && !curFilled) cls = 'regressed';
          else if (prevFilled && curFilled && JSON.stringify(prev) !== JSON.stringify(cur)) cls = 'changed';
        }

        html += `<div class="diff-val ${cls}">
          <div class="diff-val-label">
            ${cls === 'improved' ? '<span class="change-dot dot-ok"></span>' :
              cls === 'regressed' ? '<span class="change-dot dot-bad"></span>' :
              cls === 'changed' ? '<span class="change-dot dot-warn"></span>' : ''}
            Run ${i} · ${esc(run.filename)}
          </div>
          <div style="max-height:120px;overflow:auto;white-space:pre-wrap">${esc(truncate(displayVal, 500))}</div>
        </div>`;
      });

      html += '</div></div>';
    });

    body.innerHTML = html;
  } catch (err) {
    body.innerHTML = `<div class="empty-state"><div class="icon">❌</div><h3>${esc(err.message)}</h3></div>`;
  }
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('modalBackdrop')) return;
  document.getElementById('modalBackdrop').style.display = 'none';
}

// =================================================================
// 🔬 ML QUALITY AUDIT TAB
// =================================================================
function renderQuality() {
  const r = comparisonResult;
  if (!r || !r.quality_audit || !r.quality_audit.available) {
    document.getElementById('qualityContent').innerHTML =
      '<div class="empty-state"><div class="icon">🔬</div><h3>Quality audit not available</h3></div>';
    return;
  }
  const qa = r.quality_audit;
  const wrap = document.getElementById('qualityContent');
  let html = '';

  // ── Summary stat cards ──
  const latestNonRef = qa.quality_by_run.length >= 2 ? qa.quality_by_run[qa.quality_by_run.length - 2] : null;
  const firstRun = qa.quality_by_run[0];
  const et = qa.error_taxonomy;

  html += '<div class="stats-row">';
  html += statCard('📏 Reference', shortName(qa.reference_run), 'var(--accent)');
  if (firstRun && !firstRun.is_reference) {
    const c = firstRun.overall_similarity_pct >= 90 ? 'var(--ok)' : firstRun.overall_similarity_pct >= 70 ? 'var(--warn)' : 'var(--bad)';
    html += statCard('🥇 First Run Similarity', firstRun.overall_similarity_pct + '%', c);
  }
  if (latestNonRef && !latestNonRef.is_reference) {
    const c = latestNonRef.overall_similarity_pct >= 90 ? 'var(--ok)' : latestNonRef.overall_similarity_pct >= 70 ? 'var(--warn)' : 'var(--bad)';
    html += statCard('🏁 Prev Run Similarity', latestNonRef.overall_similarity_pct + '%', c);
  }
  html += statCard('🏷️ Total Diffs', et.total_errors, et.total_errors > 0 ? 'var(--bad)' : 'var(--ok)');
  html += statCard('✅ Matches', et.total_matches, 'var(--ok)');
  html += '</div>';

  // ── Quality trend chart ──
  html += '<div class="audit-section">';
  html += '<div class="audit-section-header">📈 Quality Trend — Similarity to Reference per Run</div>';
  html += '<div class="chart-box"><canvas id="qualityTrendChart"></canvas></div>';
  html += '</div>';

  // ── Per-run similarity table ──
  html += '<div class="audit-section">';
  html += '<div class="audit-section-header">📊 Per-Run Quality Score (vs ' + esc(shortName(qa.reference_run)) + ')</div>';
  html += '<table class="audit-table"><thead><tr>';
  html += '<th>Run</th><th>File</th><th>Similarity</th><th>Fields Compared</th><th>Boundary</th><th>Value Δ</th><th>Missing</th><th>Spurious</th><th>Match</th>';
  html += '</tr></thead><tbody>';

  qa.quality_by_run.forEach(function(qr) {
    const simColor = qr.overall_similarity_pct >= 90 ? 'var(--ok)' : qr.overall_similarity_pct >= 70 ? 'var(--warn)' : 'var(--bad)';
    const tag = qr.is_reference ? ' <span style="color:var(--gold);font-size:9px">★ REF</span>' : '';
    html += '<tr>' +
      '<td>Run ' + qr.run_index + '</td>' +
      '<td style="font-weight:600">' + esc(shortName(qr.filename)) + tag + '</td>' +
      '<td style="color:' + simColor + ';font-weight:700">' + qr.overall_similarity_pct + '%</td>' +
      '<td>' + qr.total_comparisons + '</td>' +
      '<td style="color:var(--warn)">' + (qr.error_counts.boundary || 0) + '</td>' +
      '<td style="color:var(--purple)">' + (qr.error_counts.value_change || 0) + '</td>' +
      '<td style="color:var(--bad)">' + (qr.error_counts.missing || 0) + '</td>' +
      '<td style="color:var(--pink)">' + (qr.error_counts.spurious || 0) + '</td>' +
      '<td style="color:var(--ok)">' + (qr.error_counts.match || 0) + '</td>' +
      '</tr>';
  });
  html += '</tbody></table></div>';

  // ── Error taxonomy donut ──
  if (et.total_errors > 0) {
    html += '<div class="audit-section">';
    html += '<div class="audit-section-header">🏷️ Error Taxonomy — Why Runs Differ from Reference</div>';
    html += '<div style="display:grid;grid-template-columns:280px 1fr;gap:0">';
    html += '<div class="chart-box"><canvas id="qualityTaxChart" width="250" height="250"></canvas></div>';
    html += '<div style="padding:16px 20px">';
    html += '<table class="audit-table"><thead><tr><th>Category</th><th>Count</th><th>%</th><th>Meaning</th></tr></thead><tbody>';
    const taxInfo = {
      boundary: {icon: '🔲', desc: 'Clipped or extended text — substring mismatch'},
      value_change: {icon: '🔀', desc: 'Completely different value extracted'},
      missing: {icon: '👻', desc: 'Field empty in earlier run, filled in reference'},
      spurious: {icon: '🎪', desc: 'Field filled in earlier run, empty in reference'}
    };
    for (const cat of ['boundary','value_change','missing','spurious']) {
      const cnt = et.counts[cat] || 0;
      const pct = et.percentages[cat] || 0;
      const info = taxInfo[cat] || {icon:'?', desc: cat};
      html += '<tr>' +
        '<td style="font-weight:700">' + info.icon + ' ' + cat.replace('_',' ') + '</td>' +
        '<td>' + cnt + '</td><td>' + pct + '%</td>' +
        '<td style="font-family:var(--body);color:var(--text-sec)">' + info.desc + '</td></tr>';
    }
    html += '</tbody></table></div></div></div>';
  }

  // ── Weakest fields table ──
  if (qa.weakest_fields && qa.weakest_fields.length > 0) {
    html += '<div class="audit-section">';
    html += '<div class="audit-section-header">🎯 Field Stability Ranking — Weakest to Strongest</div>';
    html += '<table class="audit-table"><thead><tr>';
    html += '<th>Rank</th><th>Field</th><th>Avg Similarity</th><th>Severity</th><th>Comparisons</th>';
    html += '</tr></thead><tbody>';
    qa.weakest_fields.forEach(function(f, i) {
      const simColor = f.avg_similarity_pct >= 90 ? 'var(--ok)' : f.avg_similarity_pct >= 70 ? 'var(--warn)' : 'var(--bad)';
      html += '<tr>' +
        '<td style="color:var(--text-mut)">#' + (i+1) + '</td>' +
        '<td style="font-weight:700">' + esc(f.label) + '</td>' +
        '<td style="color:' + simColor + ';font-weight:700">' + f.avg_similarity_pct + '%</td>' +
        '<td><span class="sev-dot sev-' + f.severity + '"></span>' + f.severity.replace('_',' ') + '</td>' +
        '<td>' + f.comparisons + '</td></tr>';
    });
    html += '</tbody></table></div>';
  }

  wrap.innerHTML = html;

  // ── Draw charts ──
  drawQualityTrendChart(qa);
  drawQualityTaxChart(qa);
}

function drawQualityTrendChart(qa) {
  destroyChart('qualityTrendChart');
  const ctx = document.getElementById('qualityTrendChart');
  if (!ctx) return;

  const labels = qa.quality_by_run.map(function(qr) { return shortName(qr.filename); });
  const data = qa.quality_by_run.map(function(qr) { return qr.overall_similarity_pct; });

  chartInstances['qualityTrendChart'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Similarity to Reference %',
        data: data,
        borderColor: '#c084fc',
        backgroundColor: 'rgba(192,132,252,0.1)',
        borderWidth: 3,
        tension: 0.3,
        pointRadius: 6,
        pointBackgroundColor: data.map(function(v) {
          return v >= 90 ? '#22c55e' : v >= 70 ? '#f59e0b' : '#ef4444';
        }),
        fill: true,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: '#6b5f8a' }, grid: { color: '#2a2545' } },
        x: { ticks: { color: '#9b8fc4', font: { size: 10 } }, grid: { display: false } }
      },
      plugins: { legend: { labels: { color: '#9b8fc4' } } }
    }
  });
}

function drawQualityTaxChart(qa) {
  destroyChart('qualityTaxChart');
  const ctx = document.getElementById('qualityTaxChart');
  if (!ctx || !qa.error_taxonomy || qa.error_taxonomy.total_errors === 0) return;

  const et = qa.error_taxonomy;
  chartInstances['qualityTaxChart'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Boundary', 'Value Change', 'Missing', 'Spurious'],
      datasets: [{
        data: [et.counts.boundary||0, et.counts.value_change||0, et.counts.missing||0, et.counts.spurious||0],
        backgroundColor: ['#f59e0b', '#8b5cf6', '#ef4444', '#ec4899'],
        borderColor: '#151221',
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#9b8fc4', font: { size: 10 }, padding: 8 } }
      }
    }
  });
}

// =================================================================
// 🧩 UTILITIES
// =================================================================
const FIELD_LABELS = {
  'Name': 'Full Name', 'Email': 'Email', 'Phone': 'Phone',
  'Current Company': 'Current Company', 'Current Title': 'Current Title',
  'Current Location': 'Location', 'Function': 'Function', 'Industry': 'Industry',
  'Summary': 'Summary', 'Language Skills': 'Languages',
  'Project Experience': 'Projects',
  'Work Experience': 'Experience', 'Education': 'Education', 'tags': 'Skills/Tags',
};

function statCard(label, value, color, sub) {
  return `<div class="stat-card">
    <div class="label">${label}</div>
    <div class="value" style="color:${color}">${value}</div>
    ${sub ? '<div class="sub">' + sub + '</div>' : ''}
  </div>`;
}

function shortName(filename) {
  // Shorten "extracted_resume_data_20260401_052057.json" → "0401_0520"
  const m = filename.match(/(\d{4})(\d{2})(\d{2})_?(\d{2})?(\d{2})?/);
  if (m) return `${m[2]}/${m[3]} ${m[4]||''}:${m[5]||''}`.trim();
  return filename.replace('.json', '').slice(-15);
}

function esc(str) {
  if (!str) return '';
  const d = document.createElement('div');
  d.textContent = String(str);
  return d.innerHTML;
}

function truncate(str, max) {
  if (!str) return '';
  return str.length > max ? str.slice(0, max) + '...' : str;
}

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + tab)?.classList.add('active');
  document.getElementById('view-' + tab)?.classList.add('active');
}

// Keyboard: Escape closes modal
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// =================================================================
// 🚀 INIT
// =================================================================
loadFiles();
</script>
</body>
</html>
"""


# =============================================================================
# 🚀 MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="💅✨ Fairy Codemother's Extraction Run Comparator ✨💅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Dashboard mode (port 5075)
  %(prog)s --cli                        CLI summary only
  %(prog)s --dir exports/               Scan custom directory
  %(prog)s --files a.json b.json        Compare specific files
  %(prog)s --cli --dir exports/         CLI with custom directory
        """,
    )

    parser.add_argument(
        "--cli", action="store_true",
        help="CLI mode: print comparison to terminal instead of launching dashboard"
    )
    parser.add_argument(
        "--dir", default=DEFAULT_EXPORTS_DIR,
        help=f"Directory containing JSON export files (default: {DEFAULT_EXPORTS_DIR})"
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Specific JSON files to compare (overrides --dir discovery)"
    )
    parser.add_argument(
        "--port", type=int, default=PORT,
        help=f"Dashboard port (default: {PORT})"
    )

    args = parser.parse_args()

    # ── Resolve file list ──────────────────────────────────────────────
    if args.files:
        # User specified exact files
        file_metas = []
        for filepath in args.files:
            if not os.path.isfile(filepath):
                print(f"  ❌ File not found: {filepath}")
                continue
            meta = _validate_json_file(os.path.abspath(filepath))
            if meta:
                file_metas.append(meta)
            else:
                print(f"  ⚠️ Skipping {filepath}: not a valid candidate JSON")
        file_metas.sort(key=lambda f: f["timestamp"] or "0000")
    else:
        file_metas = discover_json_files(args.dir)

    # ── CLI MODE ──────────────────────────────────────────────────────
    if args.cli:
        if len(file_metas) < 2:
            print(f"\n  ❌ Need at least 2 valid JSON files to compare!")
            print(f"     Found {len(file_metas)} in: {os.path.abspath(args.dir)}")
            print(f"     Try: python compare_runs.py --dir /path/to/exports/\n")
            sys.exit(1)

        result = compare_runs(file_metas)
        print_cli_report(result)
        sys.exit(0)

    # ── DASHBOARD MODE ────────────────────────────────────────────────
    if not FLASK_AVAILABLE:
        print("\n  ❌ Flask not installed! Install with:")
        print("     pip install flask --break-system-packages\n")
        sys.exit(1)

    set_exports_dir(args.dir)

    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + " 🧚‍♀️✨ EXTRACTION RUN COMPARATOR ✨🧚‍♀️ ".center(62) + "║")
    print("╠" + "═" * 62 + "╣")
    print("║" + f"  📂 Scanning:  {os.path.abspath(args.dir)}".ljust(62) + "║")
    print("║" + f"  📋 Files:     {len(file_metas)} JSON files found".ljust(62) + "║")
    print("║" + f"  🌐 Server:    http://localhost:{args.port}".ljust(62) + "║")
    print("║" + "".ljust(62) + "║")
    print("║" + f"  ✨ DASHBOARD: http://localhost:{args.port}/dashboard".ljust(62) + "║")
    print("║" + f"     ↑ Open this in your browser, darling! 💅".ljust(62) + "║")
    print("║" + "".ljust(62) + "║")
    print("║" + "  Port Map:".ljust(62) + "║")
    print("║" + "    5001 → Classification Dashboard".ljust(62) + "║")
    print("║" + "    5050 → Review Dashboard".ljust(62) + "║")
    print("║" + "    5055 → NER Annotation Tool".ljust(62) + "║")
    print("║" + "    5070 → Accuracy Validator".ljust(62) + "║")
    print("║" + "    5075 → Run Comparator (this!) ✨".ljust(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

    if len(file_metas) == 0:
        print(f"  ⚠️  No JSON files found in {os.path.abspath(args.dir)}")
        print(f"     The dashboard will work once you add exported JSON files!")
        print()
    else:
        for fm in file_metas:
            print(f"  📄 {fm['filename']}  ({fm['record_count']} candidates, {fm['timestamp_display']})")
        print()

    app.run(host=HOST, port=args.port, debug=DEBUG)


if __name__ == "__main__":
    main()