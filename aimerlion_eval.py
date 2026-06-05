"""
aimerlion_eval.py  —  Fairy Codemother's all-in-one diagnostic eval. ✨

RUN IT:   python aimerlion_eval.py
It works immediately on built-in DEMO data so you can see the machine work.
Then swap in YOUR data at the two spots marked  >>> SWAP HERE <<<.

What it does, darling:
  - flattens nested output into atomic "leaf" fields (the 27-jewel idea)
  - compares raw model output, final pipeline output, and ground truth
  - names ONE root cause per failure (one victim, one killer)
  - two asserts make impossible scorecards (53-out-of-42) CRASH loudly
  - rolls every case into one grand verdict
"""

import sys
try:                       # keep emojis/box-drawing happy on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ════════════════════════════════════════════════════════════════════
#  THE SCORER
# ════════════════════════════════════════════════════════════════════
def _flatten(obj, prefix=""):
    """Explode nested dict/list into {leaf_path: value}."""
    leaves = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        leaves[prefix] = obj
    return leaves


def _norm(v):
    return "" if v is None else " ".join(str(v).split()).strip().lower()


def _absent(v):
    return _norm(v) in ("", "n/a", "none", "null", "-", "not specified")


def _is_derived(path, derived_fields):
    return any(path == d or path.startswith(d + ".") or path.startswith(d + "[")
               for d in derived_fields)


def score_extraction(raw_model_output, final_output, ground_truth,
                     overrides=None, derived_fields=None):
    """Returns (card_dict, per_field_details). See top-of-file notes."""
    overrides = overrides or {}
    derived_fields = set(derived_fields or [])
    gt    = _flatten(ground_truth)
    final = _flatten(final_output)
    raw   = _flatten(raw_model_output)

    final_vals = {_norm(v) for v in final.values() if not _absent(v)}
    raw_vals   = {_norm(v) for v in raw.values()   if not _absent(v)}
    gt_vals    = {_norm(v) for v in gt.values()    if not _absent(v)}

    counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    causes = {"llm_weakness": 0, "post_processing": 0,
              "schema_issue": 0, "prompt_issue": 0, "ocr_issue": 0}
    details = []

    def fail(path, cause, ftype):
        counts["incorrect"] += 1
        causes[cause] += 1
        details.append((path, f"incorrect:{ftype}/{cause}"))

    for path, gt_val in gt.items():
        # 1) Your expert override wins.
        if path in overrides:
            cause = overrides[path]
            if cause == "not_applicable":
                counts["not_applicable"] += 1
                details.append((path, "not_applicable:override"))
            else:
                counts["incorrect"] += 1
                causes[cause] += 1
                details.append((path, f"incorrect:override/{cause}"))
            continue

        # 2) Classification field → graded in its own harness, not here.
        if _is_derived(path, derived_fields):
            counts["not_applicable"] += 1
            details.append((path, "not_applicable:derived_field"))
            continue

        # 3) Source field supposed to be EMPTY — respected, misfiled, or invented?
        if _absent(gt_val):
            placed = final.get(path)
            if _absent(placed):
                counts["not_applicable"] += 1
                details.append((path, "not_applicable"))
            elif _norm(placed) in gt_vals:           # real data, wrong drawer
                fail(path, "schema_issue", "wrong_assignment")
            else:                                     # invented from nothing
                fail(path, "llm_weakness", "hallucination")
            continue

        # 4) Source has data — match?
        if _norm(final.get(path)) == _norm(gt_val):
            counts["correct"] += 1
            details.append((path, "correct"))
            continue

        # 5) WRONG → name the ONE killer (order is sacred).
        gt_n = _norm(gt_val)
        if gt_n in raw_vals and gt_n not in final_vals:
            fail(path, "post_processing", "truncation_or_missing")
        elif gt_n in final_vals:
            fail(path, "schema_issue", "wrong_assignment")
        else:
            fail(path, "llm_weakness", "missing_or_wrong")

    total = sum(counts.values())
    denom = counts["correct"] + counts["incorrect"]
    accuracy = round(counts["correct"] / denom, 3) if denom else 0.0

    assert sum(counts.values()) == total
    assert sum(causes.values()) == counts["incorrect"], \
        "every wrong field needs EXACTLY one root cause"

    card = {"total_fields_evaluated": total, **counts,
            "field_accuracy": accuracy, "failures_by_root_cause": causes}
    return card, details


def aggregate_cards(cards):
    counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    causes = {"llm_weakness": 0, "post_processing": 0,
              "schema_issue": 0, "prompt_issue": 0, "ocr_issue": 0}
    for c in cards:
        for k in counts:  counts[k] += c[k]
        for k in causes:  causes[k] += c["failures_by_root_cause"][k]

    total = sum(counts.values())                     # RECOMPUTE, never carry
    denom = counts["correct"] + counts["incorrect"]
    accuracy = round(counts["correct"] / denom, 3) if denom else 0.0

    assert sum(counts.values()) == total
    assert sum(causes.values()) == counts["incorrect"], (
        f"causes={sum(causes.values())} but incorrect={counts['incorrect']} — "
        f"a failure has multiple causes. Hunt the if/if that should be if/elif.")

    return {"total_fields_evaluated": total, **counts,
            "field_accuracy": accuracy, "failures_by_root_cause": causes}


# ════════════════════════════════════════════════════════════════════
#  >>> SWAP HERE #1 <<<  — your real extractor
#  Right now this returns the demo's pre-baked output. Replace the body
#  with a call to YOUR extractor, merging header + deep fields.
# ════════════════════════════════════════════════════════════════════
def run_extractor(case):
    # from extraction.ai_extractor import extract_header_fields, extract_deep_fields
    # header = extract_header_fields(case["resume_text"])
    # deep   = extract_deep_fields(case["resume_text"])
    # final  = {**header, **deep}        # <-- the merge that fixes empty Name/Email/Phone
    # raw    = final                     # <-- ideally your PRE-post-processing output
    # return raw, final
    return case["raw_model_output"], case["final_output"]   # demo fallback


# ════════════════════════════════════════════════════════════════════
#  >>> SWAP HERE #2 <<<  — your real golden cases
#  Replace load_cases() body with a read of validation/golden_sg_my.jsonl
# ════════════════════════════════════════════════════════════════════
def load_cases():
    # import json
    # return [json.loads(line) for line in
    #         open("validation/golden_sg_my.jsonl", encoding="utf-8")]
    return DEMO_CASES


# ────────────── DEMO DATA (delete once you wire in the real thing) ──────────────
DEMO_CASES = [
    {   # CASE A: clean win; Industry is a derived field -> excluded
        "id": "A_clean",
        "raw_model_output": {"Name": "Siti binti Rahman", "Phone": "+65 9123 4567",
                             "Current Company": "Acme Pte Ltd", "skills": ["python", "sql"]},
        "final_output":     {"Name": "Siti binti Rahman", "Phone": "+65 9123 4567",
                             "Current Company": "Acme Pte Ltd", "skills": ["python", "sql"]},
        "ground_truth":     {"Name": "Siti binti Rahman", "Phone": "+65 9123 4567",
                             "Current Company": "Acme Pte Ltd", "skills": ["python", "sql"],
                             "Industry": "Technology"},
    },
    {   # CASE B: model got it right, a PATCH truncated the name & dropped a skill
        "id": "B_postproc",
        "raw_model_output": {"Name": "Ahmad bin Ismail", "skills": ["python", "java", "sql"]},
        "final_output":     {"Name": "Ahmad bin",         "skills": ["python", "java"]},
        "ground_truth":     {"Name": "Ahmad bin Ismail",  "skills": ["python", "java", "sql"]},
    },
    {   # CASE C: company/title SWAPPED (schema), and a location HALLUCINATED (llm)
        "id": "C_schema_halluc",
        "raw_model_output": {"Current Company": "DBS Bank", "Current Title": "Analyst"},
        "final_output":     {"Current Company": "Analyst",  "Current Title": "DBS Bank",
                             "Current Location": "Singapore"},
        "ground_truth":     {"Current Company": "DBS Bank", "Current Title": "Analyst",
                             "Current Location": ""},   # empty in source -> should stay empty
    },
]


# ════════════════════════════════════════════════════════════════════
#  THE EVAL LOOP  (this is the whole "loop" — one lap per case)
# ════════════════════════════════════════════════════════════════════
def main():
    cases = load_cases()
    cards = []

    print("\n" + "═" * 60)
    print("  ✨ FAIRY CODEMOTHER — Per-Case Runway ✨")
    print("═" * 60)

    for case in cases:
        raw, final = run_extractor(case)
        card, details = score_extraction(
            raw_model_output=raw,
            final_output=final,
            ground_truth=case["ground_truth"],
            derived_fields={"Function", "Industry"},   # classification -> graded elsewhere
            # overrides={"Name": "post_processing"},    # your eyeball verdicts go here
        )
        cards.append(card)
        cid = case.get("id", "?")
        print(f"\n  👗 {cid}:  {card['correct']}✓  {card['incorrect']}✗  "
              f"{card['not_applicable']}∅   acc={card['field_accuracy']}")
        for path, verdict in details:
            mark = "✓" if verdict == "correct" else ("∅" if "not_applicable" in verdict else "✗")
            if mark != "✓":
                print(f"        {mark} {path:24s} -> {verdict}")

    grand = aggregate_cards(cards)

    print("\n" + "╔" + "═" * 50 + "╗")
    print("║" + "  🏆 GRAND VERDICT — all cases combined".ljust(50) + "║")
    print("╠" + "═" * 50 + "╣")
    print("║" + f"  fields evaluated : {grand['total_fields_evaluated']}".ljust(50) + "║")
    print("║" + f"  correct          : {grand['correct']}".ljust(50) + "║")
    print("║" + f"  incorrect        : {grand['incorrect']}".ljust(50) + "║")
    print("║" + f"  not_applicable   : {grand['not_applicable']}".ljust(50) + "║")
    print("║" + f"  FIELD ACCURACY   : {grand['field_accuracy']}".ljust(50) + "║")
    print("╠" + "═" * 50 + "╣")
    print("║" + "  failures by root cause (your training brief):".ljust(50) + "║")
    for cause, n in grand["failures_by_root_cause"].items():
        bar = "█" * n
        print("║" + f"    {cause:16s} {n:3d} {bar}".ljust(50)[:50] + "║")
    print("╚" + "═" * 50 + "╝")
    print("\n  Biggest bucket = your Act 1 door. 💖\n")


if __name__ == "__main__":
    main()