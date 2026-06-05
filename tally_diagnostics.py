"""
tally_diagnostics.py  —  Fairy Codemother's honest counter. ✨

This does NOT auto-grade. It reads the field_evaluation block that YOU
hand-annotated (correct = true/false/null, root_cause = ...) and tallies
it into a trustworthy `summary`. Your judgment stays sovereign — this just
makes the arithmetic incapable of lying (no more 53-out-of-42).

It also CATCHES bad annotations:
  - a field marked incorrect but with no root_cause
  - a field marked correct but with a root_cause attached
  - an invalid root_cause value
  - a case you haven't annotated yet (all nulls)

RUN:   python tally_diagnostics.py "diagnostic_cases/*.json"
       (defaults to diagnostic_cases/*.json if no path given)

By default it WRITES the computed summary back into each clean case file.
Set WRITE_BACK = False to only print.
"""

import sys, json, glob, os
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WRITE_BACK = True
VALID_CAUSES = ("llm_weakness", "post_processing",
                "schema_issue", "prompt_issue", "ocr_issue")


def tally_case(case):
    """Returns (summary, problems). summary is None if the case has problems."""
    cid = case.get("case_id", "?")
    fe = case.get("field_evaluation", {})

    correct = incorrect = annotated = 0
    causes = {c: 0 for c in VALID_CAUSES}
    problems = []

    for field, ev in fe.items():
        if field.startswith("_"):          # skip "_instructions"
            continue
        if not isinstance(ev, dict):
            continue
        verdict = ev.get("correct", None)
        cause = (ev.get("root_cause") or "").strip()

        if verdict is True:
            correct += 1
            annotated += 1
            if cause:
                problems.append(f"{field}: marked CORRECT but has root_cause '{cause}'")
        elif verdict is False:
            incorrect += 1
            annotated += 1
            if not cause:
                problems.append(f"{field}: marked INCORRECT but root_cause is EMPTY")
            elif cause not in VALID_CAUSES:
                problems.append(f"{field}: invalid root_cause '{cause}'")
            else:
                causes[cause] += 1
        elif verdict is None:
            pass                            # not applicable / not yet judged
        else:
            problems.append(f"{field}: 'correct' must be true/false/null, got {verdict!r}")

    if annotated == 0:
        problems.append("NOT ANNOTATED YET (every field is still null)")

    if problems:
        return None, problems

    total = correct + incorrect
    summary = {
        "total_fields_evaluated": total,
        "correct": correct,
        "incorrect": incorrect,
        "field_accuracy": round(correct / total, 3) if total else 0.0,
        "failures_by_root_cause": causes,
    }
    # honesty guards — only reachable for clean cases, hold by construction
    assert summary["correct"] + summary["incorrect"] == summary["total_fields_evaluated"]
    assert sum(causes.values()) == summary["incorrect"]
    return summary, []


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "diagnostic_cases/*.json"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No case files matched: {pattern} 😱")

    grand = {"correct": 0, "incorrect": 0}
    grand_causes = {c: 0 for c in VALID_CAUSES}
    clean, flagged = [], []

    print("\n" + "═" * 62)
    print("  ✨ FAIRY CODEMOTHER — Diagnostic Tally ✨")
    print("═" * 62)

    for path in paths:
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
        cid = case.get("case_id", os.path.basename(path))
        summary, problems = tally_case(case)

        if problems:
            flagged.append((cid, problems))
            print(f"\n  ⚠️  {cid}: needs your attention")
            for p in problems:
                print(f"        - {p}")
            continue

        clean.append((cid, summary))
        grand["correct"] += summary["correct"]
        grand["incorrect"] += summary["incorrect"]
        for c in VALID_CAUSES:
            grand_causes[c] += summary["failures_by_root_cause"][c]

        print(f"\n  👗 {cid}:  {summary['correct']}✓  {summary['incorrect']}✗   "
              f"acc={summary['field_accuracy']}")

        if WRITE_BACK:
            case["summary"] = {"_note": "auto-filled by tally_diagnostics.py", **summary}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(case, f, indent=2, ensure_ascii=False)

    # ── grand verdict over the CLEAN cases only ──
    total = grand["correct"] + grand["incorrect"]
    acc = round(grand["correct"] / total, 3) if total else 0.0

    print("\n" + "╔" + "═" * 52 + "╗")
    print("║" + f"  🏆 GRAND VERDICT  ({len(clean)} clean / {len(flagged)} flagged)".ljust(52) + "║")
    print("╠" + "═" * 52 + "╣")
    print("║" + f"  fields evaluated : {total}".ljust(52) + "║")
    print("║" + f"  correct          : {grand['correct']}".ljust(52) + "║")
    print("║" + f"  incorrect        : {grand['incorrect']}".ljust(52) + "║")
    print("║" + f"  FIELD ACCURACY   : {acc}".ljust(52) + "║")
    print("╠" + "═" * 52 + "╣")
    print("║" + "  failures by root cause (your synth-data brief):".ljust(52) + "║")
    for c in VALID_CAUSES:
        n = grand_causes[c]
        bar = "█" * n
        print("║" + f"    {c:16s} {n:3d} {bar}".ljust(52)[:52] + "║")
    print("╚" + "═" * 52 + "╝")

    if flagged:
        print(f"\n  ⚠️  {len(flagged)} case(s) excluded from the verdict until you fix them above.")
    print("\n  Only llm_weakness earns synthetic data. The rest get a wrench. 🔧💖\n")


if __name__ == "__main__":
    main()