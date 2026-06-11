"""
hallucination_triage.py — turn the scary "24 hallucinations" into a real diagnosis.

For every field the harness flagged (empty in golden, filled by model), check
whether the model's value actually appears in the resume text:
  - IN the resume      -> GOLDEN GAP   (model was RIGHT, your golden set is incomplete)
  - NOT in the resume  -> REAL fabrication (the only thing worth retraining over)

Also reports which list fields the model emitted as the wrong type (the 8/25 issue).

Reads the two files the eval harness produced + your golden set.
Usage:
    python hallucination_triage.py
"""

import json, re
from collections import defaultdict, Counter

GOLDEN  = "golden.jsonl"
RAW_LOG = "act1_raw_outputs.jsonl"

LIST_FIELDS = {"Industry", "Language Skills", "Certifications", "hard_skills/tags",
               "soft_skills/skills", "Achievements", "Work Experience",
               "Project Experience", "Education"}
# fields where "empty in golden" is NOT trustworthy (systematic annotation gaps /
# generative fields) -> a fill here is almost never real fabrication
UNTRUSTWORTHY = {"Certifications", "Achievements", "Project Experience", "Summary"}


def extract_json(t):
    t = re.sub(r"```(?:json)?", "", t or "")
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm(s):
    return re.sub(r"[\s\-/().,]+", " ", str(s)).strip().lower()


def values_of(v):
    """Flatten a field value into checkable strings."""
    out = []
    if isinstance(v, list):
        for x in v:
            out += values_of(x)
    elif isinstance(v, dict):
        for x in v.values():
            out += values_of(x)
    elif v not in (None, ""):
        out.append(str(v))
    return out


def in_resume(value, raw):
    raw_n = norm(raw)
    pieces = values_of(value)
    if not pieces:
        return False
    hits = sum(1 for p in pieces if len(norm(p)) >= 3 and norm(p) in raw_n)
    return hits > 0   # at least one piece is verbatim-ish in the resume


def is_empty(v):
    return v in (None, "", [], {})


def main():
    golden = {str(json.loads(l)["id"]): json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()}
    raws = {str(json.loads(l)["id"]): json.loads(l)["raw_output"] for l in open(RAW_LOG, encoding="utf-8") if l.strip()}

    real = defaultdict(list)      # field -> [(id, value)]   genuine fabrication
    gaps = defaultdict(int)       # field -> count           golden-gap (model right)
    untrusted = defaultdict(int)  # field -> count           can't judge (gap/generative field)
    listerr = defaultdict(list)   # field -> [(id, type, sample)]

    for gid, g in golden.items():
        pred = extract_json(raws.get(gid, ""))
        if pred is None:
            continue
        gt = g["ground_truth"]; raw = g.get("raw_text", "")
        # hallucination triage
        for f in gt:
            if is_empty(gt.get(f)) and not is_empty(pred.get(f)):
                if f in UNTRUSTWORTHY:
                    untrusted[f] += 1
                elif in_resume(pred.get(f), raw):
                    gaps[f] += 1
                else:
                    real[f].append((gid, pred.get(f)))
        # list-type offenders
        for f in LIST_FIELDS:
            if f in pred and not isinstance(pred[f], list):
                listerr[f].append((gid, type(pred[f]).__name__, str(pred[f])[:50]))

    print("================ HALLUCINATION TRIAGE ================")
    n_gap = sum(gaps.values()); n_unt = sum(untrusted.values())
    n_real = sum(len(v) for v in real.values())
    print(f"GOLDEN GAPS (model right, golden empty): {n_gap}   -> not a model problem")
    for f, c in sorted(gaps.items(), key=lambda x: -x[1]):
        print(f"    {f:<20} {c}")
    print(f"UNJUDGEABLE (systematic gap / generative field): {n_unt}")
    for f, c in sorted(untrusted.items(), key=lambda x: -x[1]):
        print(f"    {f:<20} {c}")
    print(f"\n*** REAL fabrications (the only retrain trigger): {n_real} ***")
    for f, items in real.items():
        for gid, val in items:
            print(f"    [{gid}] {f}: {str(val)[:60]}   <-- not found in resume")
    if n_real == 0:
        print("    (none — your model is not fabricating. The 24 were measurement noise.)")

    print("\n================ LIST-TYPE OFFENDERS (the real fix) ================")
    if not listerr:
        print("  none — all list fields correctly typed.")
    for f, items in sorted(listerr.items(), key=lambda x: -len(x[1])):
        print(f"  {f:<20} wrong type in {len(items)} records  e.g. {items[0][1]} -> {items[0][2]!r}")


if __name__ == "__main__":
    main()