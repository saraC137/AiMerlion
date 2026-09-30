"""
Frozen eval for the CRM resume-extraction workflow.

Scores resume_crm_workflow extraction against data/golden.jsonl (24 real SG/MY
resumes). Per the plan, categories are scored separately:
  contacts | required fields | work history | education | languages |
  SG/MY terminology preservation | hallucinated sensitive fields

Both the prediction and the ground truth are pushed through the workflow's own
coerce_profile/normalize so we compare like with like, not raw key casing.

Usage:
    python eval_crm_workflow.py                 # run the model (Ollama)
    python eval_crm_workflow.py --no-model      # deterministic heuristic only
    python eval_crm_workflow.py --limit 5       # first 5 resumes (smoke test)
    python eval_crm_workflow.py --out eval.json # also write per-case JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resume_crm_workflow import (  # noqa: E402
    DEFAULT_MODEL,
    coerce_profile,
    extract_review,
    normalize_text,
)

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"

SENSITIVE_FIELDS = ["gender", "date_of_birth", "marital_status", "annual_salary"]
SGMY_TERMS = [
    "Pte Ltd", "Sdn Bhd", "Berhad", "CMFAS", "MAS", "GCE O Level", "GCE N Level",
    "GCE A Level", "ITE", "NITEC", "Higher NITEC", "Polytechnic", "bin", "binti",
    "binte", "s/o", "d/o", "Junior College",
]


def norm(s: Any) -> str:
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s@.+/-]", " ", str(s or "").lower())).strip()


def tokens(s: Any) -> set[str]:
    return {t for t in norm(s).replace(",", " ").split() if t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def name_match(pred: str, gt: str) -> bool:
    pt, gt_ = tokens(pred), tokens(gt)
    return bool(pt) and (pt == gt_ or jaccard(pt, gt_) >= 0.8)


def phone_match(pred: str, gt: str) -> bool:
    pd, gd = re.sub(r"\D", "", pred), re.sub(r"\D", "", gt)
    if not pd or not gd:
        return False
    return pd[-8:] == gd[-8:]


def loc_match(pred: str, gt: str) -> float:
    pp = re.search(r"\b(\d{6})\b", pred)
    gp = re.search(r"\b(\d{6})\b", gt)
    if pp and gp:
        return 1.0 if pp.group(1) == gp.group(1) else 0.0
    if pred and norm(pred) in norm(gt):
        return 0.7
    return 1.0 if jaccard(tokens(pred), tokens(gt)) >= 0.5 else 0.0


def loose_eq(pred: str, gt: str) -> bool:
    p, g = norm(pred), norm(gt)
    return bool(p) and bool(g) and (p == g or p in g or g in p)


def best_match(target: str, candidates: list[str], thresh: float = 0.5) -> int:
    tt = tokens(target)
    best_i, best_s = -1, thresh
    for i, c in enumerate(candidates):
        s = jaccard(tt, tokens(c))
        if norm(target) and (norm(target) in norm(c) or norm(c) in norm(target)):
            s = max(s, 0.9)
        if s >= best_s:
            best_i, best_s = i, s
    return best_i


def score_required(pred: dict, gt: dict) -> tuple[float, dict]:
    detail: dict[str, float] = {}
    for f in ["name", "industry", "function", "current_location", "phone", "email"]:
        g = gt.get(f, "")
        if not g:
            continue  # only score fields the gold actually has
        p = pred.get(f, "")
        if f == "name":
            detail[f] = 1.0 if name_match(p, g) else 0.0
        elif f == "phone":
            detail[f] = 1.0 if phone_match(p, g) else 0.0
        elif f == "email":
            detail[f] = 1.0 if norm(p) == norm(g) else 0.0
        elif f == "current_location":
            detail[f] = loc_match(p, g)
        else:  # industry / function
            detail[f] = 1.0 if loose_eq(p, g) else 0.0
    return (sum(detail.values()) / len(detail) if detail else None), detail


def score_contacts(pred: dict, gt: dict) -> float | None:
    parts = []
    if gt.get("phone"):
        parts.append(1.0 if phone_match(pred.get("phone", ""), gt["phone"]) else 0.0)
    if gt.get("email"):
        parts.append(1.0 if norm(pred.get("email", "")) == norm(gt["email"]) else 0.0)
    if gt.get("linkedin"):
        parts.append(1.0 if norm(gt["linkedin"]) in norm(pred.get("linkedin", "")) else 0.0)
    return sum(parts) / len(parts) if parts else None


def score_list(pred_items: list[dict], gt_items: list[dict], key: str, sub: str | None) -> tuple[float | None, float | None]:
    """Returns (key_recall, sub_accuracy_among_matched)."""
    gt_vals = [g.get(key, "") for g in gt_items if g.get(key)]
    if not gt_vals:
        return None, None
    pred_vals = [p.get(key, "") for p in pred_items]
    matched, sub_hits, sub_total = 0, 0, 0
    used: set[int] = set()
    for gi, g in enumerate(gt_items):
        if not g.get(key):
            continue
        cands = [v if i not in used else "\x00" for i, v in enumerate(pred_vals)]
        mi = best_match(g[key], cands)
        if mi >= 0:
            matched += 1
            used.add(mi)
            if sub and g.get(sub):
                sub_total += 1
                if loose_eq(pred_items[mi].get(sub, ""), g[sub]):
                    sub_hits += 1
    recall = matched / len(gt_vals)
    sub_acc = (sub_hits / sub_total) if sub_total else None
    return recall, sub_acc


def score_languages(pred: dict, gt: dict) -> float | None:
    gl = {norm(x.get("language", "")) for x in gt.get("languages", []) if x.get("language")}
    if not gl:
        return None
    pl = {norm(x.get("language", "")) for x in pred.get("languages", []) if x.get("language")}
    return len(gl & pl) / len(gl)


def score_terms(pred: dict, raw_text: str) -> tuple[float | None, list[str]]:
    raw = raw_text.lower()
    pred_text = json.dumps(pred, ensure_ascii=False).lower()
    expected = [t for t in SGMY_TERMS if t.lower() in raw]
    if not expected:
        return None, []
    # only terms the model actually surfaced in some field are checkable;
    # a term in raw but absent from output is "not extracted", not "mangled".
    surfaced = [t for t in expected if any(t.lower() in norm(v) for v in _flat_strings(pred))]
    if not surfaced:
        return None, []
    preserved = [t for t in surfaced if t.lower() in pred_text]
    mangled = [t for t in surfaced if t.lower() not in pred_text]
    return len(preserved) / len(surfaced), mangled


def _flat_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _flat_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _flat_strings(v)
    elif isinstance(obj, str):
        out.append(obj)
    return out


def score_hallucination(pred: dict, raw_text: str) -> tuple[float, list[str]]:
    raw = norm(raw_text)
    hallucinated = []
    for f in SENSITIVE_FIELDS:
        v = str(pred.get(f, "")).strip()
        if not v:
            continue
        if f == "annual_salary":
            digits = re.sub(r"\D", "", v)
            ok = bool(digits) and digits in re.sub(r"\D", "", raw_text)
        elif f == "gender":
            ok = norm(v) in raw
        else:  # dob / marital — require core token present in raw text
            core = [t for t in norm(v).split() if len(t) > 2]
            ok = any(t in raw for t in core) if core else False
        if not ok:
            hallucinated.append(f"{f}={v!r}")
    return 1.0 - len(hallucinated) / len(SENSITIVE_FIELDS), hallucinated


def evaluate(args: argparse.Namespace) -> dict:
    records = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]

    cats = ["contacts", "required", "work_company", "work_title", "education", "languages", "sgmy_terms", "no_hallucination"]
    sums = {c: 0.0 for c in cats}
    counts = {c: 0 for c in cats}
    cases = []

    for rec in records:
        raw = rec["raw_text"]
        gt = coerce_profile({"ground_truth": rec["ground_truth"]})
        review = extract_review(raw, model=args.model, timeout=args.timeout, no_model=args.no_model)
        pred = review["profile"]

        req, req_detail = score_required(pred, gt)
        contacts = score_contacts(pred, gt)
        wc, wt = score_list(pred["work_experience"], gt["work_experience"], "company", "title")
        edu, _ = score_list(pred["education"], gt["education"], "school", "degree")
        langs = score_languages(pred, gt)
        terms, mangled = score_terms(pred, raw)
        nohall, hallucinated = score_hallucination(pred, raw)

        row = {
            "contacts": contacts, "required": req, "work_company": wc, "work_title": wt,
            "education": edu, "languages": langs, "sgmy_terms": terms, "no_hallucination": nohall,
        }
        for c, v in row.items():
            if v is not None:
                sums[c] += v
                counts[c] += 1
        cases.append({
            "id": rec["id"], "name_gt": gt.get("name", ""), "scores": row,
            "required_detail": req_detail, "mangled_terms": mangled, "hallucinated": hallucinated,
        })

    means = {c: (sums[c] / counts[c] if counts[c] else None) for c in cats}
    scored = [v for v in means.values() if v is not None]
    overall = sum(scored) / len(scored) if scored else 0.0
    return {
        "mode": "heuristic" if args.no_model else f"model:{args.model}",
        "n": len(records),
        "category_means": means,
        "category_n": counts,
        "overall": overall,
        "cases": cases,
    }


def print_report(result: dict) -> None:
    print(f"\nCRM workflow eval  ({result['mode']}, n={result['n']})")
    print("-" * 52)
    labels = {
        "contacts": "Contacts (phone/email/linkedin)",
        "required": "Required fields",
        "work_company": "Work history — company recall",
        "work_title": "Work history — title accuracy",
        "education": "Education — school recall",
        "languages": "Languages",
        "sgmy_terms": "SG/MY term preservation",
        "no_hallucination": "No sensitive hallucination",
    }
    for key, label in labels.items():
        v = result["category_means"][key]
        n = result["category_n"][key]
        cell = f"{v:6.1%}" if v is not None else "   n/a"
        print(f"  {label:<34} {cell}  (n={n})")
    print("-" * 52)
    print(f"  {'OVERALL (mean of scored categories)':<34} {result['overall']:6.1%}")

    flagged = [c for c in result["cases"] if c["hallucinated"] or c["mangled_terms"]]
    if flagged:
        print("\nFlags:")
        for c in flagged:
            bits = []
            if c["hallucinated"]:
                bits.append("hallucinated " + ", ".join(c["hallucinated"]))
            if c["mangled_terms"]:
                bits.append("mangled " + ", ".join(c["mangled_terms"]))
            print(f"  {c['id']} ({c['name_gt']}): {'; '.join(bits)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--no-model", action="store_true", help="Deterministic heuristic only.")
    p.add_argument("--limit", type=int, default=0, help="Only the first N resumes.")
    p.add_argument("--out", type=Path, help="Write full per-case JSON here.")
    args = p.parse_args(argv or sys.argv[1:])

    result = evaluate(args)
    print_report(result)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
