"""
bank_stats.py

💎✨ FAIRY CODEMOTHER'S DIVERSITY STATS ANALYZER ✨💎

Reads synthetic_lite.jsonl and reports diversity statistics so you can verify:
- Ethnicity distribution matches the matrix
- No name collisions (mode collapse check)
- Company/school variety is healthy
- CMFAS / cert coverage is solid
- Fallback rate (how often Qwen invented stuff vs using banks)

Usage:
    python bank_stats.py
    python bank_stats.py --input data_gen/synthetic_lite.jsonl
    python bank_stats.py --verbose             # show top-N items per category
"""

import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict


# ════════════════════════════════════════════════════════════════════════════
# 🎨 PRETTY PRINTING
# ════════════════════════════════════════════════════════════════════════════

def bar(value: int, total: int, width: int = 30) -> str:
    """Return a horizontal bar like ████████░░░░░░░"""
    if total == 0:
        return "░" * width
    filled = int(width * value / total)
    return "█" * filled + "░" * (width - filled)


def section_header(title: str, char: str = "─") -> str:
    return f"\n{char * 70}\n  {title}\n{char * 70}"


# ════════════════════════════════════════════════════════════════════════════
# 📊 ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def analyze(records: List[Dict], verbose: bool = False) -> None:
    total = len(records)
    if total == 0:
        print("❌ No records to analyze")
        return

    # ─── 1. METADATA DISTRIBUTION ────────────────────────────────────────
    print(section_header("📋 METADATA DISTRIBUTION"))

    eth_counter = Counter(r.get("metadata", {}).get("ethnicity_tag", "?") for r in records)
    ind_counter = Counter(r.get("metadata", {}).get("industry", "?") for r in records)
    sen_counter = Counter(r.get("metadata", {}).get("seniority", "?") for r in records)
    edu_counter = Counter(r.get("metadata", {}).get("education_path", "?") for r in records)

    print(f"\n  ETHNICITY (n={total}):")
    for eth, count in eth_counter.most_common():
        pct = 100 * count / total
        print(f"    {eth:15s} {count:3d}  {bar(count, total)} {pct:.1f}%")

    print(f"\n  INDUSTRY (n={total}):")
    for ind, count in ind_counter.most_common():
        pct = 100 * count / total
        print(f"    {ind:15s} {count:3d}  {bar(count, total)} {pct:.1f}%")

    print(f"\n  SENIORITY (n={total}):")
    for sen, count in sen_counter.most_common():
        pct = 100 * count / total
        print(f"    {sen:15s} {count:3d}  {bar(count, total)} {pct:.1f}%")

    print(f"\n  EDUCATION PATH (n={total}):")
    for edu, count in edu_counter.most_common():
        pct = 100 * count / total
        print(f"    {edu:30s} {count:3d}  {bar(count, total)} {pct:.1f}%")

    # ─── 2. NAME UNIQUENESS (mode collapse check) ────────────────────────
    print(section_header("🪞 NAME UNIQUENESS"))

    names = [r.get("ground_truth", {}).get("Name", "") for r in records]
    name_counter = Counter(names)
    unique_names = len(set(names))

    print(f"\n  Total resumes:    {total}")
    print(f"  Unique names:     {unique_names}")
    print(f"  Uniqueness rate:  {100 * unique_names / total:.1f}%")

    duplicates = [(n, c) for n, c in name_counter.most_common() if c > 1 and n]
    if duplicates:
        print(f"\n  ⚠️  Duplicate names detected:")
        for name, count in duplicates[:10]:
            print(f"    {count}× {name}")
        if len(duplicates) > 10:
            print(f"    ... and {len(duplicates) - 10} more")
    else:
        print(f"\n  ✨ All names unique — no mode collapse!")

    # Verdict
    if unique_names / total >= 0.95:
        print(f"\n  🟢 EXCELLENT name diversity")
    elif unique_names / total >= 0.85:
        print(f"\n  🟡 MODERATE diversity (some duplicates)")
    else:
        print(f"\n  🔴 POOR diversity — consider regenerating!")

    # ─── 3. PHONE UNIQUENESS ─────────────────────────────────────────────
    print(section_header("📱 PHONE UNIQUENESS"))

    phones = [r.get("ground_truth", {}).get("Phone", "") for r in records if r.get("ground_truth", {}).get("Phone")]
    unique_phones = len(set(phones))
    print(f"\n  Phones present:   {len(phones)}/{total}")
    print(f"  Unique phones:    {unique_phones}")
    print(f"  Uniqueness rate:  {100 * unique_phones / max(len(phones), 1):.1f}%")

    phone_dups = [(p, c) for p, c in Counter(phones).most_common() if c > 1]
    if phone_dups:
        print(f"\n  ⚠️  Top phone collisions:")
        for p, c in phone_dups[:5]:
            print(f"    {c}× {p}")

    # ─── 4. COMPANY VARIETY ──────────────────────────────────────────────
    print(section_header("🏢 COMPANY VARIETY"))

    all_companies = []
    for r in records:
        we = r.get("ground_truth", {}).get("Work Experience", [])
        for job in we:
            if isinstance(job, dict):
                co = job.get("company", "")
                if co:
                    all_companies.append(co)

    unique_companies = len(set(all_companies))
    print(f"\n  Total job entries:    {len(all_companies)}")
    print(f"  Unique companies:     {unique_companies}")
    if all_companies:
        print(f"  Avg uses/company:     {len(all_companies) / unique_companies:.2f}")

    if verbose and all_companies:
        print(f"\n  Top 15 companies:")
        for co, c in Counter(all_companies).most_common(15):
            print(f"    {c}× {co}")

    # Verdict
    if unique_companies >= 30:
        print(f"\n  🟢 EXCELLENT company variety ({unique_companies} unique)")
    elif unique_companies >= 15:
        print(f"\n  🟡 MODERATE variety ({unique_companies} unique)")
    else:
        print(f"\n  🔴 LOW variety — same companies repeating!")

    # ─── 5. SCHOOL VARIETY ───────────────────────────────────────────────
    print(section_header("🎓 SCHOOL VARIETY"))

    all_schools = []
    for r in records:
        edu = r.get("ground_truth", {}).get("Education", [])
        for e in edu:
            if isinstance(e, dict):
                s = e.get("school", "")
                if s:
                    all_schools.append(s)

    unique_schools = len(set(all_schools))
    print(f"\n  Total education entries: {len(all_schools)}")
    print(f"  Unique schools:          {unique_schools}")

    if verbose and all_schools:
        print(f"\n  Top 15 schools:")
        for s, c in Counter(all_schools).most_common(15):
            print(f"    {c}× {s}")

    # ─── 6. CERTIFICATION COVERAGE ───────────────────────────────────────
    print(section_header("🏅 CERTIFICATION COVERAGE"))

    all_certs = []
    for r in records:
        certs = r.get("ground_truth", {}).get("Certifications", [])
        for c in certs:
            if isinstance(c, str) and c:
                all_certs.append(c)

    unique_certs = len(set(all_certs))
    cmfas_count = sum(1 for c in all_certs if "CMFAS" in c)
    aws_count = sum(1 for c in all_certs if "AWS" in c.upper())

    print(f"\n  Total cert entries:    {len(all_certs)}")
    print(f"  Unique certs:          {unique_certs}")
    print(f"  CMFAS mentions:        {cmfas_count}")
    print(f"  AWS mentions:          {aws_count}")

    if verbose and all_certs:
        print(f"\n  Top 15 certifications:")
        for c, n in Counter(all_certs).most_common(15):
            print(f"    {n}× {c}")

    # ─── 7. INDUSTRY / FUNCTION COVERAGE ─────────────────────────────────
    print(section_header("📊 INDUSTRY / FUNCTION LABELS"))

    industries = []
    for r in records:
        ind = r.get("ground_truth", {}).get("Industry", [])
        if isinstance(ind, list):
            industries.extend(ind)
        elif isinstance(ind, str) and ind:
            industries.append(ind)

    functions = [r.get("ground_truth", {}).get("Function", "") for r in records
                 if r.get("ground_truth", {}).get("Function")]

    print(f"\n  Industry labels:")
    for i, c in Counter(industries).most_common(15):
        print(f"    {c}× {i}")

    print(f"\n  Function labels:")
    for f, c in Counter(functions).most_common(15):
        print(f"    {c}× {f}")

    # ─── 8. SKILLS / FIELD POPULATION ────────────────────────────────────
    print(section_header("📐 FIELD POPULATION"))

    fields_to_check = [
        "Name", "Phone", "Email", "Current Location",
        "Current Company", "Current Title", "Summary",
        "Function", "Industry",
        "Work Experience", "Education", "Certifications",
        "hard_skills/tags", "soft_skills/skills",
    ]

    print(f"\n  Field             populated  /  total  rate   bar")
    for field in fields_to_check:
        populated = sum(1 for r in records if r.get("ground_truth", {}).get(field))
        pct = 100 * populated / total
        print(f"  {field:20s} {populated:3d}      /  {total:3d}   {pct:5.1f}%  {bar(populated, total, 20)}")

    # ─── 9. SOURCE DISTRIBUTION ──────────────────────────────────────────
    print(section_header("🦙 GENERATION SOURCE"))

    sources = Counter(r.get("metadata", {}).get("source", "?") for r in records)
    print()
    for src, count in sources.most_common():
        pct = 100 * count / total
        print(f"    {src:30s} {count:3d}  {bar(count, total)} {pct:.1f}%")

    # ─── 10. SUMMARY VERDICT ─────────────────────────────────────────────
    print(section_header("🎯 OVERALL VERDICT", char="═"))

    issues = []
    if unique_names / total < 0.95:
        issues.append(f"Name uniqueness: {100 * unique_names / total:.1f}% (target: ≥95%)")
    if unique_companies < 20:
        issues.append(f"Company variety: only {unique_companies} unique (target: ≥30)")
    if cmfas_count < 10:
        issues.append(f"CMFAS coverage: only {cmfas_count} mentions (target: ≥30 — important for SG finance)")
    if not industries:
        issues.append("No Industry labels populated")
    if not functions:
        issues.append("No Function labels populated")

    if not issues:
        print("\n  ✨ ALL CHECKS PASSED — your data looks ready for fine-tuning! 💎")
        print("\n  Next step: python prepare_lite_for_training.py")
    else:
        print(f"\n  ⚠️  {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"    • {issue}")
        print(f"\n  Suggested: review samples with `python qc_inspect.py`")
        print(f"  Or regenerate the affected cells in stage0_lite_generate.py")


# ════════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"❌ File not found: {path}")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ════════════════════════════════════════════════════════════════════════════
# 🎬 MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Diversity stats for synthetic resumes")
    parser.add_argument("--input", default="data_gen/synthetic_lite.jsonl",
                        help="Path to synthetic_lite.jsonl")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show top-N items per category")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = load_jsonl(input_path)
    if not records:
        return

    print("╔" + "═" * 68 + "╗")
    print("║" + f"  💎  DIVERSITY STATS — {len(records)} synthetic resumes  💎  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    analyze(records, verbose=args.verbose)


if __name__ == "__main__":
    main()