"""
qc_inspect.py

💎✨ FAIRY CODEMOTHER'S SYNTHETIC RESUME QC VIEWER ✨💎

Lets you eyeball random samples from synthetic_lite.jsonl to catch quality issues
BEFORE you train on them. Shows resume_text side-by-side with the ground_truth JSON.

Usage:
    python qc_inspect.py                       # 10 random samples
    python qc_inspect.py --n 20                # 20 random samples
    python qc_inspect.py --seed 42             # reproducible random pick
    python qc_inspect.py --filter malay_sg     # only Malay SG resumes
    python qc_inspect.py --id synth_lite_00042 # one specific resume
    python qc_inspect.py --export bad_ids.txt  # interactive mode: tag bad IDs as you go

Output:
    Prints each sample with formatted panels.
    Optional: saves bad-sample IDs to a file for later cleanup.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# ════════════════════════════════════════════════════════════════════════════
# 🎨 PRETTY-PRINTING
# ════════════════════════════════════════════════════════════════════════════

def line(char: str = "─", length: int = 80) -> str:
    return char * length


def box(title: str, char: str = "═", length: int = 80) -> str:
    inner = f" {title} "
    pad = (length - len(inner)) // 2
    return "╔" + char * pad + inner + char * (length - pad - len(inner)) + "╗"


def section(label: str) -> str:
    return f"\n╭─ {label} " + "─" * (78 - len(label)) + "╮"


# ════════════════════════════════════════════════════════════════════════════
# 🔍 SAMPLE FORMATTING
# ════════════════════════════════════════════════════════════════════════════

def format_sample(record: Dict, index: int, total: int) -> str:
    """Format a single synthetic resume for human inspection."""
    rid = record.get("id", "?")
    metadata = record.get("metadata", {})
    raw_text = record.get("raw_text", "")
    gt = record.get("ground_truth", {})

    out = []
    out.append("\n" + box(f"SAMPLE {index}/{total}  —  {rid}", char="═"))
    out.append("")

    # ─── Metadata panel ──────────────────────────────────────────────────
    out.append(section("📋 METADATA"))
    for k, v in metadata.items():
        out.append(f"│ {k:20s}: {v}")
    out.append("╰" + line() + "╯")

    # ─── Ground truth (compact) ──────────────────────────────────────────
    out.append(section("✨ GROUND TRUTH (key fields)"))
    for field in ["Name", "Phone", "Email", "Current Location",
                  "Current Company", "Current Title", "Function"]:
        v = gt.get(field, "")
        marker = "✓" if v else "✗"
        out.append(f"│ {marker} {field:20s}: {v}")

    # Lists - show counts + first items
    industry = gt.get("Industry", [])
    out.append(f"│ ✓ Industry          : {industry}" if industry else f"│ ✗ Industry          : (empty)")

    we = gt.get("Work Experience", [])
    out.append(f"│ {'✓' if we else '✗'} Work Experience  : {len(we)} entries")
    for i, job in enumerate(we[:5]):
        if isinstance(job, dict):
            out.append(f"│     [{i+1}] {job.get('company', '?')} | {job.get('title', '?')} | {job.get('from', '?')}–{job.get('to', '?')}")

    edu = gt.get("Education", [])
    out.append(f"│ {'✓' if edu else '✗'} Education        : {len(edu)} entries")
    for i, e in enumerate(edu[:5]):
        if isinstance(e, dict):
            out.append(f"│     [{i+1}] {e.get('school', '?')} | {e.get('degree', '?')} {e.get('major', '?')} | {e.get('dates', '?')}")

    certs = gt.get("Certifications", [])
    out.append(f"│ {'✓' if certs else '✗'} Certifications   : {len(certs)} entries")
    for c in certs[:5]:
        out.append(f"│     - {c}")

    hard = gt.get("hard_skills/tags", [])
    soft = gt.get("soft_skills/skills", [])
    out.append(f"│ {'✓' if hard else '✗'} hard_skills/tags : {len(hard)} entries — preview: {hard[:5]}")
    out.append(f"│ {'✓' if soft else '✗'} soft_skills/skills: {len(soft)} entries — preview: {soft[:5]}")

    out.append("╰" + line() + "╯")

    # ─── Raw resume text (truncated) ─────────────────────────────────────
    out.append(section("📄 RAW RESUME TEXT (first 2000 chars)"))
    preview = raw_text[:2000]
    for raw_line in preview.split("\n"):
        out.append(f"│ {raw_line}")
    if len(raw_text) > 2000:
        out.append(f"│ ... ({len(raw_text) - 2000} more characters)")
    out.append("╰" + line() + "╯")

    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# 🔍 CONSISTENCY CHECKER (auto-flags suspicious samples)
# ════════════════════════════════════════════════════════════════════════════

def quick_sanity_check(record: Dict) -> List[str]:
    """Auto-detect obvious problems. Returns list of warnings."""
    warnings = []
    gt = record.get("ground_truth", {})
    metadata = record.get("metadata", {})
    raw_text = record.get("raw_text", "")

    # Name consistency
    name = gt.get("Name", "")
    if not name:
        warnings.append("🚨 Name is empty in ground truth")
    elif name not in raw_text:
        warnings.append(f"⚠️ Name '{name}' not found verbatim in raw_text")

    # Phone consistency
    phone = gt.get("Phone", "")
    if not phone:
        warnings.append("🚨 Phone is empty")

    # Ethnicity vs name pattern
    ethnicity = metadata.get("ethnicity_tag", "")
    if ethnicity == "malay_sg" or ethnicity == "malay_my":
        if "bin" not in name.lower() and "binti" not in name.lower() and "binte" not in name.lower():
            warnings.append(f"⚠️ Malay name without bin/binti: '{name}'")
    if ethnicity == "indian_sg":
        if "s/o" not in name.lower() and "d/o" not in name.lower() and not any(
            ind in name for ind in ["Iyer", "Nair", "Sharma", "Kumar", "Pillai", "Menon", "Selvam", "Krishnan", "Subramaniam", "Murugesan", "Rajesh", "Priya", "Shivani", "Karthik", "Lakshmi", "Arjun", "Deepa", "Anjali", "Suresh", "Meera", "Ashwin", "Divya", "Rohit", "Kavitha", "Pradeep", "Lalitha", "Manikandan", "Durkeswari"]
        ):
            warnings.append(f"⚠️ Indian SG name may not look Indian: '{name}'")

    # Phone format vs country
    if ethnicity.endswith("_sg") and not (phone.startswith("+65") or phone.startswith("8") or phone.startswith("9")):
        warnings.append(f"⚠️ SG resume with non-SG phone format: '{phone}'")
    if ethnicity.endswith("_my") and not (phone.startswith("+60") or phone.startswith("01")):
        warnings.append(f"⚠️ MY resume with non-MY phone format: '{phone}'")

    # Work experience structure
    we = gt.get("Work Experience", [])
    if not we:
        warnings.append("🚨 Empty Work Experience")
    for i, job in enumerate(we):
        if not isinstance(job, dict):
            warnings.append(f"⚠️ Work[{i}] not a dict")
            continue
        if not job.get("company"):
            warnings.append(f"⚠️ Work[{i}] has no company")
        if not job.get("title"):
            warnings.append(f"⚠️ Work[{i}] has no title")
        # SG should have Pte Ltd-ish suffix, MY should have Sdn Bhd
        if ethnicity.endswith("_sg") and job.get("company", ""):
            co = job["company"]
            if not any(s in co for s in ["Pte Ltd", "Pte. Ltd", "Pvt Ltd", "Inc", "LLC", "Group", "Bank",
                                          "MAS", "Holdings", "Ventures", "Singapore", "School", "University", "Polytechnic"]):
                warnings.append(f"⚠️ SG Work[{i}] '{co}' has no obvious SG suffix")

    # Education
    edu = gt.get("Education", [])
    if not edu:
        warnings.append("🚨 Empty Education")

    # Suspicious patterns
    if "lorem ipsum" in raw_text.lower():
        warnings.append("🚨 LOREM IPSUM detected!")
    if "[insert" in raw_text.lower() or "[your " in raw_text.lower():
        warnings.append("🚨 Placeholder text detected ([insert X])")
    if raw_text.count("@email.com") > 1:
        warnings.append("⚠️ Multiple '@email.com' generic email patterns")

    return warnings


# ════════════════════════════════════════════════════════════════════════════
# 📂 DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> List[Dict]:
    """Load all records from a JSONL file."""
    if not path.exists():
        print(f"❌ File not found: {path}")
        print(f"   Did stage0_lite_generate.py finish running?")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid JSON at line {line_num}: {e}")
    return records


# ════════════════════════════════════════════════════════════════════════════
# 🎬 MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QC inspect synthetic resumes")
    parser.add_argument("--input", default="data_gen/synthetic_lite.jsonl",
                        help="Path to synthetic_lite.jsonl")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of random samples to show (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--filter", default=None,
                        help="Filter by metadata field (e.g., 'malay_sg', 'finance', 'mid_5yr')")
    parser.add_argument("--id", default=None,
                        help="Show one specific resume by ID")
    parser.add_argument("--export", default=None,
                        help="Interactive mode: tag bad samples, save IDs to this file")
    parser.add_argument("--auto", action="store_true",
                        help="Skip the raw text panel (faster, ground truth + warnings only)")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = load_jsonl(input_path)
    if not records:
        return

    print("╔" + "═" * 78 + "╗")
    print("║" + f"  🔍  QC INSPECT  —  {len(records)} resumes in {input_path.name}  ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # ─── Single-ID mode ──────────────────────────────────────────────────
    if args.id:
        matches = [r for r in records if r.get("id") == args.id]
        if not matches:
            print(f"❌ No resume with id={args.id!r}")
            return
        samples = matches
    else:
        # ─── Filter mode ─────────────────────────────────────────────────
        pool = records
        if args.filter:
            pool = [r for r in records
                    if args.filter in json.dumps(r.get("metadata", {}))]
            print(f"\n🔎 Filter '{args.filter}': {len(pool)}/{len(records)} match")

        if not pool:
            print("❌ No samples match filter")
            return

        # ─── Random sample ───────────────────────────────────────────────
        rng = random.Random(args.seed)
        n = min(args.n, len(pool))
        samples = rng.sample(pool, n)

    # ─── Display ─────────────────────────────────────────────────────────
    bad_ids = []
    for i, rec in enumerate(samples, 1):
        # Run sanity check first
        warnings = quick_sanity_check(rec)

        if args.auto:
            # Compact mode — just ID + warnings
            print(f"\n[{i}/{len(samples)}] {rec.get('id', '?'):25s}", end="")
            if warnings:
                print(f"  ⚠️ {len(warnings)} warning(s):")
                for w in warnings:
                    print(f"   {w}")
            else:
                print(f"  ✅ clean")
        else:
            print(format_sample(rec, i, len(samples)))
            if warnings:
                print(f"\n   ⚠️  AUTO-WARNINGS:")
                for w in warnings:
                    print(f"      {w}")
            else:
                print(f"\n   ✅ No auto-warnings — looks clean")

        # ─── Interactive tagging ─────────────────────────────────────────
        if args.export:
            choice = input(f"\n   Tag this as BAD? [y/N/q to quit early]: ").strip().lower()
            if choice == "q":
                print("   ⏹  Stopping inspection.")
                break
            if choice == "y":
                bad_ids.append(rec.get("id", "?"))
                print(f"   🚩 Tagged {rec.get('id')} as bad")

    # ─── Save export file ────────────────────────────────────────────────
    if args.export and bad_ids:
        export_path = Path(args.export)
        with open(export_path, "w", encoding="utf-8") as f:
            for bid in bad_ids:
                f.write(bid + "\n")
        print(f"\n💾 Saved {len(bad_ids)} bad IDs to {export_path}")
        print(f"   To remove them, run: python clean_lite.py {export_path}")

    print("\n" + "═" * 80)
    print(f"  💋 Inspection complete. {len(samples)} samples reviewed.")
    print("═" * 80)


if __name__ == "__main__":
    main()