"""
polish_synthetic_lite.py

💎✨ FAIRY CODEMOTHER'S POST-GENERATION POLISH SCRIPT ✨💎

Applies 3 quick fixes to synthetic_lite.jsonl AFTER generation:
  Fix #1: Fill empty Current Location based on country (SG/MY)
  Fix #2: Normalize Industry labels (collapse "Banking" → "Banking & Finance")
  Fix #3: Optionally expand company bank for FUTURE runs

Usage:
    python polish_synthetic_lite.py                          # apply all fixes in-place
    python polish_synthetic_lite.py --dry-run                # preview changes only
    python polish_synthetic_lite.py --expand-companies-bank  # also expand bank (Gemini call)

Output:
    data_gen/synthetic_lite.jsonl              ← polished in-place
    data_gen/synthetic_lite.pre_polish.bak     ← backup of original
    data_gen/polish_report.json                ← what changed and how many times
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ════════════════════════════════════════════════════════════════════════════
# 🎨 INDUSTRY LABEL NORMALIZATION MAP
# ════════════════════════════════════════════════════════════════════════════

# Canonical labels (right side) ← what we collapse INTO
# Variations (left side) ← what we collapse FROM
INDUSTRY_CANONICALIZATION = {
    # Banking & Finance family
    "Banking": "Banking & Finance",
    "Finance": "Banking & Finance",
    "Financial Services": "Banking & Finance",
    "Banking and Finance": "Banking & Finance",
    "Private Banking": "Banking & Finance",  # keep as subcategory? See below
    "Wealth Management": "Banking & Finance",
    "Investment Advisory": "Banking & Finance",
    "Investment Banking": "Banking & Finance",
    "Asset Management": "Banking & Finance",
    "Insurance": "Banking & Finance",

    # Technology family
    "Technology": "Technology",
    "Tech": "Technology",
    "Technology & Digital": "Technology",
    "IT": "Technology",
    "Information Technology": "Technology",
    "IT Infrastructure & Support": "Technology",
    "Software Development": "Technology",
    "Software Engineering": "Technology",
    "Cloud Computing & DevOps": "Technology",
    "Cybersecurity": "Technology",
    "Data Science & Analytics": "Technology",
    "Product Management": "Technology",
    "Digital Marketing & E-commerce": "Technology",

    # Healthcare family
    "Healthcare": "Healthcare",
    "Healthcare & Pharmaceuticals": "Healthcare",
    "Medical": "Healthcare",
    "Pharma": "Healthcare",
    "Pharmaceuticals": "Healthcare",

    # Education family
    "Education": "Education",
    "EdTech": "Education",
    "Higher Education": "Education",

    # Manufacturing family
    "Manufacturing": "Manufacturing",
    "Manufacturing & Engineering": "Manufacturing",
    "Engineering": "Manufacturing",
    "Industrial": "Manufacturing",

    # Government family
    "Government": "Government",
    "Government & Public Sector": "Government",
    "Public Sector": "Government",
    "Civil Service": "Government",

    # Logistics family
    "Logistics": "Logistics",
    "Supply Chain": "Logistics",
    "Transportation": "Logistics",
    "Shipping": "Logistics",

    # FMCG family
    "FMCG": "FMCG",
    "Retail": "FMCG",
    "Consumer Goods": "FMCG",
    "E-commerce": "FMCG",

    # Hospitality family
    "Hospitality": "Hospitality",
    "Tourism": "Hospitality",
    "F&B": "Hospitality",
    "Food & Beverage": "Hospitality",
}

# Subcategories — preserved alongside the main industry
KNOWN_SUBCATEGORIES = {
    "Banking & Finance": [
        "Private Banking", "Retail Banking", "Investment Banking",
        "Wealth Management", "Investment Advisory", "Asset Management",
        "Insurance", "Fintech"
    ],
    "Technology": [
        "Software Development", "Cloud & DevOps", "Data Science",
        "Cybersecurity", "Product Management", "IT Infrastructure"
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# 📍 LOCATION FILLER
# ════════════════════════════════════════════════════════════════════════════

SG_LOCATIONS = [
    "Singapore",
    "Singapore",  # weight: most common
    "Singapore",
    "Bukit Timah, Singapore",
    "Tampines, Singapore",
    "Jurong East, Singapore",
    "Woodlands, Singapore",
    "Punggol, Singapore",
    "Bedok, Singapore",
    "Sengkang, Singapore",
    "Pasir Ris, Singapore",
    "Yishun, Singapore",
    "Toa Payoh, Singapore",
    "Bishan, Singapore",
    "Ang Mo Kio, Singapore",
    "Hougang, Singapore",
    "Clementi, Singapore",
    "Queenstown, Singapore",
]

MY_LOCATIONS = [
    "Kuala Lumpur, Malaysia",
    "Petaling Jaya, Malaysia",
    "Shah Alam, Malaysia",
    "Subang Jaya, Malaysia",
    "Putrajaya, Malaysia",
    "Cyberjaya, Malaysia",
    "Penang, Malaysia",
    "George Town, Penang, Malaysia",
    "Johor Bahru, Malaysia",
    "Cheras, Malaysia",
    "Damansara, Malaysia",
    "Selangor, Malaysia",
    "Ampang, Malaysia",
    "Kajang, Malaysia",
    "Klang, Malaysia",
]


def fill_location(record: Dict, rng: random.Random) -> bool:
    """Fill Current Location if empty. Returns True if changed."""
    gt = record.get("ground_truth", {})
    current = gt.get("Current Location", "")
    if current and current.strip():
        return False

    metadata = record.get("metadata", {})
    ethnicity = metadata.get("ethnicity_tag", "")
    country = "sg" if ethnicity.endswith("_sg") else "my"

    if country == "sg":
        gt["Current Location"] = rng.choice(SG_LOCATIONS)
    else:
        gt["Current Location"] = rng.choice(MY_LOCATIONS)
    return True


# ════════════════════════════════════════════════════════════════════════════
# 🎯 INDUSTRY NORMALIZER
# ════════════════════════════════════════════════════════════════════════════

def normalize_industries(record: Dict) -> Dict:
    """Collapse Industry labels to canonical form. Returns dict of changes."""
    changes = {}
    gt = record.get("ground_truth", {})

    industry_raw = gt.get("Industry", [])
    if not industry_raw:
        return changes

    # Handle both string and list
    if isinstance(industry_raw, str):
        industry_raw = [industry_raw]

    # Process each label
    canonical_industries = []
    subcategories = []

    for label in industry_raw:
        if not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()

        # Check if it's a known subcategory we want to keep
        is_subcat = False
        for parent, subs in KNOWN_SUBCATEGORIES.items():
            if label in subs:
                if parent not in canonical_industries:
                    canonical_industries.append(parent)
                if label not in subcategories:
                    subcategories.append(label)
                is_subcat = True
                changes[label] = f"→ kept as subcat of {parent}"
                break

        if is_subcat:
            continue

        # Canonicalize
        canonical = INDUSTRY_CANONICALIZATION.get(label, label)
        if canonical != label:
            changes[label] = f"→ {canonical}"
        if canonical not in canonical_industries:
            canonical_industries.append(canonical)

    # Final form: [canonical_industry, *subcategories]
    final_list = canonical_industries + [s for s in subcategories if s not in canonical_industries]
    gt["Industry"] = final_list
    return changes


# ════════════════════════════════════════════════════════════════════════════
# 🏢 COMPANY BANK EXPANSION (optional, uses Gemini)
# ════════════════════════════════════════════════════════════════════════════

EXPAND_COMPANIES_PROMPT = """List MORE real Singapore and Malaysia companies, organized by industry.

I already have these companies — generate DIFFERENT ones (no duplicates):

EXISTING SG: {existing_sg}
EXISTING MY: {existing_my}

Output PURE JSON. No markdown, no commentary.

{{
  "sg": {{
    "finance": ["6-8 NEW SG finance companies with Pte Ltd suffix"],
    "tech": ["6-8 NEW SG tech companies"],
    "healthcare": ["..."],
    "hospitality": ["..."],
    "education": ["..."],
    "manufacturing": ["..."],
    "logistics": ["..."],
    "government": ["..."],
    "fmcg": ["..."]
  }},
  "my": {{
    "finance": ["6-8 NEW MY finance companies with Sdn Bhd / Berhad suffix"],
    "tech": ["..."],
    "healthcare": ["..."],
    "hospitality": ["..."],
    "education": ["..."],
    "manufacturing": ["..."],
    "logistics": ["..."],
    "government": ["..."],
    "fmcg": ["..."]
  }}
}}
"""


def expand_companies_bank() -> bool:
    """Optional: Use Gemini to expand SG + MY company banks. Returns True if successful."""
    sg_bank_path = Path("content_banks/sg_companies.json")
    my_bank_path = Path("content_banks/my_companies.json")

    if not sg_bank_path.exists() or not my_bank_path.exists():
        print("   ⚠️  Company banks not found — skipping expansion")
        return False

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from gemini_client import GeminiClient
    except ImportError:
        print("   ⚠️  gemini_client.py not found — skipping bank expansion")
        return False

    with open(sg_bank_path, encoding="utf-8") as f:
        sg_bank = json.load(f)
    with open(my_bank_path, encoding="utf-8") as f:
        my_bank = json.load(f)

    # Build "existing" summary for Gemini
    existing_sg = json.dumps({k: v[:3] for k, v in sg_bank.items()}, indent=2)
    existing_my = json.dumps({k: v[:3] for k, v in my_bank.items()}, indent=2)

    prompt = EXPAND_COMPANIES_PROMPT.format(existing_sg=existing_sg, existing_my=existing_my)

    try:
        gem = GeminiClient(model="flash-lite")
    except Exception as e:
        print(f"   ❌ Gemini setup failed: {e}")
        return False

    print("   🎨 Asking Gemini for additional companies...")
    response = gem.generate(prompt, temperature=0.4, max_output_tokens=4096)
    if not response:
        print("   ❌ Gemini returned nothing")
        return False

    # Parse JSON
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1:
        print("   ❌ Could not parse Gemini response")
        return False

    try:
        new_data = json.loads(response[start:end + 1])
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse failed: {e}")
        return False

    # Merge new entries into existing banks (preserving uniqueness)
    sg_new = new_data.get("sg", {})
    my_new = new_data.get("my", {})

    added_sg = 0
    for industry, new_list in sg_new.items():
        if industry not in sg_bank:
            sg_bank[industry] = []
        for c in new_list:
            if c not in sg_bank[industry]:
                sg_bank[industry].append(c)
                added_sg += 1

    added_my = 0
    for industry, new_list in my_new.items():
        if industry not in my_bank:
            my_bank[industry] = []
        for c in new_list:
            if c not in my_bank[industry]:
                my_bank[industry].append(c)
                added_my += 1

    # Save expanded banks
    sg_bank_path.write_text(json.dumps(sg_bank, indent=2, ensure_ascii=False), encoding="utf-8")
    my_bank_path.write_text(json.dumps(my_bank, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   ✅ Expanded SG bank with {added_sg} new companies")
    print(f"   ✅ Expanded MY bank with {added_my} new companies")
    return True


# ════════════════════════════════════════════════════════════════════════════
# 📂 LOAD / SAVE JSONL
# ════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> List[Dict]:
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


def save_jsonl(path: Path, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# 🎬 MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Polish synthetic_lite.jsonl post-generation")
    parser.add_argument("--input", default="data_gen/synthetic_lite.jsonl",
                        help="Path to synthetic_lite.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing")
    parser.add_argument("--expand-companies-bank", action="store_true",
                        help="Also call Gemini to expand company banks (1 API call)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for location filling")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + "  💎  POST-GENERATION POLISH  💎  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    records = load_jsonl(input_path)
    if not records:
        print("❌ No records loaded")
        return

    print(f"\n📂 Loaded {len(records)} records from {input_path}")

    # ─── Backup ──────────────────────────────────────────────────────────
    if not args.dry_run:
        backup_path = input_path.with_suffix(".pre_polish.bak")
        shutil.copy2(input_path, backup_path)
        print(f"💾 Backed up to {backup_path}")

    # ─── Fix #1: Fill empty Current Location ─────────────────────────────
    print("\n📍 FIX #1: Filling empty Current Location...")
    rng = random.Random(args.seed)
    filled_count = 0
    for rec in records:
        if fill_location(rec, rng):
            filled_count += 1
    print(f"   ✅ Filled {filled_count}/{len(records)} resumes")

    # ─── Fix #2: Normalize Industry labels ───────────────────────────────
    print("\n🎯 FIX #2: Normalizing Industry labels...")
    all_changes = Counter()
    for rec in records:
        changes = normalize_industries(rec)
        for old_label, info in changes.items():
            all_changes[f"{old_label}  {info}"] += 1

    if all_changes:
        print(f"   ✅ Normalized {sum(all_changes.values())} labels:")
        for change, count in all_changes.most_common(20):
            print(f"      {count}× {change}")
    else:
        print("   ✨ No changes needed — labels already canonical")

    # ─── Fix #3 (optional): Expand company bank ──────────────────────────
    if args.expand_companies_bank:
        print("\n🏢 FIX #3: Expanding company banks (Gemini call)...")
        expand_companies_bank()
    else:
        print("\n🏢 FIX #3: Skipped — use --expand-companies-bank to enable")

    # ─── Save ────────────────────────────────────────────────────────────
    if not args.dry_run:
        save_jsonl(input_path, records)
        print(f"\n💾 Saved polished {input_path}")
    else:
        print(f"\n🔬 DRY RUN — no changes written. Re-run without --dry-run to apply.")

    # ─── Save report ─────────────────────────────────────────────────────
    report = {
        "total_records": len(records),
        "locations_filled": filled_count,
        "industry_label_changes": dict(all_changes),
        "company_bank_expanded": args.expand_companies_bank,
        "dry_run": args.dry_run,
    }
    report_path = Path("data_gen/polish_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n📊 Polish report saved to {report_path}")

    print("\n" + "═" * 70)
    print("  ✨ POLISH COMPLETE ✨".center(70))
    print("═" * 70)
    print("\n💋 Next step: python bank_stats.py  (verify improvements!)")
    print("💋 Then:      python prepare_lite_for_training.py")


if __name__ == "__main__":
    main()