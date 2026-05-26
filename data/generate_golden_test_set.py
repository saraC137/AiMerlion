"""
generate_golden_test_set.py

💎✨ FAIRY CODEMOTHER'S GOLDEN TEST SET GENERATOR ✨💎

Pulls high-quality annotated SG/MY profiles from your exported profiles JSON,
matches them with raw resume text from resume_extractions.db, and produces
a locked golden test set for evaluation.

🚨 CRITICAL: The output file (validation/golden_sg_my.jsonl) is SACRED.
   - Never train on it
   - Never use these candidate_ids as Stage 0 style exemplars
   - Tag all selected IDs in data/resume_pool_assignments.json as 'golden_eligible'

Usage:
    python generate_golden_test_set.py --input profiles_20260521_160845.json --count 25
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path (same pattern as db_diagnostic.py)
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

from db_manager import DatabaseManager


# =============================================================================
# 🎨 DISPLAY HELPERS — Sparkle while we work! 💄
# =============================================================================

def print_header(title: str, width: int = 70):
    print("\n" + "╔" + "═" * width + "╗")
    print("║" + f"  {title}  ".center(width) + "║")
    print("╚" + "═" * width + "╝")


def print_divider(width: int = 70):
    print("─" * width)


# =============================================================================
# 🔍 FILTERING & SCORING — Find the cream of the crop! 👑
# =============================================================================

def completeness_score(profile: Dict) -> int:
    """Score a profile by how complete its annotations are.
    Higher score = better golden test candidate. Max possible: 14."""
    score = 0
    if profile.get('Name'): score += 2
    if profile.get('Phone'): score += 2
    if profile.get('Email'): score += 1
    if profile.get('Current Company'): score += 2
    if profile.get('Current Title'): score += 1
    work_exp = profile.get('Work Experience', [])
    if work_exp and len(work_exp) > 0: score += 3
    edu = profile.get('Education', [])
    if edu and len(edu) > 0: score += 2
    skills = profile.get('Hard Skills', [])
    if skills and len(skills) > 0: score += 1
    return score


def detect_ethnicity_tag(name: str) -> str:
    """Heuristic ethnicity tagging for diversity stratification.
    NOT for any biased decision — only to ensure test set diversity."""
    if not name:
        return 'unknown'
    name_lower = name.lower()

    # Malay indicators (strongest signal: patronymic)
    if ' bin ' in name_lower or ' binti ' in name_lower or ' binte ' in name_lower:
        return 'malay'
    if any(name_lower.startswith(p) for p in ['mohd ', 'muhammad ', 'siti ', 'ahmad ', 'nur ']):
        return 'malay'

    # Indian indicators (s/o, d/o patronymic markers)
    if ' s/o ' in name_lower or ' d/o ' in name_lower:
        return 'indian'
    if any(suffix in name_lower for suffix in [' singh', ' kumar', ' raj', ' devi']):
        return 'indian'

    # Chinese — typical SG/MY Chinese name patterns (3-character or hyphenated)
    parts = name.replace(',', '').split()
    if len(parts) >= 2 and all(len(p) <= 6 for p in parts):
        return 'chinese'

    return 'other'


def filter_real_annotations(profiles: List[Dict], min_score: int = 11) -> List[Dict]:
    """Keep only real candidates (ID < 960000) with high completeness."""
    real_complete = []
    for p in profiles:
        pid = p.get('ID', '')
        if not pid.isdigit():
            continue
        if int(pid) >= 960000:  # exclude SkillSpan/external imports
            continue
        if completeness_score(p) < min_score:
            continue
        real_complete.append(p)
    return real_complete


# =============================================================================
# 🌈 DIVERSITY STRATIFICATION — Pick a balanced mix! 🎭
# =============================================================================

def stratified_pick(candidates: List[Dict], total_count: int = 25) -> List[Dict]:
    """Pick a diverse subset across ethnicity tags.
    Target mix:
      - 40% Chinese (10/25)
      - 28% Malay (7/25) — includes bin/binti patterns
      - 20% Indian (5/25)
      - 12% Other (3/25)
    """
    # Tag every candidate
    for c in candidates:
        c['_ethnicity_tag'] = detect_ethnicity_tag(c.get('Name', ''))

    # Group by tag
    by_tag = {'chinese': [], 'malay': [], 'indian': [], 'other': [], 'unknown': []}
    for c in candidates:
        by_tag[c['_ethnicity_tag']].append(c)

    # Print distribution
    print("\n📊 Available candidates by ethnicity tag:")
    for tag, group in by_tag.items():
        print(f"   {tag:10s}: {len(group)}")

    # Target counts
    targets = {
        'chinese': int(total_count * 0.40),  # ~10
        'malay':   int(total_count * 0.28),  # ~7
        'indian':  int(total_count * 0.20),  # ~5
        'other':   int(total_count * 0.12),  # ~3
    }

    selected = []
    random.seed(42)  # reproducibility

    for tag, target in targets.items():
        pool = by_tag.get(tag, [])
        if len(pool) >= target:
            picked = random.sample(pool, target)
        else:
            # Not enough — take all of this tag, log the gap
            picked = pool
            print(f"   ⚠️  Wanted {target} {tag}, only have {len(pool)}")
        selected.extend(picked)

    # Top up to target with whatever's left (random from unused)
    used_ids = {c['ID'] for c in selected}
    remaining = [c for c in candidates if c['ID'] not in used_ids]
    while len(selected) < total_count and remaining:
        pick = random.choice(remaining)
        selected.append(pick)
        remaining.remove(pick)

    return selected[:total_count]


# =============================================================================
# 🔗 RAW TEXT LINKING — The critical step! 🪢
# =============================================================================

def link_with_raw_text(profiles: List[Dict], db: DatabaseManager) -> List[Dict]:
    """For each profile, pull the raw_text from resume_extractions.db.
    Drops profiles where raw text is missing."""
    linked = []
    for p in profiles:
        cid = int(p['ID'])
        raw_text = db.get_raw_text_for_candidate(cid)
        if raw_text and len(raw_text.strip()) > 100:  # sanity check
            p['_raw_text'] = raw_text
            linked.append(p)
        else:
            print(f"   ⚠️  Skipping ID {cid} ({p.get('Name', '?')}): no raw_text or too short")
    return linked


# =============================================================================
# 📝 GOLDEN SET ASSEMBLY — Write the sacred file! 📜
# =============================================================================

def build_golden_record(profile: Dict, golden_id: int) -> Dict:
    """Transform an annotated profile into a golden test record."""
    return {
        "id": f"golden_{golden_id:03d}",
        "source_candidate_id": profile['ID'],
        "ethnicity_tag": profile.get('_ethnicity_tag', 'unknown'),
        "industry": profile.get('Industry', '') or 'unknown',
        "function": profile.get('Function', '') or 'unknown',
        "raw_text": profile['_raw_text'],
        "ground_truth": {
            "name": profile.get('Name', ''),
            "phone": profile.get('Phone', ''),
            "email": profile.get('Email', ''),
            "linkedin": profile.get('LinkedIn', ''),
            "current_company": profile.get('Current Company', ''),
            "current_title": profile.get('Current Title', ''),
            "current_location": profile.get('Current Location', ''),
            "language_skills": profile.get('Language Skills', ''),
            "work_experience": profile.get('Work Experience', []),
            "education": profile.get('Education', []),
            "hard_skills": profile.get('Hard Skills', []),
            "soft_skills": profile.get('Soft Skills', []),
            "summary": profile.get('Summary', ''),
        }
    }


def write_golden_jsonl(records: List[Dict], output_path: Path):
    """Write the golden set as JSONL (one record per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_pool_assignments(records: List[Dict], output_path: Path):
    """Tag all golden candidate_ids as 'golden_eligible' — prevents leakage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = {}
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            assignments = json.load(f)
    for r in records:
        assignments[r['source_candidate_id']] = 'golden_eligible'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)


# =============================================================================
# 🎬 MAIN — Run the show! 🎤
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate golden test set for AiMerlion")
    parser.add_argument('--input', required=True, help='Path to profiles JSON export')
    parser.add_argument('--count', type=int, default=25, help='Number of test cases (default 25)')
    parser.add_argument('--min-score', type=int, default=11,
                        help='Minimum completeness score 0-14 (default 11)')
    parser.add_argument('--output', default='validation/golden_sg_my.jsonl',
                        help='Output JSONL path')
    parser.add_argument('--pool-file', default='data/resume_pool_assignments.json',
                        help='Pool assignment file (leakage prevention)')
    args = parser.parse_args()

    print_header("💎 GOLDEN TEST SET GENERATOR 💎")

    # Step 1: Load profiles
    print("\n📂 Loading profiles...")
    with open(args.input, 'r', encoding='utf-8') as f:
        all_profiles = json.load(f)
    print(f"   Loaded {len(all_profiles)} total profiles")

    # Step 2: Filter to real annotations
    print(f"\n🔍 Filtering to real annotations (score >= {args.min_score})...")
    real = filter_real_annotations(all_profiles, args.min_score)
    print(f"   Found {len(real)} high-quality real annotations")

    if len(real) < args.count:
        print(f"\n⚠️  WARNING: Only {len(real)} candidates meet quality threshold,")
        print(f"   but you asked for {args.count}. Will use all available.")

    # Step 3: Stratified diversity pick
    print(f"\n🌈 Stratified diversity sampling for {args.count} candidates...")
    picked = stratified_pick(real, args.count)
    print(f"   Selected {len(picked)} candidates")

    # Step 4: Link with raw text from DB
    print(f"\n🔗 Linking with raw_text from resume_extractions.db...")
    db = DatabaseManager()
    linked = link_with_raw_text(picked, db)
    print(f"   Successfully linked {len(linked)}/{len(picked)} candidates")

    if not linked:
        print("\n❌ ERROR: No candidates could be linked with raw text. Aborting.")
        sys.exit(1)

    # Step 5: Build records
    print("\n📝 Building golden records...")
    records = [build_golden_record(p, i + 1) for i, p in enumerate(linked)]

    # Step 6: Write outputs
    output_path = Path(args.output)
    pool_path = Path(args.pool_file)
    write_golden_jsonl(records, output_path)
    write_pool_assignments(records, pool_path)

    # Step 7: Summary
    print_header("✅ GOLDEN TEST SET CREATED ✅")
    print(f"\n📄 Golden file:        {output_path}  ({len(records)} cases)")
    print(f"🔒 Pool assignments:   {pool_path}  (leakage prevention)")
    print(f"\n📊 Ethnicity distribution:")
    from collections import Counter
    eth_counts = Counter(r['ethnicity_tag'] for r in records)
    for tag, count in eth_counts.most_common():
        print(f"   {tag:10s}: {count}")
    print("\n💋 The golden file is now SACRED. Never train on it. Ever. — Fairy Codemother ✨")


if __name__ == '__main__':
    main()