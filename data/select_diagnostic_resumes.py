"""
select_diagnostic_resumes.py

🩺✨ FAIRY CODEMOTHER'S DIAGNOSTIC RESUME SELECTOR ✨🩺

Pulls a diverse set of SG/MY resumes from your raw_extractions table for
Act 0 (the diagnostic gate). RESPECTS the leakage manifest — will never
pick anything tagged as 'golden_eligible'.

After selection, tags chosen IDs as 'diagnostic_eligible' so they don't
accidentally end up in training data later.

Output:
    diagnostic/case_01_raw.txt ... case_10_raw.txt   (raw resume text)
    diagnostic/case_01_template.json ... etc          (skeleton for your hand-judgment)
    diagnostic/SUMMARY.md                             (overview + instructions)

Usage:
    python select_diagnostic_resumes.py --count 10
"""

import os
import sys
import json
import random
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

from db_manager import DatabaseManager


# =============================================================================
# 🎨 DISPLAY HELPERS
# =============================================================================

def print_header(title: str, width: int = 70):
    print("\n" + "╔" + "═" * width + "╗")
    print("║" + f"  {title}  ".center(width) + "║")
    print("╚" + "═" * width + "╝")


# =============================================================================
# 🌈 ETHNICITY DETECTION (heuristic — for diversity only)
# =============================================================================

def detect_ethnicity_tag(name: str) -> str:
    """Heuristic ethnicity tagging for diversity stratification."""
    if not name:
        return 'unknown'
    name_lower = name.lower()

    if ' bin ' in name_lower or ' binti ' in name_lower or ' binte ' in name_lower or ' bte ' in name_lower:
        return 'malay'
    if any(name_lower.startswith(p) for p in ['mohd ', 'muhammad ', 'siti ', 'ahmad ', 'nur ', 'mohamed ']):
        return 'malay'

    if ' s/o ' in name_lower or ' d/o ' in name_lower:
        return 'indian'
    if any(suffix in name_lower for suffix in [' singh', ' kumar', ' raj', ' devi', ' selvam']):
        return 'indian'

    parts = name.replace(',', '').split()
    if len(parts) >= 2 and all(len(p) <= 6 for p in parts):
        return 'chinese'

    return 'other'


def looks_like_sg_my_resume(raw_text: str) -> bool:
    """Cheap heuristic — does this text look like an SG/MY resume?
    Signals: Pte Ltd, Sdn Bhd, Singapore, Malaysia, +65/+60 phones,
             polytechnic names, CMFAS, etc.
    """
    if not raw_text or len(raw_text) < 500:
        return False

    text_lower = raw_text.lower()
    sg_my_signals = [
        'singapore', 'malaysia', 'pte ltd', 'pte. ltd', 'sdn bhd', 'sdn. bhd',
        '+65', '+60', 'polytechnic', 'cmfas', 'nus', 'ntu', 'smu',
        'ngee ann', 'temasek poly', 'republic poly', 'nanyang poly',
        'kuala lumpur', 'selangor', 'penang', 'johor', 'utm', 'um',
    ]
    return any(signal in text_lower for signal in sg_my_signals)


# =============================================================================
# 🔍 CANDIDATE FETCHING
# =============================================================================

def load_pool_assignments(path: Path) -> Dict[str, str]:
    """Load the existing pool assignments manifest (leakage prevention)."""
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def fetch_candidate_pool(db: DatabaseManager, excluded_ids: set,
                         pool_size: int = 200) -> List[Dict]:
    """Fetch a large pool of candidates we can pick diagnostic cases from.
    Excludes anything already tagged in pool_assignments."""
    conn = db._connection

    # Pull from raw_extractions joined with structured_extractions
    # to get both raw text AND any name we have
    query = """
        SELECT r.candidate_id, r.raw_text, s.name
        FROM raw_extractions r
        LEFT JOIN structured_extractions s ON r.candidate_id = s.candidate_id
        WHERE r.candidate_id < 960000
          AND r.raw_text IS NOT NULL
          AND LENGTH(r.raw_text) > 500
        ORDER BY r.candidate_id
    """

    cursor = conn.execute(query)
    candidates = []
    for row in cursor.fetchall():
        cid = row['candidate_id'] if isinstance(row, sqlite3.Row) else row[0]
        raw_text = row['raw_text'] if isinstance(row, sqlite3.Row) else row[1]
        name = row['name'] if isinstance(row, sqlite3.Row) else row[2]

        if str(cid) in excluded_ids:
            continue  # already used in golden or elsewhere

        if not looks_like_sg_my_resume(raw_text):
            continue

        candidates.append({
            'candidate_id': cid,
            'name': name or 'Unknown',
            'raw_text': raw_text,
            'ethnicity_tag': detect_ethnicity_tag(name or ''),
            'raw_text_length': len(raw_text),
        })

    return candidates


# =============================================================================
# 🎭 STRATIFIED PICK FOR DIAGNOSTIC
# =============================================================================

def stratified_diagnostic_pick(candidates: List[Dict], count: int = 10) -> List[Dict]:
    """Pick diagnostic cases with diversity.
    Target: 4 Chinese, 3 Malay, 2 Indian, 1 Other (for 10 cases).
    Diagnostic prioritizes MALAY because bin/binti is your hardest pattern.
    """
    by_tag = {'chinese': [], 'malay': [], 'indian': [], 'other': [], 'unknown': []}
    for c in candidates:
        by_tag[c['ethnicity_tag']].append(c)

    print("\n📊 Available diagnostic candidates by ethnicity:")
    for tag, group in by_tag.items():
        print(f"   {tag:10s}: {len(group)}")

    # Target counts for diagnostic (slightly Malay-heavy on purpose)
    if count == 10:
        targets = {'chinese': 4, 'malay': 3, 'indian': 2, 'other': 1}
    else:
        # Scale proportionally for other counts
        targets = {
            'chinese': max(1, int(count * 0.40)),
            'malay':   max(1, int(count * 0.30)),
            'indian':  max(1, int(count * 0.20)),
            'other':   max(1, int(count * 0.10)),
        }

    selected = []
    random.seed(99)  # different seed than golden set

    for tag, target in targets.items():
        pool = by_tag.get(tag, [])
        if len(pool) >= target:
            picked = random.sample(pool, target)
        else:
            picked = pool
            print(f"   ⚠️  Wanted {target} {tag}, only have {len(pool)}")
        selected.extend(picked)

    # Top up if needed
    used_ids = {c['candidate_id'] for c in selected}
    remaining = [c for c in candidates if c['candidate_id'] not in used_ids]
    while len(selected) < count and remaining:
        pick = random.choice(remaining)
        selected.append(pick)
        remaining.remove(pick)

    return selected[:count]


# =============================================================================
# 📝 OUTPUT WRITING
# =============================================================================

def write_diagnostic_files(candidates: List[Dict], output_dir: Path):
    """Write one raw text file + one template JSON per candidate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(candidates, start=1):
        case_id = f"case_{i:02d}"

        # Raw text file (for your eyes — what you'll hand-judge against)
        raw_path = output_dir / f"{case_id}_raw.txt"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(f"# Case {i:02d} | Candidate ID: {c['candidate_id']}\n")
            f.write(f"# Annotated name (if any): {c['name']}\n")
            f.write(f"# Ethnicity tag: {c['ethnicity_tag']}\n")
            f.write(f"# Raw text length: {c['raw_text_length']} chars\n")
            f.write(f"# {'='*60}\n\n")
            f.write(c['raw_text'])

        # Template JSON (skeleton you'll fill in after running ai_extractor)
        template = {
            "case_id": case_id,
            "candidate_id": c['candidate_id'],
            "metadata": {
                "ethnicity_tag": c['ethnicity_tag'],
                "name_hint": c['name'],
                "raw_text_length": c['raw_text_length'],
            },
            "input_text": "<see {case_id}_raw.txt — too long to inline>".format(case_id=case_id),
            "model_output": {
                "_instructions": "PASTE the JSON output from your ai_extractor.py here",
            },
            "ground_truth": {
                "_instructions": "Fill in what SHOULD have been extracted by reading the raw_text",
                "name": "",
                "phone": "",
                "email": "",
                "current_company": "",
                "current_title": "",
                "work_experience": [],
                "education": [],
                "certifications": [],
                "hard_skills": [],
            },
            "field_evaluation": {
                "_instructions": "For each field, mark correct=true/false, then categorize the root_cause if wrong. root_cause MUST be one of: llm_weakness, post_processing, schema_issue, prompt_issue, ocr_issue",
                "name": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "phone": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "email": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "current_company": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "current_title": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "work_experience": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "education": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "certifications": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
                "hard_skills": {"correct": None, "failure_type": "", "root_cause": "", "notes": ""},
            },
            "summary": {
                "_instructions": "Auto-fill after field_evaluation is complete (or use a helper script)",
                "total_fields_evaluated": 0,
                "correct": 0,
                "incorrect": 0,
                "failures_by_root_cause": {
                    "llm_weakness": 0,
                    "post_processing": 0,
                    "schema_issue": 0,
                    "prompt_issue": 0,
                    "ocr_issue": 0,
                },
            }
        }

        template_path = output_dir / f"{case_id}_template.json"
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)


def write_summary_md(candidates: List[Dict], output_dir: Path):
    """Write a SUMMARY.md with instructions for completing Act 0."""
    md = ["# 🩺 Act 0 — Diagnostic Gate\n",
          "## Selected cases\n"]
    for i, c in enumerate(candidates, start=1):
        md.append(f"- **case_{i:02d}**: ID `{c['candidate_id']}` | "
                  f"{c['ethnicity_tag']} | {c['name'][:40]}")

    md.append("""

## 📋 How to complete Act 0

For each case:

1. **Run your current pipeline** on the raw text:
   ```python
   # Pseudocode — adjust to your ai_extractor entry point
   from ai_extractor import extract_resume
   raw = open('diagnostic/case_01_raw.txt').read()
   # strip the header comments (lines starting with #)
   raw_clean = '\\n'.join(l for l in raw.split('\\n') if not l.startswith('#'))
   output = extract_resume(raw_clean)
   ```

2. **Paste the model output** into `case_XX_template.json` → `model_output` section

3. **Read the raw text yourself** and fill in `ground_truth` with what
   SHOULD have been extracted

4. **For each field, fill in `field_evaluation`**:
   - `correct`: true / false
   - `failure_type`: hallucination / truncation / missing / wrong_format / wrong_assignment
   - `root_cause`: ONE of:
     - `llm_weakness` — model didn't understand the pattern
     - `post_processing` — extracted right, code mangled it
     - `schema_issue` — JSON structure rejected valid output
     - `prompt_issue` — system prompt didn't ask for it correctly
     - `ocr_issue` — input text was already corrupted

5. **Decision gate** (after all 10 cases):
   - Count total failures by root_cause
   - If >60% are `llm_weakness` → ✅ Curriculum training is the right move
   - If >40% are `post_processing` or `prompt_issue` → 🛑 Fix code first, not model
   - If >30% are `schema_issue` → 🛑 Fix schema first
   - If >20% are `ocr_issue` → 🛑 Fix input pipeline first

## 🎯 Time estimate
- Running the model: 5 min total
- Filling templates: 20-30 min for all 10 cases (2-3 min each)
- Tallying decision: 5 min

**Total: ~30-40 minutes of focused work.**

💋 — Fairy Codemother
""")

    with open(output_dir / 'SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))


# =============================================================================
# 🔒 POOL ASSIGNMENT UPDATE
# =============================================================================

def update_pool_assignments(candidates: List[Dict], pool_path: Path):
    """Tag selected candidate_ids as 'diagnostic_eligible' to prevent
    accidental reuse in training data."""
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = {}
    if pool_path.exists():
        with open(pool_path, 'r', encoding='utf-8') as f:
            assignments = json.load(f)
    for c in candidates:
        assignments[str(c['candidate_id'])] = 'diagnostic_eligible'
    with open(pool_path, 'w', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)


# =============================================================================
# 🎬 MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Select diagnostic resumes for Act 0")
    parser.add_argument('--count', type=int, default=10,
                        help='Number of diagnostic cases (default 10)')
    parser.add_argument('--output-dir', default='diagnostic',
                        help='Output directory (default ./diagnostic)')
    parser.add_argument('--pool-file', default='data/resume_pool_assignments.json',
                        help='Pool assignment manifest path')
    args = parser.parse_args()

    print_header("🩺 ACT 0 — DIAGNOSTIC RESUME SELECTOR 🩺")

    # Step 1: Load existing pool assignments (leakage prevention)
    pool_path = Path(args.pool_file)
    pool_assignments = load_pool_assignments(pool_path)
    excluded_ids = set(pool_assignments.keys())
    print(f"\n🔒 Loaded {len(excluded_ids)} excluded IDs from pool manifest")
    print(f"   (these will NEVER be picked — protects golden test set)")

    # Step 2: Fetch eligible candidates
    print("\n🔍 Fetching eligible SG/MY candidates from raw_extractions...")
    db = DatabaseManager()
    candidates = fetch_candidate_pool(db, excluded_ids)
    print(f"   Found {len(candidates)} eligible SG/MY-looking candidates")

    if len(candidates) < args.count:
        print(f"\n⚠️  WARNING: Only {len(candidates)} eligible, need {args.count}")
        print("   Will use all available.")

    # Step 3: Stratified pick
    selected = stratified_diagnostic_pick(candidates, args.count)
    print(f"\n✅ Selected {len(selected)} diagnostic cases")

    # Step 4: Write files
    output_dir = Path(args.output_dir)
    print(f"\n📝 Writing diagnostic files to {output_dir}/ ...")
    write_diagnostic_files(selected, output_dir)
    write_summary_md(selected, output_dir)
    print(f"   ✅ {len(selected)} raw text files + {len(selected)} JSON templates")
    print(f"   ✅ SUMMARY.md with instructions")

    # Step 5: Update pool manifest (leakage prevention)
    update_pool_assignments(selected, pool_path)
    print(f"\n🔒 Tagged {len(selected)} IDs as 'diagnostic_eligible' in pool manifest")

    # Step 6: Final summary
    print_header("✅ DIAGNOSTIC SET READY ✅")
    print(f"\n📁 Next steps:")
    print(f"   1. Open {output_dir}/SUMMARY.md")
    print(f"   2. Run ai_extractor on each case_XX_raw.txt")
    print(f"   3. Fill in case_XX_template.json for each one")
    print(f"   4. Tally root_cause counts → decision gate")
    print(f"\n💋 — Fairy Codemother")


if __name__ == '__main__':
    main()