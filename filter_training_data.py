"""
filter_training_data.py

💅✨ FAIRY CODEMOTHER'S TRAINING DATA FILTER ✨💅

Filters your ultimate_train.jsonl to keep ONLY examples that have
complete resume extraction outputs (name, email, phone at minimum).

This removes the SkillSpan-only examples that were teaching your model
to sometimes skip name/email/phone extraction — causing confusion!

Usage:
    python filter_training_data.py
    python filter_training_data.py --input path/to/data.jsonl --output path/to/filtered.jsonl
    python filter_training_data.py --min-fields 3
"""

import json
import sys
import os
import argparse


def filter_training_data(input_path, output_path, min_required_fields=3):
    """
    Filter training examples to keep only those with sufficient field coverage.
    
    The key insight: SkillSpan examples only have 'skills' data but no name/email/phone.
    Training on these teaches the model that empty fields are acceptable output,
    which causes it to produce incomplete or garbled extractions.
    
    We keep only examples where the GPT output JSON contains at least
    min_required_fields of the core fields (name, email, phone, experience, education).
    """
    print()
    print("=" * 60)
    print("  💅✨ TRAINING DATA FILTER ✨💅")
    print("=" * 60)
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Min required core fields: {min_required_fields}")
    print()

    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        return

    # ── Core fields that a COMPLETE resume extraction should have ──
    # These are the fields that matter for AiMerlion's use case.
    # SkillSpan examples typically only have 'skills' and nothing else.
    CORE_FIELDS = ["name", "email", "phone", "experience", "education"]

    kept = []
    removed = []
    total = 0
    broken = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                broken += 1
                continue

            total += 1

            # ── Extract the GPT response ──────────────────────────
            gpt_output = None
            if "conversations" in item:
                gpt_msgs = [
                    c for c in item["conversations"]
                    if c.get("from") == "gpt"
                ]
                if gpt_msgs:
                    try:
                        gpt_output = json.loads(gpt_msgs[0].get("value", ""))
                    except (json.JSONDecodeError, ValueError):
                        pass
            elif "output" in item:
                try:
                    gpt_output = json.loads(item["output"])
                except (json.JSONDecodeError, ValueError):
                    pass

            if not gpt_output or not isinstance(gpt_output, dict):
                removed.append((i, "Could not parse GPT output"))
                continue

            # ── Count how many core fields have actual values ─────
            filled_fields = 0
            filled_names = []
            for field in CORE_FIELDS:
                value = gpt_output.get(field)
                # Check that the field exists AND has a non-empty value
                if value and value != "" and value != [] and value != {}:
                    filled_fields += 1
                    filled_names.append(field)

            if filled_fields >= min_required_fields:
                kept.append(item)
            else:
                # Log what fields were present for debugging
                reason = f"Only {filled_fields} core fields: {filled_names}"
                removed.append((i, reason))

    # ── Write filtered output ─────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── Print results ─────────────────────────────────────────────
    print(f"  📊 RESULTS:")
    print(f"  " + "-" * 56)
    print(f"  Total input examples:     {total}")
    print(f"  Kept (complete):          {len(kept)} ✅")
    print(f"  Removed (incomplete):     {len(removed)} 🗑️")
    print(f"  Broken JSON (skipped):    {broken}")
    print()

    # ── Show what was removed ─────────────────────────────────────
    if removed:
        # Categorize removals
        skills_only = sum(1 for _, r in removed if "1 core fields" in r or "0 core fields" in r)
        partial = len(removed) - skills_only

        print(f"  📋 REMOVAL BREAKDOWN:")
        print(f"  " + "-" * 56)
        print(f"  Skills-only (0-1 core fields):  {skills_only}")
        print(f"  Partial (2 core fields):        {partial}")
        print()

        print(f"  🔍 SAMPLE REMOVALS (first 5):")
        print(f"  " + "-" * 56)
        for line_num, reason in removed[:5]:
            print(f"  Line {line_num}: {reason}")
        print()

    # ── Final summary ─────────────────────────────────────────────
    output_size = os.path.getsize(output_path)
    print(f"  ✅ Filtered data saved to: {output_path}")
    print(f"  File size: {output_size / 1024:.1f} KB")
    print()

    # ── Advice ────────────────────────────────────────────────────
    if len(kept) >= 200:
        print("  🏆 EXCELLENT! 200+ complete examples — ready for training!")
    elif len(kept) >= 100:
        print("  👑 GOOD! 100+ examples — should produce decent results.")
        print("  💡 Adding more SG/MY examples would improve accuracy further.")
    elif len(kept) >= 50:
        print("  💪 OKAY! 50+ examples — enough to try, but results may be limited.")
        print("  💡 Strongly recommend annotating or generating more data.")
    else:
        print("  ⚠️  LOW! Less than 50 examples — training may not work well.")
        print("  💡 You need more annotated resumes before training.")

    print()
    print("  " + "=" * 56)
    print()

    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter training data for quality")
    parser.add_argument(
        "--input", type=str,
        default="training_data/ultimate_train.jsonl",
        help="Input JSONL file (default: training_data/ultimate_train.jsonl)"
    )
    parser.add_argument(
        "--output", type=str,
        default="training_data/filtered_train.jsonl",
        help="Output JSONL file (default: training_data/filtered_train.jsonl)"
    )
    parser.add_argument(
        "--min-fields", type=int, default=3,
        help="Minimum core fields required (default: 3 of name/email/phone/experience/education)"
    )
    args = parser.parse_args()

    filter_training_data(args.input, args.output, args.min_fields)