"""
standardize_training_data.py

💅✨ FAIRY CODEMOTHER'S DATA STANDARDIZER ✨💅

THE PROBLEM: Your training data has 32 different JSON output schemas!
The model gets confused trying to learn all of them simultaneously.

THE FIX: Standardize every example to use ONE consistent schema.

Usage:
    python standardize_training_data.py
    python standardize_training_data.py --sgmy-only
    python standardize_training_data.py --input training_data/stage1_full_resume.jsonl
"""

import json
import os
import sys
import argparse

# ── THE standard schema — every output will have these keys ───────
# Missing values become null/empty arrays rather than missing keys.
# This teaches the model: "ALWAYS return this exact structure."
STANDARD_SCHEMA = {
    "name": None,
    "email": None,
    "phone": None,
    "date_of_birth": None,
    "location": None,
    "summary": None,
    "hard_skills": [],
    "soft_skills": [],
    "experience": [],
    "education": [],
    "certifications": [],
    "languages": [],
    "function": None,
    "industry": None,
}

# ── Field name mappings (normalize variations) ────────────────────
# Some examples use different key names for the same concept.
FIELD_ALIASES = {
    "candidate_name": "name",
    "full_name": "name",
    "skills": "hard_skills",
    "technical_skills": "hard_skills",
    "tools": "hard_skills",
    "technologies": "hard_skills",
    "work_experience": "experience",
    "employment_history": "experience",
    "career_history": "experience",
    "qualifications": "education",
    "academic": "education",
    "certs": "certifications",
    "certificates": "certifications",
    "professional_certifications": "certifications",
    "lang": "languages",
    "job_function": "function",
    "job_category": "function",
    "sector": "industry",
    "contact": "phone",
    "mobile": "phone",
    "telephone": "phone",
    "dob": "date_of_birth",
    "birth_date": "date_of_birth",
    "address": "location",
}


def standardize(input_path, output_path, sgmy_only=False):
    print()
    print("=" * 60)
    print("  💅✨ TRAINING DATA STANDARDIZER ✨💅")
    print("=" * 60)
    print(f"  Input:     {input_path}")
    print(f"  Output:    {output_path}")
    print(f"  SG/MY only: {sgmy_only}")
    print()

    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        items = [json.loads(l.strip()) for l in f if l.strip()]

    print(f"  Total input examples: {len(items)}")

    standardized = []
    skipped_sgmy = 0
    schema_fixes = 0

    for item in items:
        convos = item.get("conversations", [])
        gpt_msgs = [c for c in convos if c.get("from") == "gpt"]
        human_msgs = [c for c in convos if c.get("from") == "human"]

        if not gpt_msgs:
            continue

        # Parse the GPT output
        try:
            original = json.loads(gpt_msgs[0]["value"])
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(original, dict):
            continue

        # ── Optional: Filter SG/MY only ───────────────────────────
        if sgmy_only:
            phone = str(original.get("phone", ""))
            location = str(original.get("location", ""))
            name = str(original.get("name", ""))
            all_text = phone + location + name

            is_sgmy = any(marker in all_text for marker in [
                "+65", "+60", "Singapore", "Malaysia", "Selangor",
                "Kuala Lumpur", "Johor", "Penang", "Sabah",
                "Sarawak",
            ])
            # Also check 8-digit SG phone without country code
            if not is_sgmy and phone.replace(" ", "").replace("-", "").isdigit():
                digits = phone.replace(" ", "").replace("-", "")
                if len(digits) == 8 and digits[0] in "689":
                    is_sgmy = True

            # Check the resume text for SG/MY markers
            if not is_sgmy and human_msgs:
                resume_text = human_msgs[0]["value"].lower()
                is_sgmy = any(m in resume_text for m in [
                    "singapore", "malaysia", "pte ltd", "sdn bhd",
                    "berhad", "nric", "polytechnic", "ite college",
                    "bin ", "binti ", " d/o ",
                ])

            if not is_sgmy:
                skipped_sgmy += 1
                continue

        # ── Standardize the output to the canonical schema ────────
        new_output = {}
        for standard_key, default_val in STANDARD_SCHEMA.items():
            # Check if the key exists directly
            if standard_key in original:
                new_output[standard_key] = original[standard_key]
            else:
                # Check aliases
                found = False
                for alias, target in FIELD_ALIASES.items():
                    if target == standard_key and alias in original:
                        new_output[standard_key] = original[alias]
                        found = True
                        schema_fixes += 1
                        break
                if not found:
                    # Use default (null or empty array)
                    if isinstance(default_val, list):
                        new_output[standard_key] = []
                    else:
                        new_output[standard_key] = None

        # ── Ensure arrays are actually arrays ─────────────────────
        for key in ["hard_skills", "soft_skills", "experience",
                     "education", "certifications", "languages"]:
            val = new_output.get(key)
            if val is None:
                new_output[key] = []
            elif isinstance(val, str):
                # Convert comma-separated string to array
                new_output[key] = [s.strip() for s in val.split(",") if s.strip()]

        # ── Build the standardized example ────────────────────────
        new_gpt_value = json.dumps(new_output, ensure_ascii=False)

        # Reconstruct conversations with standardized output
        new_convos = []
        for msg in convos:
            if msg.get("from") == "gpt":
                new_convos.append({"from": "gpt", "value": new_gpt_value})
            else:
                new_convos.append(msg)

        standardized.append({"conversations": new_convos})

    # ── Write output ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in standardized:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── Verify: count unique schemas in output ────────────────────
    schema_count = {}
    for item in standardized:
        gpt = [c for c in item["conversations"] if c.get("from") == "gpt"]
        if gpt:
            parsed = json.loads(gpt[0]["value"])
            keys = tuple(sorted(parsed.keys()))
            schema_count[keys] = schema_count.get(keys, 0) + 1

    print(f"  📊 RESULTS:")
    print(f"  " + "-" * 56)
    print(f"  Output examples:        {len(standardized)}")
    if sgmy_only:
        print(f"  Skipped (non-SG/MY):    {skipped_sgmy}")
    print(f"  Schema fixes applied:   {schema_fixes}")
    print(f"  Unique output schemas:  {len(schema_count)} (should be 1!)")
    print()

    if len(schema_count) == 1:
        print(f"  ✅ PERFECT! All examples now use ONE consistent schema!")
    else:
        print(f"  ⚠️  Still {len(schema_count)} schemas — investigate further")
        for keys, count in sorted(schema_count.items(), key=lambda x: -x[1]):
            print(f"     {count:>4d}x: {list(keys)}")

    print()
    print(f"  ✅ Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print()

    # ── Training recommendation ───────────────────────────────────
    print("  " + "=" * 56)
    print("  💅 NEXT STEPS:")
    print("  " + "=" * 56)
    print()
    print("  1. Verify:  python inspect_training_data.py " + output_path)
    print()
    print("  2. Train:")
    print("     set XFORMERS_DISABLED=1")
    print("     set PYTORCH_ALLOC_CONF=expandable_segments:True")
    print()
    print(f"     python ml/finetune_unsloth.py ^")
    print(f"         --train-file {output_path} ^")
    print(f"         --epochs 3 ^")
    print(f"         --batch-size 1 ^")
    print(f"         --seq-length 2048 ^")
    print(f"         --output-dir resume_model_v5 ^")
    print(f"         --gguf q8_0 ^")
    print(f"         --model-name TG-v5")
    print()
    print("  NOTE: Using seq-length 2048 this time!")
    print("  Your VRAM can handle it with batch_size=1.")
    print("  This ensures long resumes don't get truncated.")
    print()
    print("  " + "=" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_data/stage1_full_resume.jsonl")
    parser.add_argument("--output", default="training_data/standardized_train.jsonl")
    parser.add_argument("--sgmy-only", action="store_true",
                        help="Keep only SG/MY examples")
    args = parser.parse_args()

    standardize(args.input, args.output, args.sgmy_only)