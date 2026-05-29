"""
split_training_stages.py

💅✨ FAIRY CODEMOTHER'S TWO-STAGE DATA SPLITTER ✨💅

Splits your ultimate_train.jsonl into two files:
  Stage 1: Complete resume extractions (name + email + phone + more)
  Stage 2: Skills-focused extractions (skills but missing core fields)

This supports the sequential fine-tuning strategy:
  1. Train on Stage 1 data first (teaches full extraction)
  2. Then fine-tune on Stage 2 data (teaches skill specialization)

Usage:
    python split_training_stages.py
    python split_training_stages.py --input path/to/data.jsonl
"""

import json
import os
import sys
import argparse


def split_stages(input_path, output_dir="training_data"):
    print()
    print("=" * 60)
    print("  💅✨ TWO-STAGE DATA SPLITTER ✨💅")
    print("=" * 60)
    print(f"  Input: {input_path}")
    print()

    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        return

    CORE_FIELDS = ["name", "email", "phone", "experience", "education"]

    stage1 = []  # Complete resume extractions
    stage2 = []  # Skills-focused / partial extractions
    broken = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                broken += 1
                continue

            # ── Parse the GPT output ──────────────────────────────
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

            if not gpt_output or not isinstance(gpt_output, dict):
                broken += 1
                continue

            # ── Count core fields ─────────────────────────────────
            filled = 0
            for field in CORE_FIELDS:
                value = gpt_output.get(field)
                if value and value != "" and value != [] and value != {}:
                    filled += 1

            if filled >= 3:
                stage1.append(item)
            else:
                # ── Enhance Stage 2 data for skills training ──────
                # The skills-only examples need a DIFFERENT instruction
                # that tells the model to focus on skills extraction.
                # This prevents the model from learning "don't extract names".
                #
                # We modify the human message to explicitly say
                # "focus on skills extraction" so the model understands
                # WHY there's no name/email in the output.
                enhanced = _enhance_skills_example(item)
                stage2.append(enhanced)

    # ── Write output files ────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    stage1_path = os.path.join(output_dir, "stage1_full_resume.jsonl")
    stage2_path = os.path.join(output_dir, "stage2_skills_focus.jsonl")

    with open(stage1_path, "w", encoding="utf-8") as f:
        for item in stage1:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(stage2_path, "w", encoding="utf-8") as f:
        for item in stage2:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── Print results ─────────────────────────────────────────────
    print(f"  📊 SPLIT RESULTS:")
    print(f"  " + "-" * 56)
    print(f"  Stage 1 (full resume):    {len(stage1)} examples")
    print(f"  Stage 2 (skills focus):   {len(stage2)} examples")
    print(f"  Broken (skipped):         {broken}")
    print()
    print(f"  📁 Stage 1 saved: {stage1_path}")
    print(f"  📁 Stage 2 saved: {stage2_path}")
    print()

    # ── Training commands ─────────────────────────────────────────
    print("  🎯 TRAINING COMMANDS:")
    print("  " + "-" * 56)
    print()
    print("  STAGE 1 — Full resume extraction (run first!):")
    print("  " + "~" * 56)
    print("  set XFORMERS_DISABLED=1")
    print("  set PYTORCH_ALLOC_CONF=expandable_segments:True")
    print()
    print(f"  python ml/finetune_unsloth.py ^")
    print(f"      --train-file {stage1_path} ^")
    print(f"      --epochs 5 ^")
    print(f"      --batch-size 1 ^")
    print(f"      --seq-length 1024 ^")
    print(f"      --output-dir resume_model_stage1 ^")
    print(f"      --gguf q8_0 ^")
    print(f"      --model-name TG-Stage1")
    print()
    print("  STAGE 2 — Skills specialization (run after Stage 1!):")
    print("  " + "~" * 56)
    print("  set XFORMERS_DISABLED=1")
    print("  set PYTORCH_ALLOC_CONF=expandable_segments:True")
    print()
    print(f"  python ml/finetune_unsloth.py ^")
    print(f"      --train-file {stage2_path} ^")
    print(f"      --epochs 3 ^")
    print(f"      --batch-size 1 ^")
    print(f"      --seq-length 1024 ^")
    print(f"      --output-dir resume_model_stage2 ^")
    print(f"      --gguf q8_0 ^")
    print(f"      --model-name TG-Final ^")
    print(f"      --resume-from resume_model_stage1/lora_model")
    print()
    print("  ⚠️  NOTE: The --resume-from flag in Stage 2 tells the script")
    print("  to start from Stage 1's trained weights instead of the base model.")
    print("  This is the KEY to sequential fine-tuning!")
    print()

    # ── Advice ────────────────────────────────────────────────────
    print("  " + "=" * 56)
    print("  💅 FAIRY CODEMOTHER'S ADVICE:")
    print("  " + "=" * 56)
    print()
    if len(stage1) >= 200:
        print("  🏆 Stage 1 has 200+ examples — EXCELLENT foundation!")
    elif len(stage1) >= 100:
        print("  👑 Stage 1 has 100+ examples — good enough to start!")
    else:
        print(f"  ⚠️  Stage 1 has only {len(stage1)} examples.")
        print("  Consider annotating more complete resumes first!")
    print()
    print("  📌 IMPORTANT: Your finetune_unsloth.py currently does NOT")
    print("  support --resume-from. You'll need to add this feature")
    print("  or manually edit the script to load Stage 1 LoRA weights")
    print("  instead of the base model for Stage 2.")
    print()
    print("  " + "=" * 56)
    print()

    return stage1, stage2


def _enhance_skills_example(item):
    """
    Enhance a skills-only example with a modified instruction
    that explicitly tells the model this is a skills-focused task.

    Without this, the model sees "extract resume data" but the output
    has no name/email/phone — and learns that's acceptable.
    With this, the model learns "when asked about skills specifically,
    focus on skills."

    This is crucial for preventing the model from learning
    contradictory patterns!
    """
    if "conversations" not in item:
        return item

    # Deep copy to avoid modifying original
    enhanced = json.loads(json.dumps(item))

    for msg in enhanced["conversations"]:
        if msg.get("from") == "human":
            original = msg["value"]
            # Prepend a skills-focus instruction
            # This makes the training signal consistent:
            # "Full extraction prompt" → full JSON output
            # "Skills extraction prompt" → skills-focused output
            msg["value"] = original.replace(
                "Extract structured data from this resume.",
                "Extract skills and competencies from this resume text. "
                "Focus on identifying hard_skills, soft_skills, certifications, "
                "and technical competencies."
            )
            break

    return enhanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split training data into two stages"
    )
    parser.add_argument(
        "--input", type=str,
        default="training_data/ultimate_train.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="training_data",
        help="Output directory for stage files"
    )
    args = parser.parse_args()

    split_stages(args.input, args.output_dir)
    