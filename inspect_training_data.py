"""
inspect_training_data.py

💅✨ FAIRY CODEMOTHER'S TRAINING DATA INSPECTOR ✨💅

Quick diagnostic script to check if your training JSONL is healthy.
Run this BEFORE training to catch problems early!

Usage:
    python inspect_training_data.py
    python inspect_training_data.py --file path/to/your_data.jsonl
"""

import json
import sys
import os


def inspect(filepath):
    print()
    print("=" * 60)
    print("  💅✨ TRAINING DATA INSPECTOR ✨💅")
    print("=" * 60)
    print(f"  File: {filepath}")
    print()

    # ── Check file exists ─────────────────────────────────────────
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        print("  Make sure the path is correct!")
        return

    file_size = os.path.getsize(filepath)
    print(f"  File size: {file_size / 1024:.1f} KB ({file_size:,} bytes)")
    print()

    # ── Parse every line ──────────────────────────────────────────
    lines = []
    broken_lines = []
    empty_lines = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                empty_lines += 1
                continue
            try:
                obj = json.loads(stripped)
                lines.append((i, obj))
            except json.JSONDecodeError as e:
                broken_lines.append((i, str(e), stripped[:120]))

    print(f"  Total lines in file:    {i}")
    print(f"  Empty lines:            {empty_lines}")
    print(f"  Valid JSON lines:       {len(lines)}")
    print(f"  Broken JSON lines:      {len(broken_lines)}")
    print()

    # ── Show broken lines (first 5) ───────────────────────────────
    if broken_lines:
        print("  🚨 BROKEN LINES (first 5):")
        print("  " + "-" * 56)
        for line_num, error, preview in broken_lines[:5]:
            print(f"  Line {line_num}: {error}")
            print(f"    Preview: {preview}")
            print()

    if not lines:
        print("  ❌ No valid examples found! File is empty or all broken.")
        return

    # ── Detect format ─────────────────────────────────────────────
    first_obj = lines[0][1]
    if "conversations" in first_obj:
        fmt = "sharegpt"
    elif "instruction" in first_obj:
        fmt = "alpaca"
    else:
        fmt = "unknown"
        print(f"  ⚠️  Unknown format! Keys found: {list(first_obj.keys())}")

    print(f"  Format detected:        {fmt}")
    print()

    # ── Check GPT/output quality ──────────────────────────────────
    valid_json_outputs = 0
    bad_json_outputs = 0
    bad_examples = []
    output_lengths = []
    has_name = 0
    has_email = 0
    has_phone = 0
    has_experience = 0
    has_education = 0
    has_skills = 0

    for line_num, item in lines:
        # Extract the "answer" part
        if fmt == "sharegpt":
            gpt_msgs = [
                c for c in item.get("conversations", [])
                if c.get("from") == "gpt"
            ]
            output_text = gpt_msgs[0].get("value", "") if gpt_msgs else ""
        elif fmt == "alpaca":
            output_text = item.get("output", "")
        else:
            output_text = ""

        output_lengths.append(len(output_text))

        # Try parsing the output as JSON
        try:
            parsed = json.loads(output_text)
            valid_json_outputs += 1

            # Check which fields are present
            if isinstance(parsed, dict):
                if parsed.get("name"):
                    has_name += 1
                if parsed.get("email"):
                    has_email += 1
                if parsed.get("phone"):
                    has_phone += 1
                if parsed.get("experience"):
                    has_experience += 1
                if parsed.get("education"):
                    has_education += 1
                if parsed.get("hard_skills") or parsed.get("skills"):
                    has_skills += 1

        except (json.JSONDecodeError, ValueError):
            bad_json_outputs += 1
            if len(bad_examples) < 3:
                bad_examples.append((line_num, output_text[:150]))

    total = len(lines)
    print("  📊 OUTPUT QUALITY:")
    print("  " + "-" * 56)
    print(f"  Valid JSON outputs:     {valid_json_outputs}/{total} ({valid_json_outputs/total*100:.1f}%)")
    print(f"  Broken JSON outputs:    {bad_json_outputs}/{total} ({bad_json_outputs/total*100:.1f}%)")
    print()

    if bad_examples:
        print("  🚨 BROKEN OUTPUT SAMPLES (first 3):")
        print("  " + "-" * 56)
        for line_num, preview in bad_examples:
            print(f"  Line {line_num}: {preview}")
            print()

    # ── Field coverage ────────────────────────────────────────────
    if valid_json_outputs > 0:
        print("  📋 FIELD COVERAGE (in valid outputs):")
        print("  " + "-" * 56)
        fields = [
            ("name", has_name),
            ("email", has_email),
            ("phone", has_phone),
            ("experience", has_experience),
            ("education", has_education),
            ("skills", has_skills),
        ]
        for field_name, count in fields:
            pct = count / valid_json_outputs * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            status = "✅" if pct >= 70 else "⚠️" if pct >= 40 else "❌"
            print(f"  {status} {field_name:<15s} {bar} {pct:5.1f}% ({count}/{valid_json_outputs})")
        print()

    # ── Output length stats ───────────────────────────────────────
    if output_lengths:
        avg_len = sum(output_lengths) / len(output_lengths)
        min_len = min(output_lengths)
        max_len = max(output_lengths)
        print(f"  Output length:  avg={avg_len:.0f} chars, min={min_len}, max={max_len}")

        # Flag suspiciously short/long outputs
        very_short = sum(1 for l in output_lengths if l < 20)
        very_long = sum(1 for l in output_lengths if l > 5000)
        if very_short > 0:
            print(f"  ⚠️  {very_short} outputs are suspiciously short (< 20 chars)")
        if very_long > 0:
            print(f"  ⚠️  {very_long} outputs are very long (> 5000 chars)")

    # ── Show a sample ─────────────────────────────────────────────
    print()
    print("  🔍 SAMPLE EXAMPLE (first one):")
    print("  " + "-" * 56)
    first = lines[0][1]
    if fmt == "sharegpt":
        convos = first.get("conversations", [])
        for msg in convos:
            role = msg.get("from", "?")
            val = msg.get("value", "")
            preview = val[:120] + "..." if len(val) > 120 else val
            print(f"  [{role:>6s}]: {preview}")
    elif fmt == "alpaca":
        for key in ["instruction", "input", "output"]:
            val = first.get(key, "")
            preview = val[:120] + "..." if len(val) > 120 else val
            print(f"  [{key:>11s}]: {preview}")

    # ── Final verdict ─────────────────────────────────────────────
    print()
    print("  " + "=" * 56)
    print("  📋 VERDICT:")
    print("  " + "=" * 56)

    issues = []
    if broken_lines:
        issues.append(f"{len(broken_lines)} broken JSON lines in file")
    if bad_json_outputs > total * 0.1:
        issues.append(f"{bad_json_outputs} outputs are not valid JSON ({bad_json_outputs/total*100:.0f}%)")
    if total < 50:
        issues.append(f"Only {total} examples — need at least 100, ideally 200+")
    if valid_json_outputs > 0 and has_name / valid_json_outputs < 0.5:
        issues.append("Less than 50% of outputs have a 'name' field")

    if not issues:
        print("  ✅ Data looks HEALTHY! Ready for training.")
        if total < 100:
            print(f"  💡 But {total} examples is on the low side.")
            print("     Consider adding more for better results (target: 200+)")
    else:
        print("  ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"     • {issue}")
        print()
        print("  💡 FIX: Regenerate with prepare_training_data.py")
        print("     or check the source data for encoding issues.")

    print()
    print("  " + "=" * 56)
    print()


if __name__ == "__main__":
    # Default path — change if your file is elsewhere
    default_path = "training_data/ultimate_train.jsonl"

    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        filepath = sys.argv[2] if len(sys.argv) > 2 else default_path
    elif len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = default_path

    inspect(filepath)