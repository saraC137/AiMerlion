"""
prepare_lite_for_training.py

💎✨ FAIRY CODEMOTHER'S UNSLOTH GUI DATA PREP ✨💎

Converts synthetic_lite.jsonl into Alpaca-format JSON files ready for
Unsloth GUI fine-tuning.

Output:
    data_gen/training/train_alpaca.json   — 75 examples (training set)
    data_gen/training/val_alpaca.json     — 19 examples (validation set)
    data_gen/training/data_card.md        — Human-readable description

Each record looks like:
    {
        "instruction": "Extract structured information from this resume...",
        "input": "<raw resume text>",
        "output": "<canonical JSON ground truth>"
    }

Usage:
    python prepare_lite_for_training.py
    python prepare_lite_for_training.py --split 0.8        # custom split ratio
    python prepare_lite_for_training.py --seed 42          # reproducible split
    python prepare_lite_for_training.py --stratify         # balanced ethnicity in val set
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict


# ════════════════════════════════════════════════════════════════════════════
# 📋 INSTRUCTION TEMPLATE
# ════════════════════════════════════════════════════════════════════════════

# This is what the model will see at INFERENCE time — keep it consistent
# with how you'll actually use the fine-tuned model in production.
INSTRUCTION = """You are a resume extraction model specialized in Singapore and Malaysia resumes. Extract the candidate's information from the resume text below into a structured JSON object using the canonical schema.

Canonical schema fields:
- Name, Phone, Email, Current Location, Current Company, Current Title
- Summary, Function, Industry (list), Language Skills (list)
- Work Experience: list of {company, title, from, to, responsibility (list)}
- Education: list of {school, major, degree, dates}
- Certifications (list), hard_skills/tags (list), soft_skills/skills (list)

Pay special attention to:
- Singapore patronymics (bin/binti for Malay, s/o or d/o for Indian)
- Singapore 8-digit phone numbers (+65 8XXX XXXX or +65 9XXX XXXX)
- Malaysia phone numbers (+60 1X-XXX XXXX)
- Pte Ltd / Sdn Bhd / Berhad company suffixes
- CMFAS module numbers (Module 1A, 5, 6, 6A, 8, 8A, 9, 9A, HI)
- Singapore polytechnics and universities

Output ONLY the JSON object, no commentary."""


# ════════════════════════════════════════════════════════════════════════════
# 🎨 CANONICAL OUTPUT BUILDER
# ════════════════════════════════════════════════════════════════════════════

# These are the fields we want the model to LEARN to extract
# (keep ordering stable across all training examples — helps the model)
OUTPUT_SCHEMA_FIELDS = [
    "Name", "Phone", "Email", "Current Location",
    "Current Company", "Current Title",
    "Summary", "Function", "Industry", "Language Skills",
    "Certifications", "hard_skills/tags", "soft_skills/skills",
    "Achievements", "Work Experience", "Project Experience", "Education",
]

LIST_FIELDS = {
    "Industry", "Language Skills", "Work Experience", "Education",
    "Certifications", "hard_skills/tags", "soft_skills/skills",
    "Achievements", "Project Experience",
}

def _as_list(v):
    if v in (None, ""): return []
    if isinstance(v, list): return [x for x in v if x not in (None, "")]
    return [v]                      # wraps a stray "Technology" into ["Technology"]

def build_output_json(ground_truth: dict) -> str:
    ordered = {}
    for field in OUTPUT_SCHEMA_FIELDS:
        val = ground_truth.get(field, "")
        val = _as_list(val) if field in LIST_FIELDS else ("" if val is None else val)
        ordered[field] = val
    return json.dumps(ordered, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════════════
# 🔄 RECORD CONVERTER
# ════════════════════════════════════════════════════════════════════════════

def convert_to_alpaca(record: Dict) -> Dict:
    """Convert one synthetic_lite record into Alpaca format."""
    raw_text = record.get("raw_text", "").strip()
    ground_truth = record.get("ground_truth", {})

    if not raw_text or not ground_truth:
        return None

    output_json = build_output_json(ground_truth)

    return {
        "instruction": INSTRUCTION,
        "input": raw_text,
        "output": output_json,
    }


# ════════════════════════════════════════════════════════════════════════════
# 📊 STRATIFIED SPLIT (keeps ethnicity distribution balanced)
# ════════════════════════════════════════════════════════════════════════════

def stratified_split(records: List[Dict], val_ratio: float,
                     rng: random.Random) -> tuple:
    """Split records keeping ethnicity proportional in train + val sets."""
    by_ethnicity = defaultdict(list)
    for rec in records:
        eth = rec.get("metadata", {}).get("ethnicity_tag", "unknown")
        by_ethnicity[eth].append(rec)

    train_set, val_set = [], []
    for eth, items in by_ethnicity.items():
        rng.shuffle(items)
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_set.extend(items[:val_count])
        train_set.extend(items[val_count:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    return train_set, val_set


def random_split(records: List[Dict], val_ratio: float,
                 rng: random.Random) -> tuple:
    """Simple random split."""
    shuffled = records[:]
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


# ════════════════════════════════════════════════════════════════════════════
# 📂 LOADING / SAVING
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


def save_json(path: Path, data: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════════════
# 📊 DATA CARD GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def build_data_card(records: List[Dict], train_set: List[Dict],
                    val_set: List[Dict], args) -> str:
    """Generate a human-readable description of the dataset."""
    eth_counter = Counter(r.get("metadata", {}).get("ethnicity_tag", "?") for r in records)
    ind_counter = Counter(r.get("metadata", {}).get("industry", "?") for r in records)
    sen_counter = Counter(r.get("metadata", {}).get("seniority", "?") for r in records)

    lines = [
        "# AiMerlion Stage 0 LITE — Training Dataset Card",
        "",
        f"**Generated:** Stage 0 LITE — synthetic SG/MY resumes",
        f"**Total records:** {len(records)}",
        f"**Training set:** {len(train_set)}",
        f"**Validation set:** {len(val_set)}",
        f"**Split strategy:** {'Stratified by ethnicity' if args.stratify else 'Random'}",
        f"**Random seed:** {args.seed}",
        "",
        "## Format",
        "",
        "Alpaca-style JSON array. Each record has:",
        "- `instruction`: SG/MY resume extraction directive (constant across all examples)",
        "- `input`: Raw resume text",
        "- `output`: JSON object with 15 canonical fields",
        "",
        "## Distribution",
        "",
        "### Ethnicity",
    ]
    for eth, count in eth_counter.most_common():
        pct = 100 * count / len(records)
        lines.append(f"- **{eth}**: {count} ({pct:.1f}%)")
    lines.append("")
    lines.append("### Industry")
    for ind, count in ind_counter.most_common():
        pct = 100 * count / len(records)
        lines.append(f"- **{ind}**: {count} ({pct:.1f}%)")
    lines.append("")
    lines.append("### Seniority")
    for sen, count in sen_counter.most_common():
        pct = 100 * count / len(records)
        lines.append(f"- **{sen}**: {count} ({pct:.1f}%)")

    lines.extend([
        "",
        "## How to use in Unsloth GUI",
        "",
        "1. Open Unsloth GUI",
        "2. **Base model**: `unsloth/Qwen2.5-7B-Instruct` (or local path)",
        "3. **Dataset**: Load `train_alpaca.json` as Alpaca format",
        "4. **Validation dataset** (optional): Load `val_alpaca.json`",
        "5. **Recommended LoRA settings**:",
        "   - LoRA rank (r): 16",
        "   - LoRA alpha: 32 (2 × rank)",
        "   - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`",
        "   - Dropout: 0",
        "6. **Recommended training settings**:",
        "   - Epochs: 3 (small dataset, watch for overfitting)",
        "   - Batch size: 2 per device",
        "   - Gradient accumulation: 4",
        "   - Learning rate: 2e-4",
        "   - Warmup ratio: 0.03",
        "   - Optimizer: adamw_8bit",
        "   - Precision: BF16 (you have RTX 5070 Ti Blackwell)",
        "   - Max sequence length: 4096",
        "",
        "## After training",
        "",
        "1. Save LoRA adapter to `models/qwen7b_lite_v1/`",
        "2. Run: `python evaluate_lite.py` (next script we'll write)",
        "3. Compare F1 against baseline `ai_extractor.py` on golden test set",
        "",
        "## Decision threshold",
        "",
        "Per `tasks/todo.md`:",
        "- F1 lift ≥ +5% on SG/MY fields → ✅ Commit to full Stage 0 (1000 resumes)",
        "- F1 lift 0% to +5% → 🤔 Investigate before scaling",
        "- F1 lift < 0% → 🛑 Don't scale, rethink",
    ])

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 🎬 MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic_lite for Unsloth GUI training")
    parser.add_argument("--input", default="data_gen/synthetic_lite.jsonl",
                        help="Path to synthetic_lite.jsonl")
    parser.add_argument("--output-dir", default="data_gen/training",
                        help="Output directory")
    parser.add_argument("--split", type=float, default=0.2,
                        help="Validation ratio (default: 0.2 = 80/20 split)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    parser.add_argument("--stratify", action="store_true", default=True,
                        help="Stratify split by ethnicity (default: True)")
    parser.add_argument("--no-stratify", dest="stratify", action="store_false",
                        help="Use random split instead of stratified")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + "  💎  UNSLOTH GUI DATA PREP  💎  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # ─── Load source ─────────────────────────────────────────────────────
    input_path = Path(args.input)
    records = load_jsonl(input_path)
    if not records:
        print("❌ No records loaded — exiting")
        return
    print(f"\n📂 Loaded {len(records)} records from {input_path}")

    # ─── Convert each record ─────────────────────────────────────────────
    print(f"\n🔄 Converting to Alpaca format...")
    converted = []
    skipped = 0
    for rec in records:
        alpaca = convert_to_alpaca(rec)
        if alpaca:
            converted.append({"_record": rec, "_alpaca": alpaca})
        else:
            skipped += 1
    if skipped:
        print(f"   ⚠️  Skipped {skipped} records (missing raw_text or ground_truth)")
    print(f"   ✅ Converted {len(converted)} records")

    # ─── Split ───────────────────────────────────────────────────────────
    print(f"\n📊 Splitting train/val ({(1 - args.split) * 100:.0f}/{args.split * 100:.0f})...")
    rng = random.Random(args.seed)
    if args.stratify:
        train_pairs, val_pairs = stratified_split(converted, args.split, rng)
        print(f"   ✨ Stratified by ethnicity")
    else:
        train_pairs, val_pairs = random_split(converted, args.split, rng)
        print(f"   ✨ Random split")
    print(f"   Train: {len(train_pairs)}")
    print(f"   Val:   {len(val_pairs)}")

    # Print split breakdown by ethnicity
    train_eth = Counter(p["_record"].get("metadata", {}).get("ethnicity_tag", "?") for p in train_pairs)
    val_eth = Counter(p["_record"].get("metadata", {}).get("ethnicity_tag", "?") for p in val_pairs)
    print(f"\n   Ethnicity distribution:")
    all_eth = set(train_eth.keys()) | set(val_eth.keys())
    for eth in sorted(all_eth):
        print(f"      {eth:15s} train={train_eth.get(eth, 0):3d}  val={val_eth.get(eth, 0):3d}")

    # ─── Extract just the alpaca content ─────────────────────────────────
    train_set = [p["_alpaca"] for p in train_pairs]
    val_set = [p["_alpaca"] for p in val_pairs]

    # ─── Save ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    train_path = output_dir / "train_alpaca.json"
    val_path = output_dir / "val_alpaca.json"
    card_path = output_dir / "data_card.md"

    save_json(train_path, train_set)
    save_json(val_path, val_set)

    print(f"\n💾 Saved:")
    print(f"   {train_path}  ({len(train_set)} records)")
    print(f"   {val_path}  ({len(val_set)} records)")

    # ─── Data card ───────────────────────────────────────────────────────
    full_records = [p["_record"] for p in (train_pairs + val_pairs)]
    train_records = [p["_record"] for p in train_pairs]
    val_records = [p["_record"] for p in val_pairs]
    card = build_data_card(full_records, train_records, val_records, args)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    print(f"   {card_path}  (instructions for Unsloth GUI)")

    # ─── Quick stats on the converted data ───────────────────────────────
    if train_set:
        avg_input_chars = sum(len(r["input"]) for r in train_set) / len(train_set)
        avg_output_chars = sum(len(r["output"]) for r in train_set) / len(train_set)
        avg_total_chars = avg_input_chars + avg_output_chars + len(INSTRUCTION)
        # Rough token estimate: ~4 chars/token
        approx_tokens = avg_total_chars / 4
        print(f"\n📐 Length stats (training set):")
        print(f"   Avg input chars:   {avg_input_chars:.0f}")
        print(f"   Avg output chars:  {avg_output_chars:.0f}")
        print(f"   Avg total chars:   {avg_total_chars:.0f}  (~{approx_tokens:.0f} tokens)")
        if approx_tokens > 4000:
            print(f"   ⚠️  Some examples may exceed 4096 token limit — consider raising max_seq_length to 6144")

    print("\n" + "═" * 70)
    print("  ✨ READY FOR UNSLOTH GUI  ✨".center(70))
    print("═" * 70)
    print(f"\n💋 Next step: Open Unsloth GUI and load these files!")
    print(f"   Read {card_path} for recommended training settings.")


if __name__ == "__main__":
    main()