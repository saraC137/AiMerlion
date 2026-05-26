"""
finetune_unsloth.py

💅✨ FAIRY CODEMOTHER'S UNSLOTH FINE-TUNING PIPELINE ✨💅

Upgraded from the original finetune_model.py (which used raw HuggingFace + PEFT)
to use Unsloth — 2x faster training, 70% less VRAM! Perfect for your RTX 5070 Ti.

This script handles the FULL pipeline:
  Phase 1: Load & validate training data (JSONL)
  Phase 2: Load base model with QLoRA via Unsloth
  Phase 3: Fine-tune with SFTTrainer
  Phase 4: Export to GGUF for Ollama
  Phase 5: Generate Modelfile for Ollama deployment

Hardware Target: NVIDIA RTX 5070 Ti (16GB VRAM, Blackwell sm120)
Prerequisites:
    pip install unsloth trl accelerate datasets bitsandbytes

Usage:
    # Basic fine-tune with defaults (Llama 3.2 3B)
    python finetune_unsloth.py --train-file train_data.jsonl

    # Use a different model
    python finetune_unsloth.py --train-file train_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit

    # Full options
    python finetune_unsloth.py \\
        --train-file train_data.jsonl \\
        --val-file val_data.jsonl \\
        --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit \\
        --epochs 3 \\
        --batch-size 2 \\
        --seq-length 2048 \\
        --output-dir resume_model_finetuned \\
        --gguf q4_k_m

    # Export ONLY (skip training, just convert existing LoRA to GGUF)
    python finetune_unsloth.py --export-only --lora-path resume_model_finetuned/lora_model --gguf q4_k_m
"""

# ─── BLACKWELL RTX 5070 Ti FIX: Disable Unsloth's incompatible CUDA kernels ──
# Unsloth's fused CE loss and some attention kernels can't run on sm120 GPUs.
# Set env vars BEFORE any unsloth import so they take effect on first load.
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FUSED_CROSS_ENTROPY"] = "1"

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path


# ─── Pretty console output (inherited from the OG finetune_model.py) ─────────
class Console:
    """Glamorous console output because training logs deserve SPARKLE! 💅"""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def banner(text: str):
        width = 60
        print(f"\n{'═' * width}")
        print(f"  {Console.BOLD}{Console.HEADER}{text}{Console.RESET}")
        print(f"{'═' * width}")

    @staticmethod
    def success(msg: str):
        print(f"  {Console.GREEN}✅ {msg}{Console.RESET}")

    @staticmethod
    def warning(msg: str):
        print(f"  {Console.YELLOW}⚠️  {msg}{Console.RESET}")

    @staticmethod
    def error(msg: str):
        print(f"  {Console.RED}❌ {msg}{Console.RESET}")

    @staticmethod
    def info(msg: str):
        print(f"  {Console.CYAN}💡 {msg}{Console.RESET}")

    @staticmethod
    def stat(label: str, value, indent: int = 2):
        spaces = "  " * indent
        print(f"{spaces}{Console.DIM}{label}:{Console.RESET} {Console.BOLD}{value}{Console.RESET}")


# =============================================================================
# 📋 PHASE 1: DATA LOADING & VALIDATION
# Loading the training data — the foundation of everything!
# =============================================================================

def load_training_data(train_file: str, val_file: str = None) -> tuple:
    """
    📋 Load training data from JSONL files.

    Supports TWO formats:
    1. ShareGPT/Conversational (RECOMMENDED for Unsloth):
       {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

    2. Alpaca/Instruction format:
       {"instruction": "...", "input": "...", "output": "..."}

    The script auto-detects which format you're using!

    Args:
        train_file: Path to training JSONL
        val_file:   Path to validation JSONL (optional)

    Returns:
        (train_dataset, val_dataset, detected_format)
    """
    Console.banner("📋 PHASE 1: Loading Training Data")

    # ── Validate file exists ──────────────────────────────────────────
    if not os.path.exists(train_file):
        Console.error(f"Training file not found: {train_file}")
        Console.info("Create your training data first! Format options:")
        Console.info('  ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}')
        Console.info('  Alpaca:   {"instruction": "...", "input": "...", "output": "..."}')
        sys.exit(1)

    # ── Load and detect format ────────────────────────────────────────
    train_data = []
    detected_format = None

    with open(train_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                train_data.append(item)

                # Auto-detect format from first valid line
                if detected_format is None:
                    if "conversations" in item:
                        detected_format = "sharegpt"
                    elif "instruction" in item:
                        detected_format = "alpaca"
                    else:
                        Console.error(f"Line {line_num}: Unrecognized format. Need 'conversations' or 'instruction' key.")
                        sys.exit(1)

            except json.JSONDecodeError as e:
                Console.warning(f"Skipping malformed JSON at line {line_num}: {e}")

    if not train_data:
        Console.error("No valid training examples found!")
        sys.exit(1)

    Console.success(f"Loaded {len(train_data)} training examples")
    Console.stat("Format detected", detected_format)

    # ── Data quality checks ───────────────────────────────────────────
    # (Because bad data = bad model, darling! 🚫)
    empty_count = 0
    short_count = 0

    for item in train_data:
        if detected_format == "sharegpt":
            texts = [c.get("value", "") for c in item.get("conversations", [])]
            combined = " ".join(texts)
        else:
            combined = f"{item.get('instruction', '')} {item.get('input', '')} {item.get('output', '')}"

        if len(combined.strip()) < 10:
            empty_count += 1
        elif len(combined.strip()) < 50:
            short_count += 1

    if empty_count > 0:
        Console.warning(f"{empty_count} examples are nearly empty — consider removing them")
    if short_count > 0:
        Console.warning(f"{short_count} examples are very short (< 50 chars)")

    # ── Load validation data (if provided) ────────────────────────────
    val_data = []
    if val_file and os.path.exists(val_file):
        with open(val_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        val_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        Console.success(f"Loaded {len(val_data)} validation examples")
    elif val_file:
        Console.warning(f"Validation file not found: {val_file} — will auto-split 10%")
    else:
        Console.info("No validation file — will auto-split 10% from training data")

    # ── Convert to HuggingFace Dataset ────────────────────────────────
    from datasets import Dataset

    train_dataset = Dataset.from_list(train_data)

    if val_data:
        val_dataset = Dataset.from_list(val_data)
    else:
        # Auto-split: 90% train, 10% validation
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
        Console.info(f"Auto-split: {len(train_dataset)} train, {len(val_dataset)} validation")

    return train_dataset, val_dataset, detected_format


# =============================================================================
# 🧠 PHASE 2: MODEL SETUP (Unsloth + QLoRA)
# Loading the base model with Unsloth's memory-efficient magic!
# =============================================================================

def setup_model(model_name: str, max_seq_length: int = 2048):
    """
    🧠 Load base model with Unsloth + QLoRA.

    Unsloth replaces HuggingFace's standard training kernels with
    custom Triton/CUDA kernels that are 2x faster and use 70% less VRAM.
    Think of it like upgrading from economy to first class — same destination,
    WAY better experience! ✈️💅

    QLoRA = Quantized LoRA:
    - The base model weights are frozen and quantized to 4-bit
    - Only tiny LoRA adapter layers (rank 16-64) are trainable
    - This means ~1-2% of parameters are trained — HUGE memory savings!

    Args:
        model_name:     HuggingFace model identifier (use unsloth/ prefix for speed)
        max_seq_length: Max tokens per training example (2048 is plenty for resumes)

    Returns:
        (model, tokenizer)
    """
    Console.banner("🧠 PHASE 2: Loading Model with Unsloth")

    import torch
    from unsloth import FastLanguageModel

    Console.info(f"Base model: {model_name}")
    Console.info(f"Max sequence length: {max_seq_length}")

    # ── Verify GPU ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        Console.stat("GPU", f"{gpu_name} ({gpu_mem:.1f} GB)")
        Console.stat("CUDA", torch.version.cuda)

        # Blackwell detection (compute capability 12.x)
        cc = torch.cuda.get_device_capability(0)
        if cc[0] >= 12:
            Console.success(f"Blackwell architecture detected (sm{cc[0]}{cc[1]}) — you're on the cutting edge! 💎")
    else:
        Console.error("No GPU detected! Training on CPU will be PAINFULLY slow.")
        Console.info("Make sure CUDA 12.8+ is installed for your RTX 5070 Ti")
        sys.exit(1)

    # ── Load model with Unsloth ───────────────────────────────────────
    # Unsloth's FastLanguageModel handles:
    # - 4-bit quantization (QLoRA ready)
    # - Optimized attention kernels
    # - Memory-efficient gradient handling
    print()
    Console.info("Loading model (this may take 1-3 minutes on first run)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,             # Auto-detect: float16 for most GPUs
        load_in_4bit=True,      # QLoRA: quantize base weights to 4-bit NF4
    )

    Console.success("Base model loaded!")

    # ── Apply LoRA adapters ───────────────────────────────────────────
    # LoRA injects small trainable matrices into the attention layers.
    # Only these matrices are updated during training — everything else is FROZEN.
    #
    # Analogy: Imagine the base model is a fully decorated cake (frozen).
    # LoRA is just the icing — we only change the icing, not the whole cake! 🎂
    #
    # Key parameters:
    #   r=16        → LoRA rank (higher = more params = more expressive but slower)
    #   lora_alpha  → Scaling factor (typically 2x rank)
    #   target_modules → Which layers get LoRA adapters
    #                    (attention layers are where the "thinking" happens)

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,                        # LoRA rank: 8 for 3B models, 16 for 7B-14B
        target_modules=[            # Standard attention + MLP targets
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
            "gate_proj", "up_proj", "down_proj",       # MLP layers (better accuracy)
        ],
        lora_alpha=16,              # Scaling factor = 2 * rank (8*2=16)
        lora_dropout=0.1,           # Higher dropout to prevent catastrophic forgetting
        bias="none",                # No bias training (saves memory)
        use_gradient_checkpointing="unsloth",  # Unsloth's special GC — extra VRAM savings!
        random_state=42,
    )

    # ── Report parameter counts ───────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100

    Console.success(f"LoRA adapters applied!")
    Console.stat("Total parameters", f"{total:,}")
    Console.stat("Trainable parameters", f"{trainable:,} ({pct:.2f}%)")
    Console.stat("VRAM estimate", f"~{trainable * 2 / 1e9:.1f} GB for training")

    return model, tokenizer


# =============================================================================
# 🏋️ PHASE 3: TRAINING
# The main event! Where your model learns from your annotated resumes!
# =============================================================================

def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    data_format: str,
    output_dir: str = "resume_model_finetuned",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_seq_length: int = 2048,
):
    """
    🏋️ Fine-tune the model using SFTTrainer (Supervised Fine-Tuning).

    SFTTrainer from the TRL library handles:
    - Proper chat template formatting
    - Tokenization with padding/truncation
    - Training loop with gradient accumulation
    - Logging and checkpointing

    For your RTX 5070 Ti (16GB VRAM):
    - batch_size=2 with gradient_accumulation=4 → effective batch of 8
    - This balances speed vs. VRAM usage perfectly!

    Args:
        model:          The LoRA-wrapped model from Phase 2
        tokenizer:      The tokenizer from Phase 2
        train_dataset:  Training data (HF Dataset)
        val_dataset:    Validation data (HF Dataset)
        data_format:    "sharegpt" or "alpaca"
        output_dir:     Where to save checkpoints
        num_epochs:     Number of training passes (3 is usually enough)
        batch_size:     Per-device batch size (2 for 16GB VRAM)
        learning_rate:  Learning rate (2e-4 is standard for LoRA)
        max_seq_length: Max tokens per example

    Returns:
        trainer object (for saving)
    """
    Console.banner("🏋️ PHASE 3: Training")

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    Console.info(f"Epochs: {num_epochs}")
    Console.info(f"Batch size: {batch_size} (effective: {batch_size * 4} with grad accumulation)")
    Console.info(f"Learning rate: {learning_rate}")
    Console.info(f"Data format: {data_format}")

    # ── Configure the chat template mapping ───────────────────────────
    # This tells the SFTTrainer how to interpret your JSONL format.
    # ShareGPT format uses "conversations" with "from"/"value" pairs.
    # Alpaca format uses "instruction"/"input"/"output" fields.

    if data_format == "sharegpt":
        def format_sharegpt(example):
            """Convert ShareGPT format to a single text field (Llama 3 template)."""
            text_parts = ["<|begin_of_text|>"]
            for msg in example.get("conversations", []):
                role = msg.get("from", "")
                content = msg.get("value", "")
                if role == "system":
                    text_parts.append(
                        f"<|start_header_id|>system<|end_header_id|>\n\n"
                        f"{content}<|eot_id|>"
                    )
                elif role == "human":
                    text_parts.append(
                        f"<|start_header_id|>user<|end_header_id|>\n\n"
                        f"{content}<|eot_id|>"
                    )
                elif role == "gpt":
                    text_parts.append(
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                        f"{content}<|eot_id|>"
                    )
            return {"text": "".join(text_parts)}

        train_dataset = train_dataset.map(format_sharegpt)
        val_dataset = val_dataset.map(format_sharegpt)
        dataset_text_field = "text"
    else:
        def format_alpaca(example):
            """Convert Alpaca format to a single text field (Llama 3 template)."""
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            output = example.get("output", "")

            if inp:
                user_msg = f"{instruction}\n\n{inp}"
            else:
                user_msg = instruction

            text = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_msg}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{output}<|eot_id|>"
            )
            return {"text": text}

        train_dataset = train_dataset.map(format_alpaca)
        val_dataset = val_dataset.map(format_alpaca)
        dataset_text_field = "text"

    Console.success("Data formatted for training!")

    # ── Instruction masking (completion-only training) ─────────────────
    # Without this, the model trains on ALL tokens — including the long
    # resume text in the prompt. That floods the loss with noise and
    # causes the unnaturally high initial loss (>>10). With this collator,
    # only the assistant response tokens (the JSON output) contribute to
    # the loss. This is the correct setup for instruction fine-tuning.
    #
    # The response_template is the exact token sequence that marks where
    # the assistant answer starts. Everything BEFORE it is masked (loss=0).
    from trl import DataCollatorForCompletionOnlyLM

    # Llama 3 assistant header — must match the format_sharegpt/format_alpaca output exactly.
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )
    Console.info(f"Completion-only collator active — loss computed on assistant response only")

    # ── Training arguments ────────────────────────────────────────────
    # These are carefully tuned for RTX 5070 Ti (16GB VRAM):
    #
    # - per_device_train_batch_size=2: Fits comfortably in 16GB
    # - gradient_accumulation_steps=4: Effective batch of 8
    # - fp16=True / bf16: Mixed precision for speed
    #   (Blackwell supports bf16 natively which is more stable!)
    # - warmup_steps=10: Gentle start to avoid early divergence
    # - optim="adamw_8bit": 8-bit optimizer saves ~2GB VRAM!

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,          # Effective batch = batch_size * 4
        learning_rate=learning_rate,
        weight_decay=0.1,                       # Stronger regularization to prevent forgetting
        warmup_steps=50,                        # Gentle warmup — 50 steps for 3B model
        optim="adamw_8bit",                     # 8-bit Adam: saves ~2GB VRAM!
        fp16=not is_bfloat16_supported(),       # Use fp16 if bf16 not available
        bf16=is_bfloat16_supported(),           # bf16 preferred on Blackwell/Ampere+
        logging_steps=5,                        # Log every 5 steps
        eval_strategy="no",                     # Skip eval — Blackwell CE fix only covers training path
        # eval_steps=50,
        save_strategy="steps",
        save_steps=100,                         # Save checkpoint every 100 steps
        save_total_limit=3,                     # Keep only 3 latest checkpoints
        load_best_model_at_end=False,           # No eval = no "best model" tracking
        # metric_for_best_model="eval_loss",
        report_to="none",                       # No WandB/TensorBoard (keep it simple)
        seed=42,
    )

    # ── Initialize SFTTrainer ─────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=data_collator,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        packing=False,
    )

    # ── VRAM check before we go ───────────────────────────────────────
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        Console.stat("VRAM allocated", f"{allocated:.1f} GB")
        Console.stat("VRAM reserved", f"{reserved:.1f} GB")
        Console.stat("VRAM total", f"{total:.1f} GB")
        Console.stat("VRAM headroom", f"{total - reserved:.1f} GB")

        if total - reserved < 1.0:
            Console.warning("Less than 1GB VRAM headroom! Consider reducing batch_size to 1")

    # ── START TRAINING! 🚀 ────────────────────────────────────────────
    Console.info("⚠️  Close other GPU apps (games, browsers with GPU accel, etc.)")
    Console.info("Training started... This typically takes 30min - 2hrs depending on dataset size")
    print()

    start_time = datetime.now()
    trainer.train()
    duration = datetime.now() - start_time

    Console.success(f"Training complete! Duration: {duration}")

    # ── Save the LoRA adapters ────────────────────────────────────────
    lora_path = os.path.join(output_dir, "lora_model")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    Console.success(f"LoRA adapters saved to: {lora_path}")

    return trainer, lora_path


# =============================================================================
# 📊 PHASE 3b: POST-TRAINING EVALUATION
# The morning-after reviews! Did the audience love it? 🎭
# Because training loss alone is like judging a singer by how
# confident they LOOK — you gotta actually LISTEN to them sing! 🎤
# =============================================================================

def evaluate_model(
    model,
    tokenizer,
    val_dataset,
    data_format: str,
    max_seq_length: int = 2048,
    num_samples: int = 20,
):
    """
    📊 Evaluate the fine-tuned model on validation examples.

    For LLM fine-tuning (unlike spaCy NER), we can't just compute
    token-level F1. Instead, we measure:

    1. EXACT MATCH RATE — Did the model produce valid JSON?
    2. FIELD-LEVEL ACCURACY — For each field (name, phone, email, etc.),
       did the extracted value match the expected value?
    3. JSON VALIDITY RATE — Can we even parse the output?

    Think of it like a cooking competition:
    - Exact Match = "Is this EXACTLY the dish we asked for?"
    - Field-Level = "Did they get the sauce right? The protein? The garnish?"
    - JSON Validity = "Did they at least put it on a plate?!" 🍽️💅

    Args:
        model:          The fine-tuned model
        tokenizer:      The tokenizer
        val_dataset:    Validation dataset
        data_format:    "sharegpt" or "alpaca"
        max_seq_length: Max generation length
        num_samples:    How many examples to evaluate (default: 20)

    Returns:
        Dict with evaluation metrics
    """
    Console.banner("📊 PHASE 3b: Post-Training Evaluation")

    import torch
    from unsloth import FastLanguageModel

    # ── Blackwell sm120 (RTX 5070 Ti) inference warning ──────────────
    # Unsloth injects its custom attention kernels at model *load time*,
    # not at for_inference() time. On sm120, these kernels produce broken
    # attention output during generate(), causing complete gibberish.
    # model.eval() does NOT undo the kernel injection.
    # The in-process evaluation is unreliable on Blackwell — skip it and
    # test the exported GGUF with Ollama (uses llama.cpp, no Unsloth kernels).
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    if cc[0] >= 12:
        Console.warning("Blackwell GPU (sm120) detected — skipping in-process evaluation.")
        Console.info("Unsloth's attention kernels are injected at load time and produce")
        Console.info("broken output during generate() on sm120, even with model.eval().")
        Console.info("Use the exported GGUF + Ollama to evaluate your model instead:")
        Console.info("  ollama create aimerlion-resume -f <output_dir>/Modelfile")
        Console.info("  ollama run aimerlion-resume")
        return {
            "total": 0,
            "json_valid": 0,
            "json_invalid": 0,
            "exact_match": 0,
            "field_scores": {},
            "avg_length_ratio": 0.0,
            "json_valid_rate": 0.0,
            "exact_match_rate": 0.0,
            "errors": [],
            "sample_outputs": [],
            "skipped_reason": "Blackwell sm120 — use GGUF/Ollama for evaluation",
        }

    FastLanguageModel.for_inference(model)

    # ── Sample validation examples ────────────────────────────────────
    # We don't need to eval ALL examples — a representative sample
    # gives us a reliable signal without taking forever.
    total_available = len(val_dataset)
    num_samples = min(num_samples, total_available)
    Console.info(f"Evaluating on {num_samples} / {total_available} validation examples")

    # ── Extract prompts and expected outputs ──────────────────────────
    results = {
        "total": num_samples,
        "json_valid": 0,            # Model output is parseable JSON
        "json_invalid": 0,          # Model output is NOT parseable JSON
        "exact_match": 0,           # Output exactly matches expected
        "field_scores": {},         # Per-field accuracy
        "avg_length_ratio": 0.0,    # Output length vs expected length
        "errors": [],               # Detailed error log
        "sample_outputs": [],       # First few outputs for manual review
    }

    length_ratios = []

    for i in range(num_samples):
        example = val_dataset[i]

        # ── Extract the prompt (user message) and expected output ──────
        if data_format == "sharegpt":
            convos = example.get("conversations", [])
            system_msg = ""
            prompt_parts = []
            expected_output = ""
            for msg in convos:
                role = msg.get("from") if isinstance(msg, dict) else msg.get("from", "")
                value = msg.get("value") if isinstance(msg, dict) else msg.get("value", "")
                if role == "system":
                    system_msg = value or ""
                elif role == "human":
                    prompt_parts.append(value or "")
                elif role == "gpt":
                    expected_output = value or ""
            prompt = prompt_parts[-1] if prompt_parts else ""
        else:
            # Alpaca format
            system_msg = ""
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            prompt = f"{instruction}\n\n{inp}" if inp else instruction
            expected_output = example.get("output", "")

        if not prompt or not expected_output:
            continue

        # ── Format as Llama 3 chat (include system message if present) ──
        # Must match the training format exactly so the model sees the same
        # token pattern it learned from — system msg was in every training example.
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ── Generate the model's response ─────────────────────────────
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,        # Resume JSON shouldn't exceed this
                temperature=0.1,            # Low temp = deterministic output
                top_p=0.9,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # ── Decode only the NEW tokens (skip the prompt) ──────────────
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # ── Score: JSON validity ──────────────────────────────────────
        generated_json = None
        expected_json = None
        try:
            generated_json = json.loads(generated_text)
            results["json_valid"] += 1
        except (json.JSONDecodeError, ValueError):
            results["json_invalid"] += 1
            results["errors"].append({
                "sample": i,
                "error": "Invalid JSON output",
                "output_preview": generated_text[:200],
            })

        try:
            expected_json = json.loads(expected_output)
        except (json.JSONDecodeError, ValueError):
            # Expected output isn't JSON — do string comparison instead
            expected_json = None

        # ── Score: Exact match ────────────────────────────────────────
        if generated_text.strip() == expected_output.strip():
            results["exact_match"] += 1

        # ── Score: Field-level accuracy (if both are valid JSON) ──────
        # This is the REAL gold — checking each field individually!
        # Like grading each answer on an exam separately 📝
        if generated_json and expected_json and isinstance(expected_json, dict):
            for field_name, expected_value in expected_json.items():
                if field_name not in results["field_scores"]:
                    results["field_scores"][field_name] = {
                        "correct": 0, "total": 0, "missing": 0
                    }
                results["field_scores"][field_name]["total"] += 1

                generated_value = generated_json.get(field_name)
                if generated_value is None:
                    results["field_scores"][field_name]["missing"] += 1
                elif _normalize_value(generated_value) == _normalize_value(expected_value):
                    results["field_scores"][field_name]["correct"] += 1

        # ── Length ratio (detect truncation or hallucination) ─────────
        if expected_output:
            ratio = len(generated_text) / max(len(expected_output), 1)
            length_ratios.append(ratio)

        # ── Save sample outputs for manual review ─────────────────────
        if i < 5:  # Save first 5 for inspection
            results["sample_outputs"].append({
                "prompt_preview": prompt[:150] + "..." if len(prompt) > 150 else prompt,
                "expected_preview": expected_output[:200] + "..." if len(expected_output) > 200 else expected_output,
                "generated_preview": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                "json_valid": generated_json is not None,
            })

    # ── Calculate summary metrics ─────────────────────────────────────
    results["avg_length_ratio"] = (
        sum(length_ratios) / len(length_ratios) if length_ratios else 0.0
    )
    results["json_valid_rate"] = results["json_valid"] / max(results["total"], 1)
    results["exact_match_rate"] = results["exact_match"] / max(results["total"], 1)

    # ── Print the GORGEOUS results table ──────────────────────────────
    print()
    print("═" * 60)
    print("  📊 EVALUATION RESULTS")
    print("═" * 60)
    Console.stat("Total evaluated", results["total"])
    Console.stat("JSON valid", f"{results['json_valid']}/{results['total']} ({results['json_valid_rate']:.1%})")
    Console.stat("Exact match", f"{results['exact_match']}/{results['total']} ({results['exact_match_rate']:.1%})")
    Console.stat("Avg length ratio", f"{results['avg_length_ratio']:.2f}x (1.0 = perfect)")

    # ── Per-field accuracy table ──────────────────────────────────────
    if results["field_scores"]:
        print()
        print(f"  {'Field':<25s} {'Correct':>8s} {'Missing':>8s} {'Total':>8s} {'Acc':>8s}")
        print(f"  {'-'*57}")
        for field, scores in sorted(
            results["field_scores"].items(),
            key=lambda x: x[1]["correct"] / max(x[1]["total"], 1),
            reverse=True,
        ):
            acc = scores["correct"] / max(scores["total"], 1)
            print(
                f"  {field:<25s} {scores['correct']:>8d} "
                f"{scores['missing']:>8d} {scores['total']:>8d} "
                f"{acc:>7.1%}"
            )

    # ── Health verdicts ───────────────────────────────────────────────
    print()
    if results["json_valid_rate"] >= 0.9:
        Console.success("JSON validity: EXCELLENT — model produces structured output! 🎉")
    elif results["json_valid_rate"] >= 0.7:
        Console.warning("JSON validity: OKAY — some outputs aren't valid JSON")
    else:
        Console.error("JSON validity: POOR — model struggles to produce valid JSON 😢")
        Console.info("Try: more training data, more epochs, or lower temperature")

    if results["avg_length_ratio"] < 0.3:
        Console.warning("Outputs are MUCH shorter than expected — possible truncation!")
        Console.info("Check your Modelfile stop tokens — they might be cutting output early")
    elif results["avg_length_ratio"] > 3.0:
        Console.warning("Outputs are MUCH longer than expected — possible hallucination!")
        Console.info("The model may be generating extra fields or repeating itself")

    # ── Sample outputs for manual review ──────────────────────────────
    if results["sample_outputs"]:
        print()
        print("═" * 60)
        print("  🔍 SAMPLE OUTPUTS (first 5 — for manual review)")
        print("═" * 60)
        for idx, sample in enumerate(results["sample_outputs"]):
            print(f"\n  ── Sample {idx + 1} {'✅' if sample['json_valid'] else '❌'} ──")
            print(f"  Prompt:    {sample['prompt_preview']}")
            print(f"  Expected:  {sample['expected_preview']}")
            print(f"  Generated: {sample['generated_preview']}")

    print("═" * 60)

    return results


def _normalize_value(value) -> str:
    """
    Normalize a value for comparison — handles strings, lists, numbers.
    
    We lowercase, strip whitespace, and sort lists so that
    ["Python", "Java"] matches ["java", "python"]. 
    Because ORDER shouldn't matter for skills, darling! 💅
    """
    if isinstance(value, str):
        return value.strip().lower()
    elif isinstance(value, list):
        return str(sorted([str(v).strip().lower() for v in value]))
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return str(value).strip().lower()


# =============================================================================
# 📦 PHASE 4: EXPORT TO GGUF
# Converting to Ollama's native format — the final transformation!
# =============================================================================

def export_to_gguf(
    model,
    tokenizer,
    output_dir: str = "resume_model_finetuned",
    quantization: str = "q4_k_m",
):
    """
    📦 Export the fine-tuned model to GGUF format for Ollama.

    GGUF = GPT-Generated Unified Format — it's what Ollama and llama.cpp
    speak natively. Think of it like converting a movie to the right format
    for your TV — same movie, just packaged differently! 📺

    Quantization levels (from smallest to largest):
    - q4_k_m:  Best quality/size tradeoff (RECOMMENDED) ~4GB for 7B
    - q5_k_m:  Slightly better quality, ~5GB for 7B
    - q8_0:    High quality, ~8GB for 7B
    - f16:     Full precision, ~14GB for 7B (for debugging only)

    For your 16GB RTX 5070 Ti:
    - q4_k_m lets you run 7B models + leave room for context window
    - q8_0 works if you want max quality and don't need huge context

    Args:
        model:          The fine-tuned model
        tokenizer:      The tokenizer
        output_dir:     Where to save the GGUF file
        quantization:   Quantization method (q4_k_m, q5_k_m, q8_0, f16)

    Returns:
        Path to the GGUF file
    """
    Console.banner("📦 PHASE 4: Exporting to GGUF")

    Console.info(f"Quantization: {quantization}")
    Console.info(f"Output directory: {output_dir}")

    gguf_dir = os.path.join(output_dir, "gguf_model")
    os.makedirs(gguf_dir, exist_ok=True)

    # ── Unsloth handles the entire merge + convert + quantize pipeline! ─
    # This replaces the old 3-step process:
    #   1. merge_lora.py (merge LoRA into base)
    #   2. convert_hf_to_gguf.py (convert to GGUF)
    #   3. quantize (compress weights)
    #
    # Unsloth does ALL THREE in one call! 🎉

    Console.info("Merging LoRA, converting to GGUF, and quantizing...")
    Console.info("(This may take 5-15 minutes depending on model size)")

    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method=quantization,
    )

    # ── Find the output file ──────────────────────────────────────────
    # Unsloth appends "_gguf" to the folder name automatically,
    # so we search both locations to be safe. Pick the most recently
    # modified .gguf so multiple training runs don't pick a stale file.
    gguf_files = list(Path(gguf_dir).glob("*.gguf"))
    if not gguf_files:
        gguf_files = list(Path(gguf_dir + "_gguf").glob("*.gguf"))
    if gguf_files:
        gguf_path = str(max(gguf_files, key=lambda f: f.stat().st_mtime))
        file_size = os.path.getsize(gguf_path) / (1024**3)
        Console.success(f"GGUF exported: {gguf_path}")
        Console.stat("File size", f"{file_size:.2f} GB")
    else:
        Console.error("GGUF export failed — no .gguf file found!")
        Console.info("Try running manually: model.save_pretrained_gguf(...)")
        gguf_path = None

    return gguf_path


# =============================================================================
# 🚀 PHASE 5: GENERATE OLLAMA MODELFILE
# The final step — creating the Ollama deployment configuration!
# =============================================================================

def create_modelfile(
    gguf_path: str,
    output_dir: str = "resume_model_finetuned",
    model_name: str = "aimerlion-resume",
):
    """
    🚀 Generate an Ollama Modelfile for deploying your fine-tuned model.

    The Modelfile is like a Dockerfile but for AI models — it tells Ollama
    how to package and serve your model. It includes:
    - The GGUF weights path
    - The chat template (how user/assistant messages are formatted)
    - Inference parameters (temperature, stop tokens, etc.)
    - A system prompt (the model's "personality")

    After this, you just run:
        ollama create aimerlion-resume -f Modelfile
        ollama run aimerlion-resume

    Args:
        gguf_path:   Path to the .gguf file from Phase 4
        output_dir:  Where to save the Modelfile
        model_name:  Name for the Ollama model

    Returns:
        Path to the Modelfile
    """
    Console.banner("🚀 PHASE 5: Creating Ollama Modelfile")

    if not gguf_path:
        Console.error("No GGUF path provided — cannot create Modelfile!")
        return None

    # ── Resolve to absolute path (Ollama needs this) ──────────────────
    abs_gguf_path = os.path.abspath(gguf_path)

    # ── Build the Modelfile ───────────────────────────────────────────
    # NOTE: The TEMPLATE uses Go template syntax ({{ .System }} etc.)
    # which is what Ollama expects. The Llama 3 format uses
    # <|start_header_id|> tokens — this alignment is CRITICAL!

   # ── Build the Modelfile ───────────────────────────────────────────
    # NOTE: The TEMPLATE uses Go template syntax ({{ .System }} etc.)
    # which is what Ollama expects. The Llama 3 format uses
    # <|start_header_id|> tokens — this alignment is CRITICAL!

    modelfile_content = f"""# ═══════════════════════════════════════════════════════════
# AiMerlion Resume Extraction Model (Llama 3)
# Fine-tuned for SG/MY resume parsing
# Generated by finetune_unsloth.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}
# ═══════════════════════════════════════════════════════════

# Base: your fine-tuned GGUF weights
FROM {abs_gguf_path}

# Chat template: Llama 3 format (matches training format!)
TEMPLATE \"\"\"<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

\"\"\"

# System prompt: tells the model its role
SYSTEM \"\"\"You are a precise resume data extraction assistant specializing in Singapore and Malaysia resumes. Extract the requested information and return valid JSON only. Handle 8-digit phone numbers (+65/+60), date-first experience formats, local company names, and multilingual content accurately.\"\"\"

# Inference parameters (tuned for structured extraction)
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
"""

    # ── Write the Modelfile ───────────────────────────────────────────
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    Console.success(f"Modelfile created: {modelfile_path}")

    # ── Print deployment instructions ─────────────────────────────────
    print()
    print("═" * 60)
    print("  🎯 DEPLOYMENT INSTRUCTIONS")
    print("═" * 60)
    print(f"  1. Create the Ollama model:")
    print(f"     ollama create {model_name} -f {os.path.abspath(modelfile_path)}")
    print()
    print(f"  2. Test it:")
    print(f"     ollama run {model_name}")
    print()
    print(f"  3. Use in AiMerlion (update ai_extractor.py):")
    print(f'     model_name = "{model_name}"')
    print("═" * 60)

    return modelfile_path


# =============================================================================
# 🎬 MAIN — THE GRAND PRODUCTION!
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="💅✨ Fairy Codemother's Unsloth Fine-Tuning Pipeline ✨💅\n"
                    "Fine-tune LLMs for resume extraction and deploy to Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data arguments ────────────────────────────────────────────────
    parser.add_argument(
        "--train-file", type=str, required=False, default="train_data.jsonl",
        help="Path to training data JSONL (ShareGPT or Alpaca format)"
    )
    parser.add_argument(
        "--val-file", type=str, default=None,
        help="Path to validation data JSONL (optional — auto-splits 10%% if not provided)"
    )

    # ── Model arguments ───────────────────────────────────────────────
    parser.add_argument(
        "--model", type=str, default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        help="Base model (default: unsloth/Llama-3.2-3B-Instruct-bnb-4bit). "
             "Options: unsloth/Llama-3.2-3B-Instruct-bnb-4bit (best for 16GB VRAM), "
             "unsloth/Llama-3.2-1B-Instruct-bnb-4bit (lighter, faster), "
             "unsloth/Qwen2.5-7B-Instruct-bnb-4bit (larger, needs ChatML template)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=2048,
        help="Max sequence length (default: 2048 — plenty for resumes)"
    )

    # ── Training arguments ────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 16GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5 for 3B models)")
    parser.add_argument("--output-dir", type=str, default="resume_model_finetuned", help="Output directory")

    # ── Export arguments ──────────────────────────────────────────────
    parser.add_argument(
        "--gguf", type=str, default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization (default: q4_k_m — best quality/size tradeoff)"
    )
    parser.add_argument(
        "--model-name", type=str, default="aimerlion-resume",
        help="Name for the Ollama model (default: aimerlion-resume)"
    )

    # ── Export-only mode ──────────────────────────────────────────────
    parser.add_argument(
        "--export-only", action="store_true",
        help="Skip training, just convert existing LoRA to GGUF"
    )
    parser.add_argument(
        "--lora-path", type=str, default=None,
        help="Path to existing LoRA adapters (for --export-only mode)"
    )

    args = parser.parse_args()

    # ── Beautiful startup banner ──────────────────────────────────────
    print()
    print("═" * 60)
    print("  💅✨ FAIRY CODEMOTHER'S UNSLOTH PIPELINE ✨💅")
    print("  Fine-tune → GGUF → Ollama — all in one script!")
    print("═" * 60)
    print(f"  🧠 Model:    {args.model}")
    print(f"  📂 Data:     {args.train_file}")
    print(f"  📁 Output:   {args.output_dir}")
    print(f"  📦 GGUF:     {args.gguf}")
    print(f"  🏷️  Name:     {args.model_name}")
    print("═" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Export-only mode ──────────────────────────────────────────────
    if args.export_only:
        Console.info("Export-only mode — skipping training")

        lora_path = args.lora_path or os.path.join(args.output_dir, "lora_model")
        if not os.path.exists(lora_path):
            Console.error(f"LoRA path not found: {lora_path}")
            Console.info("Train first, or specify --lora-path")
            sys.exit(1)

        from unsloth import FastLanguageModel

        Console.info(f"Loading LoRA from: {lora_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=args.seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        gguf_path = export_to_gguf(model, tokenizer, args.output_dir, args.gguf)
        create_modelfile(gguf_path, args.output_dir, args.model_name)

        print()
        Console.success("Export complete! 🎉")
        return

    # ══════════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ══════════════════════════════════════════════════════════════════

    # Phase 1: Load data
    train_ds, val_ds, data_format = load_training_data(args.train_file, args.val_file)

    # Phase 2: Setup model
    model, tokenizer = setup_model(args.model, args.seq_length)

    # Phase 3: Train!
    trainer, lora_path = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        data_format=data_format,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.seq_length,
    )

    # Phase 3b: Evaluate! (The part we were MISSING, darling! 💅)
    eval_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        val_dataset=val_ds,
        data_format=data_format,
        max_seq_length=args.seq_length,
        num_samples=20,
    )

    # Save evaluation results for tracking over time
    eval_path = os.path.join(args.output_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        # Remove sample_outputs from saved file (too verbose for JSON)
        save_results = {k: v for k, v in eval_results.items() if k != "sample_outputs"}
        json.dump(save_results, f, indent=2, default=str)
    Console.success(f"Evaluation results saved: {eval_path}")

    # Phase 4: Export to GGUF
    gguf_path = export_to_gguf(model, tokenizer, args.output_dir, args.gguf)

    # Phase 5: Create Modelfile
    modelfile_path = create_modelfile(gguf_path, args.output_dir, args.model_name)

    # ── Grand finale! ─────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  💅✨ TRAINING COMPLETE! THE CROWD GOES WILD! ✨💅")
    print("═" * 60)
    print(f"  📁 LoRA adapters:  {lora_path}")
    if gguf_path:
        print(f"  📦 GGUF model:     {gguf_path}")
    if modelfile_path:
        print(f"  📄 Modelfile:      {modelfile_path}")
    print()
    print("  🎯 NEXT STEPS:")
    print(f"  1. ollama create {args.model_name} -f {os.path.abspath(modelfile_path or '')}")
    print(f"  2. ollama run {args.model_name}")
    print(f'  3. Update ai_extractor.py: model_name = "{args.model_name}"')
    print()
    print("  💅 The Fairy Codemother is SO PROUD of you! 💅")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()