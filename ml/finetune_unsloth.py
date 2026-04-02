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
    # Basic fine-tune with defaults (Qwen 2.5 7B)
    python finetune_unsloth.py --train-file train_data.jsonl

    # Use a larger model (tight fit on 16GB!)
    python finetune_unsloth.py --train-file train_data.jsonl --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit

    # Full options
    python finetune_unsloth.py \\
        --train-file train_data.jsonl \\
        --val-file val_data.jsonl \\
        --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \\
        --epochs 3 \\
        --batch-size 2 \\
        --seq-length 2048 \\
        --output-dir resume_model_finetuned \\
        --gguf q4_k_m

    # Export ONLY (skip training, just convert existing LoRA to GGUF)
    python finetune_unsloth.py --export-only --lora-path resume_model_finetuned/lora_model --gguf q4_k_m
"""

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
        r=16,                       # LoRA rank: 16 is the sweet spot for 7B-14B
        target_modules=[            # Standard attention + MLP targets
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
            "gate_proj", "up_proj", "down_proj",       # MLP layers (better accuracy)
        ],
        lora_alpha=32,              # Scaling factor = 2 * rank
        lora_dropout=0.05,          # Small dropout for regularization
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
    learning_rate: float = 2e-4,
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
        # ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        dataset_kwargs = {
            "dataset_text_field": None,  # Not used for ShareGPT
        }
        # For ShareGPT, we need to format the conversations manually
        # because SFTTrainer expects a specific format
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
        # Alpaca: {"instruction": "...", "input": "...", "output": "..."}
        # Format into ChatML template for Qwen models
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
        weight_decay=0.01,                      # Standard L2 regularization
        warmup_steps=10,                        # Gentle learning rate warmup
        optim="adamw_8bit",                     # 8-bit Adam: saves ~2GB VRAM!
        fp16=not is_bfloat16_supported(),       # Use fp16 if bf16 not available
        bf16=is_bfloat16_supported(),           # bf16 preferred on Blackwell/Ampere+
        logging_steps=5,                        # Log every 5 steps
        eval_strategy="steps",                  # Evaluate during training
        eval_steps=50,                          # Evaluate every 50 steps
        save_strategy="steps",
        save_steps=100,                         # Save checkpoint every 100 steps
        save_total_limit=3,                     # Keep only 3 latest checkpoints
        load_best_model_at_end=True,            # Load best model after training
        metric_for_best_model="eval_loss",      # Track validation loss
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
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        packing=False,  # Don't pack multiple examples into one sequence
                        # (resume texts vary too much in length for packing to help)
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
    gguf_files = list(Path(gguf_dir).glob("*.gguf"))
    if gguf_files:
        gguf_path = str(gguf_files[0])
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
    # which is what Ollama expects. The ChatML format (<|im_start|>)
    # matches how we trained the model — this alignment is CRITICAL!

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
        "--model", type=str, default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Base model (default: unsloth/Qwen2.5-7B-Instruct-bnb-4bit). "
             "Options: unsloth/Qwen2.5-7B-Instruct-bnb-4bit (comfortable on 16GB), "
             "unsloth/Qwen2.5-14B-Instruct-bnb-4bit (tight on 16GB), "
             "unsloth/Llama-3.2-8B-Instruct-bnb-4bit (alternative)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=2048,
        help="Max sequence length (default: 2048 — plenty for resumes)"
    )

    # ── Training arguments ────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2 for 16GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
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