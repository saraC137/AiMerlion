# AiMerlion — Training Data & ML Guide

This guide covers how to go from raw resumes (or external datasets) to a trained model you can deploy with Ollama.

---

## Overview

There are two ML models you can train:

| Model | Script | Input | Use |
|-------|--------|-------|-----|
| **LLM (Unsloth)** | `ml/finetune_unsloth.py` | `train_data.jsonl` | Structured JSON extraction from resumes |
| **spaCy NER** | `train_ner.py` | `resume_extractions.db` | Token-level entity recognition |

Both can use your manually annotated resumes **and** external datasets. They complement each other — the LLM handles full extraction, spaCy handles entity tagging.

---

## Part 1 — Your Annotated Data

### Step 1: Annotate resumes

Start the annotation web tool:

```bash
python annotation_tool.py
```

Open `http://localhost:5055` in your browser.

**How to annotate:**
- Drag to select text spans, then press a number key to assign an entity label (1–9)
- Press `H` for the keyboard shortcut reference
- Labels include: `PERSON_NAME`, `JOB_TITLE`, `HARD_SKILL`, `COMPANY_NAME`, `DEGREE`, `INSTITUTION`, `LOCATION`, `CERTIFICATION`, etc.
- Set the document status to **completed** when done — only completed documents go into training

Annotations are saved automatically to `resume_extractions.db`.

---

### Step 2: Prepare training data from your annotations

Convert your completed annotations into training-ready JSONL:

```bash
python data_prep/prepare_training_data.py --db resume_extractions.db
```

This creates:
- `train_data.jsonl` — training split
- `train_data_val.jsonl` — validation split (10%)

Each line is a ShareGPT-format conversation:

```json
{
  "conversations": [
    {"from": "system", "value": "You are a precise resume data extraction assistant..."},
    {"from": "human", "value": "Extract structured data from this resume...\n\nResume:\n[raw text]"},
    {"from": "gpt",   "value": "{\"name\": \"...\", \"hard_skills\": [...], ...}"}
  ]
}
```

The expected JSON schema extracted from each resume:

```json
{
  "name": "string",
  "email": "string",
  "phone": "string",
  "date_of_birth": "string",
  "location": "string",
  "summary": "string",
  "hard_skills": ["array"],
  "soft_skills": ["array"],
  "experience": [{"title": "...", "company": "...", "duration": "..."}],
  "education": [{"degree": "...", "institution": "..."}],
  "certifications": ["array"],
  "languages": ["array"],
  "function": "job function category",
  "industry": "industry category"
}
```

---

### Step 3: Train the LLM

```bash
python ml/finetune_unsloth.py \
  --train-file train_data.jsonl \
  --val-file train_data_val.jsonl \
  --epochs 3
```

This runs 5 phases automatically:
1. Load and validate your JSONL
2. Load base model (Qwen2.5-7B by default) with 4-bit quantization
3. Fine-tune with LoRA adapters
4. Merge LoRA into base model
5. Export to GGUF for Ollama

Output goes to `resume_model_finetuned/`.

---

### Step 4: Train the spaCy NER model (optional, complementary)

```bash
python train_ner.py --db resume_extractions.db --all
```

Requires at least 5 completed documents and 10 examples per entity type. Output goes to `ml_output/`.

---

## Part 2 — External Datasets

Two external datasets are available for supplementing your training data.

---

### Dataset A: Majinuub Resume Parsing (305 international resumes)

**Source:** HuggingFace `Majinuub/Resume_Parsing`
**License:** Apache-2.0 (commercial use OK)
**Coverage:** 40+ countries — useful for general resume diversity, but has **zero Singapore/Malaysia resumes**

**Import and generate JSONL:**

```bash
python import_majinuub_resume_parsing.py
```

This creates:
- `training_data/majinuub_train.jsonl` — 274 training examples
- `training_data/majinuub_val.jsonl` — 31 validation examples

**To merge with your own annotated data:**

```bash
python import_majinuub_resume_parsing.py --merge-with train_data.jsonl
```

This appends the Majinuub examples to your existing `train_data.jsonl` and outputs a merged file.

**Train directly on Majinuub data:**

```bash
python ml/finetune_unsloth.py --train-file training_data/majinuub_train.jsonl
```

**Train on merged data (recommended):**

```bash
python ml/finetune_unsloth.py \
  --train-file training_data/merged_train.jsonl \
  --val-file train_data_val.jsonl \
  --epochs 3
```

> **Note:** Majinuub data supplements your SG/MY annotated resumes — it doesn't replace them. For best accuracy on Singapore/Malaysia resumes, always include your own annotations.

---

### Dataset B: SkillSpan (11,543 skill-tagged sentences)

**Source:** HuggingFace `jjzha/skillspan`
**License:** CC-BY-4.0 (commercial use OK)
**Coverage:** StackOverflow posts + construction job postings, tagged at token level for skills and technical knowledge

SkillSpan is best suited for the **spaCy NER model** (token-level tagging), not the LLM.

**Import into the database (for spaCy training):**

```bash
python import_skillspan.py --db resume_extractions.db
```

**Import as JSONL (for LLM training, experimental):**

```bash
python import_skillspan.py --db resume_extractions.db --jsonl
```

**Then train spaCy NER with SkillSpan included:**

```bash
python train_ner.py --db resume_extractions.db --all
```

---

### Dataset C: Graduate Employment Survey (reference data)

**File:** `GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv`
**Coverage:** NTU, NUS, SIT, SMU, SUSS, SUTD graduate employment rates and salary statistics (2013+)

This dataset is **not used for model training** directly — it's reference data for benchmarking salary and employment fields extracted from resumes against Singapore university norms.

---

## Part 3 — Full Pipeline Cheatsheet

### Quickstart: annotated data only

```bash
# 1. Annotate
python annotation_tool.py
# → http://localhost:5055, mark documents as "completed"

# 2. Prepare training data
python data_prep/prepare_training_data.py --db resume_extractions.db

# 3a. Fine-tune LLM
python ml/finetune_unsloth.py --train-file train_data.jsonl --epochs 3

# 3b. (Optional) Train spaCy NER
python train_ner.py --db resume_extractions.db --all

# 4. Deploy via Ollama
ollama create aimerlion-resume -f resume_model_finetuned/Modelfile
```

---

### Quickstart: external datasets only (no annotations yet)

```bash
# LLM path — Majinuub data
python import_majinuub_resume_parsing.py
python ml/finetune_unsloth.py --train-file training_data/majinuub_train.jsonl

# NER path — SkillSpan data
python import_skillspan.py --db resume_extractions.db
python train_ner.py --db resume_extractions.db --all
```

---

### Quickstart: combined (recommended for best results)

```bash
# 1. Annotate your SG/MY resumes
python annotation_tool.py

# 2. Prepare your annotated data
python data_prep/prepare_training_data.py --db resume_extractions.db

# 3. Import external data and merge
python import_majinuub_resume_parsing.py --merge-with train_data.jsonl
python import_skillspan.py --db resume_extractions.db

# 4. Train LLM on merged data
python ml/finetune_unsloth.py \
  --train-file training_data/merged_train.jsonl \
  --val-file train_data_val.jsonl \
  --epochs 3

# 5. Train spaCy NER (uses both your annotations + SkillSpan)
python train_ner.py --db resume_extractions.db --all

# 6. Deploy
ollama create aimerlion-resume -f resume_model_finetuned/Modelfile
```

---

## Part 4 — Key Options Reference

### `ml/finetune_unsloth.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-file` | `train_data.jsonl` | Training data (JSONL, ShareGPT or Alpaca format) |
| `--val-file` | auto 10% split | Validation data |
| `--model` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Base model. Also try `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `2` | Batch size (2 fits 16GB VRAM) |
| `--gguf` | `q4_k_m` | Quantization: `q4_k_m`, `q5_k_m`, `q8_0`, `f16` |
| `--output-dir` | `resume_model_finetuned` | Where to save the model |
| `--export-only` | — | Skip training, just convert an existing LoRA to GGUF |
| `--lora-path` | — | Path to existing LoRA adapters (use with `--export-only`) |

### `train_ner.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--db` | `resume_extractions.db` | Database with annotations |
| `--all` | — | Run all phases |
| `--status` | `completed` | Which documents to use: `completed`, `in_progress`, `all` |
| `--min-docs` | `5` | Minimum documents required to proceed |
| `--epochs` | `30` | NER training epochs |
| `--output-dir` | `ml_output` | Where to save the spaCy model |
| `--force` | — | Continue even if validation fails |

---

## Part 5 — Tips

**How many annotations do you need?**
- spaCy NER: minimum 5 documents, 10 examples per entity type. 50+ documents gives good results.
- LLM fine-tuning: even 20–30 annotated resumes can meaningfully improve accuracy on SG/MY-specific patterns. More is always better.

**Improving GGUF export on Windows:**
If the GGUF export fails with a llama.cpp build error, download a pre-built `llama-quantize.exe` from the llama.cpp GitHub releases page and place it at:
```
C:\Users\<you>\.unsloth\llama.cpp\build\bin\Release\llama-quantize.exe
```

**Re-exporting an existing trained model without re-training:**
```bash
python ml/finetune_unsloth.py \
  --export-only \
  --lora-path resume_model_finetuned \
  --gguf q4_k_m
```

**Checking annotation progress:**
Open `http://localhost:5055` → the dashboard shows pending / in-progress / completed counts per annotator.
