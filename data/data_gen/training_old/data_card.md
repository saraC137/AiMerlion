# AiMerlion Stage 0 LITE — Training Dataset Card

**Generated:** Stage 0 LITE — synthetic SG/MY resumes
**Total records:** 94
**Training set:** 75
**Validation set:** 19
**Split strategy:** Stratified by ethnicity
**Random seed:** 42

## Format

Alpaca-style JSON array. Each record has:
- `instruction`: SG/MY resume extraction directive (constant across all examples)
- `input`: Raw resume text
- `output`: JSON object with 15 canonical fields

## Distribution

### Ethnicity
- **?**: 94 (100.0%)

### Industry
- **?**: 94 (100.0%)

### Seniority
- **?**: 94 (100.0%)

## How to use in Unsloth GUI

1. Open Unsloth GUI
2. **Base model**: `unsloth/Qwen2.5-7B-Instruct` (or local path)
3. **Dataset**: Load `train_alpaca.json` as Alpaca format
4. **Validation dataset** (optional): Load `val_alpaca.json`
5. **Recommended LoRA settings**:
   - LoRA rank (r): 16
   - LoRA alpha: 32 (2 × rank)
   - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
   - Dropout: 0
6. **Recommended training settings**:
   - Epochs: 3 (small dataset, watch for overfitting)
   - Batch size: 2 per device
   - Gradient accumulation: 4
   - Learning rate: 2e-4
   - Warmup ratio: 0.03
   - Optimizer: adamw_8bit
   - Precision: BF16 (you have RTX 5070 Ti Blackwell)
   - Max sequence length: 4096

## After training

1. Save LoRA adapter to `models/qwen7b_lite_v1/`
2. Run: `python evaluate_lite.py` (next script we'll write)
3. Compare F1 against baseline `ai_extractor.py` on golden test set

## Decision threshold

Per `tasks/todo.md`:
- F1 lift ≥ +5% on SG/MY fields → ✅ Commit to full Stage 0 (1000 resumes)
- F1 lift 0% to +5% → 🤔 Investigate before scaling
- F1 lift < 0% → 🛑 Don't scale, rethink