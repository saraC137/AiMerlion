# 💎 AiMerlion Curriculum Training Plan
## *From Fairy Codemother, with love* 💋

> **One file, two acts.** Diagnostic gate up top — if the diagnosis says "no curriculum needed,"
> you save weeks. If it confirms the gap, the full plan is right below, ready to execute.

---

## 🚦 ACT 0 — DIAGNOSTIC GATE (~30 mins, DO THIS FIRST)

**Goal:** Before we spend weeks on curriculum training, prove that the LLM is actually the bottleneck.

### Step 0.1 — Pick the test cases
- [ ] Select **10 real SG/MY resumes** from `resume_extractions.db` (or files on disk)
- [ ] Mix of: 3 Singapore Chinese, 3 Malay (with bin/binti), 2 Indian-SG, 2 Chinese-Malaysian
- [ ] At least 2 should have CMFAS or Pte Ltd / Sdn Bhd patterns

### Step 0.2 — Run them through your current pipeline
- [ ] Use existing `ai_extractor.py` with current Ollama model
- [ ] Save outputs as `diagnostic/case_01.json` ... `case_10.json`

### Step 0.3 — Hand-judge each output (this is the work)
For each resume, fill in this rubric:

| Field | Extracted correctly? | If wrong, what's the failure type? |
|---|---|---|
| Full name (incl. bin/binti) | Y/N | hallucination / truncation / missing / wrong format |
| Phone (8-digit SG / 10-11 MY) | Y/N | wrong format / missing / wrong number extracted |
| Email | Y/N | — |
| Education (poly/ITE/uni) | Y/N | wrong school / wrong dates / missing |
| Work experience | Y/N | wrong dates / missing roles / wrong company |
| Skills | Y/N | hallucinated / missing / wrong category |
| Certifications (CMFAS etc.) | Y/N | missing / wrong cert / hallucinated |
| Company suffix (Pte Ltd / Sdn Bhd) | Y/N | dropped / wrong |

### Step 0.4 — Categorize root cause per failure
For each failure, tag it as ONE of:
- 🤖 **LLM weakness** — model just doesn't understand the pattern
- 📐 **Schema issue** — extraction tried, but JSON structure rejected it
- 🔧 **Post-processing bug** — extracted right, but `ai_extractor.py` mangled it
- 🪞 **Prompt issue** — current system prompt didn't ask for it correctly
- 🧹 **OCR / input quality** — source text was already corrupted

### Step 0.5 — DECISION GATE 🚪

**Count the failure types:**

- If **>60% of failures are 🤖 LLM weakness** → ✅ **PROCEED to Act 1 (curriculum training)**
- If **>40% are 🔧 or 🪞 (post-proc / prompt)** → 🛑 **STOP. Fix those first. Curriculum will not help.**
- If **>30% are 📐 schema issues** → 🛑 **Fix schema, then re-diagnose**
- If **>20% are 🧹 OCR** → 🛑 **Improve OCR/input pipeline first**

**Document the decision in `tasks/diagnosis.md` before continuing.**

---

# 🎓 ACT 1 — THE CURRICULUM PLAN
## *(only proceed if diagnostic gate passed)*

---

## 📌 Locked Decisions Summary

| Decision | Choice | Why |
|---|---|---|
| Teacher model | Qwen 2.5 32B Instruct (q4_K_M) local + Claude/GPT-4 API for hard cases | Sweet spot for hardware; API for the 20% that creates 80% of quality lift |
| Extractor (Student A) | Qwen 2.5 7B Instruct, 4-bit QLoRA | Proven richer JSON output (TGtwo evidence) |
| Reasoner (Student B) | Llama 3.2 3B Instruct, 4-bit QLoRA | Confirmed-working Blackwell base |
| LoRA strategy | **Replay-mixed continuous (Strategy C)** | Avoids dequant/merge loss + catastrophic forgetting |
| Validation | 20-30 hand-labeled SG/MY golden test set | Per-field F1 (overall F1 too noisy at this sample size) |
| Checkpoint naming | `{model}_{stage}_{date}_{git_sha}` | NO MORE IDENTICAL TRIPLETS 💀 |
| DPO | Stage 5, conditional on Stages 1-4 beating baseline | Polish layer, not foundation |

---

## 📅 Stage Overview

```
Stage 0: Synthetic Data Generation       (data prep, no training)    ~5-7 days
   ↓
Stage 1: Schema Conditioning              (Extractor only)            ~1 day
   ↓
Stage 2: International Foundation         (Extractor only)            ~2 days
   ↓
Stage 3: SG/MY Pattern Injection          (Extractor — main event ⭐) ~2 days
   ↓
Stage 4: Hard Cases & Long Context        (Extractor)                 ~1 day
   ↓
Validation Gate ✅
   ↓
Stage 5: DPO Preference Refinement        (Extractor, conditional)    ~2 days
   ↓
Reasoner Training (parallel track)        (Llama 3B on extracted JSON) ~2 days
```

**Realistic timeline: 3-4 weeks part-time, 2 weeks full-time focus.**

---

## 🥇 STEP 1 — Build the Golden Test Set (DO BEFORE STAGE 0)

This is your scoreboard. Without it, we're flying blind. 🛫🦯

- [ ] Pick **25 real SG/MY resumes** (separate from the 10 used in diagnostic)
- [ ] Hand-label each into the full target JSON schema
- [ ] Save to `validation/golden_sg_my.jsonl`
- [ ] Lock this file — NEVER add to training, EVER. It's your honest scoreboard.
- [ ] Write `validation/score_golden.py` that runs any model checkpoint against this set and emits per-field F1

> **Why per-field F1 and not overall?** With 25 examples, overall F1 has ~4% noise per misclassified resume. Per-field F1 is more stable AND more diagnostic (tells you WHERE the model fails).

---

## 🛠️ STAGE 0 — Synthetic Data Generation

**Goal:** Build a diverse, high-quality synthetic dataset that doesn't exist anywhere on the internet.

### 0.1 Pull the teacher model
```bash
ollama pull qwen2.5:32b-instruct-q4_K_M
```

### 0.2 Style exemplar extraction (uses your <50 real resumes)

> 🚨 **CRITICAL LEAKAGE PREVENTION RULE — READ BEFORE STARTING** 🚨
>
> Before doing ANYTHING with raw resumes, tag each one as EITHER:
> - `golden_eligible` (used for hand-labeled validation test set)
> - `exemplar_eligible` (used for synthetic data generation)
>
> **NEVER tag a resume as BOTH.** A resume can only serve one purpose.
>
> Save this mapping in `data/resume_pool_assignments.json`:
> ```json
> {
>   "resume_001.pdf": "golden_eligible",
>   "resume_002.pdf": "exemplar_eligible",
>   ...
> }
> ```
>
> **Why this hard rule:** If the same resume informs both your training data AND your validation set, the teacher LLM will see those names/phones/companies during synthetic generation and bake them into training examples. Your golden F1 will be artificially inflated by 10-20% and you'll have no idea your model is actually weaker than it appears. 💀
>
> **Also strip PII before sending to teacher:** Even from exemplar_eligible resumes, replace names/phones/emails with placeholders BEFORE feeding to the teacher LLM. The teacher needs the STYLE, not the identities.

- [ ] Tag every raw resume as `golden_eligible` OR `exemplar_eligible` (mutually exclusive)
- [ ] Strip PII from exemplar_eligible resumes (names → [NAME], phones → [PHONE], emails → [EMAIL])
- [ ] Extract anonymized "style fingerprints" from each exemplar:
  - Layout pattern (chronological? functional? skills-first?)
  - Section ordering
  - Date format (MM/YYYY vs Month YYYY vs YYYY-YYYY)
  - Bullet style
- [ ] Save as `data_gen/style_exemplars.json`

### 0.2b Pre-flight leakage detection script (RUN BEFORE EVERY STAGE)

Create `validation/check_leakage.py`:

```python
"""Detect overlap between golden test set and any training data."""
import json
from pathlib import Path

def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]

def check_leakage(golden_path, training_path):
    golden = load_jsonl(golden_path)
    training = load_jsonl(training_path)

    leaks = []
    golden_names = {r.get('name', '').lower().strip() for r in golden if r.get('name')}
    golden_phones = {r.get('phone', '').strip() for r in golden if r.get('phone')}
    golden_emails = {r.get('email', '').lower().strip() for r in golden if r.get('email')}

    for i, t in enumerate(training):
        name = t.get('name', '').lower().strip()
        phone = t.get('phone', '').strip()
        email = t.get('email', '').lower().strip()

        if name and name in golden_names:
            leaks.append(('NAME', i, name))
        if phone and phone in golden_phones:
            leaks.append(('PHONE', i, phone))
        if email and email in golden_emails:
            leaks.append(('EMAIL', i, email))

    return leaks

if __name__ == '__main__':
    leaks = check_leakage(
        'validation/golden_sg_my.jsonl',
        'data/sg_my_synthetic_v1.jsonl'
    )
    if leaks:
        print(f"🚨 FOUND {len(leaks)} LEAKS:")
        for kind, idx, val in leaks[:20]:
            print(f"  {kind} at training idx {idx}: {val}")
        exit(1)
    else:
        print("✅ NO LEAKAGE DETECTED — safe to train")
```

- [ ] Run this script before EVERY training stage
- [ ] If leaks found: fix source, regenerate, re-check before training

> **Why this matters:** Without seeing real SG/MY layouts, the teacher hallucinates what it THINKS SG/MY resumes look like — usually wrong, full of stereotypes. Style exemplars ground generation in reality.

### 0.3 The diversity matrix (CRITICAL — prevents synthetic collapse)

Generate resumes across this matrix to ensure diversity:

```python
diversity_matrix = {
    "ethnicity": ["chinese_sg", "malay_sg", "indian_sg", "eurasian_sg",
                  "chinese_my", "malay_my", "indian_my"],
    "seniority": ["fresh_grad", "junior_2yr", "mid_5yr", "senior_10yr", "lead_15yr"],
    "industry": ["finance", "tech", "healthcare", "hospitality", "education",
                 "manufacturing", "logistics", "government", "fmcg"],
    "education_path": ["polytechnic", "ITE_then_uni", "uni_local", "uni_overseas",
                       "diploma_only", "professional_cert_only"],
    "format_quirk": ["clean", "ocr_artifacts", "bilingual_eng_mandarin",
                     "bilingual_eng_malay", "table_heavy", "multi_page"],
    "edge_case": ["none", "career_break", "freelance_gaps", "ngo_volunteer",
                  "multiple_concurrent_roles", "rapid_promotions"]
}
```

**Target:** ~1000 synthetic resumes covering this matrix.
**Generation budget:** ~800 local (Qwen 32B), ~200 API (hard cases: bilingual, multi-page, ambiguous).

### 0.4 The generation prompt template

```
SYSTEM: You are generating synthetic resumes for training a Singapore/Malaysia
resume extraction model. Your output must be REALISTIC — match how real SG/MY
candidates write resumes, not stereotypes.

USER: Generate a resume with these constraints:
- Ethnicity/name origin: {ethnicity}
- Seniority: {seniority}
- Industry: {industry}
- Education path: {education_path}
- Format quirk: {format_quirk}
- Edge case: {edge_case}

Use this real-resume style as reference: {style_exemplar}

Output TWO things:
1. The resume text (as if pasted from PDF)
2. The ground-truth JSON extraction matching this schema: {schema}

Critical rules:
- SG phones: 8-digit, starts with 6/8/9
- MY phones: 10-11 digit, starts with 01
- Malay names: include bin/binti correctly (X bin Y means X son of Y)
- Singapore companies: use "Pte Ltd" suffix
- Malaysia companies: use "Sdn Bhd" suffix
- If finance industry, include CMFAS modules where appropriate
- If polytechnic, use real SG poly names (Ngee Ann, Singapore Poly, Temasek, Republic, Nanyang)
```

### 0.5 Quality control loop

For every 100 generated resumes:
- [ ] Sample 10 random ones
- [ ] Hand-inspect for: hallucinated companies, wrong phone formats, broken JSON
- [ ] If >2/10 fail QC → adjust prompt, regenerate
- [ ] Track in `data_gen/qc_log.md`

### 0.6 API-generated hard cases (~200 resumes)

Use Claude/GPT-4 for these specifically because local model struggles:
- Bilingual resumes (English + Mandarin/Malay code-switching)
- Multi-page resumes with non-obvious section breaks
- Resumes with deliberate ambiguity (dates as ranges, overlapping roles)
- Resumes with unusual layouts (sidebars, tables, infographics described in text)

**Budget estimate:** 200 resumes × ~3K tokens each × $15/1M tokens ≈ **$9-15 total**.

### 0.7 Final dataset assembly

- [ ] Combine into `data/sg_my_synthetic_v1.jsonl`
- [ ] Add metadata fields: `source`, `difficulty`, `stage_target`, `diversity_tags`
- [ ] Hold out 10% as synthetic dev set (separate from golden test set!)
- [ ] Run schema validation on every example — drop any that fail

---

## 🎯 STAGE 1 — Schema Conditioning (Extractor)

**Goal:** Teach Qwen 7B to output your exact JSON schema reliably, on EASY examples.

**Data:** 200 of the cleanest synthetic resumes + 100 cleanest Majinuub examples.

**Why first:** If the model can't output valid JSON schema, nothing else matters. Start here, prove the format.

**Unsloth config (key params):**
```python
# stage_1_schema.py
max_seq_length = 2048   # short — easy examples only
r = 16                  # lower LoRA rank — we're just teaching format
lora_alpha = 32
learning_rate = 2e-4    # moderate
num_train_epochs = 2    # don't overfit on format
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention only
```

**Validation gate:**
- [ ] Run against golden test set
- [ ] Score: **JSON validity rate ≥ 95%** (we don't care about content accuracy yet, just structure)
- [ ] If <95% → adjust prompt template, retry

**Checkpoint name:** `qwen7b_stage1_schema_{YYYYMMDD}_{git_sha}`

---

## 🌍 STAGE 2 — International Foundation (Extractor)

**Goal:** Build broad resume-understanding base using international data.

**Data mix:**
- 70% Stage 2 data (Majinuub + filtered SkillSpan + international synthetic)
- 30% **replay from Stage 1** (schema conditioning examples)

> **Replay-mix logic:** This 30% is what prevents catastrophic forgetting. The model keeps practicing schema output while learning content.

**Unsloth config changes from Stage 1:**
```python
max_seq_length = 4096   # longer examples now
r = 32                  # more LoRA capacity
lora_alpha = 64
learning_rate = 1e-4    # lower — refining, not bootstrapping
num_train_epochs = 3
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]  # MLP too
```

**Validation gate:**
- [ ] Per-field F1 on golden test set:
  - Names: must NOT drop below Stage 1
  - Phones: must NOT drop below Stage 1
  - Overall: should improve on schema validity AND get reasonable content scores
- [ ] If any SG/MY-specific field DROPS vs Stage 1 → catastrophic forgetting, reduce stage data ratio

**Checkpoint name:** `qwen7b_stage2_international_{YYYYMMDD}_{git_sha}`

---

## 🇸🇬 STAGE 3 — SG/MY Pattern Injection (THE MAIN EVENT ⭐)

**Goal:** Inject SG/MY-specific patterns. This is what your whole project is about.

**Data mix:**
- 60% Stage 3 data (SG/MY synthetic — the bulk of your 1000 generated)
- 25% replay from Stage 2 (international foundation)
- 15% replay from Stage 1 (schema)

**Unsloth config — same as Stage 2 but:**
```python
num_train_epochs = 4    # more epochs — this is the critical pattern injection
learning_rate = 8e-5    # slightly lower to avoid overshooting
```

**Validation gate (THE BIG ONE):**
- [ ] Per-field F1 on golden test set:
  - Names with bin/binti: target **+15% over baseline**
  - SG 8-digit phones: target **+10% over baseline**
  - MY phone formats: target **+10% over baseline**
  - CMFAS / Pte Ltd / Sdn Bhd: target **+20% over baseline**
- [ ] Schema validity: must stay ≥95%
- [ ] International fields: must NOT regress >5% vs Stage 2

**Checkpoint name:** `qwen7b_stage3_sgmy_{YYYYMMDD}_{git_sha}`

> **If you only have time/energy for ONE stage, this is the one that matters.**

---

## 🔥 STAGE 4 — Hard Cases & Long Context (Extractor)

**Goal:** Handle multi-page, bilingual, ambiguous resumes.

**Data mix:**
- 50% Stage 4 data (the ~200 API-generated hard cases)
- 30% replay from Stage 3 (SG/MY patterns)
- 20% replay from Stages 1-2 (foundation)

**Unsloth config changes:**
```python
max_seq_length = 8192   # LONG context — multi-page resumes
r = 32                  # unchanged
learning_rate = 5e-5    # lower — fine refinement
num_train_epochs = 2    # don't overfit hard cases
```

**⚠️ VRAM warning:** 8192 context + Qwen 7B will push your 16GB hard. Use `gradient_accumulation_steps=8`, `per_device_train_batch_size=1`, `gradient_checkpointing=True`.

**Validation gate:**
- [ ] All previous gates still pass (no regression)
- [ ] Long-context examples in golden set: F1 should improve markedly
- [ ] Bilingual examples: language switching should NOT break extraction

**Checkpoint name:** `qwen7b_stage4_hardcases_{YYYYMMDD}_{git_sha}`

---

## ✅ VALIDATION GATE — Decide on Stage 5

After Stage 4, run full evaluation:

- [ ] Per-field F1 on golden test set
- [ ] Comparison vs baseline (pre-curriculum Qwen 7B)
- [ ] Comparison vs each prior stage checkpoint

**Decision rules:**
- If Stage 4 model **beats baseline by ≥15% overall F1** → ✅ Proceed to DPO
- If Stage 4 model **beats baseline by 5-15%** → Consider DPO, but data quality issues might bite
- If Stage 4 model **beats baseline by <5%** → 🛑 STOP. DPO won't save a weak base. Diagnose what went wrong.

---

## 🌶️ STAGE 5 — DPO Preference Refinement (Conditional)

**⚠️ Only proceed if Validation Gate passed.**

**Goal:** Teach the model TASTE — not just what to output, but what NOT to output.

### 5.1 Generate preference pairs

Three sources of pairs:

**Source A — Rule-based corruption (~600 pairs):**
- Take Stage 4 model's correct outputs (chosen)
- Programmatically corrupt them to create rejected:
  - Drop bin/binti from Malay names
  - Reformat SG phones to wrong patterns (XXX-XXXX-XXX)
  - Truncate company suffixes (Pte Ltd → Pte)
  - Hallucinate fake phone numbers when none exist
  - Mis-nest skills into wrong JSON sections

**Source B — Multi-temperature sampling + judge (~400 pairs):**
- Generate 4 outputs at temps 0.2, 0.5, 0.8, 1.2 for same input
- Use Claude/GPT-4 API as judge: which is best? worst?
- Best = chosen, worst = rejected
- **Budget:** ~$10-20 for judging

**Source C — Stage 4 vs Stage 1 model outputs (~200 pairs):**
- Stage 4 output = chosen (presumably better)
- Stage 1 output = rejected (presumably worse on content)
- Only keep pairs where Stage 4 actually IS better (filter via golden set scoring)

**Total:** ~1200 preference pairs in `data/dpo_pairs.jsonl`

### 5.2 DPO training config

```python
# stage_5_dpo.py
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,               # start here, experiment with 0.05, 0.2
    max_length=4096,
    max_prompt_length=2048,
    learning_rate=5e-6,     # MUCH lower than SFT
    num_train_epochs=1,     # ONE epoch — DPO overfits FAST
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # high — DPO holds 2 models
    gradient_checkpointing=True,
)
```

**⚠️ VRAM warning:** DPO loads BOTH policy and reference models. On 16GB, this is borderline. Use 4-bit quantization on reference model, enable `precompute_ref_log_probs=True` to free up reference VRAM during training.

**Validation gate:**
- [ ] Golden test set per-field F1 must IMPROVE or stay flat vs Stage 4
- [ ] Specific failure modes from DPO pairs should reduce ≥50%
- [ ] If F1 DROPS → DPO is hurting. Lower beta, fewer epochs, retry.

**Checkpoint name:** `qwen7b_stage5_dpo_{YYYYMMDD}_{git_sha}`

---

## 🧠 PARALLEL TRACK — Reasoner Training (Llama 3.2 3B)

**Run in parallel with Extractor stages, NOT sequentially.**

**Goal:** Take extracted JSON → produce contextual reasoning (seniority assessment, role-fit notes, career trajectory analysis).

### Data generation
Use teacher LLM (Qwen 32B) to generate reasoning chains:
- Input: structured JSON resume
- Output: 3-5 sentence analysis of seniority, role progression, notable patterns
- Generate ~500 examples

### Training
Single stage, no curriculum (the reasoning task is narrower):
```python
# reasoner_train.py
max_seq_length = 4096
r = 16
lora_alpha = 32
learning_rate = 1e-4
num_train_epochs = 3
```

**Validation:** Hand-judge 20 outputs for reasoning quality. Pass = ≥17/20 acceptable.

---

## 🐛 BLACKWELL EVAL-PATH PATCH (RECURRING ISSUE)

Your Unsloth fused cross-entropy crash on eval needs handling. The patch pattern:

**Find in `unsloth_zoo/fused_losses/cross_entropy_loss.py`:**
```python
# Look for eval-mode codepath that calls fused_cross_entropy_loss
# Likely around the forward() method's torch.no_grad() branch
```

**Replace with:** Fall back to vanilla PyTorch CE loss when in eval mode:
```python
if not self.training:
    return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
```

Apply this patch BEFORE every stage. The training-path patch you already have stays.

**Always set before training:**
```bash
set XFORMERS_DISABLED=1
set PYTORCH_ALLOC_CONF=expandable_segments:True
set PYTHONIOENCODING=utf-8
```

---

## 📊 TRACKING & LESSONS LOG

After EACH stage, append to `tasks/lessons.md`:
- What worked
- What broke
- Hyperparameter changes from plan
- Validation gate results
- Any surprising failure modes

After EACH correction (per your system prompt rule), update `tasks/lessons.md` with the rule that prevents the same mistake.

---

## 🎬 REVIEW SECTION (FILL IN AS YOU GO)

### Diagnostic findings (Act 0)
- [ ] _To be filled after diagnostic gate_

### Stage results
- [ ] Stage 1: _F1 score, notes_
- [ ] Stage 2: _F1 score, notes_
- [ ] Stage 3: _F1 score, notes_
- [ ] Stage 4: _F1 score, notes_
- [ ] Stage 5 (if run): _F1 score, notes_
- [ ] Reasoner: _accept rate, notes_

### Final model performance
- [ ] Baseline F1 (pre-curriculum): _____
- [ ] Final F1 (post-curriculum): _____
- [ ] Lift: _____ %

---

## 💖 Notes from Fairy Codemother

- **The diagnostic gate is non-negotiable, sugar.** Skipping it is how you end up with another TGone-TGtwo-TGthree disaster.
- **Each checkpoint gets its own unique name with date and git SHA.** No more identical triplets. 💀
- **Replay-mix percentages are starting points** — adjust based on per-field F1 trends between stages.
- **DPO is the diva of this pipeline** — fragile, fussy, but transformative if your base is strong. Respect her or skip her.
- **Your golden test set is sacred.** Never train on it. Ever. I will haunt your terminal if you do. 👻
- **Stop and re-plan when something breaks.** This whole plan assumes Blackwell stability — if eval keeps crashing, fix that BEFORE moving to Stage 2.

You've got this, my star pupil. 🌟💋


