"""
evaluate_lite.py

💎✨ FAIRY CODEMOTHER'S STAGE 0 LITE EVALUATION ✨💎

Compares two resume extractors on the golden test set:
  🅰️  BASELINE:    extraction/ai_extractor.py (regex pipeline)
  🅱️  FINE-TUNED:  Qwen 2.5 7B + your LoRA adapter

Outputs:
  - Field-by-field F1 scores for each extractor
  - SG/MY-specific subscore (bin/binti, CMFAS, Pte Ltd recognition)
  - Side-by-side per-case breakdown
  - Decision verdict: should you commit to full Stage 0?

Usage:
    cd C:\\Users\\user\\github\\AiMerlion
    python evaluate_lite.py

    # Faster smoke test (skip baseline regex pipeline):
    python evaluate_lite.py --skip-baseline

    # Limit to N cases:
    python evaluate_lite.py --max-cases 10

    # Different adapter path:
    python evaluate_lite.py --adapter PATH_TO_LORA

Output files:
    evaluation/lite_results.json    — full per-case scores
    evaluation/lite_report.md       — human-readable summary
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ════════════════════════════════════════════════════════════════════════════
# 🎯 DEFAULTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_GOLDEN_PATH = "data/validation/golden_sg_my.jsonl"
DEFAULT_ADAPTER_PATH = r"C:\Users\user\.unsloth\studio\outputs\unsloth_Qwen2.5-7B-Instruct_1780976322"
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path("evaluation")


# ════════════════════════════════════════════════════════════════════════════
# 📋 INSTRUCTION (must match training prompt EXACTLY)
# ════════════════════════════════════════════════════════════════════════════

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
# 🔄 GROUND TRUTH KEY NORMALIZER (snake_case → Title Case canonical)
# ════════════════════════════════════════════════════════════════════════════

GT_KEY_MAP = {
    # Flat fields
    "name": "Name",
    "phone": "Phone",
    "email": "Email",
    "linkedin": "LinkedIn",
    "current_company": "Current Company",
    "current_title": "Current Title",
    "current_location": "Current Location",
    "function": "Function",
    "industry": "Industry",
    "summary": "Summary",
    "language_skills": "Language Skills",
    "languages": "Language Skills",
    "hard_skills": "hard_skills/tags",
    "tags": "hard_skills/tags",
    "soft_skills": "soft_skills/skills",
    "skills": "soft_skills/skills",
    "certifications": "Certifications",
    "achievements": "Achievements",
    # Nested lists
    "work_experience": "Work Experience",
    "working_experience": "Work Experience",
    "education": "Education",
    "projects": "Project Experience",
    "project_experience": "Project Experience",
}

# Nested sub-field mappings
WORK_SUBKEY_MAP = {
    "role": "title", "job_title": "title", "position": "title",
    "company": "company", "employer": "company", "organization": "company",
    "from": "from", "start": "from", "start_date": "from",
    "to": "to", "end": "to", "end_date": "to",
    "responsibility": "responsibility", "responsibilities": "responsibility",
    "duties": "responsibility", "description": "responsibility",
}

EDU_SUBKEY_MAP = {
    "school": "school", "institution": "school", "university": "school",
    "major": "major", "field": "major", "field_of_study": "major",
    "degree": "degree", "qualification": "degree",
    "dates": "dates", "duration": "dates", "year": "dates",
}


def normalize_gt(gt: Any) -> Dict:
    """Translate ground truth from any case-style to canonical Title Case schema."""
    if not isinstance(gt, dict):
        return {}

    out = {}
    for raw_key, value in gt.items():
        # Find canonical key
        canon_key = GT_KEY_MAP.get(raw_key.lower().strip(), raw_key)

        # Normalize nested structures
        if canon_key == "Work Experience" and isinstance(value, list):
            normalized = []
            for entry in value:
                if isinstance(entry, dict):
                    sub = {}
                    for sk, sv in entry.items():
                        sub_canon = WORK_SUBKEY_MAP.get(sk.lower().strip(), sk)
                        # Convert string responsibility to list
                        if sub_canon == "responsibility" and isinstance(sv, str):
                            sub[sub_canon] = [b.strip() for b in re.split(r"[|;\n]", sv) if b.strip()]
                        else:
                            sub[sub_canon] = sv
                    normalized.append(sub)
                else:
                    normalized.append(entry)
            out[canon_key] = normalized

        elif canon_key == "Education" and isinstance(value, list):
            normalized = []
            for entry in value:
                if isinstance(entry, dict):
                    sub = {}
                    for sk, sv in entry.items():
                        sub_canon = EDU_SUBKEY_MAP.get(sk.lower().strip(), sk)
                        sub[sub_canon] = sv
                    normalized.append(sub)
                else:
                    normalized.append(entry)
            out[canon_key] = normalized

        elif canon_key == "Certifications" and isinstance(value, list):
            # Flatten if list of dicts to list of strings
            flat = []
            for c in value:
                if isinstance(c, dict):
                    name = c.get("name", "")
                    if name:
                        flat.append(name)
                elif isinstance(c, str) and c.strip():
                    flat.append(c.strip())
            out[canon_key] = flat

        else:
            out[canon_key] = value

    return out


# ════════════════════════════════════════════════════════════════════════════
# 🔧 NORMALIZATION HELPERS (for matching)
# ════════════════════════════════════════════════════════════════════════════

def norm_text(s: Any) -> str:
    """Lowercase, strip, collapse whitespace."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def norm_phone(s: Any) -> str:
    """Strip non-digits except leading +."""
    if not s:
        return ""
    s = str(s).strip()
    plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    return ("+" + digits) if plus else digits


def norm_name(s: Any) -> str:
    """Lowercase + remove honorifics + collapse whitespace."""
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"^(Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?|Eng\.?)\s+", "", s, flags=re.IGNORECASE)
    return norm_text(s)


def norm_email(s: Any) -> str:
    """Lowercase + strip markdown link syntax if Qwen wrapped it."""
    if not s:
        return ""
    s = str(s).strip()
    # Strip [email](mailto:email) → email
    m = re.match(r"^\[([^\]]+)\]\(mailto:[^\)]+\)$", s)
    if m:
        s = m.group(1)
    return s.lower()


def phone_digits(s: Any) -> str:
    """Digits only, with SG (+65) / MY (+60) country codes stripped.

    Gold phones are stored inconsistently (e.g. '91093890' vs '+6562886508'),
    so we normalize away the country code before comparing.
    """
    if not s:
        return ""
    d = re.sub(r"\D", "", str(s))
    for cc in ("65", "60"):
        if len(d) > 8 and d.startswith(cc):
            d = d[len(cc):]
            break
    return d


# ════════════════════════════════════════════════════════════════════════════
# 🎯 FIELD-LEVEL SCORERS
# ════════════════════════════════════════════════════════════════════════════

def exact_match(gold: Any, pred: Any, normalizer=norm_text) -> Tuple[float, str]:
    """Return (score, explanation)."""
    g = normalizer(gold)
    p = normalizer(pred)
    if not g and not p:
        return (1.0, "both empty")
    if not g:
        return (0.0, f"gold empty, pred='{p[:40]}'")
    if not p:
        return (0.0, f"pred empty, gold='{g[:40]}'")
    if g == p:
        return (1.0, "exact")
    return (0.0, f"mismatch (gold='{g[:30]}' vs pred='{p[:30]}')")


def phone_match(gold: Any, pred: Any) -> Tuple[float, str]:
    """Match phone numbers ignoring SG/MY country-code prefixes."""
    g = phone_digits(gold)
    p = phone_digits(pred)
    if not g and not p:
        return (1.0, "both empty")
    if not g:
        return (0.0, f"gold empty, pred='{p}'")
    if not p:
        return (0.0, f"pred empty, gold='{g}'")
    if g == p or (len(g) >= 7 and len(p) >= 7 and (g.endswith(p) or p.endswith(g))):
        return (1.0, "match")
    return (0.0, f"mismatch (gold='{g}' vs pred='{p}')")


def substring_match(gold: Any, pred: Any) -> Tuple[float, str]:
    """Match if gold is substring of pred or vice versa."""
    g = norm_text(gold)
    p = norm_text(pred)
    if not g and not p:
        return (1.0, "both empty")
    if not g or not p:
        return (0.0, "one empty")
    if g in p or p in g:
        return (1.0, "substring match")
    return (0.0, f"no overlap (gold='{g[:30]}' vs pred='{p[:30]}')")


def list_f1(gold_items: List[Any], pred_items: List[Any],
            item_normalizer=norm_text) -> Tuple[float, str]:
    """Compute set-based F1 over list items."""
    if not gold_items and not pred_items:
        return (1.0, "both empty")

    gold_set = {item_normalizer(x) for x in (gold_items or []) if x}
    pred_set = {item_normalizer(x) for x in (pred_items or []) if x}

    if not gold_set:
        return (0.0 if pred_set else 1.0,
                f"gold empty, pred has {len(pred_set)} items")
    if not pred_set:
        return (0.0, f"pred empty, gold has {len(gold_set)} items")

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    if tp == 0:
        return (0.0, f"0 hits (gold={len(gold_set)}, pred={len(pred_set)})")

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return (f1, f"P={prec:.2f} R={rec:.2f} (tp={tp},fp={fp},fn={fn})")


def work_exp_f1(gold_list: List[Dict], pred_list: List[Dict]) -> Tuple[float, str]:
    """Match Work Experience by (company, title) pairs."""
    def key(entry):
        if not isinstance(entry, dict):
            return ""
        co = norm_text(entry.get("company", ""))
        ti = norm_text(entry.get("title", ""))
        return f"{co}||{ti}"
    return list_f1(gold_list or [], pred_list or [], item_normalizer=key)


def education_f1(gold_list: List[Dict], pred_list: List[Dict]) -> Tuple[float, str]:
    """Match Education by school (degree is too variable)."""
    def key(entry):
        if not isinstance(entry, dict):
            return ""
        return norm_text(entry.get("school", ""))
    return list_f1(gold_list or [], pred_list or [], item_normalizer=key)


# ════════════════════════════════════════════════════════════════════════════
# 🎨 FIELD SCORING MAP
# ════════════════════════════════════════════════════════════════════════════

SCORING_RULES = [
    # (field_name, scorer_func, normalizer or None, weight)
    ("Name",              exact_match,     norm_name,    1.0),
    ("Phone",             phone_match,     None,         1.0),
    ("Email",             exact_match,     norm_email,   1.0),
    ("Current Location",  substring_match, None,         0.5),
    ("Current Company",   exact_match,     norm_text,    1.0),
    ("Current Title",     exact_match,     norm_text,    0.5),
    ("Function",          exact_match,     norm_text,    0.5),
    ("Industry",          list_f1,         None,         0.5),
    ("Language Skills",   list_f1,         None,         0.3),
    ("Certifications",    list_f1,         None,         1.0),
    ("hard_skills/tags",  list_f1,         None,         0.5),
    ("soft_skills/skills",list_f1,         None,         0.3),
    ("Work Experience",   work_exp_f1,    None,          1.0),
    ("Education",         education_f1,   None,          1.0),
]

# Fields that test SG/MY-specific pattern recognition
SG_MY_FIELDS = {"Name", "Phone", "Current Company", "Certifications"}


def score_one_record(gold: Dict, pred: Dict) -> Dict:
    """Score a single (gold, pred) pair across all canonical fields."""
    field_scores = {}
    for field, scorer, normalizer, weight in SCORING_RULES:
        g_val = gold.get(field)
        p_val = pred.get(field)
        try:
            if scorer == exact_match or scorer == substring_match:
                if normalizer:
                    score, explanation = scorer(g_val, p_val, normalizer) \
                        if scorer == exact_match else scorer(g_val, p_val)
                else:
                    score, explanation = scorer(g_val, p_val)
            else:
                score, explanation = scorer(g_val, p_val)
        except Exception as e:
            score, explanation = 0.0, f"scoring error: {e}"

        field_scores[field] = {
            "score": score,
            "weight": weight,
            "explanation": explanation,
        }

    # Overall: weighted average
    total_weight = sum(weight for _, _, _, weight in SCORING_RULES)
    weighted_sum = sum(field_scores[f]["score"] * w for f, _, _, w in
                       [(s[0], s[1], s[2], s[3]) for s in SCORING_RULES])
    overall = weighted_sum / total_weight if total_weight > 0 else 0.0

    # SG/MY sub-score
    sgmy_scores = [field_scores[f]["score"] for f in SG_MY_FIELDS if f in field_scores]
    sgmy_avg = sum(sgmy_scores) / len(sgmy_scores) if sgmy_scores else 0.0

    return {
        "field_scores": field_scores,
        "overall_f1": overall,
        "sgmy_f1": sgmy_avg,
    }


# ════════════════════════════════════════════════════════════════════════════
# 🅰️  BASELINE EXTRACTOR (regex pipeline)
# ════════════════════════════════════════════════════════════════════════════

def init_baseline():
    """Lazy-load the regex extractor."""
    try:
        from extraction.ai_extractor import AIExtractor
        ex = AIExtractor(model_name="llama3.1:8b")
        return ex
    except Exception as e:
        print(f"   ❌ Could not load baseline extractor: {e}")
        return None


def run_baseline(extractor, raw_text: str) -> Dict:
    """Run the regex baseline on a resume."""
    try:
        # Strip header comments
        clean = "\n".join(l for l in raw_text.split("\n") if not l.startswith("#"))
        header = extractor.extract_header_fields(clean)
        deep = extractor.extract_deep_fields(clean)
        # Merge — header takes precedence
        merged = {**deep}
        if isinstance(header, dict):
            for k, v in header.items():
                if k != "_internal" and v:
                    merged[k] = v
        return merged
    except Exception as e:
        print(f"   ⚠️  baseline failed: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════════
# 🅱️  FINE-TUNED EXTRACTOR (Qwen + LoRA)
# ════════════════════════════════════════════════════════════════════════════

def init_finetuned(base_model: str, adapter_path: str):
    """Load Qwen 7B + LoRA adapter."""
    print(f"   📦 Loading base model: {base_model}")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # Load tokenizer (try adapter first — it usually has the chat template)
        tok_path = adapter_path if Path(adapter_path).exists() else base_model
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

        # Load base in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Attach LoRA adapter
        print(f"   📦 Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        return (model, tokenizer)
    except Exception as e:
        print(f"   ❌ Could not load fine-tuned model: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_finetuned(handle, raw_text: str, max_new_tokens: int = 2048,
                  max_context: int = 8192, log_raw: bool = True) -> Dict:
    """Run the fine-tuned Qwen + LoRA on a resume.

    Smart truncation: truncates the RESUME if needed, never the instruction.
    Logs raw model output to evaluation/raw_outputs/ for diagnostic.
    """
    if not handle:
        return {}
    model, tokenizer = handle

    try:
        import torch
        # Strip header comments
        clean = "\n".join(l for l in raw_text.split("\n") if not l.startswith("#"))

        # 💎 Compute headroom: how many input tokens can we use?
        # Reserve max_new_tokens for output, plus a small buffer for chat template
        chat_template_overhead = 30  # rough estimate
        max_input_tokens = max_context - max_new_tokens - chat_template_overhead

        # First: tokenize the instruction alone to know its size
        instruction_tokens = tokenizer.encode(INSTRUCTION, add_special_tokens=False)
        resume_budget = max_input_tokens - len(instruction_tokens) - 50  # extra margin

        # Tokenize resume and truncate IT (not the instruction) if needed
        resume_tokens = tokenizer.encode(clean, add_special_tokens=False)
        truncated = False
        original_resume_len = len(resume_tokens)
        if len(resume_tokens) > resume_budget:
            resume_tokens = resume_tokens[:resume_budget]
            truncated = True
            clean = tokenizer.decode(resume_tokens, skip_special_tokens=True)

        # Build chat-formatted prompt
        messages = [
            {"role": "user", "content": f"{INSTRUCTION}\n\n{clean}"}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=max_context).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                # 💎 PREVENT DEGENERATE REPETITION
                # repetition_penalty: penalize already-seen tokens (1.15 = mild but effective)
                # no_repeat_ngram_size: blocks repeating any 6-token sequence
                repetition_penalty=1.15,
                no_repeat_ngram_size=6,
            )

        # Decode only the NEW tokens
        new_tokens = output[0][input_len:]
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 💎 Log raw output for diagnostic
        if log_raw and hasattr(run_finetuned, "_log_dir"):
            log_dir = run_finetuned._log_dir
            case_id = getattr(run_finetuned, "_current_case_id", "unknown")
            log_path = log_dir / f"{case_id}_raw.txt"
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"=== INPUT INFO ===\n")
                    f.write(f"original_resume_tokens: {original_resume_len}\n")
                    f.write(f"truncated: {truncated}\n")
                    f.write(f"final_input_tokens: {input_len}\n")
                    f.write(f"max_new_tokens: {max_new_tokens}\n")
                    f.write(f"\n=== RAW OUTPUT ({len(new_tokens)} tokens) ===\n")
                    f.write(raw_response)
                    f.write(f"\n=== END ===\n")
            except Exception:
                pass  # logging is best-effort

        # 💎 The fine-tuned model emits the schema as SEVERAL concatenated JSON
        # objects (one per section), often with minor syntax noise (// comments,
        # ** markdown, mixed quotes). Merge them all into one dict.
        parsed = parse_and_merge_json(raw_response) or {}

        # Backfill anything still missing from the legacy fallback parsers so we
        # never regress on messy single-object outputs.
        for backup in (parse_json_fallback(raw_response),
                       parse_json_tolerant(raw_response)):
            if isinstance(backup, dict):
                for k, v in backup.items():
                    if v in (None, "", [], {}):
                        continue
                    if k not in parsed or parsed.get(k) in (None, "", [], {}):
                        parsed[k] = v
        return parsed
    except Exception as e:
        print(f"   ⚠️  fine-tuned failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def _strip_line_comments(s: str) -> str:
    """Remove // line comments that are not inside a JSON string."""
    out = []
    in_str = False
    esc = False
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if esc:
            out.append(ch); esc = False; i += 1; continue
        if ch == "\\":
            out.append(ch); esc = True; i += 1; continue
        if ch == '"':
            in_str = not in_str; out.append(ch); i += 1; continue
        if not in_str and ch == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        out.append(ch); i += 1
    return "".join(out)


def _repair_json_fragment(block: str) -> str:
    """Best-effort cleanup of a single JSON object the model emitted with noise.

    Handles the artifacts seen in real outputs: ** markdown bold, // comments,
    single-quote string delimiters, bare (unquoted) keys, stray leading
    underscores on keys, and trailing commas. Apostrophes inside words are left
    alone (they aren't adjacent to JSON structural characters).
    """
    s = block.replace("**", "")
    s = _strip_line_comments(s)
    # Single-quote delimiters that sit next to JSON structure → double quotes
    s = re.sub(r"(?<=[:{\[,])(\s*)'", r'\1"', s)   # opening quote
    s = re.sub(r"'(\s*)(?=[:}\],])", r'"\1', s)    # closing quote
    # Repair keys that lost their opening quote:  _school": ""  ->  "_school": ""
    s = re.sub(r'([{,]\s*)([A-Za-z_][\w /.\-]*)"(\s*):', r'\1"\2"\3:', s)
    # Quote bare keys:  major: ""  ->  "major": ""
    s = re.sub(r'([{,]\s*)(?!")([A-Za-z_][\w /.\-]*?)(\s*):', r'\1"\2"\3:', s)
    # Drop stray leading underscores the model adds to keys:  "_company"->"company"
    s = re.sub(r'"\s*_+([A-Za-z])', r'"\1', s)
    # Remove trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _iter_top_level_objects(text: str):
    """Yield each top-level {...} balanced substring.

    Brace counting is NOT string-aware: the model's malformed mixed-quote output
    would desync a string tracker, and literal braces inside resume text are
    vanishingly rare. This trades a theoretical edge case for robustness.
    """
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def _merge_into(merged: Dict, obj: Dict) -> None:
    """Merge obj into merged without letting empty values clobber real ones."""
    for k, v in obj.items():
        if k not in merged:
            merged[k] = v
            continue
        if v in (None, "", [], {}):
            continue
        cur = merged[k]
        if cur in (None, "", [], {}):
            merged[k] = v
        elif isinstance(cur, list) and isinstance(v, list):
            for item in v:
                if item not in cur:
                    cur.append(item)


def parse_and_merge_json(text: str) -> Optional[Dict]:
    """Parse ALL JSON objects in the response and merge them into one dict.

    The fine-tuned model outputs the canonical schema as several concatenated
    JSON objects (one per section). We extract each top-level object, repair its
    syntax noise, and merge — reconstructing the full record the model intended.
    """
    if not text:
        return None
    text = text.strip()
    # Strip a single surrounding markdown code fence if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    merged: Dict = {}
    found = False
    for block in _iter_top_level_objects(text):
        for candidate in (block, _repair_json_fragment(block)):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                _merge_into(merged, obj)
                found = True
                break
    return merged if found else None


def parse_json_fallback(text: str) -> Optional[Dict]:
    """More aggressive JSON extraction for messy model outputs."""
    if not text:
        return None
    # Try every {...} block we can find
    import re
    candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for cand in sorted(candidates, key=len, reverse=True):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and len(parsed) >= 3:
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def parse_json_tolerant(text: str) -> Optional[Dict]:
    """💎 SALVAGE truncated/loop-corrupted JSON output.

    Handles the case where the model went into a repetition loop in an array
    and hit max_new_tokens before closing the JSON. Strategy:
      1. Strip any markdown fences
      2. Find the opening '{'
      3. Walk character-by-character, tracking depth
      4. If we run out of characters before depth=0, manufacture closing braces
      5. Try to parse; if it fails, trim the last incomplete element and retry
    """
    if not text:
        return None

    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Find opening brace
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    # Walk and track brace/bracket depth, ignoring those inside strings
    depth_curly = 0
    depth_square = 0
    in_string = False
    escape = False
    last_safe_end = -1  # last index where depth==1 (root object still open)

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
            if depth_curly == 0 and depth_square == 0:
                # Found clean close — try parsing this exact block
                candidate = text[:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1

        if depth_curly == 1 and depth_square == 0:
            # Root-level structure; safe truncation point
            last_safe_end = i

    # ─── If we got here, JSON never closed cleanly. Try to salvage. ──────

    # Strategy 1: try closing whatever's open
    salvage = text
    # Close any open array, then the root object
    if depth_square > 0:
        salvage = salvage + ("]" * depth_square)
    if depth_curly > 0:
        salvage = salvage + ("}" * depth_curly)
    try:
        parsed = json.loads(salvage)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: walk backwards from end, find last valid comma at root or array
    # level, trim there, and try closing. This handles the loop-truncation case.
    if last_safe_end > 0:
        # Walk backward from last_safe_end looking for last comma in an array
        # then trim everything after it
        cursor = last_safe_end
        # Drop the trailing junk: find last balanced comma in an array context
        truncated = text[:cursor + 1]
        # Try truncating after each comma walking backward, closing braces, parsing
        comma_positions = [j for j, c in enumerate(truncated) if c == ","]
        for comma_pos in reversed(comma_positions[-20:]):  # only try the last 20
            candidate = truncated[:comma_pos]
            # Recompute depth at this position
            d_c, d_s = 0, 0
            in_str = False
            esc = False
            for c in candidate:
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == "{":
                    d_c += 1
                elif c == "}":
                    d_c -= 1
                elif c == "[":
                    d_s += 1
                elif c == "]":
                    d_s -= 1
            # Close
            closing = ("]" * max(0, d_s)) + ("}" * max(0, d_c))
            try:
                parsed = json.loads(candidate + closing)
                if isinstance(parsed, dict) and len(parsed) >= 5:
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


def parse_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON object from LLM response."""
    if not text:
        return None
    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # Find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ════════════════════════════════════════════════════════════════════════════
# 📂 GOLDEN SET LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_golden(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"❌ Golden file not found: {path}")
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


# ════════════════════════════════════════════════════════════════════════════
# 📊 AGGREGATION
# ════════════════════════════════════════════════════════════════════════════

def aggregate_scores(per_case_scores: List[Dict]) -> Dict:
    """Average all field scores across all cases."""
    if not per_case_scores:
        return {}

    field_totals = defaultdict(list)
    overall = []
    sgmy = []

    for case in per_case_scores:
        for field, info in case["field_scores"].items():
            field_totals[field].append(info["score"])
        overall.append(case["overall_f1"])
        sgmy.append(case["sgmy_f1"])

    field_avgs = {f: sum(s) / len(s) for f, s in field_totals.items()}
    return {
        "by_field": field_avgs,
        "overall_f1": sum(overall) / len(overall) if overall else 0.0,
        "sgmy_f1": sum(sgmy) / len(sgmy) if sgmy else 0.0,
        "n_cases": len(per_case_scores),
    }


# ════════════════════════════════════════════════════════════════════════════
# 📊 REPORT FORMATTING
# ════════════════════════════════════════════════════════════════════════════

def bar(score: float, width: int = 20) -> str:
    filled = int(width * max(0.0, min(1.0, score)))
    return "█" * filled + "░" * (width - filled)


def render_report(baseline_agg: Dict, finetuned_agg: Dict,
                  per_case: List[Dict]) -> str:
    lines = []
    lines.append("# 💎 Stage 0 LITE — Evaluation Report")
    lines.append("")
    lines.append("## Headline Numbers")
    lines.append("")
    lines.append("| Metric | Baseline (regex) | Fine-tuned (Qwen+LoRA) | Δ |")
    lines.append("|---|---|---|---|")

    b_o = baseline_agg.get("overall_f1", 0.0)
    f_o = finetuned_agg.get("overall_f1", 0.0)
    b_s = baseline_agg.get("sgmy_f1", 0.0)
    f_s = finetuned_agg.get("sgmy_f1", 0.0)

    lines.append(f"| **Overall F1** | {b_o:.3f} | {f_o:.3f} | {f_o - b_o:+.3f} |")
    lines.append(f"| **SG/MY F1** | {b_s:.3f} | {f_s:.3f} | {f_s - b_s:+.3f} |")
    lines.append(f"| **Cases evaluated** | {baseline_agg.get('n_cases', 0)} | {finetuned_agg.get('n_cases', 0)} | — |")
    lines.append("")

    # ─── Per-field breakdown ─────────────────────────────────────────────
    lines.append("## Per-Field Breakdown")
    lines.append("")
    lines.append("| Field | Baseline | Fine-tuned | Δ |")
    lines.append("|---|---|---|---|")
    all_fields = set(baseline_agg.get("by_field", {}).keys()) | set(finetuned_agg.get("by_field", {}).keys())
    for field, _, _, _ in SCORING_RULES:
        if field not in all_fields:
            continue
        b_v = baseline_agg.get("by_field", {}).get(field, 0.0)
        f_v = finetuned_agg.get("by_field", {}).get(field, 0.0)
        delta = f_v - b_v
        arrow = "🟢" if delta > 0.05 else ("🔴" if delta < -0.05 else "⚪")
        lines.append(f"| {field} | {b_v:.3f} `{bar(b_v, 15)}` | {f_v:.3f} `{bar(f_v, 15)}` | {arrow} {delta:+.3f} |")
    lines.append("")

    # ─── Decision verdict ────────────────────────────────────────────────
    lines.append("## 🎯 Decision Verdict")
    lines.append("")
    sgmy_delta = f_s - b_s
    if sgmy_delta >= 0.05:
        lines.append("### ✅ COMMIT TO FULL STAGE 0")
        lines.append(f"SG/MY F1 lift of {sgmy_delta:+.3f} (≥ +0.05 threshold) confirms curriculum direction.")
        lines.append(f"Scale up to 1000 synthetic resumes and run Stages 1-4.")
    elif sgmy_delta >= 0.0:
        lines.append("### 🤔 INVESTIGATE BEFORE SCALING")
        lines.append(f"SG/MY F1 lift of {sgmy_delta:+.3f} is positive but below +0.05 threshold.")
        lines.append("Possible actions: improve synthetic data quality, increase epochs, expand banks.")
    else:
        lines.append("### 🛑 DO NOT SCALE")
        lines.append(f"SG/MY F1 lift of {sgmy_delta:+.3f} is negative.")
        lines.append("Fine-tuning hurt SG/MY recognition. Investigate corruption issues before any scale-up.")
    lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 🎬 MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Formal F1 evaluation of Stage 0 LITE")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN_PATH,
                        help="Path to golden_sg_my.jsonl")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_PATH,
                        help="Path to LoRA adapter folder")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="HF model ID for base")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit to N cases (for fast smoke test)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the regex baseline (fine-tuned only)")
    parser.add_argument("--skip-finetuned", action="store_true",
                        help="Skip the fine-tuned model (baseline only)")
    parser.add_argument("--max-context", type=int, default=8192,
                        help="Max context tokens for inference (default: 8192). "
                             "Increase if VRAM allows; resumes are truncated to fit.")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + "  💎  STAGE 0 LITE — F1 EVALUATION  💎  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Load golden set ─────────────────────────────────────────────────
    print(f"\n📂 Loading golden set from {args.golden}...")
    golden = load_golden(Path(args.golden))
    if not golden:
        print("❌ No golden records loaded. Exiting.")
        return
    print(f"   ✅ Loaded {len(golden)} golden cases")

    if args.max_cases:
        golden = golden[:args.max_cases]
        print(f"   🔪 Limiting to first {len(golden)} cases")

    # ─── Init extractors ─────────────────────────────────────────────────
    baseline = None
    finetuned = None

    if not args.skip_baseline:
        print(f"\n🅰️  Initializing baseline extractor...")
        baseline = init_baseline()
        if not baseline:
            print(f"   ⚠️  Continuing without baseline")

    if not args.skip_finetuned:
        print(f"\n🅱️  Initializing fine-tuned extractor...")
        finetuned = init_finetuned(args.base_model, args.adapter)
        if not finetuned:
            print(f"   ⚠️  Continuing without fine-tuned")
        else:
            # 💎 Set up raw-output logging directory
            raw_dir = OUTPUT_DIR / "raw_outputs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            run_finetuned._log_dir = raw_dir
            print(f"   💾 Raw outputs will be saved to: {raw_dir}")

    if not baseline and not finetuned:
        print("\n❌ No extractors available. Exiting.")
        return

    # ─── Run evaluation ──────────────────────────────────────────────────
    baseline_results = []
    finetuned_results = []

    for i, rec in enumerate(golden, 1):
        raw_text = rec.get("raw_text") or rec.get("raw") or ""
        gold_raw = rec.get("ground_truth") or rec.get("gold") or {}
        # 💎 Normalize gold from snake_case to Title Case canonical schema
        gold = normalize_gt(gold_raw)
        case_id = rec.get("id") or rec.get("candidate_id") or f"case_{i:03d}"

        print(f"\n[{i}/{len(golden)}] {case_id}")

        if baseline:
            t0 = time.time()
            pred = run_baseline(baseline, raw_text)
            elapsed = time.time() - t0
            scored = score_one_record(gold, pred)
            baseline_results.append({
                "case_id": case_id, "elapsed_s": elapsed,
                "pred": pred, **scored
            })
            print(f"   🅰️  baseline: overall={scored['overall_f1']:.3f}, "
                  f"sgmy={scored['sgmy_f1']:.3f}  ({elapsed:.1f}s)")

        if finetuned:
            t0 = time.time()
            # 💎 Tag the current case for raw-output logging
            run_finetuned._current_case_id = case_id
            pred = run_finetuned(finetuned, raw_text,
                                  max_new_tokens=args.max_new_tokens,
                                  max_context=args.max_context)
            elapsed = time.time() - t0
            scored = score_one_record(gold, pred)
            finetuned_results.append({
                "case_id": case_id, "elapsed_s": elapsed,
                "pred": pred, **scored
            })
            print(f"   🅱️  fine-tune: overall={scored['overall_f1']:.3f}, "
                  f"sgmy={scored['sgmy_f1']:.3f}  ({elapsed:.1f}s)  "
                  f"[pred has {len(pred)} keys]")

    # ─── Aggregate ───────────────────────────────────────────────────────
    baseline_agg = aggregate_scores(baseline_results) if baseline_results else {}
    finetuned_agg = aggregate_scores(finetuned_results) if finetuned_results else {}

    # ─── Save raw results ────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "lite_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {"aggregate": baseline_agg, "per_case": baseline_results},
            "finetuned": {"aggregate": finetuned_agg, "per_case": finetuned_results},
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 Raw results: {results_path}")

    # ─── Build + save markdown report ────────────────────────────────────
    if baseline_agg and finetuned_agg:
        report = render_report(baseline_agg, finetuned_agg,
                               baseline_results + finetuned_results)
        report_path = OUTPUT_DIR / "lite_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"📊 Markdown report: {report_path}")

    # ─── Print summary to stdout ─────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  ✨ EVALUATION COMPLETE ✨".center(70))
    print("═" * 70)

    if baseline_agg:
        print(f"\n🅰️  BASELINE   — Overall: {baseline_agg['overall_f1']:.3f}, "
              f"SG/MY: {baseline_agg['sgmy_f1']:.3f}")
    if finetuned_agg:
        print(f"🅱️  FINE-TUNED — Overall: {finetuned_agg['overall_f1']:.3f}, "
              f"SG/MY: {finetuned_agg['sgmy_f1']:.3f}")

    if baseline_agg and finetuned_agg:
        sgmy_delta = finetuned_agg['sgmy_f1'] - baseline_agg['sgmy_f1']
        overall_delta = finetuned_agg['overall_f1'] - baseline_agg['overall_f1']
        print(f"\n📊 SG/MY F1 lift:   {sgmy_delta:+.3f}")
        print(f"📊 Overall F1 lift: {overall_delta:+.3f}")
        print()
        if sgmy_delta >= 0.05:
            print("  ✅ VERDICT: COMMIT TO FULL STAGE 0  ✨")
        elif sgmy_delta >= 0.0:
            print("  🤔 VERDICT: INVESTIGATE BEFORE SCALING")
        else:
            print("  🛑 VERDICT: DO NOT SCALE — diagnose first")
        print()


if __name__ == "__main__":
    main()