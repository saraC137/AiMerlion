"""
ml/bert_ner_train.py
====================
Fine-tunes a BERT token classifier for resume NER.

Entities
--------
  NAME        – candidate's full name (header)
  TITLE       – job title (experience)
  COMPANY     – employer name (experience)
  WORK_DATE   – tenure date range (experience)
  INSTITUTION – school / university (education)
  MAJOR       – degree / field of study (education)
  SKILL       – technical or soft skill (skills section)
  LANGUAGE    – spoken language (languages section)

Data format (train_data.jsonl)
-------------------------------
ShareGPT conversations where:
  conversations[1]["value"]  = raw resume text (after "Resume:\n")
  conversations[2]["value"]  = JSON string with extracted fields

Usage
-----
  python ml/bert_ner_train.py
  python ml/bert_ner_train.py --train train_data.jsonl --val train_data_val.jsonl
                               --base dslim/bert-base-NER
                               --output ml_models/bert_ner
                               --epochs 30 --lr 2e-5 --batch 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Project root on sys.path so we can import from extraction/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.text_preprocessor import clean, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Label scheme ──────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "NAME",
    "TITLE",
    "COMPANY",
    "WORK_DATE",
    "INSTITUTION",
    "MAJOR",
    "SKILL",
    "LANGUAGE",
]

LABELS = ["O"] + [f"{bio}-{et}" for et in ENTITY_TYPES for bio in ("B", "I")]
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: l for l, i in LABEL2ID.items()}

LABEL_CONFIG = {"label2id": LABEL2ID, "id2label": ID2LABEL, "entity_types": ENTITY_TYPES}

# ── Span finder ───────────────────────────────────────────────────────────────

def _find_span(text: str, entity: str) -> Optional[tuple[int, int]]:
    """
    Locate `entity` string inside `text`.
    Tries exact → case-insensitive → whitespace-normalised.
    Returns (start, end) char offsets or None.
    """
    if not entity or not entity.strip():
        return None
    ent = entity.strip()

    # 1. Exact
    idx = text.find(ent)
    if idx != -1:
        return (idx, idx + len(ent))

    # 2. Case-insensitive
    idx = text.lower().find(ent.lower())
    if idx != -1:
        return (idx, idx + len(ent))

    # 3. Normalised whitespace (PDF extraction sometimes collapses spaces)
    norm_text = re.sub(r"\s+", " ", text)
    norm_ent  = re.sub(r"\s+", " ", ent)
    idx = norm_text.lower().find(norm_ent.lower())
    if idx != -1:
        return (idx, idx + len(norm_ent))

    return None


def _resolve_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """
    Sort spans by position and remove overlaps (earlier / longer span wins).
    """
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    result: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, label in spans:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    return result


# ── Training record → char spans ─────────────────────────────────────────────

def record_to_char_spans(
    raw_text: str,
    labels_dict: dict,
) -> tuple[str, list[tuple[int, int, str]]]:
    """
    Convert one training record to (cleaned_text, [(start, end, label), ...]).

    Section-aware: experience entities are searched within the experience section,
    skills within the skills section, etc.  Falls back to full text if a section
    is absent so no entity is silently dropped.
    """
    prep = preprocess(raw_text)
    text = prep.cleaned

    def section_text(name: str) -> tuple[str, int]:
        """Return (section_text, offset_in_full_text)."""
        s = prep.sections.get(name, "")
        if not s:
            return text, 0
        off = text.find(s[:60])
        return s, (off if off != -1 else 0)

    header_text, header_off = (prep.header, 0) if prep.header else (text, 0)
    exp_text,  exp_off  = section_text("experience")
    edu_text,  edu_off  = section_text("education")
    skill_text, skill_off = section_text("skills")
    lang_text, lang_off = section_text("languages")
    if not prep.sections.get("languages"):
        # Sometimes labelled "language" (singular) in header
        lang_text, lang_off = section_text("language") if "language" in prep.sections else (text, 0)

    spans: list[tuple[int, int, str]] = []

    # ── NAME (search header first, then full text) ────────────────────────
    name = labels_dict.get("name", "")
    if name:
        sp = _find_span(header_text, name) or _find_span(text, name)
        if sp:
            off = header_off if _find_span(header_text, name) else 0
            spans.append((sp[0] + off, sp[1] + off, "NAME"))

    # ── EXPERIENCE ────────────────────────────────────────────────────────
    for exp in labels_dict.get("experience", []):
        for field, label in [("title", "TITLE"), ("company", "COMPANY"), ("duration", "WORK_DATE")]:
            val = exp.get(field, "")
            if val:
                sp = _find_span(exp_text, val) or _find_span(text, val)
                if sp:
                    off = exp_off if _find_span(exp_text, val) else 0
                    spans.append((sp[0] + off, sp[1] + off, label))

    # ── EDUCATION ─────────────────────────────────────────────────────────
    for edu in labels_dict.get("education", []):
        for field, label in [("institution", "INSTITUTION"), ("degree", "MAJOR")]:
            val = edu.get(field, "")
            if val:
                sp = _find_span(edu_text, val) or _find_span(text, val)
                if sp:
                    off = edu_off if _find_span(edu_text, val) else 0
                    spans.append((sp[0] + off, sp[1] + off, label))

    # ── SKILLS ───────────────────────────────────────────────────────────
    for skill in labels_dict.get("hard_skills", []) + labels_dict.get("soft_skills", []):
        if skill:
            # Prefer skills section to avoid tagging skill words in descriptions
            sp = _find_span(skill_text, skill) or _find_span(text, skill)
            if sp:
                off = skill_off if _find_span(skill_text, skill) else 0
                spans.append((sp[0] + off, sp[1] + off, "SKILL"))

    # ── LANGUAGES ────────────────────────────────────────────────────────
    for lang in labels_dict.get("languages", []):
        if lang:
            sp = _find_span(lang_text, lang) or _find_span(text, lang)
            if sp:
                off = lang_off if _find_span(lang_text, lang) else 0
                spans.append((sp[0] + off, sp[1] + off, "LANGUAGE"))

    return text, _resolve_spans(spans)


# ── Tokenise and align BIO tags to wordpieces ─────────────────────────────────

def tokenise_and_align(
    tokenizer: BertTokenizerFast,
    text: str,
    char_spans: list[tuple[int, int, str]],
    max_length: int = 512,
) -> dict:
    """
    Tokenise `text` and align char-level entity spans to token-level BIO labels.

    Rules:
    - Special tokens ([CLS], [SEP], [PAD]) → -100 (ignored by loss)
    - First subword of an entity → B-<TYPE>
    - Continuation subwords → I-<TYPE>
    - Non-entity tokens → O (label id 0)
    """
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )

    # Build a character-level label array
    char_label = ["O"] * len(text)
    for start, end, etype in char_spans:
        for ci in range(start, min(end, len(text))):
            char_label[ci] = f"B-{etype}" if ci == start else f"I-{etype}"

    offset_mapping = encoding["offset_mapping"]
    word_ids       = encoding.word_ids()   # None for special tokens
    token_labels: list[int] = []
    prev_word_id: Optional[int] = None

    for tok_idx, word_id in enumerate(word_ids):
        if word_id is None:
            token_labels.append(-100)
            prev_word_id = None
            continue

        tok_start, tok_end = offset_mapping[tok_idx]
        raw_label = char_label[tok_start] if tok_start < len(char_label) else "O"

        # Continuation subword of the same original word → force I- tag
        if prev_word_id is not None and word_id == prev_word_id:
            if raw_label.startswith("B-"):
                raw_label = "I-" + raw_label[2:]

        token_labels.append(LABEL2ID.get(raw_label, 0))
        prev_word_id = word_id

    encoding.pop("offset_mapping")
    encoding["labels"] = token_labels
    return {k: list(v) for k, v in encoding.items()}


# ── Dataset ───────────────────────────────────────────────────────────────────

class ResumeNERDataset(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: torch.tensor(v) for k, v in self.samples[idx].items()}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str, tokenizer: BertTokenizerFast, max_length: int = 512) -> ResumeNERDataset:
    """
    Load a ShareGPT-format JSONL file and return a ResumeNERDataset.
    Records where GPT JSON is unparseable or raw text is missing are skipped.
    """
    samples: list[dict] = []
    skipped = 0

    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                convs  = record.get("conversations", [])

                # Locate human message (raw text) and gpt message (labels)
                human_msg = next((c["value"] for c in convs if c["from"] == "human"), "")
                gpt_msg   = next((c["value"] for c in convs if c["from"] == "gpt"),   "")

                # Extract raw resume text from human message
                resume_marker = "Resume:"
                resume_pos    = human_msg.find(resume_marker)
                raw_text      = human_msg[resume_pos + len(resume_marker):].strip() if resume_pos != -1 else human_msg

                if len(raw_text) < 50:
                    skipped += 1
                    continue

                labels_dict = json.loads(gpt_msg)
            except (json.JSONDecodeError, StopIteration, KeyError):
                skipped += 1
                continue

            text, char_spans = record_to_char_spans(raw_text, labels_dict)
            enc = tokenise_and_align(tokenizer, text, char_spans, max_length)
            samples.append(enc)

    log.info("Loaded %d samples from %s (%d skipped)", len(samples), path, skipped)
    return ResumeNERDataset(samples)


# ── Metrics ───────────────────────────────────────────────────────────────────

def make_compute_metrics(id2label: dict[int, str]):
    """Return a compute_metrics function that uses seqeval."""
    try:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
        USE_SEQEVAL = True
    except ImportError:
        USE_SEQEVAL = False
        log.warning("seqeval not installed — using token-level accuracy only. "
                    "Install with: pip install seqeval")

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)

        if USE_SEQEVAL:
            pred_seqs  = []
            label_seqs = []
            for pred_row, label_row in zip(preds, label_ids):
                pred_seq  = []
                label_seq = []
                for p, l in zip(pred_row, label_row):
                    if l == -100:
                        continue
                    pred_seq.append(id2label.get(int(p), "O"))
                    label_seq.append(id2label.get(int(l), "O"))
                pred_seqs.append(pred_seq)
                label_seqs.append(label_seq)

            return {
                "f1":        f1_score(label_seqs, pred_seqs, zero_division=0),
                "precision": precision_score(label_seqs, pred_seqs, zero_division=0),
                "recall":    recall_score(label_seqs, pred_seqs, zero_division=0),
            }
        else:
            # Token-level accuracy (ignoring -100)
            mask  = label_ids != -100
            total = mask.sum()
            correct = ((preds == label_ids) & mask).sum()
            return {"accuracy": float(correct) / float(total) if total else 0.0}

    return compute_metrics


# ── Training entry point ──────────────────────────────────────────────────────

def train(
    train_path:  str   = "train_data.jsonl",
    val_path:    str   = "train_data_val.jsonl",
    base_model:  str   = "dslim/bert-base-NER",
    output_dir:  str   = "ml_models/bert_ner",
    epochs:      int   = 30,
    lr:          float = 2e-5,
    batch_size:  int   = 8,
    max_length:  int   = 512,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading tokenizer from %s", base_model)
    tokenizer = BertTokenizerFast.from_pretrained(base_model)

    log.info("Building datasets …")
    train_ds = load_jsonl(train_path, tokenizer, max_length)
    val_ds   = load_jsonl(val_path,   tokenizer, max_length)

    if len(train_ds) == 0:
        raise ValueError(f"No usable training records found in {train_path}")

    log.info("Loading base model %s with %d labels", base_model, len(LABELS))
    model = BertForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # base NER model has different label head
    )

    # Save label config alongside the model for inference
    with open(output_path / "label_config.json", "w") as fh:
        json.dump(LABEL_CONFIG, fh, indent=2)
    tokenizer.save_pretrained(output_path)

    training_args = TrainingArguments(
        output_dir                  = str(output_path),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate               = lr,
        warmup_ratio                = warmup_ratio,
        weight_decay                = weight_decay,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        save_total_limit            = 2,
        fp16                        = torch.cuda.is_available(),
        dataloader_num_workers      = 0,
        report_to                   = "none",
        logging_steps               = 10,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    compute_metrics = make_compute_metrics(ID2LABEL)

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    log.info("Starting training …")
    trainer.train()

    log.info("Saving best model to %s", output_dir)
    trainer.save_model(output_dir)
    log.info("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT NER for resume entities")
    parser.add_argument("--train",   default="train_data.jsonl",         help="Training JSONL")
    parser.add_argument("--val",     default="train_data_val.jsonl",      help="Validation JSONL")
    parser.add_argument("--base",    default="dslim/bert-base-NER",       help="HuggingFace base model")
    parser.add_argument("--output",  default="ml_models/bert_ner",        help="Output directory")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--maxlen",  type=int,   default=512)
    args = parser.parse_args()

    train(
        train_path  = args.train,
        val_path    = args.val,
        base_model  = args.base,
        output_dir  = args.output,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch,
        max_length  = args.maxlen,
    )


if __name__ == "__main__":
    main()
