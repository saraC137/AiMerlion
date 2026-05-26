"""
extraction/bert_extractor.py
=============================
BERT NER inference for resume entities.

Loads the fine-tuned model produced by ml/bert_ner_train.py and extracts:
  - Name              (from header section)
  - Work experience   [{title, company, work_date}] (from experience section)
  - Education         [{institution, major}]        (from education section)
  - Skills            [str, ...]                    (from skills section)
  - Languages         [str, ...]                    (from languages section)

Section-aware: each section's text is fed separately so the model operates
in the right context (e.g. "Python" in the skills section → SKILL, not noise).

Usage
-----
  from extraction.bert_extractor import BERTExtractor
  from extraction.text_preprocessor import preprocess

  extractor = BERTExtractor("ml_models/bert_ner")
  prep      = preprocess(raw_text)
  result    = extractor.extract_from_prepared(prep)
  # result = {
  #   "name": "Wayne Chiam",
  #   "work_experience": [{"title": "...", "company": "...", "work_date": "..."}],
  #   "education":       [{"institution": "...", "major": "..."}],
  #   "skills":          ["Python", "SQL", ...],
  #   "languages":       ["English", "Chinese"],
  # }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

log = logging.getLogger(__name__)

# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class WorkEntry:
    title:     Optional[str] = None
    company:   Optional[str] = None
    work_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EducationEntry:
    institution: Optional[str] = None
    major:       Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class NERResult:
    name:            Optional[str]       = None
    work_experience: list[WorkEntry]     = field(default_factory=list)
    education:       list[EducationEntry] = field(default_factory=list)
    skills:          list[str]           = field(default_factory=list)
    languages:       list[str]           = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name":            self.name,
            "work_experience": [e.to_dict() for e in self.work_experience],
            "education":       [e.to_dict() for e in self.education],
            "skills":          self.skills,
            "languages":       self.languages,
        }


# ── Main class ────────────────────────────────────────────────────────────────

class BERTExtractor:
    """
    Thin wrapper around a fine-tuned BertForTokenClassification model.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing the saved model, tokenizer, and label_config.json.
    device : str | None
        "cuda", "cpu", or None (auto-detect).
    max_length : int
        Maximum token sequence length fed to BERT (default 512).
    stride : int
        Overlap stride for long texts that exceed max_length.
        Text is split into overlapping windows; entities from the centre
        of each window are kept.
    """

    # Minimum character length for an extracted entity to be kept
    _MIN_ENTITY_LEN = 2

    def __init__(
        self,
        model_dir: str | Path,
        device:     Optional[str] = None,
        max_length: int = 512,
        stride:     int = 128,
    ) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"BERT NER model not found at {model_dir}. "
                "Train it first with: python ml/bert_ner_train.py"
            )

        log.info("Loading BERT NER model from %s", model_dir)
        self.tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
        self.model     = BertForTokenClassification.from_pretrained(str(model_dir))
        self.model.eval()

        cfg_path = model_dir / "label_config.json"
        if cfg_path.exists():
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            self.id2label: dict[int, str] = {int(k): v for k, v in cfg["id2label"].items()}
        else:
            # Fall back to model config
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.max_length = max_length
        self.stride     = stride
        self.model.to(self.device)
        log.info("BERTExtractor ready on %s with %d labels", device, len(self.id2label))

    # ── Low-level predict ─────────────────────────────────────────────────────

    def _predict_tokens(self, text: str) -> list[tuple[str, str]]:
        """
        Run the model over `text` (with sliding window for long texts).
        Returns [(word_token, predicted_label), ...] for non-special tokens only.
        Continuation subwords (##…) are merged into their head token.
        """
        if not text.strip():
            return []

        # Tokenise with sliding window
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Move to device
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
            )

        logits          = outputs.logits
        pred_ids        = torch.argmax(logits, dim=-1).cpu().tolist()
        offset_mapping  = encoding["offset_mapping"].tolist()
        overflow_to_sample = encoding.get("overflow_to_sample_mapping", [[0] * input_ids.shape[1]])

        # Collect (char_start, char_end, label) from all windows,
        # keeping only the "authoritative" part (not the overlap) of each window.
        char_label: dict[int, str] = {}   # char_start → label

        for win_idx, (win_preds, win_offsets) in enumerate(zip(pred_ids, offset_mapping)):
            # Determine the "safe zone" — middle of the window for all windows
            # except the first (keep from 0) and last (keep to end).
            n_wins = len(pred_ids)
            for tok_idx, (offset, pred_id) in enumerate(zip(win_offsets, win_preds)):
                tok_start, tok_end = offset
                if tok_start == tok_end:        # special token
                    continue
                label = self.id2label.get(pred_id, "O")

                # For overlapping windows, only trust tokens in the non-overlap zone
                # (first half of stride on the left, last half on the right).
                # Simple heuristic: skip the first `stride` tokens of non-first windows.
                if win_idx > 0 and tok_idx < self.stride:
                    continue
                if win_idx < n_wins - 1 and tok_idx >= self.max_length - self.stride:
                    continue

                if tok_start not in char_label:
                    char_label[tok_start] = label

        # Reconstruct tokens with their labels in order
        # Use first-window tokenisation as the token sequence reference
        ref_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        tokens  = self.tokenizer.convert_ids_to_tokens(ref_encoding["input_ids"])
        offsets = ref_encoding["offset_mapping"]

        result: list[tuple[str, str]] = []
        for token, (tok_start, tok_end) in zip(tokens, offsets):
            if tok_start == tok_end:            # [CLS], [SEP], [PAD]
                continue
            label = char_label.get(tok_start, "O")
            result.append((token, label))

        return result

    # ── BIO → entity spans ────────────────────────────────────────────────────

    def _collect_entities(
        self, token_labels: list[tuple[str, str]]
    ) -> list[tuple[str, str, int]]:
        """
        Collapse BIO token sequence into (entity_text, entity_type, position) triples.
        Wordpiece subwords (##…) are merged back into the surface form.
        """
        entities: list[tuple[str, str, int]] = []
        current_tokens: list[str] = []
        current_type:   str = ""
        current_pos:    int = 0

        for pos, (token, label) in enumerate(token_labels):
            if label.startswith("B-"):
                # Flush previous entity
                if current_tokens and current_type:
                    text = self._merge_subwords(current_tokens)
                    if len(text) >= self._MIN_ENTITY_LEN:
                        entities.append((text, current_type, current_pos))
                current_tokens = [token]
                current_type   = label[2:]
                current_pos    = pos

            elif label.startswith("I-") and current_type == label[2:]:
                current_tokens.append(token)

            else:
                # O or unexpected tag — flush
                if current_tokens and current_type:
                    text = self._merge_subwords(current_tokens)
                    if len(text) >= self._MIN_ENTITY_LEN:
                        entities.append((text, current_type, current_pos))
                current_tokens = []
                current_type   = ""

        # Final flush
        if current_tokens and current_type:
            text = self._merge_subwords(current_tokens)
            if len(text) >= self._MIN_ENTITY_LEN:
                entities.append((text, current_type, current_pos))

        return entities

    @staticmethod
    def _merge_subwords(tokens: list[str]) -> str:
        """Join wordpiece tokens: ['word', '##piece'] → 'wordpiece'."""
        merged = ""
        for tok in tokens:
            if tok.startswith("##"):
                merged += tok[2:]
            elif merged:
                merged += " " + tok
            else:
                merged = tok
        return merged.strip()

    # ── High-level extraction on plain text ───────────────────────────────────

    def extract(self, text: str) -> dict[str, list]:
        """
        Run NER on `text` and return a dict of entity type → list of strings.

        Example::
            {
              "NAME":        ["Wayne Chiam"],
              "TITLE":       ["IT Support Engineer"],
              "COMPANY":     ["Aetos Technologies"],
              "WORK_DATE":   ["01/2015 - Present"],
              "INSTITUTION": ["Nanyang Polytechnic"],
              "MAJOR":       ["Diploma in Multimedia"],
              "SKILL":       ["Python", "SQL"],
              "LANGUAGE":    ["English", "Chinese"],
            }
        """
        token_labels = self._predict_tokens(text)
        entities     = self._collect_entities(token_labels)

        result: dict[str, list] = {}
        for surface, etype, _pos in entities:
            result.setdefault(etype, []).append(surface)
        return result

    # ── Section-aware structured extraction ───────────────────────────────────

    def extract_from_prepared(self, prep) -> NERResult:
        """
        Run section-aware NER on a `PreparedResume` (from text_preprocessor.preprocess).
        Each section is fed to the model independently to maximise context relevance.

        Returns a `NERResult` with name, work_experience, education, skills, languages.
        """
        result = NERResult()

        # ── NAME (header section) ─────────────────────────────────────────
        header = prep.header or prep.cleaned[:500]
        if header:
            header_ents = self.extract(header)
            names = header_ents.get("NAME", [])
            if names:
                result.name = names[0]

        # ── WORK EXPERIENCE ───────────────────────────────────────────────
        exp_text = prep.sections.get("experience", "")
        if exp_text:
            exp_ents = self.extract(exp_text)
            result.work_experience = self._group_experience(
                exp_ents.get("TITLE",     []),
                exp_ents.get("COMPANY",   []),
                exp_ents.get("WORK_DATE", []),
                exp_text,
            )

        # ── EDUCATION ─────────────────────────────────────────────────────
        edu_text = prep.sections.get("education", "")
        if edu_text:
            edu_ents = self.extract(edu_text)
            result.education = self._group_education(
                edu_ents.get("INSTITUTION", []),
                edu_ents.get("MAJOR",       []),
                edu_text,
            )

        # ── SKILLS ────────────────────────────────────────────────────────
        skill_text = prep.sections.get("skills", "")
        if skill_text:
            skill_ents = self.extract(skill_text)
            result.skills = self._dedupe(skill_ents.get("SKILL", []))

        # ── LANGUAGES ─────────────────────────────────────────────────────
        lang_text = (
            prep.sections.get("languages")
            or prep.sections.get("language")
            or ""
        )
        if lang_text:
            lang_ents = self.extract(lang_text)
            result.languages = self._dedupe(lang_ents.get("LANGUAGE", []))

        return result

    # ── Entity grouping helpers ───────────────────────────────────────────────

    def _group_experience(
        self,
        titles:     list[str],
        companies:  list[str],
        work_dates: list[str],
        section_text: str,
    ) -> list[WorkEntry]:
        """
        Group TITLE / COMPANY / WORK_DATE entities into WorkEntry records.

        Strategy: each TITLE found in the text anchors one job record.
        COMPANY and WORK_DATE that appear between two consecutive TITLE positions
        are assigned to the earlier TITLE.  Surplus COMPANY/WORK_DATE without
        a TITLE are collected into a final catch-all entry.
        """
        if not titles:
            # No titles — bundle everything into one entry if something exists
            if companies or work_dates:
                return [WorkEntry(
                    company   = companies[0]  if companies  else None,
                    work_date = work_dates[0] if work_dates else None,
                )]
            return []

        # Find approximate position of each entity in the section text
        def first_pos(s: str) -> int:
            idx = section_text.lower().find(s.lower())
            return idx if idx != -1 else len(section_text)

        title_positions   = sorted([(first_pos(t), t) for t in titles],   key=lambda x: x[0])
        company_positions = sorted([(first_pos(c), c) for c in companies], key=lambda x: x[0])
        date_positions    = sorted([(first_pos(d), d) for d in work_dates], key=lambda x: x[0])

        entries: list[WorkEntry] = []
        for i, (t_pos, title) in enumerate(title_positions):
            next_pos = title_positions[i + 1][0] if i + 1 < len(title_positions) else len(section_text)

            entry = WorkEntry(title=title)

            # Assign the first COMPANY that falls within this title's range
            for c_pos, company in company_positions:
                if t_pos <= c_pos < next_pos:
                    entry.company = company
                    break

            # Assign the first WORK_DATE within this title's range
            for d_pos, work_date in date_positions:
                if t_pos <= d_pos < next_pos:
                    entry.work_date = work_date
                    break

            entries.append(entry)

        return entries

    def _group_education(
        self,
        institutions: list[str],
        majors:       list[str],
        section_text: str,
    ) -> list[EducationEntry]:
        """
        Group INSTITUTION / MAJOR entities into EducationEntry records.
        Each INSTITUTION anchors one education entry; MAJOR is the closest
        degree text above or below it.
        """
        if not institutions and not majors:
            return []

        def first_pos(s: str) -> int:
            idx = section_text.lower().find(s.lower())
            return idx if idx != -1 else len(section_text)

        if not institutions:
            # Only majors — list them without institution
            return [EducationEntry(major=m) for m in majors]

        inst_positions  = sorted([(first_pos(i), i) for i in institutions], key=lambda x: x[0])
        major_positions = sorted([(first_pos(m), m) for m in majors],       key=lambda x: x[0])

        entries: list[EducationEntry] = []
        for i, (inst_pos, institution) in enumerate(inst_positions):
            next_pos = inst_positions[i + 1][0] if i + 1 < len(inst_positions) else len(section_text)

            entry = EducationEntry(institution=institution)

            # Find the MAJOR closest to this institution (within its range)
            for m_pos, major in major_positions:
                if inst_pos - 200 <= m_pos < next_pos:   # allow major slightly before institution
                    entry.major = major
                    break

            entries.append(entry)

        return entries

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        """Remove duplicates while preserving order (case-insensitive)."""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result

    # ── Convenience: run full pipeline from raw text ──────────────────────────

    def extract_raw(self, raw_text: str) -> NERResult:
        """
        Convenience method: preprocess `raw_text` then run section-aware NER.
        Returns a NERResult.
        """
        # Import here to avoid circular imports at module load time
        from extraction.text_preprocessor import preprocess
        prep = preprocess(raw_text)
        return self.extract_from_prepared(prep)
