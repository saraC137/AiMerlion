"""
extraction/text_preprocessor.py
================================
Standalone resume text preprocessing pipeline.

Three stages:
  1. clean()           - encoding fixes, invisible chars, whitespace, dashes, bullets, markdown
  2. detect_sections() - returns {section_name: (start, end)} character positions in cleaned text
  3. slice_sections()  - returns {section_name: text} ready for downstream models

Convenience entry point:
  preprocess(raw_text) -> PreparedResume(cleaned, header, sections)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class PreparedResume:
    cleaned: str                         # full cleaned text
    header:  str                         # contact-info block (before first section)
    sections: Dict[str, str]             # section_name -> section text
    boundaries: Dict[str, Tuple[int,int]] = field(default_factory=dict)  # positions in `cleaned`

    def get(self, section: str, default: str = "") -> str:
        return self.sections.get(section, default)


# ---------------------------------------------------------------------------
# Stage 1 – Clean
# ---------------------------------------------------------------------------

# Characters to strip completely (invisible, directional, BOM, etc.)
_STRIP_CHARS: str = (
    "​‌‍‎‏"   # zero-width space / joiners / marks
    "‪‫‬‭‮"   # directional embeddings
    "⁠⁡⁢⁣⁤"   # word joiner / invisible operators
    "﻿￾￿"               # BOM / non-chars
    "­"                           # soft hyphen
)

# Unicode whitespace variants → plain space
_WHITESPACE_VARIANTS: str = (
    " "   # non-breaking space
    "           "
    "  　"               # narrow/medium/ideographic space
)

# Dash variants → ASCII hyphen-minus
_DASH_VARIANTS: str = (
    "‐‑‒–—―"  # hyphen-minus variants, en/em/horizontal
    "−"                                 # minus sign
    "⁃﹘﹣－"              # other dash-likes
)

# Bullet variants → "* " (preserves list structure)
_BULLET_VARIANTS: str = (
    "•●○◦■□▪▫"
    "►▸▶▷‣∙⋅·"
    "→➢➤◆◇★☆"
    "✓✔✗✘☞➔"
)

# Smart/curly double-quote variants → "
_DOUBLE_QUOTES: str = "“”„‟«»″"

# Smart/curly single-quote variants → '
_SINGLE_QUOTES: str = "‘’‚‛‹›′`´"

# Line separator variants → \n
_LINE_SEPS: str = "  "

# Mojibake fragments that appear when UTF-8 is misread as Latin-1
_MOJIBAKE: Dict[str, str] = {
    "â": "-",   # en-dash
    "â": "-",   # em-dash
    "â": "'",   # left single quote
    "â": "'",   # right single quote
    "â": '"',   # left double quote
    "â": '"',   # right double quote
    "â¢": "* ",  # bullet
    "Ã©": "e",         # é
    "Ã¨": "e",         # è
    "Ã¢": "a",         # â
    "Ã®": "i",         # î
    "Ã´": "o",         # ô
    "Ã»": "u",         # û
    "Ã§": "c",         # ç
    "Ã±": "n",         # ñ
}

# Markdown header/bold/italic → plain text
_RE_MD_HEADER = re.compile(r"^#{1,6}\s*\**([^*\n]+?)\**\s*$", re.MULTILINE)
_RE_BOLD      = re.compile(r"\*\*\*([^*]+)\*\*\*|\*\*([^*]+)\*\*|__([^_]+)__")
_RE_ITALIC    = re.compile(r"(?<!\n)\*([^*\n]+)\*(?!\*)| (?<!\n)_([^_\n]+)_(?!_)")
_RE_MD_BULLET = re.compile(r"^\s*[-*•▪►▸◗¦◗‹◗‡]\s+", re.MULTILINE)


def _translate(text: str, chars: str, replacement: str) -> str:
    """Replace every character in `chars` with `replacement`."""
    table = str.maketrans({c: replacement for c in chars})
    return text.translate(table)


def clean(text: str) -> str:
    """
    Stage 1: normalise raw resume text.

    Order matters — encoding fixes first, then invisible chars,
    whitespace, line-endings, dashes, quotes, bullets, markdown,
    control chars, final whitespace collapse.
    """
    if not text:
        return ""

    # NFKC decomposition first so composed chars are canonical
    text = unicodedata.normalize("NFKC", text)

    # Mojibake fragments (rare but painful when present)
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)

    # Strip invisible / directional chars
    text = _translate(text, _STRIP_CHARS, "")

    # Whitespace variants → plain space
    text = _translate(text, _WHITESPACE_VARIANTS, " ")

    # Line separator variants → newline
    text = _translate(text, _LINE_SEPS, "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Dash variants → ASCII hyphen
    text = _translate(text, _DASH_VARIANTS, "-")

    # Quote normalisation
    text = _translate(text, _DOUBLE_QUOTES, '"')
    text = _translate(text, _SINGLE_QUOTES, "'")

    # Bullet variants → "* "
    text = _translate(text, _BULLET_VARIANTS, "*")
    # Ensure bullets are followed by a space
    text = re.sub(r"\*(?!\s)", "* ", text)

    # Markdown → plain text
    text = _RE_MD_HEADER.sub(lambda m: (m.group(1) or "").strip(), text)
    text = _RE_BOLD.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    text = _RE_ITALIC.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    text = _RE_MD_BULLET.sub("", text)

    # Strip C0/C1 control characters (except tab and newline)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Collapse runs of spaces/tabs; collapse 3+ blank lines to 2
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    return text.strip()


# ---------------------------------------------------------------------------
# Stage 2 – Detect section boundaries
# ---------------------------------------------------------------------------

# Section header patterns in priority order.
# More specific compound headers come first to prevent shorter patterns
# from stealing their match (e.g. "Work Experience" before "Experience").
_SECTION_PATTERNS: list[tuple[str, str]] = [
    # ── Contact / personal block ──────────────────────────────────────────
    ("personal_info",    r"(?:^|\n)\s*PERSONAL\s+(?:INFORMATION|PARTICULARS|DETAILS)\s*[:\n]"),

    # ── Summary ───────────────────────────────────────────────────────────
    ("summary",          r"(?:^|\n)\s*(?:PROFESSIONAL\s+)?(?:SUMMARY|PROFILE|CAREER\s+(?:SUMMARY|PROFILE)|OBJECTIVE|CAREER\s+OBJECTIVE|ABOUT\s*ME|PERSONAL\s+STATEMENT)\s*[:\n]"),

    # ── Skills ────────────────────────────────────────────────────────────
    ("skills",           r"(?:^|\n)\s*(?:CORE\s+COMPETENCIES|KEY\s+SKILLS?|TECHNICAL\s+SKILLS?|PROFESSIONAL\s+SKILLS?|SKILLS?\s*(?:SUMMARY|PROFILE)?|COMPETENCIES|AREAS?\s+OF\s+EXPERTISE)\s*[:\n]"),

    # ── Work experience (compound headers first) ──────────────────────────
    ("experience",       r"(?:^|\n)\s*(?:WORK[\s\-/]+EXPERIENCE|WORKING\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT\s+(?:HISTORY|EXPERIENCE)|CAREER\s+(?:HISTORY|EXPERIENCE)|WORK\s+HISTORY)\s*[:\n]"),
    ("experience",       r"(?:^|\n)\s*EXPERIENCE[S]?\s*[:\n]"),

    # ── Education ─────────────────────────────────────────────────────────
    ("education",        r"(?:^|\n)\s*EDUCATION(?:AL)?(?:\s*(?:&|AND)\s*TRAINING|\s+(?:BACKGROUND|HISTORY|QUALIFICATIONS?))?\s*[:\n]"),
    ("qualifications",   r"(?:^|\n)\s*(?:ACADEMIC\s+|PROFESSIONAL\s+)?QUALIFICATIONS?\s*[:\n]"),

    # ── Certifications ────────────────────────────────────────────────────
    ("certifications",   r"(?:^|\n)\s*(?:CERTIFICATIONS?|PROFESSIONAL\s+CERTIFICATIONS?|LICENSES?\s+(?:AND\s+CERTIFICATIONS?)?|CREDENTIALS?)\s*[:\n]"),

    # ── Achievements / awards ─────────────────────────────────────────────
    ("achievements",     r"(?:^|\n)\s*(?:ACHIEVEMENTS?|AWARDS?\s+(?:AND\s+ACHIEVEMENTS?)?|ACCOMPLISHMENTS?|HONORS?|RECOGNITIONS?)\s*[:\n]"),

    # ── Co-curricular (Singapore specific) ───────────────────────────────
    ("cocurricular",     r"(?:^|\n)\s*CO-?CURRICULAR(?:\s+ACTIVITIES)?\s*[:\n]"),

    # ── Projects ──────────────────────────────────────────────────────────
    ("projects",         r"(?:^|\n)\s*(?:(?:KEY|PERSONAL|ACADEMIC|NOTABLE)\s+)?PROJECTS?\s*[:\n]"),

    # ── Languages ────────────────────────────────────────────────────────
    ("languages",        r"(?:^|\n)\s*LANGUAGES?\s*(?:SPOKEN\s*)?\s*[:\n]"),

    # ── References ───────────────────────────────────────────────────────
    ("references",       r"(?:^|\n)\s*REFERENCES?\s*[:\n]"),

    # ── Hobbies / interests ───────────────────────────────────────────────
    ("hobbies",          r"(?:^|\n)\s*(?:HOBBIES?|INTERESTS?|LEISURE\s+ACTIVITIES?)\s*[:\n]"),
]

# Minimum content length for a section to be kept
_MIN_SECTION_CHARS = 10
# Two pattern matches within this many chars are considered the same header line
# (handles compound vs. simple patterns matching the same heading, e.g.
#  "WORK EXPERIENCE" matched by both the compound and simple experience pattern)
_DEDUP_DISTANCE = 10


def detect_sections(text: str) -> Dict[str, Tuple[int, int]]:
    """
    Stage 2: locate section boundaries in `text`.

    Returns a dict mapping canonical section name → (content_start, content_end)
    where the span covers only the body text (header line excluded).
    The positions reference character offsets in the supplied `text`.

    The text should already be cleaned (pass through clean() first).
    """
    # --- find all header matches -----------------------------------------
    hits: list[dict] = []
    for canonical, pattern in _SECTION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            hits.append({
                "name":       canonical,
                "start":      m.start(),
                "header_end": m.end(),
                "header_len": m.end() - m.start(),
            })

    if not hits:
        return {}

    # Sort by position; for ties prefer the longer (more specific) match
    hits.sort(key=lambda h: (h["start"], -h["header_len"]))

    # De-duplicate: keep first match within each _DEDUP_DISTANCE window
    unique: list[dict] = []
    last_pos = -_DEDUP_DISTANCE - 1
    for h in hits:
        if h["start"] - last_pos > _DEDUP_DISTANCE:
            unique.append(h)
            last_pos = h["start"]

    # --- assign content spans --------------------------------------------
    boundaries: Dict[str, Tuple[int, int]] = {}
    for i, h in enumerate(unique):
        name          = h["name"]
        content_start = h["header_end"]
        content_end   = unique[i + 1]["start"] if i + 1 < len(unique) else len(text)

        if content_end - content_start < _MIN_SECTION_CHARS:
            continue  # empty / stub section

        # If the same canonical name already exists take the earlier one
        if name not in boundaries or content_start < boundaries[name][0]:
            boundaries[name] = (content_start, content_end)

    return boundaries


# ---------------------------------------------------------------------------
# Stage 3 – Slice into labelled sections
# ---------------------------------------------------------------------------

def slice_sections(text: str) -> Dict[str, str]:
    """
    Stage 3: return {section_name: section_text} from `text`.

    Runs detect_sections() internally; you do not need to call it separately
    unless you also need the character positions.
    """
    boundaries = detect_sections(text)
    return {
        name: text[start:end].strip()
        for name, (start, end) in boundaries.items()
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(raw_text: str) -> PreparedResume:
    """
    Full pipeline: clean → detect → slice.

    Returns a PreparedResume with:
      .cleaned    — normalised full text
      .header     — text before the first section header (contact / name block)
      .sections   — {section_name: section_text}
      .boundaries — {section_name: (start, end)} in .cleaned
    """
    cleaned    = clean(raw_text)
    boundaries = detect_sections(cleaned)

    # Header = everything before the first section header
    if boundaries:
        first_section_start = min(start for start, _ in boundaries.values())
        # Walk back to find the actual header start (before the header line)
        header_end = 0
        for name, (start, _) in boundaries.items():
            if start == first_section_start:
                # The header line itself sits just before content_start;
                # we want the text before the header keyword begins.
                # Re-detect the header keyword start by scanning backwards.
                header_end = _find_section_header_start(cleaned, start)
                break
        header = cleaned[:header_end].strip()
    else:
        # No sections found — treat first 2000 chars as header area
        header = cleaned[:2000].strip()

    sections = {
        name: cleaned[start:end].strip()
        for name, (start, end) in boundaries.items()
    }

    return PreparedResume(
        cleaned    = cleaned,
        header     = header,
        sections   = sections,
        boundaries = boundaries,
    )


def _find_section_header_start(text: str, content_start: int) -> int:
    """
    Given the content_start position (end of a section header line),
    walk backwards to find where the header line itself begins.
    Returns the position of the start of that header line.
    """
    # Scan back from content_start to find the preceding newline
    pos = content_start - 1
    while pos > 0 and text[pos] != "\n":
        pos -= 1
    # pos is now at the newline before the header, or 0
    # Walk back further past any blank lines
    while pos > 0 and text[pos - 1] in ("\n", " ", "\t"):
        pos -= 1
    return max(0, pos)
