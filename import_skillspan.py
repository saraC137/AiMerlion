"""
import_skillspan.py

💅✨ FAIRY CODEMOTHER'S SKILLSPAN IMPORTER ✨💅

Downloads and converts the SkillSpan dataset (Zhang et al., NAACL 2022)
into AiMerlion's TWO training paths:
  PATH A → SQLite (ner_documents + ner_annotations) for spaCy NER via train_ner.py
  PATH B → ShareGPT JSONL for LLM fine-tuning via finetune_unsloth.py

Dataset source: https://huggingface.co/datasets/jjzha/skillspan
License: CC-BY-4.0 (free for commercial use! 💃)
Paper: "SkillSpan: Hard and Soft Skill Extraction from English Job Postings"

The SkillSpan dataset is a SUPERSTAR for skills extraction:
  - 11,543 sentences from StackOverflow (tech) + STAR (house/construction)
  - Two tag tracks: tags_skill (soft/hard skills) + tags_knowledge (technical knowledge)
  - BIO format — same convention AiMerlion already uses! 🎯
  - Pre-split into train/dev/test (4,800 / 3,174 / 3,569)

BUT — it labels JOB POSTINGS, not resumes! The skill vocabulary transfers
beautifully though. Think of it as learning vocabulary from a DICTIONARY
before reading the novel — the words are the same, just the context shifts! 📖

Entity Mapping (SkillSpan → AiMerlion):
  ┌─────────────────────────┬──────────────────────────────────┐
  │ SKILLSPAN TAG           │ AIMERLION ENTITY                 │
  ├─────────────────────────┼──────────────────────────────────┤
  │ tags_skill (B/I)        │ HARD_SKILL                       │
  │ tags_knowledge (B/I)    │ TECHNICAL_KNOWLEDGE              │
  │ (merged mode)           │ SKILL (both combined)            │
  └─────────────────────────┴──────────────────────────────────┘

Usage:
    # Download + import into SQLite for spaCy training
    python import_skillspan.py --db resume_extractions.db

    # Also generate ShareGPT JSONL for LLM fine-tuning
    python import_skillspan.py --db resume_extractions.db --jsonl

    # Preview first N documents without writing
    python import_skillspan.py --preview 3

    # Use merged SKILL label instead of separate HARD_SKILL/TECHNICAL_KNOWLEDGE
    python import_skillspan.py --db resume_extractions.db --merge-labels

    # Only import the "tech" source (StackOverflow, more relevant for IT resumes)
    python import_skillspan.py --db resume_extractions.db --source tech

    # Filter by minimum entity density (skip mostly-O sentences)
    python import_skillspan.py --db resume_extractions.db --min-entities 1

Dependencies:
    pip install requests --break-system-packages
"""

import os
import sys
import json
import argparse
import logging
import sqlite3
import hashlib
import random
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ── Optional: requests for downloading ────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# 🎨 CONSOLE — Pretty output matching AiMerlion's signature style
# =============================================================================

class Console:
    """Pretty console output matching the AiMerlion pipeline style."""

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
        print(f"\n{'=' * width}")
        print(f"  {Console.BOLD}{Console.HEADER}{text}{Console.RESET}")
        print(f"{'=' * width}")

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
    def stat(label: str, value: Any, indent: int = 2):
        spaces = "  " * indent
        print(f"{spaces}{Console.DIM}{label}:{Console.RESET} {Console.BOLD}{value}{Console.RESET}")

    @staticmethod
    def progress_bar(current: int, total: int, width: int = 30):
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        pct = (current / total * 100) if total > 0 else 0
        print(f"\r  [{bar}] {pct:.0f}% ({current}/{total})", end="", flush=True)
        if current >= total:
            print()  # newline when done


# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 💼 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HuggingFace direct download URLs for the dataset splits
DATASET_URLS = {
    "train": "https://huggingface.co/datasets/jjzha/skillspan/resolve/main/train.json",
    "dev":   "https://huggingface.co/datasets/jjzha/skillspan/resolve/main/dev.json",
    "test":  "https://huggingface.co/datasets/jjzha/skillspan/resolve/main/test.json",
}

# Default cache directory for downloaded files
DEFAULT_CACHE_DIR = "training_data/skillspan_cache"

# Candidate ID offset — using 960,000+ to avoid collision with:
#   Real data:     1-899,999
#   Kaggle NER:    900,000-949,999
#   Majinuub:      950,000-959,999
#   SkillSpan:     960,000+ (this import!)
CANDIDATE_ID_OFFSET = 960_000

# ── AiMerlion system prompt (MUST match prepare_training_data.py) ──────
SYSTEM_PROMPT = (
    "You are a precise resume data extraction assistant specializing in "
    "Singapore and Malaysia resumes. Extract the requested information "
    "and return valid JSON only. Handle 8-digit phone numbers (+65/+60), "
    "date-first experience formats, local company names, and multilingual "
    "content accurately."
)

# ── Instruction template for skill extraction ──────────────────────────
SKILL_INSTRUCTION_TEMPLATE = (
    "Extract all skills and technical knowledge mentioned in this text. "
    "Return valid JSON with these fields: "
    "hard_skills (array of skill phrases), "
    "technical_knowledge (array of knowledge/technology phrases).\n\n"
    "Text:\n{text}"
)


# =============================================================================
# 📊 DATA MODEL — Intermediate representation for a SkillSpan document
#
# A "document" in SkillSpan is a group of sentences sharing the same `idx`.
# We assemble them into a single text with char-offset annotations.
# =============================================================================

@dataclass
class SkillSpanDocument:
    """One complete job posting assembled from SkillSpan sentence rows."""
    idx: int                              # SkillSpan document index
    source: str                           # "tech" or "house"
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""                    # Assembled full text
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    # Stats for filtering
    total_tokens: int = 0
    skill_count: int = 0
    knowledge_count: int = 0

    @property
    def candidate_id(self) -> int:
        """Generate a unique candidate ID in the 960,000+ range."""
        return CANDIDATE_ID_OFFSET + self.idx

    @property
    def doc_id(self) -> str:
        """Generate a doc_id matching AiMerlion convention."""
        return f"skillspan_{self.idx:06d}"


# =============================================================================
# 📥 PHASE 1: DOWNLOAD DATASET SPLITS
# =============================================================================

def download_split(split: str, cache_dir: str) -> str:
    """
    📥 Download one split (train/dev/test) from HuggingFace.

    Think of this as ordering costume pieces from the supplier —
    we need all three acts before the show can go on! 🎭📦

    Args:
        split:      Which split to download ("train", "dev", "test")
        cache_dir:  Where to cache the downloaded file

    Returns:
        Local file path to the downloaded JSON

    Raises:
        SystemExit: If download fails and no cached copy exists
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{split}.json")

    # ── Skip download if cached ────────────────────────────────────
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        if file_size > 1000:  # Sanity check: file isn't empty/corrupt
            Console.info(f"Using cached {split}.json ({file_size:,} bytes)")
            return local_path
        else:
            Console.warning(f"Cached {split}.json is too small ({file_size} bytes), re-downloading")

    # ── Download from HuggingFace ──────────────────────────────────
    if not HAS_REQUESTS:
        Console.error(
            "requests library not installed! Run:\n"
            "    pip install requests --break-system-packages\n"
            f"Or manually download {DATASET_URLS[split]}\n"
            f"    and save to {local_path}"
        )
        sys.exit(1)

    url = DATASET_URLS[split]
    Console.info(f"Downloading {split}.json from HuggingFace...")

    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        # Stream to file to handle large downloads gracefully
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(local_path)
        Console.success(f"Downloaded {split}.json ({file_size:,} bytes)")
        return local_path

    except requests.exceptions.RequestException as e:
        Console.error(f"Download failed for {split}: {e}")
        if os.path.exists(local_path):
            Console.warning("Using stale cached copy (may be outdated)")
            return local_path
        sys.exit(1)


def download_all_splits(cache_dir: str) -> Dict[str, str]:
    """
    Download all three splits and return {split_name: file_path}.

    Returns:
        Dict mapping split names to local file paths
    """
    paths = {}
    for split in ["train", "dev", "test"]:
        paths[split] = download_split(split, cache_dir)
    return paths


# =============================================================================
# 📂 PHASE 2: LOAD & PARSE THE DATASET
#
# SkillSpan's JSON is JSONL-like: each line is a JSON object representing
# one sentence. Sentences sharing the same `idx` belong to the same
# job posting. We group them back into documents.
# =============================================================================

def load_split(file_path: str) -> List[Dict]:
    """
    📂 Load a SkillSpan split file.

    The files are JSON arrays of objects (not JSONL), where each object
    is one sentence with tokens and BIO tags.

    Edge cases handled:
      - File might be a JSON array OR line-delimited JSON
      - Malformed entries are skipped with a warning
      - Empty files return an empty list

    Args:
        file_path:  Path to the JSON file

    Returns:
        List of sentence dicts with keys: idx, tokens, tags_skill,
        tags_knowledge, source
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # ── Try JSON array first (the expected format) ─────────────
        if content.startswith("["):
            data = json.loads(content)
            if isinstance(data, list):
                return data

        # ── Fall back to line-delimited JSON ───────────────────────
        data = []
        for line_num, line in enumerate(content.split("\n"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed line {line_num} in {file_path}: {e}")
        return data

    except FileNotFoundError:
        Console.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        Console.error(f"Invalid JSON in {file_path}: {e}")
        return []


def group_sentences_into_documents(
    sentences: List[Dict],
    source_filter: Optional[str] = None,
    min_entities: int = 0
) -> List[SkillSpanDocument]:
    """
    🏗️ Group sentence-level records into document-level SkillSpanDocument objects.

    SkillSpan stores one sentence per JSON row. The `idx` field groups
    sentences that belong to the same original job posting. We reassemble
    the full posting text and compute character offsets for each entity span.

    This is like reconstructing a shredded document — we know the pieces
    belong together because they share the same serial number! 🔍📄

    The CHARACTER OFFSET calculation is the trickiest part:
    We reconstruct the full text by joining sentences with newlines,
    then track the cumulative character position to compute offsets
    for each BIO-tagged entity span.

    Args:
        sentences:      List of sentence dicts from load_split()
        source_filter:  If set, only include sentences from this source ("tech"/"house")
        min_entities:   Skip documents with fewer than this many entity spans

    Returns:
        List of SkillSpanDocument objects with char-offset annotations
    """
    # ── Step 1: Group by document idx ──────────────────────────────
    doc_groups = defaultdict(list)
    for sent in sentences:
        idx = sent.get("idx", 0)
        source = sent.get("source", "unknown")

        # Apply source filter if specified
        if source_filter and source != source_filter:
            continue

        doc_groups[idx].append(sent)

    # ── Step 2: Assemble each document ─────────────────────────────
    documents = []
    skipped_low_entity = 0

    for idx in sorted(doc_groups.keys()):
        sents = doc_groups[idx]
        source = sents[0].get("source", "unknown") if sents else "unknown"

        doc = SkillSpanDocument(idx=idx, source=source)

        # Rebuild full text and compute character offsets
        char_offset = 0
        assembled_lines = []

        for sent in sents:
            tokens = sent.get("tokens", [])
            tags_skill = sent.get("tags_skill", [])
            tags_knowledge = sent.get("tags_knowledge", [])

            if not tokens:
                continue

            # ── Validate tag alignment ─────────────────────────
            # Edge case: tags array might be shorter than tokens
            # (shouldn't happen in clean data, but defensive coding!)
            if len(tags_skill) != len(tokens):
                logger.warning(
                    f"Doc {idx}: tags_skill length ({len(tags_skill)}) != "
                    f"tokens length ({len(tokens)}). Padding with O."
                )
                tags_skill = tags_skill + ["O"] * (len(tokens) - len(tags_skill))
                tags_skill = tags_skill[:len(tokens)]

            if len(tags_knowledge) != len(tokens):
                logger.warning(
                    f"Doc {idx}: tags_knowledge length ({len(tags_knowledge)}) != "
                    f"tokens length ({len(tokens)}). Padding with O."
                )
                tags_knowledge = tags_knowledge + ["O"] * (len(tokens) - len(tags_knowledge))
                tags_knowledge = tags_knowledge[:len(tokens)]

            # ── Reconstruct sentence text with token positions ──
            # We join tokens with spaces (SkillSpan's tokenization is
            # whitespace-based, so this is a faithful reconstruction)
            sentence_text = " ".join(tokens)

            # Track individual token char positions within this sentence
            token_positions = []
            pos = 0
            for token in tokens:
                token_positions.append((pos, pos + len(token)))
                pos += len(token) + 1  # +1 for the space separator

            # ── Extract entity spans from BIO tags ─────────────
            # For BOTH tag tracks: tags_skill and tags_knowledge
            for tag_track, entity_type in [
                (tags_skill, "HARD_SKILL"),
                (tags_knowledge, "TECHNICAL_KNOWLEDGE")
            ]:
                span_start = None
                span_label = None

                for i, tag in enumerate(tag_track):
                    if tag == "B":
                        # Start a new entity span
                        if span_start is not None:
                            # Close the previous span first
                            abs_start = char_offset + token_positions[span_start][0]
                            abs_end = char_offset + token_positions[i - 1][1]
                            span_text = sentence_text[
                                token_positions[span_start][0]:token_positions[i - 1][1]
                            ]
                            doc.annotations.append({
                                "entity_type": entity_type,
                                "char_start": abs_start,
                                "char_end": abs_end,
                                "text": span_text,
                            })
                            if entity_type == "HARD_SKILL":
                                doc.skill_count += 1
                            else:
                                doc.knowledge_count += 1

                        span_start = i

                    elif tag == "I":
                        # Continue current span — if no B preceded, treat as B
                        # (orphaned I-tag edge case)
                        if span_start is None:
                            logger.debug(
                                f"Doc {idx}: Orphaned I-tag at token {i} "
                                f"('{tokens[i]}'), treating as B"
                            )
                            span_start = i

                    elif tag == "O":
                        # Close any open span
                        if span_start is not None:
                            abs_start = char_offset + token_positions[span_start][0]
                            abs_end = char_offset + token_positions[i - 1][1]
                            span_text = sentence_text[
                                token_positions[span_start][0]:token_positions[i - 1][1]
                            ]
                            doc.annotations.append({
                                "entity_type": entity_type,
                                "char_start": abs_start,
                                "char_end": abs_end,
                                "text": span_text,
                            })
                            if entity_type == "HARD_SKILL":
                                doc.skill_count += 1
                            else:
                                doc.knowledge_count += 1
                            span_start = None

                # Close any span left open at end of sentence
                if span_start is not None:
                    abs_start = char_offset + token_positions[span_start][0]
                    abs_end = char_offset + token_positions[-1][1]
                    span_text = sentence_text[
                        token_positions[span_start][0]:token_positions[-1][1]
                    ]
                    doc.annotations.append({
                        "entity_type": entity_type,
                        "char_start": abs_start,
                        "char_end": abs_end,
                        "text": span_text,
                    })
                    if entity_type == "HARD_SKILL":
                        doc.skill_count += 1
                    else:
                        doc.knowledge_count += 1

            doc.total_tokens += len(tokens)
            assembled_lines.append(sentence_text)

            # Move the char offset past this sentence + newline
            char_offset += len(sentence_text) + 1  # +1 for '\n'

        doc.raw_text = "\n".join(assembled_lines)
        doc.sentences = sents

        # ── Apply minimum entity filter ────────────────────────────
        total_entities = doc.skill_count + doc.knowledge_count
        if total_entities < min_entities:
            skipped_low_entity += 1
            continue

        # ── Clean and validate char offsets ─────────────────────────
        # This is our "measure twice, cut once" moment! 📐✂️
        # spaCy's E024 error happens when spans have leading/trailing
        # whitespace or punctuation. We strip them here so the parser
        # doesn't throw a tantrum on the runway! 💅👠
        validation_ok = True
        cleaned_annotations = []
        for ann in doc.annotations:
            start = ann["char_start"]
            end = ann["char_end"]
            text = ann["text"]

            # ── Strip leading whitespace/punctuation from span ─────
            while start < end and doc.raw_text[start] in " \t\n\r.,;:!?()[]{}\"'":
                start += 1

            # ── Strip trailing whitespace/punctuation from span ────
            while end > start and doc.raw_text[end - 1] in " \t\n\r.,;:!?()[]{}\"'":
                end -= 1

            # Skip empty spans after stripping
            if start >= end:
                logger.debug(f"Doc {idx}: Span became empty after stripping: '{text}'")
                continue

            # Update the annotation with cleaned offsets
            ann["char_start"] = start
            ann["char_end"] = end
            ann["text"] = doc.raw_text[start:end]

            # Final validation
            extracted = doc.raw_text[start:end]
            if extracted != ann["text"]:
                logger.warning(
                    f"Doc {idx}: Offset mismatch! "
                    f"Expected '{ann['text']}' but got '{extracted}' "
                    f"at [{start}:{end}]"
                )
                validation_ok = False
            else:
                cleaned_annotations.append(ann)

        doc.annotations = cleaned_annotations

        if not validation_ok:
            Console.warning(
                f"Doc {idx} has offset mismatches — check tokenization. "
                "Importing anyway (spaCy will re-tokenize)."
            )

        documents.append(doc)

    if skipped_low_entity > 0:
        Console.info(
            f"Skipped {skipped_low_entity} documents with < {min_entities} entities"
        )

    return documents


# =============================================================================
# 🔄 PHASE 3: OPTIONAL LABEL MERGING
#
# SkillSpan distinguishes between "skills" (abilities like "design tests")
# and "knowledge" (technologies like "Python", "Docker"). AiMerlion can
# benefit from keeping them separate OR merging them into one SKILL label.
# =============================================================================

def merge_labels(documents: List[SkillSpanDocument], target_label: str = "SKILL"):
    """
    🔄 Merge HARD_SKILL and TECHNICAL_KNOWLEDGE into a single label.

    Sometimes you want ONE category instead of two — like combining
    "warm colors" and "cool colors" into just "colors" when the
    distinction doesn't matter for your use case. 🎨

    This modifies documents in-place.

    Args:
        documents:     List of SkillSpanDocument objects to modify
        target_label:  The unified label name (default: "SKILL")
    """
    for doc in documents:
        for ann in doc.annotations:
            ann["entity_type"] = target_label


# =============================================================================
# 💾 PHASE 4A: IMPORT INTO SQLITE (for spaCy NER via train_ner.py)
#
# This stores documents and annotations into the same tables that the
# annotation_tool.py uses, so train_ner.py can pick them up seamlessly.
# =============================================================================

def import_to_sqlite(
    documents: List[SkillSpanDocument],
    db_path: str,
    split_name: str = "train"
) -> Dict[str, int]:
    """
    💾 Import SkillSpan documents into AiMerlion's SQLite database.

    Writes to:
      - ner_documents: One row per job posting (text + metadata)
      - ner_annotations: One row per entity span (char offsets + label)

    Uses candidate IDs starting at 960,000 to avoid collision with:
      - Real resumes (1-899,999)
      - Kaggle imports (900,000-949,999)
      - Majinuub imports (950,000-959,999)

    Args:
        documents:   Parsed SkillSpanDocument objects
        db_path:     Path to resume_extractions.db
        split_name:  "train", "dev", or "test" (stored as metadata)

    Returns:
        Dict with import statistics
    """
    Console.banner(f"💾 Importing {split_name} split into SQLite")

    if not os.path.exists(db_path):
        Console.error(f"Database not found: {db_path}")
        Console.info("Run your extraction pipeline first to create the database.")
        return {"error": "Database not found"}

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for safety

    # ── Ensure tables exist ────────────────────────────────────────
    # We use IF NOT EXISTS so this doesn't break if tables already
    # exist from annotation_tool.py setup
    # ── Don't try to CREATE — the table already exists from
    # annotation_tool.py with a DIFFERENT schema! The real table
    # doesn't have raw_text (that lives in raw_extractions).
    # We need to INSERT into raw_extractions too! ──────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_extractions (
            candidate_id INTEGER PRIMARY KEY,
            raw_text TEXT,
            file_name TEXT DEFAULT '',
            extraction_timestamp TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ner_annotations (
            annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            text TEXT,
            layer INTEGER DEFAULT 0,
            annotator TEXT DEFAULT 'skillspan_import',
            created_at TEXT,
            FOREIGN KEY (doc_id) REFERENCES ner_documents(doc_id)
        )
    """)

    conn.commit()

    # ── Import documents ───────────────────────────────────────────
    stats = {
        "docs_inserted": 0,
        "docs_skipped": 0,
        "annotations_inserted": 0,
        "errors": 0,
    }

    now = datetime.now().isoformat()

    for i, doc in enumerate(documents):
        doc_id = doc.doc_id

        # ── Check for duplicate doc_id ─────────────────────────────
        existing = conn.execute(
            "SELECT doc_id FROM ner_documents WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()

        if existing:
            stats["docs_skipped"] += 1
            continue

        # ── Insert document ────────────────────────────────────────
        metadata = json.dumps({
            "source_dataset": "skillspan",
            "source_split": split_name,
            "source_type": doc.source,
            "skill_count": doc.skill_count,
            "knowledge_count": doc.knowledge_count,
            "total_tokens": doc.total_tokens,
            "import_date": now,
            "license": "CC-BY-4.0",
        })

        try:
            # Step A: Store the raw text in raw_extractions
            # (This is WHERE your schema keeps full text, darling!)
            conn.execute(
                """INSERT OR IGNORE INTO raw_extractions
                   (candidate_id, folder_path, filenames, raw_text,
                    resume_language, text_length, extraction_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc.candidate_id, "skillspan_import",
                 f"skillspan_{doc.idx}.txt", doc.raw_text,
                 "English", len(doc.raw_text), now)
            )

            # Step B: Store the document metadata in ner_documents
            conn.execute(
                """INSERT INTO ner_documents
                   (doc_id, candidate_id, status, annotator, notes,
                    function, industry)
                   VALUES (?, ?, 'completed', 'skillspan_import', ?, '', '')""",
                (doc_id, doc.candidate_id, metadata)
            )
            stats["docs_inserted"] += 1

            # ── Insert annotations ─────────────────────────────────
            for ann in doc.annotations:
                conn.execute(
                    """INSERT INTO ner_annotations
                       (candidate_id, doc_id, entity_type, char_start, char_end,
                        text_content, layer, confidence, annotator)
                       VALUES (?, ?, ?, ?, ?, ?, 0, 1.0, 'skillspan_import')""",
                    (doc.candidate_id, doc_id, ann["entity_type"],
                     ann["char_start"], ann["char_end"], ann["text"])
                )
                stats["annotations_inserted"] += 1

        except sqlite3.Error as e:
            logger.warning(f"Error importing doc {doc_id}: {e}")
            stats["errors"] += 1

        # Progress bar every 10 documents
        if (i + 1) % 10 == 0 or i == len(documents) - 1:
            Console.progress_bar(i + 1, len(documents))

    conn.commit()
    conn.close()

    Console.success(f"Inserted {stats['docs_inserted']} documents")
    Console.success(f"Inserted {stats['annotations_inserted']} annotations")
    if stats["docs_skipped"] > 0:
        Console.info(f"Skipped {stats['docs_skipped']} already-imported documents")
    if stats["errors"] > 0:
        Console.warning(f"{stats['errors']} errors during import")

    return stats


# =============================================================================
# 📝 PHASE 4B: GENERATE SHAREGPT JSONL (for LLM fine-tuning)
#
# Converts SkillSpan documents into the same ShareGPT format used by
# finetune_unsloth.py, so you can fine-tune your Llama 3.2 model to
# extract skills from any text!
# =============================================================================

def generate_sharegpt_jsonl(
    documents: List[SkillSpanDocument],
    output_path: str,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[str, str, int, int]:
    """
    📝 Convert SkillSpan documents to ShareGPT JSONL for Unsloth training.

    Each document becomes a conversation:
      - system: AiMerlion's standard extraction prompt
      - human:  The job posting text
      - gpt:    The extracted skills as structured JSON

    Args:
        documents:    Parsed SkillSpanDocument objects
        output_path:  Base path for the output JSONL file
        val_split:    Fraction to hold out for validation (default: 10%)
        seed:         Random seed for reproducibility

    Returns:
        Tuple of (train_path, val_path, train_count, val_count)
    """
    Console.banner("📝 Generating ShareGPT JSONL")

    conversations = []

    for doc in documents:
        # ── Build the expected output JSON ─────────────────────────
        hard_skills = []
        tech_knowledge = []

        for ann in doc.annotations:
            text = ann["text"].strip()
            if not text:
                continue

            if ann["entity_type"] in ("HARD_SKILL", "SKILL"):
                hard_skills.append(text)
            elif ann["entity_type"] == "TECHNICAL_KNOWLEDGE":
                tech_knowledge.append(text)

        # Skip documents with no extracted entities (all-O sentences)
        if not hard_skills and not tech_knowledge:
            continue

        # Deduplicate while preserving order
        hard_skills = list(dict.fromkeys(hard_skills))
        tech_knowledge = list(dict.fromkeys(tech_knowledge))

        output_json = {}
        if hard_skills:
            output_json["hard_skills"] = hard_skills
        if tech_knowledge:
            output_json["technical_knowledge"] = tech_knowledge

        # ── Build the conversation ─────────────────────────────────
        instruction = SKILL_INSTRUCTION_TEMPLATE.format(text=doc.raw_text)

        sharegpt_entry = {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": json.dumps(output_json, indent=2)},
            ]
        }
        conversations.append(sharegpt_entry)

    if not conversations:
        Console.error("No conversations generated! Check entity counts.")
        return ("", "", 0, 0)

    # ── Split into train/val ───────────────────────────────────────
    random.seed(seed)
    random.shuffle(conversations)

    val_count = max(1, int(len(conversations) * val_split))
    train_data = conversations[val_count:]
    val_data = conversations[:val_count]

    # ── Write JSONL files ──────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    train_path = output_path
    val_path = output_path.replace(".jsonl", "_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    Console.success(f"Training JSONL: {train_path} ({len(train_data)} examples)")
    Console.success(f"Validation JSONL: {val_path} ({len(val_data)} examples)")

    return (train_path, val_path, len(train_data), len(val_data))


# =============================================================================
# 📊 PHASE 5: DATASET ANALYSIS & REPORTING
# =============================================================================

def analyze_documents(documents: List[SkillSpanDocument]) -> Dict[str, Any]:
    """
    📊 Analyze the parsed SkillSpan documents and print a summary.

    Like a dress rehearsal REVIEW — we want to know what we're working
    with before the big show! 👗📋

    Args:
        documents:  List of parsed SkillSpanDocument objects

    Returns:
        Dict with analysis statistics
    """
    Console.banner("📊 Dataset Analysis")

    stats = {
        "total_documents": len(documents),
        "total_annotations": sum(len(d.annotations) for d in documents),
        "total_tokens": sum(d.total_tokens for d in documents),
        "sources": Counter(),
        "entity_types": Counter(),
        "skill_examples": [],
        "knowledge_examples": [],
        "docs_with_entities": 0,
        "docs_without_entities": 0,
    }

    for doc in documents:
        stats["sources"][doc.source] += 1
        has_entities = False

        for ann in doc.annotations:
            stats["entity_types"][ann["entity_type"]] += 1
            has_entities = True

            # Collect sample spans for display
            text = ann["text"].strip()
            if ann["entity_type"] == "HARD_SKILL" and len(stats["skill_examples"]) < 10:
                stats["skill_examples"].append(text)
            elif ann["entity_type"] == "TECHNICAL_KNOWLEDGE" and len(stats["knowledge_examples"]) < 10:
                stats["knowledge_examples"].append(text)

        if has_entities:
            stats["docs_with_entities"] += 1
        else:
            stats["docs_without_entities"] += 1

    # ── Print summary ──────────────────────────────────────────────
    Console.stat("Total documents", stats["total_documents"])
    Console.stat("Total annotations", stats["total_annotations"])
    Console.stat("Total tokens", f"{stats['total_tokens']:,}")
    Console.stat("Docs with entities", stats["docs_with_entities"])
    Console.stat("Docs without entities", stats["docs_without_entities"])

    print()
    Console.info("Source distribution:")
    for source, count in stats["sources"].most_common():
        Console.stat(f"  {source}", count, indent=2)

    print()
    Console.info("Entity type counts:")
    for etype, count in stats["entity_types"].most_common():
        Console.stat(f"  {etype}", count, indent=2)

    if stats["skill_examples"]:
        print()
        Console.info("Sample HARD_SKILL spans:")
        for ex in stats["skill_examples"][:5]:
            print(f"    • \"{ex}\"")

    if stats["knowledge_examples"]:
        print()
        Console.info("Sample TECHNICAL_KNOWLEDGE spans:")
        for ex in stats["knowledge_examples"][:5]:
            print(f"    • \"{ex}\"")

    return stats


# =============================================================================
# 👀 PREVIEW MODE
# =============================================================================

def preview_documents(documents: List[SkillSpanDocument], num_examples: int = 3):
    """
    👀 Preview assembled documents without importing.

    Like a fitting room try-on — see how the data looks before committing! 👗

    Args:
        documents:     Parsed SkillSpanDocument objects
        num_examples:  How many to show
    """
    Console.banner(f"👀 Preview: First {num_examples} Documents")

    for doc in documents[:num_examples]:
        print(f"\n  ────── Document {doc.doc_id} (source: {doc.source}) ──────")
        print(f"  Candidate ID: {doc.candidate_id}")
        print(f"  Tokens: {doc.total_tokens} | Skills: {doc.skill_count} | Knowledge: {doc.knowledge_count}")

        # Show first 300 chars of text
        text_preview = doc.raw_text[:300]
        if len(doc.raw_text) > 300:
            text_preview += "... [truncated]"
        print(f"\n  TEXT:")
        for line in text_preview.split("\n")[:6]:
            print(f"    {line}")

        # Show annotations
        if doc.annotations:
            print(f"\n  ANNOTATIONS ({len(doc.annotations)} total):")
            for ann in doc.annotations[:8]:
                print(
                    f"    [{ann['entity_type']:25s}] "
                    f"chars {ann['char_start']:5d}-{ann['char_end']:5d}: "
                    f"\"{ann['text']}\""
                )
            if len(doc.annotations) > 8:
                print(f"    ... and {len(doc.annotations) - 8} more")
        print()


# =============================================================================
# 🧹 CLEANUP — Remove previously imported SkillSpan data
# =============================================================================

def cleanup_previous_import(db_path: str) -> int:
    """
    🧹 Remove all previously imported SkillSpan data from the database.

    Useful when you want a fresh re-import (e.g., after changing
    --merge-labels or --source filter settings).

    Args:
        db_path:  Path to resume_extractions.db

    Returns:
        Number of documents removed
    """
    Console.banner("🧹 Cleaning Up Previous Import")

    if not os.path.exists(db_path):
        Console.warning("Database not found, nothing to clean")
        return 0

    conn = sqlite3.connect(db_path)

    # Count what we'll remove
    count = conn.execute(
        "SELECT COUNT(*) FROM ner_documents WHERE doc_id LIKE 'skillspan_%'"
    ).fetchone()[0]

    if count == 0:
        Console.info("No previous SkillSpan data found — nothing to clean!")
        conn.close()
        return 0

    # Remove annotations first (foreign key constraint)
    conn.execute(
        "DELETE FROM ner_annotations WHERE doc_id LIKE 'skillspan_%'"
    )
    conn.execute(
        "DELETE FROM ner_documents WHERE doc_id LIKE 'skillspan_%'"
    )
    conn.commit()
    conn.close()

    Console.success(f"Removed {count} previously imported SkillSpan documents")
    return count


# =============================================================================
# 🎬 MAIN — THE GRAND PRODUCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "💅✨ Fairy Codemother's SkillSpan Importer ✨💅\n"
            "Downloads and converts the SkillSpan dataset (CC-BY-4.0)\n"
            "into AiMerlion's SQLite + JSONL training formats.\n\n"
            "Dataset: https://huggingface.co/datasets/jjzha/skillspan\n"
            "Paper:   Zhang et al., NAACL 2022"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required arguments ─────────────────────────────────────────
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to resume_extractions.db (required for SQLite import)"
    )

    # ── Optional arguments ─────────────────────────────────────────
    parser.add_argument(
        "--jsonl", action="store_true",
        help="Also generate ShareGPT JSONL for LLM fine-tuning"
    )
    parser.add_argument(
        "--jsonl-output", type=str,
        default=os.path.join("training_data", "skillspan_train.jsonl"),
        help="Output path for ShareGPT JSONL (default: training_data/skillspan_train.jsonl)"
    )
    parser.add_argument(
        "--merge-labels", action="store_true",
        help="Merge HARD_SKILL and TECHNICAL_KNOWLEDGE into a single SKILL label"
    )
    parser.add_argument(
        "--source", type=str, default=None, choices=["tech", "house"],
        help="Only import from this source (default: both)"
    )
    parser.add_argument(
        "--min-entities", type=int, default=0,
        help="Skip documents with fewer than N entity spans (default: 0 = import all)"
    )
    parser.add_argument(
        "--preview", type=int, default=0, metavar="N",
        help="Preview first N documents without importing"
    )
    parser.add_argument(
        "--splits", type=str, nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Which splits to import (default: all three)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=DEFAULT_CACHE_DIR,
        help=f"Cache directory for downloads (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove previously imported SkillSpan data before importing"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="JSONL validation split ratio (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # ── Validate arguments ─────────────────────────────────────────
    if not args.db and args.preview == 0 and not args.jsonl:
        parser.print_help()
        print("\n💡 Tip: Use --db resume_extractions.db to import, or --preview 3 to peek!")
        return

    # ── Grand opening banner ───────────────────────────────────────
    print()
    print("═" * 60)
    print("  💅✨ SKILLSPAN DATASET IMPORTER ✨💅")
    print("  HuggingFace → AiMerlion Training Pipeline")
    print("═" * 60)
    if args.db:
        print(f"  📂 Database:     {args.db}")
    print(f"  📦 Splits:       {', '.join(args.splits)}")
    print(f"  🏷️  Labels:       {'SKILL (merged)' if args.merge_labels else 'HARD_SKILL + TECHNICAL_KNOWLEDGE'}")
    if args.source:
        print(f"  🔍 Source:       {args.source} only")
    if args.min_entities > 0:
        print(f"  📊 Min entities: {args.min_entities}")
    if args.jsonl:
        print(f"  📝 JSONL output: {args.jsonl_output}")
    print("═" * 60)

    # ── Step 1: Download dataset splits ────────────────────────────
    Console.banner("📥 Phase 1: Downloading Dataset")
    split_paths = {}
    for split in args.splits:
        split_paths[split] = download_split(split, args.cache_dir)

    # ── Step 2: Load and parse all splits ──────────────────────────
    Console.banner("📂 Phase 2: Loading & Parsing")
    all_documents = []

    for split, path in split_paths.items():
        Console.info(f"Loading {split} split...")
        sentences = load_split(path)
        Console.stat(f"  {split} sentences", len(sentences))

        documents = group_sentences_into_documents(
            sentences,
            source_filter=args.source,
            min_entities=args.min_entities,
        )
        Console.stat(f"  {split} documents", len(documents))

        # Tag documents with their split for metadata tracking
        for doc in documents:
            doc._split = split  # Temporary attribute for import tracking

        all_documents.extend(documents)

    Console.success(f"Total: {len(all_documents)} documents across {len(args.splits)} splits")

    # ── Step 3: Merge labels (optional) ────────────────────────────
    if args.merge_labels:
        Console.info("Merging HARD_SKILL + TECHNICAL_KNOWLEDGE → SKILL")
        merge_labels(all_documents, target_label="SKILL")

    # ── Step 4: Analyze ────────────────────────────────────────────
    stats = analyze_documents(all_documents)

    # ── Step 5: Preview mode (exit early) ──────────────────────────
    if args.preview > 0:
        preview_documents(all_documents, num_examples=args.preview)
        print("  💡 Preview complete! Add --db resume_extractions.db to import.")
        return

    # ── Step 6: Clean previous import (optional) ───────────────────
    if args.clean and args.db:
        cleanup_previous_import(args.db)

    # ── Step 7: Import to SQLite ───────────────────────────────────
    import_stats = {}
    if args.db:
        # Import each split separately for metadata tracking
        for split in args.splits:
            split_docs = [d for d in all_documents if getattr(d, '_split', '') == split]
            if split_docs:
                split_stats = import_to_sqlite(split_docs, args.db, split_name=split)
                import_stats[split] = split_stats

    # ── Step 8: Generate JSONL (optional) ──────────────────────────
    jsonl_stats = None
    if args.jsonl:
        # Only use training split for JSONL (dev/test reserved for eval)
        train_docs = [d for d in all_documents if getattr(d, '_split', '') == "train"]
        if not train_docs:
            train_docs = all_documents  # Fallback if no split info
            Console.warning("No split info found, using all documents for JSONL")

        train_path, val_path, train_n, val_n = generate_sharegpt_jsonl(
            train_docs,
            args.jsonl_output,
            val_split=args.val_split,
            seed=args.seed,
        )
        jsonl_stats = {
            "train_path": train_path,
            "val_path": val_path,
            "train_count": train_n,
            "val_count": val_n,
        }

    # ── Final summary ──────────────────────────────────────────────
    Console.banner("🎯 Import Complete!")

    total_docs = sum(
        s.get("docs_inserted", 0) for s in import_stats.values()
    ) if import_stats else 0
    total_anns = sum(
        s.get("annotations_inserted", 0) for s in import_stats.values()
    ) if import_stats else 0

    if total_docs > 0:
        Console.stat("Documents imported", total_docs)
        Console.stat("Annotations imported", total_anns)

    # ── Next steps ─────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  🎬 WHAT'S NEXT?")
    print("═" * 60)

    if args.db and total_docs > 0:
        print("  1. Train your spaCy NER model with the new data:")
        print(f"     python train_ner.py --db {args.db} --all --status completed")
        print()
        print("  2. The SkillSpan entities will appear as:")
        if args.merge_labels:
            print("     SKILL (combined soft/hard skills + technical knowledge)")
        else:
            print("     HARD_SKILL (abilities like 'design end-to-end tests')")
            print("     TECHNICAL_KNOWLEDGE (technologies like 'Python', 'Docker')")

    if jsonl_stats and jsonl_stats.get("train_path"):
        print()
        print("  3. Fine-tune Llama for skill extraction:")
        print(f"     python finetune_unsloth.py \\")
        print(f"       --train-file {jsonl_stats['train_path']} \\")
        print(f"       --val-file {jsonl_stats['val_path']}")

    print()
    print("  💡 TIP: To merge with your SG/MY data for combined training:")
    print("     python train_ner.py --db resume_extractions.db --all --force")
    print("     (SkillSpan + your annotations train TOGETHER! 🤝)")

    print()
    print("═" * 60)
    print("  💅✨ The Fairy Codemother is PROUD of you! ✨💅")
    print(f"  📜 License: CC-BY-4.0 — cite Zhang et al., NAACL 2022")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()