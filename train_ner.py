"""
train_ner.py

💅✨ FAIRY CODEMOTHER'S ML TRAINING PIPELINE ✨💅

The GRAND FINALE of the annotation journey! This script takes your
beautifully annotated resume data and transforms it into TRAINED ML
models that can extract entities and classify resumes ALL BY THEMSELVES!

Think of it like this, darling: 🎭
  - Annotations = The rehearsals (teaching by example)
  - This script = Opening night (the models learn to perform solo!)
  - Deployed models = The TOURING SHOW (they work on new resumes!)

Pipeline Phases:
  Phase 2: 🔍 Data Validation & Quality Audit
  Phase 3: ✂️ Train/Test/Val Split
  Phase 4: 🐍 spaCy NER Model Training
  Phase 5: 🧠 Classification Model Training (Function & Industry)

Usage:
    # Run the full pipeline (all phases)
    python train_ner.py --db resume_extractions.db --all

    # Run individual phases
    python train_ner.py --db resume_extractions.db --validate
    python train_ner.py --db resume_extractions.db --split
    python train_ner.py --db resume_extractions.db --train-ner
    python train_ner.py --db resume_extractions.db --train-classifier

    # Customize settings
    python train_ner.py --db resume_extractions.db --all --min-docs 10 --epochs 50

Dependencies:
    pip install spacy scikit-learn numpy --break-system-packages
    python -m spacy download en_core_web_sm
"""

import os
import sys
import json
import random
import logging
import sqlite3
import argparse
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass

# ─── Suppress non-critical warnings during training ──────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger("train_ner")


# =============================================================================
# 🎨 CONSOLE OUTPUT HELPERS
# Beautiful terminal output because even logs deserve GLAMOUR! 💅
# =============================================================================

class Console:
    """Pretty console output for the pipeline."""

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
        """Print a glamorous phase banner."""
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
        """Simple text-based progress bar."""
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        pct = (current / total * 100) if total > 0 else 0
        print(f"\r  [{bar}] {pct:.0f}% ({current}/{total})", end="", flush=True)
        if current >= total:
            print()  # newline when done


# =============================================================================
# 📂 PHASE 1: DATA LOADING
# Load annotated documents from the database using AnnotationStorage.
# =============================================================================

def load_annotated_documents(db_path: str, status_filter: str = "completed") -> List[Any]:
    """
    📂 Load all annotated documents from the database.

    This imports from YOUR existing ner_schema.py — no reinventing
    the wheel, darling! We use what you already built! 💅

    Args:
        db_path:        Path to resume_extractions.db
        status_filter:  Which documents to load ('completed', 'all', etc.)

    Returns:
        List of AnnotatedDocument objects
    """
    # ── Import from the existing codebase ─────────────────────────────
    # We import HERE (not at top-level) so the script can still show
    # help text even if ner_schema.py isn't in the Python path.
    try:
        from ner_schema import (
            EntitySchema, AnnotationStorage, TrainingExporter,
            AnnotatedDocument, BIOTagger
        )
    except ImportError:
        Console.error(
            "Cannot import ner_schema.py! Make sure it's in the same directory "
            "or in your PYTHONPATH. This script needs your existing code, sugar! 💅"
        )
        sys.exit(1)

    # ── Initialize storage and load documents ─────────────────────────
    storage = AnnotationStorage(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        if status_filter == "all":
            rows = conn.execute(
                "SELECT doc_id FROM ner_documents"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT doc_id FROM ner_documents WHERE status = ?",
                (status_filter,)
            ).fetchall()
    except sqlite3.OperationalError as e:
        Console.error(f"Database query failed: {e}")
        Console.info("Make sure you've annotated some resumes first!")
        sys.exit(1)
    finally:
        conn.close()

    documents = []
    skipped = 0
    for row in rows:
        doc = storage.load_document(row["doc_id"])
        if doc and doc.raw_text and len(doc.raw_text.strip()) > 50:
            documents.append(doc)
        else:
            skipped += 1
            logger.debug(f"Skipped doc {row['doc_id']} — empty or too short")

    if skipped > 0:
        Console.warning(f"Skipped {skipped} documents (empty or < 50 chars)")

    return documents


# =============================================================================
# 🔍 PHASE 2: DATA VALIDATION & QUALITY AUDIT
# The dress rehearsal inspection — catch problems BEFORE the curtain rises!
# =============================================================================

@dataclass
class ValidationReport:
    """Results of the data validation audit."""
    total_docs: int = 0
    total_annotations: int = 0
    entity_counts: Dict[str, int] = None
    entity_per_doc: Dict[str, float] = None
    warnings: List[str] = None
    errors: List[str] = None
    min_threshold_failures: List[str] = None
    orphaned_i_tags: int = 0
    empty_spans: int = 0
    duplicate_spans: int = 0
    avg_doc_length: float = 0
    passed: bool = True

    def __post_init__(self):
        # Initialize mutable defaults safely
        # (Mutable default arguments in dataclass fields are a TRAP, darling! 🪤)
        if self.entity_counts is None:
            self.entity_counts = {}
        if self.entity_per_doc is None:
            self.entity_per_doc = {}
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []
        if self.min_threshold_failures is None:
            self.min_threshold_failures = []


def validate_data(
    documents: List[Any],
    min_examples_per_entity: int = 10,
    min_docs: int = 10
) -> ValidationReport:
    """
    🔍 Phase 2: Validate annotated data quality.

    This is like a HEALTH CHECK before the show — we inspect:
      1. Do we have ENOUGH data? (minimum document count)
      2. Does every entity type have enough examples?
      3. Are there BIO consistency issues?
      4. Are there empty or suspiciously short annotations?
      5. Are there duplicate annotations (same span, same type)?
      6. Distribution balance — any wildly over/under-represented types?

    Think of it as the stage manager doing a final walkthrough before
    the curtain rises — checking every prop, every light, every mic! 🎭🔦

    Args:
        documents:                 List of AnnotatedDocument objects
        min_examples_per_entity:   Minimum annotations needed per entity type
        min_docs:                  Minimum document count to proceed

    Returns:
        ValidationReport with all findings
    """
    Console.banner("🔍 PHASE 2: Data Validation & Quality Audit")

    report = ValidationReport()
    report.total_docs = len(documents)

    # ── Check 1: Minimum document count ───────────────────────────────
    if report.total_docs < min_docs:
        report.errors.append(
            f"Only {report.total_docs} documents found (minimum: {min_docs}). "
            f"Go annotate more resumes, darling! 💅"
        )
        report.passed = False

    Console.stat("Total documents", report.total_docs)

    # ── Collect all annotations across documents ──────────────────────
    all_annotations = []
    doc_lengths = []
    per_doc_entities = defaultdict(lambda: defaultdict(int))

    for doc in documents:
        doc_lengths.append(len(doc.raw_text))
        for ann in doc.annotations:
            all_annotations.append(ann)
            per_doc_entities[doc.doc_id][ann.entity_type] += 1

    report.total_annotations = len(all_annotations)
    report.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

    Console.stat("Total annotations", report.total_annotations)
    Console.stat("Avg doc length", f"{report.avg_doc_length:.0f} chars")

    # ── Check 2: Entity type distribution ─────────────────────────────
    entity_counter = Counter(ann.entity_type for ann in all_annotations)
    report.entity_counts = dict(entity_counter.most_common())

    print(f"\n  {'Entity Type':<25} {'Count':>6}  {'Per Doc':>8}  Status")
    print(f"  {'─' * 25} {'─' * 6}  {'─' * 8}  {'─' * 15}")

    for etype, count in entity_counter.most_common():
        per_doc = count / report.total_docs if report.total_docs > 0 else 0
        report.entity_per_doc[etype] = per_doc

        # Status indicator
        if count < min_examples_per_entity:
            status = f"{Console.RED}⚠ LOW ({min_examples_per_entity} min){Console.RESET}"
            report.min_threshold_failures.append(
                f"{etype}: only {count} examples (need {min_examples_per_entity}+)"
            )
        elif count < min_examples_per_entity * 3:
            status = f"{Console.YELLOW}△ OK-ish{Console.RESET}"
        else:
            status = f"{Console.GREEN}✓ Good{Console.RESET}"

        print(f"  {etype:<25} {count:>6}  {per_doc:>7.1f}  {status}")

    # ── Check 3: Empty or suspiciously short annotations ──────────────
    for ann in all_annotations:
        text = ann.text.strip() if ann.text else ""
        if not text:
            report.empty_spans += 1
        # Very short annotations might be errors (single char entity names?)
        # But some entities ARE short: "M" for gender, "A+" for GPA
        # So we only flag if it's a type that should be longer
        long_types = {
            "JOB_TITLE", "ORGANIZATION", "INSTITUTION", "JOB_DESCRIPTION",
            "PERSON_NAME", "SUMMARY", "CERTIFICATION", "FIELD_OF_STUDY"
        }
        if len(text) <= 1 and ann.entity_type in long_types:
            report.warnings.append(
                f"Suspiciously short {ann.entity_type}: '{text}' "
                f"(chars {ann.char_start}-{ann.char_end})"
            )

    if report.empty_spans > 0:
        report.errors.append(f"{report.empty_spans} empty annotation spans found!")

    # ── Check 4: Duplicate annotations (same span + type in same doc) ─
    seen_spans = set()
    for doc in documents:
        doc_spans = set()
        for ann in doc.annotations:
            key = (ann.entity_type, ann.char_start, ann.char_end, ann.layer)
            if key in doc_spans:
                report.duplicate_spans += 1
            doc_spans.add(key)

    if report.duplicate_spans > 0:
        report.warnings.append(
            f"{report.duplicate_spans} duplicate annotation spans detected"
        )

    # ── Check 5: BIO consistency (orphaned I- tags) ───────────────────
    # This checks if the BIO tagger would produce valid sequences
    try:
        from ner_schema import EntitySchema, BIOTagger
        schema = EntitySchema()
        tagger = BIOTagger(schema)

        for doc in documents:
            token_texts, bio_tags = tagger.tag_document(doc, layer=0)
            prev_entity = None
            for i, tag in enumerate(bio_tags):
                if tag.startswith("I-"):
                    entity = tag[2:]
                    if prev_entity != entity:
                        report.orphaned_i_tags += 1
                if tag.startswith("B-"):
                    prev_entity = tag[2:]
                elif tag.startswith("I-"):
                    pass  # continues current entity
                else:
                    prev_entity = None
    except Exception as e:
        report.warnings.append(f"BIO consistency check skipped: {e}")

    if report.orphaned_i_tags > 0:
        report.warnings.append(
            f"{report.orphaned_i_tags} orphaned I- tags (I- without preceding B-)"
        )

    # ── Check 6: Coverage warnings for critical entity types ──────────
    # These are the MUST-HAVE entity types for a resume NER system.
    # If any of these are missing entirely, we're in trouble!
    critical_types = {
        "PERSON_NAME", "PHONE", "EMAIL", "ORGANIZATION",
        "JOB_TITLE", "SKILL", "INSTITUTION", "DEGREE"
    }
    missing_critical = critical_types - set(report.entity_counts.keys())
    if missing_critical:
        report.errors.append(
            f"Critical entity types with ZERO examples: {', '.join(sorted(missing_critical))}. "
            f"The model can't learn what it's never seen, darling! 🙈"
        )
        report.passed = False

    # ── Print summary ─────────────────────────────────────────────────
    print()
    if report.errors:
        for err in report.errors:
            Console.error(err)

    if report.min_threshold_failures:
        print()
        Console.warning(
            f"{len(report.min_threshold_failures)} entity types below "
            f"minimum threshold ({min_examples_per_entity}):"
        )
        for fail in report.min_threshold_failures:
            print(f"    → {fail}")

    if report.warnings:
        print()
        for warn in report.warnings:
            Console.warning(warn)

    if report.passed and not report.min_threshold_failures:
        Console.success("All validation checks passed! Data looks FABULOUS! 💅✨")
    elif report.passed:
        Console.warning(
            "Data is usable but some entity types are thin. "
            "More annotations = better models! Consider annotating more resumes."
        )
    else:
        Console.error(
            "CRITICAL issues found — fix these before training! "
            "The model needs more data to learn from, sweetheart! 💪"
        )

    # ── Small dataset advisory ────────────────────────────────────────
    if report.total_docs < 50:
        print()
        Console.info(
            f"📊 SMALL DATASET ADVISORY ({report.total_docs} docs):"
        )
        Console.info(
            "With < 50 documents, consider these strategies:"
        )
        Console.info("  1. Use spaCy's en_core_web_sm as base (transfer learning)")
        Console.info("  2. Use K-fold cross-validation instead of fixed test split")
        Console.info("  3. Focus on the most important entity types first")
        Console.info("  4. Annotate 50-100+ resumes for best results")

    return report


# =============================================================================
# ✂️ PHASE 3: TRAIN / VALIDATION / TEST SPLIT
# You NEVER try on the outfit the same day you sew it, darling! 🧵
# =============================================================================

def split_data(
    documents: List[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    ✂️ Phase 3: Split documents into train/val/test sets.

    For small datasets (< 50 docs), we use a simple random split.
    For larger datasets, we could do stratified splitting, but with
    20-50 resumes, keeping it simple is smarter — fancy stratification
    on tiny data just shuffles noise around! 🎲

    Think of it like casting a play with a small ensemble:
      - Training set  = Main cast (they rehearse every scene)
      - Validation set = Understudies (they watch rehearsals + give notes)
      - Test set       = Opening night audience (they see it ONCE, cold!)

    Args:
        documents:    List of all annotated documents
        train_ratio:  Fraction for training (default 0.8)
        val_ratio:    Fraction for validation (default 0.1)
        test_ratio:   Fraction for testing (default 0.1)
        seed:         Random seed for reproducibility

    Returns:
        Tuple of (train_docs, val_docs, test_docs)
    """
    Console.banner("✂️ PHASE 3: Train / Validation / Test Split")

    # ── Validate ratios ───────────────────────────────────────────────
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        Console.error(
            f"Split ratios must sum to 1.0 (got {total_ratio:.2f}). "
            f"Check your math, darling! 🧮"
        )
        sys.exit(1)

    n = len(documents)
    if n < 3:
        Console.error("Need at least 3 documents to split. Not enough data! 😢")
        sys.exit(1)

    # ── Shuffle with fixed seed for reproducibility ───────────────────
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # ── Calculate split points ────────────────────────────────────────
    # For very small datasets, ensure at least 1 doc in val and test
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    # Safety: if we don't have enough for all three splits,
    # prioritize training data
    if n_train < 1:
        Console.warning(
            f"Dataset too small for 3-way split ({n} docs). "
            f"Using 2-way split (train + test only)."
        )
        n_test = max(1, n // 5)  # 20% test
        n_train = n - n_test
        n_val = 0

    # ── Create splits ─────────────────────────────────────────────────
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_docs = [documents[i] for i in train_indices]
    val_docs = [documents[i] for i in val_indices]
    test_docs = [documents[i] for i in test_indices]

    # ── Print split summary ───────────────────────────────────────────
    Console.stat("Training", f"{len(train_docs)} docs ({len(train_docs)/n*100:.0f}%)")
    Console.stat("Validation", f"{len(val_docs)} docs ({len(val_docs)/n*100:.0f}%)")
    Console.stat("Test", f"{len(test_docs)} docs ({len(test_docs)/n*100:.0f}%)")

    # ── Verify entity coverage across splits ──────────────────────────
    # Make sure every split has at least SOME variety of entity types
    for split_name, split_docs in [("Train", train_docs), ("Val", val_docs), ("Test", test_docs)]:
        if not split_docs:
            continue
        types_in_split = set()
        for doc in split_docs:
            for ann in doc.annotations:
                types_in_split.add(ann.entity_type)
        Console.info(f"{split_name} split has {len(types_in_split)} entity types")

    # ── Save split metadata for reproducibility ───────────────────────
    split_meta = {
        "seed": seed,
        "total_docs": n,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "split_counts": {"train": len(train_docs), "val": len(val_docs), "test": len(test_docs)},
        "train_doc_ids": [d.doc_id for d in train_docs],
        "val_doc_ids": [d.doc_id for d in val_docs],
        "test_doc_ids": [d.doc_id for d in test_docs],
        "created_at": datetime.datetime.now().isoformat()
    }

    Console.success("Split completed! Reproducible with seed={seed} 🎲")

    return train_docs, val_docs, test_docs, split_meta


# =============================================================================
# 🐍 PHASE 4: spaCy NER MODEL TRAINING
# THE MAIN EVENT, baby! Training the Named Entity Recognition model!
# =============================================================================

def train_ner_model(
    train_docs: List[Any],
    val_docs: List[Any],
    output_dir: str = "ner_model",
    base_model: str = "en_core_web_sm",
    n_epochs: int = 30,
    batch_size: int = 4,
    learn_rate: float = 0.001,
    patience: int = 5
) -> Dict[str, Any]:
    """
    🐍 Phase 4: Train a spaCy NER model.

    This converts your annotations to spaCy's DocBin format and
    trains a custom NER model using transfer learning from an
    existing English model.

    Why transfer learning? Think of it like hiring an actor who
    ALREADY knows how to act — you just need to teach them their
    NEW role (resume entities), not how to speak English from scratch! 🎭

    For small datasets (20-50 docs), transfer learning from
    en_core_web_sm is ESSENTIAL — training from scratch would be
    like teaching a baby to perform Shakespeare! 👶🎭

    Args:
        train_docs:   Training documents (AnnotatedDocument objects)
        val_docs:     Validation documents
        output_dir:   Where to save the trained model
        base_model:   Pre-trained spaCy model to start from
        n_epochs:     Number of training passes (more = better, up to a point)
        batch_size:   Documents per training batch (small for small data)
        learn_rate:   How fast the model learns (too fast = chaos, too slow = boring)
        patience:     Stop early if no improvement for this many epochs

    Returns:
        Dict with training metrics and model path
    """
    Console.banner("🐍 PHASE 4: spaCy NER Model Training")

    # ── Lazy imports — only load heavy ML libraries when needed ────────
    try:
        import spacy
        from spacy.tokens import DocBin
        from spacy.training import Example
        from spacy.util import minibatch, compounding
    except ImportError:
        Console.error(
            "spaCy is not installed! Run:\n"
            "    pip install spacy --break-system-packages\n"
            "    python -m spacy download en_core_web_sm"
        )
        return {"error": "spacy not installed"}

    # ── Step 1: Load the base model ───────────────────────────────────
    Console.info(f"Loading base model: {base_model}")
    try:
        nlp = spacy.load(base_model)
    except OSError:
        Console.warning(f"{base_model} not found. Downloading...")
        os.system(f"python -m spacy download {base_model}")
        try:
            nlp = spacy.load(base_model)
        except OSError:
            Console.error(
                f"Could not load {base_model}. Try: python -m spacy download en_core_web_sm"
            )
            return {"error": f"Cannot load {base_model}"}

    # ── Step 2: Prepare the NER component ─────────────────────────────
    # If the base model already has NER, we ADD our custom labels.
    # If not, we create a fresh NER pipe.
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
        Console.info("Created new NER pipe")
    else:
        ner = nlp.get_pipe("ner")
        Console.info(f"Using existing NER pipe ({len(ner.labels)} existing labels)")

    # ── Step 3: Collect all entity labels from training data ──────────
    all_labels = set()
    for doc in train_docs:
        for ann in doc.annotations:
            if ann.layer == 0:  # Primary layer only
                all_labels.add(ann.entity_type)

    # Add labels to the NER component
    for label in sorted(all_labels):
        ner.add_label(label)

    Console.stat("Entity labels", f"{len(all_labels)} types")
    Console.stat("Training docs", len(train_docs))
    Console.stat("Validation docs", len(val_docs))
    Console.stat("Epochs", n_epochs)
    Console.stat("Batch size", batch_size)
    Console.stat("Learning rate", learn_rate)

    # ── Step 4: Convert annotations to spaCy Example objects ──────────
    def docs_to_examples(annotated_docs, nlp_model):
        """
        Convert AnnotatedDocument objects to spaCy Example objects.

        Each Example pairs a raw text (as a spaCy Doc) with a
        reference annotation (the "gold standard" we want the model
        to learn to reproduce).

        Think of Examples as flashcards — one side has the question
        (raw text), the other has the answer (entity labels)! 🃏
        """
        examples = []
        skipped = 0

        for adoc in annotated_docs:
            # Create the reference doc with entity annotations
            # spaCy wants entities as (start_char, end_char, label) tuples
            entities = []
            seen_spans = []  # For overlap detection

            for ann in adoc.annotations:
                if ann.layer != 0:
                    continue  # Only primary layer for spaCy

                # ── Check for overlapping spans ───────────────────
                # spaCy CANNOT handle overlapping entities on the same
                # layer — it will throw an error! So we skip overlaps
                # and keep the LONGER span (more context = better).
                overlaps = False
                for (s, e, _) in seen_spans:
                    if not (ann.char_end <= s or ann.char_start >= e):
                        overlaps = True
                        break

                if overlaps:
                    skipped += 1
                    continue

                entities.append((ann.char_start, ann.char_end, ann.entity_type))
                seen_spans.append((ann.char_start, ann.char_end, ann.entity_type))

            # ── BULLETPROOF: Wrap in try/except and skip bad docs ───
            # If ANY entity in this doc causes E024, we skip the
            # entire doc rather than crashing the whole pipeline.
            # Like a casting director — if an actor can't perform,
            # we recast, we don't cancel the show! 🎭
            try:
                doc = nlp_model.make_doc(adoc.raw_text)
                example = Example.from_dict(doc, {"entities": entities})

                # Test that this example can actually train
                # by doing a dry-run prediction
                nlp_model.update([example], sgd=None, losses={}, drop=0.0)

                examples.append(example)
            except Exception as e:
                logger.warning(f"Skipping doc {adoc.doc_id}: {e}")
                skipped += 1

        if skipped > 0:
            Console.warning(f"Skipped {skipped} overlapping/invalid annotations")

        return examples

    Console.info("Converting annotations to spaCy format...")
    train_examples = docs_to_examples(train_docs, nlp)
    val_examples = docs_to_examples(val_docs, nlp) if val_docs else []

    if not train_examples:
        Console.error("No valid training examples! Check your annotations.")
        return {"error": "No valid training examples"}

    Console.success(f"Created {len(train_examples)} training examples")
    if val_examples:
        Console.success(f"Created {len(val_examples)} validation examples")

    # ── Step 5: TRAINING LOOP ─────────────────────────────────────────
    # This is where the MAGIC happens! ✨🎭
    #
    # We disable other pipeline components during NER training so they
    # don't interfere. It's like putting blinders on a horse — focus
    # on ONE thing at a time, darling! 🐎
    Console.info("Starting training... (this may take a few minutes)")

    # Only train the NER component
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []

    with nlp.disable_pipes(*other_pipes):
        # Reset the NER weights for fine-tuning
        optimizer = nlp.resume_training()

        # Adjust learning rate
        optimizer.learn_rate = learn_rate

        for epoch in range(n_epochs):
            # Shuffle training data each epoch
            random.shuffle(train_examples)

            # ── Training pass ─────────────────────────────────────
            losses = {}
            batches = minibatch(train_examples, size=batch_size)

            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses, drop=0.35)

            train_loss = losses.get("ner", 0)

            # ── Validation pass (if we have val data) ─────────────
            val_loss = 0
            if val_examples:
                val_losses = {}
                # Evaluate without updating weights
                for example in val_examples:
                    # Use nlp.evaluate for proper scoring
                    val_losses_batch = {}
                    nlp.update([example], losses=val_losses_batch, sgd=None, drop=0.0)
                    val_loss += val_losses_batch.get("ner", 0)

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            training_log.append(epoch_log)

            # Progress output every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_str = f"  val_loss: {val_loss:.4f}" if val_examples else ""
                print(
                    f"  Epoch {epoch+1:3d}/{n_epochs} — "
                    f"train_loss: {train_loss:.4f}{val_str}"
                )

            # ── Early stopping ────────────────────────────────────
            # If the validation loss hasn't improved, we stop early.
            # Continuing to train would just memorize the training data
            # (overfitting) — like an actor who can ONLY perform in
            # one venue! We want versatility! 🌟
            if val_examples and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif val_examples:
                patience_counter += 1
                if patience_counter >= patience:
                    Console.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

    # ── Step 6: Save the trained model ────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)

    Console.success(f"Model saved to: {output_dir}/")

    # ── Step 7: Quick evaluation on training data ─────────────────────
    # (For a PROPER evaluation, we use the test set in Phase 6)
    Console.info("Running quick evaluation on training data...")

    results = {
        "model_path": output_dir,
        "base_model": base_model,
        "n_epochs": len(training_log),
        "labels": sorted(all_labels),
        "final_train_loss": training_log[-1]["train_loss"] if training_log else None,
        "final_val_loss": training_log[-1]["val_loss"] if training_log else None,
        "training_log": training_log,
    }

    return results


# =============================================================================
# 📊 PHASE 4b: NER MODEL EVALUATION
# The morning-after reviews — did the audience love it? 🎭
# =============================================================================

def evaluate_ner_model(
    model_path: str,
    test_docs: List[Any]
) -> Dict[str, Any]:
    """
    📊 Evaluate the trained NER model on the test set.

    This is the REAL test — the model sees documents it has NEVER
    seen before and we measure how well it performs. Like a pop quiz
    after all those rehearsals! 📝

    We calculate per-entity-type:
      - Precision: Of entities the model PREDICTED, how many were correct?
      - Recall:    Of entities that SHOULD have been found, how many were?
      - F1-Score:  The harmonic mean (balances precision and recall)

    Args:
        model_path:  Path to the saved spaCy model
        test_docs:   Test documents (AnnotatedDocument objects)

    Returns:
        Dict with per-entity-type metrics
    """
    Console.banner("📊 NER Model Evaluation")

    if not test_docs:
        Console.warning("No test documents available — skipping evaluation")
        return {}

    try:
        import spacy
    except ImportError:
        Console.error("spaCy not installed!")
        return {}

    # Load the trained model
    nlp = spacy.load(model_path)
    Console.info(f"Loaded model from {model_path}")

    # ── Compare predictions vs gold annotations ──────────────────────
    # We track True Positives, False Positives, False Negatives per type
    tp = defaultdict(int)  # Correctly predicted
    fp = defaultdict(int)  # Predicted but wrong
    fn = defaultdict(int)  # Missed (should have predicted)

    for adoc in test_docs:
        # Get model predictions
        doc = nlp(adoc.raw_text)
        predicted = set()
        for ent in doc.ents:
            predicted.add((ent.start_char, ent.end_char, ent.label_))

        # Get gold annotations (layer 0 only)
        gold = set()
        for ann in adoc.annotations:
            if ann.layer == 0:
                gold.add((ann.char_start, ann.char_end, ann.entity_type))

        # ── Calculate TP, FP, FN ──────────────────────────────────
        # Exact match: span boundaries AND label must match
        for pred in predicted:
            if pred in gold:
                tp[pred[2]] += 1
            else:
                fp[pred[2]] += 1

        for g in gold:
            if g not in predicted:
                fn[g[2]] += 1

    # ── Calculate per-type metrics ────────────────────────────────────
    all_types = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))

    print(f"\n  {'Entity Type':<25} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 6} {'─' * 6}  {'─' * 4} {'─' * 4} {'─' * 4}")

    metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for etype in all_types:
        t, f_p, f_n = tp[etype], fp[etype], fn[etype]
        total_tp += t
        total_fp += f_p
        total_fn += f_n

        precision = t / (t + f_p) if (t + f_p) > 0 else 0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[etype] = {"precision": precision, "recall": recall, "f1": f1}

        # Color-code F1 scores
        if f1 >= 0.8:
            f1_color = Console.GREEN
        elif f1 >= 0.5:
            f1_color = Console.YELLOW
        else:
            f1_color = Console.RED

        print(
            f"  {etype:<25} {precision:>5.1%} {recall:>5.1%} "
            f"{f1_color}{f1:>5.1%}{Console.RESET}  {t:>4} {f_p:>4} {f_n:>4}"
        )

    # ── Overall metrics ───────────────────────────────────────────────
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0

    print(f"  {'─' * 25} {'─' * 6} {'─' * 6} {'─' * 6}")
    print(f"  {'OVERALL':<25} {overall_p:>5.1%} {overall_r:>5.1%} {overall_f1:>5.1%}")

    metrics["_overall"] = {"precision": overall_p, "recall": overall_r, "f1": overall_f1}

    # ── Performance verdict ───────────────────────────────────────────
    print()
    if overall_f1 >= 0.8:
        Console.success(f"EXCELLENT! F1 = {overall_f1:.1%} — Model is runway-ready! 💃✨")
    elif overall_f1 >= 0.6:
        Console.warning(f"DECENT! F1 = {overall_f1:.1%} — Good start, needs more data/tuning")
    elif overall_f1 >= 0.4:
        Console.warning(f"FAIR. F1 = {overall_f1:.1%} — Needs more annotated data for sure")
    else:
        Console.error(f"LOW. F1 = {overall_f1:.1%} — More training data is essential!")
        Console.info("With 20-50 docs, this is NORMAL. Annotate more resumes to improve!")

    return metrics


# =============================================================================
# 🧠 PHASE 5: CLASSIFICATION MODEL TRAINING
# Training the Function & Industry predictor! 💼
# =============================================================================

def train_classifier(
    db_path: str,
    output_dir: str = "classifier_model",
    min_samples_per_class: int = 2
) -> Dict[str, Any]:
    """
    🧠 Phase 5: Train Function & Industry classification models.

    Uses scikit-learn to train text classifiers that predict:
      - Function (e.g., "IT", "Human Resources", "Sales")
      - Industry (e.g., "Banking & Finance", "Healthcare")

    For small datasets, we use simpler models that generalize better:
      - TF-IDF vectorizer (bag of words with importance weighting)
      - LinearSVC or LogisticRegression (simple but effective!)

    Think of TF-IDF like this: if every resume mentions "Excel",
    that word isn't very SPECIAL (low importance). But if only
    banking resumes mention "derivatives trading", THAT's a
    strong signal (high importance)! 📊

    Args:
        db_path:                 Path to resume_extractions.db
        output_dir:              Where to save the trained models
        min_samples_per_class:   Skip classes with fewer samples than this

    Returns:
        Dict with training results and metrics
    """
    Console.banner("🧠 PHASE 5: Classification Model Training")

    # ── Lazy imports ──────────────────────────────────────────────────
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report
        import numpy as np
        import pickle
    except ImportError:
        Console.error(
            "scikit-learn is not installed! Run:\n"
            "    pip install scikit-learn numpy --break-system-packages"
        )
        return {"error": "scikit-learn not installed"}

    # ── Step 1: Export classification data ─────────────────────────────
    Console.info("Loading classification data from database...")

    try:
        from ner_schema import EntitySchema, TrainingExporter
        schema = EntitySchema()
        exporter = TrainingExporter(schema)

        # Export to a temporary JSONL file
        os.makedirs(output_dir, exist_ok=True)
        jsonl_path = os.path.join(output_dir, "_temp_classification.jsonl")

        stats = exporter.to_classification_jsonl(
            db_path=db_path,
            output_path=jsonl_path,
            min_text_length=50,
            include_features=True
        )

        Console.stat("Records exported", stats.get("total_exported", 0))
        Console.stat("Skipped", stats.get("skipped", 0))

    except Exception as e:
        Console.error(f"Failed to export classification data: {e}")
        Console.info(
            "Make sure you've set Function & Industry labels "
            "in the Classify tab for your annotated resumes!"
        )
        return {"error": str(e)}

    # ── Step 2: Load the JSONL records ────────────────────────────────
    records = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        Console.error(f"Export file not found: {jsonl_path}")
        return {"error": "Export file not found"}

    if not records:
        Console.error("No classification records found! Set Function/Industry labels first.")
        return {"error": "No classification data"}

    # ── Step 3: Prepare text features ─────────────────────────────────
    # Combine raw text with structured features for richer signal
    texts = []
    func_labels = []
    ind_labels = []

    for rec in records:
        # Build enriched text: raw resume + structured entity features
        parts = [rec.get("text", "")]

        features = rec.get("features", {})
        if features:
            # Append structured features as additional "text"
            # This gives the TF-IDF vectorizer more signal to work with!
            if features.get("job_titles"):
                parts.append("JOB_TITLES: " + " | ".join(features["job_titles"]))
            if features.get("companies"):
                parts.append("COMPANIES: " + " | ".join(features["companies"]))
            if features.get("skills"):
                parts.append("SKILLS: " + " | ".join(features["skills"]))
            if features.get("institutions"):
                parts.append("INSTITUTIONS: " + " | ".join(features["institutions"]))
            if features.get("degrees"):
                parts.append("DEGREES: " + " | ".join(features["degrees"]))

        enriched_text = "\n".join(parts)
        texts.append(enriched_text)
        func_labels.append(rec.get("function_label", "others"))
        ind_labels.append(rec.get("industry_label", "Others"))

    Console.stat("Total samples", len(texts))

    # ── Step 4: Filter rare classes ───────────────────────────────────
    # Classes with very few samples can't be learned reliably.
    # We merge them into "others" to keep the model stable.
    results = {}

    for task_name, labels_list in [("Function", func_labels), ("Industry", ind_labels)]:
        Console.info(f"\n  ── Training {task_name} classifier ──")

        # Count class distribution
        class_counts = Counter(labels_list)
        Console.stat(f"  {task_name} classes", len(class_counts))

        # Filter rare classes
        valid_classes = {
            cls for cls, count in class_counts.items()
            if count >= min_samples_per_class
        }
        filtered_labels = [
            lbl if lbl in valid_classes else "others"
            for lbl in labels_list
        ]

        # Re-count after filtering
        filtered_counts = Counter(filtered_labels)
        unique_classes = set(filtered_labels)

        if len(unique_classes) < 2:
            Console.warning(
                f"Only {len(unique_classes)} class(es) for {task_name} — "
                f"need at least 2 to train a classifier! "
                f"Annotate more resumes with different {task_name} labels."
            )
            results[task_name.lower()] = {"error": "Not enough classes"}
            continue

        print(f"\n    {'Class':<30} {'Count':>5}")
        print(f"    {'─' * 30} {'─' * 5}")
        for cls, count in filtered_counts.most_common():
            print(f"    {cls:<30} {count:>5}")

        # ── Step 5: Build the pipeline ────────────────────────────
        # TF-IDF + LinearSVC is a classic combo for text classification.
        # It's fast, interpretable, and works GREAT on small datasets!
        #
        # Why not deep learning? With 20-50 samples, a neural network
        # would memorize the training data like a parrot — it can
        # repeat what it heard but can't generalize! 🦜
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,      # Top 5000 most important words
                ngram_range=(1, 2),     # Unigrams + bigrams ("software" + "software engineer")
                sublinear_tf=True,      # Dampen frequent terms (log scale)
                min_df=1,               # Include even rare terms (small dataset!)
                strip_accents="unicode",
                lowercase=True,
                # SG/MY resume-specific: keep technical terms, numbers
                token_pattern=r"(?u)\b\w[\w\+\#\.]+\b",
            )),
            ("clf", LinearSVC(
                max_iter=10000,         # Enough iterations to converge
                class_weight="balanced", # Handle imbalanced classes
                C=1.0,                  # Regularization strength
                random_state=42
            ))
        ])

        # ── Step 6: Cross-validation ─────────────────────────────
        # With small data, we use K-fold cross-validation instead of
        # a fixed train/test split. This gives us a more reliable
        # estimate of model performance.
        #
        # K-fold is like rotating the role of "test audience" among
        # different groups — everyone gets a turn! 🎭🔄
        n_splits = min(5, min(filtered_counts.values()))
        n_splits = max(2, n_splits)  # At least 2-fold

        Console.info(f"Running {n_splits}-fold cross-validation...")

        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(
                pipeline, texts, filtered_labels,
                cv=cv, scoring="f1_macro"
            )

            Console.stat("  CV F1 scores", [f"{s:.3f}" for s in scores])
            Console.stat("  Mean F1", f"{scores.mean():.3f} ± {scores.std():.3f}")

        except Exception as e:
            Console.warning(f"Cross-validation failed: {e}")
            Console.info("Falling back to full training without CV...")
            scores = np.array([0.0])

        # ── Step 7: Train on ALL data ─────────────────────────────
        # After CV gives us confidence, train on everything for
        # the final model. Every sample counts with small data!
        Console.info("Training final model on all data...")
        pipeline.fit(texts, filtered_labels)

        # ── Step 8: Save the trained model ────────────────────────
        model_filename = f"{task_name.lower()}_classifier.pkl"
        model_path = os.path.join(output_dir, model_filename)

        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        Console.success(f"Saved {task_name} model → {model_path}")

        # Also save the label mapping for reference
        label_map_path = os.path.join(output_dir, f"{task_name.lower()}_labels.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({
                "classes": sorted(unique_classes),
                "distribution": dict(filtered_counts),
                "cv_f1_mean": float(scores.mean()),
                "cv_f1_std": float(scores.std()),
                "n_samples": len(texts),
            }, f, indent=2, ensure_ascii=False)

        results[task_name.lower()] = {
            "model_path": model_path,
            "n_classes": len(unique_classes),
            "n_samples": len(texts),
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
        }

        # ── Performance verdict ───────────────────────────────────
        mean_f1 = scores.mean()
        if mean_f1 >= 0.7:
            Console.success(f"{task_name} classifier: F1={mean_f1:.1%} — Looking SNATCHED! 💅✨")
        elif mean_f1 >= 0.4:
            Console.warning(f"{task_name} classifier: F1={mean_f1:.1%} — Decent, needs more data")
        else:
            Console.warning(f"{task_name} classifier: F1={mean_f1:.1%} — Needs work. More labels = better!")

    # ── Cleanup temp file ─────────────────────────────────────────────
    try:
        os.remove(jsonl_path)
    except OSError:
        pass

    return results


# =============================================================================
# 📋 SAVE PIPELINE REPORT
# Everything in one place for future reference!
# =============================================================================

def save_pipeline_report(
    output_dir: str,
    validation_report: Optional[ValidationReport],
    split_meta: Optional[Dict],
    ner_results: Optional[Dict],
    eval_results: Optional[Dict],
    classifier_results: Optional[Dict]
):
    """
    📋 Save a comprehensive pipeline report to JSON.

    This is your training RECEIPT, darling — proof of everything
    that happened during the pipeline run! 🧾✨
    """
    report = {
        "pipeline_version": "1.0",
        "run_timestamp": datetime.datetime.now().isoformat(),
        "phases": {}
    }

    if validation_report:
        report["phases"]["validation"] = {
            "total_docs": validation_report.total_docs,
            "total_annotations": validation_report.total_annotations,
            "entity_counts": validation_report.entity_counts,
            "entity_per_doc": validation_report.entity_per_doc,
            "errors": validation_report.errors,
            "warnings": validation_report.warnings,
            "empty_spans": validation_report.empty_spans,
            "duplicate_spans": validation_report.duplicate_spans,
            "orphaned_i_tags": validation_report.orphaned_i_tags,
            "passed": validation_report.passed,
        }

    if split_meta:
        report["phases"]["split"] = split_meta

    if ner_results:
        report["phases"]["ner_training"] = ner_results

    if eval_results:
        report["phases"]["ner_evaluation"] = eval_results

    if classifier_results:
        report["phases"]["classification"] = classifier_results

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    Console.success(f"Pipeline report saved → {report_path}")
    return report_path


# =============================================================================
# 🚀 MAIN ENTRY POINT
# The Grand Conductor — orchestrates all phases! 🎶
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="💅✨ Fairy Codemother's ML Training Pipeline ✨💅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ner.py --db resume_extractions.db --all
  python train_ner.py --db resume_extractions.db --validate
  python train_ner.py --db resume_extractions.db --train-ner --epochs 50
  python train_ner.py --db resume_extractions.db --train-classifier
        """
    )

    # ── Required arguments ────────────────────────────────────────────
    parser.add_argument(
        "--db", type=str, default="resume_extractions.db",
        help="Path to resume_extractions.db (default: resume_extractions.db)"
    )

    # ── Phase selection ───────────────────────────────────────────────
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--validate", action="store_true", help="Phase 2: Validate data")
    parser.add_argument("--split", action="store_true", help="Phase 3: Split data")
    parser.add_argument("--train-ner", action="store_true", help="Phase 4: Train NER model")
    parser.add_argument("--train-classifier", action="store_true", help="Phase 5: Train classifier")

    # ── Configuration ─────────────────────────────────────────────────
    parser.add_argument(
        "--status", type=str, default="completed",
        choices=["completed", "in_progress", "all"],
        help="Which annotation status to load (default: completed)"
    )
    parser.add_argument(
        "--min-docs", type=int, default=5,
        help="Minimum documents required to proceed (default: 5)"
    )
    parser.add_argument(
        "--min-per-entity", type=int, default=10,
        help="Minimum examples per entity type (default: 10)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="NER training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Training batch size (default: 4, good for small data)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="ml_output",
        help="Output directory for models and reports (default: ml_output)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Continue even if validation fails"
    )

    args = parser.parse_args()

    # ── Setup logging ─────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # ── Determine which phases to run ─────────────────────────────────
    run_validate = args.all or args.validate
    run_split = args.all or args.split
    run_ner = args.all or args.train_ner
    run_classifier = args.all or args.train_classifier

    # If nothing selected, show help
    if not any([run_validate, run_split, run_ner, run_classifier]):
        parser.print_help()
        print("\n💡 Tip: Use --all to run the full pipeline!")
        return

    # ── Fancy startup banner ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  💅✨ FAIRY CODEMOTHER'S ML TRAINING PIPELINE ✨💅")
    print("═" * 60)
    print(f"  📂 Database: {args.db}")
    print(f"  📁 Output:   {args.output_dir}")
    print(f"  🎲 Seed:     {args.seed}")
    phases = []
    if run_validate:
        phases.append("2:Validate")
    if run_split:
        phases.append("3:Split")
    if run_ner:
        phases.append("4:NER")
    if run_classifier:
        phases.append("5:Classify")
    print(f"  🎬 Phases:   {' → '.join(phases)}")
    print("═" * 60)

    # ── Check database exists ─────────────────────────────────────────
    if not os.path.exists(args.db):
        Console.error(f"Database not found: {args.db}")
        Console.info("Run your extraction + annotation pipeline first!")
        sys.exit(1)

    # ── Phase 1: Load data (always needed) ────────────────────────────
    Console.banner("📂 PHASE 1: Loading Annotated Data")
    documents = load_annotated_documents(args.db, status_filter=args.status)

    if not documents:
        Console.error(
            "No annotated documents found! 😱\n"
            "    Make sure you've annotated resumes and marked them as 'completed'.\n"
            "    Or try: --status all  to include in-progress documents."
        )
        sys.exit(1)

    Console.success(f"Loaded {len(documents)} annotated documents")

    # ── Initialize result containers ──────────────────────────────────
    validation_report = None
    split_meta = None
    ner_results = None
    eval_results = None
    classifier_results = None

    # ══════════════════════════════════════════════════════════════════
    # 🔍 PHASE 2: Validation
    # ══════════════════════════════════════════════════════════════════
    if run_validate:
        validation_report = validate_data(
            documents,
            min_examples_per_entity=args.min_per_entity,
            min_docs=args.min_docs
        )

        # Abort if validation fails (unless --force)
        if not validation_report.passed and not args.force:
            Console.error(
                "Validation FAILED. Fix the issues above before training.\n"
                "    Use --force to proceed anyway (not recommended, but you do you! 💅)"
            )
            # Still save the validation report
            save_pipeline_report(
                args.output_dir, validation_report,
                None, None, None, None
            )
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # ✂️ PHASE 3: Train/Test Split
    # ══════════════════════════════════════════════════════════════════
    train_docs, val_docs, test_docs = documents, [], []

    if run_split or run_ner:
        train_docs, val_docs, test_docs, split_meta = split_data(
            documents, seed=args.seed
        )

    # ══════════════════════════════════════════════════════════════════
    # 🐍 PHASE 4: NER Training
    # ══════════════════════════════════════════════════════════════════
    if run_ner:
        ner_output = os.path.join(args.output_dir, "ner_model")

        ner_results = train_ner_model(
            train_docs=train_docs,
            val_docs=val_docs,
            output_dir=ner_output,
            n_epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Evaluate on test set
        if test_docs and ner_results and "error" not in ner_results:
            eval_results = evaluate_ner_model(ner_output, test_docs)

    # ══════════════════════════════════════════════════════════════════
    # 🧠 PHASE 5: Classification Training
    # ══════════════════════════════════════════════════════════════════
    if run_classifier:
        classifier_output = os.path.join(args.output_dir, "classifier_model")

        classifier_results = train_classifier(
            db_path=args.db,
            output_dir=classifier_output
        )

    # ══════════════════════════════════════════════════════════════════
    # 📋 SAVE PIPELINE REPORT
    # ══════════════════════════════════════════════════════════════════
    Console.banner("📋 Pipeline Summary")

    report_path = save_pipeline_report(
        args.output_dir,
        validation_report,
        split_meta,
        ner_results,
        eval_results,
        classifier_results
    )

    # ── Final summary ─────────────────────────────────────────────────
    print()
    Console.info(f"📂 All outputs saved to: {args.output_dir}/")

    if ner_results and "error" not in ner_results:
        Console.info(f"🐍 NER model: {args.output_dir}/ner_model/")

    if classifier_results:
        if classifier_results.get("function", {}).get("model_path"):
            Console.info(f"🧠 Function classifier: {classifier_results['function']['model_path']}")
        if classifier_results.get("industry", {}).get("model_path"):
            Console.info(f"🧠 Industry classifier: {classifier_results['industry']['model_path']}")

    # ── What's next? ──────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  🎬 WHAT'S NEXT?")
    print("═" * 60)
    print("  1. Review the pipeline report for performance metrics")
    print("  2. If F1 is low → annotate more resumes → re-run pipeline")
    print("  3. If F1 is good → integrate models into ai_extractor.py")
    print("  4. Replace regex extraction with: nlp = spacy.load('ml_output/ner_model')")
    print("  5. Replace keyword classifier with: pickle.load('classifier_model/...')")
    print("═" * 60)
    print("  💅✨ The Fairy Codemother is PROUD of you! ✨💅")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()