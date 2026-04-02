"""
train_aimerlion.py

💅✨ FAIRY CODEMOTHER'S COMPLETE ML TRAINING PIPELINE ✨💅

Trains BOTH models from your annotation tool data:
  1. spaCy NER      — Token-level entity extraction (replaces regex)
  2. scikit-learn    — Function/Industry classification (replaces keyword matching)

Prerequisites:
    pip install spacy scikit-learn joblib --break-system-packages
    python -m spacy download en_core_web_sm

Usage:
    python train_aimerlion.py                          # Train both models
    python train_aimerlion.py --mode ner               # NER only
    python train_aimerlion.py --mode classifier         # Classifier only
    python train_aimerlion.py --mode export             # Export data only (no training)
    python train_aimerlion.py --db path/to/resume_extractions.db

Output:
    models/ner_model/          — Trained spaCy NER model
    models/function_clf.pkl    — Function classifier (joblib)
    models/industry_clf.pkl    — Industry classifier (joblib)
    training_data/             — Exported CoNLL, JSONL, and split files
"""

import os
import sys
import json
import re
import sqlite3
import random
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 🎓 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DB = "resume_extractions.db"
OUTPUT_DIR = "training_data"
MODEL_DIR = "models"

# NER training config
NER_TRAIN_SPLIT = 0.8       # 80% train, 20% dev
NER_MAX_ITERATIONS = 50     # spaCy training iterations
NER_DROP_RATE = 0.35        # Dropout rate for regularization
NER_BATCH_SIZE = 8          # Batch size for training

# Classification config
CLF_TEST_SPLIT = 0.2        # 20% held out for testing
CLF_MIN_SAMPLES = 3         # Minimum samples per class to include
MIN_TEXT_LENGTH = 100        # Skip resumes shorter than this


# =============================================================================
# 📊 STEP 1: EXPORT TRAINING DATA FROM DATABASE
# =============================================================================

class DataExporter:
    """
    📊 Exports annotated data from the AiMerlion database.
    
    Reads from:
      - ner_documents + ner_annotations → NER training data
      - structured_extractions → Classification training data
    
    Think of this as the FABRIC SOURCING phase — before you can
    sew anything, you need to collect all the raw materials! 🧵
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"❌ Database not found: {db_path}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def export_ner_conll(self) -> Tuple[str, str]:
        """
        📤 Export NER data as CoNLL format, split into train/dev files.
        
        Uses the BIOTagger from ner_schema.py to convert span annotations
        into proper BIO-tagged token sequences.
        
        Returns (train_path, dev_path)
        """
        logger.info("📤 Exporting NER data to CoNLL format...")
        
        # 💅 Handle both old and new import paths
        try:
            from extraction.ner_schema import (
                EntitySchema, BIOTagger, AnnotationStorage,
                SpanAnnotation, AnnotatedDocument
            )
        except ImportError:
            from ner_schema import (
                EntitySchema, BIOTagger, AnnotationStorage,
                SpanAnnotation, AnnotatedDocument
            )
        
        schema = EntitySchema()
        tagger = BIOTagger(schema)
        storage = AnnotationStorage(self.db_path)
        
        conn = self._conn()
        
        # 💅 FIX: raw_text lives in raw_extractions, NOT ner_documents!
        docs_rows = conn.execute("""
            SELECT nd.doc_id, nd.candidate_id
            FROM ner_documents nd
            JOIN raw_extractions re ON re.candidate_id = nd.candidate_id
            WHERE nd.status = 'completed'
            AND re.raw_text IS NOT NULL
            AND LENGTH(re.raw_text) > ?
            GROUP BY nd.doc_id
        """, (MIN_TEXT_LENGTH,)).fetchall()
        
        if not docs_rows:
            logger.warning("⚠️ No completed documents found for NER export!")
            conn.close()
            return ("", "")
        
        # Load full AnnotatedDocument objects
        documents = []
        for row in docs_rows:
            doc = storage.load_document(row["doc_id"])
            if doc and doc.raw_text and len(doc.raw_text) > MIN_TEXT_LENGTH:
                documents.append(doc)
        
        conn.close()
        
        logger.info(f"📋 Found {len(documents)} completed documents")
        
        if len(documents) < 5:
            logger.warning("⚠️ Very few documents — model quality will be limited!")
        
        # Shuffle and split into train/dev
        random.seed(42)  # Reproducible splits
        random.shuffle(documents)
        
        split_idx = int(len(documents) * NER_TRAIN_SPLIT)
        train_docs = documents[:split_idx]
        dev_docs = documents[split_idx:]
        
        logger.info(f"📊 Split: {len(train_docs)} train / {len(dev_docs)} dev")
        
        # Generate CoNLL format using BIOTagger
        def docs_to_conll(docs: List[AnnotatedDocument]) -> str:
            """Convert documents to CoNLL-2003 format using BIOTagger."""
            lines = []
            entity_counts = Counter()
            
            for doc in docs:
                # Use BIOTagger to get token-tag pairs
                try:
                    token_texts, bio_tags = tagger.tag_document(doc, layer=0)
                except Exception as e:
                    logger.warning(f"⚠️ Skipping {doc.doc_id}: {e}")
                    continue
                
                # Document separator
                lines.append("-DOCSTART- -X- -X- O")
                lines.append("")
                
                for text, tag in zip(token_texts, bio_tags):
                    lines.append(f"{text}\t{tag}")
                    
                    # Count entity types for stats
                    if tag != "O":
                        # Extract entity type from BIO tag (e.g., "B-SKILL" → "SKILL")
                        entity_type = tag.split("-", 1)[1] if "-" in tag else tag
                        entity_counts[entity_type] += 1
                
                lines.append("")  # Blank line between documents
            
            # Log entity distribution
            if entity_counts:
                logger.info("📊 Entity distribution in training data:")
                for etype, count in entity_counts.most_common(15):
                    logger.info(f"   {etype:25s}: {count}")
            
            return "\n".join(lines)
        
        # Write train file
        train_path = os.path.join(OUTPUT_DIR, "train.conll")
        train_content = docs_to_conll(train_docs)
        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_content)
        
        # Write dev file
        dev_path = os.path.join(OUTPUT_DIR, "dev.conll")
        dev_content = docs_to_conll(dev_docs)
        with open(dev_path, "w", encoding="utf-8") as f:
            f.write(dev_content)
        
        logger.info(f"✅ NER CoNLL exported: {train_path} + {dev_path}")
        return (train_path, dev_path)
    
    # 💅 Delete old cached .spacy files to force fresh export!
    for old_file in ["train.spacy", "dev.spacy"]:
        old_path = os.path.join(OUTPUT_DIR, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)
            logger.info(f"🗑️ Deleted old {old_file}")


    def export_ner_spacy(self) -> Tuple[str, str]:
        """
        📤 Export NER data directly to spaCy DocBin format.
        
        This is the format spaCy v3 training ACTUALLY needs.
        Skips the CoNLL→spaCy conversion step!
        
        Returns (train_path, dev_path)
        """
        logger.info("📤 Exporting NER data to spaCy DocBin format...")
        
        try:
            import spacy
            from spacy.tokens import DocBin
        except ImportError:
            logger.error(
                "❌ spaCy not installed!\n"
                "   pip install spacy --break-system-packages\n"
                "   python -m spacy download en_core_web_sm"
            )
            return ("", "")
        
        # 💅 Handle both old and new import paths
        try:
            from extraction.ner_schema import AnnotationStorage
        except ImportError:
            from ner_schema import AnnotationStorage
        
        # Load base English model for tokenization
        try:
            nlp = spacy.blank("en")
        except Exception:
            nlp = spacy.blank("en")
        
        storage = AnnotationStorage(self.db_path)
        conn = self._conn()
        
        # 💅 FIX: raw_text lives in raw_extractions, NOT ner_documents!
        # Join the tables to filter by text length.
        docs_rows = conn.execute("""
            SELECT nd.doc_id, nd.candidate_id
            FROM ner_documents nd
            JOIN raw_extractions re ON re.candidate_id = nd.candidate_id
            WHERE nd.status = 'completed'
            AND re.raw_text IS NOT NULL
            AND LENGTH(re.raw_text) > ?
            GROUP BY nd.doc_id
        """, (MIN_TEXT_LENGTH,)).fetchall()
        
        # Load and convert documents
        all_examples = []
        skipped = 0
        entity_counts = Counter()
        
        for row in docs_rows:
            doc = storage.load_document(row["doc_id"])
            if not doc or not doc.raw_text:
                skipped += 1
                continue
            
            # Get layer-0 annotations (primary annotations only)
            annotations = [a for a in doc.annotations if a.layer == 0]
            
            if not annotations:
                skipped += 1
                continue
            
            # Build spaCy entities from annotations
            # Entity format: (start_char, end_char, label)
            entities = []
            seen_spans = set()
            
            for ann in annotations:
                # Skip META category entities — they're not useful for NER
                if ann.entity_type in ("SECTION_HEADER", "BULLET_MARKER"):
                    continue
                
                # 💅 FIX E024: Trim leading/trailing whitespace and punctuation
                # spaCy HATES entity spans that start or end with spaces/punctuation!
                # It's like submitting a pageant application with mustard stains! 🌭❌
                start = ann.char_start
                end = ann.char_end
                
                # Safety: bounds check
                if start < 0 or end > len(doc.raw_text) or start >= end:
                    continue
                
                span_text = doc.raw_text[start:end]
                
                # Trim leading whitespace and punctuation
                while start < end and doc.raw_text[start] in ' \t\n\r.,;:!?()[]{}"\'-/*\\':
                    start += 1
                
                # Trim trailing whitespace and punctuation
                while end > start and doc.raw_text[end - 1] in ' \t\n\r.,;:!?()[]{}"\'-/*\\':
                    end -= 1
                
                # Skip if nothing left after trimming
                if start >= end:
                    continue
                
                # Skip very short spans (likely noise)
                if (end - start) < 2:
                    continue
                
                span_key = (start, end)
                
                # Skip overlapping spans (spaCy doesn't allow overlaps)
                is_overlapping = False
                for existing_start, existing_end in seen_spans:
                    if (ann.char_start < existing_end and ann.char_end > existing_start):
                        is_overlapping = True
                        break
                
                if is_overlapping:
                    continue
                
                # Validate span boundaries
                if (ann.char_start >= 0 and 
                    ann.char_end <= len(doc.raw_text) and 
                    ann.char_start < ann.char_end):
                    entities.append((ann.char_start, ann.char_end, ann.entity_type))
                    seen_spans.add(span_key)
                    entity_counts[ann.entity_type] += 1
            
            if entities:
                all_examples.append({
                    "text": doc.raw_text,
                    "entities": entities
                })
        
        conn.close()
        
        if not all_examples:
            logger.warning("⚠️ No valid NER examples found!")
            return ("", "")
        
        logger.info(f"📋 {len(all_examples)} documents with entities (skipped {skipped})")
        logger.info("📊 Entity distribution:")
        for etype, count in entity_counts.most_common(15):
            logger.info(f"   {etype:25s}: {count}")
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * NER_TRAIN_SPLIT)
        train_examples = all_examples[:split_idx]
        dev_examples = all_examples[split_idx:]
        
        # Convert to spaCy DocBin
        def examples_to_docbin(examples: List[Dict]) -> DocBin:
            """
            💅 BULLETPROOF DocBin converter!
            Validates EACH example through spaCy's training pipeline
            before including it. Skips anything that would cause E024.
            """
            from spacy.training import Example
            
            db = DocBin()
            failed = 0
            validated = 0
            
            for ex in examples:
                doc = nlp.make_doc(ex["text"])
                ents = []
                
                for start, end, label in ex["entities"]:
                    # Try both alignment modes — contract first, then expand
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is None:
                        span = doc.char_span(start, end, label=label, alignment_mode="expand")
                    if span is not None:
                        ents.append(span)
                    else:
                        failed += 1
                
                # Greedy non-overlapping filter (longest span wins)
                sorted_ents = sorted(ents, key=lambda s: s.end - s.start, reverse=True)
                filtered = []
                taken = set()
                for ent in sorted_ents:
                    ent_tokens = set(range(ent.start, ent.end))
                    if not ent_tokens & taken:
                        filtered.append(ent)
                        taken |= ent_tokens
                
                try:
                    doc.ents = filtered
                except ValueError:
                    failed += 1
                    continue
                
                # 💅 THE KEY FIX: Actually TEST this example through spaCy
                # If it would cause E024 during training, skip it NOW!
                try:
                    reference = doc.copy()
                    example = Example(nlp.make_doc(ex["text"]), reference)
                    # Try to get the gold parse — this is what crashes during training
                    example.get_aligned_ner()
                    db.add(reference)
                    validated += 1
                except Exception as e:
                    failed += 1
                    logger.debug(f"   Skipping doc (validation failed): {str(e)[:80]}")
            
            logger.info(f"   ✅ {validated} docs validated, {failed} spans/docs skipped")
            return db
        
        # Write DocBin files
        train_path = os.path.join(OUTPUT_DIR, "train.spacy")
        dev_path = os.path.join(OUTPUT_DIR, "dev.spacy")
        
        train_db = examples_to_docbin(train_examples)
        train_db.to_disk(train_path)
        logger.info(f"✅ Train: {len(train_examples)} docs → {train_path}")
        
        dev_db = examples_to_docbin(dev_examples)
        dev_db.to_disk(dev_path)
        logger.info(f"✅ Dev: {len(dev_examples)} docs → {dev_path}")
        
        # Also save the entity labels list (needed for config)
        all_labels = sorted(entity_counts.keys())
        labels_path = os.path.join(OUTPUT_DIR, "ner_labels.json")
        with open(labels_path, "w") as f:
            json.dump(all_labels, f, indent=2)
        logger.info(f"📋 {len(all_labels)} entity labels → {labels_path}")
        
        return (train_path, dev_path)
    
    def export_classification_jsonl(self) -> str:
        """
        📤 Export classification data for Function/Industry training.
        
        Reads from structured_extractions where function and industry
        have been set by human annotators (not just keyword matching).
        """
        logger.info("📤 Exporting classification JSONL...")
        
        conn = self._conn()
        
        # Get documents with human-reviewed function/industry labels
        # 💅 FIX: JOIN raw_extractions for the resume text!
        rows = conn.execute("""
            SELECT 
                se.candidate_id,
                re.raw_text,
                se.function,
                se.industry,
                se.skills_json,
                se.experience_json,
                se.education_json,
                se.hard_skills_json,
                se.soft_skills_json
            FROM structured_extractions se
            JOIN ner_documents nd ON nd.doc_id = 'doc_' || se.candidate_id
            JOIN raw_extractions re ON re.candidate_id = se.candidate_id
            WHERE nd.status = 'completed'
            AND se.function IS NOT NULL
            AND se.function != ''
            AND se.industry IS NOT NULL
            AND se.industry != ''
            AND re.raw_text IS NOT NULL
            AND LENGTH(re.raw_text) > ?
            AND se.reviewed = 1
        """, (MIN_TEXT_LENGTH,)).fetchall()
        
        conn.close()
        
        if not rows:
            # Fallback: try without reviewed=1 filter
            logger.info("⚠️ No reviewed rows found, trying all completed with function/industry...")
            conn = self._conn()
            # 💅 FIX: Same JOIN fix for fallback query!
            rows = conn.execute("""
                SELECT 
                    se.candidate_id,
                    re.raw_text,
                    se.function,
                    se.industry,
                    se.skills_json,
                    se.experience_json,
                    se.education_json,
                    se.hard_skills_json,
                    se.soft_skills_json
                FROM structured_extractions se
                JOIN ner_documents nd ON nd.doc_id = 'doc_' || se.candidate_id
                JOIN raw_extractions re ON re.candidate_id = se.candidate_id
                WHERE nd.status = 'completed'
                AND se.function IS NOT NULL
                AND se.function != ''
                AND se.industry IS NOT NULL
                AND se.industry != ''
                AND re.raw_text IS NOT NULL
                AND LENGTH(re.raw_text) > ?
            """, (MIN_TEXT_LENGTH,)).fetchall()
            conn.close()
        
        if not rows:
            logger.warning("⚠️ No classification data found!")
            return ""
        
        # Build JSONL records
        records = []
        func_dist = Counter()
        ind_dist = Counter()
        
        def safe_json(raw, default):
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default
        
        for row in rows:
            func_label = row["function"].strip()
            ind_label = row["industry"].strip()
            
            # Build features from structured data
            features = {}
            
            # Skills
            features["skills"] = safe_json(row["skills_json"], [])
            features["hard_skills"] = safe_json(row["hard_skills_json"], [])
            features["soft_skills"] = safe_json(row["soft_skills_json"], [])
            
            # Experience
            exp = safe_json(row["experience_json"], [])
            if isinstance(exp, list):
                features["job_titles"] = [
                    e.get("title", e.get("role", "")) for e in exp
                    if isinstance(e, dict)
                ]
                features["companies"] = [
                    e.get("company", e.get("organization", "")) for e in exp
                    if isinstance(e, dict)
                ]
            elif isinstance(exp, dict) and "positions" in exp:
                positions = exp.get("positions", [])
                features["job_titles"] = [
                    p.get("title", "") for p in positions if isinstance(p, dict)
                ]
                features["companies"] = [
                    p.get("organization", "") for p in positions if isinstance(p, dict)
                ]
            else:
                features["job_titles"] = []
                features["companies"] = []
            
            # Education
            edu = safe_json(row["education_json"], [])
            if isinstance(edu, list):
                features["degrees"] = [
                    e.get("degree", "") for e in edu if isinstance(e, dict)
                ]
                features["institutions"] = [
                    e.get("institution", "") for e in edu if isinstance(e, dict)
                ]
            else:
                features["degrees"] = []
                features["institutions"] = []
            
            records.append({
                "text": row["raw_text"],
                "function_label": func_label,
                "industry_label": ind_label,
                "features": features
            })
            
            func_dist[func_label] += 1
            ind_dist[ind_label] += 1
        
        # Write JSONL
        output_path = os.path.join(OUTPUT_DIR, "classification_train.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        logger.info(f"✅ Classification: {len(records)} records → {output_path}")
        logger.info(f"📊 Function distribution ({len(func_dist)} classes):")
        for label, count in func_dist.most_common():
            logger.info(f"   {label:30s}: {count}")
        logger.info(f"📊 Industry distribution ({len(ind_dist)} classes):")
        for label, count in ind_dist.most_common():
            logger.info(f"   {label:30s}: {count}")
        
        return output_path
    
    def get_stats(self) -> Dict:
        """📊 Quick stats about available training data."""
        conn = self._conn()
        stats = {}
        
        try:
            # Document counts
            for status in ["completed", "in_progress", "pending"]:
                row = conn.execute(
                    "SELECT COUNT(*) as c FROM ner_documents WHERE status = ?", (status,)
                ).fetchone()
                stats[f"docs_{status}"] = row["c"] if row else 0
            
            # Total annotations
            row = conn.execute(
                "SELECT COUNT(*) as c FROM ner_annotations WHERE layer = 0"
            ).fetchone()
            stats["total_annotations"] = row["c"] if row else 0
            
            # Classification data
            row = conn.execute("""
                SELECT COUNT(*) as c FROM structured_extractions
                WHERE function IS NOT NULL AND function != ''
                AND industry IS NOT NULL AND industry != ''
            """).fetchone()
            stats["classified_resumes"] = row["c"] if row else 0
            
        except sqlite3.OperationalError as e:
            logger.warning(f"⚠️ Stats query failed: {e}")
        finally:
            conn.close()
        
        return stats


# =============================================================================
# 🧠 STEP 2: TRAIN spaCy NER MODEL
# =============================================================================

class NERTrainer:
    """
    🧠 Trains a spaCy NER model from your annotated data.
    
    This REPLACES all the regex patterns in ai_extractor.py!
    Instead of 17+ regex patterns trying to find skills, the trained
    model reads each word and says "this is a SKILL" or "this is a 
    COMPANY" — just like a human annotator does!
    
    Think of it as teaching a STUDENT (the model) by showing them
    hundreds of GRADED EXAMS (your annotations). After enough examples,
    they can grade new exams on their own! 🎓📝
    """
    
    @staticmethod
    def create_config(labels: List[str], output_dir: str) -> str:
        """
        📋 Generate a spaCy training config file.
        
        💅 FAIRY CODEMOTHER'S FIX: Let spaCy generate its OWN config!
        Hand-written configs break across spaCy versions, but spaCy's
        init config command always produces a VALID config for YOUR version.
        It's like letting the tailor take their OWN measurements! 📏✨
        """
        config_path = os.path.join(output_dir, "config.cfg")
        
        logger.info("📋 Generating spaCy config using spaCy's own init...")
        
        # Let spaCy generate a valid config for THIS version
        # --pipeline ner = we only want Named Entity Recognition
        # --optimize efficiency = CPU-friendly settings
        exit_code = os.system(
            f"python -m spacy init config {config_path} "
            f"--lang en --pipeline ner --optimize efficiency --force"
        )
        
        if exit_code != 0 or not os.path.exists(config_path):
            logger.error("❌ spaCy init config failed!")
            return ""
        
        # Now patch the generated config with our training data paths
        # and custom hyperparameters
        with open(config_path, "r", encoding="utf-8") as f:
            config_text = f.read()
        
        # Update paths to point to our training data
        config_text = config_text.replace(
            'train = null',
            f'train = "{OUTPUT_DIR}/train.spacy"'
        )
        config_text = config_text.replace(
            'dev = null',
            f'dev = "{OUTPUT_DIR}/dev.spacy"'
        )
        
        # Update max_epochs if present
        import re as _re
        config_text = _re.sub(
            r'max_epochs\s*=\s*\d+',
            f'max_epochs = {NER_MAX_ITERATIONS}',
            config_text
        )
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_text)
        
        logger.info(f"📋 spaCy config generated and patched → {config_path}")
        return config_path
    
    @staticmethod
    def train(config_path: str, output_dir: str) -> Optional[str]:
        """
        🏋️ Train spaCy NER — DIRECT FROM DATABASE VERSION!
        
        💅 FAIRY CODEMOTHER'S NUCLEAR OPTION: Skip DocBin entirely!
        Build training examples DIRECTLY from the annotation database.
        No serialization bugs, no lost entities, no drama! 🎭
        """
        model_path = os.path.join(output_dir, "ner_model")
        best_path = os.path.join(model_path, "model-best")
        last_path = os.path.join(model_path, "model-last")
        
        logger.info("🏋️ Starting spaCy NER training (direct from DB)...")
        
        try:
            import spacy
            from spacy.training import Example
            from spacy.util import minibatch, compounding
            
            try:
                from extraction.ner_schema import AnnotationStorage
            except ImportError:
                from ner_schema import AnnotationStorage
            
            storage = AnnotationStorage(DEFAULT_DB)
            
            # Load completed documents directly from database
            import sqlite3
            conn = sqlite3.connect(DEFAULT_DB)
            conn.row_factory = sqlite3.Row
            
            doc_rows = conn.execute("""
                SELECT nd.doc_id, nd.candidate_id
                FROM ner_documents nd
                JOIN raw_extractions re ON re.candidate_id = nd.candidate_id
                WHERE nd.status = 'completed'
                AND re.raw_text IS NOT NULL
                AND LENGTH(re.raw_text) > ?
                GROUP BY nd.doc_id
            """, (MIN_TEXT_LENGTH,)).fetchall()
            conn.close()
            
            logger.info(f"📋 Found {len(doc_rows)} completed documents in database")
            
            # Build training data directly from AnnotationStorage
            nlp = spacy.blank("en")
            
            all_examples = []
            label_counts = Counter()
            skipped = 0
            
            for row in doc_rows:
                doc = storage.load_document(row["doc_id"])
                if not doc or not doc.raw_text or len(doc.raw_text) < MIN_TEXT_LENGTH:
                    skipped += 1
                    continue
                
                # Get layer-0 annotations only
                annotations = [a for a in doc.annotations if a.layer == 0]
                if not annotations:
                    skipped += 1
                    continue
                
                # Build entity tuples: (start, end, label)
                entities = []
                seen_spans = set()
                
                for ann in annotations:
                    # Skip META entities
                    if ann.entity_type in ("SECTION_HEADER", "BULLET_MARKER"):
                        continue
                    
                    start = ann.char_start
                    end = ann.char_end
                    
                    # Bounds check
                    if start < 0 or end > len(doc.raw_text) or start >= end:
                        continue
                    
                    # Trim whitespace/punctuation from span boundaries
                    while start < end and doc.raw_text[start] in ' \t\n\r.,;:!?()[]{}"\'-/*\\':
                        start += 1
                    while end > start and doc.raw_text[end - 1] in ' \t\n\r.,;:!?()[]{}"\'-/*\\':
                        end -= 1
                    
                    if start >= end or (end - start) < 2:
                        continue
                    
                    # Check for overlaps
                    is_overlap = False
                    for es, ee in seen_spans:
                        if start < ee and end > es:
                            is_overlap = True
                            break
                    
                    if not is_overlap:
                        entities.append((start, end, ann.entity_type))
                        seen_spans.add((start, end))
                        label_counts[ann.entity_type] += 1
                
                if entities:
                    all_examples.append({
                        "text": doc.raw_text,
                        "entities": entities
                    })
            
            logger.info(f"✅ Built {len(all_examples)} examples from DB (skipped {skipped})")
            
            # Show entity distribution
            logger.info("📊 Entity counts:")
            for label, count in label_counts.most_common():
                logger.info(f"   {label:25s}: {count}")
            
            # Filter to entity types with enough examples
            MIN_ENTITY_EXAMPLES = 10
            keep_labels = {l for l, c in label_counts.items() if c >= MIN_ENTITY_EXAMPLES}
            
            if not keep_labels:
                keep_labels = {l for l, _ in label_counts.most_common(8)}
                logger.warning(f"⚠️ No type has {MIN_ENTITY_EXAMPLES}+ examples, using top 8")
            
            dropped = set(label_counts.keys()) - keep_labels
            if dropped:
                logger.info(f"🗑️ Dropping rare types: {sorted(dropped)}")
            logger.info(f"✅ Keeping: {sorted(keep_labels)}")
            
            # Filter entities in examples
            filtered_examples = []
            for ex in all_examples:
                filtered_ents = [(s, e, l) for s, e, l in ex["entities"] if l in keep_labels]
                if filtered_ents:
                    filtered_examples.append({"text": ex["text"], "entities": filtered_ents})
            
            logger.info(f"📊 After filtering: {len(filtered_examples)} examples")
            
            # Shuffle and split
            random.seed(42)
            random.shuffle(filtered_examples)
            split_idx = int(len(filtered_examples) * NER_TRAIN_SPLIT)
            train_data = filtered_examples[:split_idx]
            dev_data = filtered_examples[split_idx:]
            
            logger.info(f"📊 Split: {len(train_data)} train / {len(dev_data)} dev")
            
            # Add NER pipe and labels
            ner = nlp.add_pipe("ner")
            for label in sorted(keep_labels):
                ner.add_label(label)
            
            # Convert to spaCy Examples with validation
            def make_examples(data_list):
                examples = []
                bad = 0
                for item in data_list:
                    try:
                        spacy_doc = nlp.make_doc(item["text"])
                        ex = Example.from_dict(spacy_doc, {"entities": item["entities"]})
                        # Validate alignment
                        aligned = ex.get_aligned_ner()
                        if aligned is not None and any(a != "O" and a is not None for a in aligned):
                            examples.append(ex)
                        else:
                            bad += 1
                    except Exception:
                        bad += 1
                return examples, bad
            
            train_examples, train_bad = make_examples(train_data)
            dev_examples, dev_bad = make_examples(dev_data)
            
            logger.info(f"✅ Valid train examples: {len(train_examples)} (dropped {train_bad})")
            logger.info(f"✅ Valid dev examples: {len(dev_examples)} (dropped {dev_bad})")
            
            # 🔍 DEBUG: Show what the model actually sees
            if train_examples:
                sample = train_examples[0]
                ref_ents = [(e.start_char, e.end_char, e.label_) for e in sample.reference.ents]
                logger.info(f"🔍 Sample training example:")
                logger.info(f"   Text (first 200 chars): {sample.reference.text[:200]}")
                logger.info(f"   Entities: {ref_ents[:5]}...")
            
            if len(train_examples) < 3:
                logger.error(f"❌ Only {len(train_examples)} valid examples! Need more annotated data.")
                return None
            
            # TRAIN!
            optimizer = nlp.begin_training()
            best_f1 = 0.0
            patience_counter = 0
            max_patience = 8
            
            os.makedirs(best_path, exist_ok=True)
            os.makedirs(last_path, exist_ok=True)
            
            logger.info(f"\n🏋️ Training started! {NER_MAX_ITERATIONS} epochs max...")
            
            for epoch in range(NER_MAX_ITERATIONS):
                random.shuffle(train_examples)
                losses = {}
                batch_errors = 0
                
                batches = list(minibatch(train_examples, size=compounding(4.0, 32.0, 1.001)))
                
                for batch in batches:
                    try:
                        nlp.update(batch, drop=NER_DROP_RATE, sgd=optimizer, losses=losses)
                    except ValueError:
                        batch_errors += 1
                
                # Evaluate
                f1 = p = r = 0.0
                if dev_examples:
                    try:
                        scores = nlp.evaluate(dev_examples)
                        f1 = scores.get("ents_f", 0)
                        p = scores.get("ents_p", 0)
                        r = scores.get("ents_r", 0)
                    except Exception:
                        pass
                
                ner_loss = losses.get("ner", 0)
                
                logger.info(
                    f"   Epoch {epoch+1:3d}/{NER_MAX_ITERATIONS} | "
                    f"Loss: {ner_loss:8.2f} | "
                    f"F1: {f1:.1%} | P: {p:.1%} | R: {r:.1%}"
                    + (f" | ⚠️ {batch_errors} bad" if batch_errors else "")
                )
                
                if f1 > best_f1:
                    best_f1 = f1
                    nlp.to_disk(best_path)
                    patience_counter = 0
                    logger.info(f"   🏆 New best F1: {f1:.1%} — saved!")
                else:
                    patience_counter += 1
                
                nlp.to_disk(last_path)
                
                if patience_counter >= max_patience and epoch >= 10:
                    logger.info(f"   ⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # Final per-entity scores
            logger.info(f"\n🏆 Training complete! Best F1: {best_f1:.1%}")
            if dev_examples:
                try:
                    final_scores = nlp.evaluate(dev_examples)
                    per_type = final_scores.get("ents_per_type", {})
                    if per_type:
                        logger.info(f"\n{'Entity':25s} {'P':>8s} {'R':>8s} {'F1':>8s}")
                        logger.info("-" * 50)
                        for etype, s in sorted(per_type.items(), key=lambda x: x[1].get("f", 0), reverse=True):
                            logger.info(f"   {etype:25s} {s.get('p',0):7.1%} {s.get('r',0):7.1%} {s.get('f',0):7.1%}")
                except Exception:
                    pass
            
            # Return whichever model exists
            if os.path.exists(os.path.join(best_path, "config.cfg")):
                return best_path
            elif os.path.exists(os.path.join(last_path, "config.cfg")):
                return last_path
            else:
                logger.warning("⚠️ No valid model saved")
                return None
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def evaluate(model_path: str, dev_path: str):
        """
        📊 Evaluate the trained NER model on dev data.
        
        💅 FIX: Uses Python-native evaluation instead of CLI command.
        Our custom training loop saves the model differently than
        spaCy's CLI trainer, so we load and evaluate in Python!
        """
        logger.info(f"📊 Evaluating NER model: {model_path}")
        
        try:
            import spacy
            from spacy.tokens import DocBin
            from spacy.training import Example
            
            # Load the trained model
            nlp = spacy.load(model_path)
            
            # Load dev data
            dev_db = DocBin().from_disk(dev_path)
            dev_docs = list(dev_db.get_docs(nlp.vocab))
            
            # Build evaluation examples
            dev_examples = []
            for doc in dev_docs:
                try:
                    ex = Example.from_dict(
                        nlp.make_doc(doc.text),
                        {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}
                    )
                    dev_examples.append(ex)
                except Exception:
                    pass
            
            if not dev_examples:
                logger.warning("⚠️ No valid dev examples for evaluation!")
                return
            
            # Run evaluation
            scores = nlp.evaluate(dev_examples)
            
            # Pretty print results
            logger.info(f"\n📊 NER Evaluation Results ({len(dev_examples)} docs):")
            logger.info(f"{'Entity':25s} {'P':>8s} {'R':>8s} {'F1':>8s}")
            logger.info("-" * 50)
            
            per_type = scores.get("ents_per_type", {})
            for etype, s in sorted(per_type.items(), key=lambda x: x[1].get("f", 0), reverse=True):
                logger.info(
                    f"   {etype:25s} {s.get('p', 0):7.1%} "
                    f"{s.get('r', 0):7.1%} {s.get('f', 0):7.1%}"
                )
            
            overall_f1 = scores.get("ents_f", 0)
            overall_p = scores.get("ents_p", 0)
            overall_r = scores.get("ents_r", 0)
            logger.info("-" * 50)
            logger.info(f"   {'OVERALL':25s} {overall_p:7.1%} {overall_r:7.1%} {overall_f1:7.1%}")
            logger.info(f"\n🏆 Overall NER F1: {overall_f1:.1%}")
            
        except Exception as e:
            logger.warning(f"⚠️ Evaluation failed: {e}")


# =============================================================================
# 🧠 STEP 3: TRAIN CLASSIFICATION MODEL
# =============================================================================

class ClassifierTrainer:
    """
    🧠 Trains Function & Industry classifiers from JSONL data.
    
    Uses scikit-learn's TF-IDF + Logistic Regression pipeline — fast,
    lightweight, and surprisingly effective for text classification!
    
    No GPU needed! Trains in seconds! 🚀
    
    Think of it as teaching a filing clerk to sort resumes into the
    right department folders. After seeing enough examples of "IT resumes
    go in the IT folder", they learn the PATTERNS (keywords, job titles,
    skills) that predict each department! 📁✨
    """
    
    @staticmethod
    def train(jsonl_path: str, output_dir: str) -> Dict[str, str]:
        """
        🏋️ Train both Function and Industry classifiers.
        
        Returns dict of {"function": model_path, "industry": model_path}
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import classification_report
            import joblib
        except ImportError:
            logger.error(
                "❌ scikit-learn not installed!\n"
                "   pip install scikit-learn joblib --break-system-packages"
            )
            return {}
        
        # Load data
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        if len(records) < 10:
            logger.warning(f"⚠️ Only {len(records)} records — need 20+ for meaningful training!")
            if len(records) < 5:
                return {}
        
        logger.info(f"📋 Loaded {len(records)} classification records")
        
        # Enrich text with features for better classification
        # This is like adding spices to the dish — the base text is good,
        # but structured features make it BETTER! 🧑‍🍳
        enriched_texts = []
        for rec in records:
            parts = [rec["text"][:8000]]  # Cap text length
            
            features = rec.get("features", {})
            
            # Add job titles (strong signal for function!)
            if features.get("job_titles"):
                parts.append("JOB_TITLES: " + " | ".join(features["job_titles"]))
            
            # Add skills (strong signal for both!)
            for skill_field in ["hard_skills", "skills"]:
                if features.get(skill_field):
                    parts.append("SKILLS: " + " | ".join(features[skill_field]))
            
            # Add companies (signal for industry!)
            if features.get("companies"):
                parts.append("COMPANIES: " + " | ".join(features["companies"]))
            
            enriched_texts.append(" ".join(parts))
        
        results = {}
        os.makedirs(output_dir, exist_ok=True)
        
        # Train FUNCTION classifier
        func_labels = [r["function_label"] for r in records]
        results["function"] = ClassifierTrainer._train_one(
            enriched_texts, func_labels, "Function",
            os.path.join(output_dir, "function_clf.pkl")
        )
        
        # Train INDUSTRY classifier
        ind_labels = [r["industry_label"] for r in records]
        results["industry"] = ClassifierTrainer._train_one(
            enriched_texts, ind_labels, "Industry",
            os.path.join(output_dir, "industry_clf.pkl")
        )
        
        return results
    
    @staticmethod
    def _train_one(
        texts: List[str],
        labels: List[str],
        name: str,
        output_path: str
    ) -> Optional[str]:
        """Train a single classifier."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report
        import joblib
        
        logger.info(f"\n🏋️ Training {name} classifier...")
        
        # Filter classes with too few samples
        label_counts = Counter(labels)
        valid_indices = [
            i for i, label in enumerate(labels)
            if label_counts[label] >= CLF_MIN_SAMPLES
        ]
        
        if len(valid_indices) < len(texts):
            removed = len(texts) - len(valid_indices)
            logger.info(f"   Filtered {removed} samples from rare classes (< {CLF_MIN_SAMPLES} samples)")
        
        filtered_texts = [texts[i] for i in valid_indices]
        filtered_labels = [labels[i] for i in valid_indices]
        
        unique_labels = set(filtered_labels)
        logger.info(f"   Classes: {len(unique_labels)} | Samples: {len(filtered_texts)}")
        
        if len(unique_labels) < 2:
            logger.warning(f"⚠️ Need at least 2 classes for {name} — skipping!")
            return None
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            filtered_texts, filtered_labels,
            test_size=CLF_TEST_SPLIT,
            random_state=42,
            stratify=filtered_labels if len(unique_labels) > 1 else None
        )
        
        # Build pipeline: TF-IDF + Logistic Regression
        # TF-IDF converts text to numbers, LogReg finds the patterns
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,       # Top 5000 most important words
                ngram_range=(1, 2),      # Single words + word pairs ("data analyst")
                min_df=2,                # Word must appear in 2+ documents
                max_df=0.95,             # Skip words in 95%+ documents (too common)
                sublinear_tf=True,       # Log-scale term frequency (reduces noise)
                strip_accents="unicode", # Handle accented characters
            )),
            ("clf", LogisticRegression(
                max_iter=1000,           # Enough iterations to converge
                C=1.0,                   # Regularization strength
                class_weight="balanced", # Handle imbalanced classes
                solver="lbfgs",          # Fast solver
            ))
        ])
        
        # Train!
        logger.info(f"   Training on {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = sum(1 for a, b in zip(y_test, y_pred) if a == b) / len(y_test)
        
        logger.info(f"   ✅ {name} Accuracy: {accuracy:.1%}")
        
        # Detailed report
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info(f"\n📊 {name} Classification Report:\n{report}")
        
        # Save report to file
        report_path = output_path.replace(".pkl", "_report.txt")
        with open(report_path, "w") as f:
            f.write(f"{name} Classification Report\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write(report)
        
        # Cross-validation for more robust estimate
        if len(filtered_texts) >= 10:
            n_folds = min(5, len(unique_labels))
            try:
                cv_scores = cross_val_score(pipeline, filtered_texts, filtered_labels, cv=n_folds)
                logger.info(f"   📊 Cross-validation ({n_folds}-fold): {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
            except Exception:
                pass
        
        # Save model
        joblib.dump(pipeline, output_path)
        logger.info(f"   💾 Model saved → {output_path}")
        
        return output_path


# =============================================================================
# 🚀 MAIN — THE GRAND CONDUCTOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="💅 AiMerlion Complete ML Training Pipeline"
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Database path")
    parser.add_argument(
        "--mode", 
        choices=["all", "ner", "classifier", "export", "stats"],
        default="all",
        help="What to do"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("💅✨ AiMerlion ML Training Pipeline ✨💅")
    print(f"   Database: {args.db}")
    print(f"   Mode: {args.mode}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize exporter
    exporter = DataExporter(args.db)
    
    # Show stats
    stats = exporter.get_stats()
    print(f"\n📊 Database Stats:")
    print(f"   Completed docs:      {stats.get('docs_completed', 0)}")
    print(f"   In-progress docs:    {stats.get('docs_in_progress', 0)}")
    print(f"   Total annotations:   {stats.get('total_annotations', 0)}")
    print(f"   Classified resumes:  {stats.get('classified_resumes', 0)}")
    
    if args.mode == "stats":
        return
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # =========================================================
    # 🟢 NER TRAINING PIPELINE
    # =========================================================
    if args.mode in ("all", "ner", "export"):
        print("\n" + "=" * 70)
        print("🟢 NER Pipeline — Entity Extraction")
        print("=" * 70)
        
        # Export to spaCy format
        train_path, dev_path = exporter.export_ner_spacy()
        
        # Also export CoNLL for reference/debugging
        exporter.export_ner_conll()
        
        if args.mode != "export" and train_path and dev_path:
            # Load entity labels
            labels_path = os.path.join(OUTPUT_DIR, "ner_labels.json")
            if os.path.exists(labels_path):
                with open(labels_path) as f:
                    labels = json.load(f)
            else:
                labels = []
            
            # Generate spaCy config
            config_path = NERTrainer.create_config(labels, OUTPUT_DIR)
            
            # Train!
            model_path = NERTrainer.train(config_path, MODEL_DIR)
            
             # Evaluate (skip if model wasn't saved properly)
            if model_path and os.path.exists(os.path.join(model_path, "config.cfg")):
                NERTrainer.evaluate(model_path, dev_path)
            elif model_path:
                logger.info("ℹ️ Skipping separate evaluation — scores were shown during training above")
    
    # =========================================================
    # 🟠 CLASSIFICATION TRAINING PIPELINE
    # =========================================================
    if args.mode in ("all", "classifier", "export"):
        print("\n" + "=" * 70)
        print("🟠 Classification Pipeline — Function & Industry")
        print("=" * 70)
        
        # Export JSONL
        jsonl_path = exporter.export_classification_jsonl()
        
        if args.mode != "export" and jsonl_path:
            # Train classifiers
            results = ClassifierTrainer.train(jsonl_path, MODEL_DIR)
            
            if results.get("function"):
                print(f"\n✅ Function classifier: {results['function']}")
            if results.get("industry"):
                print(f"✅ Industry classifier: {results['industry']}")
    
    # =========================================================
    # 📋 SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("📋 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Training data:  {OUTPUT_DIR}/")
    print(f"📁 Trained models: {MODEL_DIR}/")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Test NER model:")
    print("      python -c \"import spacy; nlp=spacy.load('models/ner_model/model-best'); print(nlp('Python developer at DBS Bank').ents)\"")
    print("   2. Test classifier:")
    print("      python -c \"import joblib; clf=joblib.load('models/function_clf.pkl'); print(clf.predict(['Software engineer with Python Java AWS']))\"")
    print("   3. Wire into ai_extractor.py (see integration guide)")
    print("   4. Keep annotating more resumes → retrain for better accuracy! 📈")


if __name__ == "__main__":
    main()