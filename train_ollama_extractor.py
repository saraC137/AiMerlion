"""
train_ollama_extractor.py

💅✨ FAIRY CODEMOTHER'S OLLAMA TRAINING PIPELINE ✨💅

Converts your hand-annotated resume data (from AiMerlion's annotation tool)
into training data for fine-tuning or few-shot prompting with Ollama.

This is the BRIDGE between your annotation tool (the data collection engine)
and a better-performing AI model. Think of it like this:
  - Annotation tool = Dance school (humans teach the moves)
  - This script = Choreography sheet (converts lessons into steps)
  - Ollama fine-tune = The student who LEARNS the choreography! 💃

Architecture:
  Step 1: Export annotations from SQLite (resume_extractions.db)
  Step 2: Convert to Ollama-compatible training format (JSONL)
  Step 3: Generate few-shot examples for system prompts
  Step 4: Create custom Modelfile with baked-in examples
  Step 5: (Optional) Fine-tune with Unsloth/LoRA for serious accuracy gains

Usage:
    python train_ollama_extractor.py --db resume_extractions.db --output training_data/
    python train_ollama_extractor.py --db resume_extractions.db --mode few-shot --num-examples 10
    python train_ollama_extractor.py --db resume_extractions.db --mode modelfile

Dependencies:
    pip install --break-system-packages json sqlite3
    # For fine-tuning (optional):
    pip install --break-system-packages unsloth transformers datasets
"""

import json
import sqlite3
import os
import re
import argparse
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 🎓 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default database path (same as annotation_tool.py)
DEFAULT_DB_PATH = "resume_extractions.db"

# Output directory for training data
DEFAULT_OUTPUT_DIR = "training_data"

# Minimum text length for usable training examples
MIN_TEXT_LENGTH = 100

# Maximum resume text length to send to model (prevents context overflow)
MAX_RESUME_TEXT_LENGTH = 12000


# =============================================================================
# 📊 STEP 1: DATA EXTRACTION FROM SQLITE
# =============================================================================

class AnnotationDataExtractor:
    """
    🗄️ Extracts annotated resume data from the AiMerlion database.
    
    Reads from both ner_documents/ner_annotations tables (annotation tool)
    and structured_extractions table (AI extractor output that humans corrected).
    
    Think of this as raiding the costume vault — we're pulling out all the
    FINISHED outfits (completed annotations) to use as training patterns! 👗
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"❌ Database not found at: {db_path}\n"
                f"   Make sure you're running from the AiMerlion project directory!"
            )
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_completed_documents(self) -> List[Dict]:
        """
        📋 Get all COMPLETED annotated documents.
        
        Only completed documents are reliable enough for training.
        In-progress annotations might have missing or incorrect labels.
        
        Returns list of dicts with:
          - candidate_id, doc_id, raw_text
          - annotations (list of entity spans)
          - metadata (function, industry, etc.)
        """
        conn = self._get_connection()
        
        try:
            # Get completed documents from ner_documents
            docs = conn.execute("""
                SELECT 
                    nd.doc_id,
                    nd.candidate_id,
                    nd.raw_text,
                    nd.status,
                    nd.metadata
                FROM ner_documents nd
                WHERE nd.status = 'completed'
                AND nd.raw_text IS NOT NULL
                AND LENGTH(nd.raw_text) > ?
                ORDER BY nd.candidate_id
            """, (MIN_TEXT_LENGTH,)).fetchall()
            
            results = []
            
            for doc in docs:
                doc_id = doc["doc_id"]
                
                # Get annotations for this document
                annotations = conn.execute("""
                    SELECT 
                        entity_type,
                        char_start,
                        char_end,
                        text,
                        layer,
                        confidence,
                        annotator
                    FROM ner_annotations
                    WHERE doc_id = ?
                    AND layer = 0
                    ORDER BY char_start
                """, (doc_id,)).fetchall()
                
                # Get structured extraction data (human-corrected)
                structured = conn.execute("""
                    SELECT *
                    FROM structured_extractions
                    WHERE candidate_id = ?
                """, (doc["candidate_id"],)).fetchone()
                
                result = {
                    "candidate_id": doc["candidate_id"],
                    "doc_id": doc_id,
                    "raw_text": doc["raw_text"],
                    "annotations": [dict(a) for a in annotations],
                    "metadata": json.loads(doc["metadata"]) if doc["metadata"] else {},
                }
                
                # Add structured extraction data if available
                if structured:
                    result["structured"] = {
                        "name": structured["name"],
                        "email": structured["email"],
                        "phone": structured["phone"],
                        "date_of_birth": structured["date_of_birth"] if "date_of_birth" in structured.keys() else None,
                        "location": structured["location"] if "location" in structured.keys() else None,
                        "nationality": structured["nationality"] if "nationality" in structured.keys() else None,
                        "function": structured["function"] if "function" in structured.keys() else None,
                        "industry": structured["industry"] if "industry" in structured.keys() else None,
                        "skills_json": structured["skills_json"] if "skills_json" in structured.keys() else "[]",
                        "experience_json": structured["experience_json"] if "experience_json" in structured.keys() else "[]",
                        "education_json": structured["education_json"] if "education_json" in structured.keys() else "[]",
                        "hard_skills": structured["hard_skills"] if "hard_skills" in structured.keys() else "[]",
                        "soft_skills": structured["soft_skills"] if "soft_skills" in structured.keys() else "[]",
                        "tags": structured["tags"] if "tags" in structured.keys() else "[]",
                    }
                
                results.append(result)
            
            logger.info(f"📋 Found {len(results)} completed documents for training")
            return results
            
        finally:
            conn.close()
    
    def get_annotation_stats(self) -> Dict:
        """📊 Get statistics about available training data."""
        conn = self._get_connection()
        
        try:
            stats = {}
            
            # Total documents by status
            status_counts = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM ner_documents
                GROUP BY status
            """).fetchall()
            stats["documents_by_status"] = {row["status"]: row["count"] for row in status_counts}
            
            # Entity type distribution
            entity_counts = conn.execute("""
                SELECT entity_type, COUNT(*) as count
                FROM ner_annotations
                WHERE layer = 0
                GROUP BY entity_type
                ORDER BY count DESC
            """).fetchall()
            stats["entity_distribution"] = {row["entity_type"]: row["count"] for row in entity_counts}
            
            # Total annotations
            total = conn.execute("SELECT COUNT(*) as c FROM ner_annotations WHERE layer = 0").fetchone()
            stats["total_annotations"] = total["c"]
            
            # Annotators
            annotators = conn.execute("""
                SELECT DISTINCT annotator FROM ner_annotations
            """).fetchall()
            stats["annotators"] = [row["annotator"] for row in annotators]
            
            return stats
            
        finally:
            conn.close()


# =============================================================================
# 🔄 STEP 2: CONVERT TO TRAINING FORMATS
# =============================================================================

class TrainingDataConverter:
    """
    🔄 Converts annotated data into various training formats.
    
    Supports:
    1. Ollama Chat JSONL — For fine-tuning with Ollama's train command
    2. Few-shot Examples — For prompt engineering (immediate improvement!)
    3. Modelfile Generation — Custom model with baked-in examples
    4. Alpaca/ShareGPT Format — For Unsloth/LoRA fine-tuning
    
    Think of this as the fashion translator — taking beautiful designs
    and converting them into sewing patterns that MACHINES can follow! 🪡
    """
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # -----------------------------------------------------------------
    # 📝 FORMAT 1: Header Extraction Training Data
    # -----------------------------------------------------------------
    
    def generate_header_training(self, documents: List[Dict]) -> str:
        """
        📝 Generate training data for HEADER FIELD extraction.
        
        Each example teaches the model to extract:
        name, email, phone, date_of_birth, nationality, location, language
        
        This is the field that Ollama struggles with MOST because
        SG/MY resumes put contact info in the HEADER (first ~2000 chars)
        with varying label formats (colon, tab, or no separator).
        """
        training_examples = []
        skipped = 0
        
        for doc in documents:
            structured = doc.get("structured", {})
            
            # Skip if we don't have structured data (human-corrected truth)
            if not structured:
                skipped += 1
                continue
            
            # Build the "correct answer" from human-corrected data
            expected_output = {
                "name": structured.get("name") or None,
                "email": structured.get("email") or None,
                "phone": structured.get("phone") or None,
                "date_of_birth": structured.get("date_of_birth") or None,
                "nationality": structured.get("nationality") or None,
                "location": structured.get("location") or None,
                "language": None,  # Will try to extract from annotations
                "linkedin": None,
                "github": None,
                "website": None,
            }
            
            # Extract language from annotations if available
            for ann in doc.get("annotations", []):
                if ann["entity_type"] == "LANGUAGE_SKILL" and not expected_output["language"]:
                    expected_output["language"] = ann["text"]
                elif ann["entity_type"] == "LINKEDIN" and not expected_output["linkedin"]:
                    expected_output["linkedin"] = ann["text"]
                elif ann["entity_type"] == "GITHUB" and not expected_output["github"]:
                    expected_output["github"] = ann["text"]
            
            # Skip if we have almost nothing extracted
            non_null_count = sum(1 for v in expected_output.values() if v)
            if non_null_count < 2:
                skipped += 1
                continue
            
            # Truncate resume text to header area for training
            header_text = doc["raw_text"][:3000]
            
            # Build the training example in chat format
            user_message = (
                "Extract the candidate's contact and personal information from this resume header.\n"
                "Return ONLY valid JSON with these fields: name, email, phone, date_of_birth, "
                "nationality, location, language, linkedin, github, website.\n"
                "Use null for missing fields.\n\n"
                f"RESUME TEXT:\n{header_text}"
            )
            
            assistant_message = json.dumps(expected_output, ensure_ascii=False)
            
            training_examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are AiMerlion, a resume data extraction engine for Singapore/Malaysia recruitment. "
                            "Extract contact information EXACTLY as written. Output ONLY valid JSON."
                        )
                    },
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            })
        
        # Write JSONL
        output_path = os.path.join(self.output_dir, "header_extraction_train.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        logger.info(
            f"📝 Header training: {len(training_examples)} examples → {output_path} "
            f"(skipped {skipped})"
        )
        return output_path
    
    # -----------------------------------------------------------------
    # 🎯 FORMAT 2: Skills Extraction Training Data
    # -----------------------------------------------------------------
    
    def generate_skills_training(self, documents: List[Dict]) -> str:
        """
        🎯 Generate training data for SKILLS extraction.
        
        This is the HARDEST field for AI to get right because:
        1. Skills can be listed as flat text, bullets, or nested categories
        2. Job descriptions LOOK like skills but aren't
        3. Certifications (CMFAS, NEBOSH) are NOT skills
        4. Section headers ("Skills & Abilities") are NOT skills themselves
        
        Each example teaches the model the BOUNDARY between skills and other fields.
        """
        training_examples = []
        skipped = 0
        
        for doc in documents:
            structured = doc.get("structured", {})
            annotations = doc.get("annotations", [])
            
            # Extract skills from annotations (most reliable source)
            hard_skills = []
            soft_skills = []
            certifications = []
            not_skills = []  # Things commonly confused with skills
            
            for ann in annotations:
                text = ann["text"].strip()
                etype = ann["entity_type"]
                
                if etype == "SKILL":
                    hard_skills.append(text)
                elif etype == "SOFT_SKILL":
                    soft_skills.append(text)
                elif etype == "CERTIFICATION":
                    certifications.append(text)
                elif etype in ("JOB_TITLE", "ORGANIZATION", "JOB_DESCRIPTION", 
                              "SECTION_HEADER", "DEGREE", "INSTITUTION"):
                    not_skills.append({"text": text, "actual_type": etype})
            
            # Also try structured extraction data
            if not hard_skills and structured.get("hard_skills"):
                try:
                    hard_skills = json.loads(structured["hard_skills"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if not soft_skills and structured.get("soft_skills"):
                try:
                    soft_skills = json.loads(structured["soft_skills"])
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Skip if no skills found at all
            if not hard_skills and not soft_skills:
                skipped += 1
                continue
            
            # Deduplicate
            hard_skills = list(dict.fromkeys(hard_skills))
            soft_skills = list(dict.fromkeys(soft_skills))
            certifications = list(dict.fromkeys(certifications))
            
            # Build training example
            # Truncate text but try to include the skills section
            resume_text = doc["raw_text"][:MAX_RESUME_TEXT_LENGTH]
            
            user_message = (
                "Extract ALL skills from this resume. Categorize into hard_skills (technical) "
                "and soft_skills (interpersonal).\n\n"
                "CRITICAL RULES:\n"
                "- Skills are abilities/proficiencies: 'Python', 'Project Management', 'Data Analysis'\n"
                "- NOT skills: company names, job titles, job duties, certifications, education\n"
                "- CMFAS/NEBOSH/etc are CERTIFICATIONS, not skills\n"
                "- Section headers like 'Skills & Abilities' are not skills themselves\n"
                "- Job descriptions like 'Managed team of 5' are NOT skills\n\n"
                "Return JSON: {\"hard_skills\": [...], \"soft_skills\": [...]}\n\n"
                f"RESUME TEXT:\n{resume_text}"
            )
            
            expected = {
                "hard_skills": hard_skills,
                "soft_skills": soft_skills
            }
            
            training_examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are AiMerlion, a resume skills extraction engine for Singapore/Malaysia recruitment. "
                            "Extract skills EXACTLY as written. Distinguish skills from certifications, job titles, "
                            "company names, and job descriptions. Output ONLY valid JSON."
                        )
                    },
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)}
                ]
            })
        
        output_path = os.path.join(self.output_dir, "skills_extraction_train.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        logger.info(
            f"🎯 Skills training: {len(training_examples)} examples → {output_path} "
            f"(skipped {skipped})"
        )
        return output_path
    
    # -----------------------------------------------------------------
    # 💼 FORMAT 3: Experience Extraction Training Data
    # -----------------------------------------------------------------
    
    def generate_experience_training(self, documents: List[Dict]) -> str:
        """
        💼 Generate training data for WORK EXPERIENCE extraction.
        
        Teaches the model to identify:
        - Company name vs job title (comma-separated in SG format)
        - Date ranges (date-first format with TAB separator)
        - Job descriptions (bullet points under each role)
        - Where one job ends and another begins
        """
        training_examples = []
        skipped = 0
        
        for doc in documents:
            structured = doc.get("structured", {})
            
            # Get experience from structured data
            experience_json = structured.get("experience_json", "[]")
            try:
                experience = json.loads(experience_json) if experience_json else []
            except (json.JSONDecodeError, TypeError):
                experience = []
            
            if not experience or not isinstance(experience, list):
                skipped += 1
                continue
            
            # Validate experience entries have meaningful data
            valid_jobs = [
                job for job in experience
                if isinstance(job, dict) and (job.get("company") or job.get("title"))
            ]
            
            if not valid_jobs:
                skipped += 1
                continue
            
            resume_text = doc["raw_text"][:MAX_RESUME_TEXT_LENGTH]
            
            # Normalize experience format for training
            normalized_experience = []
            for job in valid_jobs:
                normalized_experience.append({
                    "company": job.get("company", ""),
                    "role": job.get("title", job.get("role", "")),
                    "dates": job.get("dates", ""),
                    "description": job.get("description", job.get("responsibility", ""))
                })
            
            user_message = (
                "Extract ALL work experience entries from this resume.\n\n"
                "For each job, extract: company, role, dates, description.\n"
                "Singapore resumes use DATE-FIRST format:\n"
                "  'Feb 2016 to Present   Financial Consultant, Prudential (Pte) Ltd'\n"
                "The comma separates Role from Company.\n\n"
                "Return JSON: {\"working_experience\": [{\"company\": ..., \"role\": ..., "
                "\"dates\": ..., \"description\": ...}, ...]}\n\n"
                f"RESUME TEXT:\n{resume_text}"
            )
            
            expected = {"working_experience": normalized_experience}
            
            training_examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are AiMerlion, a resume work experience extraction engine for "
                            "Singapore/Malaysia recruitment. Extract experience EXACTLY as written. "
                            "Output ONLY valid JSON."
                        )
                    },
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)}
                ]
            })
        
        output_path = os.path.join(self.output_dir, "experience_extraction_train.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        logger.info(
            f"💼 Experience training: {len(training_examples)} examples → {output_path} "
            f"(skipped {skipped})"
        )
        return output_path
    
    # -----------------------------------------------------------------
    # 🎓 FORMAT 4: Education Extraction Training Data
    # -----------------------------------------------------------------
    
    def generate_education_training(self, documents: List[Dict]) -> str:
        """🎓 Generate training data for EDUCATION extraction."""
        training_examples = []
        skipped = 0
        
        for doc in documents:
            structured = doc.get("structured", {})
            
            education_json = structured.get("education_json", "[]")
            try:
                education = json.loads(education_json) if education_json else []
            except (json.JSONDecodeError, TypeError):
                education = []
            
            if not education or not isinstance(education, list):
                skipped += 1
                continue
            
            valid_edu = [
                e for e in education
                if isinstance(e, dict) and (e.get("institution") or e.get("degree"))
            ]
            
            if not valid_edu:
                skipped += 1
                continue
            
            resume_text = doc["raw_text"][:MAX_RESUME_TEXT_LENGTH]
            
            user_message = (
                "Extract ALL education entries from this resume.\n"
                "For each entry: institution, degree, dates.\n\n"
                "Return JSON: {\"education\": [{\"institution\": ..., \"degree\": ..., "
                "\"dates\": ...}, ...]}\n\n"
                f"RESUME TEXT:\n{resume_text}"
            )
            
            expected = {"education": valid_edu}
            
            training_examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are AiMerlion, a resume education extraction engine for "
                            "Singapore/Malaysia recruitment. Output ONLY valid JSON."
                        )
                    },
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": json.dumps(expected, ensure_ascii=False)}
                ]
            })
        
        output_path = os.path.join(self.output_dir, "education_extraction_train.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        logger.info(
            f"🎓 Education training: {len(training_examples)} examples → {output_path} "
            f"(skipped {skipped})"
        )
        return output_path
    
    # -----------------------------------------------------------------
    # 🌟 FORMAT 5: Few-Shot Example Generator
    # -----------------------------------------------------------------
    
    def generate_few_shot_examples(
        self, 
        documents: List[Dict], 
        num_examples: int = 5,
        field: str = "all"
    ) -> Dict[str, List[Dict]]:
        """
        🌟 Generate few-shot examples for prompt engineering.
        
        These examples can be IMMEDIATELY injected into the Ollama prompts
        in ai_extractor.py for instant accuracy improvement — no fine-tuning needed!
        
        Selects the BEST examples (most complete annotations) for each field type.
        
        Args:
            documents: Annotated documents
            num_examples: Number of examples per field
            field: "all", "header", "skills", "experience", or "education"
        
        Returns:
            Dict with field names → list of example dicts
        """
        examples = {"header": [], "skills": [], "experience": [], "education": []}
        
        # Score documents by completeness
        scored_docs = []
        for doc in documents:
            score = 0
            structured = doc.get("structured", {})
            annotations = doc.get("annotations", [])
            
            # Score based on how many fields are populated
            if structured.get("name"): score += 2
            if structured.get("email"): score += 1
            if structured.get("phone"): score += 1
            if structured.get("date_of_birth"): score += 2
            
            # Score annotations
            entity_types = set(a["entity_type"] for a in annotations)
            score += len(entity_types)
            
            # Penalize very long texts (harder examples)
            text_len = len(doc.get("raw_text", ""))
            if text_len > 8000:
                score -= 1
            
            scored_docs.append((score, doc))
        
        # Sort by score (best first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Generate examples for each field
        for _, doc in scored_docs[:num_examples * 2]:  # Get more than needed, then filter
            structured = doc.get("structured", {})
            annotations = doc.get("annotations", [])
            header_text = doc["raw_text"][:2000]
            
            # Header example
            if len(examples["header"]) < num_examples:
                header_output = {
                    "name": structured.get("name"),
                    "email": structured.get("email"),
                    "phone": structured.get("phone"),
                    "date_of_birth": structured.get("date_of_birth"),
                    "nationality": structured.get("nationality"),
                    "location": structured.get("location"),
                }
                non_null = sum(1 for v in header_output.values() if v)
                if non_null >= 2:
                    examples["header"].append({
                        "input_preview": header_text[:500],
                        "output": header_output
                    })
            
            # Skills example
            if len(examples["skills"]) < num_examples:
                hard = []
                soft = []
                for ann in annotations:
                    if ann["entity_type"] == "SKILL":
                        hard.append(ann["text"])
                    elif ann["entity_type"] == "SOFT_SKILL":
                        soft.append(ann["text"])
                
                if hard or soft:
                    examples["skills"].append({
                        "input_preview": doc["raw_text"][:500],
                        "output": {
                            "hard_skills": list(dict.fromkeys(hard)),
                            "soft_skills": list(dict.fromkeys(soft))
                        }
                    })
        
        # Save examples to JSON
        output_path = os.path.join(self.output_dir, "few_shot_examples.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🌟 Few-shot examples saved to {output_path}")
        for field_name, field_examples in examples.items():
            logger.info(f"   {field_name}: {len(field_examples)} examples")
        
        return examples


# =============================================================================
# 🏗️ STEP 3: MODELFILE GENERATOR WITH TRAINING EXAMPLES
# =============================================================================

class ModelfileGenerator:
    """
    🏗️ Generates a custom Ollama Modelfile with baked-in training examples.
    
    This is the FASTEST path to better extraction accuracy:
    instead of fine-tuning (which requires GPU time and technical setup),
    we bake the BEST examples directly into the system prompt.
    
    The model sees these examples EVERY time it runs, building
    pattern recognition for SG/MY resume formats.
    """
    
    @staticmethod
    def generate(
        few_shot_examples: Dict[str, List[Dict]],
        base_model: str = "llama3.1:8b",
        output_path: str = "Modelfile.trained"
    ) -> str:
        """
        Generate a Modelfile with training examples baked into the system prompt.
        
        Args:
            few_shot_examples: Output from TrainingDataConverter.generate_few_shot_examples()
            base_model: Base Ollama model to build on
            output_path: Where to save the Modelfile
        """
        # Build example sections for the system prompt
        example_sections = []
        
        # Header examples
        for i, ex in enumerate(few_shot_examples.get("header", [])[:3], 1):
            example_sections.append(f"""
### Trained Example {i}: Header Extraction
Input (first 300 chars): "{ex['input_preview'][:300]}..."
Correct output: {json.dumps(ex['output'], ensure_ascii=False)}
""")
        
        # Skills examples
        for i, ex in enumerate(few_shot_examples.get("skills", [])[:3], 1):
            example_sections.append(f"""
### Trained Example {i}: Skills Extraction  
Input preview: "{ex['input_preview'][:300]}..."
Correct output: {json.dumps(ex['output'], ensure_ascii=False)}
""")
        
        examples_text = "\n".join(example_sections)
        
        modelfile_content = f'''# Auto-generated by train_ollama_extractor.py
# Generated: {datetime.now().isoformat()}
# Training examples: {sum(len(v) for v in few_shot_examples.values())}

FROM {base_model}

PARAMETER temperature 0.1
PARAMETER top_p 0.4
PARAMETER top_k 20
PARAMETER num_ctx 16384
PARAMETER num_predict 2000
PARAMETER repeat_penalty 1.1

SYSTEM """You are AiMerlion, a precision resume data extraction engine for Singapore/Malaysia recruitment.

## RULES
1. Output ONLY valid JSON
2. Use null for missing fields
3. Extract EXACTLY as written — never paraphrase
4. Skills are abilities (Python, Excel) — NOT job titles, companies, certifications, or duties

## TRAINED EXAMPLES FROM REAL SG/MY RESUMES
{examples_text}

## FIELD DISAMBIGUATION
- SKILL: "Python", "Data Analysis", "Project Management", "AutoCAD"
- NOT SKILL: "DBS Bank" (company), "Software Engineer" (job title), "CMFAS M5" (certification)
- NOT SKILL: "Managed team of 5" (job duty), "Skills & Abilities" (section header)

Follow these examples when extracting. Match the format and field assignments exactly.
"""
'''
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        logger.info(f"🏗️ Modelfile generated: {output_path}")
        logger.info(f"   To create model: ollama create aimerlion-trained -f {output_path}")
        
        return output_path


# =============================================================================
# 📊 STEP 4: TRAINING REPORT
# =============================================================================

def generate_training_report(stats: Dict, documents: List[Dict], output_dir: str) -> str:
    """
    📊 Generate a human-readable training data report.
    
    Shows what data is available, what was generated, and recommendations
    for improving extraction accuracy.
    """
    report_lines = [
        "=" * 70,
        "💅✨ AiMerlion Training Data Report ✨💅",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "📊 DATABASE STATISTICS",
        "-" * 40,
    ]
    
    # Document counts
    doc_stats = stats.get("documents_by_status", {})
    for status, count in doc_stats.items():
        emoji = "✅" if status == "completed" else "🔵" if status == "in_progress" else "⬜"
        report_lines.append(f"  {emoji} {status}: {count}")
    
    report_lines.append(f"  📝 Total annotations: {stats.get('total_annotations', 0)}")
    report_lines.append(f"  👥 Annotators: {', '.join(stats.get('annotators', ['none']))}")
    
    # Entity distribution
    report_lines.extend(["", "📋 ENTITY TYPE DISTRIBUTION", "-" * 40])
    for etype, count in sorted(
        stats.get("entity_distribution", {}).items(), 
        key=lambda x: x[1], reverse=True
    )[:15]:
        bar = "█" * min(count // 5, 30)
        report_lines.append(f"  {etype:25s} {count:5d} {bar}")
    
    # Training data summary
    report_lines.extend(["", "🎓 TRAINING DATA GENERATED", "-" * 40])
    
    training_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
    for tf in training_files:
        filepath = os.path.join(output_dir, tf)
        line_count = sum(1 for _ in open(filepath, encoding="utf-8"))
        report_lines.append(f"  📄 {tf}: {line_count} examples")
    
    # Recommendations
    completed = doc_stats.get("completed", 0)
    report_lines.extend(["", "💡 RECOMMENDATIONS", "-" * 40])
    
    if completed < 20:
        report_lines.append(
            "  ⚠️ Only {completed} completed annotations — aim for 50+ for reliable training."
        )
        report_lines.append(
            "  💡 Focus on annotating diverse resume formats (IT, finance, admin, fresh grad)."
        )
    elif completed < 50:
        report_lines.append(
            f"  🔵 {completed} completed — good start! 50+ recommended for fine-tuning."
        )
        report_lines.append(
            "  💡 Use the few-shot examples NOW for immediate improvement via prompt engineering."
        )
    else:
        report_lines.append(
            f"  ✅ {completed} completed — excellent! Ready for fine-tuning."
        )
        report_lines.append(
            "  💡 Consider LoRA fine-tuning with Unsloth for maximum accuracy gains."
        )
    
    report_lines.extend([
        "",
        "🚀 NEXT STEPS",
        "-" * 40,
        "  1. IMMEDIATE: Create custom model with Modelfile",
        "     ollama create aimerlion-extractor -f Modelfile",
        "",
        "  2. QUICK WIN: Apply surgical edits to ai_extractor.py",
        "     (See the edit instructions in the companion guide)",
        "",
        "  3. BEST ACCURACY: Fine-tune with training data",
        "     (Requires 50+ examples and GPU for LoRA training)",
        "",
        "=" * 70,
    ])
    
    report = "\n".join(report_lines)
    
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)
    return report_path


# =============================================================================
# 🚀 MAIN — CLI ENTRY POINT
# =============================================================================

def main():
    """
    🚀 Main entry point for the training pipeline.
    
    Usage:
        python train_ollama_extractor.py
        python train_ollama_extractor.py --db path/to/resume_extractions.db
        python train_ollama_extractor.py --mode few-shot --num-examples 5
        python train_ollama_extractor.py --mode modelfile --base-model llama3.1:8b
        python train_ollama_extractor.py --mode all
    """
    parser = argparse.ArgumentParser(
        description="💅 AiMerlion Ollama Training Pipeline"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_PATH,
        help=f"Path to resume_extractions.db (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--mode", choices=["all", "training-data", "few-shot", "modelfile", "stats"],
        default="all",
        help="What to generate (default: all)"
    )
    parser.add_argument(
        "--num-examples", type=int, default=5,
        help="Number of few-shot examples per field (default: 5)"
    )
    parser.add_argument(
        "--base-model", default="llama3.1:8b",
        help="Base Ollama model for Modelfile (default: llama3.1:8b)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("💅✨ AiMerlion Ollama Training Pipeline ✨💅")
    print("=" * 70)
    
    # Initialize
    extractor = AnnotationDataExtractor(args.db)
    converter = TrainingDataConverter(args.output)
    
    # Get stats
    stats = extractor.get_annotation_stats()
    
    if args.mode == "stats":
        generate_training_report(stats, [], args.output)
        return
    
    # Get completed documents
    documents = extractor.get_completed_documents()
    
    if not documents:
        print("\n❌ No completed annotations found!")
        print("   Complete some annotations in the annotation tool first:")
        print("   python annotation_tool.py → http://localhost:5055")
        print(f"\n📊 Current stats: {json.dumps(stats.get('documents_by_status', {}))}")
        return
    
    print(f"\n📋 Found {len(documents)} completed documents")
    
    # Generate based on mode
    if args.mode in ("all", "training-data"):
        print("\n🎓 Generating training data...")
        converter.generate_header_training(documents)
        converter.generate_skills_training(documents)
        converter.generate_experience_training(documents)
        converter.generate_education_training(documents)
    
    few_shot_examples = {}
    if args.mode in ("all", "few-shot", "modelfile"):
        print("\n🌟 Generating few-shot examples...")
        few_shot_examples = converter.generate_few_shot_examples(
            documents, num_examples=args.num_examples
        )
    
    if args.mode in ("all", "modelfile"):
        print("\n🏗️ Generating Modelfile...")
        ModelfileGenerator.generate(
            few_shot_examples,
            base_model=args.base_model,
            output_path=os.path.join(args.output, "Modelfile.trained")
        )
    
    # Generate report
    generate_training_report(stats, documents, args.output)
    
    print("\n✅ Training pipeline complete!")
    print(f"📁 Output: {args.output}/")


if __name__ == "__main__":
    main()