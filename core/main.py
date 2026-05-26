"""
main.py

This module serves as the primary entry point and orchestrator for the AiMerlion resume
extraction system. It provides an interactive command-line interface (CLI) for users
to process resumes, generate reports, and diagnose AI performance.

It initializes the UltimateResumeExtractor, which handles the complex logic of
document parsing, text extraction (including OCR fallbacks), and AI-powered
(or regex-based) data extraction from various resume formats (PDF, DOCX).

Key functionalities include:
- Interactive menu for user interaction.
- Orchestration of resume processing across candidate folders.
- Graceful handling of interruptions (e.g., Ctrl+C) with checkpointing.
- Generation of detailed CSV and JSON reports summarizing extraction results and AI assistance.
- Dynamic selection and initialization of OCR engines.
"""
# 🧙‍♀️✨ Fairy Codemother's ULTIMATE Resume Extractor - THE ID DIVA EDITION! ✨🧙‍♀️

import numpy as np
import os
import re
import datetime
import pandas as pd
from tqdm import tqdm
import logging
import coloredlogs
import ollama
import pypdf
import docx
import docx2txt
import pdfplumber
from pathlib import Path
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
from paddleocr import PaddleOCR
from PIL import Image
import pdf2image
from typing import Dict, List, Tuple, Optional
import math
import time
import unicodedata
import json
import config
from utils import display_menu, save_checkpoint, load_checkpoint, get_checkpoint_info, clear_checkpoint, print_batch_table, FeedbackLoopSystem, InteractiveCorrectionSystem, PatternLearningSystem, PerformanceMonitor, standardize_phone_number, standardize_date
# 🗄️ Database Manager for raw + structured storage
if config.DATABASE_ENABLED:
    from db_manager import DatabaseManager
from extraction.ai_validator import AIValidator
from extraction.marker_extractor import get_marker_extractor
from extraction.document_parser import DocumentParser
from pdf_inspector import analyze_pdf_type

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - 💄 %(levelname)s - %(message)s')

def find_candidate_folders(base_folder: str) -> List[str]:
    """
    🔍 Find all candidate folders within the base directory structure.
    Handles structure like: merlion_resumes/2025-07-10_16-54-36/[candidate_folders]
    """
    candidate_folders = []
    
    # First, check if base_folder has date-like subfolders
    for item in os.scandir(base_folder):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this is a date folder (like 2025-07-10_16-54-36)
            if re.match(r'\d{4}-\d{2}-\d{2}', item.name):
                logger.info(f"📁 Found date folder: {item.name}")
                # Look for candidate folders inside
                for candidate in os.scandir(item.path):
                    if candidate.is_dir() and not candidate.name.startswith('.'):
                        # Any folder inside the date folder is a candidate folder
                        logger.debug(f"   📂 Found candidate folder: {candidate.name}")
                        candidate_folders.append(candidate.path)
            # Or if it's directly a candidate folder (fallback)
            elif re.match(r'^\d+', item.name):
                candidate_folders.append(item.path)
    
    logger.info(f"🎯 Total candidate folders found: {len(candidate_folders)}")
    return sorted(candidate_folders)


class UltimateResumeExtractor:
    """
    🎭 THE ULTIMATE DIVA! This queen relies on POWERFUL regex patterns
    and strategic AI usage. She's learned that sometimes, you need to
    do the heavy lifting yourself instead of relying on others!
    """


    def __init__(self, model_name: str):
        self.model_name = model_name

        # 📄 Marker PDF Extractor
        self.use_marker = config.USE_MARKER_PDF
        self.marker_extractor = None
        if self.use_marker:
            try:
                self.marker_extractor = get_marker_extractor()
                if self.marker_extractor.available:
                    logger.info("📄 Marker PDF extraction ENABLED!")
                else:
                    logger.warning("⚠️ Marker not available - using pdfplumber")
            except Exception as e:
                logger.warning(f"⚠️ Marker init failed: {e} - using pdfplumber")

        # 🆕 OCR Engine Setup
        self.use_ocr = getattr(config, 'USE_OCR', True)
        self.ocr_engine = getattr(config, 'OCR_ENGINE', 'pytesseract')
        self.ocr_available = False
        self.ocr_instance = None

        if self.use_ocr:
            self._initialize_ocr_engine()

        # 📄 Document Parser (for vector/outline text PDFs)
        self.document_parser = None
        try:
            self.document_parser = DocumentParser()
            logger.info("📄 DocumentParser initialized for OCR fallback")
        except Exception as e:
            logger.warning(f"⚠️ DocumentParser init failed: {e}")

        # 🔗 SINGLE SOURCE OF TRUTH for raw text extraction.
        # get_text_from_file() delegates to this so the text fed to the
        # AI/regex extractor is byte-identical to what extract_raw_text.py
        # writes into raw_text_output/. (Marker is a singleton — no double load.)
        self.hybrid_extractor = None
        try:
            from extract_raw_text import HybridExtractor
            self.hybrid_extractor = HybridExtractor()
            logger.info("🔗 HybridExtractor initialized (shared raw-text pipeline)")
        except Exception as e:
            logger.warning(f"⚠️ HybridExtractor init failed: {e} — falling back to legacy path")

        # AI setup
        self.use_ai = config.USE_AI_EXTRACTION
        self.ai_extractor = None
        self.ai_enabled = False

        if self.use_ai:
            try:
                from extraction.ai_extractor import AIExtractor
                self.ai_extractor = AIExtractor(model_name, logger=logger)
                self.ai_enabled = self.ai_extractor.available
                if self.ai_enabled:
                    logger.info("🌸 AI extraction ENABLED!")
                else:
                    logger.warning("⚠️ AI not available - using regex only")
            except Exception as e:
                logger.error(f"❌ Could not initialize AI: {e}")
        else:
            logger.info("🎯 AI extraction disabled - using regex only")

    def _clean_text_for_extraction(self, text: str) -> str:
        """
        🧹 Clean resume text BEFORE extraction!
        
        This fixes encoding issues that break regex patterns - 
        often THE reason why extraction fails!
        """
        if not text:
            return ""
        
        # Fix common encoding garbage
        replacements = {
            '\u00a0': ' ',      # Non-breaking space → regular space
            '\u2028': '\n',     # Line separator
            '\u2029': '\n',     # Paragraph separator
            '\u200b': '',       # Zero-width space (REMOVE)
            '\u200c': '',       # Zero-width non-joiner
            '\u200d': '',       # Zero-width joiner
            '\ufeff': '',       # BOM (REMOVE)
            '–': '-',           # En-dash → hyphen
            '—': '-',           # Em-dash → hyphen
            ''': "'",           # Smart quote → regular
            ''': "'",           # Smart quote → regular
            '"': '"',           # Smart quote → regular
            '"': '"',           # Smart quote → regular
            '•': '* ',          # Bullet → asterisk
            '●': '* ',          # Bullet
            '○': '* ',          # Bullet
            '■': '* ',          # Bullet
            '□': '* ',          # Bullet
            '▪': '* ',          # Bullet
            '►': '* ',          # Bullet
            '　': ' ',          # Full-width space → regular
            '\r\n': '\n',       # Windows line endings
            '\r': '\n',         # Old Mac line endings
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove control characters (except newlines and tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize multiple spaces (but preserve newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize multiple newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()

    def _initialize_ocr_engine(self):
        """
        🔧 Initialize the configured OCR engine
        Supports: pytesseract, paddleocr, easyocr
        """
        ocr_engine = self.ocr_engine.lower()

        if ocr_engine == "pytesseract":
            try:
                import pytesseract
                # Test if tesseract is available
                pytesseract.get_tesseract_version()
                self.ocr_available = True
                self.ocr_instance = pytesseract
                logger.info("✅ Pytesseract OCR initialized!")
            except Exception as e:
                logger.warning(f"⚠️ Pytesseract not available: {e}")
                self._try_fallback_ocr()

        elif ocr_engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                lang = getattr(config, 'OCR_LANGUAGE', 'en')
                # Map common language codes
                paddle_lang = 'en' if lang == 'eng' else lang
                self.ocr_instance = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    show_log=False
                )
                self.ocr_available = True
                logger.info("✅ PaddleOCR initialized!")
            except Exception as e:
                logger.warning(f"⚠️ PaddleOCR not available: {e}")
                self._try_fallback_ocr()

        elif ocr_engine == "easyocr":
            try:
                import easyocr
                lang = getattr(config, 'OCR_LANGUAGE', 'en')
                # Map to easyocr language code
                easy_lang = ['en'] if lang == 'eng' else [lang]
                self.ocr_instance = easyocr.Reader(easy_lang, gpu=False)
                self.ocr_available = True
                logger.info("✅ EasyOCR initialized!")
            except Exception as e:
                logger.warning(f"⚠️ EasyOCR not available: {e}")
                self._try_fallback_ocr()
        else:
            logger.warning(f"⚠️ Unknown OCR engine: {ocr_engine}, trying pytesseract")
            self.ocr_engine = "pytesseract"
            self._initialize_ocr_engine()

    def _try_fallback_ocr(self):
        """Try alternative OCR engines if primary fails"""
        fallback_engines = ["pytesseract", "paddleocr", "easyocr"]
        current = self.ocr_engine.lower()

        for engine in fallback_engines:
            if engine != current:
                logger.info(f"🔄 Trying fallback OCR: {engine}")
                self.ocr_engine = engine
                try:
                    self._initialize_ocr_engine()
                    if self.ocr_available:
                        return
                except:
                    continue

        logger.warning("❌ No OCR engine available - OCR disabled")
        self.ocr_available = False

    def _extract_with_ocr(self, file_path: str) -> Optional[str]:
        """
        📸 Extract text from PDF using OCR
        Uses the same simple approach as test_ocr.py for reliability
        """
        if not self.ocr_available:
            return None

        file_name = os.path.basename(file_path)
        ocr_dpi = getattr(config, 'OCR_DPI', 200)  # Default 200 like test_ocr.py

        try:
            logger.info(f"📸 Running OCR on {file_name} using {self.ocr_engine} (DPI: {ocr_dpi})...")

            # Convert PDF to images (same as test_ocr.py)
            images = pdf2image.convert_from_path(file_path, dpi=ocr_dpi)
            logger.info(f"   🖼️ Converted {len(images)} pages to images")

            all_text = []

            for i, img in enumerate(images):
                logger.info(f"   📄 OCR processing page {i+1}/{len(images)}")
                page_text = ""

                if self.ocr_engine == "pytesseract":
                    # Simple call like test_ocr.py - NO extra config for reliability!
                    page_text = self.ocr_instance.image_to_string(img, lang='eng')

                elif self.ocr_engine == "paddleocr":
                    # Convert PIL Image to numpy array
                    img_array = np.array(img)
                    result = self.ocr_instance.ocr(img_array, cls=True)

                    # Extract ALL text from PaddleOCR (no confidence filter)
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], tuple) else line[1]
                                page_text += str(text) + "\n"

                elif self.ocr_engine == "easyocr":
                    # Convert PIL Image to numpy array
                    img_array = np.array(img)
                    result = self.ocr_instance.readtext(img_array)

                    # Extract ALL text from EasyOCR (no confidence filter)
                    for detection in result:
                        text = detection[1]
                        page_text += text + "\n"
                else:
                    page_text = ""

                if page_text.strip():
                    all_text.append(page_text.strip())
                    word_count = len(page_text.split())
                    logger.info(f"      ✅ Page {i+1}: {word_count} words extracted")
                else:
                    logger.warning(f"      ⚠️ Page {i+1}: No text extracted")

            combined_text = "\n\n".join(all_text)

            if combined_text.strip():
                total_words = len(combined_text.split())
                logger.info(f"✅ OCR complete: {len(combined_text)} chars, {total_words} words from {file_name}")
                return combined_text
            else:
                logger.warning(f"⚠️ OCR produced no text from {file_name}")
                return None

        except Exception as e:
            logger.error(f"❌ OCR failed for {file_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def process_individual_file(self, file_path: str) -> Dict:
        """
        🎯 Process a single resume file as an individual candidate
        """
        logger.info(f"📄 Processing individual file: {os.path.basename(file_path)}")
        
        # Extract ID from filename if possible
        filename = os.path.basename(file_path)
        id_match = re.search(r'(\d+)', filename)
        
        result = {
            "ID": int(id_match.group(1)) if id_match else None,
            "Name": None,
            "Email": None,
            "Phone": None,
            "Skills": None,
            "Working_Experience": None,
            "Location": None,
            "School_University": None,
            "Language": "English",
            "Extraction_Status": "Failed",
            "Notes": "",
            "AI_Assisted": False,
            "Filenames_Processed": filename
        }

        # Get text and extract data
        text = self.get_text_from_file(file_path)
        if text and len(text.strip()) >= 50:
            extracted_data, ai_used = self._extract_data_from_text(text)

            if ai_used:
                result["AI_Assisted"] = True

            # Update result with extracted data - ALL fields!
            for field in ["Name", "Email", "Phone", "Skills",
                          "Working_Experience", "Location", "School_University"]:
                if extracted_data.get(field):
                    result[field] = extracted_data[field]
            result["Phone_Method"] = extracted_data.get("Phone_Method", "Regex")
            result["Email_Method"] = extracted_data.get("Email_Method", "Regex")
            
            # Try to extract name from filename if still missing
            if not result["Name"]:
                name_from_file = self._extract_name_from_filename(filename)
                if name_from_file:
                    result["Name"] = name_from_file
                    result["Notes"] += "Name extracted from filename. "
        
        self._set_extraction_status(result, text if text else "")
        return result

    def _apply_learned_patterns(self):
        """
        🧠 Apply patterns we've learned from corrections!
        """
        patterns = self.feedback_system.load_learned_patterns()
        
        # Check if we have significant learnings
        total_corrections = sum(
            len(patterns.get(field, {}).get('transformations', []))
            for field in patterns
        )
        
        if total_corrections > 10:
            logger.info(f"🧠 Loaded {total_corrections} learned patterns!")
            
            # Suggest improvements
            suggestions = self.pattern_learner.generate_regex_suggestions()
            for field, field_suggestions in suggestions.items():
                if field_suggestions:
                    logger.info(f"💡 Suggestions for {field}: {field_suggestions[0]}")

    def _extract_with_mega_regex(self, text: str) -> Dict[str, Optional[str]]:
        """
        🌟 MEGA REGEX for ENGLISH resumes only!
        Cleaner, faster, more accurate!
        
        🧚‍♀️ NOW WITH TEXT CLEANING to fix encoding issues!
        """
        logger.debug("🎭 ENGLISH-ONLY REGEX BEGINS!")
        
        # 🧹 CLEAN TEXT FIRST - fixes encoding issues that break regex!
        text = self._clean_text_for_extraction(text)
        
        data: Dict[str, Optional[str]] = {
            "name": None,
            "email": None,
            "phone": None,
            "skills": None,
            "working_experience": None,
            "location": None,
            "school_university": None
        }
        
        # Try table extraction first
        table_data = self._extract_from_table_format(text)
        data.update(table_data)
        
        # --- 📧 EMAIL EXTRACTION (sanitized + mailto-aware) ---
        if not data["email"]:
            # Pass 1: text with only invisible/zero-width junk stripped — this
            #         is lossless for clean resumes but repairs addresses split
            #         by ZWSP/BOM (e.g. "ka<ZWSP>rthik@host" → "karthik@host")
            #         so the raw pass can't "succeed" with a truncated address.
            # Pass 2: a fully de-obfuscated copy that also repairs PDF-split,
            #         spaced and "(at)(dot)" addresses ("john . doe @ gmail .
            #         com", "john [at] gmail [dot] com", "john＠gmail.com").
            for source, label in ((self._strip_invisible(text), "raw"),
                                  (self._normalize_for_email(text), "normalized")):
                for candidate in self._gather_email_candidates(source):
                    clean = self._sanitize_email(candidate)
                    if clean:
                        data["email"] = clean
                        logger.info(f"✨ Found email ({label}): {clean}")
                        break
                if data["email"]:
                    break
        
        # --- 📱 PHONE EXTRACTION ---
        if not data["phone"]:
            phone = self._extract_phone_english(text, data.get("email"))
            if phone:
                data["phone"] = phone
        
        # --- 👤 NAME EXTRACTION ---
        if not data["name"]:
            name = self._extract_name_english(text)
            if name:
                data["name"] = name
        
        # --- 🎯 SKILLS ---
        if not data["skills"]:
            skills_match = re.search(r"(?:Skills|SKILLS|Technical Skills|Core Competencies)[\s:]*([^\n\r]+(?:\n(?![\n])[^\n]+)*)", 
                                    text, re.IGNORECASE)
            if skills_match:
                data["skills"] = skills_match.group(1).strip()
        
        # --- 💼 WORK EXPERIENCE ---
        if not data["working_experience"]:
            exp_match = re.search(r"(?:Experience|EXPERIENCE|Work Experience|Employment History)[\s:]*([^\n\r]+(?:\n(?![\n])[^\n]+)*)", 
                                text, re.IGNORECASE)
            if exp_match:
                data["working_experience"] = exp_match.group(1).strip()
        
        # --- 📍 LOCATION ---
        if not data["location"]:
            location_match = re.search(r"(?:Location|Address|City|Residence)[\s:]*([^\n\r]+)", 
                                    text, re.IGNORECASE)
            if location_match:
                data["location"] = location_match.group(1).strip()
        
        # --- 🎓 EDUCATION ---
        if not data["school_university"]:
            edu_match = re.search(r"(?:Education|EDUCATION|University|College|School)[\s:]*([^\n\r]+)", 
                                text, re.IGNORECASE)
            if edu_match:
                data["school_university"] = edu_match.group(1).strip()
        
        return data

    def _extract_phone_english(self, text: str, email: Optional[str]) -> Optional[str]:
        """
        📱 BULLETPROOF phone extraction with INTERNATIONAL support!
        
        NOW SUPPORTS:
        - Singapore: +65 9XXX XXXX, 9XXX XXXX (8 digits)
        - Malaysia: +60 12-XXX XXXX, 012-XXX XXXX
        - India: +91 XXXXX XXXXX
        - US/Canada: (XXX) XXX-XXXX, XXX-XXX-XXXX
        - UK: +44 7XXX XXXXXX
        - Australia: +61 4XX XXX XXX
        - Generic international formats
        """
        
        # If we have an email, search near it first (contact info is usually grouped)
        search_areas = []
        if email:
            email_index = text.lower().find(email.lower())
            if email_index != -1:
                start = max(0, email_index - 500)
                end = min(len(text), email_index + 500)
                search_areas.append(("near email", text[start:end]))
        
        # Header area (contact info is usually in first 3000 chars)
        search_areas.append(("header", text[:3000] if len(text) > 3000 else text))
        
        # Full text as last resort
        search_areas.append(("full text", text))
        
        # ==========================================================================
        # 📱 COMPREHENSIVE PHONE PATTERNS - ALL COUNTRIES!
        # ==========================================================================
        phone_patterns = [
            # ========== LABELED PATTERNS (highest priority!) ==========
            # "Phone: +65 9123 4567" or "Tel: 91234567" or "HP: 012-3456789"
            (r'(?:phone|tel|mobile|cell|contact|hp|h/p|handphone|no\.?\s*(?:tel|hp))[\s.:]*\+?[\d\s.()\-]{7,20}', 'LABELED', 7),
            
            # ========== SINGAPORE (+65) ==========
            # +65 9123 4567 or +65 91234567 or +6591234567
            (r'\+65[\s.-]?[689]\d{3}[\s.-]?\d{4}', 'SG', 8),
            # 65 9123 4567 (without +)
            (r'(?<!\d)65[\s.-]?[689]\d{3}[\s.-]?\d{4}(?!\d)', 'SG', 8),
            # Local SG: 9123 4567 or 91234567 (8 digits starting with 6, 8, or 9)
            (r'(?<!\d)[689]\d{3}[\s.-]?\d{4}(?!\d)', 'SG_LOCAL', 8),
            
            # ========== MALAYSIA (+60) ==========
            # +60 12-345 6789 or +60 123456789
            (r'\+60[\s.-]?\d{1,2}[\s.-]?\d{3,4}[\s.-]?\d{4}', 'MY', 9),
            # Local MY: 012-345 6789 or 0123456789
            (r'(?<!\d)0\d{1,2}[\s.-]?\d{3,4}[\s.-]?\d{4}(?!\d)', 'MY_LOCAL', 10),
            
            # ========== INDIA (+91) ==========
            # +91 98765 43210
            (r'\+91[\s.-]?[6-9]\d{4}[\s.-]?\d{5}', 'IN', 10),
            # Local IN: 98765 43210 (10 digits starting with 6-9)
            (r'(?<!\d)[6-9]\d{4}[\s.-]?\d{5}(?!\d)', 'IN_LOCAL', 10),
            
            # ========== US/CANADA (+1) ==========
            # +1 (123) 456-7890
            (r'\+1[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', 'US', 10),
            # (123) 456-7890
            (r'\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}', 'US_LOCAL', 10),
            # 123-456-7890
            (r'(?<!\d)\d{3}[\s.-]\d{3}[\s.-]\d{4}(?!\d)', 'US_LOCAL', 10),
            
            # ========== UK (+44) ==========
            # +44 7XXX XXXXXX
            (r'\+44[\s.-]?7\d{3}[\s.-]?\d{6}', 'UK', 10),
            # 07XXX XXXXXX
            (r'(?<!\d)07\d{3}[\s.-]?\d{6}(?!\d)', 'UK_LOCAL', 11),
            
            # ========== AUSTRALIA (+61) ==========
            # +61 4XX XXX XXX
            (r'\+61[\s.-]?4\d{2}[\s.-]?\d{3}[\s.-]?\d{3}', 'AU', 9),
            # 04XX XXX XXX
            (r'(?<!\d)04\d{2}[\s.-]?\d{3}[\s.-]?\d{3}(?!\d)', 'AU_LOCAL', 10),
            
            # ========== GENERIC INTERNATIONAL ==========
            (r'\+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{0,4}', 'INTL', 8),
            
            # ========== GENERIC FALLBACK ==========
            (r'(?<!\d)[\d][\d\s.\-()]{6,18}[\d](?!\d)', 'GENERIC', 8),
        ]
        
        for area_name, search_text in search_areas:
            for pattern, pattern_type, min_digits in phone_patterns:
                try:
                    matches = list(re.finditer(pattern, search_text, re.IGNORECASE))
                except re.error:
                    continue
                
                for match in matches:
                    phone_raw = match.group(0).strip()
                    
                    # Extract digits only
                    digits_only = re.sub(r'\D', '', phone_raw)
                    
                    # =========================================================
                    # 🔧 THE CRITICAL FIX: Accept 8+ digits (not 10+!)
                    # Singapore phones are 8 digits!
                    # =========================================================
                    if len(digits_only) < min_digits:
                        continue
                    if len(digits_only) > 15:
                        continue
                    
                    # Skip fake numbers (all same digit)
                    if len(set(digits_only)) <= 2:
                        continue
                    
                    # Skip obvious false positives
                    if digits_only.startswith('0000') or digits_only.startswith('1234567'):
                        continue
                    
                    # Format, sanitize, and return
                    formatted = self._format_phone_international(phone_raw, pattern_type, digits_only)
                    if formatted:
                        # Country-format branches are already clean; this is a
                        # no-op safety net for GENERIC/INTL/LABELED leftovers.
                        formatted = self._sanitize_phone(formatted) or formatted
                        logger.info(f"✨ Found phone ({pattern_type}): {formatted}")
                        return formatted

        return None
    
    def _format_phone_international(self, raw: str, phone_type: str, digits: str) -> str:
        """📱 Format phone number based on detected country"""
        # Remove label prefix if present
        # 🧹 Remove ALL label prefix variations - comprehensive cleanup!
        # This prevents "HP: 91234567" from being returned as "HP: 91234567"
        cleaned = re.sub(
            r'^(?:'
            r'phone\s*(?:no|number|num)?\.?|'
            r'tel(?:ephone)?\.?|'
            r'mobile\s*(?:no|number|phone)?\.?|'
            r'cell(?:\s*phone)?\.?|'
            r'contact\s*(?:no|number)?\.?|'
            r'h/?p\s*(?:no)?\.?|'
            r'hand\s*phone\.?|'
            r'no\.?\s*(?:tel|hp|phone|mobile)|'
            r'portable\.?|'
            r'whatsapp\.?'
            r')[\s.:)\-]*',
            '', raw, flags=re.IGNORECASE
        ).strip()
        
        # 🛡️ Safety check: if cleanup failed and result still has letters,
        # fall back to extracting just the digits
        if cleaned and re.search(r'[a-zA-Z]{3,}', cleaned):
            # Still has words — extract phone-like digits only
            digit_match = re.search(r'[\+]?[\d][\d\s\-\.\(\)]{6,18}[\d]', cleaned)
            if digit_match:
                cleaned = digit_match.group(0).strip()
            else:
                cleaned = re.sub(r'[^\d+\-\s()]', '', cleaned).strip()
        
        # Singapore
        if phone_type.startswith('SG'):
            if len(digits) == 8:
                return f"+65 {digits[:4]} {digits[4:]}"
            elif len(digits) == 10 and digits.startswith('65'):
                return f"+65 {digits[2:6]} {digits[6:]}"
        
        # Malaysia
        elif phone_type.startswith('MY'):
            if digits.startswith('60') and len(digits) >= 11:
                return f"+60 {digits[2:4]}-{digits[4:7]} {digits[7:]}"
            elif digits.startswith('0') and len(digits) >= 10:
                return f"+60 {digits[1:3]}-{digits[3:6]} {digits[6:]}"
        
        # India
        elif phone_type.startswith('IN'):
            if digits.startswith('91') and len(digits) == 12:
                return f"+91 {digits[2:7]} {digits[7:]}"
            elif len(digits) == 10 and digits[0] in '6789':
                return f"+91 {digits[:5]} {digits[5:]}"
        
        # US/Canada
        elif phone_type.startswith('US'):
            if len(digits) == 10:
                return f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits.startswith('1'):
                return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        # UK
        elif phone_type.startswith('UK'):
            if digits.startswith('44') and len(digits) >= 12:
                return f"+44 {digits[2:6]} {digits[6:]}"
            elif digits.startswith('0') and len(digits) >= 11:
                return f"+44 {digits[1:5]} {digits[5:]}"
        
        # Australia
        elif phone_type.startswith('AU'):
            if digits.startswith('61') and len(digits) >= 11:
                return f"+61 {digits[2:5]} {digits[5:8]} {digits[8:]}"
            elif digits.startswith('04') and len(digits) == 10:
                return f"+61 {digits[1:4]} {digits[4:7]} {digits[7:]}"
        
        # Labeled or generic - clean up and return (sanitized)
        return self._sanitize_phone(cleaned if cleaned else raw) or (cleaned if cleaned else raw.strip())

    def _gather_email_candidates(self, source: str) -> list:
        """
        📨 Collect email candidates from `source`, most-authoritative first:
          1) mailto: hyperlink targets   2) "Email:"-labelled addresses
          3) any bare address token anywhere in the text.
        """
        candidates = []
        # 1️⃣ Markdown / HTML mailto: links are the most authoritative target
        #    e.g. [display@x.com](mailto:real@x.com)  → real@x.com
        candidates += re.findall(
            r'mailto:\s*[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
            source, re.IGNORECASE
        )
        # 2️⃣ Labelled emails ("Email: a@b.com", "Email :** a@b.com", "e-mail – a@b.com")
        candidates += re.findall(
            r'(?:e[\s\-]?mail|mail)[\s:*\-–—]*[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
            source, re.IGNORECASE
        )
        # 3️⃣ Any bare email token anywhere
        candidates += re.findall(
            r'[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}',
            source
        )
        return candidates

    def _strip_invisible(self, text: str) -> str:
        """
        🫥 Remove invisible/zero-width characters that PDF extraction or
        anti-scrape tricks inject *inside* otherwise-clean addresses
        (e.g. "ka<ZWSP>rthik@host" -> "karthik@host"). Lossless for normal
        resume text, so it is safe to run before the first email pass.
        NBSP is a real space, so it becomes one.
        """
        if not text:
            return ""
        # Zero-width junk carries NO spacing -- DELETE it.
        #   ZWSP ZWNJ ZWJ word-joiner BOM/ZWNBSP
        s = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
        # NBSP -> normal space so later @/dot collapsing can act on it.
        return s.replace('\u00a0', ' ')

    def _normalize_for_email(self, text: str) -> str:
        """
        🧽 Produce an email-search-only copy of `text` that repairs
        addresses broken by PDF extraction or anti-scrape obfuscation.
        Used ONLY as a fallback for email detection -- never mutates the
        text other fields use.

        Repairs:
          - Invisible/zero-width chars (via _strip_invisible)
          - Unicode lookalikes:  fullwidth @ -> @,  fullwidth/ideographic dot -> .
          - HTML entities:  &#64; &commat; -> @,  &#46; -> .
          - Worded obfuscation:  "(at)" "[at]" "{at}" " at "  -> @
                                 "(dot)" "[dot]" "{dot}" " dot " -> .
          - PDF-split addresses:  "john . doe @ gmail . com" / line breaks
            inside the address  -> "john.doe@gmail.com"
        """
        if not text:
            return ""
        s = self._strip_invisible(text)

        # Unicode lookalikes
        s = s.replace('＠', '@')          # ＠ fullwidth at
        s = re.sub(r'[．。]', '.', s)  # ． fullwidth / 。 ideographic dot

        # HTML entities for @ and .
        s = re.sub(r'&#0*64;|&commat;|&#x40;', '@', s, flags=re.IGNORECASE)
        s = re.sub(r'&#0*46;|&period;|&#x2e;', '.', s, flags=re.IGNORECASE)

        # Worded obfuscation: (at) [at] {at} " at " / arroba  → @
        s = re.sub(r'\s*[\(\[\{]\s*(?:at|arroba)\s*[\)\]\}]\s*', '@', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+(?:at|arroba)\s+(?=[A-Za-z0-9.\-]+\s*(?:[\(\[\{]\s*)?dot)',
                   '@', s, flags=re.IGNORECASE)
        # Worded obfuscation: (dot) [dot] {dot} " dot "  → .
        s = re.sub(r'\s*[\(\[\{]\s*(?:dot|punto)\s*[\)\]\}]\s*', '.', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+(?:dot|punto)\s+', '.', s, flags=re.IGNORECASE)

        # Collapse whitespace (incl. newlines) hugging an @
        s = re.sub(r'\s*@\s*', '@', s)
        # Collapse whitespace around a dot only when it sits between
        # email-ish chars — repairs "gmail . com" / "john .\n doe" without
        # gluing ordinary sentences together.
        s = re.sub(r'(?<=[A-Za-z0-9])[ \t]*\.[ \t\r\n]*(?=[A-Za-z0-9])', '.', s)
        # Repair a single newline splitting the local part right before '@'
        s = re.sub(r'(?<=[A-Za-z0-9._%+\-])\s*\r?\n\s*(?=@)', '', s)

        return s

    def _sanitize_email(self, raw: Optional[str]) -> Optional[str]:
        """
        🧼 Return a single clean email address from messy input.

        Handles:
          - Markdown links:  [disp@x.com](mailto:real@x.com)  → real@x.com
                              (the mailto: target is authoritative)
          - mailto: prefixes, angle brackets <a@b.com>
          - Leading labels:  "address: a@b.com", "Email :** a@b.com"
          - Markdown emphasis (**, *, `), trailing punctuation
        Returns lowercased email, or None if nothing valid.
        """
        if not raw:
            return None
        s = str(raw)

        EMAIL = r'[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}'

        # 1️⃣ Prefer the mailto: target (the real hyperlink) if present
        m = re.search(r'mailto:\s*(' + EMAIL + r')', s, re.IGNORECASE)
        if m:
            candidate = m.group(1)
        else:
            # 2️⃣ Otherwise take the first bare email token anywhere in the string
            m = re.search(EMAIL, s)
            if not m:
                return None
            candidate = m.group(0)

        # Trim wrappers / markdown / stray punctuation
        candidate = candidate.strip(' \t\r\n<>()[]{}"\'.,;:|*`').lower()

        # Final strict validation
        if re.fullmatch(EMAIL, candidate):
            excluded = ('example.com', 'test.com', 'noreply@', 'support@',
                        'info@', 'donotreply@', 'no-reply@')
            if not any(x in candidate for x in excluded):
                return candidate
        return None

    def _sanitize_phone(self, raw: Optional[str]) -> Optional[str]:
        """
        🧼 Return a single clean phone token from messy input.

        Handles:
          - Markdown emphasis:  "** 87654321 **"          → 87654321
          - Field bleed:        "87654321 **Email :** a@b" → 87654321
          - Leading labels:     "Number: +65-12345678"     → +65-12345678
          - Trailing labels:    "+65 1234 5678 (HP)"       → +65 1234 5678
        Preserves a leading '+' and internal spaces/dashes; validates digit count.
        """
        if not raw:
            return None
        s = str(raw)

        # Strip markdown emphasis / backticks / underscores used as emphasis
        s = re.sub(r'[*`_]{1,3}', ' ', s)

        # Strip a leading label (incl. bare "Number"/"No"/"Contact"/"HP" etc.)
        s = re.sub(
            r'(?i)^[\s\-:.#)]*'
            r'(?:phone|tel(?:ephone)?|mobile|cell(?:\s*phone)?|contact|'
            r'hp|h/?p|hand\s*phone|whatsapp|portable|'
            r'no\.?\s*(?:tel|hp|phone|mobile)?|number|num)'
            r'\b[\s.:#)\-]*',
            '', s
        ).strip()

        # Cut off at the first field-bleed boundary (email/@, address, fax, etc.)
        s = re.split(
            r'(?i)\b(?:e-?mail|address|fax|name|nationality|nric|fin|'
            r'date\s*of\s*birth|dob|d\.o\.b)\b|@',
            s
        )[0]

        # Extract the phone-shaped token (optional +, digits, spaces, . - ( ))
        m = re.search(r'\+?\d[\d\s().\-]{5,18}\d', s)
        if not m:
            return None
        token = m.group(0).strip()

        # Drop a trailing parenthetical label: "(HP)", "(Mobile)", "(O)", "(R)"
        token = re.sub(r'\s*\([^)]*\)\s*$', '', token).strip()
        # Trim stray leading/trailing separators
        token = token.strip(' -.')

        digits = re.sub(r'\D', '', token)
        if not (7 <= len(digits) <= 15):
            return None
        if len(set(digits)) <= 2:               # 0000000, 1111111 etc.
            return None
        if digits.startswith(('0000', '1234567')):
            return None

        # Collapse runs of whitespace inside the token
        return re.sub(r'\s{2,}', ' ', token)

    def _find_contact_area(self, text: str) -> Optional[str]:
        """
        🔍 Find the contact information section
        Expanded search radius for SG-style resumes with separated sections!
        """
        # 🎯 Strategy 1: Look for explicit DOB/personal details labels
        # These are often in a separate "Personal Particulars" section in SG resumes
        personal_match = re.search(
            r'(?:PERSONAL\s+(?:PARTICULARS|DETAILS|INFORMATION|DATA)|BIODATA|BIO\s*DATA)',
            text, re.IGNORECASE
        )
        if personal_match:
            pos = personal_match.start()
            # Personal details section — grab a generous chunk after it
            return text[pos:min(len(text), pos + 1500)]

        # 🎯 Strategy 2: Look for email as anchor (expanded radius!)
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if email_match:
            pos = email_match.start()
            start = max(0, pos - 800)
            end = min(len(text), pos + 800)
            return text[start:end]

        # 🎯 Strategy 3: Look for phone as anchor
        phone_match = re.search(r'[\+]?\d[\d\s\-\.]{6,15}\d', text)
        if phone_match:
            pos = phone_match.start()
            start = max(0, pos - 800)
            end = min(len(text), pos + 800)
            return text[start:end]

        # 🎯 Strategy 4: Just use the first 3000 chars as fallback
        # Most resumes put contact info at the top anyway
        return text[:3000] if len(text) > 3000 else text

    def _extract_phone(self, text: str, email: Optional[str]) -> Optional[str]:
        """
        📱 ENHANCED Phone extraction with VERTICAL number support!
        Now handles those DRAMATIC vertical layouts!
        """
        # First, let's fix potential vertical phone numbers
        text = self._fix_vertical_phone_numbers(text)
        
        email_vicinity_text = ""
        if email:
            # Get text around the email (±300 characters for better coverage)
            email_index = text.find(email)
            if email_index != -1:
                start = max(0, email_index - 300)
                end = min(len(text), email_index + 300)
                email_vicinity_text = text[start:end]
                logger.debug(f"🔍 Searching for phone near email at index {email_index}")

        # Comprehensive phone patterns - GLOBAL EDITION!
        phone_patterns = [
            # Japanese mobile phones (090, 080, 070, 050)
            (r'(?:0[5789]0)[\-\s\(\)\.・－]*(\d{4})[\-\s\(\)\.・－]*(\d{4})', 'JP'),
            # Japanese with parentheses like (090) 1234-5678
            (r'\(0[5789]0\)\s*(\d{4})[\-\s・－]*(\d{4})', 'JP'),
            # Full-width Japanese
            (r'[（(]?[０0][５5-９9][０0][）)]?[\s\-－・]*([０-９0-9]{4})[\s\-－・]*([０-９0-9]{4})', 'JP'),
            # After headers (any language)
            (r'(?:電話|携帯電話|TEL|Tel|Phone|Mobile|Cell|携帯)[\s:：]*([（(]?[+\d\-\s\(\)\.・－０-９]{10,20}[）)]?)', 'HEADER'),
            # International format with +81
            (r'(?:\+81|８１)[\-\s\(\)]*(\d{1,2})[\-\s\(\)]*(\d{4})[\-\s\(\)]*(\d{4})', 'JP_INTL'),
            # Generic number sequences
            (r'(?:^|\s)([+\d\-\s\(\)\.]{10,20})(?:\s|$)', 'GENERIC'),
        ]

        # Try email vicinity first if we have an email
        search_areas = []
        if email_vicinity_text:
            search_areas.append(("email vicinity", email_vicinity_text))
        search_areas.append(("full text", text))
        
        for area_name, search_text in search_areas:
            logger.debug(f"🎯 Searching in {area_name}")
            
            for pattern, pattern_type in phone_patterns:
                matches = re.finditer(pattern, search_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    phone = self._extract_and_validate_phone(match, pattern_type)
                    if phone:
                        # Standardize the format
                        phone = self._standardize_phone_format(phone, pattern_type)
                        logger.info(f"✨ Found phone in {area_name}: {phone} (type: {pattern_type})")
                        return phone
        
        # EMERGENCY: Look for any 10+ digit number
        logger.warning("🚨 No standard phone found - trying desperate measures!")
        desperate_pattern = r'(?:^|\s)(\d[\d\-\s\(\)\.]{9,19})(?:\s|$)'
        matches = re.findall(desperate_pattern, text)
        for match in matches:
            digits_only = re.sub(r'\D', '', match)
            if 10 <= len(digits_only) <= 15:
                if not re.match(r'^(\d)\1+$', digits_only):  # Not all same digit
                    phone = self._standardize_phone_format(match, 'EMERGENCY')
                    logger.info(f"✨ Found phone (desperate mode): {phone}")
                    return phone
        
        return None

    def _fix_vertical_phone_numbers(self, text: str) -> str:
        """
        🔧 Fix vertical phone numbers like:
        (090)
        6074-6688
        """
        # Pattern for vertical phone numbers
        vertical_patterns = [
            # (090)\n1234-5678
            r'\((\d{3})\)\s*\n\s*(\d{4}[-\s]\d{4})',
            # (090)\n1234\n5678
            r'\((\d{3})\)\s*\n\s*(\d{4})\s*\n\s*(\d{4})',
            # 090\n1234\n5678
            r'(\d{3})\s*\n\s*(\d{4})\s*\n\s*(\d{4})',
            # Full-width versions
            r'[（(]([０-９0-9]{3})[）)]\s*\n\s*([０-９0-9]{4}[-\s－][０-９0-9]{4})',
        ]
        
        for pattern in vertical_patterns:
            def replacer(match):
                if len(match.groups()) == 2:
                    return f"({match.group(1)}) {match.group(2)}"
                elif len(match.groups()) == 3:
                    return f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
                return match.group(0)
            
            text = re.sub(pattern, replacer, text, flags=re.MULTILINE)
        
        return text

    def _standardize_phone_format(self, phone: str, pattern_type: str) -> str:
        """
        📱 Standardize phone to Japanese or International format
        Remove + from international format as requested!
        """
        # First normalize full-width to half-width
        phone = phone.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        phone = phone.translate(str.maketrans('（）', '()'))
        
        # Extract just digits
        digits = re.sub(r'\D', '', phone)
        
        if pattern_type in ['JP', 'JP_INTL'] or (pattern_type == 'GENERIC' and digits.startswith('0')):
            # Japanese format
            if digits.startswith('81'):  # Remove country code
                digits = '0' + digits[2:]
            
            if len(digits) == 11 and digits.startswith('0'):
                # Format as 090-1234-5678
                return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
            elif len(digits) == 10 and digits.startswith('0'):
                # Landline format 03-1234-5678
                if digits[1] in '3456789':  # Major cities
                    return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
                else:
                    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        
        # For international numbers, remove the + as requested
        if digits.startswith('81'):
            return f"81-{digits[2:4]}-{digits[4:8]}-{digits[8:]}"
        
        # Default format for other numbers
        if len(digits) >= 10:
            # Just group nicely
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11:
                return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
            else:
                return phone.strip()  # Return as is
        
        return phone.strip()

    def _extract_english_name(self, text: str) -> Optional[str]:
        """Extract English names only"""
        patterns = [
            r'(?:Name|Full\s*)?Name[\s:：]*([A-Za-z\s\'.-]{2,50})',
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$',  # Standard English name
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                name = match.strip()
                if self._is_valid_name_strict(name):
                    return name
        return None

    def _extract_japanese_name(self, text: str) -> Optional[str]:
        """Extract Japanese names only"""
        patterns = [
            r'氏\s*名[\s:：]*([一-龯]{1,4}[\s　]*[一-龯]{1,4})',
            r'([一-龯]{1,4}[\s　]*[一-龯]{1,4})[\s　]*[（(]([ぁ-ゖァ-ヾ\s　]+)[）)]',
            r'^[一-龯]{2,4}[\s　]+[一-龯]{2,4}$',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # If the pattern has groups, the result is a tuple
                name = (match[0] if isinstance(match, tuple) else match).strip()
                if self._is_valid_name_strict(name):
                    return name
        return None

    def _normalize_phone(self, phone: str) -> str:
        phone = phone.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        phone = phone.replace('－', '-').replace('・', '-')
        return re.sub(r'\s+', '', phone).strip()

    def _find_contact_information_area(self, text: str) -> Optional[str]:
        """
        🔍 Find the area where contact information is clustered
        Email, phone, and DOB are usually together!
        """
        # Find email position as anchor
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if email_match:
            email_pos = email_match.start()
            # Get 500 chars before and after email
            start = max(0, email_pos - 500)
            end = min(len(text), email_pos + 500)
            return text[start:end]
        
        # Find phone position as anchor
        phone_patterns = [
            r'(?:電話|携帯|TEL|Phone|Mobile)',
            r'\b0[5789]0[-\s]?\d{4}[-\s]?\d{4}\b',
            r'\([0-9]{3}\)',
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pos = match.start()
                start = max(0, pos - 500)
                end = min(len(text), pos + 500)
                return text[start:end]
        
        return None

    def _extract_name_english(self, text: str) -> Optional[str]:
        """
        👤 Extract names from ENGLISH resumes!
        Much simpler without Japanese complexity!
        """
        logger.debug("👑 English Name Detective activated!")
        
        text_lines = text.split('\n')
        
        # Words that are DEFINITELY NOT names
        not_names = {
            'resume', 'curriculum', 'vitae', 'cv', 'profile', 'summary',
            'objective', 'experience', 'education', 'skills', 'contact',
            'references', 'work', 'employment', 'professional', 'personal',
            'qualifications', 'achievements', 'certifications', 'projects',
            'engineer', 'developer', 'manager', 'designer', 'analyst',
            'coordinator', 'specialist', 'consultant', 'director', 'senior',
            'junior', 'lead', 'head', 'chief', 'vice', 'assistant',
            # Resume labels & section headers
            'email', 'address', 'phone', 'mobile', 'tel', 'details',
            'particulars', 'information', 'data', 'career', 'goal',
            'objectives', 'key', 'attributes', 'competencies', 'languages',
            'language', 'written', 'spoken', 'salary', 'expectation',
            'availability', 'commencement', 'immediate', 'singapore',
        }
        
        # Strategy 1: Look for explicit name headers
        name_patterns = [
            r'(?:Name|Full Name|Candidate Name)[\s:]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'^([A-Z][A-Z\s]+)$',  # All caps name on its own line
            r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',  # John M. Smith
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                candidate = match.strip()
                if self._is_valid_english_name(candidate, not_names):
                    logger.info(f"💎 Found name: {candidate}")
                    return candidate
        
        # Strategy 2: First few lines (names usually at top)
        for i, line in enumerate(text_lines[:10]):
            line = line.strip()
            
            # Skip short, long, or lines with special chars
            if len(line) < 5 or len(line) > 50:
                continue
            
            if any(char in line for char in ['@', 'http', '://', '.com', '|']):
                continue
            
            # Must be title case and 2-4 words
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(word[0].isupper() and word[1:].islower() for word in words if len(word) > 1):
                    if self._is_valid_english_name(line, not_names):
                        logger.info(f"💎 Found name at line {i}: {line}")
                        return line
        
        # Strategy 3: Look near email
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if email_match:
            email_pos = email_match.start()
            context = text[max(0, email_pos-300):email_pos]
            
            name_candidates = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', context)
            for candidate in reversed(name_candidates):
                if self._is_valid_english_name(candidate, not_names):
                    logger.info(f"💎 Found name near email: {candidate}")
                    return candidate
        
        return None

    def _is_valid_english_name(self, name: str, not_names: set) -> bool:
        """
        ✅ Validate if text is really an English name
        """
        if not name or len(name) < 3 or len(name) > 50:
            return False
        
        name_lower = name.lower()
        
        # Check against blacklist
        if name_lower in not_names:
            return False
        
        if any(word in not_names for word in name_lower.split()):
            return False
        
        # Must be only letters, spaces, hyphens, apostrophes
        if not re.match(r"^[A-Za-z\s\-'\.]+$", name):
            return False
        
        # Should have proper capitalization
        words = name.split()
        for word in words:
            if len(word) > 1:
                # Should start with capital (except Jr, Sr, etc.)
                if word not in ['Jr', 'Sr', 'II', 'III', 'IV'] and not word[0].isupper():
                    return False
        
        # Should not be all caps (unless 2-3 letters)
        if name.isupper() and len(name) > 5:
            return False

        return True

    def _is_valid_extracted_name(self, name: str, email: str = None) -> bool:
        """
        Validate a name AFTER extraction to catch garbage like email usernames,
        email aliases, resume labels, etc.
        Returns False if the name is clearly not a real person name.
        """
        if not name or len(name.strip()) < 2:
            return False

        name = name.strip()

        # Contains digits → likely email username like "Sanker22"
        if re.search(r'\d', name):
            return False

        # Single word → likely username or label, not a full name
        if len(name.split()) < 2:
            return False

        # Contains @ or looks like email
        if '@' in name or '.com' in name.lower():
            return False

        # Common resume labels that AI might return as names
        label_words = {
            'email', 'address', 'phone', 'mobile', 'tel', 'personal',
            'details', 'particulars', 'data', 'contact', 'information',
            'objective', 'career', 'goal', 'summary', 'profile',
            'curriculum', 'vitae', 'resume', 'nil', 'n/a', 'none',
            'not', 'available', 'confidential', 'private',
        }
        name_words_lower = {w.lower() for w in name.split()}
        if name_words_lower & label_words:
            return False

        # Check if name was derived from email address
        # e.g., "Bee Happy" from "bee.happy.house@gmail.com"
        if email:
            email_local = email.split('@')[0].lower() if '@' in email else ''
            if email_local:
                # Split email local part by common separators
                email_parts = set(re.split(r'[._\-+]', email_local))
                # If ALL name words appear in email local part, name is from email
                if name_words_lower and name_words_lower.issubset(email_parts):
                    return False

        return True

    def _is_definitely_a_name(self, text: str, not_names: set) -> bool:
        """
        🔍 ULTRA STRICT validation - is this REALLY a name?
        """
        if not text or len(text) < 2:
            return False
        
        text_lower = text.lower()
        
        # Check against our NOT names list
        if text_lower in not_names:
            return False
        
        # Check each word in multi-word names
        words = text.split()
        for word in words:
            if word.lower() in not_names:
                return False
        
        # Additional validation for English names
        if re.match(r'^[A-Za-z\s\-\.\']+$', text):
            # Should have at least one capital letter
            if not any(c.isupper() for c in text):
                return False
            # Shouldn't be all caps (unless 2-3 chars like "JR")
            if text.isupper() and len(text) > 3:
                return False
            # Each word should start with capital
            for word in words:
                if len(word) > 2 and not word[0].isupper():
                    return False
        
        # Additional validation for Japanese names
        if re.search(r'[一-龯ァ-ヾ]', text):
            # Should not contain numbers or special chars
            if re.search(r'[\d@#$%^&*()_+=\[\]{};:"\\|,.<>/?]', text):
                return False
        
        return True

    def _looks_like_name_part(self, word: str) -> bool:
        """Check if a single word looks like part of a name"""
        # Too short or too long
        if len(word) < 2 or len(word) > 20:
            return False
        # Should start with capital
        if not word[0].isupper():
            return False
        # Shouldn't be all caps (unless short like "JR", "III")
        if word.isupper() and len(word) > 3:
            return False
        # Common name suffixes are OK
        if word.upper() in ['JR', 'SR', 'III', 'II', 'IV']:
            return True
        # Should be mostly letters
        if not re.match(r'^[A-Za-z\'\-\.]+$', word):
            return False
        return True

    def _is_valid_name_strict(self, name: str) -> bool:
        """
        🔍 ULTRA STRICT name validation - NO job titles allowed!
        """
        if not name or len(name) < 2 or len(name) > 50:
            return False
        
        not_names = {
            'profile', 'summary', 'objective', 'experience', 'education',
            'skills', 'references', 'career', 'professional', 'personal',
            'address', 'location', 'portfolio', 'contact', 'information',
            'programmer', 'analyst', 'developer', 'engineer', 'manager',
            'coordinator', 'specialist', 'consultant', 'designer', 'architect',
            'administrator', 'executive', 'director', 'supervisor', 'lead',
            'senior', 'junior', 'intern', 'trainee', 'associate', 'assistant',
            'officer', 'technician', 'expert', 'advisor', 'coach',
            '履歴書', '職務経歴書', '経歴', '学歴', '職歴', 'スキル',
            'プロフィール', 'サマリー', '概要', '自己紹介',
            '日本語能力試験', '試験', 'テスト', '検定'
        }
        
        name_lower = name.lower()
        if name_lower in not_names or any(word in not_names for word in name_lower.split()):
            return False
        
        if name.isupper() and len(name) > 4:
            return False
        
        if any(char in name for char in ['@', 'http', 'www', '/', '\\', '|', '{', '}', '[', ']']):
            return False
        
        if re.match(r'^[A-Za-z\s\-\.\']+$', name):
            if not any(c.isupper() for c in name) or (name.islower() and len(name) > 3):
                return False
        
        return True

    def _clean_name(self, name: str) -> str:
        return re.sub(r'[（(].*?[）)]', '', name).strip()

    def _extract_via_shared_pipeline(self, file_path: str) -> Optional[str]:
        """
        Run the EXACT same per-file extraction branching as
        extract_raw_text.py::extract_raw_text() via the shared HybridExtractor.

        DOCX            → HybridExtractor.extract_docx()
        PDF needs_ocr   → HybridExtractor.extract_hybrid()   (Marker + OCR concat)
        PDF text-only   → HybridExtractor.extract_text_only() (Marker only)

        Returns the raw (un-normalized) text, or None.
        """
        file_name = os.path.basename(file_path)
        is_docx = file_name.lower().endswith('.docx')

        if is_docx:
            logger.info(f"📄 DOCX → shared python-docx/XML extraction for {file_name}")
            return self.hybrid_extractor.extract_docx(file_path)

        # PDF: inspect, then branch identically to extract_raw_text.py
        pdf_info = analyze_pdf_type(file_path)
        logger.info(
            f"🔍 PDF Inspector: {file_name} → type={pdf_info['pdf_type']}, "
            f"images={pdf_info['image_count']} "
            f"(deep={pdf_info.get('image_count_deep', 0)}), "
            f"text={pdf_info['text_length']} chars "
            f"({pdf_info.get('chars_per_page', 0):.0f}/page over "
            f"{pdf_info.get('page_count', 0)} pages), "
            f"needs_ocr={pdf_info['needs_ocr']}"
        )
        if pdf_info.get('detection_reason'):
            logger.info(f"   └─ reason: {pdf_info['detection_reason']}")

        if pdf_info['needs_ocr']:
            logger.info(
                f"🔀 needs_ocr=True ({pdf_info['image_count']} images) "
                f"- shared HYBRID extraction for {file_name}"
            )
            return self.hybrid_extractor.extract_hybrid(file_path)

        logger.info(f"📄 Text-only PDF - shared Marker extraction for {file_name}")
        return self.hybrid_extractor.extract_text_only(file_path)

    def get_text_from_file(self, file_path: str) -> Optional[str]:
        """📄 Enhanced text extraction with PDF inspection - uses OCR only when images detected"""
        file_name = os.path.basename(file_path)
        if file_name.startswith('~$'):
            return None

        # ── SHARED RAW-TEXT PIPELINE ──────────────────────────────────────────
        # Delegate to the SAME HybridExtractor that extract_raw_text.py uses, so
        # the text fed to the AI/regex extractor is identical to what is written
        # into raw_text_output/*.xlsx. _clean_text() is applied afterwards as the
        # ONLY intentional difference — it is downstream normalization the
        # regex/AI layer requires; the raw_text_output dump is intentionally
        # left un-normalized for human inspection.
        if self.hybrid_extractor is not None:
            try:
                raw = self._extract_via_shared_pipeline(file_path)
                return self._clean_text(raw) if raw else None
            except Exception as e:
                logger.warning(
                    f"⚠️ Shared pipeline failed for {file_name} ({e}) — using legacy path"
                )

        text = ""
        ocr_min_threshold = getattr(config, 'OCR_MIN_TEXT_THRESHOLD', 100)
        hybrid_merge_mode = getattr(config, 'HYBRID_MERGE_MODE', 'combine')

        try:
            if file_path.lower().endswith('.pdf'):
                # 🔍 STEP 1: Inspect PDF to determine extraction strategy
                pdf_info = analyze_pdf_type(file_path)
                logger.info(
                    f"🔍 PDF Inspector: {file_name} → type={pdf_info['pdf_type']}, "
                    f"images={pdf_info['image_count']} "
                    f"(deep={pdf_info.get('image_count_deep', 0)}), "
                    f"text={pdf_info['text_length']} chars "
                    f"({pdf_info.get('chars_per_page', 0):.0f}/page over "
                    f"{pdf_info.get('page_count', 0)} pages), "
                    f"needs_ocr={pdf_info['needs_ocr']}"
                )
                if pdf_info.get('detection_reason'):
                    logger.info(f"   └─ reason: {pdf_info['detection_reason']}")

                # 📄 CASE 1: Text-only PDF - no OCR needed
                if not pdf_info['needs_ocr']:
                    logger.info(f"📄 Text-only PDF detected - skipping OCR for {file_name}")

                    # Try Marker first (best quality)
                    if self.use_marker and self.marker_extractor and self.marker_extractor.available:
                        logger.info(f"📄 Using Marker extraction for {file_name}")
                        text = self.marker_extractor.extract_pdf(file_path, use_ocr_fallback=False)

                        if text and len(text.strip()) >= ocr_min_threshold:
                            logger.info(f"✅ Marker successfully extracted {len(text)} chars")
                            stats = self.marker_extractor.get_extraction_stats(text)
                            logger.info(f"   📊 Sections: {stats.get('sections_found', 0)}, Tables: {stats.get('table_rows', 0)} rows")
                            return self._clean_text(text)

                    # Fallback to pdfplumber
                    if not text or len(text.strip()) < ocr_min_threshold:
                        logger.info(f"📄 Using pdfplumber for {file_name}")
                        with pdfplumber.open(file_path) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"

                        if text and len(text.strip()) >= 50:
                            logger.info(f"✅ pdfplumber extracted {len(text)} chars")
                            return self._clean_text(text)

                # 📸 CASE 2: PDF has images - use OCR (hybrid extraction)
                else:
                    logger.info(f"🔀 Images detected ({pdf_info['image_count']}) - using OCR for {file_name}")

                    if self.use_ocr and self.ocr_available:
                        # Step 1: Get text from Marker/pdfplumber (selectable text)
                        extracted_text = self._extract_selectable_text(file_path)

                        # Step 2: Get text from OCR (image-based text)
                        logger.info(f"📸 Running OCR to capture image-based text...")
                        ocr_text = self._extract_with_ocr(file_path)

                        # Step 3: Merge results
                        if extracted_text and ocr_text:
                            if hybrid_merge_mode == "combine":
                                text = self._merge_extracted_texts(extracted_text, ocr_text, file_name)
                            else:  # "longest" mode
                                text = extracted_text if len(extracted_text) >= len(ocr_text) else ocr_text
                                logger.info(f"📊 Using {'extracted' if len(extracted_text) >= len(ocr_text) else 'OCR'} text (longer)")
                        elif extracted_text:
                            text = extracted_text
                            logger.info(f"📄 Using extracted text only ({len(text)} chars)")
                        elif ocr_text:
                            text = ocr_text
                            logger.info(f"📸 Using OCR text only ({len(text)} chars)")

                        if text:
                            return self._clean_text(text)

                    # Fallback if OCR not available but images detected
                    else:
                        logger.warning(f"⚠️ Images detected but OCR not available - using text extraction only")
                        if self.use_marker and self.marker_extractor and self.marker_extractor.available:
                            text = self.marker_extractor.extract_pdf(file_path, use_ocr_fallback=False)
                        if not text or len(text.strip()) < ocr_min_threshold:
                            with pdfplumber.open(file_path) as pdf:
                                for page in pdf.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text += page_text + "\n"

            elif file_path.lower().endswith('.docx'):
                # Try python-docx first
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])

                # Also extract from tables
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            text += "\n" + cell.text

                # Fallback to docx2txt if python-docx returns empty (handles text boxes, etc.)
                if not text or len(text.strip()) < 50:
                    logger.info(f"📄 python-docx returned minimal text, trying docx2txt for {file_name}")
                    try:
                        text = docx2txt.process(file_path)
                        if text:
                            logger.info(f"✅ docx2txt extracted {len(text)} chars")
                    except Exception as e:
                        logger.warning(f"⚠️ docx2txt failed: {e}")

            return self._clean_text(text)

        except Exception as e:
            logger.error(f"❌ Extraction failed for {file_name}: {e}")
            return None

    def _extract_selectable_text(self, file_path: str) -> Optional[str]:
        """Extract selectable/highlightable text from PDF using Marker, pdfplumber, or DocumentParser"""
        file_name = os.path.basename(file_path)
        text = ""

        # Try Marker first
        if self.use_marker and self.marker_extractor and self.marker_extractor.available:
            text = self.marker_extractor.extract_pdf(file_path, use_ocr_fallback=False)
            if text and len(text.strip()) >= 50:
                logger.info(f"✅ Marker extracted {len(text)} chars of selectable text")
                return text

        # Fallback to pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if text and len(text.strip()) >= 50:
                logger.info(f"✅ pdfplumber extracted {len(text)} chars of selectable text")
                return text
        except Exception as e:
            logger.warning(f"⚠️ pdfplumber failed: {e}")

        # Fallback to DocumentParser (handles vector/outline text PDFs with OCR)
        if self.document_parser and (not text or len(text.strip()) < 50):
            logger.info(f"📄 Trying DocumentParser for {file_name} (may use OCR)...")
            try:
                doc_text, method = self.document_parser.parse(file_path)
                if doc_text and len(doc_text.strip()) >= 50:
                    logger.info(f"✅ DocumentParser extracted {len(doc_text)} chars using {method}")
                    return doc_text
            except Exception as e:
                logger.warning(f"⚠️ DocumentParser failed: {e}")

        return text if text else None

    def _merge_extracted_texts(self, extracted_text: str, ocr_text: str, file_name: str) -> str:
        """
        🔀 Concatenate text extraction + OCR results
        Keeps ALL text from both sources to ensure nothing is missed
        """
        logger.info(f"🔀 Concatenating extracted text ({len(extracted_text)} chars) + OCR ({len(ocr_text)} chars)")

        # Concatenate both - don't filter anything out
        merged = extracted_text.strip() + "\n\n" + ocr_text.strip()
        logger.info(f"✅ Concatenated result: {len(merged)} total chars")
        return merged

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts (0.0 to 1.0)"""
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        if text1 == text2:
            return 1.0

        # Check if one contains the other
        if text1 in text2 or text2 in text1:
            return 0.9

        # Use difflib for more accurate comparison
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1, text2).ratio()
        except:
            # Fallback: simple word overlap
            words1 = set(text1.split())
            words2 = set(text2.split())
            if not words1 or not words2:
                return 0.0
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0

    def _extract_data_from_text(self, text: str) -> Tuple[Dict, bool]:
        """
        🎯 Extract data with proper field mapping!
        Returns formatted dict + whether AI was used
        """
        from utils import standardize_phone_number, standardize_date
        
        final_results = {}
        ai_assisted = False

        # ── STEP 1: Regex for Email and Phone (always runs first) ──────────────
        # Phone and Email are reliably captured by regex; AI is unnecessary for them.
        regex_data = self._extract_with_mega_regex(text)
        regex_email  = regex_data.get('email')
        regex_phone  = regex_data.get('phone')
        if regex_email:
            final_results['email'] = regex_email
        if regex_phone:
            final_results['phone'] = regex_phone

        # ── STEP 2: AI for Name + structured fields (experience, skills, etc.) ──
        if self.ai_enabled:
            try:
                logger.info("🤖 Attempting AI extraction...")
                header_data = self.ai_extractor.extract_header_fields(text)
                deep_data   = self.ai_extractor.extract_deep_fields(text)
                ai_results  = {**header_data, **deep_data}

                if ai_results:
                    for key, val in ai_results.items():
                        # Keep regex email/phone; let AI fill everything else
                        if key in ('email', 'phone') and final_results.get(key):
                            continue
                        final_results[key] = val
                    ai_assisted = True
                    logger.info(f"✅ AI extracted: {list(ai_results.keys())}")
            except Exception as e:
                logger.warning(f"⚠️ AI extraction failed: {e}")

        # ── STEP 3: Regex fills any remaining gaps ──────────────────────────────
        for field in ('name', 'email', 'phone'):
            if not final_results.get(field) and regex_data.get(field):
                final_results[field] = regex_data[field]

        # ── STEP 4: Final sanitize — guarantees clean output regardless of
        #            whether the value came from regex OR the AI model ─────────
        # An email MUST be a real address: local@domain.tld. _sanitize_email
        # returns None for anything that isn't (no '@', a name, a LinkedIn URL,
        # "N/A", "see resume", AI noise). When it fails we DROP the value — a
        # string without an '@' is NOT an email, so it never belongs here.
        if final_results.get('email'):
            cleaned_email = self._sanitize_email(final_results['email'])
            if not cleaned_email:
                logger.warning(
                    "🚫 Rejected non-email value for Email field: "
                    f"{str(final_results['email'])[:60]!r}"
                )
            final_results['email'] = cleaned_email
        if final_results.get('phone'):
            final_results['phone'] = (
                self._sanitize_phone(final_results['phone']) or final_results['phone']
            )

        # Record per-field extraction methods for reporting (reflect the FINAL
        # value: if email was rejected above, it is 'Not found', not 'Regex').
        phone_method = 'Regex' if regex_phone else ('AI' if final_results.get('phone') else 'Not found')
        if not final_results.get('email'):
            email_method = 'Not found'
        else:
            email_method = 'Regex' if regex_email else 'AI'

        # 🎯 FORMAT THE OUTPUT PROPERLY!
        # Pass raw text for fallback extraction if structured parsing fails
        formatted_data = {
            "Name": final_results.get("name"),
            "Email": final_results.get("email"),
            "Phone": final_results.get("phone"),
            "Phone_Method": phone_method,
            "Email_Method": email_method,
            # 💎 THE CRITICAL FIX: Convert lists to JSON strings for CSV export!
            # 🆘 Now with RAW TEXT FALLBACK when structured parsing fails!
            "Skills": self._format_skills_for_export(final_results, text),
            "Working_Experience": self._format_experience_for_export(final_results, text),

            "Location": final_results.get("location"),
            "School_University": self._format_education_for_export(final_results, text),
            "Summary": final_results.get("summary", ""),

            # 🗂️ Raw structured data — preserved for JSON export formatting
            "_raw_experience": final_results.get("working_experience", []),
            "_raw_education": final_results.get("education", []),
            "_raw_hard_skills": final_results.get("hard_skills", []),
            "_raw_soft_skills": final_results.get("soft_skills", []),
            "_raw_languages": final_results.get("languages", []),
        }

        return formatted_data, ai_assisted

    def _extract_section_raw(self, text: str, section_keywords: List[str]) -> Optional[str]:
        """
        🆘 EMERGENCY FALLBACK: Extract raw text content from a section.
        Used when structured parsing fails but we still want to capture content.
        """
        import re

        if not text:
            return None

        # Common section terminators (next section headers)
        terminators = [
            'EXPERIENCE', 'WORK EXPERIENCE', 'EMPLOYMENT', 'EDUCATION', 'SKILLS',
            'TECHNICAL SKILLS', 'CERTIFICATIONS', 'AWARDS', 'PROJECTS', 'REFERENCES',
            'ACHIEVEMENTS', 'PUBLICATIONS', 'LANGUAGES', 'HOBBIES', 'INTERESTS',
            'SUMMARY', 'OBJECTIVE', 'PROFILE', 'QUALIFICATIONS'
        ]

        for keyword in section_keywords:
            # Create pattern to find section and capture until next section
            terminator_pattern = '|'.join([t for t in terminators if t != keyword])
            pattern = rf'(?:^|\n)\s*{re.escape(keyword)}[S]?\s*[:\n]\s*(.*?)(?=\n\s*(?:{terminator_pattern})\s*(?:[:|\n]|$)|$)'

            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                # Clean up the content
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                content = re.sub(r'[•●○▪■►]', '|', content)  # Replace bullets with |

                if len(content) > 50:
                    return content

        return None

    def _format_skills_for_export(self, data: Dict, raw_text: str = "") -> Optional[str]:
        """
        🎨 Format skills for CSV export
        Combines hard + soft skills into readable format
        WITH FALLBACK: If no structured skills found, extract from raw text!
        """
        import json

        hard_skills = data.get("hard_skills", [])
        soft_skills = data.get("soft_skills", [])

        if hard_skills or soft_skills:
            # Option 2: Readable list format (better for humans)
            all_skills = []
            if hard_skills:
                all_skills.extend(hard_skills)
            if soft_skills:
                all_skills.extend(soft_skills)
            return " | ".join(all_skills)

        # 🆘 FALLBACK: Extract skills section as raw text
        if raw_text:
            skills_fallback = self._extract_section_raw(raw_text,
                ['SKILLS', 'TECHNICAL SKILLS', 'KEY SKILLS', 'CORE COMPETENCIES',
                 'PROFICIENCIES', 'EXPERTISE', 'CAPABILITIES', 'QUALIFICATIONS'])
            if skills_fallback:
                return skills_fallback

        return None

    def _format_experience_for_export(self, data: Dict, raw_text: str = "") -> Optional[str]:
        """
        💼 Format work experience for CSV export
        Creates readable summary of jobs
        WITH FALLBACK: If no structured experience found, extract from raw text!
        """
        import json

        experience = data.get("working_experience", [])

        if experience and isinstance(experience, list) and len(experience) > 0:
            # Option 2: Readable summary format
            job_summaries = []
            for job in experience:
                if isinstance(job, dict):
                    company = job.get('company', 'Unknown')
                    role = job.get('role', 'N/A')
                    dates = job.get('dates', 'N/A')
                    description = job.get('description', '').strip()

                    # 💅 Build the job entry with description included!
                    # Skip placeholder descriptions that add no value
                    skip_descriptions = {
                        '', 'description not available', 'n/a', 'none',
                        'see description', 'no description'
                    }

                    if description and description.lower() not in skip_descriptions:
                        # Include the full job description after the header
                        job_summaries.append(
                            f"{company} - {role} ({dates}): {description}"
                        )
                    else:
                        # No valid description — just header info
                        job_summaries.append(f"{company} - {role} ({dates})")

            if job_summaries:
                return " || ".join(job_summaries)

        # 🆘 FALLBACK: Extract experience section as raw text
        if raw_text:
            exp_fallback = self._extract_section_raw(raw_text,
                ['EXPERIENCE', 'WORK EXPERIENCE', 'EMPLOYMENT', 'EMPLOYMENT HISTORY',
                 'PROFESSIONAL EXPERIENCE', 'CAREER HISTORY', 'WORK HISTORY'])
            if exp_fallback:
                return exp_fallback

        return None

    def _format_education_for_export(self, data: Dict, raw_text: str = "") -> Optional[str]:
        """
        🎓 Format education for CSV export
        Creates readable summary of education entries
        WITH FALLBACK: If no structured education found, extract from raw text!
        """
        import json

        education = data.get("education", [])

        if education and isinstance(education, list) and len(education) > 0:
            # Option 2: Readable summary format
            edu_summaries = []
            for edu in education:
                if isinstance(edu, dict):
                    institution = edu.get('institution', 'Unknown')
                    degree = edu.get('degree', 'N/A')
                    dates = edu.get('dates', 'N/A')
                    description = edu.get('description', '').strip()

                    if description and description.lower() not in {'', 'n/a', 'none'}:
                        edu_summaries.append(
                            f"{degree} from {institution} ({dates}): {description}"
                        )
                    else:
                        edu_summaries.append(f"{degree} from {institution} ({dates})")
            if edu_summaries:
                return " || ".join(edu_summaries)

        # 🆘 FALLBACK: Extract education section as raw text
        if raw_text:
            edu_fallback = self._extract_section_raw(raw_text,
                ['EDUCATION', 'ACADEMIC', 'QUALIFICATIONS', 'ACADEMIC BACKGROUND',
                 'EDUCATIONAL BACKGROUND', 'SCHOOLING', 'DEGREES'])
            if edu_fallback:
                return edu_fallback

        return None

    def _extract_name_from_folder(self, folder_path: str) -> Optional[str]:
        """
        Extract name from folder name like "43228_Suhana Binte Salim" or "108_Mr ATWAL Prateek"
        """
        folder_name = os.path.basename(folder_path)

        # Remove the numeric ID prefix (e.g., "43228_")
        name_part = re.sub(r'^\d+_', '', folder_name).strip()

        if not name_part or len(name_part) < 2:
            return None

        # Remove titles
        name_part = re.sub(r'^(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+', '', name_part, flags=re.IGNORECASE).strip()

        # Remove D/O, S/O markers (common in Singapore)
        # But keep them as part of name for now — they're cultural identifiers

        # Filter out non-name words that sometimes appear in folder names
        skip_words = {'temp', 'school', 'admin', 'ok', 'raw', 'kiv', 'sent'}
        cleaned_words = []
        for word in name_part.split():
            if word.lower().strip('()') not in skip_words:
                cleaned_words.append(word)
        name_part = ' '.join(cleaned_words).strip()

        if name_part and len(name_part) > 2:
            logger.info(f"📛 Extracted name from folder: {name_part}")
            return name_part

        return None

    def process_candidate_folder(self, folder_path: str) -> Dict:
        """
        ✨ Processes all resume files - ENGLISH ONLY!
        """
        logger.info(f"🎤 Processing: {os.path.basename(folder_path)}")
        
        # Find resume files
        resume_files = []
        try:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.pdf', '.docx')) and not filename.startswith('~$'):
                    resume_files.append(os.path.join(folder_path, filename))
                    logger.info(f"   ✅ Found: {filename}")
        except Exception as e:
            logger.error(f"❌ Error reading folder: {e}")
            return {}
        
        if not resume_files:
            logger.warning(f"🤷‍♀️ No resume files found")
            return {}
        
        # Initialize result
        # ✨ FIXED: Initialize result with ALL fields (including previously missing ones!)
        result = {
            "ID": None,
            "Name": None,
            "Email": None,
            "Phone": None,
            "Skills": None,
            "Working_Experience": None,
            "Location": None,
            "School_University": None,

            # 🆕 NEW FIELDS - Previously extracted but not saved!
            "Summary": None,
            "Certifications": None,
            "Languages": None,
            "Projects": None,
            "Achievements": None,
            "References": None,
            "Hobbies": None,
            
            "Language": "English",  # Resume language
            "Extraction_Status": "Failed",
            "Notes": "",
            "AI_Assisted": False,
            "Filenames_Processed": ", ".join([os.path.basename(f) for f in resume_files])
        }
        
        # Extract ID from folder name
        folder_name = os.path.basename(folder_path)
        id_match = re.match(r'^(\d+)', folder_name)
        if id_match:
            result["ID"] = int(id_match.group(1))
        
        # Combine text from all files
        combined_text = ""
        for file_path in resume_files:
            text = self.get_text_from_file(file_path)
            if text:
                combined_text += text + "\n\n"
        
        
        # Extract data
        if combined_text and len(combined_text.strip()) > 50:
            extracted_data, ai_used = self._extract_data_from_text(combined_text)
            
            # 🔍 DEBUG: See what we actually got
            logger.info("📊 Extraction results:")
            logger.info(f"   Name: {extracted_data.get('Name', 'NOT FOUND')}")
            logger.info(f"   Email: {extracted_data.get('Email', 'NOT FOUND')}")
            logger.info(f"   Skills: {extracted_data.get('Skills', 'NOT FOUND')[:100] if extracted_data.get('Skills') else 'NOT FOUND'}...")
            logger.info(f"   Experience: {extracted_data.get('Working_Experience', 'NOT FOUND')[:100] if extracted_data.get('Working_Experience') else 'NOT FOUND'}...")
            
            # 🆕 NEW: Debug for additional fields
            logger.info(f"   Summary: {extracted_data.get('Summary', 'NOT FOUND')[:100] if extracted_data.get('Summary') else 'NOT FOUND'}...")
            logger.info(f"   Certifications: {extracted_data.get('Certifications', 'NOT FOUND')}")
            logger.info(f"   Projects: {extracted_data.get('Projects', 'NOT FOUND')}")
            
            if ai_used:
                result["AI_Assisted"] = True
            

            # ✨ FIXED: Update result with ALL fields (including new ones!)
            # Basic contact fields
            for field in ["Name", "Email", "Phone", "Location"]:
                if extracted_data.get(field):
                    result[field] = extracted_data[field]
                    logger.debug(f"✅ Updated {field}: {str(extracted_data[field])[:100]}...")

            # Carry forward per-field extraction method tags
            result["Phone_Method"] = extracted_data.get("Phone_Method", "Regex")
            result["Email_Method"] = extracted_data.get("Email_Method", "Regex")

            # Validate extracted name — catch garbage like email usernames or labels
            extracted_email = result.get("Email", "")
            if result.get("Name") and not self._is_valid_extracted_name(result["Name"], extracted_email):
                logger.warning(f"⚠️ Extracted name looks invalid: '{result['Name']}' — trying fallbacks")
                result["Name"] = None

            # Fallback chain for missing/invalid name
            if not result.get("Name"):
                # Try folder name first
                folder_name_candidate = self._extract_name_from_folder(folder_path)

                # If folder name is multi-word (2+), use it directly
                if folder_name_candidate and len(folder_name_candidate.split()) >= 2:
                    result["Name"] = folder_name_candidate
                    logger.info(f"📛 Name from folder: {folder_name_candidate}")
                else:
                    # Folder name is single-word or unavailable — try filenames
                    for file_path in resume_files:
                        fn_candidate = self._extract_name_from_filename(os.path.basename(file_path))
                        if fn_candidate and self._is_valid_extracted_name(fn_candidate):
                            result["Name"] = fn_candidate
                            logger.info(f"📛 Name from filename: {fn_candidate}")
                            break

                    # If filename didn't work either, use the single-word folder name
                    if not result.get("Name") and folder_name_candidate:
                        result["Name"] = folder_name_candidate
                        logger.info(f"📛 Name from folder (partial): {folder_name_candidate}")
            
            # 🆕 NEW: Update skills (already formatted as string)
            if extracted_data.get('Skills'):
                result['Skills'] = extracted_data['Skills']
                logger.debug(f"✅ Updated Skills: {str(extracted_data['Skills'])[:100]}...")
            
            # 🆕 NEW: Update experience (already formatted as string)
            if extracted_data.get('Working_Experience'):
                result['Working_Experience'] = extracted_data['Working_Experience']
                logger.debug(f"✅ Updated Working_Experience: {str(extracted_data['Working_Experience'])[:100]}...")
            
            # 🆕 NEW: Update education (already formatted as string)
            if extracted_data.get('School_University'):
                result['School_University'] = extracted_data['School_University']
                logger.debug(f"✅ Updated School_University: {str(extracted_data['School_University'])[:100]}...")
            
            # 🆕 NEW: Update additional text fields
            if extracted_data.get('Summary'):
                result['Summary'] = extracted_data['Summary']
                logger.debug(f"✅ Updated Summary: {str(extracted_data['Summary'])[:100]}...")

            # 🗂️ Store raw structured data for JSON export
            result['_raw_experience'] = extracted_data.get('_raw_experience', [])
            result['_raw_education'] = extracted_data.get('_raw_education', [])
            result['_raw_hard_skills'] = extracted_data.get('_raw_hard_skills', [])
            result['_raw_soft_skills'] = extracted_data.get('_raw_soft_skills', [])
            result['_raw_languages'] = extracted_data.get('_raw_languages', [])
            
            # 🆕 NEW: Update list fields (convert to readable format)
            if extracted_data.get('Certifications'):
                result['Certifications'] = self._format_list_field(extracted_data['Certifications'])
                logger.debug(f"✅ Updated Certifications: {result['Certifications']}")
            
            if extracted_data.get('Languages'):
                result['Languages'] = self._format_list_field(extracted_data['Languages'])
                logger.debug(f"✅ Updated Languages: {result['Languages']}")
            
            if extracted_data.get('Projects'):
                result['Projects'] = self._format_list_field(extracted_data['Projects'])
                logger.debug(f"✅ Updated Projects: {result['Projects']}")
            
            if extracted_data.get('Achievements'):
                result['Achievements'] = self._format_list_field(extracted_data['Achievements'])
                logger.debug(f"✅ Updated Achievements: {result['Achievements']}")
            
            if extracted_data.get('References'):
                result['References'] = self._format_list_field(extracted_data['References'])
                logger.debug(f"✅ Updated References: {result['References']}")
            
            if extracted_data.get('Hobbies'):
                result['Hobbies'] = self._format_list_field(extracted_data['Hobbies'])
                logger.debug(f"✅ Updated Hobbies: {result['Hobbies']}")

            # Set status after ALL fields are updated
            self._set_extraction_status(result, combined_text)

            logger.info(f"✅ Completed: {result['Extraction_Status']}")
            return result
        else:
            # No text extracted or text too short
            logger.warning(f"⚠️ Could not extract text from files (got {len(combined_text.strip()) if combined_text else 0} chars)")
            result["Notes"] += "⚠️ Text extraction failed or returned minimal content. "
            result["Extraction_Status"] = "Failed"
            return result
    
    def _format_list_field(self, field_data) -> Optional[str]:
        """
        🆕 NEW: Format list fields for CSV export
        Handles both lists and strings intelligently
        """
        if not field_data:
            return None
        
        # If already a string, return it
        if isinstance(field_data, str):
            return field_data
        
        # If it's a list
        if isinstance(field_data, list):
            # If list is empty, return None
            if not field_data:
                return None
            
            # If list contains dictionaries (structured data)
            if field_data and isinstance(field_data[0], dict):
                formatted_items = []
                for item in field_data:
                    # Extract key fields from dict
                    parts = []
                    for key in ['name', 'title', 'description', 'institution', 'organization', 'date', 'dates']:
                        if key in item and item[key]:
                            parts.append(str(item[key]))
                    if parts:
                        formatted_items.append(" - ".join(parts))
                return " | ".join(formatted_items) if formatted_items else None
            
            # If list contains simple strings
            else:
                return " | ".join([str(item) for item in field_data if item])
        
        # Fallback: convert to string
        return str(field_data)

    def _extract_name_from_filename(self, file_name: str) -> Optional[str]:
        filename_without_ext = os.path.splitext(file_name)[0]

        # Standard space-separated patterns
        filename_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:[\s*-_])',
            r'^(?:CV[-_])?([A-Z][a-z]+\s+[A-Z]+)(?:\s+\d|$)',
            r'^([A-Z]+\s+[A-Z][a-z]+)',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]
        for pattern in filename_patterns:
            match = re.match(pattern, filename_without_ext)
            if match:
                potential_name = match.group(1).strip()
                words = [word.capitalize() if word.isupper() and len(word) > 2 else word for word in potential_name.split()]
                potential_name = ' '.join(words)
                if self._is_valid_name_strict(potential_name):
                    return potential_name

        # Underscore-separated format: "Yen_Beng_Lim_Singapore_12.00_yrs"
        # Extract consecutive capitalized word parts from underscored filename
        parts = filename_without_ext.split('_')
        name_parts = []
        for part in parts:
            # Stop at non-name parts (country names, numbers, common suffixes)
            if re.match(r'^\d', part) or part.lower() in {
                'singapore', 'malaysia', 'india', 'china', 'philippines',
                'yrs', 'years', 'resume', 'cv', 'raw', 'ok', 'temp',
                'sent', 'kiv', 'school', 'admin',
            }:
                break
            if part and part[0].isupper() and len(part) >= 2:
                name_parts.append(part)
        if len(name_parts) >= 2:
            potential_name = ' '.join(name_parts)
            if self._is_valid_name_strict(potential_name):
                return potential_name

        # "Sent - DATE - Name - Title" format
        # e.g., "Sent - 070916 - Shivasanker S - Ex-Jurong Ports..."
        sent_match = re.match(r'^Sent\s*-\s*\d+\s*-\s*([^-]+)', filename_without_ext)
        if sent_match:
            potential_name = sent_match.group(1).strip()
            if potential_name and self._is_valid_name_strict(potential_name):
                return potential_name

        return None

    def _set_extraction_status(self, result: Dict, text: str):
        if not result["Name"] or not result["ID"]:
            result["Extraction_Status"] = "Failed"
            if not result["Name"]:
                result["Notes"] += "🚨 CRITICAL: No name found! "
            if not result["ID"]:
                result["Notes"] += "🚨 No ID found in folder name!"
        else:
            extracted_count = sum(1 for f in ["ID", "Name", "Email", "Phone"] if result[f] is not None)
            if extracted_count >= 5:
                result["Extraction_Status"] = "Complete"
            elif extracted_count >= 3:
                result["Extraction_Status"] = "Success"
            else:
                result["Extraction_Status"] = "Partial"

        if not result["Email"] and not result["Phone"]:
            result["Notes"] += " 🚨 NO CONTACT INFO FOUND!"
            self._emergency_contact_extraction(result, text)

    def _emergency_contact_extraction(self, result: Dict, text: str):
        logger.warning(f"🆘 No contact info for {result.get('Filenames_Processed', 'N/A')} - attempting emergency extraction!")
        emergency_patterns = [
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'(?:Contact|連絡先|联系).*?(\d[\d\-\s\.]{9,})',
        ]
        for pattern in emergency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if '@' in match.group(0) and not result["Email"]:
                    result["Email"] = match.group(1).lower()
                    logger.info(f"🆘 Found emergency email: {result['Email']}")
                elif not result["Phone"]:
                    result["Phone"] = match.group(1)
                    logger.info(f"🆘 Found emergency phone: {result['Phone']}")

    def _normalize_japanese_phone(self, phone: str) -> str:
        phone = phone.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        phone = re.sub(r'^(\+81|０８１|81)', '0', phone)
        phone = re.sub(r'[^\d\-]', '', phone)
        return phone.replace('・', '-').replace('－', '-').replace(' ', '')

    def _is_valid_japanese_phone(self, phone: str) -> bool:
        digits_only = re.sub(r'\D', '', phone)
        if not 10 <= len(digits_only) <= 11:
            return False
        if re.match(r'^0[5789]0', digits_only):
            return len(digits_only) == 11
        if re.match(r'^0\d{9}$', digits_only):
            return True
        return False

    def _extract_and_validate_phone(self, match: re.Match, pattern_type: str) -> Optional[str]:
        try:
            phone = match.group(0).strip()
            if pattern_type == 'JP':
                return self._normalize_japanese_phone(phone)
            elif pattern_type == 'INTL':
                phone = re.sub(r'[^\d+\-]', '', phone)
                if not phone.startswith('+') and len(re.sub(r'\D', '', phone)) > 10:
                    phone = '+' + phone
                return phone if self._is_valid_international_phone(phone) else None
            elif pattern_type in ['HEADER', 'TABLE', 'BULLET', 'GENERIC']:
                phone = match.group(1) if match.lastindex else phone
                phone = phone.strip()
                if self._looks_like_japanese_phone(phone):
                    return self._normalize_japanese_phone(phone)
                else:
                    phone = re.sub(r'[^\d+\-\s]', '', phone)
                    return phone if len(re.sub(r'\D', '', phone)) >= 10 else None
        except Exception as e:
            logger.debug(f"Phone extraction error: {e}")
        return None
    
    def _looks_like_japanese_phone(self, phone: str) -> bool:
        digits = re.sub(r'\D', '', phone)
        if digits.startswith('81'):
            return True
        if digits.startswith('0') and 10 <= len(digits) <= 11:
            return any(digits.startswith(prefix) for prefix in ['050', '070', '080', '090', '03', '06', '011'])
        return False
    
    def _is_valid_international_phone(self, phone: str) -> bool:
        digits = re.sub(r'\D', '', phone)
        if not 7 <= len(digits) <= 15:
            return False
        if re.match(r'^(\d)\1+$', digits):
            return False
        return True
    
    def _is_sequential(self, digits: str) -> bool:
        if len(digits) < 6: return False
        is_asc = all(int(digits[i]) == int(digits[i-1]) + 1 for i in range(1, min(6, len(digits))))
        is_desc = all(int(digits[i]) == int(digits[i-1]) - 1 for i in range(1, min(6, len(digits))))
        return is_asc or is_desc

    def _extract_from_table_format(self, text: str) -> Dict[str, Optional[str]]:
        """
        📊 Extract data from table/structured resume formats.
        Optimized for Singapore English resumes! 🇸🇬
        """
        logger.debug("📊 Attempting table format extraction!")
        data = {}
        # 🇸🇬 Singapore English patterns only
        patterns = {
            'name': r'(?:Name|Full\s*Name|Candidate\s*Name)[\s|│:]\s*([^|│\n]+)',
            'email': r'(?:Email|E-mail|Email\s*Address)[\s|│:]\s*([^|│\n]+)',
            'phone': r'(?:Phone|Tel|Mobile|HP|H/P|Handphone|Hand\s*Phone|Contact\s*(?:No|Number)?)[\s|│:]\s*([^|│\n]+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                if key == 'email':
                    # 🛡️ VALIDATE: the label-capture grabs everything up to the
                    # next pipe/newline, so "Email and phone." yields "and
                    # phone.". Only accept it if it's a real address — otherwise
                    # leave email UNSET so the proper _gather_email_candidates /
                    # _sanitize_email pass (and the AI header) can find it.
                    clean = self._sanitize_email(value)
                    if clean:
                        data[key] = clean
                    else:
                        logger.debug(f"⚠️ Table email rejected (not an email): '{value[:60]}'")
                elif key == 'phone':
                    # 🛡️ VALIDATE: Phone must contain at least 7 digits!
                    # Prevents capturing "NIL", "Available upon request", etc.
                    digits = re.sub(r'\D', '', value)
                    if len(digits) >= 7 and len(digits) <= 15:
                        data[key] = value
                    else:
                        logger.debug(f"⚠️ Table phone rejected (not enough digits): '{value}'")
                else:
                    data[key] = value
        return data

    def check_empty_folders(self, resume_folder: str) -> List[str]:
        """
        🔍 Check for folders without resume files
        """
        empty_folders = []
        
        # Get all candidate folders
        candidate_folders = find_candidate_folders(resume_folder)
        
        for folder_path in candidate_folders:
            # Check if folder has any resume files
            resume_files = find_resume_files(folder_path)
            
            if not resume_files:
                folder_name = os.path.basename(folder_path)
                empty_folders.append(folder_name)
        
        return empty_folders

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        text = unicodedata.normalize('NFKC', text)
        return text.replace('｜', '|').replace('—', '-')

    def _extract_with_ai_assistance(self, text: str, regex_results: Dict) -> Dict:
        """
        🤖 Use AI to validate regex results and extract missing fields!
        This is our BACKUP DANCER that helps when regex struggles!
        """
        if not self.ai_enabled:
            return regex_results
        
        logger.info("🤖 AI ASSISTANCE MODE ACTIVATED!")
        
        # Copy results to avoid modifying original
        ai_enhanced_results = regex_results.copy()
        
        # 1️⃣ VALIDATE EXTRACTED NAME
        if regex_results.get("name"):
            # Get context around where we found the name
            name_context = text[:500]  # Usually names are at the top
            is_valid, confidence = self.ai_validator.validate_name(
                regex_results["name"], 
                name_context
            )
            
            if not is_valid or confidence < 0.6:
                logger.warning(f"🚨 AI rejected name '{regex_results['name']}' (confidence: {confidence})")
                # Try to get a better name from AI
                ai_name = self.ai_validator.extract_from_messy_text(text, 'name')
                if ai_name:
                    ai_enhanced_results["name"] = ai_name
                    ai_enhanced_results["ai_extracted_name"] = True
        
        # 2️⃣ FILL IN MISSING FIELDS WITH AI
        missing_fields = []
        if not regex_results.get("name"):
            missing_fields.append("name")
        if not regex_results.get("phone"):
            missing_fields.append("phone")
        
        if missing_fields:
            logger.info(f"🤖 AI attempting to extract missing fields: {missing_fields}")
            
            # First, try to fix vertical formatting issues
            if "phone" in missing_fields:
                fixed_text = self._fix_vertical_phone_numbers(text[:2000])
                if fixed_text != text[:2000]:
                    # Re-run phone extraction on fixed text
                    phone = self._extract_phone(fixed_text, regex_results.get("email"))
                    if phone:
                        ai_enhanced_results["phone"] = phone
                        missing_fields.remove("phone")
            
            # Use AI for remaining missing fields
            for field in missing_fields:
                ai_result = self.ai_validator.extract_from_messy_text(text, field)
                if ai_result:
                    if field == "name":
                        ai_enhanced_results["name"] = ai_result
                    elif field == "phone":
                        # Standardize AI-extracted phone
                        ai_enhanced_results["phone"] = self._standardize_phone_format(ai_result, 'AI')
        
        # 3️⃣ ADD AI CONFIDENCE SCORES
        ai_enhanced_results["ai_assisted"] = True
        ai_enhanced_results["extraction_method"] = "Regex + AI" if any([
            ai_enhanced_results.get("ai_extracted_name"),
            regex_results != ai_enhanced_results
        ]) else "Regex Only"
        
        return ai_enhanced_results

def save_extraction_feedback(self, filename: str, extracted_data: Dict, 
                           corrections: Optional[Dict] = None):
    """
    💾 Save extraction results for future improvement!
    """
    feedback_file = "extraction_feedback.json"
    
    try:
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    except:
        feedback_data = []
    
    entry = {
        "filename": filename,
        "timestamp": datetime.datetime.now().isoformat(),
        "extracted": extracted_data,
        "corrections": corrections or {},
        "ai_assisted": extracted_data.get("ai_assisted", False)
    }
    
    feedback_data.append(entry)
    
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    
    logger.info("💾 Saved extraction feedback for learning!")

def get_user_settings():
    """Gets batch size and max number of resumes to process from the user."""
    while True:
        try:
            batch_size_input = input(f"💅 Batch size? (Enter for {config.DEFAULT_BATCH_SIZE}): ")
            batch_size = int(batch_size_input) if batch_size_input else config.DEFAULT_BATCH_SIZE
            if batch_size <= 0:
                print("😱 Positive numbers only, sweetie!")
                continue

            total_input = input("💅 How many resumes total? (Enter for ALL): ")
            max_to_process = int(total_input) if total_input else None
            if max_to_process is not None and max_to_process <= 0:
                print("😱 Must be positive, darling!")
                continue
            
            return batch_size, max_to_process
        except ValueError:
            print("💔 Numbers only, honey!")

def process_resumes(extractor, folder_list, processed_folders, batch_size, existing_results=None):
    """
    Orchestrates the resume processing, batching them for efficiency.
    Handles Ctrl+C gracefully by saving progress immediately.
    """
    results = existing_results if existing_results else []
    total_processed_resumes = len(results)  # Start from existing count
    start_time = time.time()
    interrupted = False

    # 🗄️ Initialize database manager
    db_manager = None
    if config.DATABASE_ENABLED:
        try:
            db_manager = DatabaseManager(config.DATABASE_FILE)
        except Exception as e:
            logger.warning(f"⚠️ Database init failed (continuing without DB): {e}")

    performance_monitor = PerformanceMonitor(FeedbackLoopSystem())

    # Process each candidate folder
    total_folders_to_process = len(folder_list)
    pbar = tqdm(total=total_folders_to_process, desc="✨ Processing candidate folders", unit="folder")

    try:
        for candidate_folder_path in folder_list:
            if candidate_folder_path in processed_folders:
                pbar.update(1)
                continue


            # Process this candidate's folder
            result = extractor.process_candidate_folder(candidate_folder_path)
            if result and result.get("ID"):  # Only add if we got a valid result with ID
                results.append(result)
                total_processed_resumes += 1

                # 🗄️ NEW: Save to database alongside CSV/JSON!
                if config.DATABASE_ENABLED and db_manager:
                    try:
                        # Get raw text for DB storage (re-extract from files)
                        raw_text = ""
                        for fname in os.listdir(candidate_folder_path):
                            if fname.lower().endswith(('.pdf', '.docx')) and not fname.startswith('~$'):
                                file_text = extractor.get_text_from_file(
                                    os.path.join(candidate_folder_path, fname)
                                )
                                if file_text:
                                    raw_text += file_text + "\n\n"

                        db_manager.save_extraction(
                            result=result,
                            raw_text=raw_text,
                            folder_path=candidate_folder_path
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ DB save failed (non-fatal): {e}")

            processed_folders.append(candidate_folder_path)
            # Save checkpoint with results included
            save_checkpoint(processed_folders, config.CHECKPOINT_FILE, results)
            pbar.update(1)

    except KeyboardInterrupt:
        interrupted = True
        pbar.close()
        print("\n")
        logger.warning("⚠️ Ctrl+C detected! Saving progress immediately...")

        # Save checkpoint with results
        save_checkpoint(processed_folders, config.CHECKPOINT_FILE, results)
        logger.info(f"💾 Checkpoint saved! {len(processed_folders)} folders, {len(results)} candidates extracted.")

        # Save current results to files immediately
        if results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save to CSV
            df = pd.DataFrame(results)
            emergency_csv = f"{config.OUTPUT_FILENAME_PREFIX}_interrupted_{timestamp}.csv"
            df.to_csv(emergency_csv, index=False, encoding='utf-8-sig')
            logger.info(f"💾 Emergency CSV saved: {emergency_csv}")

            # Save to JSON (formatted to standard export schema)
            emergency_json = f"{config.OUTPUT_FILENAME_PREFIX}_interrupted_{timestamp}.json"
            emergency_export = [format_result_as_export_json(r) for r in results]
            with open(emergency_json, 'w', encoding='utf-8') as f:
                json.dump(emergency_export, f, ensure_ascii=False, indent=8, default=str)
            logger.info(f"💾 Emergency JSON saved: {emergency_json}")

            print(f"\n✅ Saved {len(results)} candidates before exit!")
            print(f"💡 Use option [2] to resume extraction from where you left off.")
        else:
            logger.info("No results to save yet.")

        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"⏱️ Processing interrupted after {total_duration:.2f} seconds.")
        logger.info(f"📊 Candidates processed before interruption: {total_processed_resumes}")

        # 🗄️ Close database connection
        if db_manager:
            db_manager.print_stats_report()
            db_manager.close()

        # Return results so caller can still use them if needed
        return results

    if not interrupted:
        pbar.close()

    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"🏁 Processing completed in {total_duration:.2f} seconds.")
    logger.info(f"📊 Total candidates processed: {total_processed_resumes}")

    performance_monitor.generate_performance_report()

    # 🗄️ Close database connection
    if db_manager:
        db_manager.print_stats_report()
        db_manager.close()
    
     # 🧠 ML: Auto-train models after batch processing
    if config.ML_ENABLED and hasattr(config, 'ML_AUTO_TRAIN_AFTER_BATCH') and config.ML_AUTO_TRAIN_AFTER_BATCH:
        try:
            from ml_engine import MLEngine
            logger.info("🧠 Auto-training ML models on latest data...")
            engine = MLEngine(db_path=config.DATABASE_FILE)
            train_results = engine.train_all()
            logger.info(f"✅ ML training complete: {train_results}")
        except ImportError:
            logger.info("ℹ️ ML engine not installed — skipping auto-train")
        except Exception as e:
            logger.warning(f"⚠️ ML auto-train failed (non-critical): {e}")

    return results


# =============================================================================
# JSON EXPORT FORMAT TRANSFORMATION
# =============================================================================

def _classify_job_function(title: str) -> str:
    """Map a job title string to a broad function category."""
    if not title:
        return ""
    t = title.lower()
    if any(k in t for k in ["sales", "retail", "cashier", "store", "shop", "boutique", "merchandise"]):
        return "Sales"
    if any(k in t for k in ["finance", "account", "audit", "tax", "banking", "investment", "treasury", "credit"]):
        return "Finance"
    if any(k in t for k in ["software", "developer", "programmer", "it ", " it", "tech", "system", "data", "network", "devops", "cloud", "security", "infrastructure"]):
        return "Technology"
    if any(k in t for k in ["hr", "human resource", "recruit", "talent", "people", "compensation", "payroll"]):
        return "Human Resources"
    if any(k in t for k in ["market", "brand", "digital", "content", "social media", "pr", "communications", "advertis"]):
        return "Marketing"
    if any(k in t for k in ["operation", "logistic", "supply chain", "procurement", "warehouse", "inventory", "distribution"]):
        return "Operations"
    if any(k in t for k in ["nurse", "doctor", "medical", "healthcare", "health", "clinic", "pharma", "dental"]):
        return "Healthcare"
    if any(k in t for k in ["admin", "secretary", "receptionist", "clerk", "office manager", "personal assistant"]):
        return "Admin"
    if any(k in t for k in ["teacher", "lecturer", "educator", "tutor", "instructor", "trainer"]):
        return "Education"
    if any(k in t for k in ["customer service", "customer support", "help desk", "call center", "contact center"]):
        return "Customer Service"
    if any(k in t for k in ["manager", "director", "head of", "chief", "vp ", "president", "ceo", "coo", "cfo"]):
        return "Management"
    return ""


# Ordered list of known language names. More-specific dialects come before
# the parent ("Mandarin" before "Chinese") so that "Chinese (Mandarin)" yields
# "Mandarin" rather than "Chinese" when both match.
_KNOWN_LANGUAGES: List[str] = [
    "Mandarin", "Cantonese", "Hokkien", "Teochew", "Hakka",
    "English", "Chinese", "Malay", "Bahasa Melayu", "Tamil",
    "Hindi", "Japanese", "Korean", "French", "German", "Spanish",
    "Portuguese", "Italian", "Arabic", "Bengali", "Urdu", "Punjabi",
    "Indonesian", "Bahasa Indonesia", "Burmese", "Sinhalese", "Thai",
    "Vietnamese", "Tagalog", "Filipino", "Greek", "Russian", "Dutch",
    "Swedish", "Norwegian", "Danish", "Finnish", "Polish", "Czech",
    "Hungarian", "Romanian", "Turkish",
]
_KNOWN_LANGUAGES_LOWER: set = {l.lower() for l in _KNOWN_LANGUAGES}


def _extract_languages_from_raw(text: str) -> List[str]:
    """
    Scan `text` for any known language names and return them in the canonical
    capitalisation.  Handles cases like:
      "Writes & Speaksenglish & Malay."  → ["English", "Malay"]
      "Reads"                            → []
      "Others"                           → []
      "Page 2"                           → []

    Tries word-boundary match first; falls back to substring match to catch
    verb-concatenated tokens like "Speaksenglish".
    """
    found: List[str] = []
    seen: set = set()
    for lang in _KNOWN_LANGUAGES:
        key = lang.lower()
        if key in seen:
            continue
        # Word-boundary search (normal case)
        if re.search(rf'\b{re.escape(lang)}\b', text, re.IGNORECASE):
            found.append(lang)
            seen.add(key)
            # If we matched a dialect, suppress the parent "Chinese"
            if key in {"mandarin", "cantonese", "hokkien", "teochew", "hakka"}:
                seen.add("chinese")
        # Substring fallback for concatenated tokens, e.g. "Speaksenglish"
        elif re.search(re.escape(lang), text, re.IGNORECASE):
            found.append(lang)
            seen.add(key)
            if key in {"mandarin", "cantonese", "hokkien", "teochew", "hakka"}:
                seen.add("chinese")
    return found


def _classify_degree_type(degree_text: str, institution: str) -> str:
    """Classify a degree into a standard type."""
    combined = f"{degree_text} {institution}".lower()
    if any(k in combined for k in ["phd", "ph.d", "doctor", "doctorate"]):
        return "PhD"
    if any(k in combined for k in ["master", "msc", "m.sc", "mba", "m.eng", "m.a.", "m.b.a", "m.ed", "mphil"]):
        return "Master"
    if any(k in combined for k in ["bachelor", "bsc", "b.sc", "b.eng", "b.a.", "bba", "b.b.a", "honours", "honor", "hons", "degree in"]):
        return "Bachelor"
    if any(k in combined for k in ["diploma", "dip.", " dip "]):
        return "Diploma"
    return "Others"


def format_result_as_export_json(result: Dict) -> Dict:
    """
    Transform an internal extraction result dict into the standard candidate
    export JSON format used for downstream systems.
    """
    # --- Work Experience ---
    raw_exp = result.get("_raw_experience") or []
    work_experience = []
    for job in raw_exp:
        if not isinstance(job, dict):
            continue
        dates_str = job.get("dates", "")
        from_date, to_date = "", ""
        if " - " in dates_str:
            parts = dates_str.split(" - ", 1)
            from_date = parts[0].strip()
            to_date = parts[1].strip()
        elif dates_str:
            from_date = dates_str

        description = job.get("description", "")
        if description and description.lower() not in {"description not available", "n/a", "none", ""}:
            responsibilities = [r.strip() for r in description.split("|") if r.strip() and len(r.strip()) > 3]
        else:
            responsibilities = []

        work_experience.append({
            "company": job.get("company", ""),
            "title": job.get("role", ""),
            "from": from_date,
            "to": to_date,
            "responsibility": responsibilities,
        })

    # --- Education ---
    raw_edu = result.get("_raw_education") or []
    education = []
    for edu in raw_edu:
        if not isinstance(edu, dict):
            continue
        institution = edu.get("institution", "")
        degree_text = edu.get("degree", "")
        education.append({
            "school": institution,
            "major": degree_text,
            "Degree": _classify_degree_type(degree_text, institution),
        })

    # --- Language Skills ---
    raw_langs = result.get("_raw_languages") or []
    language_skills = []
    seen_langs: set = set()

    for lang in raw_langs:
        if isinstance(lang, dict):
            lang_name = lang.get("language", "")
        elif isinstance(lang, str):
            lang_name = lang
        else:
            continue
        if not lang_name:
            continue
        for extracted in _extract_languages_from_raw(lang_name):
            if extracted.lower() not in seen_langs:
                seen_langs.add(extracted.lower())
                language_skills.append(extracted)

    # --- Skills & Tags ---
    hard_skills = result.get("_raw_hard_skills") or []
    soft_skills = result.get("_raw_soft_skills") or []
    all_skills = [s for s in list(hard_skills) + list(soft_skills) if s]
    # Tags: short hard-skill tokens that look like tool/system names (≤3 words, ≤30 chars)
    tags = [s for s in hard_skills if s and len(s.split()) <= 3 and len(s) <= 30]

    # --- Current Company / Title from most-recent role ---
    current_company = work_experience[0].get("company", "") if work_experience else ""
    current_title = work_experience[0].get("title", "") if work_experience else ""

    return {
        "ID": str(result.get("ID", "")) if result.get("ID") is not None else "",
        "Name": result.get("Name") or "",
        "Page": "",
        "Phone": result.get("Phone") or "",
        "Email": result.get("Email") or "",
        "Current Company": current_company,
        "Current Title": current_title,
        "Team": "",
        "Current Location": result.get("Location") or "",
        "Expected Location": "",
        "Gender": "",
        "Created By": "",
        "Creation Date": "",
        "Last Contact": "",
        "Function": _classify_job_function(current_title),
        "Industry": [],
        "Summary": result.get("Summary") or "",
        "Language Skills": language_skills,
        "Work Experience": work_experience,
        "Education": education,
        "Project Experience": "",
        "tags": tags,
        "skills": all_skills,
    }


def generate_reports(results: List[Dict], empty_folders: List[str]):
    """Generates CSV reports for the extraction results."""
    print("\n" + "═" * 80)
    print("🎉 CURTAIN CALL! The show is complete! 🎉".center(80))
    print("═" * 80)
    
    df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 🌟 NEW: AI STATISTICS SECTION! 🌟
    # Calculate AI usage stats
    ai_assisted_count  = sum(1 for r in results if r.get('AI_Assisted', False))
    ai_enhanced_names  = sum(1 for r in results if r.get('AI_Assisted', False) and r.get('Name'))
    # Phone and Email are extracted by regex; count per actual method
    regex_phones = sum(1 for r in results if r.get('Phone_Method') == 'Regex')
    regex_emails = sum(1 for r in results if r.get('Email_Method') == 'Regex')
    # Main data report - CSV
    output_csv = f"{config.OUTPUT_FILENAME_PREFIX}_{timestamp}.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n📄 Main data saved to: {output_csv}")

    # Main data report - JSON (formatted to standard export schema)
    output_json = f"{config.OUTPUT_FILENAME_PREFIX}_{timestamp}.json"
    export_records = [format_result_as_export_json(r) for r in results]
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(export_records, f, ensure_ascii=False, indent=8, default=str)
    print(f"📄 Main data saved to: {output_json}")

    # 🗣️ NEW: Language Classification Report 🗣️
    if 'Language' in df.columns:
        print("\n" + "─" * 80)
        print("🗣️ LANGUAGE CLASSIFICATION".center(80))
        print("─" * 80)

        japanese_resumes = df[df['Language'] == 'Japanese']
        english_resumes = df[df['Language'] == 'English']
        unknown_resumes = df[df['Language'] == 'Unknown']

        print(f"\n🇯🇵 Japanese Resumes ({len(japanese_resumes)}):")
        if not japanese_resumes.empty:
            for _, row in japanese_resumes.iterrows():
                print(f"  - ID: {row.get('ID', 'N/A')}, Files: {row.get('Filenames_Processed', 'N/A')}")
        else:
            print("  None found.")

        print(f"\n🇬🇧 English Resumes ({len(english_resumes)}):")
        if not english_resumes.empty:
            for _, row in english_resumes.iterrows():
                print(f"  - ID: {row.get('ID', 'N/A')}, Files: {row.get('Filenames_Processed', 'N/A')}")
        else:
            print("  None found.")

        if not unknown_resumes.empty:
            print(f"\n❓ Unknown Language Resumes ({len(unknown_resumes)}):")
            for _, row in unknown_resumes.iterrows():
                print(f"  - ID: {row.get('ID', 'N/A')}, Files: {row.get('Filenames_Processed', 'N/A')}")

    # Missing fields report
    missing_report = []
    for _, row in df.iterrows():
        missing = [
            f for f in ['Name', 'Email', 'Phone', 'Skills', 'Working_Experience', 'School_University']
            if pd.isna(row.get(f)) or str(row.get(f, '')).strip() in ('', 'None', '[]', '{}')
        ]
        if missing:
            missing_report.append({
                'ID': row.get('ID', 'Unknown'),
                'Filenames_Processed': row['Filenames_Processed'],
                'Missing_Fields': ', '.join(missing)
            })
    
    if missing_report:
        missing_df = pd.DataFrame(missing_report)
        missing_csv = f"missing_fields_report_{timestamp}.csv"
        missing_df.to_csv(missing_csv, index=False, encoding='utf-8-sig')
        print(f"🚨 Missing Fields Report: {missing_csv}")

    # Empty folders report
    if empty_folders:
        empty_df = pd.DataFrame({'Empty_Folders': empty_folders})
        empty_csv = f"empty_folders_report_{timestamp}.csv"
        empty_df.to_csv(empty_csv, index=False, encoding='utf-8-sig')
        print(f"📁 Empty Folders Report: {empty_csv}")

    # Summary statistics
    print("\n📊 EXTRACTION SUMMARY:")
    print(f"✅ Total resumes processed: {len(df)}")
    print(f"🚨 Resumes with missing fields: {len(missing_report)}")
    print(f"📁 Empty candidate folders: {len(empty_folders)}")
    
    # 🗄️ NEW: Database quality report
    if config.DATABASE_ENABLED:
        try:
            db = DatabaseManager(config.DATABASE_FILE)
            db.print_stats_report()

            # Show wrong-field suspects
            suspects = db.get_wrong_field_suspects()
            if suspects:
                print(f"\n🕵️ WRONG FIELD SUSPECTS ({len(suspects)}):")
                for s in suspects[:10]:
                    print(f"   ID {s['candidate_id']}: {', '.join(s['issues'])}")

            # Show reprocessing candidates
            failed = db.get_candidates_for_reprocessing(status_filter='Failed')
            if failed:
                print(f"\n🔄 Candidates needing re-extraction: {len(failed)}")
                print(f"   IDs: {failed[:20]}{'...' if len(failed) > 20 else ''}")

            db.close()
        except Exception as e:
            logger.warning(f"⚠️ DB report failed: {e}")

    # 🌟 AI Assistance Report (OPTIONAL BUT FABULOUS!) 🌟
    if ai_assisted_count > 0:
        ai_report = []
        for _, row in df.iterrows():
            if row.get('AI_Assisted', False):
                ai_enhanced = [
                    field for field in ['Name', 'Skills', 'Working_Experience', 'School_University', 'Summary']
                    if pd.notna(row.get(field)) and str(row.get(field, '')).strip() not in ('', 'None', '[]', '{}')
                ]
                ai_report.append({
                    'ID': row.get('ID', 'Unknown'),
                    'Filenames_Processed': row['Filenames_Processed'],
                    'AI_Enhanced_Fields': ', '.join(ai_enhanced),
                    'Phone_Method': row.get('Phone_Method', 'Regex'),
                    'Email_Method': row.get('Email_Method', 'Regex'),
                    'Extraction_Status': row.get('Extraction_Status', 'Unknown')
                })
        
        if ai_report:
            ai_df = pd.DataFrame(ai_report)
            ai_csv = f"ai_assistance_report_{timestamp}.csv"
            ai_df.to_csv(ai_csv, index=False, encoding='utf-8-sig')
            print(f"\n🤖 AI Assistance Report: {ai_csv}")

    # 🌟 NEW: AI ASSISTANCE STATISTICS! 🌟
    if ai_assisted_count > 0:
        print("\n🤖 AI ASSISTANCE REPORT:")
        print(f"✨ Total AI-Assisted Extractions: {ai_assisted_count}/{len(results)} ({ai_assisted_count/len(results)*100:.1f}%)")
        print(f"   💅 Names enhanced by AI: {ai_enhanced_names}")
        print(f"   📱 Phones extracted by Regex: {regex_phones}")
        print(f"   📧 Emails extracted by Regex: {regex_emails}")
        
        # Which files needed AI help the most?
        ai_files = [r['Filenames_Processed'] for r in results if r.get('AI_Assisted', False)]
        if ai_files and len(ai_files) <= 10:
            print("\n   📁 Files that needed AI help:")
            for filename in ai_files[:5]:  # Show first 5
                print(f"      - {filename}")
            if len(ai_files) > 5:
                print(f"      ... and {len(ai_files) - 5} more")
    else:
        print("\n🤖 AI ASSISTANCE: Not needed - Regex handled everything! 💪")

    print("\n📊 Field Extraction Success Rates:")
    field_display = [
        ('ID',                 'ID'),
        ('Name',               'Name'),
        ('Email',              'Email'),
        ('Phone',              'Phone'),
        ('Location',           'Location'),
        ('Skills',             'Skills'),
        ('Working_Experience', 'Working Experience'),
        ('School_University',  'Education'),
        ('Summary',            'Summary'),
    ]
    for col, label in field_display:
        if col in df.columns:
            filled = df[col].apply(
                lambda v: bool(v) and str(v).strip() not in ('', 'None', 'nan', '[]', '{}')
            ).sum()
            success_rate = (filled / len(df)) * 100 if len(df) > 0 else 0
            method_tag = ""
            if col == 'Phone':
                method_tag = f" [Regex: {regex_phones}]"
            elif col == 'Email':
                method_tag = f" [Regex: {regex_emails}]"
            emoji = "✅" if success_rate > 80 else "⚠️" if success_rate > 50 else "🚨"
            print(f"  {emoji} {label}: {success_rate:.1f}% success{method_tag}")
    
    # 🌟 NEW: Show extraction method breakdown! 🌟
    if 'AI_Assisted' in df.columns:
        print("\n🎯 EXTRACTION METHOD BREAKDOWN:")
        regex_only = len(df[df['AI_Assisted'] == False])
        ai_assisted = len(df[df['AI_Assisted'] == True])
        
        print(f"  ⚡ Regex-only extractions: {regex_only} ({regex_only/len(df)*100:.1f}%)")
        print(f"  🤖 AI-assisted extractions: {ai_assisted} ({ai_assisted/len(df)*100:.1f}%)")
        
        # Per-field success rate: Regex-only vs AI-assisted rows
        if ai_assisted > 0:
            compare_fields = [
                ('Name',               'Name'),
                ('Skills',             'Skills'),
                ('Working_Experience', 'Working Experience'),
                ('School_University',  'Education'),
            ]
            print(f"\n  📊 Success rate comparison (Regex-only vs AI-assisted rows):")
            for col, label in compare_fields:
                if col not in df.columns:
                    continue
                r_rate = df[df['AI_Assisted'] == False][col].notna().sum() / regex_only * 100 if regex_only > 0 else 0
                a_rate = df[df['AI_Assisted'] == True][col].notna().sum() / ai_assisted * 100
                print(f"     {label:20s} — Regex-only: {r_rate:.1f}%  |  AI-assisted: {a_rate:.1f}%")
        print(f"\n  🔑 Phone/Email always use Regex — not counted as AI-assisted.")

    # In generate_reports function
    if any(r.get('AI_Used', False) for r in results):
        ai_times = [r.get('processing_time', 0) for r in results if r.get('AI_Used')]
        regex_times = [r.get('processing_time', 0) for r in results if not r.get('AI_Used')]
        
        print("\n🏎️ PERFORMANCE REPORT:")
        print(f"  🌸 AI avg time: {sum(ai_times)/len(ai_times):.2f}s per resume")
        print(f"  ⚡ Regex-only avg time: {sum(regex_times)/len(regex_times):.2f}s per resume")

def find_resume_files(folder_path: str) -> List[str]:
    """
    🔍 Find all resume files (PDF/DOCX) within a folder
    """
    resume_files = []
    
    # Look for PDF and DOCX files
    for item in os.scandir(folder_path):
        if item.is_file() and not item.name.startswith('~$'):
            if item.name.lower().endswith(('.pdf', '.docx')):
                resume_files.append(item.path)
    
    return sorted(resume_files)

def list_candidates_and_files(extractor: "UltimateResumeExtractor"):
    """📜 List all candidates, their resume files, languages, and formats."""
    print("\n" + "📜" * 40)
    print("📜 Candidate File Report 📜".center(80))
    print("📜" * 40 + "\n")

    try:
        all_folders = find_candidate_folders(config.RESUME_FOLDER)
        if not all_folders:
            print(f"😱 No candidate folders found in '{config.RESUME_FOLDER}'!")
            return
    except FileNotFoundError:
        print(f"😱 ERROR: The folder '{config.RESUME_FOLDER}' doesn't exist, darling!")
        return

    candidate_data = {}
    for folder_path in tqdm(all_folders, desc="🕵️‍♀️ Analyzing files"):
        candidate_name = os.path.basename(folder_path)
        
        if candidate_name not in candidate_data:
            candidate_data[candidate_name] = {
                "Candidate": candidate_name,
                "File Details": [],
                "Many Files": "No"
            }

        try:
            files_in_folder = os.listdir(folder_path)
        except FileNotFoundError:
            continue

        if not files_in_folder:
            candidate_data[candidate_name]["File Details"].append("N/A (Folder is empty)")
            continue

        file_count = len(files_in_folder)
        if file_count > 5:
            candidate_data[candidate_name]["Many Files"] = "Yes"

        for filename in files_in_folder:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_format = os.path.splitext(filename)[1].lower()
                language = "Unknown"
                
                if "japanese" in filename.lower() or any(char in "履歴書職務経歴書" for char in filename):
                    language = "Japanese"
                elif "english" in filename.lower() or "resume" in filename.lower():
                    language = "English"

                candidate_data[candidate_name]["File Details"].append(f"{filename} ({language}, {file_format})")

    report_data = []
    for candidate_name, data in candidate_data.items():
        report_data.append({
            "Candidate": data["Candidate"],
            "File Details": "; ".join(data["File Details"]),
            "Many Files": data["Many Files"]
        })

    if report_data:
        df = pd.DataFrame(report_data)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"all_candidates_report_{timestamp}.csv"
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n📄 Report saved to: {os.path.abspath(output_csv)}")

        # JSON export
        output_json = f"all_candidates_report_{timestamp}.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"📄 Report saved to: {os.path.abspath(output_json)}")
        
        # Display the report in the terminal
        print("\n" + df.to_string())
    else:
        print("🤷‍♀️ No candidates found to analyze.")

def main():
    """🎭 The Main Show with MENU MAGIC!"""

    from utils import display_menu, save_checkpoint, load_checkpoint, get_checkpoint_info, clear_checkpoint, print_batch_table, FeedbackLoopSystem, InteractiveCorrectionSystem, PatternLearningSystem, PerformanceMonitor, select_folders_to_process_enhanced

    folder_list = []
    processed_folders = []
    existing_results = []  # Store results from checkpoint

    while True:
        # Get checkpoint info for menu display
        checkpoint_info = get_checkpoint_info(config.CHECKPOINT_FILE)
        display_menu(checkpoint_info)
        choice = input("\n💅 What's it gonna be, sweetie? (1-7): ").strip()

        # 🌟 NEW: Get candidate folders with proper structure! 🌟
        try:
            all_folders = find_candidate_folders(config.RESUME_FOLDER)
            if not all_folders:
                print(f"😱 No candidate folders found in '{config.RESUME_FOLDER}'!")
                print("Make sure your structure is: merlion_resumes/[date_folder]/[candidate_folders]/")
                return
        except FileNotFoundError:
            print(f"😱 ERROR: The folder '{config.RESUME_FOLDER}' doesn't exist, darling!")
            return

        if choice == "1":
            print("\n✨ Starting fresh! Let's process all resumes! ✨")
            # Clear existing checkpoint
            if checkpoint_info and checkpoint_info.get("exists"):
                clear_checkpoint(config.CHECKPOINT_FILE)
                print("🗑️ Previous checkpoint cleared!")
            print(f"🎯 Found {len(all_folders)} candidate folders to process!")
            processed_folders = []
            existing_results = []
            folder_list = all_folders
            break

        elif choice == "2":
            processed_folders, existing_results, timestamp = load_checkpoint(config.CHECKPOINT_FILE)
            if processed_folders:
                print(f"\n👑 Welcome back! Resuming from checkpoint:")
                print(f"   📁 Folders already processed: {len(processed_folders)}")
                print(f"   ✅ Candidates already extracted: {len(existing_results)}")
                remaining = len(all_folders) - len(processed_folders)
                print(f"   📋 Remaining to process: {remaining}")
            else:
                print("\n🤔 No checkpoint found, darling. Starting fresh!")
                existing_results = []
            folder_list = [f for f in all_folders if f not in processed_folders]
            break
            
        elif choice == "3":
            print("\n🎯 SELECTIVE PROCESSING")

            # Uses the SAME scanner that produced the count above
            # (find_candidate_folders -> all_folders), so flat/numeric and
            # date-wrapped layouts both work. Unlike option 4 (debug: auto
            # runs the FIRST folder), this lets the user pick WHICH folders.
            if not all_folders:
                print("😱 No candidate folders found!")
                continue

            print(f"\n💡 {len(all_folders)} total candidate folders available")

            # With thousands of folders, dumping every name is unusable, so
            # offer an optional name filter first. Empty input = list all.
            term = input(
                "🔎 Type part of a folder name to narrow the list "
                "(or press Enter to list all): "
            ).strip().lower()
            if term:
                candidate_pool = [
                    f for f in all_folders
                    if term in os.path.basename(f).lower()
                ]
                if not candidate_pool:
                    print(f"😱 No folders matched '{term}', darling! Try again.")
                    continue
                print(f"✨ {len(candidate_pool)} folder(s) match '{term}'")
            else:
                candidate_pool = all_folders

            # Let the user choose exactly which folders to run
            # (numbers like "1, 3, 4", or "all").
            folder_list = select_folders_to_process_enhanced(candidate_pool)
            if not folder_list:
                print("🤷‍♀️ Nothing selected — back to the menu.")
                continue

            processed_folders = []
            break
            
        elif choice == "4":
            print("\n🔍 DEBUG MODE - Testing ONE folder")
            
            # Find first candidate folder
            test_folder = None
            all_debug_folders = find_candidate_folders(config.RESUME_FOLDER)
            if all_debug_folders:
                test_folder = all_debug_folders[0]
            
            if test_folder:
                print(f"📁 Testing with: {test_folder}")
                extractor = UltimateResumeExtractor(config.SUZUME_MODEL_NAME)
                result = extractor.process_candidate_folder(test_folder)
                
                print("\n✨ RESULTS:")
                for key, value in result.items():
                    if key != "Filenames_Processed":
                        print(f"  {key}: {value}")
            else:
                print("❌ No test folder found!")
            return
            
        elif choice == "5":
            print("\n📭 Finding candidates with missing resumes...")
            empty_folders = []
            for folder_path in all_folders:
                if not find_resume_files(folder_path):
                    empty_folders.append(os.path.basename(folder_path))
            
            if not empty_folders:
                print("🎉 All candidate folders have resumes! Nothing to report.")
            else:
                print(f"🚨 Found {len(empty_folders)} candidates with missing resumes:")
                for folder_name in empty_folders:
                    print(f"  - {folder_name}")
                
                # Save report
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                report_df = pd.DataFrame({'CandidateFolder_Missing_Resume': empty_folders})
                report_filename = f"missing_resumes_report_{timestamp}.csv"
                report_df.to_csv(report_filename, index=False, encoding='utf-8-sig')
                print(f"\n📄 Report saved to: {report_filename}")
            continue

        elif choice == "7":
            print("\n💋 Until we meet again, darling! Stay fabulous! ✨")
            return

        elif choice == "6":
            print("\n📜 Listing candidates and their resume languages...")
            extractor = UltimateResumeExtractor(config.SUZUME_MODEL_NAME)
            list_candidates_and_files(extractor)
            continue
            
        else:
            print("\n😱 That's not an option, honey! Try again!")

    if not folder_list:
        print("\n🎉 All candidate folders have been processed! Nothing left to do!")
        # If we have existing results from checkpoint, offer to generate reports
        if existing_results:
            print(f"💡 You have {len(existing_results)} extracted candidates from checkpoint.")
            generate_choice = input("Generate reports from existing data? (y/n): ").strip().lower()
            if generate_choice == 'y':
                generate_reports(existing_results, [])
        return

    batch_size, max_to_process = get_user_settings()

    extractor = UltimateResumeExtractor(config.SUZUME_MODEL_NAME)

    if max_to_process is not None:
        folder_list = folder_list[:max_to_process]

    # Pass existing_results to continue from where we left off
    results = process_resumes(extractor, folder_list, processed_folders, batch_size, existing_results)

    if results:
        generate_reports(results, [])  # Empty folders check can be integrated elsewhere if needed
    else:
        print("\nNo new data was processed to generate a report.")

if __name__ == "__main__":
    main()