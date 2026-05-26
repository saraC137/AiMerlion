"""
ai_extractor.py

This module implements the core AI-powered and regex-based data extraction logic
for resumes within the AiMerlion system. It specializes in extracting structured
information such as personal details, skills, work experience, and education.

The AIExtractor class utilizes a hybrid strategy:
- An AI-first approach for extracting header fields (e.g., name, email, phone)
  by interacting with an Ollama language model and enforcing strict JSON output.
- A robust regex-first approach for detailed sections like work experience and
  education, with AI assistance for validation or to fill in missing information.

It includes advanced text cleaning for markdown, comprehensive regex patterns
for various data types, and mechanisms to handle malformed AI responses,
ensuring high-quality and structured data output.
"""

import ollama
import json
import re
from typing import Dict, Optional, List, Any, Tuple, Union
import logging
import coloredlogs

# Import helper functions from your utils.py
from utils import standardize_phone_number, standardize_date
import config
from extraction.text_preprocessor import clean as _pp_clean, detect_sections as _pp_detect_sections

# =====================================================================
# JSON SCHEMAS FOR STRUCTURED OUTPUT (Ollama format parameter)
# Passed as format= to ollama.chat() to enforce grammar-based constrained
# generation — the model CANNOT output fields with wrong types or missing
# required keys. Eliminates the need for post-processing "fix" methods.
# =====================================================================

HEADER_SCHEMA = {
    "type": "object",
    "properties": {
        "name":          {"type": ["string", "null"]},
        "email":         {"type": ["string", "null"]},
        "phone":         {"type": ["string", "null"]},
        "nationality":   {"type": ["string", "null"]},
        "location":      {"type": ["string", "null"]},
        "language":      {"type": ["string", "null"]},
        "linkedin":      {"type": ["string", "null"]},
        "github":        {"type": ["string", "null"]},
        "website":       {"type": ["string", "null"]},
    },
    "required": [
        "name", "email", "phone", "nationality",
        "location", "language", "linkedin", "github", "website",
    ],
}

SKILLS_SCHEMA = {
    "type": "object",
    "properties": {
        "hard": {"type": "array", "items": {"type": "string"}},
        "soft": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hard", "soft"],
}

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary":        {"type": ["string", "null"]},
        "skills":         {"type": ["string", "null"]},
        "experience":     {"type": ["string", "null"]},
        "education":      {"type": ["string", "null"]},
        "certifications": {"type": ["string", "null"]},
        "languages":      {"type": ["string", "null"]},
        "others":         {"type": ["string", "null"]},
    },
    "required": [
        "summary", "skills", "experience", "education",
        "certifications", "languages", "others"
    ],
}


class AIExtractor:
    """
    💅 The English Resume Extraction Specialist!
    Separates Hard/Soft skills and guarantees structured output for experience.
    """
    
    def __init__(self, model_name: str, logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            coloredlogs.install(level='INFO', logger=self.logger,
                              fmt='%(asctime)s - 🌸 %(levelname)s - %(message)s')

        _timeout = getattr(config, 'AI_OLLAMA_TIMEOUT', 180)
        self._client = ollama.Client(timeout=_timeout)
        self.available = self._test_model()

        # BERT NER engine (lazy model load on first use)
        self._ner_engine = None
        self._ner_enabled = False
        try:
            from extraction.ner_engine import get_ner_engine
            self._ner_engine = get_ner_engine(self.logger)
            self._ner_enabled = True
            self.logger.info("🧬 BERT NER Engine ready!")
        except Exception as e:
            self.logger.warning(f"⚠️ BERT NER not available: {e}")

    def _test_model(self) -> bool:
        """Test if the Ollama model is ready to serve!"""
        try:
            self.logger.info(f"🌸 Initializing {self.model_name} model...")
            self._client.chat(model=self.model_name, messages=[{'role': 'user', 'content': 'Hi'}])
            self.logger.info(f"✨ {self.model_name} is READY! Extraction engines online!")
            return True
        except Exception as e:
            self.logger.error(f"âŒ Model error: {e}")
            self.logger.error(f"Please ensure the model '{self.model_name}' is pulled in Ollama.")
            return False
    
    def _normalize_text_for_extraction(self, text: str) -> str:
        """
        🧚‍♀️✨ FAIRY CODEMOTHER'S ULTIMATE TEXT NORMALIZER v1.0! ✨🧚‍♀️
        
        This method MUST be called ONCE at the very beginning of extraction!
        It transforms chaotic raw text into clean, regex-ready text.
        
        Handles:
        - Encoding issues (mojibake from UTF-8 misread as Latin-1)
        - Invisible characters (zero-width spaces, BOM, etc.)
        - Multiple whitespace types (NBSP, em-space, etc.)
        - Different dash characters (en-dash, em-dash, minus sign)
        - Smart quotes (curly quotes, guillemets, etc.)
        - Various bullet characters (bullet, pointer, checkmark, etc.)
        - Line ending normalization (Windows, Mac, Unix)
        
        WHY THIS MATTERS:
        Raw text: "Feb 2016 [en-dash] Aug 2016" -> Regex for "-" WON'T MATCH!
        Normalized: "Feb 2016 - Aug 2016" -> Regex for "-" WILL MATCH!
        
        Returns:
            Normalized text ready for regex extraction
        """
        if not text:
            return ""
        
        # =====================================================================
        # STEP 1: FIX ENCODING ISSUES (Mojibake from UTF-8 -> Latin-1)
        # =====================================================================
        encoding_fixes = {
            '–': '-', '–': '-',  # dashes
            ''': "'", ''': "'", '"': '"', '"': '"',  # quotes
            '•': '* ',  # bullet
            '"¦': '...',  # ellipsis
            'Ã©': 'e', 'Ã¨': 'e', 'Ã¢': 'a', 'Ã®': 'i', 'Ã´': 'o', 'Ã»': 'u', 'Ã§': 'c', 'Ã±': 'n',
            ' ': ' ', '·': '*',
        }
        for bad, good in encoding_fixes.items():
            text = text.replace(bad, good)
        
        # =====================================================================
        # STEP 2: REMOVE INVISIBLE CHARACTERS
        # =====================================================================
        invisible_chars = [
            '\u200b', '\u200c', '\u200d', '\u200e', '\u200f',
            '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
            '\u2060', '\u2061', '\u2062', '\u2063', '\u2064',
            '\ufeff', '\ufffe', '\uffff',
        ]
        for char in invisible_chars:
            text = text.replace(char, '')
        
        # =====================================================================
        # STEP 3: NORMALIZE WHITESPACE CHARACTERS
        # =====================================================================
        whitespace_map = {
            '\u00a0': ' ', '\u2000': ' ', '\u2001': ' ', '\u2002': ' ',
            '\u2003': ' ', '\u2004': ' ', '\u2005': ' ', '\u2006': ' ',
            '\u2007': ' ', '\u2008': ' ', '\u2009': ' ', '\u200a': ' ',
            '\u202f': ' ', '\u205f': ' ', '\u3000': ' ',
        }
        for char, replacement in whitespace_map.items():
            text = text.replace(char, replacement)
        
        # =====================================================================
        # STEP 4: NORMALIZE LINE SEPARATORS
        # =====================================================================
        text = text.replace('\u2028', '\n')
        text = text.replace('\u2029', '\n')
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        # =====================================================================
        # STEP 5: NORMALIZE DASHES (Critical for date ranges!)
        # =====================================================================
        dash_chars = [
            '\u2013', '\u2014', '\u2010', '\u2011', '\u2012',
            '\u2015', '\u2212', '\u2043', '\ufe58', '\ufe63', '\uff0d',
        ]
        for char in dash_chars:
            text = text.replace(char, '-')
        
        # =====================================================================
        # STEP 6: NORMALIZE QUOTES
        # =====================================================================
        double_quotes = ['\u201c', '\u201d', '\u201e', '\u201f', '\u00ab', '\u00bb', '\u2033']
        single_quotes = ['\u2018', '\u2019', '\u201a', '\u201b', '\u2039', '\u203a', '\u2032', '\u0060', '\u00b4']
        for char in double_quotes:
            text = text.replace(char, '"')
        for char in single_quotes:
            text = text.replace(char, "'")
        
        # =====================================================================
        # STEP 7: NORMALIZE BULLET CHARACTERS
        # =====================================================================
        bullet_chars = [
            '\u2022', '\u25cf', '\u25cb', '\u25e6', '\u25a0', '\u25a1',
            '\u25aa', '\u25ab', '\u25ba', '\u25b8', '\u25b6', '\u25b7',
            '\u2023', '\u2219', '\u22c5', '\u00b7', '\u2192', '\u27a2',
            '\u27a4', '\u25c6', '\u25c7', '\u2605', '\u2606', '\u2713',
            '\u2714', '\u2717', '\u2718', '\u261e', '\u2794',
        ]
        for char in bullet_chars:
            text = text.replace(char, '* ')
        
        # =====================================================================
        # STEP 8: CLEAN UP CONTROL CHARACTERS
        # =====================================================================
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # =====================================================================
        # STEP 9: NORMALIZE MULTIPLE SPACES
        # =====================================================================
        text = re.sub(r'[ ]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        # Delegate final pass to the canonical preprocessor
        result = _pp_clean(text)
        self.logger.info("Text normalization complete.")
        return result
    
    def _extract_email_regex(self, text: str) -> Optional[str]:
        """
        📝§ Extract email using PURE REGEX - bulletproof extraction!
        Returns the first valid email found, or None.
        """
        # Standard email regex pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

        # Search in the first part of the resume (header area)
        header_text = text[:3000] if len(text) > 3000 else text

        matches = re.findall(email_pattern, header_text)

        if matches:
            # Filter out common non-personal emails
            excluded_domains = ['example.com', 'test.com', 'email.com', 'domain.com']
            for email in matches:
                email_lower = email.lower()
                # Skip placeholder/example emails
                if any(domain in email_lower for domain in excluded_domains):
                    continue
                # Skip if it looks like a company support email embedded in text
                if 'support@' in email_lower or 'info@' in email_lower or 'noreply@' in email_lower:
                    continue
                self.logger.info(f"📝§ Regex found email: {email}")
                return email

        return None

    def _is_valid_email(self, email: str) -> bool:
        """
        âœ… Validate that a string is actually an email address format.
        Returns False for garbage text that AI might return.
        """
        if not email or not isinstance(email, str):
            return False

        # Basic length check - emails are typically short
        if len(email) > 100:
            return False

        # Must contain @ and at least one dot after @
        if '@' not in email:
            return False

        # Standard email regex validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email.strip()))

    def _extract_phone_regex(self, text: str) -> Optional[str]:
        """
        Extract phone number using pure regex for Singapore (+65) and Malaysia (+60).
        Searches labeled fields first (HP, Phone, Mobile, Tel), then bare numbers.
        Returns raw matched string; caller should pass through standardize_phone_number.
        """
        header_text = text[:3000] if len(text) > 3000 else text

        # --- Labeled phone patterns (highest confidence) ---
        label_pattern = (
            r'(?:HP|Handphone|H\/P|Phone|Mobile|Tel(?:ephone)?|Contact|Cel(?:l)?)'
            r'\s*[:\-]?\s*'
            r'(\+?[\d][\d\s\-\(\)\.]{6,17})'
        )
        for m in re.finditer(label_pattern, header_text, re.IGNORECASE):
            candidate = m.group(1).strip()
            if self._is_valid_phone(candidate):
                self.logger.info(f"Regex found labeled phone: {candidate}")
                return candidate

        # --- Singapore: explicit +65 or 65 country code ---
        sg_pattern = r'(?<!\d)(\+?65[\s\-\.]?[689]\d{3}[\s\-\.]?\d{4})(?!\d)'
        m = re.search(sg_pattern, header_text)
        if m:
            candidate = m.group(1).strip()
            self.logger.info(f"Regex found SG phone: {candidate}")
            return candidate

        # --- Malaysia: explicit +60 or 60 country code ---
        my_pattern = r'(?<!\d)(\+?60[\s\-\.]?1[0-9][\s\-\.]?\d{3,4}[\s\-\.]?\d{4})(?!\d)'
        m = re.search(my_pattern, header_text)
        if m:
            candidate = m.group(1).strip()
            self.logger.info(f"Regex found MY phone (with code): {candidate}")
            return candidate

        # --- Singapore bare 8-digit number starting with 6, 8, or 9 ---
        sg_bare = r'(?<!\d)([689]\d{3}[\s\-\.]?\d{4})(?!\d)'
        for m in re.finditer(sg_bare, header_text):
            candidate = m.group(1).strip()
            # Avoid matching years or postal codes by checking context
            start = max(0, m.start() - 20)
            ctx = header_text[start:m.start()].lower()
            if any(w in ctx for w in ['postal', 'zip', 'nric', 'ic no', 'year', 'batch', 'order']):
                continue
            self.logger.info(f"Regex found SG bare phone: {candidate}")
            return candidate

        # --- Malaysia bare mobile starting with 01 (10-11 digits) ---
        my_bare = r'(?<!\d)(01[0-9][\s\-\.]?\d{3,4}[\s\-\.]?\d{4})(?!\d)'
        m = re.search(my_bare, header_text)
        if m:
            candidate = m.group(1).strip()
            self.logger.info(f"Regex found MY bare phone: {candidate}")
            return candidate

        return None

    def _is_valid_phone(self, phone: str) -> bool:
        """
        Validate a candidate phone string is a plausible SG or MY number.
        Accepts strings that contain 8-12 digits (after stripping formatting).
        """
        if not phone or not isinstance(phone, str):
            return False
        digits = re.sub(r'\D', '', phone)
        if len(digits) < 8 or len(digits) > 15:
            return False
        # Must look like a real phone — reject pure years/IDs
        if re.fullmatch(r'(19|20)\d{2}', digits):
            return False
        return True

    def _find_contact_area(self, text: str) -> str:
        """
        📝 Find the contact/header area of the resume (usually first 2000 chars).
        This area typically contains personal details like DOB, phone, email.
        """
        if not text:
            return ""

        # Look for common section headers that mark the end of contact area
        section_markers = [
            r'\n\s*(?:EXPERIENCE|EMPLOYMENT|WORK|EDUCATION|SKILLS|SUMMARY|OBJECTIVE|PROFILE)\s*(?:[:|\n])',
        ]

        for marker in section_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match and match.start() > 100:  # At least 100 chars of content
                return text[:match.start()]

        # Fallback: use first 2000 characters
        return text[:2000] if len(text) > 2000 else text

    def _extract_summary_regex(self, text: str) -> str:
        """
        📝 Extract professional summary/objective section VERBATIM.
        Does NOT paraphrase - extracts exactly as written.
        """
        summary_patterns = [
            r'(?:^|\n)\s*(?:PROFESSIONAL\s+)?(?:SUMMARY|PROFILE|OBJECTIVE|ABOUT\s*ME|CAREER\s+OBJECTIVE|PERSONAL\s+STATEMENT)\s*[:\n]\s*(.*?)(?=\n\s*(?:EXPERIENCE|EDUCATION|SKILLS|EMPLOYMENT|WORK|CERTIFICATION|QUALIFICATION)\s*(?:[:|\n]|$))',
        ]

        for pattern in summary_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                summary = match.group(1).strip()
                # Clean but preserve original wording
                summary = re.sub(r'\s+', ' ', summary)
                if len(summary) > 20:
                    self.logger.info(f"📝 Found summary: {len(summary)} chars")
                    return summary[:2000]  # Allow longer summaries

        return ""

    def _extract_certifications_regex(self, text: str) -> List[Dict[str, str]]:
        """
        🏅 Extract certifications/licenses VERBATIM.
        Captures: name, issuer, date, credential ID if present.
        """
        certifications = []

        # Find certifications section
        cert_patterns = [
            r'(?:^|\n)\s*(?:CERTIFICATIONS?|LICENSES?|PROFESSIONAL\s+(?:CERTIFICATIONS?|LICENSES?)|CREDENTIALS?|ACCREDITATIONS?)\s*[:\n]\s*(.*?)(?=\n\s*(?:EXPERIENCE|EDUCATION|SKILLS|EMPLOYMENT|WORK|PROJECTS?|AWARDS?|ACHIEVEMENTS?|LANGUAGES?|REFERENCES?)\s*(?:[:|\n]|$)|$)',
        ]

        cert_text = ""
        for pattern in cert_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                cert_text = match.group(1)
                break

        if not cert_text:
            return certifications

        # Split by common delimiters and extract each certification
        lines = re.split(r'\n|[•◗◗‹▪â– ►]', cert_text)

        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue

            # Skip section headers
            if re.match(r'^(certifications?|licenses?|credentials?)$', line, re.IGNORECASE):
                continue

            cert_entry = {"name": line, "issuer": "", "date": "", "credential_id": ""}

            # Try to extract date from the line
            date_match = re.search(r'(\d{4}|\w+\s+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})', line, re.IGNORECASE)
            if date_match:
                cert_entry["date"] = date_match.group(1)

            # Try to extract credential ID
            cred_match = re.search(r'(?:Credential\s*(?:ID)?|ID|License\s*(?:No\.?|#)?)\s*[:\-]?\s*([A-Z0-9\-]+)', line, re.IGNORECASE)
            if cred_match:
                cert_entry["credential_id"] = cred_match.group(1)

            # Try to extract issuer (common patterns)
            issuer_match = re.search(r'(?:by|from|issued\s+by|issuer)\s*[:\-]?\s*([A-Za-z\s&]+)', line, re.IGNORECASE)
            if issuer_match:
                cert_entry["issuer"] = issuer_match.group(1).strip()

            certifications.append(cert_entry)

        self.logger.info(f"🏅 Found {len(certifications)} certifications")
        return certifications

    def _extract_languages_regex(self, text: str) -> List[Dict[str, str]]:
        """
        🌐 Extract languages with proficiency levels VERBATIM.
        
        🧚"â™€ FAIRY CODEMOTHER'S FIX v3.0! 💅
        - Fixed: Now tries INLINE pattern FIRST (single-line entries)
        - Fixed: Added "Work Experience" as explicit terminator  
        - Fixed: Added sanity checks to prevent slurping entire resume
        """
        languages = []
        
        # =================================================================
        # 🎯 STEP 1: TRY INLINE PATTERN FIRST!
        # This catches "Language : English & Chinese" on a single line
        # This is the MOST COMMON format in Singapore resumes!
        # =================================================================
        inline_match = re.search(r'Languages?\s*[:\-]\s*([^\n]+)', text, re.IGNORECASE)
        if inline_match:
            lang_text = inline_match.group(1).strip()
            self.logger.info(f"🌐 Found inline language pattern: '{lang_text[:100]}'")
            
            # 🛡 SANITY CHECK 1: Language entries should be SHORT (< 200 chars)!
            if len(lang_text) > 200:
                self.logger.warning(f"⚠ Inline language too long ({len(lang_text)} chars) - truncating!")
                lang_text = lang_text[:100]
            
            # 🛡 SANITY CHECK 2: Should NOT contain job-related keywords
            job_keywords = ['experience', 'present', 'consultant', 'manager', 'pte ltd', 
                        'company', 'education', 'skills', 'achievements', 'financial',
                        'carried', 'conducted', 'managed', 'worked', 'responsible']
            if not any(kw in lang_text.lower() for kw in job_keywords):
                # This is a valid language entry!
                return self._parse_language_text(lang_text)
            else:
                self.logger.warning(f"⚠ Inline pattern captured job keywords - trying section pattern")
        
        # =================================================================
        # 🎯 STEP 2: Try section pattern (multi-line language sections)
        # FIXED: Added "WORK\s+EXPERIENCE" and "WORK\s+HISTORY" as terminators!
        # =================================================================
        lang_patterns = [
            # Pattern with EXPLICIT compound terminators!
            r'(?:^|\n)\s*(?:LANGUAGES?|LANGUAGE\s+SKILLS?|LINGUISTIC\s+SKILLS?)\s*[:\n]\s*(.*?)(?=\n\s*(?:WORK\s+EXPERIENCE|WORK\s+HISTORY|EMPLOYMENT\s+HISTORY|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|EDUCATION|SKILLS|EMPLOYMENT|PROJECTS?|CERTIFICATIONS?|AWARDS?|ACHIEVEMENTS?|REFERENCES?|HOBBIES?|INTERESTS?|PERSONAL\s+PARTICULARS?)\s*(?:[:|\n]|$)|$)',
        ]
        
        lang_text = ""
        for pattern in lang_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                lang_text = match.group(1).strip()
                self.logger.info(f"🌐 Found section language pattern: '{lang_text[:100]}...'")
                break
        
        if not lang_text:
            return languages
        
        # 🛡 SANITY CHECK 3: Section should be reasonably short
        if len(lang_text) > 500:
            self.logger.warning(f"⚠ Language section too long ({len(lang_text)} chars) - likely captured too much!")
            # Only keep first 200 characters
            lang_text = lang_text[:200]
        
        return self._parse_language_text(lang_text)
    
    def _extract_language_from_header(self, text: str) -> Optional[str]:
            """
            🗣 FAIRY CODEMOTHER'S HEADER LANGUAGE EXTRACTOR! 💅
            
            Singapore resumes often have language in the HEADER area (first 2000 chars)
            like: "Language : English & Chinese" - NOT in a dedicated section!
            
            This is like finding the VIP guest list at the door, honey! 🚪✨
            """
            # Only search in header area (first 2000 chars where contact info lives)
            header_text = text[:2000] if len(text) > 2000 else text
            
            # 🎯 Patterns for header-style language (with TAB or colon separator)
            header_lang_patterns = [
                # "Language : English & Chinese" or "Language\tEnglish & Chinese"
                r'Language[\s:\t]+([A-Za-z&,\s]+?)(?=\n|Date|Nationality|Address|HP|Phone|Email|NRIC|$)',
                # Simpler fallback
                r'Language\s*[:\t]\s*([^\n]+)',
            ]
            
            for pattern in header_lang_patterns:
                match = re.search(pattern, header_text, re.IGNORECASE)
                if match:
                    language = match.group(1).strip()
                    # Clean up trailing punctuation and normalize spaces
                    language = re.sub(r'[,;.\s]+$', '', language)
                    language = re.sub(r'\s+', ' ', language)
                    
                    # Validate it contains actual language names
                    if len(language) >= 3 and len(language) <= 100:
                        known_languages = ['english', 'chinese', 'mandarin', 'malay', 'tamil', 
                                        'hindi', 'japanese', 'korean', 'french', 'german', 
                                        'spanish', 'cantonese', 'hokkien', 'teochew']
                        if any(lang in language.lower() for lang in known_languages):
                            self.logger.info(f"🗣 Found language in header: {language}")
                            return language
            
            return None

    def _parse_language_text(self, lang_text: str) -> List[Dict[str, str]]:
        """
        🧚"â™€ Helper method to parse language text into structured entries
        ADD THIS AS A NEW METHOD IN AIExtractor class (after _extract_languages_regex)!
        """
        languages = []
        
        # Split by common delimiters
        items = re.split(r'[,;|]|\band\b|\b&\b|[•◗¦◗‹▪â– ►\n]', lang_text)
        
        for item in items:
            item = item.strip()
            if len(item) < 2 or len(item) > 50:  # Language names are 2-50 chars
                continue
            
            # 🛡 Skip section headers that might have slipped through
            if re.match(r'^(?:languages?|skills?|experience|education)$', item, re.IGNORECASE):
                continue
            
            # 🛡 Skip items that look like dates (job entries bleeding in!)
            if re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}', item, re.IGNORECASE):
                self.logger.debug(f"⭐ Skipping date entry in languages: {item[:50]}")
                continue
            
            # 🛡 Skip items that look like company names
            company_keywords = ['pte ltd', 'pvt ltd', 'company', 'consultant', 'manager', 
                            'engineer', 'executive', 'analyst', 'director']
            if any(kw in item.lower() for kw in company_keywords):
                self.logger.debug(f"⭐ Skipping company-like entry in languages: {item[:50]}")
                continue
            
            # 🛡 Skip items that are action verbs (job responsibilities)
            action_verbs = ['carried', 'conducted', 'managed', 'maintained', 'established',
                        'engaged', 'created', 'held', 'ensured', 'worked', 'achieved']
            first_word = item.split()[0].lower() if item.split() else ''
            if first_word in action_verbs:
                self.logger.debug(f"⭐ Skipping action verb in languages: {item[:50]}")
                continue
            
            lang_entry = {"language": item, "proficiency": ""}
            
            # Try to extract proficiency level
            prof_match = re.search(
                r'\(?(\s*(?:Native|Fluent|Advanced|Intermediate|Basic|Beginner|Professional|Working|Conversational|Elementary|Mother\s+Tongue|Bilingual|C2|C1|B2|B1|A2|A1)\s*)\)?', 
                item, re.IGNORECASE
            )
            if prof_match:
                lang_entry["proficiency"] = prof_match.group(1).strip()
                # Clean language name by removing proficiency
                lang_entry["language"] = re.sub(
                    r'\(?\s*(?:Native|Fluent|Advanced|Intermediate|Basic|Beginner|Professional|Working|Conversational|Elementary|Mother\s+Tongue|Bilingual|C2|C1|B2|B1|A2|A1)\s*\)?', 
                    '', item, flags=re.IGNORECASE
                ).strip(' -"“":')
            
            if lang_entry["language"] and len(lang_entry["language"]) >= 2:
                languages.append(lang_entry)
        
        self.logger.info(f"🌐 Found {len(languages)} languages")
        return languages

    def _extract_projects_regex(self, text: str) -> List[Dict[str, str]]:
        """
        🚀 Extract projects with descriptions VERBATIM.
        """
        projects = []

        # Find projects section
        proj_patterns = [
            r'(?:^|\n)\s*(?:PROJECTS?|PERSONAL\s+PROJECTS?|ACADEMIC\s+PROJECTS?|KEY\s+PROJECTS?|NOTABLE\s+PROJECTS?)\s*[:\n]\s*(.*?)(?=\n\s*(?:EXPERIENCE|EDUCATION|SKILLS|EMPLOYMENT|WORK|CERTIFICATIONS?|AWARDS?|ACHIEVEMENTS?|LANGUAGES?|REFERENCES?)\s*(?:[:|\n]|$)|$)',
        ]

        proj_text = ""
        for pattern in proj_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                proj_text = match.group(1)
                break

        if not proj_text:
            return projects

        # Try to detect project entries (usually start with project name or bullet)
        # Split by blank lines or project markers
        entries = re.split(r'\n\s*\n|\n(?=[A-Z][A-Za-z\s]+(?:Project|App|System|Platform|Website|Application))', proj_text)

        for entry in entries:
            entry = entry.strip()
            if len(entry) < 10:
                continue

            lines = entry.split('\n')
            project_name = lines[0].strip().strip('•◗◗‹▪â– ►-*')

            # Skip if it looks like a section header
            if re.match(r'^projects?$', project_name, re.IGNORECASE):
                continue

            description = ' '.join(line.strip().strip('•◗◗‹▪â– ►-*') for line in lines[1:] if line.strip())

            # Try to extract technologies used
            tech_match = re.search(r'(?:Technologies?|Tech\s*Stack|Built\s+with|Using)\s*[:\-]?\s*([^\n]+)', entry, re.IGNORECASE)
            technologies = tech_match.group(1).strip() if tech_match else ""

            # Try to extract date/duration
            date_match = re.search(r'(\d{4}(?:\s*[-–“]\s*\d{4})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})', entry, re.IGNORECASE)
            dates = date_match.group(1) if date_match else ""

            projects.append({
                "name": project_name[:200],
                "description": description[:1500] if description else project_name[:1500],
                "technologies": technologies[:300],
                "dates": dates
            })

        self.logger.info(f"🚀 Found {len(projects)} projects")
        return projects

    def _extract_achievements_regex(self, text: str) -> List[str]:
        """
        🏢 Extract achievements/awards/honors VERBATIM.
        """
        achievements = []

        # Find achievements section
        ach_patterns = [
            r'(?:^|\n)\s*(?:ACHIEVEMENTS?|AWARDS?|HONORS?|ACCOMPLISHMENTS?|RECOGNITIONS?|ACCOLADES?)\s*[:\n]\s*(.*?)(?=\n\s*(?:EXPERIENCE|EDUCATION|SKILLS|EMPLOYMENT|WORK|PROJECTS?|CERTIFICATIONS?|LANGUAGES?|REFERENCES?|HOBBIES?|INTERESTS?)\s*(?:[:|\n]|$)|$)',
        ]

        ach_text = ""
        for pattern in ach_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ach_text = match.group(1)
                break

        if not ach_text:
            return achievements

        # Split by bullets or newlines
        items = re.split(r'\n|[•◗◗‹▪â– ►]', ach_text)

        for item in items:
            item = item.strip().strip('-*')
            if len(item) < 5:
                continue

            # Skip section headers
            if re.match(r'^(achievements?|awards?|honors?)$', item, re.IGNORECASE):
                continue

            achievements.append(item)

        self.logger.info(f"🏢 Found {len(achievements)} achievements")
        return achievements[:50]  # Max 50 achievements

    def _extract_references_regex(self, text: str) -> List[Dict[str, str]]:
        """
        📝ž Extract references VERBATIM if present.
        """
        references = []

        # Find references section
        ref_patterns = [
            r'(?:^|\n)\s*(?:REFERENCES?|PROFESSIONAL\s+REFERENCES?)\s*[:\n]\s*(.*?)(?=\n\s*(?:EXPERIENCE|EDUCATION|SKILLS)\s*(?:[:|\n]|$)|$)',
        ]

        ref_text = ""
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ref_text = match.group(1)
                break

        if not ref_text:
            # Check for "Available upon request"
            if re.search(r'references?\s*(?:available\s*)?(?:upon|on)\s*request', text, re.IGNORECASE):
                return [{"name": "Available upon request", "title": "", "contact": ""}]
            return references

        # Try to parse reference entries
        entries = re.split(r'\n\s*\n', ref_text)

        for entry in entries:
            entry = entry.strip()
            if len(entry) < 10:
                continue

            lines = [l.strip() for l in entry.split('\n') if l.strip()]
            if not lines:
                continue

            ref_entry = {
                "name": lines[0] if lines else "",
                "title": "",
                "company": "",
                "contact": ""
            }

            for line in lines[1:]:
                # Try to identify what each line contains
                if '@' in line or re.search(r'\+?\d[\d\s\-()]+', line):
                    ref_entry["contact"] = line
                elif any(title in line.lower() for title in ['manager', 'director', 'supervisor', 'professor', 'dr.', 'ceo', 'cto', 'lead', 'head']):
                    ref_entry["title"] = line
                else:
                    ref_entry["company"] = line

            references.append(ref_entry)

        self.logger.info(f"📝ž Found {len(references)} references")
        return references

    def _extract_hobbies_regex(self, text: str) -> List[str]:
        """
        🎮 Extract hobbies/interests VERBATIM.
        """
        hobbies = []

        # Find hobbies/interests section
        hobby_patterns = [
            r'(?:^|\n)\s*(?:HOBBIES?|INTERESTS?|PERSONAL\s+INTERESTS?|LEISURE\s+ACTIVITIES?)\s*[:\n]\s*(.*?)(?=\n\s*(?:EXPERIENCE|EDUCATION|SKILLS|EMPLOYMENT|WORK|PROJECTS?|CERTIFICATIONS?|LANGUAGES?|REFERENCES?|ACHIEVEMENTS?)\s*(?:[:|\n]|$)|$)',
        ]

        hobby_text = ""
        for pattern in hobby_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                hobby_text = match.group(1)
                break

        if not hobby_text:
            return hobbies

        # Split by bullets, commas, or newlines
        items = re.split(r'[,\n]|[•◗◗‹▪â– ►]', hobby_text)

        for item in items:
            item = item.strip().strip('-*')
            if len(item) < 2:
                continue

            # Skip section headers
            if re.match(r'^(hobbies?|interests?)$', item, re.IGNORECASE):
                continue

            hobbies.append(item)

        self.logger.info(f"🎮 Found {len(hobbies)} hobbies/interests")
        return hobbies[:20]  # Max 20 hobbies

    def extract_header_fields(self, text: str) -> Dict[str, Optional[str]]:
        """
        🎯 PASS 1: Extract HEADER fields (Personal Info) - ENHANCED VERSION!
        
        🌟 FAIRY CODEMOTHER'S IMPROVEMENTS:
        - Better prompts with examples and context
        - Multi-format DOB extraction (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD)
        - Enhanced validation for each field
        - Fallback regex extraction for missing fields
        - Better handling of edge cases
        
        Think of this like finding the STAR of the show in a lineup! 🌟
        We need to spot them quickly but accurately!
        """
        if not self.available:
            return {}

        # 📍 STEP 1: Try regex extraction for email and phone (most reliable!)
        regex_email = self._extract_email_regex(text)
        regex_phone = self._extract_phone_regex(text)

        # 📍 STEP 2: Search a LARGER text sample (header info can be anywhere!)
        # Like searching for your misplaced wig - check EVERYWHERE! 💁‍♀️
        text_sample = text[:10000] if len(text) > 10000 else text

        # 🎭 STEP 3: Create DETAILED prompt with EXAMPLES and CONTEXT
        # This is like giving directions with LANDMARKS, honey! 🗺️
        prompt = f"""You are a precise resume data parser specializing in extracting contact information.

    **IMPORTANT INSTRUCTIONS:**
    1. Search the ENTIRE resume header and contact section carefully
    2. Extract data EXACTLY as written - do not modify or format
    3. If a field is not found, return null (not empty string, not "N/A")
    4. Phone numbers may include country codes (+65, +60, etc.)
    5. Location can be "City, Country" or just "City"

    **REQUIRED FIELDS TO EXTRACT:**
    - name: Full name (first and last name together)
    - email: Email address (username@domain.com format)
    - phone: Phone number (with country code if present). SG phones are 8 digits.
    - nationality: Nationality or citizenship if mentioned (e.g. "Singaporean", "Malaysian")
    - location: Current city and/or country
    - language: Languages spoken (e.g. "English & Chinese"). Often appears as "Language : English & Chinese" in the header — DO NOT skip this!
    - linkedin: LinkedIn profile URL (full or partial)
    - github: GitHub profile URL if present
    - website: Personal website URL if present

    **EXAMPLES OF GOOD EXTRACTION:**

    Example Resume 1:
    "John Michael Doe
    Email: john.doe@email.com
    Phone: +65 9123 4567
    Location: Singapore
    LinkedIn: linkedin.com/in/johndoe"

    Correct Output:
    {{
    "name": "John Michael Doe",
    "email": "john.doe@email.com",
    "phone": "+65 9123 4567",
    "nationality": null,
    "location": "Singapore",
    "language": null,
    "linkedin": "linkedin.com/in/johndoe",
    "github": null,
    "website": null
    }}

    Example Resume 2:
    "Sarah Chen
    sarah.chen@company.com | (+65) 8765-4321
    Nationality: Singaporean
    Address: 123 Main Street, Singapore 123456"

    Correct Output:
    {{
    "name": "Sarah Chen",
    "email": "sarah.chen@company.com",
    "phone": "+65 8765-4321",
    "nationality": "Singaporean",
    "location": "Singapore",
    "language": null,
    "linkedin": null,
    "github": null,
    "website": null
    }}

    Example Resume 3 (Singapore format with Language in header):
    "Nurul Ain Binte Ismail
    Nationality	: Singaporean
    Language	: English & Chinese
    HP	: 91234567
    Email	: nurul.ain@email.com
    Address	: Blk 123 Ang Mo Kio Ave 4, Singapore 560123"

    Correct Output:
    {{
    "name": "Nurul Ain Binte Ismail",
    "email": "nurul.ain@email.com",
    "phone": "91234567",
    "nationality": "Singaporean",
    "location": "Blk 123 Ang Mo Kio Ave 4, Singapore 560123",
    "language": "English & Chinese",
    "linkedin": null,
    "github": null,
    "website": null
    }}

    **NOW EXTRACT FROM THIS RESUME:**

    {text_sample}

    **OUTPUT ONLY VALID JSON - NO MARKDOWN, NO EXPLANATIONS:**"""

        # 📍 STEP 4: Call the AI with schema-enforced structured output
        result = self._call_ollama(prompt, schema=HEADER_SCHEMA)

        # 📍 STEP 5: VALIDATE and ENHANCE the extraction!
        # This is the QUALITY CONTROL stage, sweetie! 💅
        
        # ✅ Validate email - use regex fallback if AI failed
        ai_email = result.get('email')
        if not self._is_valid_email(ai_email):
            if ai_email:
                self.logger.warning(f"⚠️ AI returned invalid email: '{str(ai_email)[:50]}...' - using regex fallback")
            if regex_email:
                result['email'] = regex_email
                self.logger.info(f"✅ Using regex-extracted email: {regex_email}")
            else:
                result['email'] = None
                self.logger.warning("⚠️ No valid email found in resume")

        # ✅ Validate phone - use regex fallback if AI failed or returned garbage
        ai_phone = result.get('phone')
        if not self._is_valid_phone(ai_phone):
            if ai_phone:
                self.logger.warning(f"⚠️ AI returned invalid phone: '{str(ai_phone)[:40]}' - using regex fallback")
            if regex_phone:
                result['phone'] = standardize_phone_number(regex_phone)
                self.logger.info(f"✅ Using regex-extracted phone: {result['phone']}")
            else:
                result['phone'] = None
                self.logger.warning("⚠️ No valid SG/MY phone found in resume")
        else:
            result['phone'] = standardize_phone_number(ai_phone)
        
        # ✅ Clean name - remove titles and suffixes
        if result.get('name'):
            result['name'] = self._clean_name(result['name'])
        
        # ✅ Validate nationality - check it's actually a nationality
        if result.get('nationality'):
            nationality = result['nationality'].strip()
            # If it looks like a location or address, it's probably wrong!
            if any(word in nationality.lower() for word in ['street', 'road', 'avenue', 'city', 'block', '#']):
                self.logger.warning(f"⚠️ Nationality looks like an address: {nationality[:30]}")
                result['nationality'] = None
            # Check length - nationalities are typically 5-20 characters
            elif len(nationality) < 3 or len(nationality) > 30:
                self.logger.warning(f"⚠️ Nationality length suspicious: {nationality[:30]}")
                result['nationality'] = None
        
        self.logger.info(f"✅ Header extraction complete! Found: {', '.join([k for k, v in result.items() if v])}")

        return result

    def extract_deep_fields(self, text: str) -> Dict[str, Any]:
        """
        🚀 MANUAL-FIRST EXTRACTION with AI assistance only for disambiguation
        Because AI can't be trusted with structure, honey! 💅
        """
        if not self.available:
            result = self._pure_manual_extraction(text)
            if self._ner_enabled:
                result = self._enhance_with_bert_ner(result, text)
            return result

        self.logger.info("🎭 Starting MANUAL-FIRST extraction...")

        # ALWAYS start with manual extraction
        result = self._pure_manual_extraction(text)

        # Use AI section detection to improve BERT NER accuracy
        ai_sections: Dict[str, str] = {}
        if self._ner_enabled:
            self.logger.info("🔍 Running AI section detection for BERT NER...")
            ai_sections = self.detect_sections_ai(text)

        # BERT NER: fill gaps that regex missed, using AI sections where available
        if self._ner_enabled:
            result = self._enhance_with_bert_ner(result, text, ai_sections=ai_sections)

        # ONLY use AI to enhance/disambiguate specific fields if needed
        if self._needs_ai_enhancement(result):
            self.logger.info("🤖 Using AI for skill categorization only...")
            result = self._enhance_skills_with_ai(result, text)

        self.logger.info(f"✅ Extraction complete: {len(result.get('hard_skills', []))} hard skills, {len(result.get('working_experience', []))} jobs")

        return result

    def _enhance_with_bert_ner(
        self,
        result: Dict[str, Any],
        full_text: str,
        ai_sections: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Use BERT NER (NEREngine) to fill in fields that regex extraction missed.
        Only adds to empty / short fields — never overwrites regex results.
        Uses ai_sections (from detect_sections_ai) when available for better context.
        """
        if not self._ner_enabled or self._ner_engine is None:
            return result

        self.logger.info("🧬 Running BERT NER enhancement...")
        ai_sections = ai_sections or {}

        # ── Work Experience ────────────────────────────────────────────────
        if not result.get('working_experience'):
            exp_text = ai_sections.get('experience') or full_text
            entities = self._ner_engine.get_entities_by_type(
                exp_text, ['Companies worked at', 'Designation', 'Years of Experience']
            )
            companies   = entities.get('Companies worked at', [])
            titles      = entities.get('Designation', [])
            years       = entities.get('Years of Experience', [])
            if companies or titles:
                n = max(len(companies), len(titles), 1)
                for i in range(n):
                    # Use the SAME internal schema as the regex / date-first
                    # extractors ({company, role, dates, description}) so the
                    # JSON exporter (format_result_as_export_json) maps it to
                    # {company, title, from, to, responsibility[]} correctly.
                    entry: Dict[str, Optional[str]] = {
                        'company':     companies[i] if i < len(companies) else None,
                        'role':        titles[i]    if i < len(titles)    else None,
                        'dates':       years[i]     if i < len(years)     else "",
                        'description': '',
                    }
                    result['working_experience'].append(entry)
                self.logger.info(f"🧬 NER added {n} experience entries")

        # ── Education ──────────────────────────────────────────────────────
        if not result.get('education'):
            edu_text = ai_sections.get('education') or full_text
            entities = self._ner_engine.get_entities_by_type(
                edu_text, ['College Name', 'Degree', 'Graduation Year']
            )
            colleges = entities.get('College Name', [])
            degrees  = entities.get('Degree', [])
            years    = entities.get('Graduation Year', [])
            if colleges or degrees:
                n = max(len(colleges), len(degrees), 1)
                for i in range(n):
                    # Match the SAME internal schema the regex education
                    # extractors use ({institution, degree, dates}) so the
                    # JSON/CSV exporters read it correctly. Previously this
                    # used qualification/graduation_year, which the exporter
                    # ignores -> blank major/Degree on NER-filled education.
                    entry = {
                        'institution': colleges[i] if i < len(colleges) else None,
                        'degree':      degrees[i]  if i < len(degrees)  else None,
                        'dates':       years[i]    if i < len(years)    else "",
                    }
                    result['education'].append(entry)
                self.logger.info(f"🧬 NER added {n} education entries")

        # ── Skills ─────────────────────────────────────────────────────────
        if not result.get('hard_skills'):
            skill_text = ai_sections.get('skills') or full_text
            entities = self._ner_engine.get_entities_by_type(skill_text, ['Skills'])
            ner_skills = entities.get('Skills', [])
            if ner_skills:
                result['hard_skills'] = ner_skills
                self.logger.info(f"🧬 NER added {len(ner_skills)} skills")

        return result

    def _pure_manual_extraction(self, text: str) -> Dict[str, Any]:
        """
        🛠 PURE MANUAL EXTRACTION - No AI involved!
        This is the BACKBONE, honey! 🚪

        COMPREHENSIVE EXTRACTION - Captures ALL resume sections VERBATIM:
        - Skills (hard & soft)
        - Work Experience (with full descriptions)
        - Education
        - Summary/Objective
        - Certifications
        - Languages
        - Projects
        - Achievements/Awards
        - References
        - Hobbies/Interests
        """
        self.logger.info("📧 Running COMPREHENSIVE manual extraction...")

        result = {
            "hard_skills": [],
            "soft_skills": [],
            "working_experience": [],
            "education": [],
            "summary": "",
            "certifications": [],
            "languages": [],
            "projects": [],
            "achievements": [],
            "references": [],
            "hobbies": []
        }

        # ===== EXTRACT SUMMARY/OBJECTIVE =====
        result['summary'] = self._extract_summary_regex(text)

        # ===== EXTRACT SKILLS =====
        skills_result = self._extract_skills_regex(text)
        result['hard_skills'] = skills_result['hard_skills']
        result['soft_skills'] = skills_result['soft_skills']

        # ===== EXTRACT EXPERIENCE =====
        result['working_experience'] = self._extract_experience_regex(text)

        # ===== EXTRACT EDUCATION =====
        result['education'] = self._extract_education_regex(text)

        # ===== EXTRACT CERTIFICATIONS =====
        result['certifications'] = self._extract_certifications_regex(text)

        # ===== EXTRACT LANGUAGES =====
        # 🏆• FAIRY CODEMOTHER'S FIX: Try header extraction FIRST (Singapore-style)!
        header_language = self._extract_language_from_header(text)
        if header_language:
            result['languages'] = [{"language": header_language, "proficiency": ""}]
            self.logger.info(f"🗣 Using header language: {header_language}")
        else:
            result['languages'] = self._extract_languages_regex(text)

        # ===== EXTRACT PROJECTS =====
        result['projects'] = self._extract_projects_regex(text)

        # ===== EXTRACT ACHIEVEMENTS =====
        result['achievements'] = self._extract_achievements_regex(text)

        # ===== EXTRACT REFERENCES =====
        result['references'] = self._extract_references_regex(text)

        # ===== EXTRACT HOBBIES =====
        result['hobbies'] = self._extract_hobbies_regex(text)

        self.logger.info(f"📝Š Comprehensive extraction complete:")
        self.logger.info(f"   - Summary: {len(result['summary'])} chars")
        self.logger.info(f"   - Hard Skills: {len(result['hard_skills'])}")
        self.logger.info(f"   - Soft Skills: {len(result['soft_skills'])}")
        self.logger.info(f"   - Experience: {len(result['working_experience'])} entries")
        self.logger.info(f"   - Education: {len(result['education'])} entries")
        self.logger.info(f"   - Certifications: {len(result['certifications'])}")
        self.logger.info(f"   - Languages: {len(result['languages'])}")
        self.logger.info(f"   - Projects: {len(result['projects'])}")
        self.logger.info(f"   - Achievements: {len(result['achievements'])}")

        return result

    def _expand_cmfas_certifications(self, text: str) -> str:
        """
        🏅 CMFAS Certification Expander - Singapore Financial Industry Certifications

        CMFAS (Capital Markets and Financial Advisory Services) modules are listed as:
        "CMFAS M5 | M8 | M8A | M9 | M9A | HI | CGI "“ BCP | ComGI | PGI"

        This function expands abbreviated module lists so each module retains the CMFAS prefix:
        "CMFAS M5 | CMFAS M8 | CMFAS M8A | CMFAS M9 | CMFAS M9A | CMFAS HI | ..."

        Also handles standalone CMFAS modules on separate lines:
        "CMFAS M5, M8, M8A
         HI
         CGI "“ BCP, ComGI, PGI"
        """
        if not text:
            return text

        # Known CMFAS module codes (Singapore MAS certifications)
        # M1-M10: Various advisory modules
        # HI: Health Insurance, CGI: Collective Investment, BCP: Bundled Coverage Product
        # ComGI: Commercial General Insurance, PGI: Personal General Insurance
        # Also includes compound forms like "CGI "“ BCP"
        cmfas_module_single = r'M\d+[A-Z]?|HI|CGI|BCP|ComGI|PGI|PHI|CLI|BCPP'
        # Compound module (e.g., "CGI "“ BCP")
        cmfas_module = r'(?:' + cmfas_module_single + r')(?:\s*["“\-]\s*(?:' + cmfas_module_single + r'))?'

        # Pattern to find CMFAS followed by a list of modules separated by | or ,
        # Greedily capture everything that looks like CMFAS modules
        cmfas_pattern = r'CMFAS\s+(' + cmfas_module + r')(\s*[|,]\s*(' + cmfas_module + r'))*'

        def expand_cmfas_match(match):
            """Expand a CMFAS certification list by adding CMFAS prefix to each module"""
            full_match = match.group(0)
            # Split by | or , and expand each module
            parts = re.split(r'\s*[|,]\s*', full_match)
            expanded = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                # If it already has CMFAS, keep it; otherwise prepend CMFAS
                if part.upper().startswith('CMFAS'):
                    expanded.append(part)
                else:
                    # Handle compound modules like "CGI "“ BCP"
                    expanded.append(f"CMFAS {part}")
            return ' | '.join(expanded)

        result = re.sub(cmfas_pattern, expand_cmfas_match, text, flags=re.IGNORECASE)

        # PHASE 2: Handle standalone CMFAS modules on separate lines
        # These appear in "Additional Qualifications" or similar sections without CMFAS prefix
        # Pattern: Standalone line containing ONLY CMFAS module codes (no other text)
        # e.g., "HI" or "CGI "“ BCP, ComGI, PGI"
        standalone_modules = ['ComGI', 'PGI', 'PHI', 'CLI', 'BCPP', 'HI', 'CGI', 'BCP']  # Longer names first

        # Only apply this if we found CMFAS somewhere in the text (indicates this is a financial resume)
        if 'CMFAS' in result.upper():
            for module in standalone_modules:
                # Add CMFAS prefix to standalone modules (word boundary match)
                pattern = r'\b' + re.escape(module) + r'\b'
                result = re.sub(pattern, f'CMFAS {module}', result)

            # Clean up: Remove double CMFAS prefixes that may have been added
            # "CMFAS CMFAS X" -> "CMFAS X"
            result = re.sub(r'CMFAS\s+CMFAS\s+', 'CMFAS ', result)

            # Clean up: Fix compound modules like "CMFAS CGI "“ CMFAS BCP" -> "CMFAS CGI "“ BCP"
            # The second module in a compound shouldn't have CMFAS prefix
            modules_alt = '|'.join(re.escape(m) for m in standalone_modules)
            compound_fix = r'(CMFAS\s+(?:' + modules_alt + r'))(\s*["“\-]\s*)CMFAS\s+(' + modules_alt + r')'
            result = re.sub(compound_fix, r'\1\2\3', result)

        if result != text:
            self.logger.info("🏅 Expanded CMFAS certifications for proper extraction")

        return result

    def _extract_skills_from_nested_categories(self, text: str) -> Dict[str, List[str]]:
        """
        🎯 Extract skills from NESTED CATEGORIES common in Singapore resumes!
        
        Handles format like:
        Skills & Abilities
        Organisation Skills
        - Prepared and hosted for seminars...
        - Organised for internal audit...
        
        Interpersonal Skills
        - Applied emotional competence...
        """
        hard_skills = []
        soft_skills = []
        
        # Find Skills section with nested categories
        skills_section_pattern = r'(?:Skills?\s*(?:&|and)?\s*Abilities?|Skills?)\s*\n(.*?)(?=\n\s*(?:Additional|Certification|Education|Experience|Achievement|Award|Reference|$))'
        
        match = re.search(skills_section_pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            return {"hard_skills": [], "soft_skills": []}
        
        skills_text = match.group(1)
        
        # 🎯 Detect category headers (end with "Skills")
        category_pattern = r'^([A-Z][A-Za-z]+(?:\s+[A-Z]?[a-z]+)*\s+Skills?)\s*$'
        
        current_category = None
        category_items = []
        
        for line in skills_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a category header
            if re.match(category_pattern, line, re.IGNORECASE):
                # Save previous category's items
                if current_category and category_items:
                    # Determine if soft or hard skill category
                    if any(soft_kw in current_category.lower() for soft_kw in 
                        ['organisation', 'organization', 'interpersonal', 'leadership', 
                            'communication', 'teamwork', 'management', 'soft']):
                        soft_skills.append(current_category)  # Add category name
                        soft_skills.extend(category_items)
                    else:
                        hard_skills.append(current_category)
                        hard_skills.extend(category_items)
                
                current_category = line
                category_items = []
            else:
                # This is a skill item under current category
                if len(line) > 10:  # Skip very short lines
                    # Clean bullet markers
                    cleaned = re.sub(r'^[\*\-•◗◗‹▪â– ►]\s*', '', line)
                    if cleaned:
                        category_items.append(cleaned)
        
        # Don't forget the last category!
        if current_category and category_items:
            if any(soft_kw in current_category.lower() for soft_kw in 
                ['organisation', 'organization', 'interpersonal', 'leadership', 
                    'communication', 'teamwork', 'management', 'soft']):
                soft_skills.append(current_category)
                soft_skills.extend(category_items)
            else:
                hard_skills.append(current_category)
                hard_skills.extend(category_items)
        
        self.logger.info(f"📝Š Nested skills: {len(hard_skills)} hard, {len(soft_skills)} soft")
        
        return {"hard_skills": hard_skills, "soft_skills": soft_skills}

    def _extract_skills_regex(self, text: str) -> Dict[str, list]:
        """
        🎯 FAIRY CODEMOTHER'S ULTIMATE SKILLS EXTRACTOR v3.0! 💅
        
        Enhanced to capture skills from ALL job types:
        - Software Engineers & Full-Stack Developers
        - QA Engineers & Testers
        - Cybersecurity & Pentesters
        - Data Analysts & Scientists
        - AI/ML Engineers
        - DevOps & Cloud Engineers
        - Business Analysts
        - Product & Project Managers
        - HR, Sales, Marketing & Admin
        
        This queen recognizes skills across ALL industries, darling! 👉
        """
        hard_skills = []
        soft_skills = []

        # 🎭 IMPROVED: Section terminators must be at LINE START to avoid false matches
        # Define terminator keywords for the skills section
        # 💅 FAIRY CODEMOTHER'S FIX: "ADDITIONAL SKILLS" should NOT be a terminator!
        # It was killing extraction of the second skills section! 
        # Only "ADDITIONAL QUALIFICATIONS" (certifications) should terminate.
        terminator_keywords = [
            r'(?:WORK\s+)?EXPERIENCE[S]?', 'EMPLOYMENT', r'WORK\s+HISTORY', 'EDUCATION',
            'CERTIFICATION[S]?',
            'AWARD[S]?', 'PROJECT[S]?', 'REFERENCE[S]?', 'ACHIEVEMENT[S]?', 'PUBLICATION[S]?',
            'LANGUAGE[S]?',
            'HOBBIES?', 'INTEREST[S]?', 'SUMMARY', 'OBJECTIVE', 'PROFILE', 'COMMENDATION[S]?',
            r'PROFESSIONAL\s+DEVELOPMENT', 'CAREER', 'CO-?CURRICULAR',
            r'ADDITIONAL\s+QUALIFICATIONS?',  # ← Certifications section = stop
            # 🚫 REMOVED: "ADDITIONAL" alone — was killing "ADDITIONAL SKILLS" section!
        ]
        
        # 1. Section headers pattern
        headers_pattern = r'\n\s*(?:' + '|'.join(terminator_keywords) + r')\s*(?:[:|\n]|$)'
        
        # 2. Date pattern for start of a new job entry
        date_pattern = r'\n\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s+to'
        
        # Combine all parts into a single lookahead.
        section_end = f'(?=({headers_pattern})|({date_pattern})|$)'

        # 🏆• ENHANCED v6.0: More patterns to catch skills in various resume formats!
        skills_patterns = [
            # Pattern 1: Generic "SKILLS" section (most common)
            (r'(?:^|\n)\s*(?:#*\s*\**)?SKILLS?\s*[:\n]\s*(.*?)' + section_end, 'skills'),
            # Pattern 2: "TECHNICAL SKILLS" or "KEY SKILLS" etc.
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:TECHNICAL|KEY|CORE|PROFESSIONAL|RELEVANT|TRANSFERABLE|OTHER|ADDITIONAL)\s+SKILLS?\s*[:\n]\s*(.*?)' + section_end, 'qualified_skills'),
            # Pattern 3: "CORE COMPETENCIES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:CORE|KEY)\s+COMPETENC(?:IES|Y)\s*[:\n]\s*(.*?)' + section_end, 'competencies'),
            # Pattern 4: "SOFTWARE" or "TOOLS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:SOFTWARE|TOOLS?\s*(?:&|AND)?\s*(?:TECHNOLOGIES|TECH)?)\s*[:\n]\s*(.*?)' + section_end, 'software'),
            # Pattern 5: "SKILLS & ABILITIES" or similar
            (r'(?:^|\n)\s*(?:#*\s*\**)?SKILLS?\s*(?:&|AND)\s*(?:ABILITIES|EXPERTISE|COMPETENCIES)\s*[:\n]\s*(.*?)' + section_end, 'skills_abilities'),
            # Pattern 6: "AREAS OF EXPERTISE" or just "EXPERTISE"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:AREAS?\s+OF\s+)?EXPERTISE\s*[:\n]\s*(.*?)' + section_end, 'expertise'),
            # Pattern 7: "ADDITIONAL QUALIFICATIONS" (catches certifications too!)
            (r'(?:^|\n)\s*(?:#*\s*\**)?ADDITIONAL\s+QUALIFICATIONS?\s*[:\n]\s*(.*?)' + section_end, 'additional_quals'),
            # Pattern 8: "TECHNOLOGIES" standalone
            (r'(?:^|\n)\s*(?:#*\s*\**)?TECHNOLOGIES\s*[:\n]\s*(.*?)' + section_end, 'technologies'),
            # Pattern 9: "PROGRAMMING LANGUAGES" or "LANGUAGES" (tech context)
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:PROGRAMMING\s+)?LANGUAGES?\s*[:\n]\s*(.*?)' + section_end, 'languages'),
            # Pattern 10: "FRAMEWORKS" or "LIBRARIES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:FRAMEWORKS?|LIBRARIES)\s*[:\n]\s*(.*?)' + section_end, 'frameworks'),
            # Pattern 11: "CERTIFICATIONS" (often contains skill keywords)
            (r'(?:^|\n)\s*(?:#*\s*\**)?CERTIFICATIONS?\s*[:\n]\s*(.*?)' + section_end, 'certifications'),
            # Pattern 12: "PROFESSIONAL QUALIFICATIONS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?PROFESSIONAL\s+QUALIFICATIONS?\s*[:\n]\s*(.*?)' + section_end, 'pro_quals'),
            # 🏆• Pattern 13: "TECH STACK" (common in developer resumes)
            (r'(?:^|\n)\s*(?:#*\s*\**)?TECH(?:NICAL)?\s+STACK\s*[:\n]\s*(.*?)' + section_end, 'tech_stack'),
            # 🏆• Pattern 14: "PROFICIENCIES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:TECHNICAL\s+)?PROFICIENC(?:IES|Y)\s*[:\n]\s*(.*?)' + section_end, 'proficiencies'),
            # 🏆• Pattern 15: "CAPABILITIES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:CORE\s+)?CAPABILITIES\s*[:\n]\s*(.*?)' + section_end, 'capabilities'),
            # 🏆• Pattern 16: "STRENGTHS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:KEY\s+)?STRENGTHS\s*[:\n]\s*(.*?)' + section_end, 'strengths'),
            # 🏆• Pattern 17: "SPECIALIZATIONS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:AREAS?\s+OF\s+)?SPECIALIZ(?:ATION|ATIONS?)\s*[:\n]\s*(.*?)' + section_end, 'specializations'),
            # 🏆• Pattern 18: "DOMAINS" or "DOMAIN KNOWLEDGE"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:DOMAIN\s+)?(?:KNOWLEDGE|EXPERTISE|DOMAINS?)\s*[:\n]\s*(.*?)' + section_end, 'domains'),
            # 🏆• Pattern 19: Singapore-specific - "LICENSES" or "LICENCES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?LICEN[CS]ES?\s*[:\n]\s*(.*?)' + section_end, 'licenses'),
            # 🏆• Pattern 20: "DATABASES" section
            (r'(?:^|\n)\s*(?:#*\s*\**)?DATABASES?\s*[:\n]\s*(.*?)' + section_end, 'databases'),
            # 🏆• Pattern 21: "CLOUD" or "CLOUD PLATFORMS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?CLOUD(?:\s+PLATFORMS?)?\s*[:\n]\s*(.*?)' + section_end, 'cloud'),
            # 🏆• Pattern 22: "OPERATING SYSTEMS" or "OS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:OPERATING\s+SYSTEMS?|OS)\s*[:\n]\s*(.*?)' + section_end, 'os'),
            # 💅 Pattern 23: "ADDITIONAL SKILLS" — the forgotten sister section!
            (r'(?:^|\n)\s*(?:#*\s*\**)?ADDITIONAL\s+SKILLS?\s*[:\n]\s*(.*?)' + section_end, 'additional_skills'),
            # 💅 Pattern 24: "OTHER SKILLS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?OTHER\s+SKILLS?\s*[:\n]\s*(.*?)' + section_end, 'other_skills'),
        ]

        all_skills_text = []
        patterns_matched = []

        for pattern, pattern_name in skills_patterns:
            try:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
                for match in matches:
                    captured = match.group(1).strip()
                    if captured and len(captured) > 5:
                        all_skills_text.append(captured)
                        patterns_matched.append(pattern_name)
                        self.logger.debug(f"🎯 Found skills via '{pattern_name}': {len(captured)} chars")
            except re.error as e:
                self.logger.warning(f"⚠ Regex error in skills pattern '{pattern_name}': {e}")
                continue

        # Combine all captured skills sections
        skills_text = "\n".join(all_skills_text)

        if not skills_text:
            self.logger.warning("⚠ No skills section found")
            return {"hard_skills": [], "soft_skills": []}

        self.logger.info(f"📝Š Found skills via patterns: {patterns_matched}")
        
        # 🧹 CLEAN THE TEXT!
        # Remove markdown formatting (bold, italic, headers)
        skills_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', skills_text)  # **bold**
        skills_text = re.sub(r'\*([^*\n]+)\*', r'\1', skills_text)    # *italic*
        skills_text = re.sub(r'^#+\s*', '', skills_text, flags=re.MULTILINE)  # # headers
        
        # Remove bullet characters
        skills_text = re.sub(r'[•◗◗‹◗¦▪▫â– □►▸"£âƒ→·∙â‹…▶▷◗†◗‡â˜…☆âœ“✔]', '\n', skills_text)
        skills_text = re.sub(r'^\s*[-–“"”*]\s*', '\n', skills_text, flags=re.MULTILINE)

        # 🎭 CMFAS CERTIFICATION EXPANSION - Singapore financial certifications
        skills_text = self._expand_cmfas_certifications(skills_text)

        # Split by newlines, commas, semicolons, AND pipes
        raw_skills = re.split(r'[\n,;|]+', skills_text)
        
        # Clean each skill
        all_skills = []

        # 🎭 FAIRY CODEMOTHER'S SENTENCE DETECTOR!
        # Words that indicate a SENTENCE, not a skill!
        sentence_start_words = {
            'prepared', 'organised', 'organized', 'applied', 'managed', 'developed',
            'created', 'led', 'built', 'designed', 'implemented', 'executed', 'worked',
            'responsible', 'achieved', 'increased', 'decreased', 'improved', 'reduced',
            'collaborated', 'coordinated', 'analyzed', 'analysed', 'maintained', 'provided',
            'ensured', 'utilized', 'utilised', 'demonstrated', 'conducted', 'supported',
            'delivered', 'performed', 'generated', 'resolved', 'diagnosed', 'authored',
            'assisted', 'established', 'trained', 'hosted', 'educated', 'facilitated',
            'oversaw', 'spearheaded', 'streamlined', 'optimized', 'optimised', 'initiated',
            'launched', 'negotiated', 'presented', 'reviewed', 'supervised', 'mentored',
            'a', 'an', 'the', 'i', 'we', 'they', 'he', 'she', 'it', 'my', 'our',
            'knowledge', 'experienced', 'instrumental', 'sound', 'ability', 'able'
        }

        # 🎭 Section headers to skip
        skip_headers = {
            'skills', 'experience', 'education', 'summary', 'objective', 'profile',
            'additional qualifications', 'qualifications', 'certifications',
            'additional skills', 'other skills', 'skills & abilities',
            'skills and abilities', 'core competencies', 'key skills',
            'professional skills', 'personal skills', 'hard skills', 'soft skills',
            'transferable skills', 'relevant skills', 'technical skills',
            'areas of expertise', 'competencies', 'abilities', 'expertise',
            'organisation skills', 'organization skills', 'interpersonal skills',
            'leadership skills', 'communication skills', 'technical', 'tools',
            'frameworks', 'languages', 'technologies', '& abilities', 'and abilities'
        }

        for skill in raw_skills:
            cleaned = skill.strip()
            # Remove leading numbers/bullets
            cleaned = re.sub(r'^\d+[\.)]\s*', '', cleaned)
            # Remove excessive whitespace
            cleaned = ' '.join(cleaned.split())
            # Remove trailing colons (often headers)
            cleaned = cleaned.rstrip(':')

            # 🎭 FAIRY CODEMOTHER'S SKILL POLLUTION FILTER v3.0!
            # Skip content that's clearly NOT a skill!
            
            # Skip if it contains date patterns (job entries bleeding in!)
            if re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}', cleaned, re.IGNORECASE):
                self.logger.debug(f"â­ Skipping date-containing entry: {cleaned[:50]}")
                continue
            
            # Skip if it contains company indicators
            company_indicators = ['pte ltd', 'pvt ltd', 'inc.', 'corp.', 'llc', 'company', 'consultancy']
            if any(ind in cleaned.lower() for ind in company_indicators):
                self.logger.debug(f"â­ Skipping company-like entry: {cleaned[:50]}")
                continue
            
            # Skip if it looks like a job title + company combination
            if re.search(r'(?:Consultant|Manager|Engineer|Developer|Assistant|Executive|Officer),\s+[A-Z]', cleaned):
                self.logger.debug(f"â­ Skipping job+company entry: {cleaned[:50]}")
                continue
            
            # Skip section headers that might have slipped through
            section_headers = ['work experience', 'education', 'achievements', 'co-curricular']
            if cleaned.lower() in section_headers:
                continue

            all_skills.append(cleaned)
        
        # 🎭 CATEGORIZE: Hard vs Soft skills using COMPREHENSIVE keyword lists!
        for skill in all_skills:  # No limit - capture ALL skills!
            skill_lower = skill.lower()
            
            # Check categorization
            is_technical = self._is_technical_skill(skill_lower)
            is_soft = self._is_soft_skill(skill_lower)
            
            if is_technical:
                hard_skills.append(skill)
            elif is_soft:
                soft_skills.append(skill)
            else:
                # Default categorization based on patterns
                # Acronyms, numbers, dots, slashes, parentheses = technical
                if re.search(r'[A-Z]{2,}|[0-9]|\.|/|\(|\)', skill):
                    hard_skills.append(skill)
                else:
                    soft_skills.append(skill)

        # Remove duplicates while preserving order
        hard_skills = list(dict.fromkeys(hard_skills))
        soft_skills = list(dict.fromkeys(soft_skills))

        self.logger.info(f"📝Š Found {len(hard_skills)} hard skills, {len(soft_skills)} soft skills")

        # 🏆• If we found very few skills, try nested category extraction
        if len(hard_skills) + len(soft_skills) < 5:
            self.logger.info("📝Š Few skills found, trying nested category extraction...")
            nested_result = self._extract_skills_from_nested_categories(text)
            if nested_result['hard_skills'] or nested_result['soft_skills']:
                return nested_result

        return {
            "hard_skills": hard_skills,  # No limit - capture all!
            "soft_skills": soft_skills   # No limit - capture all!
        }

    def _get_known_skill_acronyms(self) -> set:
        """
        🎭 FAIRY CODEMOTHER'S ACRONYM DICTIONARY v6.0! 💅
        Known skill acronyms that should NOT be filtered out even if short!
        Enhanced for Singapore job market!
        """
        return {
            # Cloud & DevOps
            'AWS', 'GCP', 'EC2', 'S3', 'RDS', 'EKS', 'ECS', 'IAM', 'VPC', 'CDN',
            'CI', 'CD', 'IaC', 'K8S', 'VM', 'DNS', 'SSL', 'TLS', 'SSH', 'VPN',
            'ELB', 'ALB', 'NLB', 'SNS', 'SQS', 'SES', 'ACM', 'WAF', 'EMR',
            'AKS', 'GKE', 'EKS', 'ACR', 'ECR', 'GCR',

            # Programming & Frameworks
            'OOP', 'API', 'REST', 'SDK', 'IDE', 'GUI', 'CLI', 'MVC', 'MVP', 'MVVM',
            'JS', 'TS', 'SQL', 'CSS', 'HTML', 'XML', 'JSON', 'YAML', 'CSV', 'JSX', 'TSX',
            'NPM', 'PIP', 'GIT', 'SVN', 'JWT', 'OAuth', 'CORS', 'CRUD', 'GRPC', 'RPC',
            'ORM', 'JPA', 'JDK', 'JRE', 'JVM', 'CLR', 'DDD', 'CQRS',

            # Data & AI
            'ML', 'AI', 'NLP', 'CV', 'DL', 'NN', 'CNN', 'RNN', 'LSTM', 'GAN', 'LLM',
            'ETL', 'ELT', 'BI', 'KPI', 'ROI', 'SLA', 'KYC', 'AML', 'RAG', 'GPT',
            'OCR', 'ASR', 'TTS', 'NLU', 'NLG', 'MLOps', 'AIOps',

            # Security
            'SIEM', 'SOC', 'IDS', 'IPS', 'WAF', 'DDoS', 'XSS', 'CSRF', 'SQLi',
            'PCI', 'DSS', 'GDPR', 'HIPAA', 'SOX', 'NIST', 'ISO', 'PDPA',
            'VAPT', 'APT', 'EDR', 'XDR', 'MDR', 'SOAR', 'IAM', 'PAM', 'MFA', '2FA',

            # Quality & Testing
            'QA', 'QC', 'UAT', 'SIT', 'TDD', 'BDD', 'E2E', 'API', 'UI', 'UX',

            # Business & Management
            'ERP', 'CRM', 'HRM', 'SCM', 'PMP', 'PMO', 'ITIL', 'SAP', 'BPM',
            'P&L', 'B2B', 'B2C', 'SaaS', 'PaaS', 'IaaS', 'DaaS', 'FaaS',
            'OKR', 'MBO', 'KRA', 'NPS', 'CSAT', 'CLV', 'CAC', 'MRR', 'ARR',

            # Singapore Financial Certifications
            'CMFAS', 'M5', 'M8', 'M8A', 'M9', 'M9A', 'HI', 'CGI', 'BCP',
            'CACS', 'CEHA', 'IIQE', 'CII', 'IBF', 'FRM', 'CFA', 'CFP', 'ACCA',
            'M1', 'M1A', 'M2', 'M3', 'M4', 'M6', 'M6A', 'M7',

            # Singapore Safety & Compliance
            'WSH', 'SHE', 'HSE', 'MOM', 'MPA', 'NEA', 'ISPS', 'OSHAS', 'NEBOSH',
            'ATP', 'LSP', 'AED', 'CPR', 'IOSH', 'BCSS', 'SCDF', 'FSM',
            'WAH', 'CS', 'PTW', 'LOTO', 'HIRA', 'JSA', 'HAZOP', 'FMEA',

            # Singapore Government/Industry
            'GeBIZ', 'SG', 'IMDA', 'EDB', 'ESG', 'STB', 'HDB', 'LTA', 'PUB',
            'CPF', 'IRAS', 'ACRA', 'MAS', 'SGX', 'PDPC',

            # Web3 & Blockchain
            'NFT', 'DAO', 'DeFi', 'DEX', 'CEX', 'Web3', 'EVM', 'TVL',

            # Modern Tech
            'IoT', 'AR', 'VR', 'XR', 'MR', '5G', 'RPA', 'BPA', 'ESB', 'iPaaS',
        }

    def _is_technical_skill(self, skill_lower: str) -> bool:
        """
        📧 FAIRY CODEMOTHER'S TECHNICAL SKILL DETECTOR v6.0! 💻
        Covers ALL job types from Software to Safety!
        Enhanced for Singapore job market and modern tech!
        """
        tech_keywords = [
            # ===== SOFTWARE ENGINEERING & DEVELOPMENT =====
            'programming', 'coding', 'software', 'development', 'developer',
            'object-oriented', 'oop', 'scalable', 'system design', 'architecture',
            'restful', 'api', 'microservices', 'frontend', 'backend', 'full-stack',
            'fullstack', 'integration', 'state management', 'performance optimization',
            'secure coding', 'code review', 'refactoring', 'debugging', 'clean code',
            'design patterns', 'solid principles', 'dry', 'kiss', 'yagni',

            # Agile & Workflow
            'agile', 'scrum', 'kanban', 'sprint', 'ci/cd', 'cicd', 'pipeline',
            'version control', 'git', 'svn', 'tdd', 'test-driven', 'bdd',
            'continuous integration', 'continuous delivery', 'continuous deployment',
            'gitops', 'devsecops', 'shift-left',

            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'csharp',
            'ruby', 'php', 'go', 'golang', 'rust', 'kotlin', 'swift', 'scala',
            'perl', 'r programming', 'matlab', 'vba', 'bash', 'shell', 'powershell',
            'dart', 'elixir', 'clojure', 'haskell', 'lua', 'groovy', 'objective-c',
            'assembly', 'fortran', 'cobol', 'abap', 'apex', 'solidity',

            # Frameworks & Libraries - Frontend
            'react', 'angular', 'vue', 'svelte', 'solid', 'qwik', 'astro',
            'jquery', 'bootstrap', 'tailwind', 'material ui', 'chakra', 'ant design',
            'next.js', 'nuxt', 'gatsby', 'remix', 'sveltekit',
            'redux', 'mobx', 'zustand', 'recoil', 'jotai', 'pinia', 'vuex',

            # Frameworks & Libraries - Backend
            'node', 'express', 'fastify', 'nest.js', 'koa', 'hapi',
            'django', 'flask', 'fastapi', 'tornado', 'aiohttp',
            'spring', 'spring boot', 'hibernate', 'quarkus', 'micronaut',
            '.net', 'asp.net', '.net core', 'blazor', 'entity framework',
            'laravel', 'symfony', 'codeigniter', 'yii',
            'rails', 'sinatra', 'hanami',
            'gin', 'echo', 'fiber', 'chi',

            # Frameworks & Libraries - AI/ML
            'tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy',
            'scipy', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
            'huggingface', 'transformers', 'langchain', 'llamaindex',
            'opencv', 'pillow', 'spacy', 'nltk', 'gensim',
            'xgboost', 'lightgbm', 'catboost', 'ray', 'dask',

            # ===== QA & TESTING =====
            'testing', 'test', 'qa', 'quality assurance', 'quality control',
            'manual testing', 'automated testing', 'automation', 'test planning',
            'test case', 'test design', 'regression', 'smoke testing', 'sanity',
            'end-to-end', 'e2e', 'defect tracking', 'bug tracking',
            'test automation', 'selenium', 'cypress', 'playwright', 'puppeteer',
            'jest', 'junit', 'pytest', 'mocha', 'chai', 'vitest',
            'postman', 'jira', 'testng', 'cucumber', 'appium', 'detox',
            'root cause analysis', 'risk-based testing', 'uat', 'sit',
            'load testing', 'performance testing', 'jmeter', 'gatling', 'k6', 'locust',
            'api testing', 'contract testing', 'pact', 'karate',

            # ===== CYBERSECURITY =====
            'security', 'cybersecurity', 'infosec', 'information security',
            'vulnerability', 'penetration testing', 'pentest', 'ethical hacking',
            'threat modeling', 'threat intelligence', 'incident response',
            'security monitoring', 'siem', 'soc', 'ids', 'ips', 'firewall',
            'owasp', 'iso 27001', 'nist', 'pci dss', 'gdpr', 'pdpa', 'compliance',
            'web application security', 'network security', 'endpoint security',
            'exploit', 'malware', 'forensics', 'red team', 'blue team', 'purple team',
            'encryption', 'cryptography', 'ssl', 'tls', 'vpn', 'authentication',
            'zero trust', 'devsecops', 'appsec', 'cloud security', 'container security',
            'vapt', 'bug bounty', 'cve', 'mitre att&ck', 'kill chain',

            # ===== DATA & ANALYTICS =====
            'data', 'analytics', 'analysis', 'statistical', 'statistics',
            'data cleaning', 'data preprocessing', 'etl', 'elt', 'data pipeline',
            'data validation', 'data modeling', 'data warehouse', 'data lake', 'data mesh',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'redis',
            'tableau', 'power bi', 'looker', 'metabase', 'superset', 'dashboarding', 'visualization',
            'excel', 'spreadsheet', 'pivot', 'vlookup', 'xlookup', 'power query',
            'snowflake', 'databricks', 'bigquery', 'redshift', 'synapse',
            'dbt', 'airflow', 'prefect', 'dagster', 'fivetran', 'airbyte',

            # ===== AI & MACHINE LEARNING =====
            'machine learning', 'ml', 'artificial intelligence', 'ai',
            'deep learning', 'neural network', 'nlp', 'natural language',
            'computer vision', 'image recognition', 'speech recognition',
            'model training', 'model tuning', 'hyperparameter', 'feature engineering',
            'model deployment', 'inference', 'mlops', 'model monitoring',
            'transformer', 'bert', 'gpt', 'llm', 'prompt engineering', 'rag',
            'gpu', 'cuda', 'tpu', 'distributed computing',
            'generative ai', 'genai', 'chatbot', 'conversational ai',
            'recommendation system', 'anomaly detection', 'time series',
            'reinforcement learning', 'federated learning', 'transfer learning',
            'embeddings', 'vector database', 'pinecone', 'weaviate', 'milvus', 'qdrant',

            # ===== DEVOPS & CLOUD =====
            'devops', 'cloud', 'infrastructure', 'aws', 'azure', 'gcp',
            'google cloud', 'amazon web services', 'microsoft azure',
            'docker', 'container', 'kubernetes', 'k8s', 'orchestration',
            'terraform', 'ansible', 'puppet', 'chef', 'cloudformation', 'pulumi',
            'infrastructure as code', 'iac', 'serverless', 'lambda', 'functions',
            'monitoring', 'logging', 'observability', 'prometheus', 'grafana', 'datadog',
            'elk', 'splunk', 'new relic', 'dynatrace', 'jaeger', 'zipkin',
            'high availability', 'disaster recovery', 'backup', 'failover',
            'load balancing', 'auto scaling', 'cdn', 'dns', 'service mesh', 'istio',
            'argocd', 'flux', 'helm', 'kustomize', 'rancher', 'openshift',

            # ===== DATABASES =====
            'database', 'dbms', 'rdbms', 'sql server', 'db2', 'sqlite',
            'elasticsearch', 'cassandra', 'dynamodb', 'cosmos db', 'firestore',
            'database design', 'normalization', 'indexing', 'query optimization',
            'neo4j', 'graph database', 'timescaledb', 'influxdb', 'clickhouse',
            'cockroachdb', 'tidb', 'vitess', 'planetscale', 'supabase', 'neon',

            # ===== NETWORKING =====
            'network', 'networking', 'tcp/ip', 'http', 'https', 'ftp', 'smtp',
            'routing', 'switching', 'vlan', 'subnet', 'dhcp', 'bgp', 'ospf',
            'cisco', 'juniper', 'palo alto', 'fortinet', 'checkpoint',
            'sd-wan', 'sdn', 'nfv', 'mpls', 'ipsec',

            # ===== MOBILE DEVELOPMENT =====
            'mobile', 'ios', 'android', 'react native', 'flutter', 'xamarin',
            'swift', 'swiftui', 'kotlin', 'jetpack compose', 'expo',
            'mobile app', 'app store', 'play store', 'push notification',

            # ===== WEB3 & BLOCKCHAIN =====
            'blockchain', 'web3', 'smart contract', 'solidity', 'ethereum',
            'defi', 'nft', 'dao', 'cryptocurrency', 'bitcoin',
            'hardhat', 'truffle', 'foundry', 'wagmi', 'ethers.js', 'web3.js',
            'polygon', 'solana', 'avalanche', 'arbitrum', 'optimism',

            # ===== TOOLS & SOFTWARE =====
            'office', 'microsoft office', 'word', 'powerpoint', 'outlook',
            'google workspace', 'gsuite', 'slack', 'teams', 'zoom', 'notion',
            'adobe', 'photoshop', 'illustrator', 'figma', 'sketch', 'xd', 'canva',
            'autocad', 'solidworks', 'matlab', 'labview', 'revit', 'blender',
            'sap', 'salesforce', 'hubspot', 'zendesk', 'servicenow', 'freshdesk',
            'confluence', 'sharepoint', 'trello', 'asana', 'monday.com', 'clickup',

            # ===== SINGAPORE SAFETY & COMPLIANCE =====
            'wsh', 'workplace safety', 'safety health', 'she', 'hse', 'ehs',
            'risk assessment', 'hazard', 'incident investigation', 'near miss',
            'audit', 'auditing', 'inspection', 'compliance',
            'iso 14001', 'oshas 18001', 'iso 45001', 'bizsafe', 'bizcare',
            'mom', 'mpa', 'nea', 'scdf', 'isps', 'marpol', 'solas', 'stcw',
            'scaffold', 'fall protection', 'working at height', 'wah',
            'permit to work', 'ptw', 'loto', 'confined space', 'hot work',
            'fire safety', 'emergency response', 'first aid', 'aed', 'cpr', 'bcss',
            'lifting', 'crane', 'forklift', 'reach truck', 'counterbalance',

            # ===== SINGAPORE FINANCIAL =====
            'cmfas', 'cfa', 'cfp', 'frm', 'acca', 'cpa', 'ca',
            'financial planning', 'wealth management', 'investment',
            'insurance', 'bancassurance', 'underwriting', 'claims',
            'kyc', 'aml', 'cft', 'fatca', 'crs', 'sanctions screening',
            'trade finance', 'treasury', 'forex', 'derivatives', 'structured products',
            'private banking', 'retail banking', 'corporate banking',
            'mas regulations', 'sgx', 'cpf', 'srs',

            # ===== CERTIFICATIONS =====
            'certified', 'certification', 'license', 'accredited',
            'nebosh', 'iosh', 'osha', 'bcss', 'wshc',
            'pmp', 'prince2', 'itil', 'togaf', 'cobit', 'six sigma', 'lean',
            'aws certified', 'azure certified', 'google certified', 'oci certified',
            'cissp', 'cism', 'cisa', 'ceh', 'oscp', 'comptia', 'security+',
            'ccna', 'ccnp', 'ccie', 'jncia', 'jncis',
            'scrum master', 'csm', 'psm', 'product owner', 'cspo', 'pspo',
            'safe', 'scaled agile', 'less', 'dassm', 'icagile',
        ]

        return any(kw in skill_lower for kw in tech_keywords)

    def _is_soft_skill(self, skill_lower: str) -> bool:
        """
        💅 FAIRY CODEMOTHER'S SOFT SKILL DETECTOR v6.0! 🌐Ÿ
        The interpersonal MAGIC that makes careers SHINE!
        Enhanced for Singapore job market!
        """
        soft_keywords = [
            # ===== COMMUNICATION =====
            'communication', 'verbal', 'written', 'presentation', 'public speaking',
            'articulate', 'listening', 'active listening', 'storytelling',
            'negotiation', 'persuasion', 'influence', 'facilitation',
            'bilingual', 'multilingual', 'english', 'mandarin', 'malay', 'tamil',
            'cantonese', 'hokkien', 'teochew', 'translation', 'interpretation',

            # ===== LEADERSHIP & MANAGEMENT =====
            'leadership', 'leader', 'leading', 'management', 'managing',
            'supervision', 'supervising', 'mentoring', 'coaching', 'training',
            'delegation', 'empowerment', 'motivation', 'inspiring',
            'decision making', 'decision-making', 'strategic thinking',
            'vision', 'direction', 'guidance', 'people management',
            'team management', 'team lead', 'team leadership', 'servant leadership',
            'situational leadership', 'transformational leadership',

            # ===== TEAMWORK & COLLABORATION =====
            'teamwork', 'team player', 'collaboration', 'collaborative',
            'cross-functional', 'cross functional', 'interpersonal',
            'relationship building', 'networking', 'stakeholder',
            'partnership', 'cooperation', 'coordination', 'synergy',
            'consensus building', 'conflict management', 'mediation',
            'virtual team', 'remote collaboration', 'multicultural',

            # ===== PROBLEM SOLVING =====
            'problem solving', 'problem-solving', 'analytical', 'analysis',
            'critical thinking', 'logical', 'reasoning', 'troubleshooting',
            'creative thinking', 'creativity', 'innovative', 'innovation',
            'resourceful', 'solution-oriented', 'solutions', 'lateral thinking',
            'design thinking', 'systems thinking', 'first principles',
            'root cause', 'hypothesis', 'experimentation',

            # ===== ORGANIZATION & PLANNING =====
            'organization', 'organisation', 'organizational', 'organisational',
            'planning', 'prioritization', 'prioritizing', 'time management',
            'multitasking', 'multi-tasking', 'scheduling', 'deadline',
            'attention to detail', 'detail-oriented', 'meticulous', 'thorough',
            'work organization', 'task management', 'workload management',
            'efficiency', 'productivity', 'goal setting', 'milestone tracking',

            # ===== ADAPTABILITY =====
            'adaptability', 'adaptable', 'flexibility', 'flexible',
            'agility', 'resilience', 'resilient', 'change management',
            'learning agility', 'quick learner', 'fast learner', 'self-learner',
            'open-minded', 'growth mindset', 'continuous learning',
            'versatile', 'versatility', 'dynamic', 'proactive',
            'initiative', 'self-motivated', 'self-starter', 'autonomous',

            # ===== CUSTOMER & CLIENT FOCUS =====
            'customer service', 'client service', 'customer focus',
            'client relations', 'customer satisfaction', 'service oriented',
            'empathy', 'patience', 'rapport building', 'customer experience',
            'customer success', 'client management', 'customer retention',
            'service excellence', 'hospitality', 'guest relations',

            # ===== PROFESSIONALISM =====
            'professionalism', 'professional', 'integrity', 'ethics', 'ethical',
            'accountability', 'responsibility', 'reliability', 'dependable',
            'confidentiality', 'discretion', 'trustworthy', 'punctual',
            'work ethic', 'diligent', 'diligence', 'committed', 'commitment',
            'ownership', 'follow through', 'follow-through',

            # ===== EMOTIONAL INTELLIGENCE =====
            'emotional intelligence', 'eq', 'self-awareness', 'self awareness',
            'empathy', 'emotional competence', 'conflict resolution',
            'stress management', 'composure', 'poise', 'mindfulness',
            'self-regulation', 'social awareness', 'relationship management',
            'cultural sensitivity', 'cultural awareness', 'diversity',
            'inclusion', 'psychological safety',

            # ===== BUSINESS ANALYSIS & STRATEGY =====
            'requirements gathering', 'stakeholder analysis', 'gap analysis',
            'process improvement', 'process optimization', 'business process',
            'user stories', 'acceptance criteria', 'functional specification',
            'business acumen', 'commercial awareness', 'industry knowledge',
            'strategic planning', 'business strategy', 'competitive analysis',

            # ===== PRODUCT & PROJECT =====
            'roadmap', 'product lifecycle', 'market research', 'prioritization',
            'cross-functional collaboration', 'stakeholder alignment',
            'mvp', 'kpi tracking', 'okr', 'backlog management',
            'project planning', 'risk management', 'budget', 'resource allocation',
            'timeline management', 'delivery', 'milestone', 'scope management',
            'vendor management', 'contract management', 'procurement',

            # ===== HR & ADMIN =====
            'recruitment', 'onboarding', 'talent management', 'talent acquisition',
            'performance evaluation', 'performance review', 'appraisal',
            'payroll', 'benefits administration', 'hr operations',
            'policy', 'documentation', 'operational efficiency',
            'employee engagement', 'employee relations', 'workforce planning',
            'succession planning', 'learning development', 'compensation',

            # ===== SALES & MARKETING =====
            'sales', 'selling', 'lead generation', 'pipeline management',
            'client acquisition', 'account management', 'business development',
            'marketing', 'campaign', 'market segmentation', 'brand',
            'conversion', 'revenue', 'target achievement', 'quota',
            'cold calling', 'prospecting', 'closing', 'upselling', 'cross-selling',
            'relationship selling', 'consultative selling', 'solution selling',

            # ===== SINGAPORE SPECIFIC =====
            'singlish', 'local culture', 'government liaison', 'statutory',
            'regulatory', 'cpf', 'ir8a', 'employment act', 'mom compliance',
        ]

        return any(kw in skill_lower for kw in soft_keywords)

    def _is_known_multi_word_skill(self, skill: str) -> bool:
        """
        🎭 FAIRY CODEMOTHER'S MULTI-WORD SKILL VALIDATOR! 💅
        Some skills are naturally multiple words - don't skip these, honey!
        """
        skill_lower = skill.lower()
        
        known_multi_word_skills = {
            # Technical
            'object-oriented programming', 'object oriented programming',
            'machine learning', 'deep learning', 'natural language processing',
            'computer vision', 'artificial intelligence', 'data science',
            'test driven development', 'test-driven development',
            'continuous integration', 'continuous delivery', 'continuous deployment',
            'version control', 'source control', 'infrastructure as code',
            'cloud computing', 'distributed systems', 'microservices architecture',
            'api development', 'web development', 'mobile development',
            'database design', 'system design', 'software architecture',
            'penetration testing', 'vulnerability assessment', 'threat modeling',
            'incident response', 'security monitoring', 'risk assessment',
            'data analysis', 'statistical analysis', 'predictive analytics',
            'business intelligence', 'data visualization', 'data modeling',
            'model training', 'model deployment', 'feature engineering',
            'prompt engineering', 'gpu optimization',
            
            # Soft Skills
            'problem solving', 'problem-solving', 'critical thinking',
            'time management', 'project management', 'change management',
            'conflict resolution', 'team building', 'relationship building',
            'customer service', 'client relations', 'stakeholder management',
            'attention to detail', 'attention-to-detail',
            'emotional intelligence', 'stress management',
            'requirements gathering', 'gap analysis', 'process improvement',
            'cross-functional collaboration', 'stakeholder alignment',
            'risk management', 'budget control', 'resource allocation',
            'lead generation', 'pipeline management', 'account management',
            
            # Safety (Singapore)
            'workplace safety', 'safety health', 'risk assessment',
            'incident investigation', 'fall protection', 'working at height',
            'permit to work', 'confined space', 'fire safety',
            'emergency response', 'first aid',
        }
        
        return skill_lower in known_multi_word_skills

    def _extract_experience_date_first_format(self, text: str) -> list:
        """
        🎭 FAIRY CODEMOTHER'S SPECIAL! 💅
        Handles Singapore/UK resume format where dates come BEFORE company/role!
        
        Format: "Feb 2016 to Present    Role, Company Name"
        
        This is like reading a mirror image, darling - we flip the script! ✨
        """
        jobs = []
        
        # 🎯 Pattern for date-first format
        # Matches: "Feb 2016 to Present    Financial Consultant, Prudential Assurance Company Singapore (Pte) Ltd"
        # 🎯 Pattern for date-first format - NOW WITH TAB SUPPORT! 💅
        # Singapore resumes use TAB characters between date and role/company
        date_first_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})[\s\t]+to[\s\t]+(Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})[\s\t]+([^\n]+)'
        
        # Find all matches
        matches = list(re.finditer(date_first_pattern, text, re.IGNORECASE | re.MULTILINE))
        
        if not matches:
            return []
        
        self.logger.info(f"🎯 Found {len(matches)} date-first format entries")
        
        for i, match in enumerate(matches):
            start_date = match.group(1).strip()
            end_date = match.group(2).strip()
            role_company = match.group(3).strip()
            
            # 🎭 FAIRY CODEMOTHER'S EDUCATION FILTER v2.0!
            # Skip entries that are clearly EDUCATION, not work experience!
            education_keywords = [
                'diploma', 'degree', 'bachelor', 'master', 'phd', 'certificate',
                'gce', 'o level', 'a level', 'n level', 'psle', 'nitec', 'ite',
                'polytechnic', 'university', 'college', 'school', 'institute',
                'secondary', 'primary', 'junior college', 'jc'
            ]
            role_company_lower = role_company.lower()
            
            # Check if this looks like an education entry
            is_education = any(edu_kw in role_company_lower for edu_kw in education_keywords)
            
            if is_education:
                self.logger.debug(f"â­ Skipping education entry in experience: {role_company[:50]}")
                continue

            # Parse role and company from "Role, Company Name"
            # Could be "Financial Consultant, Prudential Assurance Company Singapore (Pte) Ltd"
            if ',' in role_company:
                parts = role_company.split(',', 1)
                role = parts[0].strip()
                company = parts[1].strip() if len(parts) > 1 else "Company not specified"
            else:
                # If no comma, try to detect role vs company
                role = role_company
                company = "See description"
            
            # Format dates
            dates = f"{start_date} - {end_date}"
            
            # 🎯 Get description (text until next date pattern or section header)
            chunk_start = match.end()
            if i + 1 < len(matches):
                chunk_end = matches[i + 1].start()
            else:
                # Find next section header with MORE specific boundaries, darling! 💅
                next_section = re.search(
                    r'\n\s*(?:Education|Skills|Achievements|Co-Curricular|Additional|Qualifications?)\s*(?:\n|$)', 
                    text[chunk_start:], 
                    re.IGNORECASE
                )
                chunk_end = chunk_start + (next_section.start() if next_section else 8000)  # 🏆• Increased from 1500!

            desc_chunk = text[chunk_start:chunk_end]

            # 🎭 FAIRY CODEMOTHER'S ENHANCED DESCRIPTION EXTRACTOR v2.0!
            description_lines = []
            consecutive_empty = 0

            # 🎯 Action verbs that typically START job responsibilities
            action_verb_starters = (
                'carried', 'conducted', 'managed', 'developed', 'worked', 
                'maintained', 'achieved', 'established', 'engaged', 'created', 
                'held', 'ensured', 'performed', 'led', 'built', 'designed', 
                'implemented', 'executed', 'responsible', 'assisted', 'organized', 
                'organised', 'supported', 'delivered', 'generated', 'resolved', 
                'authored', 'diagnosed', 'prepared', 'trained', 'mentored', 
                'supervised', 'oversaw', 'spearheaded', 'streamlined', 'optimized', 
                'optimised', 'facilitated', 'hosted', 'educated', 'collaborated', 
                'improved', 'increased', 'decreased', 'reduced', 'enhanced', 
                'drove', 'directed', 'handled', 'processed', 'administered', 
                'monitored', 'evaluated', 'assessed', 'identified', 'formulated', 
                'defined', 'planned', 'contributed', 'participated', 'provided',
                'applied', 'utilized', 'utilised', 'demonstrated', 'initiated',
                'launched', 'negotiated', 'presented', 'reviewed', 'analyzed',
                'analysed', 'coordinated', 'enforced', 'managing', 'providing',
                'enforcement', 'providing', 'assisting'
            )

            for line in desc_chunk.split('\n'):
                line = line.strip()
                
                # Track empty lines
                if len(line) < 5:
                    consecutive_empty += 1
                    # Stop if we hit 3+ consecutive empty lines (likely end of section)
                    if consecutive_empty >= 6 and description_lines:
                        break
                    continue
                
                consecutive_empty = 0
                
                # 🛑 STOP CONDITIONS: These indicate we've left the job description
                # Check for date patterns (next job entry)
                if re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}', line, re.IGNORECASE):
                    break
                
                # Check for section headers
                if re.match(r'^(?:Education|Skills|Achievements|Co-Curricular|Additional|Awards|Certifications|References|Projects|Languages|Hobbies)\s*$', line, re.IGNORECASE):
                    break
                
                # 🎯 CAPTURE CONDITIONS:
                
                # 1. Lines with bullet markers
                is_bullet = line.startswith(('*', '-', '•', '·', '►', '➢', '◗‹', '◗¦', '▪', '▸'))
                
                # 2. Lines starting with action verbs (job responsibilities!)
                first_word = line.split()[0].lower().rstrip('.,') if line.split() else ''
                is_action_line = first_word in action_verb_starters
                
                # Clean bullet markers if present
                if is_bullet:
                    cleaned = re.sub(r'^[\*\-•·►➢◗‹◗¦▪▸]\s*', '', line).strip()
                else:
                    cleaned = line
                
                # Capture if it's a bullet OR action line
                if (is_bullet or is_action_line) and len(cleaned) > 10:
                    description_lines.append(cleaned)
                elif description_lines and len(line) > 20 and len(line) < 500:
                    # Continue capturing if we already have content and line is substantial
                    # This catches continuation lines
                    # But skip if it looks like a new section/header (all caps, short)
                    if not line.isupper() and not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', line):
                        description_lines.append(line)

           # 💎 Join with pipe delimiter + REMOVE DUPLICATES for beautiful formatting!
            if description_lines:
                # Remove duplicate lines while preserving order 🌐Ÿ
                seen = set()
                unique_lines = []
                for line in description_lines:
                    # Normalize for comparison (lowercase, strip whitespace)
                    normalized = line.lower().strip()
                    if normalized not in seen and len(normalized) > 5:
                        seen.add(normalized)
                        unique_lines.append(line)
                
                description = ' | '.join(unique_lines) if unique_lines else "Description not available"
            else:
                description = "Description not available"
            
            # 🎭 Action verbs that typically START job responsibilities
            action_verb_starters = (
                'carried', 'conducted', 'managed', 'developed', 'worked', 
                'maintained', 'achieved', 'established', 'engaged', 'created', 
                'held', 'ensured', 'performed', 'led', 'built', 'designed', 
                'implemented', 'executed', 'responsible', 'assisted', 'organized', 
                'organised', 'supported', 'delivered', 'generated', 'resolved', 
                'authored', 'diagnosed', 'prepared', 'trained', 'mentored', 
                'supervised', 'oversaw', 'spearheaded', 'streamlined', 'optimized', 
                'optimised', 'facilitated', 'hosted', 'educated', 'collaborated', 
                'improved', 'increased', 'decreased', 'reduced', 'enhanced', 
                'drove', 'directed', 'handled', 'processed', 'administered', 
                'monitored', 'evaluated', 'assessed', 'identified', 'formulated', 
                'defined', 'planned', 'contributed', 'participated', 'provided',
                'applied', 'utilized', 'utilised', 'demonstrated', 'initiated',
                'launched', 'negotiated', 'presented', 'reviewed', 'analyzed',
                'analysed', 'coordinated', 'enforced', 'managing', 'providing',
                'enforcement', 'providing', 'assisting'
            )
            
            for line in desc_chunk.split('\n'):
                line = line.strip()
                
                # Track empty lines
                if len(line) < 5:
                    consecutive_empty += 1
                    # Stop if we hit 3+ consecutive empty lines (likely end of section)
                    if consecutive_empty >= 6 and description_lines:
                        break
                    continue
                
                consecutive_empty = 0
                
                # 🛑 STOP CONDITIONS: These indicate we've left the job description
                # Check for date patterns (next job entry)
                if re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}', line, re.IGNORECASE):
                    break
                
                # Check for section headers
                if re.match(r'^(?:Education|Skills|Achievements|Co-Curricular|Additional|Awards|Certifications|References|Projects|Languages|Hobbies)\s*$', line, re.IGNORECASE):
                    break
                
                # 🎯 CAPTURE CONDITIONS:
                
                # 1. Lines with bullet markers
                is_bullet = line.startswith(('*', '-', '•', '·', '►', '➢', '◗‹', '◗¦', '▪', '▸'))
                
                # 2. Lines starting with action verbs (job responsibilities!)
                first_word = line.split()[0].lower().rstrip('.,') if line.split() else ''
                is_action_line = first_word in action_verb_starters
                
                # Clean bullet markers if present
                if is_bullet:
                    cleaned = re.sub(r'^[\*\-•·►➢◗‹◗¦▪▸]\s*', '', line).strip()
                else:
                    cleaned = line
                
                # Capture if it's a bullet OR action line
                if (is_bullet or is_action_line) and len(cleaned) > 10:
                    description_lines.append(cleaned)
                elif description_lines and len(line) > 20 and len(line) < 500:
                    # Continue capturing if we already have content and line is substantial
                    # This catches continuation lines
                    # But skip if it looks like a new section/header (all caps, short)
                    if not line.isupper() and not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', line):
                        description_lines.append(line)
            
            description = ' | '.join(description_lines) if description_lines else "Description not available"

            jobs.append({
                "company": company[:200],  # Increased limit
                "role": role[:200],        # Increased limit
                "dates": dates,
                "description": description  # No truncation - capture all!
            })
            
            self.logger.info(f"âœ… Extracted: {role[:40]} at {company[:40]}")
        
        return jobs
    
    def _extract_experience_regex(self, text: str) -> list:
        """
        💼 Extract work experience using IMPROVED REGEX v7.0
        
        🎭 FAIRY CODEMOTHER'S LATEST UPDATE with BLEEDING PREVENTION! ✨
        
        Now handles:
        - Markdown formatting cleanup
        - Date-first format (Singapore/UK style)
        - CLEAN section boundaries (no bleeding!)
        - Multiple resume formats
        - Edge cases and validation
        """

        jobs = []
        
        # 🧹 STEP 1: CLEAN THE TEXT FIRST!
        text = self._clean_markdown(text)
        self.logger.info("🧹 Cleaned markdown formatting")
        
        # 🏆 STEP 2: TRY DATE-FIRST FORMAT (Singapore/UK style)
        # This catches resumes like "Feb 2016 to Present    Financial Consultant, Prudential..."
        date_first_jobs = self._extract_experience_date_first_format(text)
        if date_first_jobs and len(date_first_jobs) >= 1:
            self.logger.info(f"🎯 Date-first format detected! Found {len(date_first_jobs)} jobs")
            return date_first_jobs
        
        # Certification keywords for filtering
        certification_keywords = [
            'certificate', 'certification', 'certified', 'diploma', 'course',
            'award', 'commendation', 'license', 'licence', 'accredit',
            'training', 'module', 'level', 'bizsafe', 'nebosh', 'iso',
            'oshas', 'mpa atp', 'mom atp', 'mom lsp', 'attestation'
        ]

        # ===================================================================
        # 🎭 STEP 3: ENHANCED SECTION DETECTION v4.0 with BLEEDING PREVENTION!
        # NEW: Uses _prevent_section_bleeding for CLEAN boundaries!
        # ===================================================================
        
        # 🛡️ Extract experience section with STRICT boundaries
        # This STOPS at Education/Certifications/Qualifications headers
        exp_text = self._prevent_section_bleeding(
            text, 
            'experience',  # Primary section to extract
            ['education', 'certifications', 'qualifications', 'achievements']  # Stop sections
        )
        
        # 🛡️ Validate we got something useful
        if not exp_text or len(exp_text) < 100:
            self.logger.warning("⚠️ No clear experience section found via boundaries - trying fallback")
            
            # Fallback: Use enhanced section detection
            sections = self._detect_section_boundaries_enhanced(text)
            if 'experience' in sections or 'experience_simple' in sections:
                section_key = 'experience' if 'experience' in sections else 'experience_simple'
                start, end = sections[section_key]
                exp_text = text[start:end]
                self.logger.info(f"🎯 Found experience section via boundaries: chars {start}-{end}")
            else:
                # Last resort: use full text
                exp_text = text
                self.logger.warning("⚠️ No clear experience section found - using full text")
        else:
            self.logger.info(f"✅ Experience section extracted with clean boundaries: {len(exp_text)} chars")

        # ===================================================================
        # 🏆 STEP 4: FIND ALL COMPANIES
        # (The rest of the method stays THE SAME as your current version!)
        # ===================================================================
        
        # Company patterns - these stay the same as your current code
        company_patterns = [
            # Singapore date-first pattern
            (r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s+to\s+(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s+([A-Z][A-Za-z\s]+(?:Consultant|Manager|Engineer|Developer|Assistant|Executive|Officer|Specialist|Coordinator|Director|Analyst)[,\s]+[A-Z][^\n]+)', 'sg_date_first'),

            # Standard Company Name Patterns (with legal suffixes)
            (r'\b([A-Z][A-Za-z0-9\s&\',\.\-]+(?:Pte\.?\s*Ltd\.?|Pvt\.?\s*Ltd\.?|Private\s+Limited|Limited|Inc\.?|Corp\.?|Corporation|LLC|Co\.?|Company|LLP|Partners))\b', 'legal_suffix'),
            
            # Standard Company Name Patterns (Capitalized without suffix)
            (r'(?:^|\n)\s*([A-Z][A-Za-z0-9\s&\',\.\-]{3,40})\s*(?:\n|,|\||$)', 'capital_name'),
            
            # Industry-specific patterns (Banks, Consulting, Tech)
            (r'\b([A-Z][A-Za-z\s]+(?:Bank|Consulting|Technologies|Solutions|Systems|Services|Group|Holdings|International))\b', 'industry'),
            
            # Government/Institution patterns
            (r'\b([A-Z][A-Za-z\s]+(?:Ministry|Government|Agency|Authority|Board|Council|Commission|Department))\b', 'government'),
        ]
        
        # ===================================================================
        # 🏆 STEP 4: DATE-ANCHORED JOB PARSING
        # One job block per date range. Each range anchors a job; company /
        # role come from the date line (+ the line above it), and the
        # description is the block of lines until the next date range or a
        # new section. Handles company-first and date-anywhere layouts
        # (pure date-first is already handled by STEP 2 above).
        # ===================================================================
        month = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?'
        date_tok = r'(?:' + month + r'\s*,?\s*\d{4}|\d{1,2}[/\-]\d{4}|\d{4})'
        end_tok = r'(?:' + date_tok + r'|Present|Current|Now|Ongoing|Till\s*Date|To\s*Date)'
        date_range_re = re.compile(
            r'(' + date_tok + r')\s*(?:-|–|—|to|till|until)\s*(' + end_tok + r')',
            re.IGNORECASE,
        )

        anchors = list(date_range_re.finditer(exp_text))
        if not anchors:
            self.logger.warning("⚠️ No date ranges found in experience section")
            return jobs

        edu_kw = (
            'diploma', 'degree', 'bachelor', 'master', 'phd', 'gce', 'o level',
            'a level', 'n level', 'psle', 'nitec', 'polytechnic', 'university',
            'college', 'secondary school', 'primary school', 'junior college',
        )

        lines = exp_text.split('\n')
        line_starts = []
        _pos = 0
        for _ln in lines:
            line_starts.append(_pos)
            _pos += len(_ln) + 1

        def _line_index_of(offset: int) -> int:
            idx = 0
            for li, s in enumerate(line_starts):
                if s <= offset:
                    idx = li
                else:
                    break
            return idx

        for i, m in enumerate(anchors):
            dates = f"{m.group(1).strip()} - {m.group(2).strip()}"

            anchor_line = _line_index_of(m.start())
            if i + 1 < len(anchors):
                nxt = _line_index_of(anchors[i + 1].start())
                # The line just above the next date is THAT job's header,
                # so end this description before it (prevents the next
                # role/company bleeding into this job's description).
                next_anchor_line = nxt - 1 if nxt - 1 > anchor_line else nxt
            else:
                next_anchor_line = len(lines)

            # Header = line above the date line + the date line (date removed)
            header_bits = []
            if anchor_line > 0:
                prev = lines[anchor_line - 1].strip(' \t|·•-')
                if prev:
                    header_bits.append(prev)
            same = date_range_re.sub('', lines[anchor_line]).strip(' \t|·•-,')
            if same:
                header_bits.append(same)
            header = ' '.join(header_bits).strip(' \t|·•-,')

            if any(k in header.lower() for k in edu_kw):
                self.logger.debug(f"⭐ Skipping education-looking block: {header[:50]}")
                continue

            if ',' in header:
                role, company = [p.strip() for p in header.split(',', 1)]
            elif re.search(r'\s+at\s+', header, re.IGNORECASE):
                parts = re.split(r'\s+at\s+', header, maxsplit=1, flags=re.IGNORECASE)
                role, company = parts[0].strip(), parts[1].strip()
            else:
                role, company = header, "See description"

            desc_lines = []
            seen = set()
            for ln in lines[anchor_line + 1:next_anchor_line]:
                ln = ln.strip()
                if len(ln) < 4:
                    continue
                if re.match(r'^(?:Education|Skills|Achievements|Awards|Certifications?|'
                            r'References|Projects|Languages|Hobbies|Co-Curricular|'
                            r'Additional|Qualifications?)\s*$', ln, re.IGNORECASE):
                    break
                ln = re.sub(r'^[\*\-•·►➢▪▸]\s*', '', ln).strip()
                key = ln.lower()
                if ln and key not in seen:
                    seen.add(key)
                    desc_lines.append(ln)

            description = ' | '.join(desc_lines) if desc_lines else "Description not available"

            jobs.append({
                "company": (company or "Company not specified")[:200],
                "role": (role or "Role not specified")[:200],
                "dates": dates,
                "description": description,
            })
            self.logger.info(f"✅ Extracted: {role[:40]} at {company[:40]}")

        self.logger.info(f"💼 Regex experience extraction found {len(jobs)} job(s)")
        return jobs

    def _clean_education_text(self, text: str) -> str:
        """
        🧹 Clean whitespace, tabs, and formatting artifacts from education text
        """
        if not text:
            return ""
        # Replace tabs with single space
        text = text.replace('\t', ' ')
        # Replace newlines with single space
        text = text.replace('\n', ' ')
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing colons and whitespace
        text = text.strip().strip(':').strip()
        return text

    def _extract_education_date_first_format(self, text: str) -> list:
        """
        🎓 FAIRY CODEMOTHER'S EDUCATION EXTRACTOR for Date-First Format! 💅
        Enhanced for Singapore resumes v6.0

        Handles: "Apr 2006 to Apr 2009    Diploma in Banking and Financial Services"
                "Singapore Polytechnic"
        """
        education = []

        # Singapore institutions for validation
        sg_institutions = [
            'singapore polytechnic', 'ngee ann polytechnic', 'temasek polytechnic',
            'republic polytechnic', 'nanyang polytechnic', 'ite', 'institute of technical education',
            'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss', 'national university of singapore',
            'nanyang technological university', 'singapore management university',
            'singapore university of technology', 'singapore institute of technology',
            'singapore university of social sciences', 'lasalle', 'nafa',
            'james cook university', 'kaplan', 'sim', 'mdis', 'psb academy',
            'junior college', 'jc', 'secondary school', 'pri school', 'primary school'
        ]

        # Pattern for date-first education entries
        edu_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s+(?:to|till|until|-|"“)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current)\s+([^\n]+)'

        # Find education section first - expanded patterns
        edu_section_patterns = [
            r'Education(?:al)?\s*(?:Background|History|Qualifications?)?\s*\n(.*?)(?=\n\s*(?:Achievements?|Skills?|Co-?Curricular|Additional|Work|Experience|Employment|Certification|Award|Language|Hobbies?|Interest|Reference|$))',
            r'Academic\s+(?:Background|Qualifications?|History)\s*\n(.*?)(?=\n\s*(?:Experience|Employment|Work|Skills|$))',
            r'Qualifications?\s*\n(.*?)(?=\n\s*(?:Experience|Employment|Work|Skills|$))',
        ]

        edu_text = ""
        for pattern in edu_section_patterns:
            edu_section_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if edu_section_match:
                edu_text = edu_section_match.group(1)
                break

        if not edu_text:
            edu_text = text

        matches = list(re.finditer(edu_pattern, edu_text, re.IGNORECASE))

        for i, match in enumerate(matches):
            start_date = match.group(1).strip()
            end_date = match.group(2).strip()
            degree = match.group(3).strip()

            dates = f"{start_date} - {end_date}"

            # Get institution (usually on the next line)
            chunk_start = match.end()
            if i + 1 < len(matches):
                chunk_end = matches[i + 1].start()
            else:
                chunk_end = min(chunk_start + 800, len(edu_text))

            chunk = edu_text[chunk_start:chunk_end]

            # First non-empty line is usually the institution
            institution = "Institution not specified"
            additional_info = []

            for line in chunk.split('\n'):
                line = line.strip()
                if not line or line.startswith(('Obtained', '*', '-', '•', '◗‹', '◗')):
                    continue

                # Check if it looks like a Singapore institution
                line_lower = line.lower()
                is_sg_inst = any(inst in line_lower for inst in sg_institutions)

                # Check if it looks like an institution name
                if institution == "Institution not specified":
                    if is_sg_inst or re.match(r'^[A-Z][A-Za-z\s\'-]+(?:School|Polytechnic|University|College|Institute|Academy|ITE|JC)', line, re.IGNORECASE):
                        institution = line
                    elif len(line) > 5 and len(line) < 150 and not any(kw in line_lower for kw in ['obtained', 'credits', 'grade', 'gpa', 'cgpa']):
                        institution = line
                else:
                    # Capture additional info like GPA, credits, subjects
                    if any(kw in line_lower for kw in ['gpa', 'cgpa', 'credit', 'grade', 'distinction', 'merit', 'obtained']):
                        additional_info.append(line)
                    elif len(additional_info) < 3 and len(line) > 10:
                        additional_info.append(line)

            # Append additional info to degree if found
            if additional_info:
                degree = f"{degree} ({'; '.join(additional_info[:2])})"

            education.append({
                "institution": institution[:250],  # Increased limit
                "degree": degree[:500],            # Increased limit for additional info
                "dates": dates
            })

            self.logger.info(f"🎓 Extracted: {degree[:50]} from {institution[:30]}")

        return education

    def _extract_education_label_value_format(self, text: str) -> list:
        """
        Extract education from label-value formats common in Singapore resumes.

        Handles:
        - Institution:/Course Attended:/Year Obtained: (Suhana format)
        - School:/Year Attended:/Passes: (Khairany format)
        - Qualification:/Year of Graduation:/Name of Institution: (Miliana format)

        Uses ORIGINAL text with newlines preserved for line-by-line parsing.
        """
        education = []

        # Check if this text has label-value education patterns
        has_inst_labels = re.search(r'(?:Institution|School)\s*:\s*\S', text, re.IGNORECASE)
        has_qual_labels = re.search(r'(?:Highest\s+)?Qualification\s*:\s*\S', text, re.IGNORECASE)

        if not has_inst_labels and not has_qual_labels:
            return education

        # Find education section boundaries on ORIGINAL text
        edu_section_patterns = [
            r'(?:Education(?:al)?\s*Qualifications?|Academic\s+Qualifications?|EDUCATION)\s*\n(.*?)(?=\n\s*(?:Other\s+Awarded|Employment|Work\s+Experience|Experience|Professional\s+Skills|Professional\s+Certificates|Salary|Language|Hobbies?|Interest|Reference|Availability|EMPLOYMENT|WORK|SKILLS)\b)',
            r'(?:Education(?:al)?\s*Qualifications?|Academic\s+Qualifications?|EDUCATION)\s*\n(.*)',
        ]

        edu_text = ""
        for pattern in edu_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                edu_text = match.group(1)
                break

        if not edu_text:
            edu_text = text

        # =========================================================
        # STRATEGY A: "Institution:" or "School:" appears as label
        # (Suhana, Khairany formats)
        # =========================================================
        if has_inst_labels:
            # Split into blocks at each Institution:/School: marker
            block_starts = list(re.finditer(
                r'(?:^|\n)\s*(?:Institution|School)\s*:\s*',
                edu_text, re.IGNORECASE
            ))

            for i, block_start in enumerate(block_starts):
                # Get block text from this marker to the next (or end)
                start = block_start.start()
                end = block_starts[i + 1].start() if i + 1 < len(block_starts) else len(edu_text)
                block_text = edu_text[start:end]

                # Extract institution name
                inst_match = re.search(
                    r'(?:Institution|School)\s*:\s*(.+?)(?:\n|$)',
                    block_text, re.IGNORECASE
                )
                institution = inst_match.group(1).strip() if inst_match else ""
                if not institution:
                    continue

                # Extract degree/course/qualification
                degree = "Qualification not specified"
                for dp in [
                    r'(?:Course\s+Attended|Certification|Certificate\s+Obtained|Passes|Qualification)\s*:\s*(.+?)(?:\n|$)',
                ]:
                    dm = re.search(dp, block_text, re.IGNORECASE)
                    if dm:
                        degree = dm.group(1).strip()
                        break

                # Extract dates/year
                dates = ""
                for dtp in [
                    r'(?:Year\s+Obtained|Year\s+Attended|Year\s+of\s+Graduation|Year)\s*:\s*(.+?)(?:\n|$)',
                ]:
                    dtm = re.search(dtp, block_text, re.IGNORECASE)
                    if dtm:
                        dates = dtm.group(1).strip()
                        break

                education.append({
                    "institution": institution[:250],
                    "degree": degree[:500],
                    "dates": dates
                })

            if education:
                self.logger.info(f"🎓 Label-value (Institution/School) format: {len(education)} entries")
                return education

        # =========================================================
        # STRATEGY B: "Qualification:" appears FIRST, then
        # "Name of Institution:" later (Miliana format)
        # =========================================================
        if has_qual_labels:
            qual_matches = list(re.finditer(
                r'(?:^|\n)\s*(?:Highest\s+)?Qualification\s*:\s*(.+?)(?:\n|$)',
                edu_text, re.IGNORECASE
            ))

            for i, qm in enumerate(qual_matches):
                degree = qm.group(1).strip()

                # Skip header-only lines like "Other Qualifications:"
                if not degree or degree.lower().rstrip(':') in ['', 'other qualifications', 'other']:
                    continue

                # Get block from this qualification to the next (or end)
                block_start = qm.start()
                block_end = qual_matches[i + 1].start() if i + 1 < len(qual_matches) else len(edu_text)
                block_text = edu_text[block_start:block_end]

                # Extract institution
                institution = "Institution not specified"
                inst_match = re.search(
                    r'(?:Name\s+of\s+Institution|Institution|School)\s*:\s*(.+?)(?:\n|$)',
                    block_text, re.IGNORECASE
                )
                if inst_match:
                    institution = inst_match.group(1).strip()

                # Extract dates/year
                dates = ""
                date_match = re.search(
                    r'(?:Year\s+of\s+Graduation|Year\s+Obtained|Year\s+Attended|Year)\s*:\s*(.+?)(?:\n|$)',
                    block_text, re.IGNORECASE
                )
                if date_match:
                    dates = date_match.group(1).strip()

                education.append({
                    "institution": institution[:250],
                    "degree": degree[:500],
                    "dates": dates
                })

            if education:
                self.logger.info(f"🎓 Label-value (Qualification-first) format: {len(education)} entries")
                return education

        return education

    def _extract_education_highest_qualification_inline(self, text: str) -> list:
        """
        Extract education from "Highest Qualification:" inline in personal details.

        Last resort for resumes where education is a single line like:
          Highest Qualification: "O" Levels - 3 credits (English, Tamil & Geography)
          Highest Qualification: G.C.E. A Level - Outram Secondary School (Pre-U Ctr)
        """
        education = []

        # Only use if there is NO education section header
        has_edu_section = re.search(
            r'(?:^|\n)\s*(?:Education|Academic)\s*(?:Qualifications?|Background|History)?\s*(?:[:\n])',
            text, re.IGNORECASE
        )
        if has_edu_section:
            return education

        hq_match = re.search(
            r'(?:^|\n)\s*Highest\s+Qualification\s*:\s*(.+?)(?:\n|$)',
            text, re.IGNORECASE
        )

        if not hq_match:
            return education

        qual_text = hq_match.group(1).strip()
        if not qual_text or len(qual_text) < 3:
            return education

        institution = "Institution not specified"
        degree = qual_text

        # Try to split "degree - institution" at separator
        separator_match = re.search(
            r'^(.+?)\s*[–\-]\s*([A-Z][A-Za-z\s\'-]+(?:School|Polytechnic|University|College|Institute|Academy|Centre|Center)[^\n]*)',
            qual_text, re.IGNORECASE
        )
        if separator_match:
            degree = separator_match.group(1).strip()
            institution = separator_match.group(2).strip()

        education.append({
            "institution": institution[:250],
            "degree": degree[:500],
            "dates": ""
        })

        self.logger.info(f"🎓 Highest Qualification inline format: {degree[:50]}")
        return education

    def _extract_education_regex(self, text: str) -> list:
        """
        Extract education using multi-format waterfall v8.0

        Extraction order:
        1. Date-first format (e.g., "Apr 2006 to Apr 2009  Diploma...")
        2. Label-value format (Institution:/Course:/Year: patterns)
        3. Highest Qualification inline (single line in personal details)
        4. Institution-pattern fallback (general regex matching)
        """

        # === STEP 1: Try date-first format (existing, working) ===
        date_first_edu = self._extract_education_date_first_format(text)
        if date_first_edu and len(date_first_edu) >= 1:
            self.logger.info(f"🎓 Date-first education format detected! Found {len(date_first_edu)} entries")
            return date_first_edu

        # === STEP 2: Try label-value format on ORIGINAL text ===
        label_value_edu = self._extract_education_label_value_format(text)
        if label_value_edu and len(label_value_edu) >= 1:
            return label_value_edu

        # === STEP 3: Try Highest Qualification inline ===
        highest_qual_edu = self._extract_education_highest_qualification_inline(text)
        if highest_qual_edu and len(highest_qual_edu) >= 1:
            return highest_qual_edu

        # === STEP 4: Fallback - institution pattern matching ===
        education = []

        sg_institutions = [
            'singapore polytechnic', 'ngee ann polytechnic', 'temasek polytechnic',
            'republic polytechnic', 'nanyang polytechnic', 'ite', 'institute of technical education',
            'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss', 'national university of singapore',
            'nanyang technological university', 'singapore management university',
            'singapore university of technology', 'singapore institute of technology',
            'singapore university of social sciences', 'lasalle', 'nafa',
            'james cook university', 'kaplan', 'sim global education', 'mdis', 'psb academy',
            'raffles', 'hwa chong', 'victoria junior college', 'anglo-chinese',
            'national junior college', 'catholic junior college', 'st. andrew',
            'anderson serangoon', 'tampines meridian', 'yishun innova', 'millennia institute'
        ]

        # Use ORIGINAL text for section boundary detection (needs newlines!)
        edu_text = self._prevent_section_bleeding(
            text,
            'education',
            ['experience', 'skills', 'certifications', 'achievements', 'projects']
        )

        # If boundary method didn't find education, try pattern matching
        if not edu_text or len(edu_text) < 50:
            self.logger.warning("⚠️ No education section found via boundaries - trying pattern matching")

            edu_patterns = [
                r'(?:EDUCATION(?:AL)?|ACADEMIC)\s*(?:BACKGROUND|HISTORY|QUALIFICATIONS?)?\s*:?\s*\n(.*?)(?=\n\s*(?:EXPERIENCE|EMPLOYMENT|WORK|SKILLS|CERTIFICATIONS?|AWARDS?)\s*(?:[:\n])|$)',
                r'QUALIFICATIONS?\s*:?\s*\n(.*?)(?=\n\s*(?:EXPERIENCE|EMPLOYMENT|WORK|SKILLS)\s*(?:[:\n])|$)',
            ]

            for pattern in edu_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    edu_text = match.group(1)
                    self.logger.info("🎓 Found education section via pattern")
                    break

        # If still nothing, search entire document for institutions
        if not edu_text or len(edu_text) < 50:
            self.logger.info("🎓 No education section header found, searching entire document...")
            sg_inst_found = any(inst in text.lower() for inst in sg_institutions)
            inst_pattern = r'([A-Z][A-Za-z\s\',\.&-]+(?:University|College|Institute|School|Academy|Polytechnic|ITE)[^\n]*)'

            if sg_inst_found or re.search(inst_pattern, text):
                edu_text = text[-3000:] if len(text) > 3000 else text
                self.logger.info("🎓 Using document tail for education search")
            else:
                self.logger.info("🎓 No education information found in document")
                return education
        else:
            self.logger.info(f"✅ Education section extracted with clean boundaries: {len(edu_text)} chars")

        # Institution patterns for fallback matching
        inst_patterns = [
            r'((?:Singapore|Ngee\s+Ann|Temasek|Republic|Nanyang)\s+Polytechnic)',
            r'((?:National\s+University\s+of\s+Singapore|NUS|Nanyang\s+Technological\s+University|NTU|Singapore\s+Management\s+University|SMU|Singapore\s+University\s+of\s+Technology|SUTD|Singapore\s+Institute\s+of\s+Technology|SIT|Singapore\s+University\s+of\s+Social\s+Sciences|SUSS))',
            r'((?:Institute\s+of\s+Technical\s+Education|ITE\s+College)\s*[A-Za-z]*)',
            r'((?:[A-Z][A-Za-z\s\'-]+(?:Junior\s+College|JC)|Millennia\s+Institute))',
            r'((?:[A-Z][A-Za-z\s\'-]+(?:Secondary\s+School|Sec\s+School)))',
            r'((?:Kaplan|SIM\s+Global|MDIS|PSB\s+Academy|James\s+Cook\s+University|LASALLE|NAFA|Raffles))',
            r'([A-Z][A-Za-z\s\',\.&-]+(?:University|College|Institute|School|Academy))',
        ]

        found_institutions = set()

        for inst_pattern in inst_patterns:
            for match in re.finditer(inst_pattern, edu_text):
                institution = match.group(1).strip()

                if len(institution) < 5 or len(institution) > 200:
                    continue

                # Skip company names (work experience bleeding)
                inst_lower = institution.lower()
                if any(kw in inst_lower for kw in ['pte ltd', 'pvt ltd', 'company', 'corporation', 'services pte']):
                    continue

                # Avoid duplicates
                if institution.lower() in found_institutions:
                    continue
                found_institutions.add(institution.lower())

                # Extract degree/qualification from surrounding context
                degree = "Qualification not specified"
                dates = ""

                context_start = max(0, match.start() - 300)
                context_end = min(len(edu_text), match.end() + 300)
                context = edu_text[context_start:context_end]

                degree_patterns = [
                    r'((?:Diploma|Higher Nitec|Nitec|Degree|Certificate|Bachelor|Master|Doctor)\s+(?:in|of)\s+[A-Za-z\s&,\'-]{5,100})',
                    r"((?:GCE\s+)?['\"]?[ONAona]['\"]?\s*Level[s]?[^\n,;]{0,50})",
                    r'(PSLE\s*(?:Certificate)?)',
                    r'((?:Bachelor|Master|Diploma|Certificate|Degree)\s+[^\n,;]{5,80})',
                ]

                for dp in degree_patterns:
                    dm = re.search(dp, context, re.IGNORECASE)
                    if dm:
                        degree = dm.group(1).strip()
                        break

                date_match = re.search(r'(\d{4})\s*(?:[-–to]+\s*(\d{4}|[Pp]resent|[Cc]urrent))?', context)
                if date_match:
                    dates = date_match.group(0).strip()

                education.append({
                    "institution": institution[:250],
                    "degree": degree[:500],
                    "dates": dates
                })

        return education

    def _needs_ai_enhancement(self, result: Dict) -> bool:
        """
        🤔 Check if we need AI to help categorize skills better
        """
        # Only use AI if skills are ambiguous
        total_skills = len(result.get('hard_skills', [])) + len(result.get('soft_skills', []))
        
        # If we found very few skills, AI might help
        if total_skills < 5:
            return True
        
        # If categorization seems off (too many in one category), AI might help
        hard_count = len(result.get('hard_skills', []))
        soft_count = len(result.get('soft_skills', []))
        
        if hard_count == 0 or soft_count == 0:
            return True
        
        return False

    def _enhance_skills_with_ai(self, result: Dict, text: str) -> Dict:
        """
        🤖 Use AI ONLY to re-categorize skills (not extract them!)
        """
        all_skills = result.get('hard_skills', []) + result.get('soft_skills', [])
        
        if not all_skills:
            return result
        
        skills_list = ', '.join(all_skills[:40])
        
        prompt = f"""Categorize these skills as hard (technical) or soft (interpersonal).

RULES:
- Hard skills = tools, systems, technical processes, software, methodologies
  Examples: "SAP HCM", "Python", "Payroll Processing", "Recruitment and Onboarding", "CPF e-Submission"
- Soft skills = interpersonal abilities, people skills, communication
  Examples: "Employee Relations", "Stakeholder Management", "Presentation Skills", "Teamwork"
- Functional skills like "Payroll Processing", "Benefits Administration", "Vendor Management" 
  are HARD skills — they are specific professional competencies, not interpersonal abilities
- Use the EXACT wording from the original skills — do NOT paraphrase or modify

SKILLS: {skills_list}

Return ONLY this JSON:
{{"hard": ["skill1", "skill2"], "soft": ["skill3", "skill4"]}}"""

        response = self._call_ollama_raw(prompt, num_predict=300, schema=SKILLS_SCHEMA)
        parsed = self._parse_ai_response(response)

        if parsed and isinstance(parsed.get('hard'), list) and isinstance(parsed.get('soft'), list):
            result['hard_skills'] = parsed['hard'][:30]
            result['soft_skills'] = parsed['soft'][:20]
            self.logger.info("âœ… AI re-categorized skills successfully (schema-enforced)")
        
        return result

    def _call_ollama_raw(self, prompt: str, num_predict: int = 500,
                         schema=None) -> str:
        """Helper to call Ollama and return raw text response.

        Args:
            schema: Optional JSON Schema dict. When provided it is passed as
                    the format parameter to ollama.chat(), enabling
                    grammar-based constrained generation.
        """
        try:
            # 💅 FAIRY CODEMOTHER'S FIX: Context window QUADRUPLED!
            # 4096 was WAY too small — resumes + prompt easily exceed 4K tokens.
            # 16384 gives us ~40,000 chars of headroom (resume + system prompt + response)
            # If your machine struggles, try 8192 as a compromise.
            options = {
                'temperature': 0.1,
                'top_p': 0.4,       # 🆕 Tighter focus (was 0.5)
                'top_k': 20,        # 🆕 Only consider top 20 tokens for each prediction
                'num_ctx': 16384,   # 🔥 Was 4096 — quadrupled!
                'num_predict': num_predict,
                'repeat_penalty': 1.1,  # 🆕 Prevent repetition in output
                # Disable Qwen3 chain-of-thought so num_predict isn't burned on <think>
                # blocks. Silently ignored by non-Qwen models and older Ollama versions.
                'think': False,
            }

            chat_kwargs = {
                'model': self.model_name,
                'messages': [
                    {
                        'role': 'system',
                        'content': (
                            'You are AiMerlion, a resume data extraction engine for Singapore/Malaysia recruitment.\n'
                            'RULES:\n'
                            '1. Output ONLY valid JSON — no markdown, no explanations\n'
                            '2. Extract EXACTLY as written — never paraphrase\n'
                            '3. Use null for missing fields — never "N/A" or empty string\n'
                            '4. SKILLS are abilities (Python, Excel, Project Management)\n'
                            '5. NOT skills: company names, job titles, job duties, certifications, section headers\n'
                            '6. CMFAS/NEBOSH/BizSafe are CERTIFICATIONS, not skills\n'
                            '7. Phone numbers in SG are 8 digits, often with +65 prefix\n'
                            '8. "Language : English & Chinese" is a HEADER field, not a section'
                        )
                    },
                    {'role': 'user', 'content': prompt}
                ],
                'options': options,
            }
            if schema is not None:
                chat_kwargs['format'] = schema

            response = self._client.chat(**chat_kwargs)

            content = response['message']['content'] or ""
            # Strip any residual Qwen3 <think>...</think> blocks as a safety net
            # in case 'think: False' isn't honored by this Ollama version.
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            content = re.sub(r'<think>.*$',         '', content, flags=re.DOTALL)
            return content.strip()

        except Exception as e:
            self.logger.error(f"âŒ AI Call error: {e}")
            return ""

    def _call_ollama(self, prompt: str, is_deep: bool = False, schema=None) -> Dict:
        """Helper to handle the actual API call"""
        try:
            # 💅 FAIRY CODEMOTHER'S FIX: Matching context window + tighter sampling
            options = {
                'temperature': 0.1,
                'top_p': 0.4,       # 🆕 Tighter focus
                'top_k': 20,        # 🆕 Top-K sampling
                'num_ctx': 16384,   # 🔥 Was 4096 — quadrupled!
                'repeat_penalty': 1.1,  # 🆕 Prevent repetition
                # Disable Qwen3 reasoning so num_predict goes entirely to JSON output
                'think': False,
            }

            if is_deep:
                options['num_predict'] = 3000  # More room for structured output
            
            # 🌐Ÿ HERE'S WHERE THE MAGIC HAPPENS, SWEETIE! 🌐Ÿ
            chat_kwargs = {
                'model': self.model_name,
                'messages': [
                    {
                        'role': 'system',
                        'content': (
                            'You are AiMerlion, a resume data extraction engine for Singapore/Malaysia recruitment.\n\n'
                            'EXTRACTION RULES:\n'
                            '1. Output ONLY valid JSON — no markdown fences, no explanations\n'
                            '2. Arrays must contain INDIVIDUAL items — never text blocks with bullets\n'
                            '3. Each work experience entry must be a SEPARATE JSON object\n'
                            '4. Extract EXACTLY as written — preserve original wording, never paraphrase\n'
                            '5. Use null for missing fields — never "N/A" or empty strings\n\n'
                            'FIELD DISAMBIGUATION (CRITICAL):\n'
                            '- SKILL = technical/soft ability: "Python", "Data Analysis", "Leadership"\n'
                            '- NOT SKILL = company name ("DBS Bank"), job title ("Software Engineer"),\n'
                            '  job duty ("Managed team"), certification ("CMFAS M5"), section header ("Skills")\n'
                            '- SG phones = 8 digits with optional +65. MY phones = 10-11 digits with +60\n'
                            '- Date-first experience: "Feb 2016 to Present [TAB] Role, Company"\n'
                            '- "Language : English & Chinese" in the header = language field, not section\n\n'
                            'Your output WILL be validated — formatting errors are rejected.'
                        )
                    },
                    {'role': 'user', 'content': prompt}
                ],
                'options': options,
            }
            if schema is not None:
                chat_kwargs['format'] = schema

            response = self._client.chat(**chat_kwargs)

            response_text = response['message']['content'] or ""
            # Strip Qwen3 <think>...</think> blocks before JSON parsing
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            response_text = re.sub(r'<think>.*$',         '', response_text, flags=re.DOTALL)
            response_text = response_text.strip()
            parsed_data = self._parse_ai_response(response_text)
            
            # 🌐Ÿ VALIDATION LAYER - Check if AI was lazy!
            if is_deep and parsed_data:
                parsed_data = self._validate_deep_extraction(parsed_data, response_text)
            
            return parsed_data
            
        except Exception as e:
            self.logger.error(f"âŒ AI Call error: {e}")
            return {}

    def _parse_ai_response(self, response_text: str) -> Optional[Dict]:
        """
        📧 Parse JSON and clean text dumps.
        """
        # 1. Clean Markdown
        cleaned = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        data = {}
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: Find JSON object
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except:
                    pass
        
        if not data:
            self.logger.warning("⚠ Could not parse JSON from AI response")
            return {}

        # 2. 🛡 SAFETY NET: Fix Malformed Lists
        # If the AI returned a bulleted string instead of a list, fix it now.
        data = self._fix_malformed_lists(data)
        
        return data

    def _fix_malformed_lists(self, data: Dict) -> Dict:
        """
        🚑 EMERGENCY ROOM for Malformed Data!
        This is like plastic surgery for badly extracted JSON, honey! 👪
        """
        self.logger.info("📧 Running malformed data repair...")
        
        # ========== FIX SKILLS ==========
        for field in ['hard_skills', 'soft_skills', 'skills']:
            if field in data:
                if isinstance(data[field], str):
                    self.logger.warning(f"⚠ {field} was a string - converting to list!")
                    raw_text = data[field]
                    
                    # AGGRESSIVE CLEANING - Remove ALL bullet characters
                    cleaned = re.sub(r'[•◗◗‹◗¦▪▫â– □►▸"£âƒ→·∙â‹…▶▷◗†◗‡â˜…☆\-"“"”*]', '', raw_text)
                    
                    # Split by newlines AND commas
                    items = re.split(r'[\n,;]+', cleaned)
                    
                    # Clean each item and filter
                    clean_list = []
                    for item in items:
                        cleaned_item = item.strip()
                        # Remove numbering like "1.", "2."
                        cleaned_item = re.sub(r'^\d+\.?\s*', '', cleaned_item)
                        # Only keep items with actual content
                        if cleaned_item and len(cleaned_item) > 2 and len(cleaned_item) < 100:
                            clean_list.append(cleaned_item)
                    
                    data[field] = clean_list
                    self.logger.info(f"âœ… Converted {field} to {len(clean_list)} items")
                
                elif isinstance(data[field], list):
                    # Even if it's a list, clean each item
                    cleaned_items = []
                    for item in data[field]:
                        if isinstance(item, str):
                            cleaned = item.strip().strip('•-””””*◗¦▪▫')
                            cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)
                            if cleaned and len(cleaned) > 2 and len(cleaned) < 100:
                                cleaned_items.append(cleaned)
                    data[field] = cleaned_items

        # ========== FIX EXPERIENCE ==========
        if 'working_experience' in data:
            if isinstance(data['working_experience'], str):
                self.logger.error("🚨 MAJOR ISSUE: AI dumped experience as text!")
                self.logger.warning("⚠ Attempting emergency extraction...")
                
                raw_text = data['working_experience']
                
                # Try to detect individual jobs by company names or date patterns
                # Look for patterns like "Company Name" followed by role and dates
                jobs = []
                
                # Split by common section separators
                sections = re.split(r'\n(?=[A-Z][a-zA-Z\s&,\.]+(?:Pvt\.|Ltd\.|Inc\.|Corp|Company))', raw_text)
                
                for section in sections:
                    if len(section.strip()) < 20:
                        continue
                    
                    # Try to extract company (first line usually)
                    lines = section.split('\n')
                    company = lines[0].strip() if lines else "Unknown Company"
                    
                    # Look for date patterns
                    date_match = re.search(r'(\w+\s+\d{4}\s*[-–“]\s*(?:\w+\s+\d{4}|Present))', section)
                    dates = date_match.group(1) if date_match else "N/A"
                    
                    # Role is often the second line or has keywords
                    role = "See Description"
                    for line in lines[1:4]:  # Check first few lines
                        if any(keyword in line.lower() for keyword in ['engineer', 'manager', 'developer', 'analyst', 'specialist', 'executive']):
                            role = line.strip()
                            break
                    
                    # Description is the rest - no truncation!
                    description = section
                    
                    jobs.append({
                        "company": company[:100],
                        "role": role[:100],
                        "dates": dates,
                        "description": description
                    })
                
                data['working_experience'] = jobs if jobs else [{
                    "company": "Parse Failed - Manual Review Needed",
                    "role": "Multiple Roles",
                    "dates": "See Description",
                    "description": raw_text
                }]
                
                self.logger.info(f"âœ… Extracted {len(data['working_experience'])} jobs from text dump")
            
            elif isinstance(data['working_experience'], list):
                # Validate each job object
                validated_jobs = []
                for job in data['working_experience']:
                    if isinstance(job, dict):
                        # Ensure required fields exist
                        validated_job = {
                            "company": str(job.get('company', 'Unknown'))[:200],
                            "role": str(job.get('role', 'N/A'))[:200],
                            "dates": str(job.get('dates', 'N/A')),
                            "description": str(job.get('description', ''))
                        }
                        validated_jobs.append(validated_job)
                
                data['working_experience'] = validated_jobs

        # ========== FIX EDUCATION ==========
        if 'education' in data:
            if isinstance(data['education'], str):
                self.logger.warning("⚠ Education was a string - attempting to structure...")
                data['education'] = [{
                    "institution": "See Description",
                    "degree": "Multiple Degrees",
                    "dates": "N/A",
                    "description": data['education'][:300]
                }]
            elif isinstance(data['education'], list):
                validated_edu = []
                for edu in data['education']:
                    if isinstance(edu, dict):
                        validated_edu.append({
                            "institution": str(edu.get('institution', 'Unknown'))[:150],
                            "degree": str(edu.get('degree', 'N/A'))[:150],
                            "dates": str(edu.get('dates', 'N/A'))
                        })
                data['education'] = validated_edu

        return data

    def _validate_deep_extraction(self, data: Dict, original_response: str) -> Dict:
        """
        🔍 THE QUALITY CONTROL DIVA checks if AI did its job properly!
        If not, she throws a FIT and demands a re-do! 💅
        """
        issues_found = []
        
        # Check 1: Are skills actually lists?
        for skill_field in ['hard_skills', 'soft_skills']:
            if skill_field in data:
                if isinstance(data[skill_field], str):
                    issues_found.append(f"{skill_field} is a string instead of array")
                elif isinstance(data[skill_field], list) and len(data[skill_field]) > 0:
                    # Check if first item looks like it has bullets
                    first_item = str(data[skill_field][0])
                    if any(char in first_item for char in ['•', '\n', '"“']):
                        issues_found.append(f"{skill_field} contains improperly formatted items")
        
        # Check 2: Is experience properly structured?
        if 'working_experience' in data:
            if isinstance(data['working_experience'], str):
                issues_found.append("working_experience is a string (should be array of objects)")
            elif isinstance(data['working_experience'], list):
                if len(data['working_experience']) > 0:
                    first_job = data['working_experience'][0]
                    if isinstance(first_job, dict):
                        # Check if description is suspiciously long (sign of text dump)
                        desc = first_job.get('description', '')
                        if len(desc) > 1000:
                            issues_found.append("Job description is too long (possible text dump)")
        
        # Check 3: Response length check - if response is > 80% of original text, likely a dump
        if len(original_response) > 5000:
            issues_found.append("Response is excessively long (possible text dump)")
        
        # Log issues
        if issues_found:
            self.logger.warning(f"⚠ VALIDATION ISSUES DETECTED:")
            for issue in issues_found:
                self.logger.warning(f"   ⚠ {issue}")
            self.logger.info("📧 Applying aggressive cleanup...")
        
        # Always run cleanup regardless
        return self._fix_malformed_lists(data)

    def validate_and_enhance(self, regex_results: Dict, text: str) -> Dict:
        """
        🔍 Merge Regex + AI results - COMPREHENSIVE VERSION

        Extracts ALL resume sections without paraphrasing:
        - Header fields (name, email, phone, etc.)
        - Skills (hard & soft)
        - Work Experience (with FULL descriptions)
        - Education
        - Summary/Objective
        - Certifications
        - Languages
        - Projects
        - Achievements
        - References
        - Hobbies/Interests
        """
        if not self.available:
            return regex_results
        
        # 🧚‍♀️ NORMALIZE TEXT FIRST! This is THE key to consistent extraction!
        text = self._normalize_text_for_extraction(text)

        final_results = regex_results.copy()

        # Pass 1: Headers
        ai_header = self.extract_header_fields(text)
        if ai_header:
            for field in ['name', 'email', 'phone', 'location', 'linkedin', 'website']:
                if ai_header.get(field):
                    final_results[field] = ai_header[field]
                    final_results[f"{field}_source"] = "AI_Header"

        # Pass 2: Deep Fields (COMPREHENSIVE)
        ai_deep = self.extract_deep_fields(text)
        if ai_deep:
            # Skills
            final_results['hard_skills'] = ai_deep.get('hard_skills', [])
            final_results['soft_skills'] = ai_deep.get('soft_skills', [])

            # Combine for backwards compatibility
            final_results['skills'] = final_results['hard_skills'] + final_results['soft_skills']

            # Core sections
            final_results['working_experience'] = ai_deep.get('working_experience', [])
            final_results['education'] = ai_deep.get('education', [])

            # NEW: Additional sections (extracted VERBATIM)
            final_results['summary'] = ai_deep.get('summary', '')
            final_results['certifications'] = ai_deep.get('certifications', [])
            final_results['languages'] = ai_deep.get('languages', [])
            final_results['projects'] = ai_deep.get('projects', [])
            final_results['achievements'] = ai_deep.get('achievements', [])
            final_results['references'] = ai_deep.get('references', [])
            final_results['hobbies'] = ai_deep.get('hobbies', [])

            final_results['experience_source'] = "AI_Deep"

        return self._post_process_results(final_results)

    def _post_process_results(self, results: Dict) -> Dict:
        """
        🎨 Final Cleanup - COMPREHENSIVE VERSION

        Ensures all extracted fields are properly formatted and validated.
        """
        processed = results.copy()

        # Phone Standardizer
        if processed.get('phone'):
            val = processed['phone']
            if isinstance(val, list): val = val[0]
            if isinstance(val, str):
                processed['phone'] = standardize_phone_number(val)

        # Name Cleaner
        if processed.get('name'):
            processed['name'] = self._clean_name(processed['name'])

        # Ensure List fields are properly initialized
        list_fields = [
            'hard_skills', 'soft_skills', 'working_experience', 'education',
            'certifications', 'languages', 'projects', 'achievements',
            'references', 'hobbies'
        ]
        for f in list_fields:
            if f not in processed or not isinstance(processed[f], list):
                processed[f] = []

        # Ensure String fields are properly initialized
        if 'summary' not in processed or not isinstance(processed['summary'], str):
            processed['summary'] = ""

        # Combine skills for backwards compatibility
        if 'skills' not in processed:
            processed['skills'] = processed.get('hard_skills', []) + processed.get('soft_skills', [])

        processed['ai_enhanced'] = True
        processed['extraction_version'] = "2.0_comprehensive"

        processed = self._validate_extracted_data(processed)

        return processed

    def _clean_name(self, name: str) -> str:
        if not isinstance(name, str): return ""
        name = name.strip()
        name = re.sub(r'^(Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.|Eng\.)\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r',\s*(PhD|MD|MBA|M\.Sc|B\.Sc)$', '', name, flags=re.IGNORECASE)
        return name
    
    def _clean_markdown(self, text: str) -> str:
        """
        🧹 Remove markdown formatting that confuses regex!
        🎭 FAIRY CODEMOTHER'S ENHANCED VERSION v2.0!
        Now handles **bold**, *italic*, and markdown headers properly!
        """
        if not text:
            return ""
        
        # 🏆• STEP 0: Normalize en-dash and em-dash to hyphen FIRST!
        text = text.replace('"“', '-')   # en-dash (U+2013)
        text = text.replace('"”', '-')   # em-dash (U+2014)
        
        # Remove markdown headers (##, ###, etc.) - must come first!
        # 🏆• Also remove the ** around header text!
        text = re.sub(r'^#+\s*\**([^*\n]+)\**\s*$', r'\1', text, flags=re.MULTILINE)
        
        # Remove bold markers (**text** or __text__)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        
        # Remove italic markers (*text* or _text_) - be careful not to remove bullet points!
        # Only match if there's text between the asterisks (not at line start)
        text = re.sub(r'(?<!\n)\*([^*\n]+)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!\n)_([^_\n]+)_(?!_)', r'\1', text)
        
        # 🏆• Handle bold+italic (***text***)
        text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
        
        # Remove bullet point markers at line start, but KEEP the content!
        text = re.sub(r'^\s*[-\*•▪►▸◗¦◗‹◗‡]\s+', '', text, flags=re.MULTILINE)
        
        # Normalize special quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove extra whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _detect_section_boundaries_enhanced(self, text: str) -> Dict[str, Tuple[int, int]]:
        """
        🎭 FAIRY CODEMOTHER'S ENHANCED SECTION DETECTOR v3.0! 💅
        
        This is like a GPS for your resume - it knows EXACTLY where
        each section starts and ends! 🗺️
        
        IMPROVEMENTS:
        - More specific patterns to avoid false matches
        - Better handling of compound headers ("Work Experience" vs "Experience")
        - Prevents sections from overlapping
        - Validates that boundaries make sense
        - Handles Asian resume formats (common in Singapore!)
        
        REPLACE the existing _detect_section_boundaries method (lines 3337-3385)
        with this enhanced version!
        """
        sections = {}
        
        # 🎯 ENHANCED Section patterns with PRIORITY ORDER!
        # More specific patterns come FIRST to avoid false matches!
        # This is like VIP entry - the most important guests get in first! 👑
        
        section_patterns = [
            # PERSONAL INFORMATION (sometimes at top of Asian resumes)
            ('personal_info', r'(?:^|\n)\s*PERSONAL\s+(?:INFORMATION|PARTICULARS|DETAILS)\s*(?:[:\n])'),
            
            # PROFESSIONAL SUMMARY (very specific to avoid matching "Summary" in job descriptions)
            ('summary', r'(?:^|\n)\s*(?:PROFESSIONAL\s+)?(?:SUMMARY|PROFILE|CAREER\s+(?:SUMMARY|PROFILE)|OBJECTIVE|ABOUT\s*ME|PERSONAL\s+STATEMENT)\s*(?:[:\n])'),
            
            # CORE COMPETENCIES / SKILLS
            ('skills', r'(?:^|\n)\s*(?:CORE\s+COMPETENCIES|KEY\s+SKILLS?|TECHNICAL\s+SKILLS?|PROFESSIONAL\s+SKILLS?|SKILLS?\s+(?:SUMMARY|PROFILE)?|COMPETENCIES)\s*(?:[:\n])'),
            
            # WORK EXPERIENCE (match the FULL phrase first!)
            ('experience', r'(?:^|\n)\s*(?:WORK\s+EXPERIENCE|WORKING\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT\s+(?:HISTORY|EXPERIENCE)|CAREER\s+(?:HISTORY|EXPERIENCE))\s*(?:[:\n])'),
            
            # Just "EXPERIENCE" (lower priority - match only if above patterns don't match)
            ('experience_simple', r'(?:^|\n)\s*EXPERIENCE[S]?\s*(?:[:\n])'),
            
            # EDUCATION (very specific to avoid matching "Education" in job descriptions!)
            ('education', r'(?:^|\n)\s*EDUCATION(?:AL)?(?:\s+(?:BACKGROUND|HISTORY|QUALIFICATIONS?|AND\s+TRAINING))?\s*(?:[:\n])'),
            
            # QUALIFICATIONS (separate from education)
            ('qualifications', r'(?:^|\n)\s*(?:PROFESSIONAL\s+)?QUALIFICATIONS?\s*(?:[:\n])'),
            
            # CERTIFICATIONS
            ('certifications', r'(?:^|\n)\s*(?:CERTIFICATIONS?|PROFESSIONAL\s+CERTIFICATIONS?|LICENSES?|CREDENTIALS?)\s*(?:[:\n])'),
            
            # ACHIEVEMENTS / AWARDS
            ('achievements', r'(?:^|\n)\s*(?:ACHIEVEMENTS?|AWARDS?|ACCOMPLISHMENTS?|HONORS?|RECOGNITIONS?)\s*(?:[:\n])'),
            
            # CO-CURRICULAR ACTIVITIES (common in Singapore resumes!)
            ('cocurricular', r'(?:^|\n)\s*CO-?CURRICULAR(?:\s+ACTIVITIES)?\s*(?:[:\n])'),
            
            # PROJECTS
            ('projects', r'(?:^|\n)\s*(?:PROJECTS?|KEY\s+PROJECTS?|PERSONAL\s+PROJECTS?)\s*(?:[:\n])'),
            
            # LANGUAGES
            ('languages', r'(?:^|\n)\s*LANGUAGES?\s*(?:[:\n])'),
            
            # REFERENCES
            ('references', r'(?:^|\n)\s*REFERENCES?\s*(?:[:\n])'),
            
            # HOBBIES / INTERESTS
            ('hobbies', r'(?:^|\n)\s*(?:HOBBIES?|INTERESTS?)\s*(?:[:\n])'),
        ]
        
        # 📍 STEP 1: Find all section starts
        found_sections = []
        
        for section_name, pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            
            for match in matches:
                # Get the matched header text for logging
                header_text = text[match.start():match.end()].strip()
                
                found_sections.append({
                    'name': section_name,
                    'start': match.start(),
                    'header_end': match.end(),
                    'header_text': header_text
                })
                
                self.logger.debug(f"📝 Found section '{section_name}' at position {match.start()}: '{header_text}'")
        
        # 📍 STEP 2: Sort by position and remove duplicates
        # If multiple patterns match the same position, keep the most specific one
        found_sections.sort(key=lambda x: (x['start'], -len(x['header_text'])))
        
        # Remove duplicates (sections within 50 chars of each other are considered duplicates)
        unique_sections = []
        last_position = -100
        
        for section in found_sections:
            if section['start'] - last_position > 50:
                unique_sections.append(section)
                last_position = section['start']
            else:
                self.logger.debug(f"⚠️ Skipping duplicate section at {section['start']}: {section['header_text']}")
        
        # 📍 STEP 3: Calculate boundaries with VALIDATION
        # Each section ends where the next section begins!
        # This is like setting up VIP sections at a club! 🎉
        
        for i, section in enumerate(unique_sections):
            section_name = section['name']
            content_start = section['header_end']
            
            # Determine where this section ends
            if i + 1 < len(unique_sections):
                # End at the START of the next section header (not the end!)
                content_end = unique_sections[i + 1]['start']
            else:
                # Last section extends to end of document
                content_end = len(text)
            
            # 🛡️ VALIDATION: Section should have some content!
            content_length = content_end - content_start
            
            if content_length < 10:
                self.logger.warning(f"⚠️ Section '{section_name}' is too short ({content_length} chars) - skipping")
                continue
            
            if content_length > 50000:
                self.logger.warning(f"⚠️ Section '{section_name}' is suspiciously long ({content_length} chars)")
                # Still include it but log the warning
            
            sections[section_name] = (content_start, content_end)
            self.logger.debug(f"📝 Section '{section_name}': chars {content_start}-{content_end} ({content_length} chars)")
        
        # 📍 STEP 4: SPECIAL HANDLING for merged "experience" sections
        # If we found both 'experience' and 'experience_simple', keep only the first one
        if 'experience' in sections and 'experience_simple' in sections:
            # Keep whichever comes first
            if sections['experience'][0] < sections['experience_simple'][0]:
                del sections['experience_simple']
                self.logger.debug("📝 Removed duplicate 'experience_simple' section")
            else:
                sections['experience'] = sections['experience_simple']
                del sections['experience_simple']
                self.logger.debug("📝 Using 'experience_simple' as main experience section")
        elif 'experience_simple' in sections:
            # Rename to 'experience' for consistency
            sections['experience'] = sections['experience_simple']
            del sections['experience_simple']
        
        # Merge experience_simple into experience (handled by text_preprocessor already,
        # but guard here in case the legacy path produced both)
        if 'experience' in sections and 'experience_simple' in sections:
            if sections['experience'][0] <= sections['experience_simple'][0]:
                del sections['experience_simple']
            else:
                sections['experience'] = sections['experience_simple']
                del sections['experience_simple']
        elif 'experience_simple' in sections:
            sections['experience'] = sections['experience_simple']
            del sections['experience_simple']

        self.logger.info(f"Section detection complete. Found {len(sections)} sections.")
        return sections

    def detect_sections_ai(self, text: str) -> Dict[str, str]:
        """
        🎭 AI-DRIVEN SECTION DETECTION!
        Uses the LLM to intelligently segment the resume into logical blocks.
        """
        if not self.available:
            self.logger.warning("⚠️ AI not available for section detection — using regex fallback")
            boundaries = _pp_detect_sections(text)
            return {name: text[start:end].strip() for name, (start, end) in boundaries.items()}

        self.logger.info("🔍 Detecting sections via AI...")
        
        # Limit text sample for detection
        text_sample = text[:15000] if len(text) > 15000 else text
        
        prompt = f"""Identify all logical sections in this resume and return them as a JSON object.
        Extract the EXACT text for each section.
        
        **REQUIRED KEYS:**
        - summary: Professional summary/objective
        - skills: Technical and soft skills
        - experience: Work history/employment
        - education: Academic background
        - certifications: Licenses/certifications
        - languages: Spoken/written languages
        - others: Projects, hobbies, references, etc.
        
        Resume text:
        ---
        {text_sample}
        ---
        
        Return ONLY valid JSON."""

        try:
            # Use _call_ollama_raw with schema for reliability
            response_text = self._call_ollama_raw(prompt, num_predict=2000, schema=SECTION_SCHEMA)

            # Guard: model returned nothing usable (common with Qwen3 reasoning models
            # when num_predict is consumed by <think> blocks)
            if not response_text or not response_text.strip():
                self.logger.warning(
                    "⚠️ AI section detector returned empty content "
                    "(likely Qwen3 reasoning burned all tokens) — using regex fallback"
                )
                boundaries = _pp_detect_sections(text)
                return {name: text[start:end].strip() for name, (start, end) in boundaries.items()}

            # Direct JSON parse
            sections_raw = json.loads(response_text)

            # Filter out null values
            sections = {k: v for k, v in sections_raw.items() if v}

            self.logger.info(f"✨ AI found {len(sections)} sections!")
            return sections

        except Exception as e:
            # Fallback to regex-based detection so callers always get usable sections
            self.logger.warning(f"⚠️ AI section detection failed ({e}) — using regex fallback")
            boundaries = _pp_detect_sections(text)
            return {name: text[start:end].strip() for name, (start, end) in boundaries.items()}

    def _detect_section_boundaries_enhanced(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Delegates to extraction.text_preprocessor.detect_sections()."""
        boundaries = _pp_detect_sections(text)
        self.logger.info(f"Section detection complete. Found {len(boundaries)} sections.")
        return boundaries

    def _extract_section_content(self, text: str, section_name: str) -> str:
        """
        🎯 Extract content from a specific section safely!
        
        This is like getting the PERFECT slice of cake - 
        not too much frosting, not too little! 🍰
        
        This is a NEW helper method - add it after _detect_section_boundaries_enhanced!
        """
        sections = self._detect_section_boundaries_enhanced(text)
        
        if section_name not in sections:
            self.logger.debug(f"⚠️ Section '{section_name}' not found")
            return ""
        
        start, end = sections[section_name]
        content = text[start:end].strip()
        
        # 🛡️ Additional cleaning - remove the section header if it's still there
        # Sometimes the header is included in the content
        header_patterns = [
            r'^(?:PROFESSIONAL\s+)?(?:SUMMARY|PROFILE|OBJECTIVE|SKILLS?|EXPERIENCE|EDUCATION|CERTIFICATIONS?|LANGUAGES?|PROJECTS?|ACHIEVEMENTS?|REFERENCES?)\s*[:\n]\s*',
        ]
        
        for pattern in header_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()

    def _prevent_section_bleeding(self, text: str, primary_section: str, stop_sections: List[str]) -> str:
        """
        🛡️ PREVENT content from bleeding between sections!
        
        This is like having BOUNCERS between different sections of a club -
        they make sure nobody wanders into the VIP area! 💂
        
        Example usage:
        work_exp_text = self._prevent_section_bleeding(text, 'experience', 
                                                        ['education', 'certifications', 'qualifications'])
        
        This ensures work experience stops BEFORE education/certifications begin!
        
        ADD this as a NEW method after _extract_section_content!
        """
        sections = self._detect_section_boundaries_enhanced(text)
        
        if primary_section not in sections:
            return ""
        
        primary_start, primary_end = sections[primary_section]
        
        # Find the EARLIEST stop section that appears after our primary section
        earliest_stop = primary_end
        
        for stop_section in stop_sections:
            if stop_section in sections:
                stop_start, _ = sections[stop_section]
                if primary_start < stop_start < earliest_stop:
                    earliest_stop = stop_start
                    self.logger.debug(f"📍 Limiting '{primary_section}' at '{stop_section}' boundary")
        
        # Extract content with the enforced boundary
        content = text[primary_start:earliest_stop].strip()
        
        return content

    def _validate_extracted_data(self, results: Dict) -> Dict:
        """
        🎭 THE ULTIMATE QUALITY CONTROL METHOD! 🎭
        
        This is like having a STRICT fashion police checking every outfit
        before it goes on the runway! 💃
        
        Ensures:
        - Work experience doesn't leak into education
        - Skills are actually skills (not job descriptions!)
        - Dates are in the right format
        - Company names aren't bullet points
        - Certifications aren't job responsibilities
        
        ADD THIS METHOD after _post_process_results (after line 3285)
        CALL THIS METHOD at the end of _post_process_results before returning!
        """
        validated = results.copy()
        
        # ===== VALIDATION 1: Clean up Skills =====
        # Remove any skills that are actually sentences or job descriptions
        # This is like removing fake eyelashes that look too natural! 👁️
        validated['hard_skills'] = self._validate_skills_list(validated.get('hard_skills', []))
        validated['soft_skills'] = self._validate_skills_list(validated.get('soft_skills', []))
        
        # ===== VALIDATION 2: Validate Work Experience =====
        # Make sure experience entries have proper structure
        # This is like checking if a dress has all its zippers! 👗
        validated['working_experience'] = self._validate_experience_entries(
            validated.get('working_experience', [])
        )
        
        # ===== VALIDATION 3: Validate Education =====
        # Make sure education entries don't contain work experience
        # This is like making sure your homework doesn't end up in your lunch! 🎒
        validated['education'] = self._validate_education_entries(
            validated.get('education', [])
        )
        
        # ===== VALIDATION 4: Validate Certifications =====
        # Make sure certifications are actual certifications, not job duties!
        validated['certifications'] = self._validate_certifications(
            validated.get('certifications', [])
        )
        
        # ===== VALIDATION 5: Validate Languages =====
        # Make sure languages are actual languages, not company names!
        validated['languages'] = self._validate_languages(
            validated.get('languages', [])
        )
        
        self.logger.info("✅ Data validation complete!")
        
        return validated

    def _validate_skills_list(self, skills: List[str]) -> List[str]:
        """
        💅 Validate that skills are ACTUALLY skills!
        
        This is like checking if someone's claiming to know "walking in heels"
        vs "I walked in heels to the store yesterday" 👠
        
        Skills should be:
        - Short (typically 1-5 words)
        - Not full sentences
        - Not job descriptions
        - Not bullet points or formatting artifacts
        """
        if not isinstance(skills, list):
            return []
        
        validated_skills = []
        
        for skill in skills:
            if not isinstance(skill, str):
                continue
            
            skill = skill.strip()
            
            # ❌ REJECT: Empty or too short
            if len(skill) < 2:
                continue
            
            # ❌ REJECT: Too long (probably a sentence/description)
            if len(skill) > 80:
                self.logger.debug(f"⚠️ Skipping long 'skill': {skill[:50]}...")
                continue
            
            # ❌ REJECT: Contains action verbs (signs of job description)
            action_verbs = [
                'managed', 'developed', 'created', 'led', 'coordinated',
                'implemented', 'designed', 'built', 'worked on', 'responsible for',
                'conducted', 'performed', 'achieved', 'established', 'maintained',
                'executed', 'delivered', 'collaborated', 'supported', 'assisted'
            ]
            if any(verb in skill.lower() for verb in action_verbs):
                self.logger.debug(f"⚠️ Skipping description as skill: {skill[:50]}")
                continue
            
            # ❌ REJECT: Contains bullet point indicators
            if any(char in skill for char in ['•', '◆', '▪', '►', '✓']):
                self.logger.debug(f"⚠️ Skipping bullet artifact: {skill[:30]}")
                continue
            
            # ❌ REJECT: Looks like a sentence (has multiple clauses)
            if skill.count(',') > 2 or any(phrase in skill.lower() for phrase in ['in order to', 'such as', 'including', 'for example']):
                self.logger.debug(f"⚠️ Skipping sentence as skill: {skill[:50]}")
                continue
            
            # ✅ ACCEPT: Looks like a valid skill!
            validated_skills.append(skill)
        
        self.logger.info(f"📊 Skills validated: {len(skills)} → {len(validated_skills)}")
        return validated_skills

    def _validate_experience_entries(self, experiences: List[Dict]) -> List[Dict]:
        """
        💼 Validate work experience entries!
        
        This is like checking if everyone at a job fair actually HAS a job
        and isn't just pretending! 🎭
        
        Each entry should have:
        - Valid company name (not a bullet point!)
        - Valid role title (not a description!)
        - Dates that make sense
        - Description that's not too short or too long
        """
        if not isinstance(experiences, list):
            return []
        
        validated = []
        
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            
            # Get fields
            company = exp.get('company', '').strip()
            role = exp.get('role', '').strip()
            dates = exp.get('dates', '').strip()
            description = exp.get('description', '').strip()
            
            # ===== VALIDATE COMPANY NAME =====
            # Company should not be empty or a bullet point
            if not company or len(company) < 2:
                self.logger.warning(f"⚠️ Skipping experience with invalid company: {company}")
                continue
            
            # Company should not start with action verbs (sign of misclassification)
            first_word = company.split()[0].lower() if company.split() else ''
            if first_word in ['managed', 'developed', 'led', 'created', 'implemented', 'designed', 'conducted']:
                self.logger.warning(f"⚠️ Company looks like a verb phrase: {company[:50]}")
                continue
            
            # ===== VALIDATE ROLE =====
            # Role should not be extremely long (might be a description)
            if len(role) > 150:
                self.logger.warning(f"⚠️ Role too long: {role[:50]}...")
                role = role[:150]  # Truncate
            
            # ===== VALIDATE DATES =====
            # Dates should contain year numbers
            if dates and not re.search(r'\d{4}', dates):
                self.logger.warning(f"⚠️ Dates don't contain year: {dates}")
                dates = "Dates not specified"
            
            # ===== CLEAN DESCRIPTION =====
            # Remove excessive whitespace and formatting
            if description:
                description = re.sub(r'\s+', ' ', description).strip()
                # Truncate if too long (but keep reasonable length)
                if len(description) > 50000:
                    description = description[:50000] + "..."
            
            # ✅ Add validated entry
            validated.append({
                'company': company,
                'role': role,
                'dates': dates,
                'description': description
            })
        
        self.logger.info(f"💼 Experience validated: {len(experiences)} → {len(validated)} entries")
        return validated

    def _validate_education_entries(self, education: List[Dict]) -> List[Dict]:
        """
        🎓 Validate education entries!
        
        This is like checking if someone's degree is REAL and not just
        "University of Hard Knocks"! 🎓😂
        
        Education should NOT contain:
        - Work experience bleeding in
        - Company names
        - Job titles
        - Action verbs from job descriptions
        """
        if not isinstance(education, list):
            return []
        
        validated = []
        
        for edu in education:
            if not isinstance(edu, dict):
                continue
            
            institution = edu.get('institution', '').strip()
            degree = edu.get('degree', '').strip()
            dates = edu.get('dates', '').strip()
            
            # ===== VALIDATE INSTITUTION =====
            # Should not contain "Pte Ltd", "Company", "Consultant" (signs of work experience!)
            job_indicators = ['pte ltd', 'pvt ltd', 'company', 'consultant', 'corporation', 'corp', 'inc']
            institution_lower = institution.lower()
            
            if any(indicator in institution_lower for indicator in job_indicators):
                self.logger.warning(f"⚠️ Institution looks like a company: {institution[:50]}")
                continue
            
            # Should not start with action verbs
            first_word = institution.split()[0].lower() if institution.split() else ''
            if first_word in ['managed', 'worked', 'developed', 'led', 'coordinated']:
                self.logger.warning(f"⚠️ Institution starts with action verb: {institution[:50]}")
                continue
            
            # ===== VALIDATE DEGREE =====
            # Should not be excessively long (might be job description)
            if len(degree) > 300:
                self.logger.warning(f"⚠️ Degree field too long: {degree[:50]}...")
                degree = degree[:300]
            
            # Should not contain multiple job-like action verbs
            job_verbs = ['responsible', 'managed', 'coordinated', 'developed', 'implemented']
            verb_count = sum(1 for verb in job_verbs if verb in degree.lower())
            if verb_count >= 2:
                self.logger.warning(f"⚠️ Degree contains multiple action verbs: {degree[:50]}")
                continue
            
            # ✅ Add validated entry
            validated.append({
                'institution': institution,
                'degree': degree,
                'dates': dates
            })
        
        self.logger.info(f"🎓 Education validated: {len(education)} → {len(validated)} entries")
        return validated

    def _validate_certifications(self, certifications: List[Dict]) -> List[Dict]:
        """
        🏅 Validate certifications!
        
        This is like checking if someone's awards are REAL awards
        and not just "Employee of the Month at McDonald's"! 🏆
        
        Certifications should be:
        - Actual certification names
        - Not job responsibilities
        - Not action phrases
        """
        if not isinstance(certifications, list):
            return []
        
        validated = []
        
        for cert in certifications:
            if not isinstance(cert, dict):
                continue
            
            name = cert.get('name', '').strip()
            
            # ===== VALIDATE CERTIFICATION NAME =====
            if not name or len(name) < 3:
                continue
            
            # Should not start with action verbs (signs of job duty!)
            first_word = name.split()[0].lower() if name.split() else ''
            action_verbs = [
                'managed', 'coordinated', 'developed', 'maintained', 'conducted',
                'performed', 'assisted', 'supported', 'implemented', 'established',
                'ensured', 'carried', 'worked', 'responsible', 'led'
            ]
            
            if first_word in action_verbs:
                self.logger.warning(f"⚠️ Cert name starts with action verb: {name[:50]}")
                continue
            
            # Should not contain phrases indicating job duties
            duty_phrases = ['responsible for', 'in charge of', 'worked on', 'assisted with']
            if any(phrase in name.lower() for phrase in duty_phrases):
                self.logger.warning(f"⚠️ Cert looks like job duty: {name[:50]}")
                continue
            
            # ✅ Add validated certification
            validated.append(cert)
        
        self.logger.info(f"🏅 Certifications validated: {len(certifications)} → {len(validated)}")
        return validated

    def _validate_languages(self, languages: List[Dict]) -> List[Dict]:
        """
        🌐 Validate languages!
        
        This is like checking if someone speaks "English" not "English Consultant"! 🗣️
        
        Languages should:
        - Be actual language names
        - Not contain company names
        - Not contain dates or job information
        """
        if not isinstance(languages, list):
            return []
        
        # Known language names for validation
        known_languages = [
            'english', 'chinese', 'mandarin', 'cantonese', 'malay', 'tamil',
            'hindi', 'japanese', 'korean', 'french', 'german', 'spanish',
            'italian', 'portuguese', 'russian', 'arabic', 'indonesian', 'thai',
            'vietnamese', 'tagalog', 'bengali', 'punjabi', 'hokkien', 'teochew',
            'hakka', 'dutch', 'polish', 'turkish', 'swedish', 'norwegian', 'danish'
        ]
        
        validated = []
        
        for lang in languages:
            if not isinstance(lang, dict):
                continue
            
            language = lang.get('language', '').strip()
            proficiency = lang.get('proficiency', '').strip()
            
            # ===== VALIDATE LANGUAGE NAME =====
            if not language or len(language) < 2:
                continue
            
            language_lower = language.lower()
            
            # Should contain at least ONE known language name
            if not any(known_lang in language_lower for known_lang in known_languages):
                self.logger.warning(f"⚠️ Doesn't look like a language: {language[:50]}")
                continue
            
            # Should NOT contain company indicators
            company_indicators = ['pte ltd', 'pvt ltd', 'company', 'consultant', 'corporation']
            if any(indicator in language_lower for indicator in company_indicators):
                self.logger.warning(f"⚠️ Language contains company indicator: {language[:50]}")
                continue
            
            # Should NOT contain dates (sign of job entry bleeding in!)
            if re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}', language, re.IGNORECASE):
                self.logger.warning(f"⚠️ Language contains dates: {language[:50]}")
                continue
            
            # Should NOT contain action verbs
            if any(verb in language_lower for verb in ['managed', 'worked', 'conducted', 'carried']):
                self.logger.warning(f"⚠️ Language contains action verbs: {language[:50]}")
                continue
            
            # ✅ Add validated language
            validated.append({
                'language': language,
                'proficiency': proficiency
            })
        
        self.logger.info(f"🌐 Languages validated: {len(languages)} → {len(validated)}")
        return validated
