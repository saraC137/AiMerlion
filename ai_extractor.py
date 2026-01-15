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
        
        self.available = self._test_model()
        
    def _test_model(self) -> bool:
        """Test if the Ollama model is ready to serve!"""
        try:
            self.logger.info(f"🌸 Initializing {self.model_name} model...")
            ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': 'Hi'}])
            self.logger.info(f"✨ {self.model_name} is READY! Extraction engines online!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Model error: {e}")
            self.logger.error(f"Please ensure the model '{self.model_name}' is pulled in Ollama.")
            return False
    
    def _extract_email_regex(self, text: str) -> Optional[str]:
        """
        📧 Extract email using PURE REGEX - bulletproof extraction!
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
                self.logger.info(f"📧 Regex found email: {email}")
                return email

        return None

    def _is_valid_email(self, email: str) -> bool:
        """
        ✅ Validate that a string is actually an email address format.
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

    def _find_contact_area(self, text: str) -> str:
        """
        📍 Find the contact/header area of the resume (usually first 2000 chars).
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
        📝 Extract professional summary/objective section VERBATIM.
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
                    self.logger.info(f"📝 Found summary: {len(summary)} chars")
                    return summary[:2000]  # Allow longer summaries

        return ""

    def _extract_certifications_regex(self, text: str) -> List[Dict[str, str]]:
        """
        🏆 Extract certifications/licenses VERBATIM.
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
        lines = re.split(r'\n|[•●○▪■►]', cert_text)

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

        self.logger.info(f"🏆 Found {len(certifications)} certifications")
        return certifications

    def _extract_languages_regex(self, text: str) -> List[Dict[str, str]]:
        """
        🌐 Extract languages with proficiency levels VERBATIM.
        """
        languages = []

        # Find languages section
        lang_patterns = [
            r'(?:^|\n)\s*(?:LANGUAGES?|LANGUAGE\s+SKILLS?|LINGUISTIC\s+SKILLS?)\s*[:\n]\s*(.*?)(?=\n\s*(?:EXPERIENCE|EDUCATION|SKILLS|EMPLOYMENT|WORK|PROJECTS?|CERTIFICATIONS?|AWARDS?|ACHIEVEMENTS?|REFERENCES?|HOBBIES?|INTERESTS?)\s*(?:[:|\n]|$)|$)',
        ]

        lang_text = ""
        for pattern in lang_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                lang_text = match.group(1)
                break

        if not lang_text:
            # Try inline pattern: "Languages: English (Native), Mandarin (Fluent)"
            inline_match = re.search(r'Languages?\s*[:\-]\s*([^\n]+)', text, re.IGNORECASE)
            if inline_match:
                lang_text = inline_match.group(1)

        if not lang_text:
            return languages

        # Split by common delimiters
        items = re.split(r'[,;|•●○▪■►\n]', lang_text)

        for item in items:
            item = item.strip()
            if len(item) < 2:
                continue

            # Skip section headers
            if re.match(r'^languages?$', item, re.IGNORECASE):
                continue

            lang_entry = {"language": item, "proficiency": ""}

            # Try to extract proficiency level
            prof_match = re.search(r'\(?(\s*(?:Native|Fluent|Advanced|Intermediate|Basic|Beginner|Professional|Working|Conversational|Elementary|Mother\s+Tongue|Bilingual|C2|C1|B2|B1|A2|A1)\s*)\)?', item, re.IGNORECASE)
            if prof_match:
                lang_entry["proficiency"] = prof_match.group(1).strip()
                # Clean language name by removing proficiency
                lang_entry["language"] = re.sub(r'\(?\s*(?:Native|Fluent|Advanced|Intermediate|Basic|Beginner|Professional|Working|Conversational|Elementary|Mother\s+Tongue|Bilingual|C2|C1|B2|B1|A2|A1)\s*\)?', '', item, flags=re.IGNORECASE).strip(' -–:')

            if lang_entry["language"]:
                languages.append(lang_entry)

        self.logger.info(f"🌐 Found {len(languages)} languages")
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
            project_name = lines[0].strip().strip('•●○▪■►-*')

            # Skip if it looks like a section header
            if re.match(r'^projects?$', project_name, re.IGNORECASE):
                continue

            description = ' '.join(line.strip().strip('•●○▪■►-*') for line in lines[1:] if line.strip())

            # Try to extract technologies used
            tech_match = re.search(r'(?:Technologies?|Tech\s*Stack|Built\s+with|Using)\s*[:\-]?\s*([^\n]+)', entry, re.IGNORECASE)
            technologies = tech_match.group(1).strip() if tech_match else ""

            # Try to extract date/duration
            date_match = re.search(r'(\d{4}(?:\s*[-–]\s*\d{4})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})', entry, re.IGNORECASE)
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
        🏅 Extract achievements/awards/honors VERBATIM.
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
        items = re.split(r'\n|[•●○▪■►]', ach_text)

        for item in items:
            item = item.strip().strip('-*')
            if len(item) < 5:
                continue

            # Skip section headers
            if re.match(r'^(achievements?|awards?|honors?)$', item, re.IGNORECASE):
                continue

            achievements.append(item)

        self.logger.info(f"🏅 Found {len(achievements)} achievements")
        return achievements[:50]  # Max 50 achievements

    def _extract_references_regex(self, text: str) -> List[Dict[str, str]]:
        """
        📞 Extract references VERBATIM if present.
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

        self.logger.info(f"📞 Found {len(references)} references")
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
        items = re.split(r'[,\n]|[•●○▪■►]', hobby_text)

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
        🎯 PASS 1: Extract HEADER fields (Personal Info).
        Fast execution on the first 3000 characters.
        Now with email validation and regex fallback!
        """
        if not self.available:
            return {}

        # 📧 FIRST: Try regex extraction for email (most reliable!)
        regex_email = self._extract_email_regex(text)

        text_sample = text[:6000] if len(text) > 6000 else text

        prompt = f"""You are a precise data parser. Extract the candidate's contact details from the resume header.

**Required fields:**
1. name (Full Name)
2. email
3. phone
4. date_of_birth (DOB, Date of Birth, Birth Date - format as DD/MM/YYYY)
5. location (City, Country)
6. linkedin (URL)
7. website (URL)

**JSON Output Format ONLY:**
{{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "555-0199",
  "date_of_birth": "15/01/1990",
  "location": "New York, USA",
  "linkedin": "linkedin.com/in/johndoe",
  "website": "null"
}}

Resume text:
{text_sample}

Response must be valid JSON only."""

        result = self._call_ollama(prompt)

        # 📧 VALIDATE EMAIL: If AI returned garbage, use regex result
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

        return result

    def extract_deep_fields(self, text: str) -> Dict[str, Any]:
        """
        🚀 MANUAL-FIRST EXTRACTION with AI assistance only for disambiguation
        Because AI can't be trusted with structure, honey! 💅
        """
        if not self.available:
            return self._pure_manual_extraction(text)
        
        self.logger.info("🎭 Starting MANUAL-FIRST extraction...")
        
        # ALWAYS start with manual extraction
        result = self._pure_manual_extraction(text)
        
        # ONLY use AI to enhance/disambiguate specific fields if needed
        if self._needs_ai_enhancement(result):
            self.logger.info("🤖 Using AI for skill categorization only...")
            result = self._enhance_skills_with_ai(result, text)
        
        self.logger.info(f"✅ Extraction complete: {len(result.get('hard_skills', []))} hard skills, {len(result.get('working_experience', []))} jobs")
        
        return result

    def _pure_manual_extraction(self, text: str) -> Dict[str, Any]:
        """
        🛠️ PURE MANUAL EXTRACTION - No AI involved!
        This is the BACKBONE, honey! 💪

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
        self.logger.info("🔧 Running COMPREHENSIVE manual extraction...")

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
        result['languages'] = self._extract_languages_regex(text)

        # ===== EXTRACT PROJECTS =====
        result['projects'] = self._extract_projects_regex(text)

        # ===== EXTRACT ACHIEVEMENTS =====
        result['achievements'] = self._extract_achievements_regex(text)

        # ===== EXTRACT REFERENCES =====
        result['references'] = self._extract_references_regex(text)

        # ===== EXTRACT HOBBIES =====
        result['hobbies'] = self._extract_hobbies_regex(text)

        self.logger.info(f"📊 Comprehensive extraction complete:")
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
        🏆 CMFAS Certification Expander - Singapore Financial Industry Certifications

        CMFAS (Capital Markets and Financial Advisory Services) modules are listed as:
        "CMFAS M5 | M8 | M8A | M9 | M9A | HI | CGI – BCP | ComGI | PGI"

        This function expands abbreviated module lists so each module retains the CMFAS prefix:
        "CMFAS M5 | CMFAS M8 | CMFAS M8A | CMFAS M9 | CMFAS M9A | CMFAS HI | ..."

        Also handles standalone CMFAS modules on separate lines:
        "CMFAS M5, M8, M8A
         HI
         CGI – BCP, ComGI, PGI"
        """
        if not text:
            return text

        # Known CMFAS module codes (Singapore MAS certifications)
        # M1-M10: Various advisory modules
        # HI: Health Insurance, CGI: Collective Investment, BCP: Bundled Coverage Product
        # ComGI: Commercial General Insurance, PGI: Personal General Insurance
        # Also includes compound forms like "CGI – BCP"
        cmfas_module_single = r'M\d+[A-Z]?|HI|CGI|BCP|ComGI|PGI|PHI|CLI|BCPP'
        # Compound module (e.g., "CGI – BCP")
        cmfas_module = r'(?:' + cmfas_module_single + r')(?:\s*[–\-]\s*(?:' + cmfas_module_single + r'))?'

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
                    # Handle compound modules like "CGI – BCP"
                    expanded.append(f"CMFAS {part}")
            return ' | '.join(expanded)

        result = re.sub(cmfas_pattern, expand_cmfas_match, text, flags=re.IGNORECASE)

        # PHASE 2: Handle standalone CMFAS modules on separate lines
        # These appear in "Additional Qualifications" or similar sections without CMFAS prefix
        # Pattern: Standalone line containing ONLY CMFAS module codes (no other text)
        # e.g., "HI" or "CGI – BCP, ComGI, PGI"
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

            # Clean up: Fix compound modules like "CMFAS CGI – CMFAS BCP" -> "CMFAS CGI – BCP"
            # The second module in a compound shouldn't have CMFAS prefix
            modules_alt = '|'.join(re.escape(m) for m in standalone_modules)
            compound_fix = r'(CMFAS\s+(?:' + modules_alt + r'))(\s*[–\-]\s*)CMFAS\s+(' + modules_alt + r')'
            result = re.sub(compound_fix, r'\1\2\3', result)

        if result != text:
            self.logger.info("🏆 Expanded CMFAS certifications for proper extraction")

        return result

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
        
        This queen recognizes skills across ALL industries, darling! 👑
        """
        hard_skills = []
        soft_skills = []

        # 🎭 IMPROVED: Section terminators must be at LINE START to avoid false matches
        section_end = r'(?=\n\s*(?:WORK\s+)?EXPERIENCE[S]?|EMPLOYMENT|WORK\s+HISTORY|EDUCATION|CERTIFICATION|AWARD|PROJECT|REFERENCE|ACHIEVEMENT|PUBLICATION|LANGUAGE[S]?|HOBBIES?|INTEREST|SUMMARY|OBJECTIVE|PROFILE|COMMENDATION|PROFESSIONAL\s+DEVELOPMENT|CAREER|CO-?CURRICULAR|ADDITIONAL\s+QUALIFICATIONS?)\s*(?:[:|\n]|$)|(?=\n\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s+to)|$)'

        # 🆕 ENHANCED v6.0: More patterns to catch skills in various resume formats!
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
            # 🆕 Pattern 13: "TECH STACK" (common in developer resumes)
            (r'(?:^|\n)\s*(?:#*\s*\**)?TECH(?:NICAL)?\s+STACK\s*[:\n]\s*(.*?)' + section_end, 'tech_stack'),
            # 🆕 Pattern 14: "PROFICIENCIES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:TECHNICAL\s+)?PROFICIENC(?:IES|Y)\s*[:\n]\s*(.*?)' + section_end, 'proficiencies'),
            # 🆕 Pattern 15: "CAPABILITIES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:CORE\s+)?CAPABILITIES\s*[:\n]\s*(.*?)' + section_end, 'capabilities'),
            # 🆕 Pattern 16: "STRENGTHS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:KEY\s+)?STRENGTHS\s*[:\n]\s*(.*?)' + section_end, 'strengths'),
            # 🆕 Pattern 17: "SPECIALIZATIONS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:AREAS?\s+OF\s+)?SPECIALIZ(?:ATION|ATIONS?)\s*[:\n]\s*(.*?)' + section_end, 'specializations'),
            # 🆕 Pattern 18: "DOMAINS" or "DOMAIN KNOWLEDGE"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:DOMAIN\s+)?(?:KNOWLEDGE|EXPERTISE|DOMAINS?)\s*[:\n]\s*(.*?)' + section_end, 'domains'),
            # 🆕 Pattern 19: Singapore-specific - "LICENSES" or "LICENCES"
            (r'(?:^|\n)\s*(?:#*\s*\**)?LICEN[CS]ES?\s*[:\n]\s*(.*?)' + section_end, 'licenses'),
            # 🆕 Pattern 20: "DATABASES" section
            (r'(?:^|\n)\s*(?:#*\s*\**)?DATABASES?\s*[:\n]\s*(.*?)' + section_end, 'databases'),
            # 🆕 Pattern 21: "CLOUD" or "CLOUD PLATFORMS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?CLOUD(?:\s+PLATFORMS?)?\s*[:\n]\s*(.*?)' + section_end, 'cloud'),
            # 🆕 Pattern 22: "OPERATING SYSTEMS" or "OS"
            (r'(?:^|\n)\s*(?:#*\s*\**)?(?:OPERATING\s+SYSTEMS?|OS)\s*[:\n]\s*(.*?)' + section_end, 'os'),
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
                self.logger.warning(f"⚠️ Regex error in skills pattern '{pattern_name}': {e}")
                continue

        # Combine all captured skills sections
        skills_text = "\n".join(all_skills_text)

        if not skills_text:
            self.logger.warning("⚠️ No skills section found")
            return {"hard_skills": [], "soft_skills": []}

        self.logger.info(f"📊 Found skills via patterns: {patterns_matched}")
        
        # 🧹 CLEAN THE TEXT!
        # Remove markdown formatting (bold, italic, headers)
        skills_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', skills_text)  # **bold**
        skills_text = re.sub(r'\*([^*\n]+)\*', r'\1', skills_text)    # *italic*
        skills_text = re.sub(r'^#+\s*', '', skills_text, flags=re.MULTILINE)  # # headers
        
        # Remove bullet characters
        skills_text = re.sub(r'[•●○◦▪▫■□►▸‣⁃→·∙⋅▶▷◆◇★☆✓✔]', '\n', skills_text)
        skills_text = re.sub(r'^\s*[-–—*]\s*', '\n', skills_text, flags=re.MULTILINE)

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
                self.logger.debug(f"⏭️ Skipping date-containing entry: {cleaned[:50]}")
                continue
            
            # Skip if it contains company indicators
            company_indicators = ['pte ltd', 'pvt ltd', 'inc.', 'corp.', 'llc', 'company', 'consultancy']
            if any(ind in cleaned.lower() for ind in company_indicators):
                self.logger.debug(f"⏭️ Skipping company-like entry: {cleaned[:50]}")
                continue
            
            # Skip if it looks like a job title + company combination
            if re.search(r'(?:Consultant|Manager|Engineer|Developer|Assistant|Executive|Officer),\s+[A-Z]', cleaned):
                self.logger.debug(f"⏭️ Skipping job+company entry: {cleaned[:50]}")
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

        self.logger.info(f"📊 Found {len(hard_skills)} hard skills, {len(soft_skills)} soft skills")

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
        🔧 FAIRY CODEMOTHER'S TECHNICAL SKILL DETECTOR v6.0! 💻
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
        💅 FAIRY CODEMOTHER'S SOFT SKILL DETECTOR v6.0! 🌟
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

    def _extract_dob_english(self, text: str) -> Optional[str]:
        """
        🎂 Extract date of birth from ENGLISH resumes!
        Returns YYYY-MM-DD format.
        🎭 FAIRY CODEMOTHER'S ENHANCEMENT: Added DD Mon YYYY pattern!
        """
        import datetime
        
        current_year = datetime.datetime.now().year
        min_birth_year = current_year - config.MAX_AGE
        max_birth_year = current_year - config.MIN_AGE
        
        # Find contact info area
        contact_area = self._find_contact_area(text)
        search_text = contact_area if contact_area else text
        
        # 🎭 FAIRY CODEMOTHER'S ENHANCED PATTERNS - Now with DD Mon YYYY support!
        dob_patterns = [
            # 🆕 NEW PATTERN: DD Mon YYYY (e.g., "13 Oct 1989") - Singapore/UK style!
            (r'(?:DOB|D\.O\.B\.|Date of Birth|Birth Date|Born)[\s:]*(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})', 'dmy_written_labeled'),
            
            # 🆕 NEW PATTERN: DD Mon YYYY without label (standalone)
            (r'\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b', 'dmy_written'),
            
            # With label: DOB: 01/15/1990
            (r'(?:DOB|D\.O\.B\.|Date of Birth|Birth Date|Born)[\s:]*(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'mdy_labeled'),
            # With label: DOB: 1990-01-15
            (r'(?:DOB|D\.O\.B\.|Date of Birth|Birth Date|Born)[\s:]*(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 'ymd_labeled'),
            # Written: January 15, 1990
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', 'written'),
            # ISO format: 1990-01-15
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'ymd'),
            # US format: 01/15/1990
            (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', 'mdy'),
            # Pattern for "22 January 1971" - DD FULL_MONTH YYYY format
            (r'(?:DOB|D\.O\.B\.|Date of Birth|Birth Date|Born)[\s:]*(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', 'dmy_full_month_labeled'),
            # Same pattern but standalone (no label)
            (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', 'dmy_full_month'),
        ]
        
        # 🎭 Month name to number mapping for the new patterns!
        month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
            'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }
        
        for pattern, date_format in dob_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE)
            for match in matches:
                try:
                    # 🆕 Handle the new DD Mon YYYY formats!
                    if date_format in ['dmy_written_labeled', 'dmy_written']:
                        day = int(match.group(1))
                        month_name = match.group(2).lower()
                        year = int(match.group(3))
                        month = month_map.get(month_name[:3], 0)  # Use first 3 chars for matching
                        
                    elif date_format == 'written':
                        month_name, day, year = match.groups()
                        month = datetime.datetime.strptime(month_name, '%B').month
                        year, day = int(year), int(day)
                    
                    elif date_format in ['mdy_labeled', 'mdy']:
                        month, day, year = map(int, match.groups())
                    
                    elif date_format in ['ymd_labeled', 'ymd']:
                        year, month, day = map(int, match.groups())
                    
                    elif date_format in ['dmy_full_month_labeled', 'dmy_full_month']:
                        day = int(match.group(1))
                        month_name = match.group(2)
                        year = int(match.group(3))
                        month = datetime.datetime.strptime(month_name, '%B').month

                    else:
                        continue
                    
                    # Validate year range
                    if min_birth_year <= year <= max_birth_year:
                        # Validate date is real
                        dt = datetime.datetime(year, month, day)
                        formatted = dt.strftime('%Y-%m-%d')
                        self.logger.info(f"✅ Found DOB: {formatted}")
                        return formatted
                
                except (ValueError, OverflowError):
                    continue
        
        return None

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
        date_first_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s+to\s+(Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s+([^\n]+)'
        
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
                self.logger.debug(f"⏭️ Skipping education entry in experience: {role_company[:50]}")
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
            
            # Get description (text until next date pattern or section header)
            chunk_start = match.end()
            if i + 1 < len(matches):
                chunk_end = matches[i + 1].start()
            else:
                # Find next section header
                next_section = re.search(r'\n\s*(?:Education|Skills|Achievements|Co-Curricular|Additional)', 
                                        text[chunk_start:], re.IGNORECASE)
                chunk_end = chunk_start + (next_section.start() if next_section else 1500)
            
            desc_chunk = text[chunk_start:chunk_end]
            
            # Extract bullet points - capture ALL of them
            bullet_lines = []
            in_bullets = False
            consecutive_empty = 0

            for line in desc_chunk.split('\n'):
                line = line.strip()

                # Track empty lines
                if len(line) < 5:
                    consecutive_empty += 1
                    if consecutive_empty >= 3 and bullet_lines:
                        break
                    continue

                consecutive_empty = 0

                # Check for bullet markers
                if line.startswith(('*', '-', '•', '·', '►', '➢', '○', '●')):
                    # Clean the bullet
                    cleaned = re.sub(r'^[\*\-•·►➢○●]\s*', '', line).strip()
                    if len(cleaned) > 10:
                        bullet_lines.append(cleaned)
                        in_bullets = True
                elif in_bullets and len(line) > 10:
                    # Continue capturing non-bullet lines if we're in the description
                    bullet_lines.append(line)

            description = ' | '.join(bullet_lines) if bullet_lines else "Description not available"

            jobs.append({
                "company": company[:200],  # Increased limit
                "role": role[:200],        # Increased limit
                "dates": dates,
                "description": description  # No truncation - capture all!
            })
            
            self.logger.info(f"✅ Extracted: {role[:40]} at {company[:40]}")
        
        return jobs
    
    def _extract_experience_regex(self, text: str) -> list:
        """
        💼 Extract work experience using IMPROVED REGEX
        Now handles markdown, flexible formats, and edge cases!
        🎭 FAIRY CODEMOTHER'S UPDATE: Now tries date-first format first! ✨
        """

        jobs = []
        
        # 🧹 STEP 1: CLEAN THE TEXT FIRST!
        text = self._clean_markdown(text)
        self.logger.info("🧹 Cleaned markdown formatting")
        
        # 🆕 STEP 1.5: TRY DATE-FIRST FORMAT (Singapore/UK style)
        # This catches resumes like "Feb 2016 to Present    Financial Consultant, Prudential..."
        date_first_jobs = self._extract_experience_date_first_format(text)
        if date_first_jobs and len(date_first_jobs) >= 2:
            self.logger.info(f"🎯 Date-first format detected! Found {len(date_first_jobs)} jobs")
            return date_first_jobs
        
        certification_keywords = [
            'certificate', 'certification', 'certified', 'diploma', 'course',
            'award', 'commendation', 'license', 'licence', 'accredit',
            'training', 'module', 'level', 'bizsafe', 'nebosh', 'iso',
            'oshas', 'mpa atp', 'mom atp', 'mom lsp', 'attestation'
        ]


        # 🎭 FAIRY CODEMOTHER'S ENHANCED SECTION DETECTION v3.0!
        # First, try to find the EXACT experience section boundaries
        
        # Look for Work Experience section header
        exp_section_match = re.search(
            r'(?:^|\n)\s*(?:WORK\s+)?EXPERIENCE[S]?\s*\n',
            text, 
            re.IGNORECASE
        )
        
        # Look for Education section (this is where experience ENDS!)
        edu_section_match = re.search(
            r'(?:^|\n)\s*EDUCATION(?:AL)?\s*(?:BACKGROUND|HISTORY|QUALIFICATIONS?)?\s*\n',
            text,
            re.IGNORECASE
        )
        
        if exp_section_match:
            start = exp_section_match.end()
            
            # End at Education section if found, otherwise use a reasonable chunk
            if edu_section_match and edu_section_match.start() > start:
                end = edu_section_match.start()
                self.logger.info(f"🎯 Found experience section: chars {start}-{end} (stops at Education)")
            else:
                # Fallback: Look for other section terminators
                other_sections = re.search(
                    r'(?:^|\n)\s*(?:ACHIEVEMENTS?|CO-?CURRICULAR|SKILLS|ADDITIONAL)\s*\n',
                    text[start:],
                    re.IGNORECASE
                )
                if other_sections:
                    end = start + other_sections.start()
                else:
                    end = min(start + 5000, len(text))
                self.logger.info(f"🎯 Found experience section: chars {start}-{end}")
            
            exp_text = text[start:end]
        else:
            # Fallback: Use section boundary detection
            sections = self._detect_section_boundaries(text)
            if 'experience' in sections:
                start, end = sections['experience']
                exp_text = text[start:end]
                self.logger.info(f"🎯 Found experience section via boundaries: chars {start}-{end}")
            else:
                exp_text = text
                self.logger.warning("⚠️ No clear experience section found - using full text")


        # 🔍 STEP 2: FIND EXPERIENCE SECTION
        # 🎭 FAIRY CODEMOTHER'S ULTIMATE FIX v5.0! 💅
        # 
        # THE PROBLEM: The old pattern used keywords like "SKILLS" in the lookahead,
        # but "skills" appears INSIDE bullet points (e.g., "troubleshooting skills")
        # causing premature termination! Drama queen behavior! 😱
        #
        # THE FIX: Only match section HEADERS (word alone on a line or at line start)
        # not words embedded in sentences!
        
        exp_patterns = [
            # Pattern 1: EXPERIENCE followed by content until a SECTION HEADER
            # Section headers are words at the START of a line, possibly followed by colon
            # The key is \n before the section name to ensure it's a header!
            r'(?:WORK\s+)?EXPERIENCE[S]?\s*:?\s*\n(.+?)(?=\n(?:EDUCATION|SKILL[S]?|CERTIFICATION|AWARD|PROJECT|REFERENCE|ACHIEVEMENT|PUBLICATION|TRAINING|LANGUAGE|HOBBY|HOBBIES|INTEREST|SUMMARY|OBJECTIVE|PROFILE)\s*(?:[:|\n]|$))',
            
            # Pattern 2: Same but with ## markdown headers
            r'(?:WORK\s+)?EXPERIENCE[S]?\s*\n(.+?)(?=\n#+\s*(?:EDUCATION|SKILL|CERTIFICATION|AWARD))',
            
            # Pattern 3: GREEDY FALLBACK - capture everything after EXPERIENCE until end
            # This is the SAFETY NET when no clear section boundary exists!
            r'(?:WORK\s+)?EXPERIENCE[S]?\s*:?\s*\n(.+)$',
        ]
        
        exp_text = ""
        pattern_used = None
        
        for i, pattern in enumerate(exp_patterns):
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    exp_text = match.group(1)
                    pattern_used = i + 1
                    self.logger.info(f"💼 Found experience section ({len(exp_text)} chars) using pattern {pattern_used}")
                    break
            except re.error as e:
                self.logger.warning(f"⚠️ Regex error in pattern {i+1}: {e}")
                continue
        
        # 🚨 FALLBACK: If no section header found, use full text
        if not exp_text or len(exp_text) < 100:
            self.logger.warning("⚠️ No clear experience section - scanning full document")
            exp_text = text
        
        # 🎭 CRITICAL SAFETY CHECK: If exp_text is suspiciously short, USE FULL TEXT!
        # This catches the case where "skills" appears early in the text
        original_len = len(text)
        captured_len = len(exp_text)
        
        if captured_len < 500 and original_len > 1000:
            self.logger.warning(f"⚠️ Experience section too short ({captured_len} chars vs {original_len} total)")
            self.logger.warning("⚠️ Likely hit an embedded keyword like 'skills' - using full document!")
            exp_text = text
        elif captured_len < original_len * 0.25:  # Less than 25% of original
            self.logger.warning(f"⚠️ Experience section is only {captured_len}/{original_len} chars ({100*captured_len//original_len}%)")
            self.logger.warning("⚠️ Using full document for safety")
            exp_text = text
        
        # 🏢 STEP 3: FIND ALL COMPANIES
        # 🎭 FAIRY CODEMOTHER'S PRECISION PATTERNS v6.0 - ENHANCED! 💅
        # Handles MULTIPLE resume formats: Indian, Singapore, US, UK, EU, Startups, Freelance!
        company_patterns = [

            (r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s+to\s+(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s+([A-Z][A-Za-z\s]+(?:Consultant|Manager|Engineer|Developer|Assistant|Executive|Officer|Specialist|Coordinator|Director|Analyst)[,\s]+[A-Z][^\n]+)', 'sg_date_first'),

            # ===== LEGAL SUFFIX PATTERNS (High confidence) =====

            # Pattern 1: Indian style - "Pvt. Ltd." or "Private Limited"
            (r'^([A-Z][^\n]{3,60}?(?:Pvt\.?\s*Ltd\.?|Private\s+Limited))(?:\s*[-–][^\n]*)?$', 'pvt_ltd'),

            # Pattern 2: Singapore style - "Pte Ltd" or "Pte. Ltd."
            (r'^([A-Z][^\n]{3,60}?(?:Pte\.?\s*Ltd\.?))(?:\s*[-–][^\n]*)?$', 'pte_ltd'),

            # Pattern 3: US/International - "Ltd", "Inc", "Corp", "LLC", "LLP", "GmbH", "S.A.", etc.
            (r'^([A-Z][^\n]{3,50}?(?:Ltd\.?|Inc\.?|Corp\.?|Corporation|LLC|LLP|LP|GmbH|S\.?A\.?|PLC|N\.?V\.?|B\.?V\.?|AG|SE|SARL|SRL|SpA|KG|OHG|UG))(?:\s*[-–][^\n]*)?$', 'legal_suffix'),

            # Pattern 4: Tech/Business keywords - "Technologies", "Solutions", "Software", etc.
            (r'^([A-Z][^\n]{3,50}?(?:Technologies|Solutions|Services|Systems|Consulting|Enterprises|Software|Digital|Labs?|Studio|Agency|Media|Group|Partners|Holdings|Ventures|Capital|Networks|Logistics|Industries|Manufacturing|International|Global|Worldwide|Asia|Pacific))(?:\s+(?:Pvt\.?\s*Ltd\.?|Pte\.?\s*Ltd\.?|Inc\.?|LLC))?(?:\s*[-–][^\n]*)?$', 'tech_company'),

            # ===== CONTEXTUAL PATTERNS (Medium confidence) =====

            # Pattern 5: Company name on line BEFORE a job title
            (r'^([A-Z][A-Za-z0-9\s&\.,\'-]{5,50})\s*$(?=\s*\n\s*(?:Senior|Junior|Lead|Staff|Principal|Chief|Head|VP|Director|Manager|Engineer|Developer|Designer|Analyst|Consultant|Specialist|Architect|Administrator|Coordinator|Executive|Officer|Intern|Trainee|Associate|Representative))', 'before_job_title'),

            # Pattern 6: Company with date range on SAME LINE
            (r"^([A-Z][A-Za-z0-9\s&\.,\'-]{5,50}?)(?:\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*['\"]?\d{2,4}|\s+\d{4})\s*[-–]\s*(?:Present|Current|Now|Ongoing|\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec))", 'company_with_date'),

            # Pattern 7: Company followed by location
            (r'^([A-Z][A-Za-z0-9\s&\'-]{5,40})\s*[-–,]\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,?\s*(?:[A-Z]{2}|Singapore|India|USA|UK|Germany|France|Australia|Canada|Japan|China|Malaysia|Indonesia|Philippines|Vietnam|Thailand|Hong Kong|Taiwan))$', 'company_with_location'),

            # ===== SPECIAL PATTERNS =====

            # Pattern 8: Trading Co / Co. pattern
            (r'^([A-Z][^\n]+?(?:Trading\s+)?Co\.?[^\n]*)$', 'trading_company'),

            # Pattern 9: Company with parenthetical info (e.g., "ABC Corp (Singapore)")
            (r'^([A-Z][^\n]+?\([^)\n]+\))$', 'parenthetical_company'),

            # Pattern 10: Freelance/Self-employed/Contract
            (r'^((?:Freelance|Self[- ]?Employed|Independent|Contract(?:or|ing)?|Consultant|Consultancy|Remote)[^\n]*)$', 'freelance'),

            # Pattern 11: Known tech companies (no suffix needed)
            (r'^((?:Google|Meta|Facebook|Amazon|Apple|Microsoft|Netflix|Uber|Airbnb|Stripe|Shopify|Twitter|LinkedIn|Oracle|IBM|Intel|Cisco|Adobe|Salesforce|SAP|VMware|Dell|HP|Accenture|Infosys|TCS|Wipro|HCL|Cognizant|Capgemini|Deloitte|EY|PwC|KPMG|McKinsey|BCG|Bain|Goldman|JPMorgan|Morgan Stanley|Citibank|HSBC|Standard Chartered|DBS|OCBC|UOB|Grab|Sea|Shopee|Lazada|ByteDance|TikTok|Alibaba|Tencent|Baidu|Huawei|Xiaomi|Samsung|Sony|Toyota|Honda|BMW|Mercedes|Tesla)[^\n]*)$', 'known_company'),

            # Pattern 12: Bank/Financial institution
            (r'^([A-Z][^\n]{3,50}?(?:Bank|Finance|Financial|Insurance|Investment|Asset\s+Management|Securities|Capital|Credit|Trust))(?:\s*[-–][^\n]*)?$', 'financial'),

            # Pattern 13: Hospital/Healthcare
            (r'^([A-Z][^\n]{3,50}?(?:Hospital|Medical|Healthcare|Health\s+Care|Clinic|Pharma|Pharmaceutical|Biotech|Life\s+Sciences))(?:\s*[-–][^\n]*)?$', 'healthcare'),

            # Pattern 14: University/Institute/School
            (r'^([A-Z][^\n]{3,50}?(?:University|Institute|College|School|Academy|Polytechnic))(?:\s*[-–][^\n]*)?$', 'education_org'),

            # Pattern 15: Government/Public sector
            (r'^([A-Z][^\n]{3,50}?(?:Ministry|Department|Government|Authority|Agency|Council|Commission|Board|Bureau))(?:\s*[-–][^\n]*)?$', 'government'),

            # ===== FALLBACK PATTERNS (Lower confidence but broad) =====

            # Pattern 16: Any capitalized multi-word line that's NOT a job description
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})$', 'capitalized_name'),

            # Pattern 17: Company name followed by ":" or "|" (common formatting)
            (r'^([A-Z][A-Za-z0-9\s&\.,\'-]{5,50})\s*[:|]', 'company_with_separator'),
        ]
        
        all_matches = []
        for pattern, pattern_type in company_patterns:
            try:
                matches = list(re.finditer(pattern, exp_text, re.MULTILINE | re.IGNORECASE))
                if matches:
                    self.logger.info(f"🏢 Found {len(matches)} companies using '{pattern_type}' pattern")
                    # 🎭 FAIRY CODEMOTHER'S DEBUG TIP: Log what we found!
                    for m in matches[:3]:  # Show first 3 matches for debugging
                        self.logger.debug(f"   → Match: '{m.group(1)[:50]}...' at pos {m.start()}")
                    all_matches.extend([(m, pattern_type) for m in matches])
            except re.error as regex_err:
                # 🚨 Handle malformed regex gracefully - don't let one bad pattern crash everything!
                self.logger.error(f"❌ Regex error in '{pattern_type}' pattern: {regex_err}")
                continue
        
        if not all_matches:
            # 🎭 ENHANCED ERROR HANDLING: Log sample of text for debugging
            self.logger.warning("⚠️ No companies found with standard patterns, trying AGGRESSIVE fallback...")
            
            # 🚨 AGGRESSIVE FALLBACK: Look for ANY line that might be a company
            # This catches edge cases that slip through the patterns
            fallback_matches = []
            lines = exp_text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 5 or len(line) > 80:
                    continue
                
                # Skip lines that start with action verbs (job descriptions)
                # 🎭 FAIRY CODEMOTHER'S EXPANDED LIST - matches skip_start_words!
                action_verbs = [
                    'developed', 'created', 'managed', 'led', 'built', 'designed',
                    'implemented', 'executed', 'worked', 'responsible', 'achieved',
                    'increased', 'decreased', 'improved', 'reduced', 'collaborated',
                    'coordinated', 'analyzed', 'maintained', 'provided', 'ensured',
                    'utilized', 'demonstrated', 'conducted', 'supported', 'delivered',
                    'performed', 'generated', 'resolved', 'diagnosed', 'authored',
                    'assisted', 'prepared', 'established', 'organized', 'trained',
                    # 🎭 NEW: Additional verbs (present participle forms too!)
                    'managing', 'providing', 'working', 'assisting', 'establishing',
                    'organising', 'organizing', 'training', 'mentoring', 'supervising',
                    'overseeing', 'spearheading', 'initiating', 'launching', 'negotiating',
                    'presenting', 'reviewing', 'authoring', 'diagnosing', 'streamlining',
                    'optimizing', 'optimising', 'facilitating', 'hosting', 'educating',
                    'collaborating', 'improving', 'increasing', 'decreasing', 'reducing',
                    'enhancing', 'driving', 'directing', 'handling', 'processing',
                    'administering', 'monitoring', 'evaluating', 'assessing', 'identifying',
                    'formulating', 'defining', 'planning', 'contributing', 'participating',
                    'engaging', 'liaising', 'interfacing', 'communicating', 'performs'
                ]

                first_word = line.split()[0].lower() if line.split() else ''
                if first_word in action_verbs:
                    continue

                # 🎭 NEW: Skip if line contains phrases indicating job description
                description_phrases = [
                    'responsible for', 'in charge of', 'duties include', 'tasked with',
                    'worked with', 'worked on', 'assist in', 'assist with', 'helped to',
                    'in order to', 'to ensure', 'to provide', 'to support', 'to maintain'
                ]
                if any(phrase in line.lower() for phrase in description_phrases):
                    continue

                # 🎭 NEW: Skip if too many words (likely a description)
                if len(line.split()) > 8:
                    continue

                # Skip common non-company lines
                skip_patterns = [
                    r'^(education|skills?|experience|summary|objective|profile|contact|references?|projects?|certifications?|awards?|languages?|hobbies|interests)$',
                    r'^\d+',  # Starts with number
                    r'^[-•●○▪]',  # Bullet points
                    r'@',  # Email addresses
                    r'^\+?\d[\d\s\-()]+$',  # Phone numbers
                ]
                if any(re.match(pat, line, re.IGNORECASE) for pat in skip_patterns):
                    continue
                
                # Check if this line might be a company (followed by job title or date)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip().lower()
                    job_indicators = ['engineer', 'developer', 'manager', 'analyst', 'designer',
                                     'consultant', 'specialist', 'architect', 'director', 'lead',
                                     'senior', 'junior', 'intern', 'trainee', 'executive', 'officer']
                    date_indicators = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 
                                      'sep', 'oct', 'nov', 'dec', 'present', '2019', '2020', 
                                      '2021', '2022', '2023', '2024', '2025']
                    
                    if any(ind in next_line for ind in job_indicators + date_indicators):
                        # Create a fake match object
                        class FakeMatch:
                            def __init__(self, text, pos):
                                self._text = text
                                self._start = pos
                            def group(self, n=0):
                                return self._text if n <= 1 else self._text
                            def start(self):
                                return self._start
                        
                        fallback_matches.append((FakeMatch(line, exp_text.find(line)), 'fallback'))
                        self.logger.info(f"🎯 Fallback found: '{line[:50]}'")
            
            if fallback_matches:
                all_matches = fallback_matches
            else:
                self.logger.error("❌ No companies found in text!")
                self.logger.debug(f"📄 Experience text sample (first 500 chars):\n{exp_text[:500]}")
                return jobs
        
        # Remove duplicates based on position AND content (improved deduplication)
        # Sometimes different patterns match the same company - we keep the first match
        seen_positions = set()
        seen_companies = set()
        unique_matches = []
        
        for match, ptype in all_matches:
            pos = match.start()
            company_text = match.group(1).strip().lower()[:50]  # Normalize for comparison
            
            # Skip if we've seen this position OR very similar company name
            position_key = pos // 10  # Allow 10-char tolerance for position
            if position_key not in seen_positions and company_text not in seen_companies:
                seen_positions.add(position_key)
                seen_companies.add(company_text)
                unique_matches.append((match, ptype))
        
        # Sort by position in text
        unique_matches.sort(key=lambda x: x[0].start())
        
        self.logger.info(f"💼 Processing {len(unique_matches)} unique job entries")
        
        # 👔 STEP 4: EXTRACT DETAILS FOR EACH JOB
        for i, (company_match, pattern_type) in enumerate(unique_matches):
            company_name = company_match.group(1).strip()
            
            # Clean company name
            company_name = re.sub(r'\s+', ' ', company_name)  # Remove extra spaces
            company_name = company_name.strip('.,;:-')  # Remove trailing punctuation
            
            # Define job chunk (text for this specific job)
            chunk_start = company_match.start()
            if i + 1 < len(unique_matches):
                chunk_end = unique_matches[i + 1][0].start()
            else:
                chunk_end = min(chunk_start + 3000, len(exp_text))
            
            job_chunk = exp_text[chunk_start:chunk_end]
            
            # 👔 EXTRACT ROLE
            # 🎭 FAIRY CODEMOTHER'S ENHANCED ROLE PATTERNS v6.0!
            # These patterns are like a talent scout - they spot the job titles hiding in the crowd! 💅
            role_patterns = [
                # Pattern 1: Comprehensive job titles with seniority prefixes
                r'((?:Senior|Junior|Lead|Principal|Chief|Head\s+of|VP\s+of|Vice\s+President|Associate|Assistant|Staff|Trainee|Entry[- ]?Level|Mid[- ]?Level|Executive|Managing|General)?\s*(?:Field\s+Service\s+|Full[- ]?Stack\s+|Front[- ]?End\s+|Back[- ]?End\s+|Data\s+|Business\s+|Product\s+|Project\s+|Program\s+|Operations\s+|Marketing\s+|Sales\s+|HR\s+|Human\s+Resources\s+|Finance\s+|Accounting\s+|IT\s+|Technical\s+|Software\s+|Hardware\s+|Network\s+|Security\s+|Cloud\s+|DevOps\s+|QA\s+|Quality\s+)?(?:Engineer|Manager|Developer|Analyst|Specialist|Executive|Consultant|Director|Officer|Coordinator|Lead|Architect|Technician|Administrator|Designer|Supervisor|Intern|Trainee|Associate|Representative|Agent|Advisor|Strategist|Planner|Controller|Accountant|Recruiter|Scientist|Researcher|Writer|Editor|Producer|Creator))',
                # Pattern 2: Service/Sales/Support specific roles
                r'((?:Senior|Junior|Lead)?\s*(?:Service|Sales|Quality\s+Control|Technical\s+Support|Customer\s+Service|Customer\s+Success|Account|Client\s+Relations|Business\s+Development)\s*(?:Engineer|Executive|Manager|Specialist|Representative|Agent|Lead|Director))',
                # Pattern 3: C-Suite and VP roles
                r'((?:Chief\s+)?(?:Executive|Technology|Operating|Financial|Marketing|Information|Product|Revenue|Data|Human\s+Resources)\s+Officer|C[ETOFMIP]O|VP\s+(?:of\s+)?[A-Za-z\s]+|Vice\s+President\s+(?:of\s+)?[A-Za-z\s]+)',
                # Pattern 4: Role followed by "at" or "for" company
                r'([A-Z][A-Za-z\s]+(?:Engineer|Manager|Developer|Analyst|Designer|Architect|Lead|Director|Specialist|Consultant))\s+(?:at|for|with)\s+',
                # Pattern 5: Common tech roles
                r'((?:Full[- ]?Stack|Front[- ]?End|Back[- ]?End|Mobile|iOS|Android|Web|UI/?UX|DevOps|SRE|ML|AI|Data|Cloud|Platform|Infrastructure|Security|QA|Test)\s+(?:Engineer|Developer|Architect|Designer|Specialist|Lead))',
                # Pattern 6: Industry-specific roles
                r'((?:Financial|Investment|Insurance|Banking|Healthcare|Pharmaceutical|Legal|Real\s+Estate|Logistics|Supply\s+Chain|Manufacturing|Retail|Hospitality)\s+(?:Analyst|Consultant|Manager|Specialist|Advisor|Agent|Officer|Director))',
                # Pattern 7: Any capitalized multi-word phrase that looks like a title
                r'\n\s*([A-Z][A-Za-z]+(?:\s+[A-Z]?[a-z]+){1,4})\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})',
                # Pattern 8: Role on its own line (common format)
                r'^\s*([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,4}(?:Engineer|Manager|Developer|Analyst|Designer|Director|Lead|Specialist|Consultant|Coordinator|Officer|Administrator|Supervisor|Executive))\s*$',
            ]
            
            role = "Position not specified"
            for pattern in role_patterns:
                role_match = re.search(pattern, job_chunk, re.MULTILINE | re.IGNORECASE)
                if role_match:
                    potential_role = role_match.group(1).strip()
                    potential_role = re.sub(r'\s+', ' ', potential_role)
                    
                    # Validate it's not the company name again and meets length requirements
                    if (potential_role.lower() != company_name.lower()[:len(potential_role)] and 
                        len(potential_role) >= 5 and 
                        len(potential_role) <= 60):
                        # 🎭 Additional validation: shouldn't contain location indicators
                        location_words = ['india', 'mumbai', 'delhi', 'bangalore', 'usa', 'uk', 'pvt', 'ltd']
                        if not any(loc in potential_role.lower() for loc in location_words):
                            role = potential_role
                            break
            
            # 📅 EXTRACT DATES - Enhanced v6.0
            date_patterns = [
                # Pattern 1: Full/abbreviated month + year range (most common)
                r"((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s*['\"]?\d{2,4}\s*(?:[-\u2013\u2014]|to|till|until|through)\s*(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s*['\"]?\d{2,4}|Present|Current|Now|Ongoing|Till\s+Date|To\s+Date|present|current|ongoing))",
                # Pattern 2: MM/YYYY format
                r"(\d{1,2}/\d{2,4}\s*(?:[-\u2013\u2014]|to|till)\s*(?:\d{1,2}/\d{2,4}|Present|Current|Now|Ongoing))",
                # Pattern 3: MM-YYYY format
                r"(\d{1,2}-\d{4}\s*(?:[-\u2013\u2014]|to|till)\s*(?:\d{1,2}-\d{4}|Present|Current|Now|Ongoing))",
                # Pattern 4: YYYY-MM format (ISO style)
                r"(\d{4}-\d{1,2}\s*(?:[-\u2013\u2014]|to|till)\s*(?:\d{4}-\d{1,2}|Present|Current|Now|Ongoing))",
                # Pattern 5: Year only range
                r"(\d{4}\s*(?:[-\u2013\u2014]|to|till|through)\s*(?:\d{4}|Present|Current|Now|Ongoing|present|current))",
                # Pattern 6: Quarter-based dates (Q1 2020 - Q4 2022)
                r"(Q[1-4]\s*['\"]?\d{2,4}\s*(?:[-\u2013\u2014]|to|till)\s*(?:Q[1-4]\s*['\"]?\d{2,4}|Present|Current|Now))",
                # Pattern 7: Season-based dates (Spring 2020 - Fall 2022)
                r"((?:Spring|Summer|Fall|Autumn|Winter)\s+\d{4}\s*(?:[-\u2013\u2014]|to|till)\s*(?:(?:Spring|Summer|Fall|Autumn|Winter)\s+\d{4}|Present|Current|Now))",
                # Pattern 8: Single date with "since" (Since Jan 2020)
                r"((?:Since|From)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s*\d{2,4})",
                # Pattern 9: DD/MM/YYYY - DD/MM/YYYY format
                r"(\d{1,2}/\d{1,2}/\d{2,4}\s*(?:[-\u2013\u2014]|to|till)\s*(?:\d{1,2}/\d{1,2}/\d{2,4}|Present|Current|Now))",
                # Pattern 10: Parenthetical dates (2020 - Present)
                r"\((\d{4}\s*[-\u2013\u2014]\s*(?:\d{4}|Present|Current|Now))\)",
            ]
            
            dates = "Dates not specified"
            for pattern in date_patterns:
                date_match = re.search(pattern, job_chunk, re.IGNORECASE)
                if date_match:
                    dates = date_match.group(1)
                    # Normalize date separators
                    dates = dates.replace('–', ' - ').replace('—', ' - ')
                    dates = re.sub(r'\s+to\s+', ' - ', dates, flags=re.IGNORECASE)
                    dates = re.sub(r'\s+till\s+', ' - ', dates, flags=re.IGNORECASE)
                    dates = re.sub(r'\s+until\s+', ' - ', dates, flags=re.IGNORECASE)
                    dates = re.sub(r'\s+through\s+', ' - ', dates, flags=re.IGNORECASE)
                    # Normalize end indicators
                    dates = re.sub(r'(present|current|ongoing|now|till\s+date|to\s+date)', 'Present', dates, flags=re.IGNORECASE)
                    # Normalize "Since/From" format
                    dates = re.sub(r'^(Since|From)\s+', '', dates, flags=re.IGNORECASE)
                    if 'Present' not in dates and not re.search(r'-\s*\d', dates):
                        dates = dates + ' - Present'
                    dates = re.sub(r'\s+', ' ', dates).strip()
                    break
            
            # 📝 EXTRACT DESCRIPTION - Enhanced v6.0
            # Capture ALL job responsibilities without truncation!
            # Find where dates end
            if dates != "Dates not specified":
                date_pos = job_chunk.find(dates)
                desc_start = date_pos + len(dates) if date_pos != -1 else 0
            else:
                desc_start = len(company_name) + len(role) + 50

            # COMPREHENSIVE: Use entire job chunk for description extraction
            desc_chunk = job_chunk[desc_start:]

            # 🎯 Enhanced bullet point detection patterns
            bullet_markers = [
                r'^[\*\-•●○▪■►◆➢➤✓✔→]',  # Standard bullets
                r'^\d+[\.\)]\s',            # Numbered lists (1. or 1))
                r'^[a-z][\.\)]\s',          # Lettered lists (a. or a))
                r'^(?:Key|Main|Core)\s+(?:Responsibilities|Duties|Tasks|Achievements)',  # Section headers
            ]

            # Look for bullet points - capture ALL of them
            bullet_lines = []
            in_description = False
            consecutive_empty = 0

            for line in desc_chunk.split('\n'):
                original_line = line
                line = line.strip()

                # Skip empty lines but track them
                if len(line) < 5:
                    consecutive_empty += 1
                    # Stop if we hit 3+ consecutive empty lines (likely end of section)
                    if consecutive_empty >= 3 and bullet_lines:
                        break
                    continue

                consecutive_empty = 0

                # Skip lines that look like headers or locations
                if re.match(r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$', line):  # "Mumbai, India"
                    continue

                # Skip lines that look like next company/role headers
                if re.match(r'^(?:Senior|Junior|Lead|Staff|Principal|Chief|Head|VP|Director|Manager)\s+', line, re.IGNORECASE):
                    if in_description and len(bullet_lines) > 0:
                        break

                # Skip dates that indicate a new job entry
                if re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}', line, re.IGNORECASE):
                    if in_description and len(bullet_lines) > 0:
                        break

                # Check if this is a bullet point
                is_bullet = any(re.match(pat, line) for pat in bullet_markers)

                # Clean bullet markers from the line
                cleaned_line = re.sub(r'^[\*\-•●○▪■►◆➢➤✓✔→]\s*', '', line)
                cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', cleaned_line)
                cleaned_line = re.sub(r'^[a-z][\.\)]\s*', '', cleaned_line)
                cleaned_line = cleaned_line.strip()

                # Skip if cleaned line is too short
                if len(cleaned_line) < 10:
                    continue

                # Add to description
                if is_bullet or in_description or len(bullet_lines) == 0:
                    in_description = True
                    bullet_lines.append(cleaned_line)

                # No limit on bullets - capture everything!

            if bullet_lines:
                # Join with pipe delimiter but preserve full content
                description = ' | '.join(bullet_lines)
            else:
                # Fallback: take entire desc_chunk
                description = desc_chunk.replace('\n', ' ').strip()

            # Clean description - preserve all content
            description = re.sub(r'\s+', ' ', description)
            # Remove the character limit - capture EVERYTHING
            # Only apply a very generous limit as a safety net
            description = description[:10000]  # 10K chars max as safety net

            # Skip entries that are actually bullet points/descriptions, not company names
            # 🎭 FAIRY CODEMOTHER'S EXPANDED LIST - catches job responsibilities!
            skip_start_words = [
                # Original words
                'performs', 'developed', 'created', 'managed', 'led', 'responsible',
                'achieved', 'implemented', 'conducted', 'provided', 'ensured', 'utilized',
                'demonstrated', 'maintained', 'coordinated', 'executed', 'built', 'designed',
                'analyzed', 'prepared', 'supported', 'delivered', 'generated', 'resolved',
                # 🎭 NEW: Additional action verbs commonly found in job descriptions
                'established', 'managing', 'providing', 'worked', 'working', 'assist',
                'assisted', 'assisting', 'organised', 'organized', 'organising', 'organizing',
                'trained', 'training', 'mentored', 'mentoring', 'supervised', 'supervising',
                'oversaw', 'overseeing', 'spearheaded', 'spearheading', 'initiated', 'initiating',
                'launched', 'launching', 'negotiated', 'negotiating', 'presented', 'presenting',
                'reviewed', 'reviewing', 'authored', 'authoring', 'diagnosed', 'diagnosing',
                'streamlined', 'streamlining', 'optimized', 'optimizing', 'optimised', 'optimising',
                'facilitated', 'facilitating', 'hosted', 'hosting', 'educated', 'educating',
                'collaborated', 'collaborating', 'improved', 'improving', 'increased', 'increasing',
                'decreased', 'decreasing', 'reduced', 'reducing', 'enhanced', 'enhancing',
                'drove', 'driving', 'directed', 'directing', 'handled', 'handling',
                'processed', 'processing', 'administered', 'administering', 'monitored', 'monitoring',
                'evaluated', 'evaluating', 'assessed', 'assessing', 'identified', 'identifying',
                'formulated', 'formulating', 'defined', 'defining', 'planned', 'planning',
                'contributed', 'contributing', 'participated', 'participating', 'engaged', 'engaging',
                'liaised', 'liaising', 'interfaced', 'interfacing', 'communicated', 'communicating'
            ]

            company_lower = company_name.lower()
            if any(cert_kw in company_lower for cert_kw in certification_keywords):
                self.logger.debug(f"⭐️ Skipping certification entry: {company_name[:50]}")
                continue

            if any(company_name.lower().startswith(word) for word in skip_start_words):
                self.logger.debug(f"⏭️ Skipping bullet point: {company_name[:50]}")
                continue

            # 🎭 NEW: Skip if the text contains phrases that indicate it's a job description
            description_indicators = [
                'responsible for', 'in charge of', 'duties include', 'tasked with',
                'worked with', 'worked on', 'assist in', 'assist with', 'helped to',
                'in order to', 'to ensure', 'to provide', 'to support', 'to maintain'
            ]
            if any(indicator in company_name.lower() for indicator in description_indicators):
                self.logger.debug(f"⏭️ Skipping description phrase: {company_name[:50]}")
                continue

            # Skip if too many words (likely a description, not a company)
            # 🎭 Reduced from 12 to 8 for stricter filtering
            if company_name.count(' ') > 8:
                self.logger.debug(f"⏭️ Skipping long phrase ({company_name.count(' ')+1} words): {company_name[:50]}")
                continue

            # 🎭 NEW: Skip if it contains common non-company words
            non_company_words = ['portfolio', 'assist', 'planning', 'networks', 'businesses', 'clients', 'customers']
            word_count = company_name.count(' ') + 1
            # Only apply this check for multi-word "companies" that don't have legal suffixes
            has_legal_suffix = any(suffix in company_name.lower() for suffix in ['ltd', 'inc', 'corp', 'llc', 'pte', 'pvt'])
            if word_count >= 4 and not has_legal_suffix:
                if any(word in company_name.lower() for word in non_company_words):
                    self.logger.debug(f"⏭️ Skipping non-company phrase: {company_name[:50]}")
                    continue

            # 🎯 STEP 5: ADD TO RESULTS - COMPREHENSIVE (capture all content)
            jobs.append({
                "company": company_name[:200],  # Increased limit for long company names
                "role": role[:200],             # Increased limit for long titles
                "dates": dates,
                "description": description if description else "Description not available"  # No truncation!
            })
            
            self.logger.info(f"✅ Extracted: {company_name[:40]} - {role[:40]}")
        
        self.logger.info(f"💼 Total jobs extracted: {len(jobs)}")
        
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
        edu_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s+(?:to|till|until|-|–)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current)\s+([^\n]+)'

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
                if not line or line.startswith(('Obtained', '*', '-', '•', '○', '●')):
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

    def _extract_education_regex(self, text: str) -> list:
        """
        🎓 Extract education using PURE REGEX
        Enhanced for Singapore resumes v6.0
        """

        date_first_edu = self._extract_education_date_first_format(text)
        if date_first_edu and len(date_first_edu) >= 1:
            self.logger.info(f"🎓 Date-first education format detected! Found {len(date_first_edu)} entries")
            return date_first_edu

        education = []

        # Singapore institutions for detection
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

        # 🧹 Clean the text first to remove tabs/newlines
        cleaned_text = self._clean_education_text(text)

        # Find EDUCATION section - try multiple patterns
        edu_patterns = [
            r'EDUCATION(?:AL)?\s*(?:BACKGROUND|HISTORY|QUALIFICATIONS?)?\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|WORK\s*HISTORY|SKILLS|CERTIFICATIONS?|AWARDS?|PUBLICATIONS?|REFERENCES?|PROJECTS?|ACHIEVEMENTS?|LANGUAGES?|HOBBIES?|INTERESTS?|$)',
            r'ACADEMIC\s+(?:BACKGROUND|QUALIFICATIONS?|CREDENTIALS|HISTORY)\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|SKILLS|$)',
            r'QUALIFICATIONS?\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|SKILLS|CERTIFICATIONS?|$)',
            r'TRAINING\s+(?:AND|&)\s+EDUCATION\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|SKILLS|$)',
        ]

        edu_text = ""
        for pattern in edu_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
            if match:
                edu_text = match.group(1)
                self.logger.info(f"🎓 Found education section")
                break

        # If no section header found, try to find education entries in the entire text
        if not edu_text:
            self.logger.info("🎓 No education section header found, searching entire document...")
            # Look for Singapore institutions or university/college patterns anywhere
            sg_inst_found = any(inst in cleaned_text.lower() for inst in sg_institutions)
            inst_pattern = r'([A-Z][A-Za-z\s\',\.&-]+(?:University|College|Institute|School|Academy|Polytechnic|ITE)[^\n]*)'
            if sg_inst_found or re.search(inst_pattern, cleaned_text):
                edu_text = cleaned_text[-3000:] if len(cleaned_text) > 3000 else cleaned_text
            else:
                self.logger.info("🎓 No education information found in document")
                return education

        # Enhanced pattern for institutions - includes Singapore-specific
        inst_patterns = [
            # Singapore Polytechnics
            r'((?:Singapore|Ngee\s+Ann|Temasek|Republic|Nanyang)\s+Polytechnic[^\n]*)',
            # Singapore Universities
            r'((?:National\s+University\s+of\s+Singapore|NUS|Nanyang\s+Technological\s+University|NTU|Singapore\s+Management\s+University|SMU|Singapore\s+University\s+of\s+Technology|SUTD|Singapore\s+Institute\s+of\s+Technology|SIT|Singapore\s+University\s+of\s+Social\s+Sciences|SUSS)[^\n]*)',
            # ITE
            r'((?:Institute\s+of\s+Technical\s+Education|ITE\s+College)[^\n]*)',
            # Junior Colleges
            r'((?:[A-Z][A-Za-z\s\'-]+(?:Junior\s+College|JC)|Millennia\s+Institute)[^\n]*)',
            # Secondary Schools
            r'((?:[A-Z][A-Za-z\s\'-]+(?:Secondary\s+School|Sec\s+School))[^\n]*)',
            # Private institutions
            r'((?:Kaplan|SIM\s+Global|MDIS|PSB\s+Academy|James\s+Cook\s+University|LASALLE|NAFA|Raffles)[^\n]*)',
            # General pattern for other institutions
            r'([A-Z][A-Za-z\s\',\.&-]+(?:University|College|Institute|School|Academy)[^\n]*)',
        ]

        all_inst_matches = []
        for pattern in inst_patterns:
            matches = list(re.finditer(pattern, edu_text, re.IGNORECASE))
            all_inst_matches.extend(matches)

        # Remove duplicates and sort by position
        seen_positions = set()
        unique_matches = []
        for m in sorted(all_inst_matches, key=lambda x: x.start()):
            pos_key = m.start() // 20  # Allow some tolerance
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_matches.append(m)

        for i, inst_match in enumerate(unique_matches):
            institution = inst_match.group(1).strip()
            institution = self._clean_education_text(institution)

            # Get chunk for this education entry
            chunk_start = inst_match.start()
            if i + 1 < len(unique_matches):
                chunk_end = unique_matches[i + 1].start()
            else:
                chunk_end = min(chunk_start + 1000, len(edu_text))

            edu_chunk = edu_text[chunk_start:chunk_end]

            # Extract DEGREE - Enhanced for Singapore qualifications
            degree_patterns = [
                # Singapore-specific qualifications
                r'((?:NITEC|Higher\s+NITEC|Master\s+NITEC|Technical\s+Diploma|Technician\s+Diploma)[^\n,;]*)',
                r'((?:GCE\s*["\']?\s*[NOAN]\s*["\']?\s*[-\s]?Levels?|[NOAN][-\s]?Levels?)[^\n,;]*)',
                r'(PSLE[^\n,;]*)',
                r'((?:Diploma\s+in|Advanced\s+Diploma\s+in|Specialist\s+Diploma\s+in)[^\n,;]*)',
                # Standard degrees
                r'((?:Bachelor\s+of|Master\s+of|Doctor\s+of|PhD|Ph\.?\s*D\.?|Doctorate)[^\n,;]*)',
                r'((?:B\.?\s*[A-Z][A-Za-z]*\.?|M\.?\s*[A-Z][A-Za-z]*\.?)\s+(?:in\s+)?[A-Za-z\s]+)',
                # IB and international
                r'((?:International\s+Baccalaureate|IB\s+Diploma)[^\n,;]*)',
                # Generic diploma/certificate
                r'((?:Diploma|Certificate|Associate\s+Degree)[^\n,;]*)',
                # Higher Secondary
                r'((?:Higher\s+(?:National|Secondary)|Secondary\s+(?:\d|Education))[^\n,;]*)',
            ]

            degree = "Degree not specified"
            for deg_pattern in degree_patterns:
                degree_match = re.search(deg_pattern, edu_chunk, re.IGNORECASE)
                if degree_match:
                    degree = degree_match.group(1).strip()
                    degree = self._clean_education_text(degree)
                    break

            # Try to extract subjects/credits/major mentioned
            subjects_patterns = [
                r'(?:Credits?\s+in|Subjects?|Major(?:ing)?\s+in|Minor(?:ing)?\s+in|Specializ(?:ation|ing)\s+in|Concentration\s+in|Focus(?:ing)?\s+on)\s*:?\s*([^,\n]+(?:,\s*[^,\n]+)*)',
                r'(?:with\s+)?(?:Merit|Distinction|Honours?|Honors?|First\s+Class|Second\s+Class|Upper|Lower)',
            ]

            for subj_pattern in subjects_patterns:
                subjects_match = re.search(subj_pattern, edu_chunk, re.IGNORECASE)
                if subjects_match:
                    if degree == "Degree not specified":
                        subjects = self._clean_education_text(subjects_match.group(1) if subjects_match.lastindex else subjects_match.group(0))
                        degree = f"Credits in: {subjects}"
                    else:
                        subjects = self._clean_education_text(subjects_match.group(1) if subjects_match.lastindex else subjects_match.group(0))
                        if subjects not in degree:
                            degree = f"{degree} - {subjects}"
                    break

            # Extract GPA/CGPA if present
            gpa_pattern = r'(?:GPA|CGPA|Grade\s+Point\s+Average)\s*:?\s*([\d\.]+(?:\s*/\s*[\d\.]+)?)'
            gpa_match = re.search(gpa_pattern, edu_chunk, re.IGNORECASE)
            if gpa_match:
                gpa = gpa_match.group(1)
                if gpa not in degree:
                    degree = f"{degree} (GPA: {gpa})"

            # Extract DATES - more flexible patterns
            date_patterns = [
                r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current))',
                r'(\d{4}\s*[-–—]\s*(?:\d{4}|Present|Current))',
                r'(\d{2}/\d{4}\s*[-–—]\s*(?:\d{2}/\d{4}|Present|Current))',
                r'(?:Graduated|Completed|Awarded)\s*:?\s*(\d{4})',
                r'(?:Class\s+of|Batch\s+of)\s*[\'"]?(\d{4})',
            ]

            dates = "Dates not specified"
            for date_pat in date_patterns:
                date_match = re.search(date_pat, edu_chunk, re.IGNORECASE)
                if date_match:
                    dates = date_match.group(1) if date_match.lastindex else date_match.group(0)
                    dates = dates.replace('–', '-').replace('—', '-')
                    break

            education.append({
                "institution": institution[:300],  # Increased limit
                "degree": degree[:500],            # Increased limit
                "dates": dates
            })

        self.logger.info(f"🎓 Found {len(education)} education entries")

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
IMPORTANT: Use the EXACT wording from the original skills - do NOT paraphrase, rephrase, or modify them in any way.

    SKILLS: {skills_list}

    Return ONLY this JSON format:
    {{"hard": ["skill1", "skill2"], "soft": ["skill3", "skill4"]}}"""

        response = self._call_ollama_raw(prompt, num_predict=300)
        parsed = self._parse_ai_response(response)
        
        if parsed and isinstance(parsed.get('hard'), list) and isinstance(parsed.get('soft'), list):
            result['hard_skills'] = parsed['hard'][:30]
            result['soft_skills'] = parsed['soft'][:20]
            self.logger.info("✅ AI re-categorized skills successfully")
        
        return result

    def _call_ollama_raw(self, prompt: str, num_predict: int = 500) -> str:
        """Helper to call Ollama and return raw text response"""
        try:
            options = {
                'temperature': 0.1,
                'top_p': 0.5,
                'num_ctx': 4096,
                'num_predict': num_predict
            }

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a precise JSON extraction API. Output ONLY valid JSON - no markdown, no text before/after. Extract content EXACTLY as written - preserve original wording word for word, do NOT paraphrase.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options=options
            )

            return response['message']['content']

        except Exception as e:
            self.logger.error(f"❌ AI Call error: {e}")
            return ""

    def _call_ollama(self, prompt: str, is_deep: bool = False) -> Dict:
        """Helper to handle the actual API call"""
        try:
            options = {
                'temperature': 0.1,  # 💅 Tiny bit of warmth for better parsing
                'top_p': 0.5,
                'num_ctx': 4096,
            }
            
            if is_deep:
                options['num_predict'] = 3000  # More room for structured output
            
            # 🌟 HERE'S WHERE THE MAGIC HAPPENS, SWEETIE! 🌟
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': '''You are a precise JSON extraction API.
    Rules:
    1. Output ONLY valid JSON - no markdown, no text before/after
    2. Arrays must contain individual items, never text blocks with bullets
    3. Each experience entry must be a separate object
    4. Extract skills and work experience descriptions EXACTLY as written in the resume - word for word, do NOT paraphrase or summarize
    5. Preserve the original wording and terminology used by the candidate
    Your output will be validated - formatting errors will be rejected.'''
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options=options
            )
            
            response_text = response['message']['content']
            parsed_data = self._parse_ai_response(response_text)
            
            # 🌟 VALIDATION LAYER - Check if AI was lazy!
            if is_deep and parsed_data:
                parsed_data = self._validate_deep_extraction(parsed_data, response_text)
            
            return parsed_data
            
        except Exception as e:
            self.logger.error(f"❌ AI Call error: {e}")
            return {}

    def _parse_ai_response(self, response_text: str) -> Optional[Dict]:
        """
        🔧 Parse JSON and clean text dumps.
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
            self.logger.warning("⚠️ Could not parse JSON from AI response")
            return {}

        # 2. 🛡️ SAFETY NET: Fix Malformed Lists
        # If the AI returned a bulleted string instead of a list, fix it now.
        data = self._fix_malformed_lists(data)
        
        return data

    def _fix_malformed_lists(self, data: Dict) -> Dict:
        """
        🚑 EMERGENCY ROOM for Malformed Data!
        This is like plastic surgery for badly extracted JSON, honey! 💉
        """
        self.logger.info("🔧 Running malformed data repair...")
        
        # ========== FIX SKILLS ==========
        for field in ['hard_skills', 'soft_skills', 'skills']:
            if field in data:
                if isinstance(data[field], str):
                    self.logger.warning(f"⚠️ {field} was a string - converting to list!")
                    raw_text = data[field]
                    
                    # AGGRESSIVE CLEANING - Remove ALL bullet characters
                    cleaned = re.sub(r'[•●○◦▪▫■□►▸‣⁃→·∙⋅▶▷◆◇★☆\-–—*]', '', raw_text)
                    
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
                    self.logger.info(f"✅ Converted {field} to {len(clean_list)} items")
                
                elif isinstance(data[field], list):
                    # Even if it's a list, clean each item
                    cleaned_items = []
                    for item in data[field]:
                        if isinstance(item, str):
                            cleaned = item.strip().strip('•\-–—*◦▪▫')
                            cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)
                            if cleaned and len(cleaned) > 2 and len(cleaned) < 100:
                                cleaned_items.append(cleaned)
                    data[field] = cleaned_items

        # ========== FIX EXPERIENCE ==========
        if 'working_experience' in data:
            if isinstance(data['working_experience'], str):
                self.logger.error("🚨 MAJOR ISSUE: AI dumped experience as text!")
                self.logger.warning("⚠️ Attempting emergency extraction...")
                
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
                    date_match = re.search(r'(\w+\s+\d{4}\s*[-–]\s*(?:\w+\s+\d{4}|Present))', section)
                    dates = date_match.group(1) if date_match else "N/A"
                    
                    # Role is often the second line or has keywords
                    role = "See Description"
                    for line in lines[1:4]:  # Check first few lines
                        if any(keyword in line.lower() for keyword in ['engineer', 'manager', 'developer', 'analyst', 'specialist', 'executive']):
                            role = line.strip()
                            break
                    
                    # Description is the rest (truncated)
                    description = section[:400] if len(section) > 400 else section
                    
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
                    "description": raw_text[:500]
                }]
                
                self.logger.info(f"✅ Extracted {len(data['working_experience'])} jobs from text dump")
            
            elif isinstance(data['working_experience'], list):
                # Validate each job object
                validated_jobs = []
                for job in data['working_experience']:
                    if isinstance(job, dict):
                        # Ensure required fields exist
                        validated_job = {
                            "company": str(job.get('company', 'Unknown'))[:100],
                            "role": str(job.get('role', 'N/A'))[:100],
                            "dates": str(job.get('dates', 'N/A')),
                            "description": str(job.get('description', ''))[:500]
                        }
                        validated_jobs.append(validated_job)
                
                data['working_experience'] = validated_jobs

        # ========== FIX EDUCATION ==========
        if 'education' in data:
            if isinstance(data['education'], str):
                self.logger.warning("⚠️ Education was a string - attempting to structure...")
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
                    if any(char in first_item for char in ['•', '\n', '–']):
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
            self.logger.warning(f"⚠️ VALIDATION ISSUES DETECTED:")
            for issue in issues_found:
                self.logger.warning(f"   ⚠️ {issue}")
            self.logger.info("🔧 Applying aggressive cleanup...")
        
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
        
        # 🆕 STEP 0: Normalize en-dash and em-dash to hyphen FIRST!
        text = text.replace('–', '-')   # en-dash (U+2013)
        text = text.replace('—', '-')   # em-dash (U+2014)
        
        # Remove markdown headers (##, ###, etc.) - must come first!
        # 🆕 Also remove the ** around header text!
        text = re.sub(r'^#+\s*\**([^*\n]+)\**\s*$', r'\1', text, flags=re.MULTILINE)
        
        # Remove bold markers (**text** or __text__)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        
        # Remove italic markers (*text* or _text_) - be careful not to remove bullet points!
        # Only match if there's text between the asterisks (not at line start)
        text = re.sub(r'(?<!\n)\*([^*\n]+)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!\n)_([^_\n]+)_(?!_)', r'\1', text)
        
        # 🆕 Handle bold+italic (***text***)
        text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
        
        # Remove bullet point markers at line start, but KEEP the content!
        text = re.sub(r'^\s*[-\*•▪►▸◦○◇]\s+', '', text, flags=re.MULTILINE)
        
        # Normalize special quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove extra whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _detect_section_boundaries(self, text: str) -> Dict[str, Tuple[int, int]]:
        """
        🎭 FAIRY CODEMOTHER'S SECTION DETECTOR v2.0! 💅
        Now with PROPER ordering and boundary detection!
        Stops certifications AND education from bleeding into work experience!
        """
        sections = {}
        
        # 🎯 Section patterns in ORDER OF APPEARANCE in typical resumes
        # More specific patterns first to avoid false matches!
        section_patterns = [
            ('summary', r'(?:^|\n)\s*(?:PROFESSIONAL\s+)?(?:SUMMARY|PROFILE|OBJECTIVE|ABOUT\s*ME)'),
            ('skills', r'(?:^|\n)\s*(?:SKILLS?|CORE\s+COMPETENCIES|TECHNICAL\s+SKILLS?|KEY\s+SKILLS?)'),
            ('experience', r'(?:^|\n)\s*(?:WORK\s+)?EXPERIENCE[S]?|EMPLOYMENT(?:\s+HISTORY)?|CAREER(?:\s+HISTORY)?'),
            ('education', r'(?:^|\n)\s*EDUCATION(?:AL)?(?:\s+(?:BACKGROUND|HISTORY|QUALIFICATIONS?))?'),
            ('achievements', r'(?:^|\n)\s*ACHIEVEMENTS?|AWARDS?|ACCOMPLISHMENTS?'),
            ('cocurricular', r'(?:^|\n)\s*CO-?CURRICULAR(?:\s+ACTIVITIES)?'),
            ('qualifications', r'(?:^|\n)\s*(?:ADDITIONAL\s+)?QUALIFICATIONS?'),
            ('certifications', r'(?:^|\n)\s*CERTIFICATIONS?|LICENSES?'),
            ('languages', r'(?:^|\n)\s*LANGUAGES?'),
            ('references', r'(?:^|\n)\s*REFERENCES?'),
        ]
        
        # Find all section starts
        found_sections = []
        for section_name, pattern in section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                found_sections.append((section_name, match.start(), match.end()))
                self.logger.debug(f"📍 Found section '{section_name}' at position {match.start()}")
        
        # Sort by position in document
        found_sections.sort(key=lambda x: x[1])
        
        # Calculate boundaries (end of each section is start of next section header)
        for i, (section_name, start_pos, header_end) in enumerate(found_sections):
            # Content starts after the header
            content_start = header_end
            
            if i + 1 < len(found_sections):
                # End at the START of next section header (not end)
                end_pos = found_sections[i + 1][1]
            else:
                end_pos = len(text)
            
            sections[section_name] = (content_start, end_pos)
            self.logger.debug(f"📐 Section '{section_name}': chars {content_start}-{end_pos}")
        
        return sections