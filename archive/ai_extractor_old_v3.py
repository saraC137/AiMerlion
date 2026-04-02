"""
🌸 AI EXTRACTOR - English Resume Specialist 🌸
Refactored to enforce strict JSON structure and prevent text dumping.
"""

import ollama
import json
import re
from typing import Dict, Optional, List, Any, Union
import logging
import coloredlogs

# Import helper functions from your utils.py
from utils import standardize_phone_number, standardize_date

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
    
    def extract_header_fields(self, text: str) -> Dict[str, Optional[str]]:
        """
        🎯 PASS 1: Extract HEADER fields (Personal Info).
        Fast execution on the first 3000 characters.
        """
        if not self.available:
            return {}
        
        text_sample = text[:3000] if len(text) > 3000 else text
        
        prompt = f"""You are a precise data parser. Extract the candidate's contact details from the resume header.

**Required fields:**
1. name (Full Name)
2. email
3. phone
4. location (City, Country)
5. linkedin (URL)
6. website (URL)

**JSON Output Format ONLY:**
{{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "555-0199",
  "location": "New York, USA",
  "linkedin": "linkedin.com/in/johndoe",
  "website": "null"
}}

Resume text:
{text_sample}

Response must be valid JSON only."""
        
        return self._call_ollama(prompt)

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
        """
        self.logger.info("🔧 Running pure manual extraction...")
        
        result = {
            "hard_skills": [],
            "soft_skills": [],
            "working_experience": [],
            "education": []
        }
        
        # ===== EXTRACT SKILLS =====
        skills_result = self._extract_skills_regex(text)
        result['hard_skills'] = skills_result['hard_skills']
        result['soft_skills'] = skills_result['soft_skills']
        
        # ===== EXTRACT EXPERIENCE =====
        result['working_experience'] = self._extract_experience_regex(text)
        
        # ===== EXTRACT EDUCATION =====
        result['education'] = self._extract_education_regex(text)
        
        return result

    def _extract_skills_regex(self, text: str) -> Dict[str, list]:
        """
        🎯 Extract skills using PURE REGEX
        """
        hard_skills = []
        soft_skills = []
        
        # Find SKILLS section
        skills_patterns = [
            r'SKILLS?\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|WORK|EDUCATION|PROFESSIONAL|$)',
            r'TECHNICAL\s+SKILLS?\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|WORK|EDUCATION|$)',
            r'CORE\s+COMPETENCIES\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|WORK|EDUCATION|$)'
        ]
        
        skills_text = ""
        for pattern in skills_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                skills_text = match.group(1)
                break
        
        if not skills_text:
            self.logger.warning("⚠️ No skills section found")
            return {"hard_skills": [], "soft_skills": []}
        
        # Clean and split skills
        # Remove bullet characters first
        skills_text = re.sub(r'[•●○◦▪▫■□▸▹►▻⦿⦾]', '\n', skills_text)
        skills_text = re.sub(r'^\s*[-–—*]\s*', '\n', skills_text, flags=re.MULTILINE)
        
        # Split by newlines, commas, semicolons
        raw_skills = re.split(r'[\n,;]+', skills_text)
        
        # Clean each skill
        all_skills = []
        for skill in raw_skills:
            cleaned = skill.strip()
            # Remove leading numbers/bullets
            cleaned = re.sub(r'^\d+[\.)]\s*', '', cleaned)
            # Remove excessive whitespace
            cleaned = ' '.join(cleaned.split())
            
            # Filter valid skills
            if 3 <= len(cleaned) <= 80 and cleaned and not cleaned.isdigit():
                # Skip common section headers
                skip_words = ['skills', 'experience', 'education', 'summary', 'objective', 'profile']
                if not any(skip.lower() == cleaned.lower() for skip in skip_words):
                    all_skills.append(cleaned)
        
        # Categorize: Hard vs Soft skills
        tech_keywords = [
            'software', 'system', 'tool', 'platform', 'framework', 'language',
            'python', 'java', 'sql', 'javascript', 'c++', 'c#', 'ruby', 'php',
            'aws', 'azure', 'docker', 'kubernetes', 'linux', 'windows',
            'api', 'database', 'server', 'cloud', 'network', 'security',
            'machine learning', 'ai', 'data', 'analytics', 'crm', 'erp',
            'office', 'excel', 'powerpoint', 'word', 'adobe', 'autocad',
            'testing', 'qa', 'devops', 'agile', 'scrum', 'git', 'ci/cd'
        ]
        
        soft_keywords = [
            'communication', 'leadership', 'management', 'teamwork', 'collaboration',
            'problem solving', 'analytical', 'critical thinking', 'creativity',
            'time management', 'organization', 'presentation', 'negotiation',
            'interpersonal', 'customer service', 'conflict resolution', 'adaptability',
            'decision making', 'strategic', 'planning', 'mentoring', 'coaching'
        ]
        
        for skill in all_skills[:50]:  # Limit to 50 total skills
            skill_lower = skill.lower()
            
            # Check if it's a technical skill
            is_technical = any(keyword in skill_lower for keyword in tech_keywords)
            is_soft = any(keyword in skill_lower for keyword in soft_keywords)
            
            if is_technical:
                hard_skills.append(skill)
            elif is_soft:
                soft_skills.append(skill)
            else:
                # Default categorization based on common patterns
                # If it contains specific tools/software names, it's technical
                if re.search(r'[A-Z]{2,}|[0-9]|\.|/', skill):  # Acronyms, numbers, dots, slashes
                    hard_skills.append(skill)
                else:
                    soft_skills.append(skill)
        
        self.logger.info(f"📊 Found {len(hard_skills)} hard skills, {len(soft_skills)} soft skills")
        
        return {
            "hard_skills": hard_skills[:30],  # Max 30 hard skills
            "soft_skills": soft_skills[:20]   # Max 20 soft skills
        }

    def _extract_experience_regex(self, text: str) -> list:
        """
        💼 Extract work experience using IMPROVED REGEX
        Now handles markdown, flexible formats, and edge cases!
        """
        jobs = []
        
        # 🧹 STEP 1: CLEAN THE TEXT FIRST!
        text = self._clean_markdown(text)
        self.logger.info("🧹 Cleaned markdown formatting")
        
        # 🔍 STEP 2: FIND EXPERIENCE SECTION
        exp_patterns = [
            r'(?:WORK\s+)?EXPERIENCE\s*:?\s*(.*?)(?=EDUCATION|SKILLS|CERTIFICATIONS|AWARDS|$)',
            r'EMPLOYMENT\s+(?:HISTORY|BACKGROUND)\s*:?\s*(.*?)(?=EDUCATION|SKILLS|$)',
            r'PROFESSIONAL\s+(?:EXPERIENCE|BACKGROUND)\s*:?\s*(.*?)(?=EDUCATION|SKILLS|$)',
        ]
        
        exp_text = ""
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                exp_text = match.group(1)
                self.logger.info(f"💼 Found experience section ({len(exp_text)} chars)")
                break
        
        # 🚨 FALLBACK: If no section header, look for job entries in full text
        if not exp_text or len(exp_text) < 100:
            self.logger.warning("⚠️ No clear experience section - scanning full document")
            exp_text = text
        
        # 🏢 STEP 3: FIND ALL COMPANIES
        # Try multiple patterns in order of specificity
        company_patterns = [
            # Pattern 1: Company with legal suffix
            (r'([A-Z][A-Za-z0-9\s&,\.\'\-]+\b(?:Pvt\.?\s*Ltd\.?|Private\s+Limited|Ltd\.?|Limited|Inc\.?|Incorporated|Corp\.?|Corporation|LLC|Company|Co\.)[^\n]*)', 'legal_suffix'),
            
            # Pattern 2: Company with business keywords
            (r'([A-Z][A-Za-z\s&,\.\'\-]+\b(?:Technologies|Solutions|Services|Systems|Group|International|Industries|Consulting|Partners|Holdings|Enterprises)[^\n]*)', 'business_keyword'),
            
            # Pattern 3: Line before a job title
            (r'([A-Z][A-Za-z\s&,\.\'\-]{10,80})\s*\n\s*(?:Senior|Junior|Lead|Chief|Principal)?\s*(?:Engineer|Manager|Developer|Analyst|Consultant|Director|Executive|Coordinator|Specialist)', 'before_title'),
        ]
        
        all_matches = []
        for pattern, pattern_type in company_patterns:
            matches = list(re.finditer(pattern, exp_text, re.MULTILINE))
            if matches:
                self.logger.info(f"🏢 Found {len(matches)} companies using '{pattern_type}' pattern")
                all_matches.extend([(m, pattern_type) for m in matches])
        
        if not all_matches:
            self.logger.error("❌ No companies found in text!")
            return jobs
        
        # Remove duplicates based on position
        seen_positions = set()
        unique_matches = []
        for match, ptype in all_matches:
            pos = match.start()
            if pos not in seen_positions:
                seen_positions.add(pos)
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
                chunk_end = min(chunk_start + 1500, len(exp_text))
            
            job_chunk = exp_text[chunk_start:chunk_end]
            
            # 👔 EXTRACT ROLE
            role_patterns = [
                r'\n\s*((?:Senior|Junior|Lead|Principal|Chief|Head\s+of|VP\s+of)?\s*[A-Z][A-Za-z\s]+(?:Engineer|Manager|Developer|Analyst|Specialist|Executive|Consultant|Director|Officer|Coordinator|Lead|Architect|Technician|Administrator|Designer|Associate|Assistant|Supervisor|Trainee|Intern))',
                r'\n\s*([A-Z][A-Za-z\s]{5,60})',  # Any capitalized phrase (fallback)
            ]
            
            role = "Position not specified"
            for pattern in role_patterns:
                role_match = re.search(pattern, job_chunk, re.MULTILINE)
                if role_match:
                    potential_role = role_match.group(1).strip()
                    potential_role = re.sub(r'\s+', ' ', potential_role)
                    
                    # Validate it's not the company name again
                    if potential_role.lower() != company_name.lower() and len(potential_role) >= 5:
                        role = potential_role
                        break
            
            # 📅 EXTRACT DATES
            date_patterns = [
                r'((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}\s*[-–—to]+\s*(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}|Present|Current|present|current|ongoing|Ongoing))',
                r'(\d{1,2}/\d{4}\s*[-–—to]+\s*(?:\d{1,2}/\d{4}|Present|Current))',
                r'(\d{4}\s*[-–—to]+\s*(?:\d{4}|Present|Current|present|current))',
            ]
            
            dates = "Dates not specified"
            for pattern in date_patterns:
                date_match = re.search(pattern, job_chunk, re.IGNORECASE)
                if date_match:
                    dates = date_match.group(1)
                    # Normalize
                    dates = dates.replace('–', ' - ').replace('—', ' - ').replace('to', '-')
                    dates = re.sub(r'\s+', ' ', dates).strip()
                    dates = re.sub(r'(present|current|ongoing)', 'Present', dates, flags=re.IGNORECASE)
                    break
            
            # 📝 EXTRACT DESCRIPTION (First 3-4 bullet points)
            # Find where dates end
            if dates != "Dates not specified":
                date_pos = job_chunk.find(dates)
                desc_start = date_pos + len(dates) if date_pos != -1 else 0
            else:
                desc_start = len(company_name) + len(role) + 50
            
            desc_chunk = job_chunk[desc_start:desc_start + 800]
            
            # Look for bullet points
            bullet_lines = []
            for line in desc_chunk.split('\n'):
                line = line.strip()
                # Skip empty lines and very short lines
                if len(line) < 10:
                    continue
                # Skip lines that look like headers or locations
                if re.match(r'^[A-Z][a-z]+,\s+[A-Z][a-z]+$', line):  # "Mumbai, India"
                    continue
                # Add to description
                bullet_lines.append(line)
                if len(bullet_lines) >= 4:  # Max 4 bullets
                    break
            
            if bullet_lines:
                description = ' | '.join(bullet_lines)
            else:
                # Fallback: take first 300 chars
                description = desc_chunk[:300].replace('\n', ' ').strip()
            
            # Clean description
            description = re.sub(r'\s+', ' ', description)
            description = description[:500]  # Max 500 chars
            
            # 🎯 STEP 5: ADD TO RESULTS
            jobs.append({
                "company": company_name[:100],
                "role": role[:100],
                "dates": dates,
                "description": description if description else "Description not available"
            })
            
            self.logger.info(f"✅ Extracted: {company_name[:40]} - {role[:40]}")
        
        self.logger.info(f"💼 Total jobs extracted: {len(jobs)}")
        
        return jobs

    def _extract_education_regex(self, text: str) -> list:
        """
        🎓 Extract education using PURE REGEX
        """
        education = []
        
        # Find EDUCATION section - try multiple patterns
        edu_patterns = [
            r'EDUCATION(?:AL)?\s*(?:BACKGROUND|HISTORY)?\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|WORK|SKILLS|CERTIFICATIONS|AWARDS|PUBLICATIONS|REFERENCES|PROJECTS|$)',
            r'ACADEMIC\s+(?:BACKGROUND|QUALIFICATIONS|CREDENTIALS|HISTORY)\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|SKILLS|$)',
            r'QUALIFICATIONS?\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|SKILLS|CERTIFICATIONS|$)',
            r'TRAINING\s+(?:AND|&)\s+EDUCATION\s*:?\s*(.*?)(?=EXPERIENCE|EMPLOYMENT|SKILLS|$)',
        ]

        edu_text = ""
        for pattern in edu_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                edu_text = match.group(1)
                self.logger.info(f"🎓 Found education section")
                break

        # If no section header found, try to find education entries in the entire text
        if not edu_text:
            self.logger.info("🎓 No education section header found, searching entire document...")
            # Look for university/college patterns anywhere in the document
            inst_pattern = r'([A-Z][A-Za-z\s\',\.&-]+(?:University|College|Institute|School|Academy)[^\n]*)'
            if re.search(inst_pattern, text):
                # Use a reasonable chunk of text that might contain education info
                # Usually education is at the bottom of resume
                edu_text = text[-2000:] if len(text) > 2000 else text
            else:
                self.logger.info("🎓 No education information found in document")
                return education
        
        # Pattern for institutions
        inst_pattern = r'([A-Z][A-Za-z\s\',\.&-]+(?:University|College|Institute|School|Academy)[^\n]*)'
        inst_matches = list(re.finditer(inst_pattern, edu_text))
        
        for i, inst_match in enumerate(inst_matches):
            institution = inst_match.group(1).strip()
            
            # Get chunk for this education entry
            chunk_start = inst_match.start()
            if i + 1 < len(inst_matches):
                chunk_end = inst_matches[i + 1].start()
            else:
                chunk_end = min(chunk_start + 300, len(edu_text))
            
            edu_chunk = edu_text[chunk_start:chunk_end]
            
            # Extract DEGREE
            degree_pattern = r'((?:Bachelor|Master|PhD|Ph\.?\s*D\.?|Doctorate|Diploma|Associate|B\.?\s*[ASE]\.?|M\.?\s*[ASE]\.?|B\.?\s*Tech|M\.?\s*Tech)[^\n]+)'
            degree_match = re.search(degree_pattern, edu_chunk, re.IGNORECASE)
            degree = degree_match.group(1).strip() if degree_match else "Degree not specified"
            
            # Extract DATES
            date_pattern = r'\d{4}\s*[-–—]\s*\d{4}'
            date_match = re.search(date_pattern, edu_chunk)
            dates = date_match.group(0) if date_match else "Dates not specified"
            
            education.append({
                "institution": institution[:150],
                "degree": degree[:150],
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
                        'content': 'You are a precise JSON extraction API. Output ONLY valid JSON - no markdown, no text before/after.'
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
    4. Keep descriptions concise (under 300 chars per job)
    5. Never copy-paste resume sections verbatim
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
                    cleaned = re.sub(r'[•\-–—*â€¢◦▪▫]', '', raw_text)
                    
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
        🔍 Merge Regex + AI results
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

        # Pass 2: Deep Fields
        ai_deep = self.extract_deep_fields(text)
        if ai_deep:
            final_results['hard_skills'] = ai_deep.get('hard_skills', [])
            final_results['soft_skills'] = ai_deep.get('soft_skills', [])
            
            # Combine for backwards compatibility
            final_results['skills'] = final_results['hard_skills'] + final_results['soft_skills']
            
            final_results['working_experience'] = ai_deep.get('working_experience', [])
            final_results['education'] = ai_deep.get('education', [])
            final_results['experience_source'] = "AI_Deep"

        return self._post_process_results(final_results)

    def _post_process_results(self, results: Dict) -> Dict:
        """
        🎨 Final Cleanup
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

        # Ensure Lists
        for f in ['hard_skills', 'soft_skills', 'working_experience', 'education']:
            if f not in processed or not isinstance(processed[f], list):
                processed[f] = []

        processed['ai_enhanced'] = True
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
        This is like removing makeup before bed, darling! 💄
        """
        # Remove markdown headers (##, ###, etc.)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic markers (**text** or *text*)
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *italic* -> italic
        
        # Remove bullet point markers (-, *, •)
        # Keep the content, just remove the marker at line start
        text = re.sub(r'^\s*[-\*•]\s+', '', text, flags=re.MULTILINE)
        
        # Normalize dashes (em-dash, en-dash -> regular dash)
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines -> 2 lines
        
        return text