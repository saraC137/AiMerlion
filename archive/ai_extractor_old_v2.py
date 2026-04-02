"""
🌸 AI EXTRACTOR - The Multilingual Extraction DIVA! 🌸
Powered by AI for resume excellence!
"""

import ollama
import json
import re
from typing import Dict, Optional, List, Tuple
import logging
import coloredlogs
from datetime import datetime
import unicodedata

class AIExtractor:
    """
    💅 The MULTILINGUAL AI Queen! Specializes in Japanese resumes!
    """
    
    def __init__(self, model_name: str, logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            coloredlogs.install(level='INFO', logger=self.logger,
                              fmt='%(asctime)s - 🌸 %(levelname)s - %(message)s')
        
        # Test model availability
        self.available = self._test_model()
        
    def _test_model(self) -> bool:
        """Test if the model is ready to serve!"""
        try:
            self.logger.info(f"🌸 Initializing {self.model_name} model...")
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'こんにちは'}]
            )
            self.logger.info(f"✨ {self.model_name} is READY! Multilingual power activated!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Model error: {e}")
            self.logger.error(f"Please ensure the model '{self.model_name}' is available.")
            return False
    
    def extract_all_fields(self, text: str) -> Dict[str, Optional[str]]:
        """
        🎯 Extract ALL header fields in one go using Suzume's power!
        This is efficient for personal details.
        """
        if not self.available:
            return {}
        
        # INCREASED LIMIT for better context, or remove if you trust your token limit.
        text_sample = text[:5000] if len(text) > 5000 else text
        
        # 🌟 POWER PROMPT for AI Model!
        prompt = f"""あなたは履歴書から情報を抽出する専門家です。以下のテキストから**候補者本人**の個人情報を正確に抽出してください。

Extract the following information for the **primary candidate** from this resume. The resume may be in Japanese, English, or mixed.
Return your answer in JSON format ONLY, with no additional text.

**Extraction Strategy:**
- The candidate's personal information is almost always at the top of the resume. Prioritize information found in the header.
- Do not extract contact details from references or previous employers.

**Required fields:**
1.  **name**: The candidate's full name (名前/氏名). This is usually the most prominent text at the top.
2.  **email**: The candidate's personal email address (メールアドレス).
3.  **phone**: The candidate's primary contact phone number (電話番号).
4.  **date_of_birth**: The candidate's date of birth (生年月日).
    - **Look for multiple formats**: "January 1, 1990", "1990/01/22", "01-22-1990", "昭和64年1月1日".
    - **Convert to YYYY-MM-DD format** in the final JSON output.
5.  **location**: The candidate's current location (現住所).
6.  **school_university**: The candidate's school or university (学歴).
**// Removed skills and working_experience from this prompt**

**JSON Output Format:**
{{
  "name": "Taro Tanaka",
  "email": "taro.tanaka@example.com",
  "phone": "090-1234-5678",
  "date_of_birth": "1989-01-01",
  "location": "Tokyo, Japan",
  "school_university": "University of Tokyo"
}}

If a field is not found, use a `null` value.

Resume text:
{text_sample}

Remember: Return ONLY the valid JSON object, nothing else."""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a precise data extraction assistant. Always respond with valid JSON only.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 500,
                }
            )
            
            response_text = response['message']['content']
            self.logger.debug(f"🌸 Raw AI response: {response_text[:200]}...")
            
            # Extract and parse JSON
            result = self._parse_ai_response(response_text)
            
            if result:
                # Post-process the results
                result = self._post_process_results(result)
                self.logger.info(f"✨ AI extracted: {len([v for v in result.values() if v])} fields")
                return result
            
        except Exception as e:
            self.logger.error(f"❌ AI extraction error: {e}")
            
        return {}
    
    def _parse_ai_response(self, response_text: str) -> Optional[Dict]:
        """
        🔧 Parse JSON from Suzume's response with multiple fallback strategies
        """
        # Strategy 1: Direct JSON parsing
        try:
            # Remove any markdown code blocks
            cleaned = re.sub(r'```json\s*|\s*```', '', response_text)
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Extract fields individually from response
        result = {}
        
        # Extract each field with patterns
        patterns = {
            'name': r'"name":\s*"([^"]+)"',
            'name_japanese': r'"name_japanese":\s*"([^"]+)"',
            'name_english': r'"name_english":\s*"([^"]+)"',
            'email': r'"email":\s*"([^"]+)"',
            'phone': r'"phone":\s*"([^"]+)"',
            'date_of_birth': r'"date_of_birth":\s*"([^"]+)"',
            'skills': r'"skills":\s*\[([^]]+)\]',
            'working_experience': r'"working_experience":\s*\[([^]]+)\]',
            'location': r'"location":\s*"([^"]+)"',
            'school_university': r'"school_university":\s*"([^"]+)"'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, response_text)
            if match:
                value = match.group(1)
                if value and value.lower() != 'null':
                    result[field] = value
        
        return result if result else None
    
    def _post_process_results(self, results: Dict) -> Dict:
        """
        🎨 Clean and standardize extracted results using our new utility functions!
        """
        # We need our new functions!
        from utils import standardize_phone_number, standardize_date

        processed = {}
        
        # Process name fields
        if results.get('name'):
            processed['name'] = self._clean_name(results['name'])
        elif results.get('name_japanese'):
            processed['name'] = results['name_japanese']
        elif results.get('name_english'):
            processed['name'] = results['name_english']
        
        # Store Japanese name separately if available
        if results.get('name_japanese'):
            processed['name_japanese'] = self._clean_name(results['name_japanese'])
        
        # Process email
        if results.get('email'):
            email = results['email']
            if isinstance(email, str):
                email = email.lower().strip()
                if '@' in email:
                    processed['email'] = email
        
        # 🌟 USE THE NEW STANDARDIZATION FUNCTIONS 🌟
        # Process phone
        if results.get('phone'):
            phone = results.get('phone')
            if isinstance(phone, str):
                processed['phone'] = standardize_phone_number(phone)
            elif isinstance(phone, list) and phone:
                processed['phone'] = standardize_phone_number(phone[0]) if isinstance(phone[0], str) else None
        
        # Process date of birth
        if results.get('date_of_birth'):
            dob = results.get('date_of_birth')
            if isinstance(dob, str):
                processed['date_of_birth'] = standardize_date(dob)
            elif isinstance(dob, list) and dob:
                processed['date_of_birth'] = standardize_date(dob[0]) if isinstance(dob[0], str) else None

        # Process new fields
        if results.get('skills'):
            skills_list = []
            raw_skills = results.get('skills')
            if isinstance(raw_skills, list):
                for item in raw_skills:
                    if isinstance(item, str):
                        skills_list.append(item.strip())
                    elif isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, str):
                                skills_list.append(sub_item.strip())
            elif isinstance(raw_skills, str):
                # From regex fallback
                skills_list = [s.strip().strip('"') for s in raw_skills.split(',')]
            processed['skills'] = [s for s in skills_list if s]

        if results.get('working_experience'):
            experience_list = []
            raw_experience = results.get('working_experience')
            if isinstance(raw_experience, list):
                for item in raw_experience:
                    if isinstance(item, str):
                        experience_list.append(item.strip())
                    elif isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, str):
                                experience_list.append(sub_item.strip())
            elif isinstance(raw_experience, str):
                experience_list = [s.strip().strip('"') for s in raw_experience.split(',')]
            processed['working_experience'] = [e for e in experience_list if e]

        if results.get('location'):
            location = results.get('location')
            if isinstance(location, str):
                processed['location'] = location.strip()
            elif isinstance(location, list) and location:
                processed['location'] = str(location[0]).strip()

        if results.get('school_university'):
            school = results.get('school_university')
            if isinstance(school, str):
                processed['school_university'] = school.strip()
            elif isinstance(school, list) and school:
                processed['school_university'] = str(school[0]).strip()
        
        return processed

    def _clean_name(self, name: str) -> str:
        """Clean and validate name"""
        name = name.strip()
        # Remove common prefixes/suffixes that aren't part of the name
        name = re.sub(r'^(Mr\.|Ms\.|Mrs\.|Dr\.|様|さん|氏)\s*', '', name)
        name = re.sub(r'\s*(様|さん|氏)$', '', name)
        return name
    
    def _standardize_phone(self, phone: str) -> str:
        """Standardize phone number format"""
        # Convert full-width to half-width
        phone = unicodedata.normalize('NFKC', phone)
        
        # Extract digits
        digits = re.sub(r'\D', '', phone)
        
        # Format based on length and pattern
        if len(digits) == 11 and digits.startswith('0'):
            # Japanese mobile: 090-1234-5678
            return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
        elif len(digits) == 10 and digits.startswith('0'):
            # Japanese landline: 03-1234-5678
            return f"{digits[:2]}-{digits[2:6]}-{digits[6:]}"
        elif digits.startswith('81'):
            # International format
            return f"81-{digits[2:4]}-{digits[4:8]}-{digits[8:]}"
        
        return phone  # Return as-is if can't standardize
    
    def _standardize_date(self, date_str: str) -> Optional[str]:
        """Standardize date to YYYY-MM-DD format"""
        # Already in correct format?
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # Try various formats
        date_patterns = [
            (r'(\d{4})[/年](\d{1,2})[/月](\d{1,2})', '{}-{:02d}-{:02d}'),
            (r'(\d{1,2})[/](\d{1,2})[/](\d{4})', '{2}-{0:02d}-{1:02d}'),
        ]
        
        for pattern, format_str in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    parts = [int(g) for g in match.groups()]
                    return format_str.format(*parts)
                except:
                    pass
        
        return None
    
    def validate_and_enhance(self, regex_results: Dict, text: str) -> Dict:
        """
        🔍 Validate regex results and fill in missing fields with AI.
        Uses two AI calls for maximum completeness.
        """
        if not self.available:
            return regex_results
        
        # 1. Get AI extraction for simple/header fields
        ai_header_results = self.extract_all_fields(text)
        
        # 2. Get AI extraction for deep/list fields
        ai_deep_results = self.extract_deep_fields(text) # NEW CALL
        
        # Merge results intelligently
        final_results = regex_results.copy()
        
        # Merge Header Fields (Prioritize Regex/Existing, then AI Header)
        for field in ['name', 'email', 'phone', 'date_of_birth', 'location']:
            if not final_results.get(field) and ai_header_results.get(field):
                final_results[field] = ai_header_results[field]
                final_results[f"{field}_source"] = "AI"
            elif final_results.get(field):
                final_results[f"{field}_source"] = "Regex"
        
        # Merge Deep Fields (Prioritize AI Deep Extraction)
        # For these complex lists, we trust the dedicated AI call more than simple regex/initial AI
        for field in ['skills', 'working_experience', 'school_university']:
            if ai_deep_results.get(field):
                # The deep AI call is better at lists, so we use it directly
                final_results[field] = ai_deep_results[field]
                final_results[f"{field}_source"] = "AI_Deep"
            elif final_results.get(field) and isinstance(final_results[field], list) and len(final_results[field]) > 0:
                final_results[f"{field}_source"] = "Regex" # Keep existing if AI found nothing
            else:
                final_results[field] = [] # Ensure it's an empty list if nothing is found
        
        # Special handling for names - validate with AI
        if final_results.get('name') and ai_results.get('name'):
            # If AI found a different name, it might be more accurate
            if final_results['name'].lower() != ai_results['name'].lower():
                self.logger.warning(f"🤔 Name mismatch - Regex: {final_results['name']}, AI: {ai_results['name']}")
                # You can add logic here to choose which one to trust
        
        final_results['ai_enhanced'] = True
        return final_results
    
    def extract_deep_fields(self, text: str) -> Dict[str, List[str]]:
        """
        🚀 Extract deep, list-based fields (Skills, Experience) with a dedicated call.
        """
        if not self.available:
            return {}

        # Use the FULL TEXT for deep extraction
        
        prompt = f"""あなたは履歴書から職務経歴とスキルを抽出する専門家です。以下の履歴書テキストから**候補者本人**の職務経歴、スキル、学歴をすべて正確に抽出してください。

Extract ALL items for the following list-based sections for the **primary candidate** from this resume. The resume may be in Japanese, English, or mixed.
Return your answer in JSON format ONLY, with no additional text.

**Required fields (return as lists of strings):**
1.  **skills**: A list of ALL the candidate's technical and soft skills (スキル/特技).
2.  **working_experience**: A detailed list of the candidate's ALL previous jobs (職務経歴). **Include the Company Name, Role, and Key Accomplishments/Dates for each entry.**
3.  **school_university**: A detailed list of the candidate's ALL educational history (学歴/職歴). **Include the Institution Name, Degree/Major, and Graduation Year.**

**JSON Output Format:**
{{
  "skills": ["Python", "Machine Learning", "Data Analysis", "Japanese N2"],
  "working_experience": [
    "2020-Present: Software Engineer at ABC Inc. - Developed microservices for payment processing, reducing latency by 15%.", 
    "2018-2020: Data Scientist at XYZ Corp. - Built predictive models using scikit-learn for customer churn reduction."
  ],
  "school_university": [
    "2016-2018: Master of Science in Computer Science, University of Tokyo", 
    "2012-2016: Bachelor of Arts in English, Waseda University"
  ]
}}

If a field is not found, use an empty list `[]`.

Resume text:
{text}

Remember: Return ONLY the valid JSON object, nothing else."""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a precise data extraction assistant. Always respond with valid JSON only.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 1000, # Increased prediction length for long lists
                }
            )
            response_text = response['message']['content']
            result = self._parse_ai_response(response_text)
            
            # Clean up lists that might come back as single strings from regex fallback
            if result:
                for key in ['skills', 'working_experience', 'school_university']:
                    if isinstance(result.get(key), str):
                        # Attempt to convert a string (from regex fallback) into a list
                        result[key] = [s.strip().strip('"') for s in result[key].split(',') if s.strip()]
                    elif result.get(key) is None:
                        result[key] = []
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Deep AI extraction error: {e}")
            return {}