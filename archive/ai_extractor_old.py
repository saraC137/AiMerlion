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
    
    def __init__(self, model_name: str):
        self.model_name = model_name
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
        🎯 Extract ALL fields in one go using Suzume's power!
        This is more efficient than multiple calls!
        """
        if not self.available:
            return {}
        
        # Limit text to prevent token overflow
        text_sample = text[:3000] if len(text) > 3000 else text
        
        # 🌟 POWER PROMPT for AI Model!
        prompt = f"""あなたは履歴書から情報を抽出する専門家です。以下のテキストから個人情報を抽出してください。

Extract the following information from this resume. The resume may be in Japanese, English, or mixed.
Return your answer in JSON format ONLY, with no additional text.

Required fields:
1. name: The person's full name (名前/氏名)
2. name_japanese: Japanese name if available (漢字 or カタカナ)
3. name_english: English/Romanized name if available
4. email: Email address (メールアドレス)
5. phone: Phone number (電話番号) - include full number
6. date_of_birth: Date of birth (生年月日) - format as YYYY-MM-DD

Look for name after these headers:
- "Name:" or "Full Name:" or "Candidate Name:"
- "氏名" or "名前" or "お名前"
- Or look at the VERY FIRST line (often the name)

Return the complete name (first and last name).

Look for text in format: username@domain.com

For International Phone Numbers:
- International format starting with + (e.g., +91-xxx, +81-xxx)
- If no international format, extract domestic format (090-xxx, 080-xxx)

Look for dates near:
- "Date of Birth:" or "DOB:" or "Born:"
- "生年月日"

The date may be in formats:
- DD/MM/YYYY (22/01/1997)
- YYYY-MM-DD (1997-01-22)
- Japanese format with 年月日

Return in YYYY-MM-DD format.
For Japanese dates:
- 令和5年 = 2023年
- 平成31年/令和元年 = 2019年
- 平成元年 = 1989年
- 昭和64年/平成元年 = 1989年

JSON format:
{{
  "name": "extracted name",
  "name_japanese": "日本語名前",
  "name_english": "English Name",
  "email": "email@example.com",
  "phone": "090-1234-5678",
  "date_of_birth": "1995-05-23"
}}

If a field is not found, use null.

Resume text:
{text_sample}

Remember: Return ONLY the JSON object, no explanations."""
        
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
            email = results['email'].lower().strip()
            if '@' in email:
                processed['email'] = email
        
        # 🌟 USE THE NEW STANDARDIZATION FUNCTIONS 🌟
        # Process phone
        if results.get('phone'):
            processed['phone'] = standardize_phone_number(results['phone'])
        
        # Process date of birth
        if results.get('date_of_birth'):
            processed['date_of_birth'] = standardize_date(results['date_of_birth'])
        
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
        🔍 Validate regex results and fill in missing fields with AI
        """
        if not self.available:
            return regex_results
        
        # Get AI extraction
        ai_results = self.extract_all_fields(text)
        
        # Merge results intelligently
        final_results = regex_results.copy()
        
        # For each field, prefer regex if available, otherwise use AI
        for field in ['name', 'name_japanese', 'email', 'phone', 'date_of_birth']:
            if not final_results.get(field) and ai_results.get(field):
                final_results[field] = ai_results[field]
                final_results[f"{field}_source"] = "AI"
            elif final_results.get(field):
                final_results[f"{field}_source"] = "Regex"
        
        # Special handling for names - validate with AI
        if final_results.get('name') and ai_results.get('name'):
            # If AI found a different name, it might be more accurate
            if final_results['name'].lower() != ai_results['name'].lower():
                self.logger.warning(f"🤔 Name mismatch - Regex: {final_results['name']}, AI: {ai_results['name']}")
                # You can add logic here to choose which one to trust
        
        final_results['ai_enhanced'] = True
        return final_results