"""
🤖 AI Validation Assistant - Powered by TinyLlama!
The backup dancer to our regex diva!
"""

import ollama
import json
import re
from typing import Dict, Optional, Tuple
import logging
import coloredlogs

class AIValidator:
    """
    💅 The AI Assistant that helps validate and extract tricky data!
    """
    
    def __init__(self, model_name: str = "tinyllama:latest"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='INFO', logger=self.logger,
                          fmt='%(asctime)s - 🤖 %(levelname)s - %(message)s')

        self._client = ollama.Client(timeout=60)

        # Test if model is available
        try:
            self.logger.info(f"🎭 Initializing AI Assistant with {model_name}...")
            # Quick test
            response = self._client.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': 'Say "ready"'}
            ])
            self.logger.info("✨ AI Assistant is READY to serve!")
        except Exception as e:
            self.logger.error(f"❌ AI Model error: {e}")
            self.logger.warning("🚨 AI features will be disabled!")
            self.available = False
        else:
            self.available = True
    
    def validate_name(self, candidate_name: str, context: str = "") -> Tuple[bool, float]:
        """
        🔍 Validate if the extracted text is REALLY a person's name!
        Returns: (is_valid, confidence_score)
        """
        if not self.available:
            return True, 0.5 
        
        prompt = f"""Is "{candidate_name}" a real person's name? 

Consider:
- Real names: John Smith, 田中太郎, Nguyen Van A
- NOT names: Engineer, Resume, Objective, 職歴

Answer with ONLY this format:
{{"is_name": true, "confidence": 0.8}}"""
        
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            
            # 🌟 ENHANCED: Robust JSON extraction!
            response_text = response['message']['content']
            
            # Try to extract JSON from the response
            result = self._extract_json_from_response(response_text)
            
            if result:
                is_name = result.get('is_name', True)
                confidence = float(result.get('confidence', 0.5))
                
                self.logger.info(f"🤖 AI validation for '{candidate_name}': {is_name} (confidence: {confidence})")
                return is_name, confidence
            else:
                # Fallback: Try simple pattern matching
                response_lower = response_text.lower()
                if 'not' in response_lower or 'false' in response_lower or 'no' in response_lower:
                    return False, 0.7
                elif 'yes' in response_lower or 'true' in response_lower or 'name' in response_lower:
                    return True, 0.7
                
        except Exception as e:
            self.logger.error(f"❌ AI validation error: {e}")
    
        return True, 0.3  # Default to accepting with very low confidence
    
    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """
        🔧 Extract JSON from messy AI responses!
        Handles cases where AI adds extra text around JSON.
        """
        # Method 1: Try direct parsing
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Method 2: Find JSON-like content
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{.*?"is_name".*?\}',  # JSON with is_name field
            r'\{.*?"confidence".*?\}',  # JSON with confidence field
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean up common issues
                    match = match.replace("'", '"')  # Single to double quotes
                    match = re.sub(r'(\w+):', r'"\1":', match)  # Unquoted keys
                    match = match.replace('True', 'true').replace('False', 'false')
                    
                    result = json.loads(match)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue
        
        # Method 3: Build JSON from patterns in text
        is_name = None
        confidence = None
        
        # Look for true/false
        if re.search(r'\btrue\b|\byes\b', text, re.IGNORECASE):
            is_name = True
        elif re.search(r'\bfalse\b|\bno\b', text, re.IGNORECASE):
            is_name = False
        
        # Look for confidence values
        conf_match = re.search(r'(0\.\d+|1\.0)', text)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        if is_name is not None:
            return {
                "is_name": is_name,
                "confidence": confidence or 0.5
            }
        
        return None

    def extract_from_messy_text(self, text: str, field_type: str) -> Optional[str]:
        """
        🎯 Use AI to extract specific fields from REALLY messy text!
        ENHANCED version with better prompts for TinyLlama!
        """
        if not self.available:
            return None
        
        # 🌟 SIMPLER prompts for TinyLlama!
        prompts = {
            'name': """Find the person's name in this text.
Return ONLY the name, nothing else.
Example: John Smith or 田中太郎""",
            
            'phone': """Find the phone number.
Return ONLY the number.
Example: 090-1234-5678""",
            
            'dob': """Find the birth date.
Return in format: YYYY-MM-DD
Example: 1995-05-23""",
        }
        
        if field_type not in prompts:
            return None
        
        # Limit text to avoid token limits
        text_sample = text[:1500] if len(text) > 1500 else text
        
        prompt = f"{prompts[field_type]}\n\nText:\n{text_sample}"
        
        response = None  # Ensure response is always defined
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.1,
                    'num_predict': 50,
                }
            )
            
            result = response['message']['content'].strip()
            
            # 🌟 VALIDATION based on field type!
            if field_type == 'name':
                # Clean up common AI additions
                result = re.sub(r'^(The |Name:|Person:|Answer:)', '', result, flags=re.IGNORECASE)
                result = result.strip('"\'')
                
                # Validate it looks like a name
                if 2 <= len(result) <= 50 and not any(char in result for char in '@/\\|{}[]'):
                    self.logger.info(f"🤖 AI extracted name: {result}")
                    return result
                    
            elif field_type == 'phone':
                # Extract just the phone number part
                phone_match = re.search(r'[\d\-\s\(\)\.]{10,20}', result)
                if phone_match:
                    self.logger.info(f"🤖 AI extracted phone: {phone_match.group(0)}")
                    return phone_match.group(0)
                    
            elif field_type == 'dob':
                # Look for date pattern
                date_match = re.search(r'\d{4}-\d{1,2}-\d{1,2}', result)
                if date_match:
                    self.logger.info(f"🤖 AI extracted DOB: {date_match.group(0)}")
                    return date_match.group(0)
            
        except Exception as e:
            self.logger.error(f"❌ AI extraction error for {field_type}: {e}")
            self.logger.debug(f"   Response was: {response.get('message', {}).get('content', 'No content') if response else 'No response'}")
        
        return None

    def validate_email(self, email: str) -> bool:
        """
        📧 Validate email format!
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_phone(self, phone: str, region: str = "JP") -> Tuple[bool, str]:
        """
        📱 Validate and format phone numbers!
        Returns: (is_valid, formatted_number)
        """
        # Remove all non-numeric characters for validation
        digits_only = re.sub(r'\D', '', phone)
        
        if region == "JP":
            # Japanese phone validation
            if len(digits_only) == 11 and digits_only.startswith('0'):
                # Format: 090-1234-5678
                formatted = f"{digits_only[:3]}-{digits_only[3:7]}-{digits_only[7:]}"
                return True, formatted
            elif len(digits_only) == 10 and digits_only.startswith('0'):
                # Landline format: 03-1234-5678
                if digits_only[1] in '3456789':  # Major cities
                    formatted = f"{digits_only[:2]}-{digits_only[2:6]}-{digits_only[6:]}"
                else:
                    formatted = f"{digits_only[:3]}-{digits_only[3:6]}-{digits_only[6:]}"
                return True, formatted
        
        # International format
        if 10 <= len(digits_only) <= 15:
            # Basic international validation
            return True, phone
        
        return False, phone
    
    def validate_dob(self, dob_str: str) -> Tuple[bool, Optional[str]]:
        """
        🎂 Validate and format date of birth!
        Returns: (is_valid, formatted_date)
        """
        import datetime
        
        # Target format: YYYY-MM-DD
        formats_to_try = [
            '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y年%m月%d日', '%d-%m-%Y', '%m-%d-%Y'
        ]
        
        for fmt in formats_to_try:
            try:
                date_obj = datetime.datetime.strptime(dob_str.strip(), fmt)
                # Validate reasonable age (18-70 years old)
                age = (datetime.datetime.now() - date_obj).days / 365.25
                if 18 <= age <= 70:
                    return True, date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try to extract date parts with regex
        date_match = re.search(r'(\d{4}).*?(\d{1,2}).*?(\d{1,2})', dob_str)
        if date_match:
            year, month, day = map(int, date_match.groups())
            try:
                date_obj = datetime.datetime(year, month, day)
                age = (datetime.datetime.now() - date_obj).days / 365.25
                if 18 <= age <= 70:
                    return True, date_obj.strftime('%Y-%m-%d')
            except ValueError:
                pass
        
        return False, None

    def fix_vertical_text(self, text: str) -> str:
        """
        🔧 Use AI to fix text that might be vertically formatted
        """
        if not self.available:
            return text
        
        # Check if text might have vertical formatting issues
        if text.count('\n') < 10:  # Not many newlines, probably OK
            return text
        
        prompt = f"""This text might have formatting issues where information is split across lines.
Fix any phone numbers, dates, or names that are broken across multiple lines.

Example of what to fix:
(090)
1234-5678
Should become: (090) 1234-5678

Text to fix:
{text[:1000]}

Return the fixed text, maintaining all information but fixing line breaks."""
        
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )

            fixed_text = response['message']['content']
            if len(fixed_text) > len(text) * 0.5:  # Sanity check
                self.logger.info("🤖 AI fixed vertical formatting issues")
                return fixed_text
                
        except Exception as e:
            self.logger.error(f"❌ AI text fixing error: {e}")
        
        return text