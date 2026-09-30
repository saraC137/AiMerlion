"""
AI Extractor using your fine-tuned model
Works directly with HuggingFace format (no Ollama conversion needed)
"""

import json
import re
from typing import Dict, Optional
import logging
import coloredlogs
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class AIExtractorFineTuned:
    """
    Optimized AI Extractor using YOUR fine-tuned model.
    """
    
    def __init__(self, adapter_path: str = "resume_model_finetuned/final_model"):
        self.adapter_path = adapter_path
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='INFO', logger=self.logger,
                          fmt='%(asctime)s - 🌸 %(levelname)s - %(message)s')
        
        self.model = None
        self.tokenizer = None
        self.available = self._load_model()
        
    def _load_model(self) -> bool:
        """Load the fine-tuned model."""
        try:
            self.logger.info(f"🌸 Loading fine-tuned model from {self.adapter_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_path,
                trust_remote_code=True
            )
            
            # Load base model + LoRA adapter
            base_model_name = "Qwen/Qwen2.5-14B-Instruct"
            
            self.logger.info("Loading base model (this may take a minute)...")
            from transformers import BitsAndBytesConfig
            
            # 4-bit quantization config for 12GB GPU
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.logger.info("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            self.logger.info(f"✨ Fine-tuned model loaded successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model loading error: {e}")
            return False
    
    def extract_all_fields(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract all fields using your fine-tuned model.
        """
        if not self.available:
            return {}
        
        # Limit text
        text_sample = text[:3000] if len(text) > 3000 else text
        
        # Create prompt (same format as training)
        prompt = f"""Extract the following information for the **primary candidate** from this resume. The resume may be in Japanese, English, or mixed.
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

**JSON Output Format:**
{{
  "name": "Taro Tanaka",
  "email": "taro.tanaka@example.com",
  "phone": "090-1234-5678",
  "date_of_birth": "1989-01-01"
}}

If a field is not found, use a `null` value.

Resume text:
{text_sample}

Return ONLY the valid JSON object, nothing else."""
        
        # Format with chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        self.logger.debug(f"🌸 Raw response: {response[:200]}...")
        
        # Parse JSON
        result = self._parse_json_response(response)
        
        if result:
            extracted_count = len([v for v in result.values() if v])
            self.logger.info(f"✨ AI extracted: {extracted_count} fields")
        
        return result
    
    def _parse_json_response(self, response: str) -> Dict[str, Optional[str]]:
        """Parse JSON from model response."""
        try:
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            response = response.strip()
            
            # Find JSON object
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Normalize field names
                result = {
                    'name': data.get('name'),
                    'email': data.get('email'),
                    'phone': data.get('phone'),
                    'date_of_birth': data.get('date_of_birth')
                }
                
                return result
        
        except Exception as e:
            self.logger.error(f"❌ JSON parsing error: {e}")
        
        return {}
    
    def validate_and_enhance(self, regex_results: Dict, text: str) -> Dict:
        """Use AI to validate and fill missing fields."""
        if not self.available:
            return regex_results
        
        ai_results = self.extract_all_fields(text)
        final_results = regex_results.copy()
        
        for field in ['name', 'email', 'phone', 'date_of_birth']:
            if not final_results.get(field) and ai_results.get(field):
                final_results[field] = ai_results[field]
                final_results[f"{field}_source"] = "AI"
            elif final_results.get(field):
                final_results[f"{field}_source"] = "Regex"
        
        final_results['ai_enhanced'] = True
        return final_results


# Backward compatibility alias
AIExtractor = AIExtractorFineTuned