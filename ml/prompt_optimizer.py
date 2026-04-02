"""
Prompt Optimization Framework
Tests multiple prompt strategies and measures performance against labeled data
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import ollama
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from fuzzywuzzy import fuzz
import re

class PromptOptimizer:
    """
    Tests different prompt strategies to find the best extraction approach.
    """
    
    def __init__(self, model_name: str, dataset_path: str = "dataset.jsonl"):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.labeled_data = []
        self.results = {}
        
        # Load dataset
        self._load_dataset()
        
        print(f"Initialized optimizer with model: {model_name}")
        print(f"Loaded {len(self.labeled_data)} labeled samples")
    
    def _load_dataset(self):
        """Load labeled dataset."""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.labeled_data.append(json.loads(line))
                    except:
                        pass
    
    def test_phone_extraction_strategies(self, max_samples: int = 50):
        """
        Test multiple phone extraction strategies.
        """
        print("\n" + "="*70)
        print("TESTING PHONE EXTRACTION STRATEGIES")
        print("="*70)
        
        strategies = {
            'strategy_1_international_priority': self._phone_prompt_intl_priority,
            'strategy_2_all_phones': self._phone_prompt_all_phones,
            'strategy_3_context_aware': self._phone_prompt_context_aware,
            'strategy_4_format_flexible': self._phone_prompt_format_flexible
        }
        
        results = {}
        
        for strategy_name, prompt_fn in strategies.items():
            print(f"\nTesting: {strategy_name}")
            accuracy = self._test_strategy(prompt_fn, 'phone', max_samples)
            results[strategy_name] = accuracy
            print(f"  Accuracy: {accuracy:.2f}%")
        
        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1])
        print(f"\nBest phone strategy: {best_strategy[0]} ({best_strategy[1]:.2f}%)")
        
        self.results['phone'] = {
            'strategies': results,
            'best': best_strategy[0],
            'best_accuracy': best_strategy[1]
        }
        
        return results
    
    def test_name_extraction_strategies(self, max_samples: int = 50):
        """
        Test multiple name extraction strategies.
        """
        print("\n" + "="*70)
        print("TESTING NAME EXTRACTION STRATEGIES")
        print("="*70)
        
        strategies = {
            'strategy_1_header_focus': self._name_prompt_header_focus,
            'strategy_2_position_aware': self._name_prompt_position_aware,
            'strategy_3_format_detection': self._name_prompt_format_detection,
            'strategy_4_validation_strict': self._name_prompt_validation_strict
        }
        
        results = {}
        
        for strategy_name, prompt_fn in strategies.items():
            print(f"\nTesting: {strategy_name}")
            accuracy = self._test_strategy(prompt_fn, 'name', max_samples)
            results[strategy_name] = accuracy
            print(f"  Accuracy: {accuracy:.2f}%")
        
        best_strategy = max(results.items(), key=lambda x: x[1])
        print(f"\nBest name strategy: {best_strategy[0]} ({best_strategy[1]:.2f}%)")
        
        self.results['name'] = {
            'strategies': results,
            'best': best_strategy[0],
            'best_accuracy': best_strategy[1]
        }
        
        return results
    
    def test_email_extraction_strategies(self, max_samples: int = 50):
        """
        Test email extraction strategies.
        """
        print("\n" + "="*70)
        print("TESTING EMAIL EXTRACTION STRATEGIES")
        print("="*70)
        
        strategies = {
            'strategy_1_simple': self._email_prompt_simple,
            'strategy_2_validation': self._email_prompt_validation,
            'strategy_3_multiple_handling': self._email_prompt_multiple
        }
        
        results = {}
        
        for strategy_name, prompt_fn in strategies.items():
            print(f"\nTesting: {strategy_name}")
            accuracy = self._test_strategy(prompt_fn, 'email', max_samples)
            results[strategy_name] = accuracy
            print(f"  Accuracy: {accuracy:.2f}%")
        
        best_strategy = max(results.items(), key=lambda x: x[1])
        print(f"\nBest email strategy: {best_strategy[0]} ({best_strategy[1]:.2f}%)")
        
        self.results['email'] = {
            'strategies': results,
            'best': best_strategy[0],
            'best_accuracy': best_strategy[1]
        }
        
        return results
    
    def test_dob_extraction_strategies(self, max_samples: int = 50):
        """
        Test date of birth extraction strategies.
        """
        print("\n" + "="*70)
        print("TESTING DATE OF BIRTH EXTRACTION STRATEGIES")
        print("="*70)
        
        strategies = {
            'strategy_1_format_aware': self._dob_prompt_format_aware,
            'strategy_2_era_conversion': self._dob_prompt_era_conversion,
            'strategy_3_context_search': self._dob_prompt_context_search
        }
        
        results = {}
        
        for strategy_name, prompt_fn in strategies.items():
            print(f"\nTesting: {strategy_name}")
            accuracy = self._test_strategy(prompt_fn, 'date_of_birth', max_samples)
            results[strategy_name] = accuracy
            print(f"  Accuracy: {accuracy:.2f}%")
        
        best_strategy = max(results.items(), key=lambda x: x[1])
        print(f"\nBest DOB strategy: {best_strategy[0]} ({best_strategy[1]:.2f}%)")
        
        self.results['date_of_birth'] = {
            'strategies': results,
            'best': best_strategy[0],
            'best_accuracy': best_strategy[1]
        }
        
        return results
    
    def _test_strategy(self, prompt_fn, field: str, max_samples: int) -> float:
        """
        Test a single prompt strategy against labeled data.
        """
        correct = 0
        total = 0
        
        samples = self.labeled_data[:max_samples]
        
        for sample in samples:
            text = sample['input']
            
            # Get ground truth
            gt_data = sample['output'][0] if isinstance(sample['output'], list) else sample['output']
            
            if field == 'phone':
                contact_info = gt_data.get('contact_info', {})
                gt_value = contact_info.get('phone', [None])[0]
            elif field == 'email':
                contact_info = gt_data.get('contact_info', {})
                gt_value = contact_info.get('email', [None])[0]
            elif field == 'date_of_birth':
                contact_info = gt_data.get('contact_info', {})
                gt_value = contact_info.get('date_of_birth')
            elif field == 'name':
                gt_value = gt_data.get('candidate_name')
            else:
                continue
            
            if not gt_value:
                continue
            
            total += 1
            
            # Extract using this strategy
            prompt = prompt_fn(text)
            extracted = self._call_ai(prompt)
            
            # Compare
            if self._is_match(field, extracted, gt_value):
                correct += 1
        
        if total == 0:
            return 0.0
        
        return (correct / total) * 100
    
    def _call_ai(self, prompt: str) -> Optional[str]:
        """
        Call AI model with prompt.
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a precise data extraction assistant. Extract only the requested information. Return only the extracted value, nothing else.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.1,
                    'num_predict': 200
                }
            )
            
            result = response['message']['content'].strip()
            
            # Clean common AI additions
            result = re.sub(r'^(Answer:|Result:|Output:)\s*', '', result, flags=re.IGNORECASE)
            result = result.strip('"\'')
            
            return result if result else None
            
        except Exception as e:
            print(f"AI call error: {e}")
            return None
    
    def _is_match(self, field: str, extracted: Optional[str], ground_truth: str) -> bool:
        """
        Check if extracted value matches ground truth.
        """
        if not extracted or not ground_truth:
            return False
        
        if field == 'phone':
            # Extract digits only for comparison
            ex_digits = re.sub(r'\D', '', extracted)
            gt_digits = re.sub(r'\D', '', ground_truth)
            
            # Match if digits are the same
            return ex_digits == gt_digits or ex_digits in gt_digits or gt_digits in ex_digits
        
        elif field == 'email':
            return extracted.lower().strip() == ground_truth.lower().strip()
        
        elif field == 'name':
            # Fuzzy match for names
            similarity = fuzz.ratio(extracted.lower(), ground_truth.lower())
            return similarity >= 85
        
        elif field == 'date_of_birth':
            # Parse both dates and compare
            ex_date = self._parse_date(extracted)
            gt_date = self._parse_date(ground_truth)
            
            return ex_date and gt_date and ex_date == gt_date
        
        return False
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date to YYYY-MM-DD for comparison."""
        if not date_str:
            return None
        
        # Already in YYYY-MM-DD
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # DD/MM/YYYY
        match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
        if match:
            day, month, year = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        
        # MM/DD/YYYY
        match = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', date_str)
        if match:
            month, day, year = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        
        # YYYY/MM/DD
        match = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', date_str)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        
        return None
    
    # PHONE EXTRACTION STRATEGIES
    
    def _phone_prompt_intl_priority(self, text: str) -> str:
        """Strategy 1: Prioritize international format."""
        text_sample = text[:2000]
        return f"""Extract the PRIMARY phone number from this resume.

PRIORITY ORDER:
1. International format starting with + (e.g., +91-xxx, +81-xxx)
2. If no international format, extract domestic format (090-xxx, 080-xxx)

Extract ONLY ONE phone number - the primary contact number.

Resume text:
{text_sample}

Return only the phone number, nothing else."""
    
    def _phone_prompt_all_phones(self, text: str) -> str:
        """Strategy 2: Find all phones, return first."""
        text_sample = text[:2000]
        return f"""Find ALL phone numbers in this resume and return the FIRST one you find.

Look for:
- International format: +81, +91, etc.
- Japanese format: 090-xxxx-xxxx, 080-xxxx-xxxx
- Any phone-like numbers near "Mobile:", "Phone:", "Tel:", "電話"

Resume text:
{text_sample}

Return only the first phone number found, nothing else."""
    
    def _phone_prompt_context_aware(self, text: str) -> str:
        """Strategy 3: Context-aware extraction."""
        text_sample = text[:2000]
        return f"""Extract the person's PRIMARY contact phone number.

Look near these indicators:
- "Mobile:" or "Phone:" or "Tel:" or "Contact:" 
- "電話" or "携帯電話"
- Near the person's email address
- At the top of the resume

Prefer international format (+81, +91) if present.

Resume text:
{text_sample}

Return only one phone number, nothing else."""
    
    def _phone_prompt_format_flexible(self, text: str) -> str:
        """Strategy 4: Format flexible with normalization."""
        text_sample = text[:2000]
        return f"""Extract the primary phone number from this resume.

The number may be in various formats:
- +81 90 1234 5678
- +81-90-1234-5678
- 090-1234-5678
- (090) 1234-5678

Return the number in international format if it starts with +, otherwise return as-is.

Resume text:
{text_sample}

Return only the phone number, nothing else."""
    
    # NAME EXTRACTION STRATEGIES
    
    def _name_prompt_header_focus(self, text: str) -> str:
        """Strategy 1: Focus on headers."""
        text_sample = text[:1500]
        return f"""Extract the person's full name from this resume.

Look for name after these headers:
- "Name:" or "Full Name:" or "Candidate Name:"
- "氏名" or "名前" or "お名前"
- Or look at the VERY FIRST line (often the name)

Return the complete name (first and last name).

Resume text:
{text_sample}

Return only the name, nothing else."""
    
    def _name_prompt_position_aware(self, text: str) -> str:
        """Strategy 2: Position-based extraction."""
        lines = text.split('\n')[:20]
        text_sample = '\n'.join(lines)
        return f"""Extract the person's name from this resume.

The name is usually in the first 5 lines of the resume.
Look for:
- A line with 2-4 words that looks like a person's name
- Capitalized words that aren't section headers
- Names in format: "FirstName LastName" or "LASTNAME FirstName"

Avoid extracting: Resume, CV, Profile, Objective (these are NOT names)

Resume text (first 20 lines):
{text_sample}

Return only the person's name, nothing else."""
    
    def _name_prompt_format_detection(self, text: str) -> str:
        """Strategy 3: Format detection."""
        text_sample = text[:1500]
        return f"""Extract the person's full name.

Name formats to look for:
- English: "John Smith" or "SMITH John"
- Japanese: "田中太郎" or with furigana
- Mixed: "LASTNAME Firstname"

The name is usually at the top of the resume or after "Name:"

Resume text:
{text_sample}

Return only the name, nothing else."""
    
    def _name_prompt_validation_strict(self, text: str) -> str:
        """Strategy 4: Strict validation."""
        text_sample = text[:1500]
        return f"""Extract the person's name from this resume.

IMPORTANT: The name should be:
- 2-4 words long
- NOT a job title (Engineer, Manager, etc.)
- NOT a section header (Resume, Profile, Objective)
- A real person's name

Look at the top of the resume or after "Name:" label.

Resume text:
{text_sample}

Return only the person's name, nothing else."""
    
    # EMAIL EXTRACTION STRATEGIES
    
    def _email_prompt_simple(self, text: str) -> str:
        """Strategy 1: Simple email extraction."""
        text_sample = text[:1500]
        return f"""Extract the person's email address.

Look for text in format: username@domain.com

Resume text:
{text_sample}

Return only the email address, nothing else."""
    
    def _email_prompt_validation(self, text: str) -> str:
        """Strategy 2: With validation."""
        text_sample = text[:1500]
        return f"""Extract the person's PRIMARY email address.

Must contain:
- @ symbol
- domain name (gmail.com, yahoo.com, company.com, etc.)
- valid email format

If multiple emails present, return the first one.

Resume text:
{text_sample}

Return only the email address, nothing else."""
    
    def _email_prompt_multiple(self, text: str) -> str:
        """Strategy 3: Handle multiple emails."""
        text_sample = text[:1500]
        return f"""Find the person's PRIMARY email address.

If there are multiple emails, prioritize:
1. Email near "Email:" or "E-mail:" or "メール"
2. Personal email (gmail, yahoo, outlook)
3. The first email found

Resume text:
{text_sample}

Return only ONE email address, nothing else."""
    
    # DATE OF BIRTH STRATEGIES
    
    def _dob_prompt_format_aware(self, text: str) -> str:
        """Strategy 1: Format aware."""
        text_sample = text[:2000]
        return f"""Extract the person's date of birth.

Look for dates near:
- "Date of Birth:" or "DOB:" or "Born:"
- "生年月日"

The date may be in formats:
- DD/MM/YYYY (22/01/1997)
- YYYY-MM-DD (1997-01-22)
- Japanese format with 年月日

Return in YYYY-MM-DD format.

Resume text:
{text_sample}

Return only the date in YYYY-MM-DD format, nothing else."""
    
    def _dob_prompt_era_conversion(self, text: str) -> str:
        """Strategy 2: Handle Japanese era names."""
        text_sample = text[:2000]
        return f"""Extract the person's date of birth and convert to YYYY-MM-DD format.

Handle Japanese era names:
- 令和5年 = 2023年
- 平成31年 = 2019年
- 昭和64年 = 1989年

Look near "生年月日" or "Date of Birth" or "DOB"

Resume text:
{text_sample}

Return only the date in YYYY-MM-DD format, nothing else."""
    
    def _dob_prompt_context_search(self, text: str) -> str:
        """Strategy 3: Context-based search."""
        text_sample = text[:2000]
        return f"""Find the person's date of birth.

Search near:
- Email and phone number (personal information section)
- "生年月日" or "DOB" or "Date of Birth" headers
- Top section of resume

Convert any format to YYYY-MM-DD.

Resume text:
{text_sample}

Return only the date in YYYY-MM-DD format, nothing else."""
    
    def run_full_optimization(self, max_samples: int = 50):
        """
        Run optimization for all fields.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE PROMPT OPTIMIZATION")
        print("="*70)
        print(f"Testing on {max_samples} samples")
        
        # Test each field
        self.test_phone_extraction_strategies(max_samples)
        self.test_name_extraction_strategies(max_samples)
        self.test_email_extraction_strategies(max_samples)
        self.test_dob_extraction_strategies(max_samples)
        
        # Generate report
        self.generate_optimization_report()
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        print("\n" + "="*70)
        print("OPTIMIZATION REPORT")
        print("="*70)
        
        for field, data in self.results.items():
            print(f"\n{field.upper()}:")
            print(f"  Best strategy: {data['best']}")
            print(f"  Best accuracy: {data['best_accuracy']:.2f}%")
            print(f"  All strategies:")
            for strategy, acc in data['strategies'].items():
                indicator = " <-- BEST" if strategy == data['best'] else ""
                print(f"    {strategy}: {acc:.2f}%{indicator}")
        
        # Save to JSON
        output_file = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Generate code for best prompts
        self.generate_optimized_code()
    
    def generate_optimized_code(self):
        """Generate code file with optimized prompts."""
        code = """# OPTIMIZED PROMPTS
# Generated by prompt_optimizer.py
# Use these prompts in your extraction system

def get_optimized_prompt(field: str, text: str) -> str:
    \"\"\"
    Get the optimized prompt for a specific field.
    \"\"\"
    text_sample = text[:2000]  # Limit text length
    
"""
        
        for field, data in self.results.items():
            best_strategy = data['best']
            code += f"    if field == '{field}':\n"
            code += f"        # Best strategy: {best_strategy} ({data['best_accuracy']:.2f}% accuracy)\n"
            code += f"        # TODO: Copy the prompt from {best_strategy} method\n"
            code += f"        pass\n\n"
        
        code += """    return ""
"""
        
        with open("optimized_prompts.py", 'w', encoding='utf-8') as f:
            f.write(code)
        
        print("\nGenerated optimized_prompts.py template")


def main():
    """Run prompt optimization."""
    import config
    
    print("="*70)
    print("PROMPT OPTIMIZATION SYSTEM")
    print("="*70)
    
    # Initialize optimizer
    optimizer = PromptOptimizer(
        model_name=config.MODEL_NAME,
        dataset_path="dataset.jsonl"
    )
    
    # Ask user
    print(f"\nTotal samples available: {len(optimizer.labeled_data)}")
    
    while True:
        choice = input("\nHow many samples to test each strategy? (default: 50): ").strip()
        if not choice:
            max_samples = 50
            break
        try:
            max_samples = int(choice)
            if max_samples <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Run optimization
    optimizer.run_full_optimization(max_samples)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the optimization_results_*.json file")
    print("2. Implement the best strategies in your extraction code")
    print("3. Re-run evaluator.py to measure improvement")


if __name__ == "__main__":
    main()