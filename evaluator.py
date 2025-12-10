"""
Evaluation Framework for Resume Extraction
Measures accuracy against ground truth labeled data
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
from fuzzywuzzy import fuzz
import re
from collections import defaultdict

class ExtractionEvaluator:
    """
    Evaluates extraction accuracy against ground truth labels.
    Provides detailed metrics and error analysis.
    """
    
    def __init__(self, dataset_path: str = "dataset.jsonl"):
        """
        Initialize evaluator with path to labeled dataset.
        
        Args:
            dataset_path: Path to JSONL file with format:
                {"input": "resume text", "output": {"name": "...", "email": "...", ...}}
        """
        self.dataset_path = dataset_path
        self.labeled_data = []
        self.results = []
        
        # Load labeled dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the labeled dataset from JSONL file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        self.labeled_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {e}")
        
        print(f"Loaded {len(self.labeled_data)} labeled examples")
    
    def evaluate_extraction(self, extractor, max_samples: Optional[int] = None):
        """
        Run extraction on labeled data and calculate accuracy.
        
        Args:
            extractor: Your UltimateResumeExtractor instance
            max_samples: Optional limit on number of samples to evaluate
        
        Returns:
            Dict with evaluation metrics
        """
        samples_to_evaluate = self.labeled_data[:max_samples] if max_samples else self.labeled_data
        
        print(f"\nEvaluating {len(samples_to_evaluate)} samples...")
        print("=" * 60)
        
        self.results = []
        
        for idx, sample in enumerate(samples_to_evaluate, 1):
            resume_text = sample['input']
            ground_truth = sample['output']
            
            # Run extraction
            extracted_data = self._extract_from_text(extractor, resume_text)
            
            # Compare against ground truth
            comparison = self._compare_extraction(extracted_data, ground_truth)
            
            # Store result
            result = {
                'sample_id': idx,
                'ground_truth': ground_truth,
                'extracted': extracted_data,
                'comparison': comparison
            }
            self.results.append(result)
            
            # Progress indicator
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(samples_to_evaluate)} samples...")
        
        # Calculate overall metrics
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _extract_from_text(self, extractor, text: str) -> Dict:
        """
        Extract data from resume text using the extractor.
        
        Args:
            extractor: UltimateResumeExtractor instance
            text: Resume text
        
        Returns:
            Extracted data dictionary
        """
        # Use extractor's internal method
        extracted, ai_used = extractor._extract_data_from_text(text)
        
        # Normalize field names to match ground truth format
        normalized = {
            'name': extracted.get('Name'),
            'email': extracted.get('Email'),
            'phone': extracted.get('Phone'),
            'date_of_birth': extracted.get('Date_of_Birth')
        }
        
        return normalized
    
    def _compare_extraction(self, extracted: Dict, ground_truth) -> Dict:
        """
        Compare extracted data against ground truth.
        
        Returns:
            Dictionary with comparison results for each field
        """
        comparison = {}
        
        # Handle list format: output is [{"resume_id": ..., "contact_info": {...}, ...}]
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            gt_dict = ground_truth[0]
        else:
            gt_dict = ground_truth
        
        # Extract ground truth values from the nested structure
        gt_name = gt_dict.get('candidate_name')
        
        # Email and phone are in nested contact_info
        contact_info = gt_dict.get('contact_info', {})
        gt_email = contact_info.get('email', [None])[0] if contact_info.get('email') else None
        gt_phone = contact_info.get('phone', [None])[0] if contact_info.get('phone') else None
        gt_dob = contact_info.get('date_of_birth')
        
        # Define fields to compare
        fields = {
            'name': gt_name,
            'email': gt_email,
            'phone': gt_phone,
            'date_of_birth': gt_dob
        }
        
        for field, gt_value in fields.items():
            ex_value = extracted.get(field)
            
            # Determine match type
            if gt_value is None:
                match_type = 'no_ground_truth'
            elif ex_value is None:
                match_type = 'missing'
            else:
                match_type = self._determine_match_quality(field, ex_value, gt_value)
            
            comparison[field] = {
                'ground_truth': gt_value,
                'extracted': ex_value,
                'match_type': match_type,
                'correct': match_type in ['exact', 'fuzzy']
            }
        
        return comparison
    
    def _determine_match_quality(self, field: str, extracted: str, ground_truth: str) -> str:
        """
        Determine quality of match between extracted and ground truth values.
        
        Returns:
            'exact', 'fuzzy', 'partial', or 'wrong'
        """
        if not extracted or not ground_truth:
            return 'missing'
        
        # Normalize for comparison
        extracted_norm = self._normalize_value(field, extracted)
        ground_truth_norm = self._normalize_value(field, ground_truth)
        
        # Exact match
        if extracted_norm == ground_truth_norm:
            return 'exact'
        
        # Field-specific fuzzy matching
        if field == 'name':
            # Use fuzzy string matching for names
            similarity = fuzz.ratio(extracted_norm.lower(), ground_truth_norm.lower())
            if similarity >= 90:
                return 'fuzzy'
            elif similarity >= 70:
                return 'partial'
        
        elif field == 'email':
            # Email must be exact match
            if extracted_norm.lower() == ground_truth_norm.lower():
                return 'exact'
        
        elif field == 'phone':
            # Extract digits only for phone comparison
            ex_digits = re.sub(r'\D', '', extracted)
            gt_digits = re.sub(r'\D', '', ground_truth)
            
            if ex_digits == gt_digits:
                return 'fuzzy'  # Same digits, different format
            elif gt_digits and ex_digits in gt_digits:
                return 'partial'  # Partial phone number
        
        elif field == 'date_of_birth':
            # Check if dates match
            ex_date = self._parse_date(extracted)
            gt_date = self._parse_date(ground_truth)
            
            if ex_date and gt_date and ex_date == gt_date:
                return 'fuzzy'  # Same date, different format
        
        return 'wrong'
    
    def _normalize_value(self, field: str, value: str) -> str:
        """Normalize value for comparison."""
        if not value:
            return ""
        
        # Remove extra whitespace
        normalized = ' '.join(value.split())
        
        # Field-specific normalization
        if field == 'email':
            normalized = normalized.lower().strip()
        elif field == 'phone':
            # Keep as-is for now, digit extraction happens in match quality check
            pass
        elif field == 'name':
            # Remove honorifics, normalize spacing
            normalized = re.sub(r'\b(Mr|Ms|Mrs|Dr|様|さん|氏)\.?\s*', '', normalized, flags=re.IGNORECASE)
            normalized = ' '.join(normalized.split())
        
        return normalized.strip()
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date to YYYY-MM-DD format for comparison."""
        if not date_str:
            return None
        
        # If already in YYYY-MM-DD format
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str
        
        # Try to extract date parts
        patterns = [
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if len(match.group(1)) == 4:  # YYYY-MM-DD
                        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
                    else:  # MM-DD-YYYY or DD-MM-YYYY
                        return f"{match.group(3)}-{int(match.group(1)):02d}-{int(match.group(2)):02d}"
                except:
                    pass
        
        return None
    
    def _calculate_metrics(self) -> Dict:
        """Calculate overall accuracy metrics."""
        metrics = {
            'total_samples': len(self.results),
            'field_metrics': {},
            'overall_accuracy': 0.0
        }
        
        fields = ['name', 'email', 'phone', 'date_of_birth']
        
        for field in fields:
            field_stats = {
                'total': 0,
                'correct': 0,
                'exact_match': 0,
                'fuzzy_match': 0,
                'partial_match': 0,
                'wrong': 0,
                'missing': 0,
                'accuracy': 0.0
            }
            
            for result in self.results:
                comp = result['comparison'].get(field, {})
                match_type = comp.get('match_type')
                
                if match_type == 'no_ground_truth':
                    continue  # Skip if no ground truth for this field
                
                field_stats['total'] += 1
                
                if match_type == 'exact':
                    field_stats['exact_match'] += 1
                    field_stats['correct'] += 1
                elif match_type == 'fuzzy':
                    field_stats['fuzzy_match'] += 1
                    field_stats['correct'] += 1
                elif match_type == 'partial':
                    field_stats['partial_match'] += 1
                elif match_type == 'missing':
                    field_stats['missing'] += 1
                else:  # wrong
                    field_stats['wrong'] += 1
            
            # Calculate accuracy
            if field_stats['total'] > 0:
                field_stats['accuracy'] = (field_stats['correct'] / field_stats['total']) * 100
            
            metrics['field_metrics'][field] = field_stats
        
        # Calculate overall accuracy
        total_fields = sum(m['total'] for m in metrics['field_metrics'].values())
        total_correct = sum(m['correct'] for m in metrics['field_metrics'].values())
        
        if total_fields > 0:
            metrics['overall_accuracy'] = (total_correct / total_fields) * 100
        
        return metrics
    
    def print_metrics_report(self, metrics: Dict):
        """Print a formatted metrics report."""
        print("\n" + "=" * 70)
        print("EXTRACTION ACCURACY REPORT")
        print("=" * 70)
        
        print(f"\nTotal Samples Evaluated: {metrics['total_samples']}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        
        print("\n" + "-" * 70)
        print("FIELD-LEVEL ACCURACY")
        print("-" * 70)
        
        for field, stats in metrics['field_metrics'].items():
            # Emoji indicators
            if stats['accuracy'] >= 80:
                indicator = "✅"
            elif stats['accuracy'] >= 60:
                indicator = "⚠️"
            else:
                indicator = "🚨"
            
            print(f"\n{indicator} {field.upper()}:")
            print(f"  Total with ground truth: {stats['total']}")
            print(f"  Accuracy: {stats['accuracy']:.2f}%")
            print(f"  Breakdown:")
            print(f"    Exact matches: {stats['exact_match']}")
            print(f"    Fuzzy matches: {stats['fuzzy_match']}")
            print(f"    Partial matches: {stats['partial_match']}")
            print(f"    Wrong: {stats['wrong']}")
            print(f"    Missing: {stats['missing']}")
            
            # Quick diagnosis
            if stats['total'] > 0:
                missing_rate = (stats['missing'] / stats['total']) * 100
                wrong_rate = (stats['wrong'] / stats['total']) * 100
                
                if missing_rate > 50:
                    print(f"  ⚠️ Primary Issue: Not finding field ({missing_rate:.0f}% missing)")
                elif wrong_rate > 30:
                    print(f"  ⚠️ Primary Issue: Extracting wrong values ({wrong_rate:.0f}% wrong)")
                elif stats['partial_match'] > stats['correct']:
                    print(f"  ⚠️ Primary Issue: Partial extractions (incomplete data)")

    
    def generate_error_analysis(self, output_file: str = "error_analysis.csv"):
        """
        Generate detailed error analysis report.
        Shows which samples failed and why.
        """
        errors = []
        
        for result in self.results:
            sample_id = result['sample_id']
            
            for field, comp in result['comparison'].items():
                if not comp['correct'] and comp['match_type'] != 'no_ground_truth':
                    errors.append({
                        'sample_id': sample_id,
                        'field': field,
                        'ground_truth': comp['ground_truth'],
                        'extracted': comp['extracted'],
                        'match_type': comp['match_type']
                    })
        
        if errors:
            df = pd.DataFrame(errors)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nError analysis saved to: {output_file}")
            print(f"Total errors: {len(errors)}")
            
            # Show error distribution
            print("\nError Distribution by Field:")
            error_counts = df['field'].value_counts()
            for field, count in error_counts.items():
                print(f"  {field}: {count} errors")
            
            print("\nError Types:")
            type_counts = df['match_type'].value_counts()
            for error_type, count in type_counts.items():
                print(f"  {error_type}: {count}")
        else:
            print("\nNo errors found - 100% accuracy!")
    
    def save_detailed_results(self, output_file: str = "evaluation_results.json"):
        """Save detailed evaluation results to JSON."""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.results),
            'results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")


def main():
    """
    Main evaluation function.
    Run this to evaluate your current extraction system.
    """
    print("=" * 70)
    print("RESUME EXTRACTION EVALUATOR")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = ExtractionEvaluator("dataset.jsonl")
    
    # Ask user how many samples to evaluate
    print(f"\nTotal labeled samples available: {len(evaluator.labeled_data)}")
    
    while True:
        choice = input("\nHow many samples to evaluate? (Enter for all, or number): ").strip()
        if not choice:
            max_samples = None
            break
        try:
            max_samples = int(choice)
            if max_samples <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Import your extractor
    from main import UltimateResumeExtractor
    import config
    
    # Initialize extractor
    print(f"\nInitializing extractor with model: {config.MODEL_NAME}")
    extractor = UltimateResumeExtractor(config.MODEL_NAME)
    
    # Run evaluation
    metrics = evaluator.evaluate_extraction(extractor, max_samples)
    
    # Print report
    evaluator.print_metrics_report(metrics)
    
    # Generate error analysis
    evaluator.generate_error_analysis()
    
    # Save detailed results
    evaluator.save_detailed_results()
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    # Provide recommendations based on results
    print("\nRECOMMENDATIONS:")
    
    overall_acc = metrics['overall_accuracy']
    
    if overall_acc < 50:
        print("  Current accuracy is low (<50%).")
        print("  Recommend: Focus on prompt optimization (Phase 2)")
        print("  Consider switching to Qwen2.5:14b-instruct model")
    elif overall_acc < 70:
        print("  Moderate accuracy (50-70%).")
        print("  Recommend: Prompt optimization should help significantly")
    elif overall_acc < 85:
        print("  Good accuracy (70-85%).")
        print("  Recommend: Fine-tuning will push you to >90%")
    else:
        print("  Excellent accuracy (>85%)!")
        print("  Recommend: Fine-tuning for edge cases only")
    
    # Field-specific recommendations
    print("\nFIELD-SPECIFIC ISSUES:")
    for field, stats in metrics['field_metrics'].items():
        if stats['accuracy'] < 70:
            print(f"  {field}: Low accuracy ({stats['accuracy']:.1f}%) - needs attention")
            if stats['missing'] > stats['wrong']:
                print(f"    Problem: Not finding the field ({stats['missing']} missing)")
            else:
                print(f"    Problem: Extracting wrong values ({stats['wrong']} wrong)")


if __name__ == "__main__":
    main()