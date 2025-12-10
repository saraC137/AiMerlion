import json
import os
import datetime
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
from fuzzywuzzy import fuzz
import jsonlines
import unicodedata

def display_menu() -> None:
    """✨ Display the most GORGEOUS menu ever! ✨"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "   ✨💅 FAIRY CODEMOTHER'S RESUME EXTRACTOR DELUXE 💅✨".center(78) + "║")
    print("║" + "              ~ Where Every Resume Gets Its Glow-Up ~".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╠" + "═"*78 + "╣")
    print("║" + " "*78 + "║")
    print("║" + "   🎭 Choose Your Adventure, Darling:".center(78) + "║")
    print("║" + " "*78 + "║")
    print("║" + "     [1] 💄 Fresh Start - Process ALL resumes".ljust(78) + "║")
    print("║" + "     [2] 👑 Continue the Show - Resume from last checkpoint".ljust(78) + "║")
    print("║" + "     [3] 🎯 Selective Processing - Choose specific folders".ljust(78) + "║")
    print("║" + "     [4] 🔎 Test mode - Testing the stage before the show!".ljust(78) + "║")
    print("║" + "     [5] 📭 Find Candidates with Missing Resumes".ljust(78) + "║")
    print("║" + "     [6] 📜 List down candidates with resumes and their resumes language".ljust(78) + "║")
    print("║" + "     [7] 💋 Exit - The show must end (for now!)".ljust(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")

def save_checkpoint(processed_files: List[str], checkpoint_file: str) -> None:
    """💾 Save our progress, honey!"""
    checkpoint_data = {
        "processed_files": processed_files,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_processed": len(processed_files)
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint(checkpoint_file: str) -> List[str]:
    """📂 Load our previous work!"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data.get("processed_files", [])
            except json.JSONDecodeError:
                return []
    return []

def print_batch_table(batch_num: int, total_batches: int, batch_files: List[str], total_files: int, batch_size: int) -> None:
    """Prints a dazzling ASCII art table for the current batch."""
    print("\n" + "═" * 80)
    print(f"║ 🎭 Act {batch_num} of {total_batches} 🎭".center(80))
    print("╠" + "═" * 78 + "╣")
    start_file_num = (batch_num - 1) * batch_size + 1
    end_file_num = min(batch_num * batch_size, total_files)
    print(f"║ Processing {len(batch_files)} resumes (Files {start_file_num}-{end_file_num} of {total_files})".center(80))
    print("╚" + "═" * 78 + "╝")

def select_folders_to_process(resume_folder: str) -> List[str]:
    """💅 Allow the user to select specific FOLDERS to process."""
    print("\n" + "─"*60)
    print("🎯 SELECTIVE PROCESSING - Choose Your Stars! 🎯".center(60))
    print("─"*60)

    try:
        subfolders = sorted([f.name for f in os.scandir(resume_folder) if f.is_dir()])
    except FileNotFoundError:
        print(f"😱 Oh no, darling! The folder '{resume_folder}' wasn't found!")
        return []

    if not subfolders:
        print("🤷‍♀️ There are no folders to choose from in your resume directory.")
        return []

    print("Which candidate folders would you like to process, honey?")
    for i, folder_name in enumerate(subfolders):
        print(f"  [{i+1}] {folder_name}")

    selected_folders = []
    while True:
        choice_input = input("\nEnter the numbers (e.g., 1, 3, 4), or 'all' to process everything: ").strip().lower()
        if not choice_input:
            print("You have to choose something, sweetie! Try again.")
            continue

        if choice_input == 'all':
            selected_folders = subfolders
            break

        try:
            selected_indices = [int(i.strip()) - 1 for i in choice_input.split(',')]
            if any(i < 0 or i >= len(subfolders) for i in selected_indices):
                print("😱 One of those numbers is out of range, darling! Look at the list again.")
                continue
            selected_folders = [subfolders[i] for i in selected_indices]
            break
        except ValueError:
            print("💔 Numbers and commas only, honey! Or just type 'all'.")

    print("\n👑 Fabulous choices! Processing the following folders:")
    # Return the FULL PATHS to the selected folders
    selected_folder_paths = [os.path.join(resume_folder, folder) for folder in selected_folders]
    for folder_path in selected_folder_paths:
        print(f"  - {os.path.basename(folder_path)}")

    return selected_folder_paths

def standardize_phone_number(phone: str) -> Optional[str]:
    """
    📱 Standardizes phone numbers to common English formats.
    Supports: US, UK, and international formats.
    """
    if not phone:
        return None
    
    # Clean the input
    cleaned = re.sub(r'[^\d+\-\(\)\s]', '', phone).strip()
    
    # Extract only digits
    digits = re.sub(r'\D', '', cleaned)
    
    # US/Canada format (10 digits)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    
    # US/Canada with country code (11 digits starting with 1)
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    
    # UK format (10-11 digits)
    elif len(digits) in [10, 11] and cleaned.startswith(('+44', '44')):
        if digits.startswith('44'):
            digits = digits[2:]
        return f"+44 {digits[:4]} {digits[4:7]} {digits[7:]}"
    
    # International format with + (keep as-is but format nicely)
    elif cleaned.startswith('+') and len(digits) >= 10:
        return cleaned
    
    # If we can't format it, return the cleaned version
    return cleaned if len(digits) >= 7 else None

def standardize_date(date_str: str) -> Optional[str]:
    """
    🎂 Standardizes date strings to YYYY-MM-DD format.
    Handles common English date formats only.
    """
    if not date_str:
        return None
    
    import datetime
    
    # Already in correct format?
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # Common English date formats
    date_formats = [
        '%Y-%m-%d',         # 2023-05-23
        '%Y/%m/%d',         # 2023/05/23
        '%m/%d/%Y',         # 05/23/2023 (US)
        '%d/%m/%Y',         # 23/05/2023 (UK)
        '%B %d, %Y',        # May 23, 2023
        '%b %d, %Y',        # May 23, 2023
        '%d %B %Y',         # 23 May 2023
        '%d %b %Y',         # 23 May 2023
        '%m-%d-%Y',         # 05-23-2023
        '%d-%m-%Y',         # 23-05-2023
        '%Y%m%d',           # 20230523
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.datetime.strptime(date_str.strip(), fmt)
            
            # Validate reasonable age (18-70)
            age = (datetime.datetime.now() - dt).days / 365.25
            if 18 <= age <= 70:
                return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # Try to extract date with regex if format parsing fails
    # Pattern: look for YYYY-MM-DD or MM/DD/YYYY variations
    date_patterns = [
        (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 'ymd'),
        (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'mdy'),
    ]
    
    for pattern, date_format in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                if date_format == 'ymd':
                    year, month, day = map(int, match.groups())
                else:  # mdy
                    month, day, year = map(int, match.groups())
                
                dt = datetime.datetime(year, month, day)
                age = (datetime.datetime.now() - dt).days / 365.25
                
                if 18 <= age <= 70:
                    return dt.strftime('%Y-%m-%d')
            except (ValueError, OverflowError):
                continue
    
    return None

def detect_folder_structure(resume_folder: str) -> str:
    """
    🕵️‍♀️ Detect if resumes are organized in subfolders or loose files
    Returns: 'folders' | 'files' | 'mixed'
    """
    has_subfolders = False
    has_loose_files = False
    
    for item in os.scandir(resume_folder):
        if item.is_dir() and not item.name.startswith('.'):
            has_subfolders = True
        elif item.is_file() and item.name.lower().endswith(('.pdf', '.docx')):
            has_loose_files = True
    
    if has_subfolders and has_loose_files:
        return 'mixed'
    elif has_subfolders:
        return 'folders'
    else:
        return 'files'

def select_folders_to_process_enhanced(all_folders: List[str]) -> List[str]:
    """
    💅 Allow the user to select specific candidate folders with better display.
    """
    print("\n" + "─"*60)
    print("🎯 SELECTIVE PROCESSING - Choose Your Stars! 🎯".center(60))
    print("─"*60)

    if not all_folders:
        print("🤷‍♀️ There are no candidate folders to choose from!")
        return []

    print("Which candidate folders would you like to process, honey?")
    
    # Group by parent folder for better display
    folder_groups = {}
    for folder_path in all_folders:
        parent = os.path.basename(os.path.dirname(folder_path))
        if parent not in folder_groups:
            folder_groups[parent] = []
        folder_groups[parent].append(folder_path)
    
    # Display with grouping
    idx = 0
    folder_mapping = {}
    for parent, folders in folder_groups.items():
        print(f"\n📁 {parent}:")
        for folder_path in folders:
            idx += 1
            folder_name = os.path.basename(folder_path)
            folder_mapping[idx] = folder_path
            print(f"  [{idx}] {folder_name}")

    selected_folders = []
    while True:
        choice_input = input("\nEnter the numbers (e.g., 1, 3, 4), or 'all' to process everything: ").strip().lower()
        if not choice_input:
            print("You have to choose something, sweetie! Try again.")
            continue

        if choice_input == 'all':
            selected_folders = all_folders
            break

        try:
            selected_indices = [int(i.strip()) for i in choice_input.split(',')]
            if any(i < 1 or i > len(folder_mapping) for i in selected_indices):
                print("😱 One of those numbers is out of range, darling! Look at the list again.")
                continue
            selected_folders = [folder_mapping[i] for i in selected_indices]
            break
        except ValueError:
            print("💔 Numbers and commas only, honey! Or just type 'all'.")

    print("\n👑 Fabulous choices! Processing the following folders:")
    for folder_path in selected_folders:
        print(f"  - {os.path.basename(folder_path)}")

    return selected_folders

# 🌟 STRATEGY 4 FUNCTIONS START HERE! 🌟

class FeedbackLoopSystem:
    """
    💅 The ULTIMATE Learning System without Fine-tuning!
    This fabulous system learns from EVERY extraction!
    """
    
    def __init__(self, feedback_dir: str = "feedback"):
        self.feedback_dir = feedback_dir
        os.makedirs(feedback_dir, exist_ok=True)
        
        self.feedback_file = os.path.join(feedback_dir, "extraction_feedback.jsonl")
        self.corrections_file = os.path.join(feedback_dir, "pattern_corrections.json")
        self.stats_file = os.path.join(feedback_dir, "learning_stats.json")
        self.patterns_file = os.path.join(feedback_dir, "learned_patterns.json")
        
        # Load existing data
        self.load_learned_patterns()
        self.load_stats()
    
    def save_extraction_result(self, filename: str, extracted_data: Dict, 
                              confidence_scores: Optional[Dict] = None, ai_assisted: bool = False):
        """
        💾 Save EVERY extraction result for learning!
        """
        entry = {
            "filename": filename,
            "timestamp": datetime.datetime.now().isoformat(),
            "extracted": extracted_data,
            "confidence": confidence_scores or {},
            "ai_assisted": ai_assisted,
            "version": "1.0"  # Track which version extracted this
        }
        
        # Append to JSONL file (more efficient for large datasets)
        with jsonlines.open(self.feedback_file, mode='a') as writer:
            writer.write(entry)
    
    def save_correction(self, filename: str, field: str, 
                       original_value: str, corrected_value: str, 
                       context: str = ""):
        """
        🔧 Save manual corrections for pattern learning!
        """
        corrections = self.load_corrections()
        
        correction_entry = {
            "filename": filename,
            "field": field,
            "original": original_value,
            "corrected": corrected_value,
            "context": context[:200],  # Save some context
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if field not in corrections:
            corrections[field] = []
        
        corrections[field].append(correction_entry)
        
        # Save corrections
        with open(self.corrections_file, 'w', encoding='utf-8') as f:
            json.dump(corrections, f, indent=2, ensure_ascii=False)
        
        # Learn from this correction immediately!
        self.learn_from_correction(field, original_value, corrected_value, context)
    
    def learn_from_correction(self, field: str, original: str, corrected: str, context: str):
        """
        🧠 Learn patterns from corrections!
        """
        patterns = self.load_learned_patterns()
        
        if field not in patterns:
            patterns[field] = {
                "positive_patterns": [],  # Patterns that worked
                "negative_patterns": [],  # Patterns to avoid
                "transformations": []     # How to fix common mistakes
            }
        
        # Analyze the correction
        if original and not corrected:
            # False positive - add to negative patterns
            patterns[field]["negative_patterns"].append({
                "pattern": original,
                "context_hint": self._extract_context_hint(context, original),
                "occurrences": 1
            })
        elif corrected and original != corrected:
            # Transformation needed
            patterns[field]["transformations"].append({
                "from": original,
                "to": corrected,
                "similarity": fuzz.ratio(original, corrected),
                "context_hint": self._extract_context_hint(context, original)
            })
        
        self.save_learned_patterns(patterns)
    
    def _extract_context_hint(self, context: str, value: str) -> str:
        """
        🔍 Extract hints from context around the value
        """
        if not context or not value:
            return ""
        
        # Find value in context
        value_pos = context.lower().find(value.lower())
        if value_pos == -1:
            return ""
        
        # Get surrounding text (20 chars before and after)
        start = max(0, value_pos - 20)
        end = min(len(context), value_pos + len(value) + 20)
        
        return context[start:end].strip()
    
    def suggest_pattern_improvements(self, field: str, current_patterns: List[str]) -> List[str]:
        """
        💡 Suggest improved patterns based on learning!
        """
        patterns = self.load_learned_patterns()
        suggestions = []
        
        if field not in patterns:
            return suggestions
        
        # Analyze transformations to suggest better patterns
        transformations = patterns[field].get("transformations", [])
        
        # Group similar transformations
        transformation_groups = {}
        for trans in transformations:
            key = (trans["from"][:5], trans["to"][:5])  # Group by prefix
            if key not in transformation_groups:
                transformation_groups[key] = []
            transformation_groups[key].append(trans)
        
        # Generate suggestions
        for group, transforms in transformation_groups.items():
            if len(transforms) >= 3:  # Pattern appears frequently
                # This is a common mistake!
                suggestion = f"Common correction: '{transforms[0]['from']}' → '{transforms[0]['to']}'"
                suggestions.append(suggestion)
        
        return suggestions
    
    def get_field_accuracy_stats(self) -> Dict[str, Dict]:
        """
        📊 Get accuracy statistics for each field!
        """
        stats = {
            "name": {"total": 0, "correct": 0, "ai_assisted": 0},
            "email": {"total": 0, "correct": 0, "ai_assisted": 0},
            "phone": {"total": 0, "correct": 0, "ai_assisted": 0},
            "date_of_birth": {"total": 0, "correct": 0, "ai_assisted": 0}
        }
        
        corrections = self.load_corrections()
        
        # Count extractions
        with jsonlines.open(self.feedback_file) as reader:
            for entry in reader:
                for field in stats:
                    if entry["extracted"].get(field):
                        stats[field]["total"] += 1
                        if entry.get("ai_assisted"):
                            stats[field]["ai_assisted"] += 1
        
        # Count corrections (these were wrong)
        for field, field_corrections in corrections.items():
            if field in stats:
                stats[field]["correct"] = stats[field]["total"] - len(field_corrections)
        
        # Calculate accuracy
        for field in stats:
            total = stats[field]["total"]
            if total > 0:
                stats[field]["accuracy"] = int(stats[field]["correct"] / total * 100)
            else:
                stats[field]["accuracy"] = 0
        
        return stats
    
    def generate_improvement_report(self) -> str:
        """
        📈 Generate a FABULOUS improvement report!
        """
        stats = self.get_field_accuracy_stats()
        corrections = self.load_corrections()
        
        report = []
        report.append("✨ EXTRACTION IMPROVEMENT REPORT ✨")
        report.append("=" * 50)
        
        for field, field_stats in stats.items():
            report.append(f"\n💅 {field.upper()} FIELD:")
            report.append(f"   Total extractions: {field_stats['total']}")
            report.append(f"   Accuracy: {field_stats['accuracy']:.1f}%")
            report.append(f"   AI-assisted: {field_stats['ai_assisted']} ({field_stats['ai_assisted']/field_stats['total']*100:.1f}%)" if field_stats['total'] > 0 else "   AI-assisted: 0 (0%)")
            
            # Show common corrections
            if field in corrections and corrections[field]:
                report.append("   Common mistakes:")
                # Group by similarity
                mistakes = {}
                for corr in corrections[field][:10]:  # Top 10
                    mistake = corr['original']
                    if mistake not in mistakes:
                        mistakes[mistake] = 0
                    mistakes[mistake] += 1
                
                for mistake, count in sorted(mistakes.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report.append(f"      '{mistake}' ({count} times)")
        
        report.append("\n🎯 RECOMMENDATIONS:")
        
        # Recommendations based on accuracy
        worst_field = min(stats.items(), key=lambda x: x[1]['accuracy'])
        if worst_field[1]['accuracy'] < 70:
            report.append(f"   🚨 Focus on improving {worst_field[0]} extraction (only {worst_field[1]['accuracy']:.1f}% accurate)")
        
        if sum(s['ai_assisted'] for s in stats.values()) > sum(s['total'] for s in stats.values()) * 0.5:
            report.append("   🤖 High AI dependency detected - consider improving base regex patterns")
        
        return "\n".join(report)
    
    def export_training_data(self, output_file: str = "training_data.csv"):
        """
        📤 Export data for potential future fine-tuning!
        """
        data = []
        
        # Load all feedback
        with jsonlines.open(self.feedback_file) as reader:
            for entry in reader:
                # Combine with corrections if available
                corrected_data = entry["extracted"].copy()
                
                # Apply corrections
                corrections = self.load_corrections()
                for field, field_corrections in corrections.items():
                    for corr in field_corrections:
                        if corr["filename"] == entry["filename"]:
                            corrected_data[field] = corr["corrected"]
                
                data.append({
                    "filename": entry["filename"],
                    "name": corrected_data.get("name", ""),
                    "email": corrected_data.get("email", ""),
                    "phone": corrected_data.get("phone", ""),
                    "dob": corrected_data.get("date_of_birth", ""),
                    "ai_assisted": entry.get("ai_assisted", False)
                })
        
        # Export to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        return f"💾 Exported {len(data)} entries to {output_file}"
    
    def load_corrections(self) -> Dict:
        """Load existing corrections"""
        if os.path.exists(self.corrections_file):
            with open(self.corrections_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_learned_patterns(self) -> Dict:
        """Load learned patterns"""
        if os.path.exists(self.patterns_file):
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_learned_patterns(self, patterns: Dict):
        """Save learned patterns"""
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
    
    def load_stats(self) -> Dict:
        """Load statistics"""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def update_stats(self, stats_update: Dict):
        """Update and save statistics"""
        stats = self.load_stats()
        stats.update(stats_update)
        stats["last_updated"] = datetime.datetime.now().isoformat()
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

# 🌟 INTERACTIVE CORRECTION SYSTEM 🌟

class InteractiveCorrectionSystem:
    """
    💖 Interactive system for correcting extractions!
    Makes learning from mistakes FABULOUS!
    """
    
    def __init__(self, feedback_system: FeedbackLoopSystem):
        self.feedback_system = feedback_system
    
    def review_extraction(self, filename: str, extracted_data: Dict) -> Dict:
        """
        👀 Review and optionally correct extraction results
        """
        print("\n" + "="*60)
        print(f"📋 REVIEWING: {filename}")
        print("="*60)
        
        # Display extracted data
        print("\n🎭 Extracted Data:")
        for field, value in extracted_data.items():
            if field not in ['ai_assisted', 'extraction_method']:
                emoji = self._get_field_emoji(field)
                print(f"{emoji} {field}: {value or '[NOT FOUND]'}")
        
        # Ask if corrections needed
        print("\n💅 Does this look FABULOUS? (y/n/skip): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            return extracted_data
        elif response == 'skip':
            return extracted_data
        elif response == 'n':
            return self._correct_fields(filename, extracted_data)
        
        return extracted_data
    
    def _get_field_emoji(self, field: str) -> str:
        """Get emoji for field"""
        emojis = {
            "name": "👤",
            "name_japanese": "🎌",
            "email": "📧",
            "phone": "📱",
            "date_of_birth": "🎂",
            "id": "🆔"
        }
        return emojis.get(field, "📌")
    
    def _correct_fields(self, filename: str, extracted_data: Dict) -> Dict:
        """
        🔧 Correct individual fields
        """
        corrected_data = extracted_data.copy()
        
        for field in ['name', 'email', 'phone', 'date_of_birth']:
            if field not in extracted_data:
                continue
                
            current_value = extracted_data.get(field, "")
            print(f"\n{self._get_field_emoji(field)} {field}: {current_value or '[NOT FOUND]'}")
            print("Correct value (or press Enter to keep): ", end="")
            
            new_value = input().strip()
            
            if new_value and new_value != current_value:
                # Save the correction
                self.feedback_system.save_correction(
                    filename=filename,
                    field=field,
                    original_value=current_value or "",
                    corrected_value=new_value,
                    context=""  # Could add context if needed
                )
                corrected_data[field] = new_value
                print(f"✅ Updated {field}!")
        
        return corrected_data

# 🌟 PATTERN LEARNING SYSTEM 🌟

class PatternLearningSystem:
    """
    🧠 Learn and suggest new patterns based on corrections!
    """
    
    def __init__(self, feedback_system: FeedbackLoopSystem):
        self.feedback_system = feedback_system
    
    def analyze_failures(self, min_occurrences: int = 3) -> Dict[str, List[str]]:
        """
        🔍 Analyze common extraction failures
        """
        corrections = self.feedback_system.load_corrections()
        failure_patterns = {}
        
        for field, field_corrections in corrections.items():
            if field not in failure_patterns:
                failure_patterns[field] = []
            
            # Group similar corrections
            correction_groups = {}
            for corr in field_corrections:
                original = corr.get('original', '')
                corrected = corr.get('corrected', '')
                
                # Create a key for grouping
                if not original:  # Missed extraction
                    key = "MISSED_EXTRACTION"
                else:
                    # Group by pattern similarity
                    key = self._get_pattern_key(original)
                
                if key not in correction_groups:
                    correction_groups[key] = []
                correction_groups[key].append((original, corrected))
            
            # Find patterns that occur frequently
            for key, corrections_list in correction_groups.items():
                if len(corrections_list) >= min_occurrences:
                    failure_patterns[field].append({
                        "pattern_type": key,
                        "examples": corrections_list[:5],  # Top 5 examples
                        "occurrences": len(corrections_list),
                        "suggested_fix": self._suggest_pattern_fix(field, corrections_list)
                    })
        
        return failure_patterns
    
    def _get_pattern_key(self, text: str) -> str:
        """
        🗝️ Generate a key for grouping similar patterns
        """
        # Simple pattern detection
        if '@' in text and '.' in text:
            return "EMAIL_LIKE"
        elif re.match(r'.*\d{3,}.*\d{3,}', text):
            return "PHONE_LIKE"
        elif re.match(r'.*\d{4}.*\d{1,2}.*\d{1,2}', text):
            return "DATE_LIKE"
        elif len(text.split()) >= 2:
            return "MULTI_WORD"
        else:
            return "SINGLE_WORD"
    
    def _suggest_pattern_fix(self, field: str, corrections: List[Tuple[str, str]]) -> str:
        """
        💡 Suggest how to fix a common pattern
        """
        if field == "name":
            # Analyze name corrections
            if all(not orig for orig, _ in corrections):
                return "Add pattern to look for names after headers like 'Applicant:', 'Candidate:'"
            elif all(',' in orig for orig, _ in corrections):
                return "Handle names in 'LastName, FirstName' format"
        
        elif field == "phone":
            # Analyze phone corrections
            if all(len(re.sub(r'\D', '', orig)) < 10 for orig, _ in corrections if orig):
                return "Phone numbers might be split across lines - check for vertical formatting"
        
        elif field == "email":
            # Analyze email corrections
            if any('@' not in orig for orig, _ in corrections if orig):
                return "Email might be obfuscated (e.g., 'name at domain dot com')"
        
        return "Review examples and add specific patterns"
    
    def generate_regex_suggestions(self) -> Dict[str, List[str]]:
        """
        🎯 Generate new regex patterns based on successful corrections
        """
        patterns = self.feedback_system.load_learned_patterns()
        suggestions = {}
        
        for field, field_patterns in patterns.items():
            suggestions[field] = []
            
            # Analyze transformations
            transformations = field_patterns.get('transformations', [])
            
            # Group by transformation type
            for trans in transformations:
                if trans['similarity'] > 80:  # Minor differences
                    # Suggest fuzzy matching
                    suggestions[field].append(
                        f"Add fuzzy matching for '{trans['from']}' → '{trans['to']}'"
                    )
        
        return suggestions

# 🌟 PERFORMANCE MONITORING 🌟

class PerformanceMonitor:
    """
    📊 Monitor extraction performance over time!
    """
    
    def __init__(self, feedback_system: FeedbackLoopSystem):
        self.feedback_system = feedback_system
    
    def track_performance(self, batch_results: List[Dict]):
        """
        📈 Track performance metrics for a batch
        """
        stats = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_processed": len(batch_results),
            "successful_extractions": sum(1 for r in batch_results if r.get("Extraction_Status") != "Failed"),
            "ai_assisted": sum(1 for r in batch_results if r.get("AI_Assisted", False)),
            "field_success_rates": {}
        }
        
        # Calculate field success rates
        for field in ["Name", "Email", "Phone", "Date_of_Birth"]:
            success_count = sum(1 for r in batch_results if r.get(field))
            stats["field_success_rates"][field] = (success_count / len(batch_results) * 100) if batch_results else 0
        
        # Update stats
        self.feedback_system.update_stats(stats)
        
        return stats
    
    def generate_performance_report(self) -> str:
        """
        📊 Generate a performance report over time
        """
        stats = self.feedback_system.load_stats()
        
        if not stats:
            return "No performance data available yet!"
        
        report = []
        report.append("\n✨ PERFORMANCE REPORT ✨")
        report.append("="*50)
        
        # Latest stats
        if "field_success_rates" in stats:
            report.append("\n📊 LATEST FIELD SUCCESS RATES:")
            for field, rate in stats["field_success_rates"].items():
                emoji = "✅" if rate > 80 else "⚠️" if rate > 60 else "🚨"
                report.append(f"   {emoji} {field}: {rate:.1f}%")
        
        # AI usage
        if "ai_assisted" in stats and "total_processed" in stats:
            ai_percentage = (stats["ai_assisted"] / stats["total_processed"] * 100) if stats["total_processed"] > 0 else 0
            report.append(f"\n🤖 AI ASSISTANCE: {ai_percentage:.1f}% of extractions")
        
        return "\n".join(report)

# 🌟 FILE EXTRACTION HELPERS 🌟

def get_file_type(file_path: str) -> Optional[str]:
    """
    Determines the file type based on its extension.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    if extension == '.pdf':
        return 'pdf'
    elif extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        return 'image'
    elif extension in ['.doc', '.docx']:
        return 'docx'
    elif extension in ['.xls', '.xlsx']:
        return 'xlsx'
    elif extension == '.txt':
        return 'txt'
    else:
        return None

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using pdfplumber.
    """
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return "\n".join(text)
    except Exception as e:
        # Add OCR fallback here if needed
        return ""

def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from an image file using pytesseract.
    """
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    """
    try:
        import docx
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        return ""

def extract_text_from_xlsx(file_path: str) -> str:
    """
    Extracts text from an XLSX file using openpyxl.
    """
    try:
        import openpyxl
        workbook = openpyxl.load_workbook(file_path)
        text = []
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows():
                row_text = []
                for cell in row:
                    if cell.value:
                        row_text.append(str(cell.value))
                if row_text:
                    text.append(" ".join(row_text))
        return "\n".join(text)
    except Exception as e:
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """
    Extracts text from a TXT file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return ""

def extract_text(file_path: str) -> str:
    """
    Extracts text from a file based on its type.
    """
    file_type = get_file_type(file_path)
    if file_type == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_type == 'image':
        return extract_text_from_image(file_path)
    elif file_type == 'docx':
        return extract_text_from_docx(file_path)
    elif file_type == 'xlsx':
        return extract_text_from_xlsx(file_path)
    elif file_type == 'txt':
        return extract_text_from_txt(file_path)
    else:
        return ""