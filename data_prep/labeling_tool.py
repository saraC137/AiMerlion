"""
Interactive Resume Labeling Tool
Label 1000 resumes in batches of 100 for fine-tuning preparation
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import jsonlines

import config

class InteractiveLabelingTool:
    """
    Interactive tool for labeling resume extractions.
    Designed for efficient batch labeling with progress tracking.
    """
    
    def __init__(self, 
                 resume_folder: str = config.RESUME_FOLDER,
                 output_file: str = "labeled_dataset.jsonl",
                 batch_size: int = 100):
        self.resume_folder = resume_folder
        self.output_file = output_file
        self.batch_size = batch_size
        
        # Progress tracking
        self.progress_file = "labeling_progress.json"
        self.current_batch = 0
        self.total_labeled = 0
        
        # Load existing progress
        self._load_progress()
        
        print("="*70)
        print("INTERACTIVE RESUME LABELING TOOL")
        print("="*70)
        print(f"Target: 1000 labeled resumes")
        print(f"Batch size: {batch_size} resumes")
        print(f"Current progress: {self.total_labeled} labeled")
        print(f"Remaining: {1000 - self.total_labeled} resumes")
    
    def _load_progress(self):
        """Load labeling progress from file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                self.current_batch = progress.get('current_batch', 0)
                self.total_labeled = progress.get('total_labeled', 0)
        
        # Count existing labeled samples
        if os.path.exists(self.output_file):
            with jsonlines.open(self.output_file) as reader:
                self.total_labeled = sum(1 for _ in reader)
    
    def _save_progress(self):
        """Save labeling progress."""
        progress = {
            'current_batch': self.current_batch,
            'total_labeled': self.total_labeled,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def start_labeling_session(self, extractor):
        """
        Start an interactive labeling session.
        """
        if self.total_labeled >= 1000:
            print("\n✅ Goal reached! 1000 resumes already labeled.")
            return
        
        print(f"\n🎯 Starting Batch #{self.current_batch + 1}")
        print(f"This batch: {self.batch_size} resumes")
        print("-"*70)
        
        # Get resume files with folder info
        all_files = self._get_resume_files()
        
        if not all_files:
            print("❌ No resume files found!")
            return
        
        # Get files for this batch
        batch_files = all_files[:self.batch_size]
        
        if not batch_files:
            print("✅ No more files to label!")
            return
        
        
        print("\nINSTRUCTIONS:")
        print("  [Y] - Accept extraction (correct)")
        print("  [N] - Skip this resume")
        print("  [E] - Edit fields manually")
        print("  [Q] - Quit and save progress")
        print("  [S] - Show current stats")
        print("-"*70)
        
        # Process batch
        labeled_count = 0
        
        for file_info in batch_files:
            try:
                file_path = file_info['file_path']
                folder_path = file_info['folder_path']
                candidate_id = file_info['id']
                
                # Extract resume
                text = extractor.get_text_from_file(file_path)
                if not text or len(text.strip()) < 50:
                    continue
                
                # Get extraction
                extracted_data, ai_used = extractor._extract_data_from_text(text)
                
                # Add ID to extracted data
                extracted_data['ID'] = candidate_id
                
                # Show for review
                print("\n" + "="*70)
                print(f"📄 File: {os.path.basename(file_path)}")
                print(f"📁 Folder: {os.path.basename(folder_path)}")
                print(f"🆔 ID: {candidate_id}")
                print("="*70)
                
                # Show extracted data
                self._display_extraction(extracted_data)
                
                # Show snippet of resume text
                print("\n📋 Resume Preview (first 500 chars):")
                print("-"*70)
                print(text[:500] + "..." if len(text) > 500 else text)
                print("-"*70)
                
                # Get user decision
                while True:
                    choice = input("\n[Y]es / [N]o / [E]dit / [Q]uit / [S]tats: ").strip().upper()
                    
                    if choice == 'Y':
                        # Accept as-is
                        self._save_labeled_example(text, extracted_data, candidate_id)
                        labeled_count += 1
                        self.total_labeled += 1
                        print("✅ Saved!")
                        break
                    
                    elif choice == 'N':
                        # Skip
                        print("⏭️ Skipped")
                        break
                    
                    elif choice == 'E':
                        # Edit fields
                        corrected_data = self._edit_fields(extracted_data)
                        corrected_data['ID'] = candidate_id  # Preserve ID
                        self._save_labeled_example(text, corrected_data, candidate_id)
                        labeled_count += 1
                        self.total_labeled += 1
                        print("✅ Saved with corrections!")
                        break
                    
                    elif choice == 'Q':
                        # Quit
                        print(f"\n💾 Saving progress... ({labeled_count} labeled in this session)")
                        self._save_progress()
                        print("✅ Progress saved. Run again to continue.")
                        return
                    
                    elif choice == 'S':
                        # Show stats
                        self._show_stats()
                    
                    else:
                        print("❌ Invalid choice. Try again.")
                
                # Auto-save every 10 resumes
                if labeled_count % 10 == 0 and labeled_count > 0:
                    self._save_progress()
                    print(f"💾 Auto-saved progress ({labeled_count} in this batch)")
            
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                self._save_progress()
                return
            
            except Exception as e:
                print(f"❌ Error processing file: {e}")
                continue
        
        # Batch complete
        self.current_batch += 1
        self._save_progress()
        
        print("\n" + "="*70)
        print(f"🎉 BATCH #{self.current_batch} COMPLETE!")
        print("="*70)
        print(f"Labeled in this batch: {labeled_count}")
        print(f"Total labeled: {self.total_labeled} / 1000")
        print(f"Progress: {self.total_labeled / 10:.1f}%")
        
        if self.total_labeled >= 1000:
            print("\n🏆 GOAL REACHED! 1000 resumes labeled!")
            print(f"Training dataset saved to: {self.output_file}")
        else:
            remaining = 1000 - self.total_labeled
            print(f"\n📊 Remaining: {remaining} resumes")
            print("Run the tool again to continue labeling.")
    
    def _get_resume_files(self) -> List[Dict]:
        """
        Get all resume files from the resume folder with ID information,
        skipping those that are already labeled.
        Returns list of dicts with file_path, folder_path, and id.
        """
        from main import find_candidate_folders, find_resume_files
        import re

        # Load labeled IDs
        labeled_ids = set()
        if os.path.exists(self.output_file):
            with jsonlines.open(self.output_file) as reader:
                for item in reader:
                    # Assuming the ID is stored in a consistent way
                    if item.get('output') and item['output'][0].get('candidate_id'):
                        labeled_ids.add(int(item['output'][0]['candidate_id']))

        all_files = []
        candidate_folders = find_candidate_folders(self.resume_folder)
        
        # Iterate through all candidate folders
        for folder in candidate_folders:
            folder_name = os.path.basename(folder)
            id_match = re.match(r'^(\d+)', folder_name)
            candidate_id = int(id_match.group(1)) if id_match else None
            
            if not candidate_id or candidate_id in labeled_ids:
                continue

            files = find_resume_files(folder)
            if files:
                all_files.append({
                    'file_path': files[0],
                    'folder_path': folder,
                    'id': candidate_id
                })
            
            # Stop if we have enough for the next batch + a buffer
            if len(all_files) >= 1000:
                break
        
        return all_files
    
    def _display_extraction(self, data: Dict):
        """Display extracted data in a readable format."""
        print("\n🔍 EXTRACTED DATA:")
        print("-"*70)
        
        fields = [
            ("ID", data.get("ID")),
            ("Name", data.get("Name")),
            ("Email", data.get("Email")),
            ("Phone", data.get("Phone")),
            ("Date of Birth", data.get("Date_of_Birth"))
        ]
        
        for field_name, value in fields:
            status = "✅" if value else "❌"
            display_value = value if value else "[NOT FOUND]"
            print(f"{status} {field_name:15s}: {display_value}")
    
    def _edit_fields(self, current_data: Dict) -> Dict:
        """Interactive field editing."""
        print("\n📝 EDIT FIELDS")
        print("-"*70)
        print("Press Enter to keep current value, or type new value")
        print("-"*70)
        
        edited_data = {}
        
        # ID (show but don't allow editing - it's from folder)
        current_id = current_data.get("ID", "")
        print(f"\nID (from folder): {current_id} [READ ONLY]")
        edited_data["ID"] = current_id
        
        # Name
        current = current_data.get("Name", "")
        print(f"\nName [{current}]: ", end="")
        new_value = input().strip()
        edited_data["Name"] = new_value if new_value else current
        
        # Email
        current = current_data.get("Email", "")
        print(f"Email [{current}]: ", end="")
        new_value = input().strip()
        edited_data["Email"] = new_value if new_value else current
        
        # Phone
        current = current_data.get("Phone", "")
        print(f"Phone [{current}]: ", end="")
        new_value = input().strip()
        edited_data["Phone"] = new_value if new_value else current
        
        # DOB
        current = current_data.get("Date_of_Birth", "")
        print(f"DOB (YYYY-MM-DD) [{current}]: ", end="")
        new_value = input().strip()
        edited_data["Date_of_Birth"] = new_value if new_value else current
        
        return edited_data
    
    def _save_labeled_example(self, text: str, labels: Dict, candidate_id: int):
        """Save a labeled example to the dataset."""
        # Format for training
        entry = {
            "input": text[:5000],  # Limit text length
            "output": [{
                "resume_id": str(candidate_id),
                "candidate_id": str(candidate_id),
                "candidate_name": labels.get("Name"),
                "contact_info": {
                    "email": [labels.get("Email")] if labels.get("Email") else [],
                    "phone": [labels.get("Phone")] if labels.get("Phone") else [],
                    "date_of_birth": labels.get("Date_of_Birth")
                }
            }]
        }
        
        # Append to JSONL file
        with jsonlines.open(self.output_file, mode='a') as writer:
            writer.write(entry)
    
    def _show_stats(self):
        """Display current statistics."""
        print("\n" + "="*70)
        print("📊 LABELING STATISTICS")
        print("="*70)
        print(f"Total labeled: {self.total_labeled} / 1000")
        print(f"Progress: {self.total_labeled / 10:.1f}%")
        print(f"Current batch: #{self.current_batch + 1}")
        print(f"Remaining: {1000 - self.total_labeled} resumes")
        
        if self.total_labeled > 0:
            # Calculate estimated time remaining
            # Assuming 30 seconds per resume
            remaining_time = (1000 - self.total_labeled) * 0.5  # minutes
            hours = int(remaining_time // 60)
            minutes = int(remaining_time % 60)
            print(f"Estimated time remaining: {hours}h {minutes}m (at 30s per resume)")
        
        print("="*70)
    
    def export_training_splits(self):
        """
        Export labeled data into train/validation splits.
        80% train, 20% validation
        """
        if not os.path.exists(self.output_file):
            print("❌ No labeled data found!")
            return
        
        # Load all labeled data
        all_data = []
        with jsonlines.open(self.output_file) as reader:
            all_data = list(reader)
        
        print(f"\n📦 Exporting {len(all_data)} labeled examples...")
        
        # Shuffle and split
        import random
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * 0.8)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        # Save splits
        with jsonlines.open('train_data.jsonl', mode='w') as writer:
            for item in train_data:
                writer.write(item)
        
        with jsonlines.open('val_data.jsonl', mode='w') as writer:
            for item in val_data:
                writer.write(item)
        
        print(f"✅ Training set: {len(train_data)} examples → train_data.jsonl")
        print(f"✅ Validation set: {len(val_data)} examples → val_data.jsonl")
        print("\n🎉 Ready for fine-tuning!")


def main():
    """Run the labeling tool."""
    print("="*70)
    print("RESUME LABELING TOOL - BATCH MODE")
    print("="*70)
    
    # Initialize tool
    tool = InteractiveLabelingTool(
        resume_folder=config.RESUME_FOLDER,
        batch_size=100
    )
    
    # Check if goal reached
    if tool.total_labeled >= 1000:
        print("\n✅ 1000 resumes already labeled!")
        
        export = input("\nExport train/validation splits? (y/n): ").strip().lower()
        if export == 'y':
            tool.export_training_splits()
        return
    
    # Start labeling
    print("\nInitializing extraction system...")
    from main import UltimateResumeExtractor
    
    extractor = UltimateResumeExtractor(config.MODEL_NAME)
    
    # Start session
    tool.start_labeling_session(extractor)
    
    # Ask if user wants to continue
    if tool.total_labeled < 1000:
        print("\n" + "-"*70)
        another = input("Start another batch? (y/n): ").strip().lower()
        if another == 'y':
            tool.start_labeling_session(extractor)


if __name__ == "__main__":
    main()