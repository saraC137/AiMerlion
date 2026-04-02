# Save as check_structure.py
import os
import re
import config

def check_structure():
    print("="*70)
    print("FOLDER STRUCTURE DIAGNOSTIC")
    print("="*70)
    
    resume_folder = config.RESUME_FOLDER
    print(f"\nBase folder: {resume_folder}")
    
    # Check date folders
    date_folders = []
    for item in os.scandir(resume_folder):
        if item.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', item.name):
            date_folders.append(item.path)
    
    print(f"\nDate folders found: {len(date_folders)}")
    if date_folders:
        print(f"Example: {os.path.basename(date_folders[0])}")
    
    # Check candidate folders
    candidate_count = 0
    candidates_with_files = 0
    
    for date_folder in date_folders[:1]:  # Check first date folder
        print(f"\nChecking inside: {os.path.basename(date_folder)}")
        
        for candidate in os.scandir(date_folder):
            if candidate.is_dir():
                candidate_count += 1
                
                # Check for resume files
                has_resume = False
                for file in os.scandir(candidate.path):
                    if file.is_file() and file.name.lower().endswith(('.pdf', '.docx')):
                        has_resume = True
                        candidates_with_files += 1
                        break
                
                if candidate_count <= 3:  # Show first 3
                    print(f"  Candidate {candidate_count}: {candidate.name}")
                    print(f"    Has resume files: {'YES' if has_resume else 'NO'}")
                    
                    if has_resume:
                        # Show files
                        files = [f.name for f in os.scandir(candidate.path) 
                                if f.is_file() and f.name.lower().endswith(('.pdf', '.docx'))]
                        print(f"    Files: {files[:2]}")  # Show first 2 files
    
    print(f"\n{'='*70}")
    print(f"Total candidates checked: {candidate_count}")
    print(f"Candidates with resume files: {candidates_with_files}")
    print(f"Candidates WITHOUT files: {candidate_count - candidates_with_files}")

if __name__ == "__main__":
    check_structure()