import os
import sys
import pandas as pd
import json
from main import UltimateResumeExtractor, find_candidate_folders
from utils import save_checkpoint, load_checkpoint
import config

def save_data_and_progress(extracted_data, candidate_folder):
    """
    Asks the user to save the extracted data and update the progress checkpoint.
    """
    save_choice = input("Do you want to save the extracted data? (y/n): ").lower()
    if save_choice == 'y':
        format_choice = input("Save as (csv/json): ").lower()
        if format_choice not in ['csv', 'json']:
            print("Invalid format. Data not saved.")
            return

        filename = input(f"Enter filename (without extension): ")
        if not filename:
            print("Invalid filename. Data not saved.")
            return

        if format_choice == 'csv':
            df = pd.DataFrame([extracted_data])
            df.to_csv(f"{filename}.csv", index=False, encoding='utf-8-sig')
            print(f"Data saved to {filename}.csv")
        elif format_choice == 'json':
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)
            print(f"Data saved to {filename}.json")

    progress_choice = input("Mark this candidate as extracted? (y/n): ").lower()
    if progress_choice == 'y':
        processed_folders = load_checkpoint(config.CHECKPOINT_FILE)
        if candidate_folder not in processed_folders:
            processed_folders.append(candidate_folder)
            save_checkpoint(processed_folders, config.CHECKPOINT_FILE)
            print("Progress updated.")
        else:
            print("Candidate already marked as extracted.")

def compare_extraction(candidate_id: str):
    """
    Compares the raw text of a resume with the extracted data for a given candidate.
    """
    # Find the candidate folder
    resume_folder = config.RESUME_FOLDER
    all_folders = find_candidate_folders(resume_folder)
    
    candidate_folder = None
    for folder in all_folders:
        if os.path.basename(folder).split('_')[0] == candidate_id:
            candidate_folder = folder
            break
    
    if not candidate_folder:
        print(f"Candidate folder for ID '{candidate_id}' not found.")
        return

    # Initialize the extractor
    extractor = UltimateResumeExtractor(config.SUZUME_MODEL_NAME)

    # Get the raw text
    combined_text = ""
    resume_files = [os.path.join(candidate_folder, f) for f in os.listdir(candidate_folder) if f.lower().endswith(('.pdf', '.docx'))]
    for file_path in resume_files:
        text = extractor.get_text_from_file(file_path)
        if text:
            combined_text += f"--- Raw text from {os.path.basename(file_path)} ---\n"
            combined_text += text + "\n\n"

    # Get the extracted data
    extracted_data = extractor.process_candidate_folder(candidate_folder)

    # Print the comparison
    print("--- Raw Text ---")
    print(combined_text)
    print("\n--- Extracted Data ---")

    if not extracted_data:
        print("No data was extracted. Performing diagnostic check...")
        if not combined_text.strip():
            print("DIAGNOSIS: The file appears to be empty or could not be read.")
        elif len(combined_text.strip()) < 50:
            print(f"DIAGNOSIS: The extracted text is very short ({len(combined_text.strip())} characters), which may be insufficient for data extraction.")
        else:
            lang = extractor.detect_language(combined_text)
            print(f"DIAGNOSIS: Language detected as '{lang}'. Ensure this language is supported by the extraction logic.")
            print("The raw text is provided above for manual inspection.")
    else:
        core_fields = ["Name", "Email", "Phone", "Date_of_Birth", "Skills", "Working_Experience", "Location", "School_University"]
        metadata_fields = ["Language", "Extraction_Status", "Notes", "AI_Assisted", "Filenames_Processed"]

        print("\n--- Core Extracted Data ---")
        for key in core_fields:
            value = extracted_data.get(key)
            if value is not None:
                print(f"{key}: {value}")

        print("\n--- Metadata ---")
        for key in metadata_fields:
            value = extracted_data.get(key)
            if value is not None:
                print(f"{key}: {value}")

        save_data_and_progress(extracted_data, candidate_folder)

if __name__ == "__main__":
    while True:
        print("\n--- Compare Extraction Menu ---")
        print("1. Enter Candidate ID to compare")
        print("2. Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            while True:
                candidate_id = input("Enter the candidate ID (or 'E' to exit): ")
                if candidate_id.lower() == 'e':
                    break
                compare_extraction(candidate_id)
        elif choice == '2':
            break
        else:
            print("Invalid choice. Please try again.")
