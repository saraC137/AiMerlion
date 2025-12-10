
import os
import glob
import csv

def classify_resumes():
    """
    Classifies resumes as Japanese, English, or both, and exports the results to a CSV file.
    """
    resume_dir = "merlion_resumes/2025-07-17_09-43-21"
    pdf_files = glob.glob(os.path.join(resume_dir, "**/*.pdf"), recursive=True)
    
    candidates = {}

    for file_path in pdf_files:
        parts = file_path.split(os.sep)
        candidate_name = parts[-2]

        if candidate_name not in candidates:
            candidates[candidate_name] = {"english": False, "japanese": False}

        filename = os.path.basename(file_path).lower()

        if "履歴書" in filename or "職務経歴書" in filename:
            candidates[candidate_name]["japanese"] = True
        if "resume" in filename or "cv" in filename:
            candidates[candidate_name]["english"] = True

    with open("resume_classification.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Candidate Name", "Has English Resume", "Has Japanese Resume", "Has Both"])

        for name, data in candidates.items():
            has_both = data["english"] and data["japanese"]
            writer.writerow([name, data["english"], data["japanese"], has_both])

if __name__ == "__main__":
    classify_resumes()
