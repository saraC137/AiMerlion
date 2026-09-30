import os
import pandas as pd

def find_resume_files(root_dir):
    
    resume_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            resume_files.append(os.path.join(root, file))
    return resume_files

def identify_language_and_format(files):
    
    candidate_resumes = {}
    for file_path in files:
        directory, filename = os.path.split(file_path)
        candidate_name = os.path.basename(directory)
        
        if candidate_name not in candidate_resumes:
            candidate_resumes[candidate_name] = {"japanese": [], "english": [], "others": [], "file_count": 0}
        
        candidate_resumes[candidate_name]["file_count"] += 1
        file_format = os.path.splitext(filename)[1].lower()
        
        if "japanese" in filename.lower() or any(char in "履歴書職務経歴書" for char in filename):
            candidate_resumes[candidate_name]["japanese"].append(file_format)
        elif "english" in filename.lower() or "resume" in filename.lower():
            candidate_resumes[candidate_name]["english"].append(file_format)
        else:
            candidate_resumes[candidate_name]["others"].append(file_format)
            
    return candidate_resumes

def generate_report(candidate_resumes, output_file):
    
    report_data = []
    for candidate, resumes in candidate_resumes.items():
        if resumes["japanese"] and resumes["english"]:
            jp_formats = ", ".join(sorted(list(set(resumes["japanese"]))))
            en_formats = ", ".join(sorted(list(set(resumes["english"]))))
            many_files = "Yes" if resumes["file_count"] > 5 else "No"
            report_data.append([candidate, jp_formats, en_formats, many_files])
    
    df = pd.DataFrame(report_data, columns=["Candidate", "Japanese Resume Formats", "English Resume Formats", "Many Files"])
    df.to_csv(output_file, index=False)

def main():
    
    resumes_dir = os.path.join(os.path.dirname(__file__), "merlion_resumes")
    output_file = os.path.join(os.path.dirname(__file__), "language_and_format_report.csv")
    
    resume_files = find_resume_files(resumes_dir)
    candidate_resumes = identify_language_and_format(resume_files)
    generate_report(candidate_resumes, output_file)

if __name__ == "__main__":
    main()