
import os
import pandas as pd

def find_resume_files(root_dir):
    
    resume_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            resume_files.append(os.path.join(root, file))
    return resume_files

def generate_report(files, output_file):
    
    report_data = []
    for file_path in files:
        directory, filename = os.path.split(file_path)
        candidate_name = os.path.basename(directory)
        
        file_format = os.path.splitext(filename)[1].lower()
        language = "Unknown"
        if "japanese" in filename.lower() or any(char in "履歴書職務経歴書" for char in filename):
            language = "Japanese"
        elif "english" in filename.lower() or "resume" in filename.lower():
            language = "English"
        
        report_data.append([candidate_name, filename, language, file_format])
    
    df = pd.DataFrame(report_data, columns=["Candidate", "File Name", "Language", "Format"])
    df.to_csv(output_file, index=False)

def main():
    
    resumes_dir = os.path.join(os.path.dirname(__file__), "merlion_resumes")
    output_file = os.path.join(os.path.dirname(__file__), "all_candidates_report.csv")
    
    resume_files = find_resume_files(resumes_dir)
    generate_report(resume_files, output_file)

if __name__ == "__main__":
    main()
