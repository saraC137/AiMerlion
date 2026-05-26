
import logging
import coloredlogs
import json
import re
from typing import Dict, List, Any
from extraction.ai_extractor import AIExtractor

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt='%(asctime)s - 💅 %(levelname)s - %(message)s')

# --- CHALLENGING SAMPLES ---

MESSY_RESUME_1 = """
SARA TAN
Singapore Citizen | Available June 2026

I am a marketing professional. In my last role at Global Media (2020-2024), 
I managed a budget of $2M. Before that, I was at Small Agency from 2018 to 2020.

My education includes a Bachelor of Business from NUS which I completed in 2018.
I am good at Photoshop, Excel, and Social Media Marketing.
I also have a Google Ads certification.
"""

COMPLEX_RESUME_2 = """
AHMAD BIN ABDULLAH
Senior Project Manager

--- OVERVIEW ---
15 years in construction. Leading multi-million dollar projects in Singapore and Malaysia.

--- EMPLOYMENT & KEY ACHIEVEMENTS ---
1. BuildCorp SG | Project Manager | 2015 - Present
   * Led the construction of 'The Merlion Tower'.
   * Achievement: Completed 2 months ahead of schedule.
2. KL Builders | Site Engineer | 2009 - 2015
   * Supervised 50+ workers daily.

--- ACADEMIC BACKGROUND ---
Master of Civil Engineering, UTM, 2009
Bachelor of Civil Engineering, UTM, 2007

--- OTHER INFO ---
Languages: English, Malay, Mandarin
Certifications: PMP, Safety Officer Level 3
Hobbies: Marathon running
"""

def detect_sections_ai(extractor: AIExtractor, text: str) -> Dict[str, str]:
    """Stage 1: Detect Sections using AI"""
    prompt = f"""
    Identify all logical sections in this resume and return them as a JSON object.
    Normalized keys: summary, skills, experience, education, certifications, languages, others.
    
    Resume text:
    {text}
    
    Return ONLY JSON.
    """
    try:
        response = extractor._call_ollama_raw(prompt, num_predict=1000)
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"Section detection failed: {e}")
        return {}

def extract_entities_from_section(extractor: AIExtractor, section_name: str, content: str) -> Any:
    """Stage 2: Extract specific entities from a section (Simulating NER)"""
    if not content: return None
    
    prompts = {
        "experience": "Extract a list of jobs. For each job, find: company, title, start_date, end_date. Return as JSON list.",
        "education": "Extract a list of degrees. For each, find: school, degree, year. Return as JSON list.",
        "skills": "Extract a list of technical skills. Return as a simple JSON string list.",
        "summary": "Summarize the candidate's core value proposition in 1 sentence. Return as JSON object with key 'summary'."
    }
    
    if section_name not in prompts:
        return f"Skipping detailed extraction for {section_name}"

    prompt = f"""
    Section: {section_name.upper()}
    Content: {content}
    
    Task: {prompts[section_name]}
    Return ONLY valid JSON.
    """
    
    try:
        response = extractor._call_ollama_raw(prompt, num_predict=800)
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return f"Extraction failed: {e}"

def run_complex_test():
    """🧪 Run the full AI Sectioning -> Extraction pipeline test"""
    
    print("\n" + "="*80)
    print("🚀 PROTOTYPE: AI SECTIONING + DEEP ENTITY EXTRACTION")
    print("="*80)
    
    extractor = AIExtractor("llama3.1:8b", logger)
    
    samples = [
        {"name": "Messy (No Headers)", "text": MESSY_RESUME_1},
        {"name": "Complex (Custom Delimiters)", "text": COMPLEX_RESUME_2}
    ]
    
    for sample in samples:
        print(f"\n🏃 TESTING SAMPLE: {sample['name']}")
        print("-" * 40)
        
        # 1. Detect Sections using the NEW production method!
        print("🔍 Stage 1: Detecting Sections via Production AI method...")
        sections = extractor.detect_sections_ai(sample['text'])
        
        for name, content in sections.items():
            print(f"   ✅ Identified: {name}")
            
            # 2. Extract Entities from detected section (Simulating NER)
            print(f"      🪄 Stage 2: Extracting from {name}...")
            details = extract_entities_from_section(extractor, name, content)
            
            # Print a neat summary of what was found
            if isinstance(details, list):
                print(f"      📈 Found {len(details)} entries")
                for entry in details[:2]: # Show first 2
                    print(f"         • {entry}")
            else:
                print(f"      📈 Result: {details}")

    print("\n" + "="*80)
    print("✨ TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_complex_test()
