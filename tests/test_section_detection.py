
import logging
import coloredlogs
import re
from typing import Dict, Tuple
from extraction.ai_extractor import AIExtractor

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt='%(asctime)s - 💅 %(levelname)s - %(message)s')

# Test cases for section detection
TEST_CASES = [
    {
        "name": "Standard English Resume",
        "text": """
JOHN DOE
Software Engineer

SUMMARY
Experienced software engineer with 5 years of experience in Python.

SKILLS
- Python, Django, Flask
- PostgreSQL, Redis

WORK EXPERIENCE
Senior Engineer | Tech Corp | 2020 - Present
- Built things.

EDUCATION
B.S. Computer Science | University of Nowhere | 2016 - 2020
        """,
        "expected_sections": ["summary", "skills", "experience", "education"]
    },
    {
        "name": "Asian Format (Singapore Style)",
        "text": """
TAN MEI LING
Address: 123 Ang Mo Kio Ave 1, Singapore
HP: 9123 4567

PERSONAL PARTICULARS
Nationality: Singaporean
Availability: Immediate

PROFESSIONAL PROFILE
Highly motivated individual with a background in marketing.

CORE COMPETENCIES
- Digital Marketing
- SEO/SEM

WORKING EXPERIENCE
Marketing Executive | SG Agency | Jan 2018 - Dec 2023
- Managed campaigns.

EDUCATIONAL QUALIFICATIONS
Diploma in Marketing | Singapore Polytechnic | 2015 - 2018

CO-CURRICULAR ACTIVITIES
Member of Swimming Club
        """,
        "expected_sections": ["personal_info", "summary", "skills", "experience", "education", "cocurricular"]
    },
    {
        "name": "Messy Layout & Compound Headers",
        "text": """
KEVIN LEE

About Me:
I am a developer.

Key Skills:
- JavaScript
- React

Employment History:
Lead Dev | Startup X | 2022
Built a cool app.

Academic History and Training:
B.Eng | NTU | 2021

Certifications & Licenses:
AWS Certified Developer
        """,
        "expected_sections": ["summary", "skills", "experience", "education", "certifications"]
    }
]

def run_tests():
    """🎭 Run the section detection tests"""
    
    print("\n" + "="*80)
    print("🔬 TESTING SECTION DETECTION")
    print("="*80)
    
    # Initialize extractor (model doesn't matter much for this as it's regex-based)
    extractor = AIExtractor("llama3.2:3b", logger)
    
    results = []
    
    for case in TEST_CASES:
        print(f"\n🏃 Running Case: {case['name']}")
        print("-" * 40)
        
        # Test Enhanced Section Detection
        try:
            # We need to access the protected method for testing
            sections = extractor._detect_section_boundaries_enhanced(case['text'])
            
            found_sections = list(sections.keys())
            print(f"✅ Found Sections: {found_sections}")
            
            # Print content snippets for verification
            for name, (start, end) in sections.items():
                content = case['text'][start:end].strip()
                snippet = content[:50].replace('\n', ' ') + "..." if len(content) > 50 else content
                print(f"   📍 {name:15}: {snippet}")
            
            # Validation
            missing = [s for s in case['expected_sections'] if s not in found_sections and f"{s}_simple" not in found_sections]
            extra = [s for s in found_sections if s not in case['expected_sections'] and s.replace('_simple', '') not in case['expected_sections']]
            
            status = "✅ PASS" if not missing else "❌ FAIL"
            
            if missing:
                print(f"  ❌ Missing expected sections: {missing}")
            if extra:
                print(f"  ℹ️ Found additional sections: {extra}")
                
            results.append({
                "case": case['name'],
                "status": status,
                "found": found_sections,
                "expected": case['expected_sections']
            })
        except Exception as e:
            print(f"💥 Error testing case: {e}")
            results.append({"case": case['name'], "status": "💥 ERROR"})

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    passed = sum(1 for r in results if r['status'] == "✅ PASS")
    print(f"Total Cases: {len(TEST_CASES)}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {len(TEST_CASES) - passed}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_tests()
