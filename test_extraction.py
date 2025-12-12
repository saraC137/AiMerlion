"""
🔬 DIAGNOSTIC TEST - Let's find where the problem REALLY is!
"""

import sys
from ai_extractor import AIExtractor
import logging
import coloredlogs
import json

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,
                    fmt='%(asctime)s - 💅 %(levelname)s - %(message)s')

# Sample problematic text (just the skills section for testing)
SAMPLE_TEXT = """
SKILLS
- Customer Service • Customer Relationship Management
- Service Contract Management • Inspection and Testing
- Field Technical Services • Preventive Maintenance and AMC
- Technical Support • Installations Expertise

EXPERIENCE
Sartorius India Pvt. Ltd. – Bio Analytical Division
Senior Field Service Engineer April 2021 – Present
Mumbai, India
- Executed the installation and qualification of Bio-analytical instruments
- Completed over 20+ installations Bio-analytical instruments.

Sartorius India Pvt. Ltd. – Analytical Division
Senior Field Service Engineer Feb 2016 – March 2021
Mumbai, India
- Executed the installation and qualification of analytical instruments
- Contributed significantly to revenue generation

EDUCATION
University of Mumbai
Bachelor of Engineering
Mumbai, Maharashtra
Major: Electronics Engineering
"""

def test_extraction():
    """🎭 Test the extraction and see EXACTLY what we get"""
    
    print("\n" + "="*80)
    print("🔬 DIAGNOSTIC TEST - AI EXTRACTOR")
    print("="*80)
    
    # Create extractor
    extractor = AIExtractor("llama3.1:8b", logger)
    
    if not extractor.available:
        print("❌ AI Model not available! Check Ollama!")
        return
    
    # Test extraction
    print("\n📥 Extracting from sample text...")
    results = extractor.extract_deep_fields(SAMPLE_TEXT)
    
    # Display results in DETAIL
    print("\n" + "="*80)
    print("📊 RAW RESULTS (Type and Content):")
    print("="*80)
    
    for key, value in results.items():
        print(f"\n🔑 KEY: {key}")
        print(f"   TYPE: {type(value).__name__}")
        
        if isinstance(value, str):
            print(f"   LENGTH: {len(value)} characters")
            print(f"   PREVIEW: {value[:200]}...")
            if '\n' in value or '•' in value:
                print("   ⚠️ WARNING: Contains newlines or bullets! This is WRONG!")
        
        elif isinstance(value, list):
            print(f"   LENGTH: {len(value)} items")
            if len(value) > 0:
                print(f"   FIRST ITEM TYPE: {type(value[0]).__name__}")
                print(f"   FIRST ITEM: {value[0]}")
                if len(value) > 1:
                    print(f"   SECOND ITEM: {value[1]}")
        
        elif isinstance(value, dict):
            print(f"   KEYS: {list(value.keys())}")
    
    # Save to JSON for inspection
    print("\n💾 Saving results to 'diagnostic_output.json'...")
    with open('diagnostic_output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # CRITICAL CHECK
    print("\n" + "="*80)
    print("🚨 CRITICAL CHECKS:")
    print("="*80)
    
    issues = []
    
    # Check 1: Are skills strings?
    for field in ['hard_skills', 'soft_skills']:
        if field in results:
            if isinstance(results[field], str):
                issues.append(f"❌ {field} is a STRING (should be LIST)")
            elif isinstance(results[field], list):
                if len(results[field]) > 0 and isinstance(results[field][0], str):
                    if '\n' in results[field][0] or '•' in results[field][0]:
                        issues.append(f"❌ {field} list items contain bullets/newlines")
                    else:
                        print(f"✅ {field} is properly formatted")
    
    # Check 2: Is experience a string?
    if 'working_experience' in results:
        if isinstance(results['working_experience'], str):
            issues.append(f"❌ working_experience is a STRING (should be LIST of OBJECTS)")
        elif isinstance(results['working_experience'], list):
            if len(results['working_experience']) > 0:
                if isinstance(results['working_experience'][0], dict):
                    print(f"✅ working_experience is properly formatted")
                else:
                    issues.append(f"❌ working_experience contains non-dict items")
    
    if issues:
        print("\n🚨 ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ ALL CHECKS PASSED! The extractor is working correctly!")
        print("   The problem must be in your MAIN SCRIPT or EXPORT CODE!")
    
    return results

if __name__ == "__main__":
    results = test_extraction()
    
    print("\n" + "="*80)
    print("📝 FINAL STRUCTURED OUTPUT:")
    print("="*80)
    print(json.dumps(results, indent=2, ensure_ascii=False))