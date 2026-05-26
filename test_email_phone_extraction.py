
import logging
import coloredlogs
import json
from extraction.ai_extractor import AIExtractor
from utils import standardize_phone_number

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger,
                    fmt='%(asctime)s - 💅 %(levelname)s - %(message)s')

# Test cases for email and phone extraction
TEST_CASES = [
    {
        "name": "Singapore Standard",
        "text": """
        JOHN DOE
        Email: john.doe@gmail.com
        Mobile: +65 9123 4567
        Address: 123 Ang Mo Kio Ave 1, Singapore
        """,
        "expected_email": "john.doe@gmail.com",
        "expected_phone": "+65 9123 4567"
    },
    {
        "name": "Malaysia Standard",
        "text": """
        TAN MEI LING
        HP: +60 12-345 6789
        Email: tanml@company.com.my
        Location: Kuala Lumpur, Malaysia
        """,
        "expected_email": "tanml@company.com.my",
        "expected_phone": "+60 12-345 6789"
    },
    {
        "name": "International (India)",
        "text": """
        RAHUL SHARMA
        Contact: +91 98765 43210
        rahul.sharma@example.in
        Mumbai, India
        """,
        "expected_email": "rahul.sharma@example.in",
        "expected_phone": "+91 98765 43210"
    },
    {
        "name": "Bare Numbers and Emails",
        "text": """
        SARA SMITH
        sara.smith@outlook.com
        91234567
        """,
        "expected_email": "sara.smith@outlook.com",
        "expected_phone": "+65 9123 4567" # Should be standardized to SG if 8 digits
    },
    {
        "name": "Labeled with Slashes and Dots",
        "text": """
        KEVIN LEE
        H/P: 8123.4567
        Email... kevin.lee@tech.sg
        """,
        "expected_email": "kevin.lee@tech.sg",
        "expected_phone": "+65 8123 4567"
    },
    {
        "name": "Obfuscated/Complex Labels",
        "text": """
        AMANDA WONG
        Personal Mobile: (65) 9888-7777
        Primary Email Address: amanda_wong@university.edu.sg
        """,
        "expected_email": "amanda_wong@university.edu.sg",
        "expected_phone": "+65 9888 7777"
    },
    {
        "name": "Negative: Prohibited Domains",
        "text": """
        TEST USER
        Email: test@example.com
        Phone: 91234567
        """,
        "expected_email": None, # example.com is excluded by regex
        "expected_phone": "+65 9123 4567"
    },
    {
        "name": "Negative: Support/Info Emails",
        "text": """
        COMPANY PROFILE
        info@company.com
        support@tech.sg
        Contact us at 6123 4567
        """,
        "expected_email": None, # info@ and support@ are excluded
        "expected_phone": "+65 6123 4567"
    },
    {
        "name": "Negative: Invalid Email Format",
        "text": """
        BAD EMAIL
        myemail.com (missing @)
        user@domain (missing TLD)
        Phone: 91234567
        """,
        "expected_email": None,
        "expected_phone": "+65 9123 4567"
    },
    {
        "name": "Edge: Phone-like Serial Numbers",
        "text": """
        DEVICE INFO
        Serial: 12345678901234567 (too long)
        Model: 2024 (looks like year)
        """,
        "expected_email": None,
        "expected_phone": None # Should reject very long or year-like numbers
    },
    {
        "name": "Edge: Mixed Text and Numbers",
        "text": """
        Order #12345678
        Batch 9876-5432
        This is not a phone number.
        """,
        "expected_email": None,
        "expected_phone": None # These don't have phone labels or SG/MY structures
    }
]

def run_tests():
    """🎭 Run the extraction tests"""
    
    print("\n" + "="*80)
    print("🔬 TESTING EMAIL & PHONE EXTRACTION")
    print("="*80)
    
    # Initialize extractor (using llama3.1:8b as seen in other scripts)
    # We'll try to use the most common model
    model_name = "llama3.1:8b"
    extractor = AIExtractor(model_name, logger)
    
    if not extractor.available:
        print(f"❌ AI Model '{model_name}' not available! Skipping AI part.")
        # We can still test regex methods if we want to be thorough
    
    results = []
    
    for case in TEST_CASES:
        print(f"\n🏃 Running Case: {case['name']}")
        print("-" * 40)
        
        # 1. Test Regex Methods Directly
        regex_email = extractor._extract_email_regex(case['text'])
        regex_phone_raw = extractor._extract_phone_regex(case['text'])
        regex_phone = standardize_phone_number(regex_phone_raw) if regex_phone_raw else None
        
        print(f"🔍 Regex Email: {regex_email}")
        print(f"🔍 Regex Phone: {regex_phone} (Raw: {regex_phone_raw})")
        
        # 2. Test Hybrid Extraction (Full AI Flow)
        ai_result = {}
        if extractor.available:
            try:
                # header_fields calls both regex and AI
                header_info = extractor.extract_header_fields(case['text'])
                ai_result = {
                    "email": header_info.get("email"),
                    "phone": header_info.get("phone")
                }
                print(f"🤖 AI Hybrid Email: {ai_result['email']}")
                print(f"🤖 AI Hybrid Phone: {ai_result['phone']}")
            except Exception as e:
                print(f"⚠️ AI Extraction failed: {e}")
        
        # Validation
        email_match = (regex_email == case['expected_email'])
        phone_match = (regex_phone == case['expected_phone'])
        
        if extractor.available and ai_result:
            email_match = email_match or (ai_result['email'] == case['expected_email'])
            phone_match = phone_match or (ai_result['phone'] == case['expected_phone'])
        
        status = "✅ PASS" if email_match and phone_match else "❌ FAIL"
        
        print(f"\nRESULT: {status}")
        if not email_match:
            print(f"  ❌ Email mismatch! Expected: {case['expected_email']}")
        if not phone_match:
            print(f"  ❌ Phone mismatch! Expected: {case['expected_phone']}")
            
        results.append({
            "case": case['name'],
            "status": status,
            "regex": {"email": regex_email, "phone": regex_phone},
            "ai": ai_result,
            "expected": {"email": case['expected_email'], "phone": case['expected_phone']}
        })

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
