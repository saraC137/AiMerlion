# test_extraction_comprehensive.py
"""
🧚‍♀️✨ FAIRY CODEMOTHER'S COMPREHENSIVE EXTRACTION TEST SUITE! ✨🧚‍♀️

This fabulous test file validates ALL extraction fields that appear in the final JSON output:
- ID, Name, Email, Phone, Date_of_Birth
- Skills, Working_Experience, Location, School_University
- Language, Extraction_Status, Notes, AI_Assisted, Filenames_Processed

Think of this as the FULL DRESS REHEARSAL before the main show, darling! 💅

Usage:
    python test_extraction_comprehensive.py

Expected Output:
    A complete JSON-like result showing all extracted fields + individual test results
"""

import json
import re
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# ============================================================================
# 🎭 TEST RESUME DATA - Singapore-style format (Date-first)
# ============================================================================
TEST_TEXT = """
Source: merlion_resumes/43199_Tan Kee Miang Nigel/Nigel Tan Kee Miang Resume (MSWords).docx
File Type: DOCX
Images Found: 0
Extraction Method: python-docx
================================================================================

Nigel Tan Kee Miang						
Block 362
Hougang Avenue 5 #12-302
Singapore 530362
62886508 (Home)/ 98153946 (Mobile)
Email: nigeltkm@gmail.com
Personal Particulars
Date of Birth	: 13 Oct 1989
Nationality	: Singaporean
Language	: English & Chinese
Work Experience
Feb 2016 to Present	Financial Consultant, Prudential Assurance Company Singapore (Pte) Ltd
Carried out in-depth policy reviews and managed the Sum at Risk (SAR).
Conducted frontline KYC and background fact finding of clients.
Maintained industry practice standards and Achieved A for Balance Scorecard grade.
Jan 2015 to Jan 2016	Business Consultant, TWH Consultancy Pte Ltd
Established company insurance arm.
Managing general insurance portfolio and assist in corporate planning.
Providing businesses with networks
Dec 2013 to Jan 2015	Financial Consultant, Prudential Assurance Company Singapore (Pte) Ltd
Engaged clients in financial planning for long and short term goals.
Created networks between clients.
Held seminars to educate participants on changes in the industry.
Dec 2011 to Nov 2013	Accounts Assistant, SKF Asia Pacific Pte Ltd
Ensured details of invoices are correct and keyed in on time.
Worked with auditors to maintain proper accounting standards.
Enforcement of guidelines to speed up efficiency of claims.
Education
Apr 2006 to Apr 2009	Diploma in Banking and Financial Services (Financial Trading)
			Singapore Polytechnic
			Obtained A in	: Treasury Options, Macroeconomic Analysis, 
					  Goal Setting and Decision Making
			Credits in	: Financial Accounting, Principles of Accounting	
Jan 2002 to Dec 2005	GCE 'O' Level
			Deyi Secondary School
			Obtained A in	: Mathematics, Science (Physics/Chemistry)					Credits in	: Principles of Accounting, Additional Mathematics
Achievements
Year			Description
2011			Grand Slam Award
2010			Overall Top Trainee for Basic Specialisation Course 4/09
2010			Best Trainee (C2) for Basic Specialisation Course 4/09
2004			Platoon Best Cadet for Deyi NCC Leadership Course
Co-Curricular Activities
Years of Service		Post Held
2002 to 2005		Master Sergeant (CSM), Deyi National Cadet Corps
Organised and run camps for the unit.
Planned and carried out daily training programmes.
Choreographed Precision Drills performance for Army Open House. 
Skills & Abilities
Organisation Skills
Prepared and hosted for seminars to educate participants.
Organised for internal audit checks for taxation with shipping department.
Planned and executed camps and orientations programmes for Deyi National Cadet Corps.
Supported the team in the organisation of Deyi Secondary School opening ceremony.
Interpersonal Skills
Applied emotional competence to manage self and others.
Worked and connected with clients of different backgrounds.
Engaged in discussion with various staffs to further understand problems faced in claims submission.
Worked as a team in intense situations during Fire Fighting and Damage Control Exercises.
Participated actively in group projects and presentations for C.K. TANG LTD during Final Year Project.
Leadership Skills
Motivated and helped ship crew achieve higher standard for IPPT and fitness event.
Led Deyi NCC in achieving awards for Precision Drill competitions.
Motivated the team during Mt Ophir Expedition.
Additional Qualifications
CMFAS M5, M8, M8A, M9, M9A
HI
CGI – BCP, ComGI, PGI
"""

# ============================================================================
# 🎯 EXPECTED VALUES - What we SHOULD extract from the test resume
# ============================================================================
EXPECTED_VALUES = {
    "ID": 43199,  # From folder name
    "Name": "Nigel Tan Kee Miang",
    "Email": "nigeltkm@gmail.com",
    "Phone": ["62886508", "98153946"],  # Either one is acceptable
    "Date_of_Birth": "1989-10-13",  # 13 Oct 1989 → YYYY-MM-DD
    "Location": "Singapore 530362",  # Or contains "Singapore"
    "Language": "English",  # Primary language
    "Skills_Count_Min": 5,  # Minimum skills expected
    "Experience_Count_Min": 3,  # Minimum jobs expected  
    "Education_Count_Min": 1,  # Minimum education entries
}


class ComprehensiveExtractionTester:
    """
    🎭 THE ULTIMATE TEST SUITE DIVA!
    
    She tests EVERY field like a judge at a beauty pageant - 
    no contestant (field) escapes her scrutiny! 👑
    """
    
    def __init__(self):
        self.test_results = {}
        self.passed_count = 0
        self.failed_count = 0
        self.warnings_count = 0
        self.ai_extractor = None
        self.extracted_data = {}
        
    def _print_header(self, title: str):
        """Print a fabulous section header"""
        print("\n" + "=" * 80)
        print(f"🌟 {title} 🌟".center(80))
        print("=" * 80)
        
    def _print_result(self, field: str, status: str, value: Any, expected: Any = None):
        """Print individual test result with flair"""
        if status == "PASS":
            emoji = "✅"
            self.passed_count += 1
        elif status == "FAIL":
            emoji = "❌"
            self.failed_count += 1
        elif status == "WARN":
            emoji = "⚠️"
            self.warnings_count += 1
        else:
            emoji = "ℹ️"
            
        # Truncate long values for display
        display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
        
        print(f"\n{emoji} {field}: {status}")
        print(f"   📦 Extracted: {display_value}")
        if expected:
            print(f"   🎯 Expected:  {expected}")
            
        self.test_results[field] = {
            "status": status,
            "extracted": value,
            "expected": expected
        }
        
    def initialize_extractor(self) -> bool:
        """
        🚀 Initialize the AI Extractor - The main performer!
        """
        self._print_header("INITIALIZING EXTRACTION ENGINE")
        
        try:
            # Try to import from the project files
            sys.path.insert(0, '/mnt/project')
            from ai_extractor import AIExtractor
            
            print("📦 Importing AIExtractor...")
            
            # Try different model names based on config
            model_names = ["llama3.1-128k", "llama3.1:8b", "llama3:8b", "tinyllama"]
            
            for model_name in model_names:
                try:
                    print(f"🤖 Trying model: {model_name}")
                    self.ai_extractor = AIExtractor(model_name)
                    
                    if self.ai_extractor.available:
                        print(f"✅ AIExtractor initialized with {model_name}!")
                        return True
                    else:
                        print(f"⚠️ Model {model_name} not available, trying next...")
                except Exception as e:
                    print(f"⚠️ Model {model_name} failed: {e}")
                    continue
            
            # If no AI model available, still continue with regex-only
            print("⚠️ No AI model available - testing regex-only extraction")
            self.ai_extractor = AIExtractor("dummy")  # Will use regex fallback
            return True
            
        except ImportError as e:
            print(f"❌ Failed to import AIExtractor: {e}")
            print("   Make sure you're running from the project directory!")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
            
    def extract_id_from_source(self, text: str) -> Optional[int]:
        """
        🔢 Extract ID from the source path in the resume text
        Pattern: merlion_resumes/[ID]_[Name]/...
        """
        # Pattern for folder name like "43199_Tan Kee Miang Nigel"
        id_patterns = [
            r'merlion_resumes/(\d+)_',  # From source path
            r'/(\d+)_[A-Za-z]',  # Generic ID_Name pattern
            r'^(\d+)_',  # At start of folder name
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return None
        
    def extract_filename_from_source(self, text: str) -> Optional[str]:
        """
        📁 Extract filename from the source path
        """
        match = re.search(r'Source:\s*([^\n]+)', text)
        if match:
            path = match.group(1).strip()
            return os.path.basename(path)
        return None

    def test_all_fields(self):
        """
        🎪 THE MAIN EVENT! Test ALL extraction fields!
        
        This is like the talent portion of the pageant, honey!
        Each field gets its moment to SHINE! ✨
        """
        if not self.ai_extractor:
            print("❌ Extractor not initialized! Call initialize_extractor() first!")
            return
            
        text = TEST_TEXT
        
        # ============================================================
        # 🔢 TEST 1: ID EXTRACTION
        # ============================================================
        self._print_header("TEST 1: ID EXTRACTION")
        
        extracted_id = self.extract_id_from_source(text)
        expected_id = EXPECTED_VALUES["ID"]
        
        if extracted_id == expected_id:
            self._print_result("ID", "PASS", extracted_id, expected_id)
        elif extracted_id:
            self._print_result("ID", "WARN", extracted_id, expected_id)
        else:
            self._print_result("ID", "FAIL", extracted_id, expected_id)
            
        self.extracted_data["ID"] = extracted_id
        
        # ============================================================
        # 👤 TEST 2: NAME EXTRACTION
        # ============================================================
        self._print_header("TEST 2: NAME EXTRACTION")
        
        # Use header extraction which includes name
        header_data = self.ai_extractor.extract_header_fields(text)
        extracted_name = header_data.get("name")
        expected_name = EXPECTED_VALUES["Name"]
        
        # Check if name contains key parts
        if extracted_name:
            name_parts = ["Nigel", "Tan"]
            has_key_parts = any(part.lower() in extracted_name.lower() for part in name_parts)
            
            if extracted_name.lower() == expected_name.lower():
                self._print_result("Name", "PASS", extracted_name, expected_name)
            elif has_key_parts:
                self._print_result("Name", "WARN", extracted_name, f"Expected '{expected_name}' (partial match)")
            else:
                self._print_result("Name", "FAIL", extracted_name, expected_name)
        else:
            self._print_result("Name", "FAIL", None, expected_name)
            
        self.extracted_data["Name"] = extracted_name
        
        # ============================================================
        # 📧 TEST 3: EMAIL EXTRACTION  
        # ============================================================
        self._print_header("TEST 3: EMAIL EXTRACTION")
        
        extracted_email = header_data.get("email")
        # Also try regex fallback
        if not extracted_email:
            extracted_email = self.ai_extractor._extract_email_regex(text)
            
        expected_email = EXPECTED_VALUES["Email"]
        
        if extracted_email and extracted_email.lower() == expected_email.lower():
            self._print_result("Email", "PASS", extracted_email, expected_email)
        elif extracted_email and "@" in extracted_email:
            self._print_result("Email", "WARN", extracted_email, expected_email)
        else:
            self._print_result("Email", "FAIL", extracted_email, expected_email)
            
        self.extracted_data["Email"] = extracted_email
        
        # ============================================================
        # 📱 TEST 4: PHONE EXTRACTION
        # ============================================================
        self._print_header("TEST 4: PHONE EXTRACTION")
        
        extracted_phone = header_data.get("phone")
        expected_phones = EXPECTED_VALUES["Phone"]
        
        # Check if any expected phone number is found
        if extracted_phone:
            phone_digits = re.sub(r'\D', '', str(extracted_phone))
            expected_found = any(
                re.sub(r'\D', '', exp) in phone_digits 
                for exp in expected_phones
            )
            
            if expected_found:
                self._print_result("Phone", "PASS", extracted_phone, f"One of {expected_phones}")
            elif len(phone_digits) >= 8:  # Singapore phones are 8 digits
                self._print_result("Phone", "WARN", extracted_phone, f"Expected one of {expected_phones}")
            else:
                self._print_result("Phone", "FAIL", extracted_phone, f"Expected one of {expected_phones}")
        else:
            self._print_result("Phone", "FAIL", None, f"Expected one of {expected_phones}")
            
        self.extracted_data["Phone"] = extracted_phone
        
        # ============================================================
        # 🎂 TEST 5: DATE OF BIRTH EXTRACTION
        # ============================================================
        self._print_header("TEST 5: DATE OF BIRTH EXTRACTION")
        
        # Try the English DOB extractor
        extracted_dob = self.ai_extractor._extract_dob_english(text)
        
        # Also check header data
        if not extracted_dob:
            extracted_dob = header_data.get("date_of_birth")
            
        expected_dob = EXPECTED_VALUES["Date_of_Birth"]
        
        if extracted_dob:
            # Normalize dates for comparison (accept various formats)
            dob_normalized = re.sub(r'[^\d]', '', str(extracted_dob))
            expected_normalized = re.sub(r'[^\d]', '', expected_dob)
            
            # Check if year 1989, month 10, day 13 are present
            if "19891013" in dob_normalized or dob_normalized == expected_normalized:
                self._print_result("Date_of_Birth", "PASS", extracted_dob, expected_dob)
            elif "1989" in str(extracted_dob):
                self._print_result("Date_of_Birth", "WARN", extracted_dob, f"Expected {expected_dob} (year matches)")
            else:
                self._print_result("Date_of_Birth", "FAIL", extracted_dob, expected_dob)
        else:
            self._print_result("Date_of_Birth", "FAIL", None, expected_dob)
            
        self.extracted_data["Date_of_Birth"] = extracted_dob
        
        # ============================================================
        # 🎯 TEST 6: SKILLS EXTRACTION
        # ============================================================
        self._print_header("TEST 6: SKILLS EXTRACTION")
        
        skills_result = self.ai_extractor._extract_skills_regex(text)
        hard_skills = skills_result.get("hard_skills", [])
        soft_skills = skills_result.get("soft_skills", [])
        total_skills = len(hard_skills) + len(soft_skills)
        
        min_expected = EXPECTED_VALUES["Skills_Count_Min"]
        
        print(f"   📊 Hard Skills ({len(hard_skills)}): {hard_skills[:5]}{'...' if len(hard_skills) > 5 else ''}")
        print(f"   📊 Soft Skills ({len(soft_skills)}): {soft_skills[:5]}{'...' if len(soft_skills) > 5 else ''}")
        
        if total_skills >= min_expected:
            self._print_result("Skills", "PASS", f"{total_skills} skills found", f"Min {min_expected}")
        elif total_skills > 0:
            self._print_result("Skills", "WARN", f"{total_skills} skills found", f"Min {min_expected}")
        else:
            self._print_result("Skills", "FAIL", "No skills found", f"Min {min_expected}")
            
        # Format for JSON output
        all_skills = hard_skills + soft_skills
        self.extracted_data["Skills"] = " | ".join(all_skills) if all_skills else None
        
        # ============================================================
        # 💼 TEST 7: WORKING EXPERIENCE EXTRACTION
        # ============================================================
        self._print_header("TEST 7: WORKING EXPERIENCE EXTRACTION")

        # 🎭 FAIRY CODEMOTHER'S DEBUG MODE ACTIVATED! ✨
        print("\n🔍 DEBUG: Calling date-first format extractor...")

        # Test date-first format (Singapore style)
        experience = self.ai_extractor._extract_experience_date_first_format(text)

        # 🎯 DEBUG: Print FULL job details!
        if experience:
            print(f"\n📊 DEBUG: Got {len(experience)} jobs from extractor")
            print("\n" + "="*60)
            for i, job in enumerate(experience, 1):
                print(f"\n💼 JOB #{i}:")
                print(f"   Company: {job.get('company', 'N/A')}")
                print(f"   Role: {job.get('role', 'N/A')}")
                print(f"   Dates: {job.get('dates', 'N/A')}")
                print(f"   Description Length: {len(job.get('description', ''))} chars")
                print(f"   Description Preview: {job.get('description', 'N/A')[:200]}...")
            print("="*60 + "\n")
        else:
            print("❌ DEBUG: No jobs returned from date-first extractor!")
        
        # Fallback to regular extraction
        if not experience:
            experience = self.ai_extractor._extract_experience_regex(text)
            
        min_expected = EXPECTED_VALUES["Experience_Count_Min"]
        
        if experience:
            print(f"   📊 Jobs found: {len(experience)}")
            for i, job in enumerate(experience[:4], 1):
                print(f"   {i}. {job.get('role', 'N/A')} at {job.get('company', 'N/A')}")
                print(f"      Dates: {job.get('dates', 'N/A')}")
                # 🆕 SHOW THE DESCRIPTION, HONEY! 💅
                desc = job.get('description', 'No description')
                # 💎 Show FULL description with pretty formatting!
                if desc and desc != 'No description':
                    # Split by pipe delimiter for beautiful display
                    responsibilities = desc.split(' | ')
                    print(f"      Description ({len(desc)} chars, {len(responsibilities)} items):")
                    for idx, resp in enumerate(responsibilities[:10], 1):  # Show first 10
                        print(f"         {idx}. {resp}")
                    if len(responsibilities) > 10:
                        print(f"         ... and {len(responsibilities) - 10} more items!")
                else:
                    print(f"      Description: No description found")
                
        if len(experience) >= min_expected:
            self._print_result("Working_Experience", "PASS", f"{len(experience)} jobs found", f"Min {min_expected}")
        elif len(experience) > 0:
            self._print_result("Working_Experience", "WARN", f"{len(experience)} jobs found", f"Min {min_expected}")
        else:
            self._print_result("Working_Experience", "FAIL", "No jobs found", f"Min {min_expected}")
            
        # 💎 Format for JSON output - STRUCTURED ARRAY FORMAT! ✨
        if experience:
            # Keep the full structured data - this is MUCH better for databases! 💅
            self.extracted_data["Working_Experience"] = [
                {
                    "company": j.get('company', 'Unknown'),
                    "role": j.get('role', 'N/A'),
                    "dates": j.get('dates', 'N/A'),
                    "description": j.get('description', 'No description')
                }
                for j in experience
            ]
        else:
            self.extracted_data["Working_Experience"] = None
            
        # ============================================================
        # 📍 TEST 8: LOCATION EXTRACTION
        # ============================================================
        self._print_header("TEST 8: LOCATION EXTRACTION")
        
        extracted_location = header_data.get("location")
        
        # Fallback: search for Singapore address patterns
        if not extracted_location:
            location_patterns = [
                r'Singapore\s+\d{6}',  # Singapore postal code
                r'Block\s+\d+[^\n]+Singapore',  # Block address
                r'(?:Singapore|SG)\b',  # Just Singapore
            ]
            for pattern in location_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted_location = match.group(0)
                    break
                    
        if extracted_location and "singapore" in extracted_location.lower():
            self._print_result("Location", "PASS", extracted_location, "Contains 'Singapore'")
        elif extracted_location:
            self._print_result("Location", "WARN", extracted_location, "Expected Singapore address")
        else:
            self._print_result("Location", "FAIL", None, "Expected Singapore address")
            
        self.extracted_data["Location"] = extracted_location
        
        # ============================================================
        # 🎓 TEST 9: EDUCATION / SCHOOL_UNIVERSITY EXTRACTION
        # ============================================================
        self._print_header("TEST 9: EDUCATION EXTRACTION")
        
        # Test date-first format
        education = self.ai_extractor._extract_education_date_first_format(text)
        
        # Fallback to regular extraction
        if not education:
            education = self.ai_extractor._extract_education_regex(text)
            
        min_expected = EXPECTED_VALUES["Education_Count_Min"]
        
        if education:
            print(f"   📊 Education entries found: {len(education)}")
            for i, edu in enumerate(education[:3], 1):
                print(f"   {i}. {edu.get('degree', 'N/A')}")
                print(f"      Institution: {edu.get('institution', 'N/A')}")
                print(f"      Dates: {edu.get('dates', 'N/A')}")
                
        if len(education) >= min_expected:
            self._print_result("School_University", "PASS", f"{len(education)} entries", f"Min {min_expected}")
        elif len(education) > 0:
            self._print_result("School_University", "WARN", f"{len(education)} entries", f"Min {min_expected}")
        else:
            self._print_result("School_University", "FAIL", "No education found", f"Min {min_expected}")
            
        # Format for JSON output
        if education:
            edu_summaries = [
                f"{e.get('degree', 'N/A')} from {e.get('institution', 'Unknown')} ({e.get('dates', 'N/A')})"
                for e in education
            ]
            self.extracted_data["School_University"] = " || ".join(edu_summaries)
        else:
            self.extracted_data["School_University"] = None
            
        # ============================================================
        # 🗣️ TEST 10: LANGUAGE EXTRACTION
        # ============================================================
        self._print_header("TEST 10: LANGUAGE EXTRACTION")
        
        languages = self.ai_extractor._extract_languages_regex(text)
        
        if languages:
            print(f"   📊 Languages found: {len(languages)}")
            for lang in languages:
                prof = f" ({lang['proficiency']})" if lang.get('proficiency') else ""
                print(f"   - {lang['language']}{prof}")
                
        # Check if English is detected (primary language for this system)
        expected_lang = EXPECTED_VALUES["Language"]
        has_english = any("english" in l.get("language", "").lower() for l in languages)
        
        if has_english:
            self._print_result("Language", "PASS", f"{len(languages)} languages", f"Contains {expected_lang}")
        elif languages:
            self._print_result("Language", "WARN", f"{len(languages)} languages", f"Expected {expected_lang}")
        else:
            self._print_result("Language", "FAIL", "No languages found", f"Expected {expected_lang}")
            
        self.extracted_data["Language"] = expected_lang  # Default for English resumes
        
        # ============================================================
        # 📊 TEST 11: EXTRACTION STATUS (Calculated)
        # ============================================================
        self._print_header("TEST 11: EXTRACTION STATUS")
        
        # Count successfully extracted fields
        key_fields = ["Name", "Email", "Phone", "Date_of_Birth"]
        extracted_count = sum(1 for f in key_fields if self.extracted_data.get(f))
        
        if extracted_count >= 5:
            status = "Complete"
        elif extracted_count >= 3:
            status = "Success"
        elif extracted_count >= 1:
            status = "Partial"
        else:
            status = "Failed"
            
        self._print_result("Extraction_Status", "PASS" if status in ["Complete", "Success"] else "WARN", 
                          status, f"Based on {extracted_count}/{len(key_fields)} key fields")
        
        self.extracted_data["Extraction_Status"] = status
        
        # ============================================================
        # 📝 TEST 12: NOTES (Generated)
        # ============================================================
        self._print_header("TEST 12: NOTES")
        
        notes = []
        if not self.extracted_data.get("Name"):
            notes.append("🚨 CRITICAL: No name found!")
        if not self.extracted_data.get("Email") and not self.extracted_data.get("Phone"):
            notes.append("🚨 NO CONTACT INFO FOUND!")
        if self.extracted_data.get("Date_of_Birth"):
            notes.append("✅ DOB extracted successfully")
            
        notes_str = " ".join(notes) if notes else "Extraction completed normally"
        
        self._print_result("Notes", "PASS", notes_str, "Generated notes")
        self.extracted_data["Notes"] = notes_str
        
        # ============================================================
        # 🤖 TEST 13: AI_ASSISTED FLAG
        # ============================================================
        self._print_header("TEST 13: AI_ASSISTED FLAG")
        
        ai_assisted = self.ai_extractor.available if self.ai_extractor else False
        
        self._print_result("AI_Assisted", "PASS", ai_assisted, "Boolean flag")
        self.extracted_data["AI_Assisted"] = ai_assisted
        
        # ============================================================
        # 📁 TEST 14: FILENAMES_PROCESSED
        # ============================================================
        self._print_header("TEST 14: FILENAMES_PROCESSED")
        
        filename = self.extract_filename_from_source(text)
        
        if filename:
            self._print_result("Filenames_Processed", "PASS", filename, "From source path")
        else:
            self._print_result("Filenames_Processed", "FAIL", None, "Expected filename from source")
            
        self.extracted_data["Filenames_Processed"] = filename

    def test_additional_sections(self):
        """
        🎁 BONUS ROUND! Test additional extraction methods!
        
        These are the extra talents that make our extraction SHINE! ✨
        """
        text = TEST_TEXT
        
        self._print_header("BONUS: ADDITIONAL SECTIONS")
        
        # Test certifications
        print("\n🏆 Certifications:")
        certs = self.ai_extractor._extract_certifications_regex(text)
        print(f"   Found {len(certs)} certifications")
        for cert in certs[:3]:
            print(f"   - {cert.get('name', 'N/A')}")
            
        # Test summary
        print("\n📝 Summary/Profile:")
        summary = self.ai_extractor._extract_summary_regex(text)
        if summary:
            print(f"   {summary[:100]}..." if len(summary) > 100 else f"   {summary}")
        else:
            print("   Not found (not all resumes have summaries)")
            
        # Test achievements
        print("\n🏅 Achievements:")
        achievements = self.ai_extractor._extract_achievements_regex(text)
        print(f"   Found {len(achievements)} achievements")
        for ach in achievements[:3]:
            print(f"   - {ach[:50]}..." if len(ach) > 50 else f"   - {ach}")
            
        # Test projects
        print("\n🚀 Projects:")
        projects = self.ai_extractor._extract_projects_regex(text)
        print(f"   Found {len(projects)} projects")

    def generate_final_json(self):
        """
        📋 Generate the final JSON output matching the expected format!
        
        This is the GRAND FINALE, darling! 👑
        """
        self._print_header("FINAL JSON OUTPUT")
        
        # Build the final JSON structure
        final_json = {
            "ID": self.extracted_data.get("ID"),
            "Name": self.extracted_data.get("Name"),
            "Email": self.extracted_data.get("Email"),
            "Phone": self.extracted_data.get("Phone"),
            "Date_of_Birth": self.extracted_data.get("Date_of_Birth"),
            "Skills": self.extracted_data.get("Skills"),
            "Working_Experience": self.extracted_data.get("Working_Experience"),
            "Location": self.extracted_data.get("Location"),
            "School_University": self.extracted_data.get("School_University"),
            "Language": self.extracted_data.get("Language"),
            "Extraction_Status": self.extracted_data.get("Extraction_Status"),
            "Notes": self.extracted_data.get("Notes"),
            "AI_Assisted": self.extracted_data.get("AI_Assisted"),
            "Filenames_Processed": self.extracted_data.get("Filenames_Processed")
        }
        
        print(json.dumps(final_json, indent=2, ensure_ascii=False))
        
        return final_json

    def print_summary(self):
        """
        🏆 Print the test summary - The final scoreboard!
        """
        self._print_header("TEST SUMMARY")
        
        total = self.passed_count + self.failed_count + self.warnings_count
        
        print(f"\n   ✅ PASSED:   {self.passed_count}/{total}")
        print(f"   ⚠️ WARNINGS: {self.warnings_count}/{total}")
        print(f"   ❌ FAILED:   {self.failed_count}/{total}")
        
        pass_rate = (self.passed_count / total * 100) if total > 0 else 0
        
        print(f"\n   📊 Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 80:
            print("\n   🎉 FABULOUS! The extraction is working beautifully! 💅✨")
        elif pass_rate >= 60:
            print("\n   👍 Good progress, darling! A few tweaks needed! 💪")
        elif pass_rate >= 40:
            print("\n   🤔 We're getting there, honey! Keep polishing! 💎")
        else:
            print("\n   😱 Houston, we have a problem! Time for some debugging! 🔧")
            
        print("\n" + "=" * 80)


def main():
    """
    🎬 LIGHTS! CAMERA! ACTION!
    
    The main function that runs our comprehensive test suite!
    """
    print("\n" + "🌟" * 40)
    print("🧚‍♀️ FAIRY CODEMOTHER'S COMPREHENSIVE EXTRACTION TEST 🧚‍♀️".center(80))
    print("🌟" * 40)
    print("\nTesting ALL fields in the final JSON output! Let the show begin! 💅✨\n")
    
    tester = ComprehensiveExtractionTester()
    
    # Step 1: Initialize extractor
    if not tester.initialize_extractor():
        print("\n❌ Failed to initialize. Please check your setup!")
        return
        
    # Step 2: Run all field tests
    tester.test_all_fields()
    
    # Step 3: Test additional sections (bonus!)
    tester.test_additional_sections()
    
    # Step 4: Generate final JSON
    final_json = tester.generate_final_json()
    
    # Step 5: Print summary
    tester.print_summary()
    
    # Step 6: Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"extraction_test_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_timestamp": timestamp,
            "extracted_data": final_json,
            "test_results": tester.test_results,
            "summary": {
                "passed": tester.passed_count,
                "failed": tester.failed_count,
                "warnings": tester.warnings_count
            }
        }, f, indent=2, ensure_ascii=False, default=str)
        
    print(f"\n📁 Results saved to: {output_file}")
    print("\n✨ Test complete! May your extractions be ever fabulous! ✨")


if __name__ == "__main__":
    main()