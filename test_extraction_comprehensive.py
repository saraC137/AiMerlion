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
TEST_TEXT = r"""
Source: merlion_resumes\43196_Binte Nurul\Nurul Ain Binte Ismail (ok).docx
File Type: DOCX
Images Found: 0
Extraction Method: python-docx
================================================================================

Name: Nurul Ain Binte Ismail
Add ress : Blk 756 Jurong West St 74 #05-64 S(640756 ) Hp No
: 90 927602 Email
: nurulainbte ismail @gmail.com Personal Particulars NRIC : S9243711F Race : Javanese Nationality : Singapo rean Date of Birth : 19 November 1992 Gender/Age: 
Female/22 Status : Single Objectives To obtain an interesting and challenging position with its w ide task, that will allow me to advance further enhancement while contributing for the company’s success. To obtain a challenging job that would give me an opportunity for professional growth and employment benefits. Education Singapore Institute of Retail Studies Diploma in Retail Management Oct 2015 – Sept 2016 Retailing and the Economy, Manage Marketing Mix, Contribute to the Buying of Merchandise, Lead and Manage Work Team, Manage Retail Productivity, Manage Merchandise Presentation &amp; Visual Display, Manage Quality Service Operation, Manage Operational Human Resource Policies, Manage Finance for Retail Operations Completing in end of Sept 2016 Advanced Certificate in Retail Supervision Jan 2015 – Sept 2015 Supervise Retail Staff, Provide Marketing and Promotion Support , Monitor visual merchandising presentation, Supervise Housekeeping Standards, Maintain Store Security Completed Advanced Cert in Retail Supervision Private Candidate for GCE ‘O’ Level Feb 2009 – Nov 200 9 Completed GCE ‘O’ Level Jurong West Secondary School 2005 – 200 8 Attended until Secondary 4 (Express Stream) Juying Primary School 199 9 – 2004 PSLE Experience Full-Time Sales Assistant| Ferragamo (Singapore) Pte Ltd June 2015 – Aug 2016 Sales generation. Monthly individual &amp; store targets orientated . Customer recruitment targets. Individual KPI targets. Customer service; personalize one to one service. Listening, recommending of products based on customer’s needs. Up-selling &amp; cross-selling. Closing on sales &amp; recruiting customer into CRM base. Customer orientated; creating customer rapport. Calling of customers &amp; engaging with them, on new arrivals &amp; promotions. Sending of birthdays cards &amp; catalogs. Inviting customers back to store on birthdays, inviting/rsvp customers during events &amp; interacting with customers. Create personalize customer service &amp; having own customer base &amp; VIPs. Constantly maintaining knowledgeable &amp; solid p roduct knowledge &amp; caring for product. Participates in product shipment, receiving, transferring, sensor tags all items, re-pricing , stocking up and replenishment s according to guidelines. Handle customer’s enquiries, feedback &amp; complaints. Handling after-sales services; repairs for shoes &amp; bags, defective cases &amp; handling follow-ups. Telephone etiquette, providing over the p hone enquiries. Goals &amp; target orientated. Daily morning briefings on store targets, key performance indicator goals for the day. Visual &amp; Merchandising m aintains all visual and housekeeping standards per directive . Maintain ing proper standards of display, color coordination &amp; according to styles &amp; season . Ensure all stocks are well replenished. Displaying new arrivals each week &amp; changing of walls &amp; windows for visual merchandising of items. Handling of stock takes; every 3 months. Ensure all disc repancies are found &amp; ensure SAP &amp; stocks are tally. Daily count of stocks in morning &amp; night to prevent loss. Administrative work; create reports on incentives for staff . Checking of transfers in &amp; out each day. Checking of stocks. Recording of monthly incentives . Stationeries requisition. Cashiering; End day of POS register. Settlement of credits, nets &amp; cash. Tax refunds, credit refunds, customer refunds &amp; buying of gift cards. Opening of POS register, counting of float money. Daily count of gift cards. Packing &amp; unpacking of stocks. Referring to D/N papers &amp; locations to locations transfers. Experience Full-Time Sales Assistant| Esprit Retail Pte Ltd June 2012 – May 2014 WSQ Retail Cert In Apply Color Theory. WSQ Retail Cert In Provide Advice on Fashion &amp; Apparels. WSQ Retail Cert in Maintain Professional Image. Assign to be in-charge of certain divisions within the store. Assign to various roles &amp; tasks in store. Senior sales associate. In-charge in training. Set up main store for Jem outlet. Customer service and sales generation. Actively attending to their needs in fashion, advising on fashion apparels, fitting room services &amp; cashiering service. Up-sell new arrivals, promoting promotions &amp; membership card. Giving customers best deals or saving on items they purchase. Constantly maintaining knowledgeable &amp; solid product knowledge, washing advice &amp; caring for product. Participates in product shipment, receiving, transferring, sensor tags all items, re-pricing, re-tagging, stocking up and replenishment s according to guidelines. Handle customer’s enquiries, feedback &amp; complaints. Provide alteration measurements for customers. Telephone etiquette, provide over the phone service. Goals, monthly &amp; divisions target orientated. Daily morning briefings on store targets, key performance indicator goals for the day. Individuals target for the week &amp; month. Visual &amp; Merchandising m aintains all visual and housekeeping standards per directive . Maintain proper standards of folding, hanging, steaming and sizing . Ensure all stocks are well replenished. Displaying new arrivals each week &amp; changing of walls for visual merchandising of items. Handling stock takes; scanning items, planning of store layout, planning of manpower &amp; designing staffs to designated tasks &amp; areas. Administrative work; calculations of key performance index of individual staffs. Checking of transfers in &amp; out each day. Checking of stocks. Emailing between stores to stores. Recording of petty cash, bank-in money &amp; store float money. Recording of alterations &amp; gift cards. Stationeries requisition. Cashiering; End day of POS register. Settlement of credits, nets &amp; cash. Tax refunds, credit refunds, customer refunds &amp; buying of gift cards. Opening of POS register, counting of float money. Daily count of gift cards. Packing &amp; unpacking of stocks. Referring to D/N papers &amp; locations to locations transfers. Reasons of resignation; wanting to experience &amp; explore new working environment &amp; job scope. Experience Full-Time Sales Assistant| Lachmann ( Lee Jeans ) May 2011 – Feb 2012 Customer service, handling of telephone calls (telephone etiquette), fitting room, customer complaints. Cashiering, opening &amp; closing of store. Stock display, monthly stock check, stock intake &amp; packing, weekly re-ordering of stocks &amp; replenishments, visual merchandising. experience Full-Time Sales Assistant| Wing Tai Asia ( Miss Selfridge ) April 2010 – April 2011 WSQ Retail Cert In Interact With Customers Customer service, handling of telephone calls (telephone etiquette) , fitting room. Cashiering, opening &amp; closing of store. 2 nd shop-in-charge, assist store in-charge to manage store, lead team of people. Stock display, stock check, stock intake &amp; packing , visual merchandising. experience Temp Sales Assistant | Triumph Jan 2010 – March 2010 Contract Basis for 3 months Basic customer service Stock display, stock check, stock intake &amp; packing experience Temp Sales Assistant | Burberry Boutique Apr 2008 – July 2008 Contract Basis for 3 months Basic customer service &amp; stock display languages Reads, writes &amp; speaks fluent English &amp; Malay. others Able to work in fast-pace working environment . P erform rotating s hifts &amp; able work independently or as a team. Willing to learn new t hings, take initiative &amp; adapt to new environment. Good conversational ability &amp; relates well with peers &amp; customers . Fast learner &amp; pro-active. Name: Nurul Ain Binte Ismail Name: Nurul Ain Binte Ismail Page 2 Page 2
"""

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
            sys.path.insert(0, '.')
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
            r'merlion_resumes[\\/](\d+)_',  # From source path
            r'[\\/](\d+)_[A-Za-z]',  # Generic ID_Name pattern
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
        
        if extracted_id:
            self._print_result("ID", "PASS", extracted_id)
        else:
            self._print_result("ID", "FAIL", None, "No ID extracted")
            
        self.extracted_data["ID"] = extracted_id
        
        # ============================================================
        # 👤 TEST 2: NAME EXTRACTION
        # ============================================================
        self._print_header("TEST 2: NAME EXTRACTION")
        
        # Use header extraction which includes name
        header_data = self.ai_extractor.extract_header_fields(text)
        extracted_name = header_data.get("name")
        
        if extracted_name:
            self._print_result("Name", "PASS", extracted_name)
        else:
            self._print_result("Name", "FAIL", None, "No name extracted")
            
        self.extracted_data["Name"] = extracted_name
        
        # ============================================================
        # 📧 TEST 3: EMAIL EXTRACTION  
        # ============================================================
        self._print_header("TEST 3: EMAIL EXTRACTION")
        
        extracted_email = header_data.get("email")
        # Also try regex fallback
        if not extracted_email:
            extracted_email = self.ai_extractor._extract_email_regex(text)
            
        if extracted_email:
            self._print_result("Email", "PASS", extracted_email)
        else:
            self._print_result("Email", "FAIL", None, "No email extracted")
            
        self.extracted_data["Email"] = extracted_email
        
        # ============================================================
        # 📱 TEST 4: PHONE EXTRACTION
        # ============================================================
        self._print_header("TEST 4: PHONE EXTRACTION")
        
        extracted_phone = header_data.get("phone")
        
        # Check if any expected phone number is found
        if extracted_phone:
            phone_digits = re.sub(r'\D', '', str(extracted_phone))
            if len(phone_digits) >= 8: # Basic check for phone number length
                self._print_result("Phone", "PASS", extracted_phone)
            else:
                self._print_result("Phone", "WARN", extracted_phone, "Extracted phone number might be too short")
        else:
            self._print_result("Phone", "FAIL", None, "No phone extracted")
            
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
            
        if extracted_dob:
            self._print_result("Date_of_Birth", "PASS", extracted_dob)
        else:
            self._print_result("Date_of_Birth", "FAIL", None, "No Date of Birth extracted")
            
        self.extracted_data["Date_of_Birth"] = extracted_dob
        
        # ============================================================
        # 🎯 TEST 6: SKILLS EXTRACTION
        # ============================================================
        self._print_header("TEST 6: SKILLS EXTRACTION")
        
        skills_result = self.ai_extractor._extract_skills_regex(text)
        hard_skills = skills_result.get("hard_skills", [])
        soft_skills = skills_result.get("soft_skills", [])
        total_skills = len(hard_skills) + len(soft_skills)
        
        print(f"   📊 Hard Skills ({len(hard_skills)}): {hard_skills[:5]}{'...' if len(hard_skills) > 5 else ''}")
        print(f"   📊 Soft Skills ({len(soft_skills)}): {soft_skills[:5]}{'...' if len(soft_skills) > 5 else ''}")
        
        if total_skills > 0:
            self._print_result("Skills", "PASS", f"{total_skills} skills found")
        else:
            self._print_result("Skills", "INFO", "No skills found")
            
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
                
            self._print_result("Working_Experience", "PASS", f"{len(experience)} jobs found")
        else:
            self._print_result("Working_Experience", "INFO", "No jobs found")
            
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
                    
        if extracted_location:
            self._print_result("Location", "PASS", extracted_location)
        else:
            self._print_result("Location", "FAIL", None, "No location extracted")
            
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
            
        if education:
            print(f"   📊 Education entries found: {len(education)}")
            for i, edu in enumerate(education[:3], 1):
                print(f"   {i}. {edu.get('degree', 'N/A')}")
                print(f"      Institution: {edu.get('institution', 'N/A')}")
                print(f"      Dates: {edu.get('dates', 'N/A')}")
                
            self._print_result("School_University", "PASS", f"{len(education)} entries")
        else:
            self._print_result("School_University", "INFO", "No education found")
            
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
                
            self._print_result("Language", "PASS", f"{len(languages)} languages")
        else:
            self._print_result("Language", "INFO", "No languages found")

        self.extracted_data["Language"] = languages[0]['language'] if languages else None
        
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