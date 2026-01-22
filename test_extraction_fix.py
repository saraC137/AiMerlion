# test_extraction_fix.py
"""
🧚‍♀️ Quick test to verify the extraction fixes!
"""

test_text = """
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
Jan 2002 to Dec 2005	GCE ‘O’ Level
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

from ai_extractor import AIExtractor

# Test the extractor
extractor = AIExtractor("llama3.1-128k")

# Test language extraction
print("🗣️ Testing language extraction...")
languages = extractor._extract_languages_regex(test_text)
print(f"   Languages found: {len(languages)}")
for lang in languages:
    print(f"   - {lang['language']}{' (' + lang['proficiency'] + ')' if lang['proficiency'] else ''}")

# Test experience extraction
print("\n💼 Testing experience extraction...")
exp = extractor._extract_experience_date_first_format(test_text)
print(f"   Jobs found: {len(exp)}")
for job in exp:
    print(f"   - {job['role']} at {job['company']}")

# Test skills extraction
print("\n🎯 Testing skills extraction...")
skills = extractor._extract_skills_regex(test_text)
print(f"   Hard skills: {skills['hard_skills']}")
print(f"   Soft skills: {skills['soft_skills']}")

print("\n✨ Test complete!")