"""
🧚‍♀️✨ FAIRY CODEMOTHER'S COMPREHENSIVE EXTRACTION TEST SUITE ✨🧚‍♀️

This script tests ALL field extractions for Singapore-style resumes!
It includes the FIXES for:
- Language (from header area, not section)
- Skills (nested categories like "Organisation Skills", "Interpersonal Skills")
- Working Experience (date-first format with TABS)
- Education (date-first format)
- Location/Nationality
- Additional Qualifications (CMFAS certifications)

Run this to verify your extraction fixes are working, honey! 💅
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

# Setup logging with Fairy Codemother flair! ✨
logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🧚 %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 📝 TEST RESUME TEXT (Singapore-style with ALL the quirks!)
# =============================================================================
TEST_RESUME_TEXT = """
Source: merlion_resumes\\43234_Seng Sue Huang Claudia\\Claudia Seng.docx
File Type: DOCX
Images Found: 0
Extraction Method: python-docx
================================================================================

Seng Sue Huang Claudia, 2016 Resume. 
PERSONAL INFORMATION
Name		: 	Claudia Seng
Address		: 	Hougang
Date of Birth	: 	16 Nov 1994
Nationality	: 	Singaporean
Gender		: 	Female
Race		:	Chinese
Mobile Number: 9384 8277
Email: Claudia_seng@hotmail.com
EDUCATIONAL BACKGROUND
2015 - Currently
Kaplan
Diploma in HR

2007-2011
GCE O Level
Ang Mo Kio Secondary School
WORK EXPERIENCE
May 2016 – Aug 2016
Keyence Singapore Pte Ltd
Marketing administrative 
Help in cold callings to update clients data base 
Engage client to learn about the new products and exhibitions
Outsourcing of new company and person in contact for sales purposes 
Reason of leaving: End of contract due to personal reasons.
Last drawn: $2000
June 2015 - Jan 2016
Collective Accounting Pte. LTd
Personal Assistant / Secretary
Assisting the director with their meeting
Assisting in the replying of the appointment
Basic data entry
Handle phone calls
Filling and administrative jobs
Reason of leaving:	Looking for better prospect 
Last drawn:$1900
Jan 2015 - Apr 2015
Mondania 
Personal Assistant / Secretary
Assisting the director with their meeting
Assisting in the replying of the appointment
Assisting in the leasing and inventory list
Basic data entry
Handle phone calls
Filling and administrative jobs
Reason of leaving:	Due to the requirement for daily OT.
Last drawn:$10/hr
Aug 2012- Dec 2014
3e Accounting Pte Ltd
Admin Assistant Cum Receptionist
booking of meeting room 
Assisting in the data entry
handling in walk in customers
liaising with oversea clients reply email
Print and prepare contracts for services.
Reason of leaving:	Seeking for a new opportunity.
Last draw $1900
Feb 2012 - Jul 2012
DP Group Pte Ltd
Admin
meeting of clients
liaising with suppliers to get materials
prepare quotations
Ad hoc data entry.
Reason of leaving:Temporary Contract Positon.
Last drawn $1700
Nov 2011 – Jan 2012
Customer service at changi airport 
Assist tourist with gst refunds 
Working with immigration office 
Reason for leaving: IRAS decides to do away with assistants 
Last drawn $1500
COMPUTER SKILLS
MS Word
Excel
Power Point 
LANGUAGE SKILLS
Spoken	: English, Mandarin
Written	: English, Chinese 
REMUNERATION PACKAGE
Last Drawn Salary	: S$ 2000/month (Keyence Singapore)
Expected Salary		: S$ 2200 negotiable 
Availability	:   19th September 2016
"""


# =============================================================================
# 🎯 EXTRACTION FUNCTIONS (THE FIXES!)
# =============================================================================

def clean_text_for_extraction(text: str) -> str:
    """
    🧹 Clean resume text BEFORE extraction!
    Fixes encoding issues that break regex patterns - often THE reason why extraction fails!
    """
    if not text:
        return ""
    
    # Fix common encoding garbage
    replacements = {
        '\u00a0': ' ',      # Non-breaking space → regular space
        '\u2028': '\n',     # Line separator
        '\u2029': '\n',     # Paragraph separator
        '\u200b': '',       # Zero-width space (REMOVE)
        '\u200c': '',       # Zero-width non-joiner
        '\u200d': '',       # Zero-width joiner
        '\ufeff': '',       # BOM (REMOVE)
        '–': '-',           # En-dash → hyphen
        '—': '-',           # Em-dash → hyphen
        ''': "'",           # Smart quote → regular
        ''': "'",           # Smart quote → regular
        '"': '"',           # Smart quote → regular
        '"': '"',           # Smart quote → regular
        '•': '* ',          # Bullet → asterisk
        '●': '* ',          # Bullet
        '○': '* ',          # Bullet
        '■': '* ',          # Bullet
        '□': '* ',          # Bullet
        '▪': '* ',          # Bullet
        '►': '* ',          # Bullet
        '　': ' ',          # Full-width space → regular
        '\r\n': '\n',       # Windows line endings
        '\r': '\n',         # Old Mac line endings
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize multiple spaces (but preserve newlines and tabs)
    text = re.sub(r'[ ]+', ' ', text)
    
    # Normalize multiple newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()


def extract_name(text: str) -> Optional[str]:
    """
    👤 Extract candidate name from resume
    Singapore resumes often have name at the VERY TOP
    """
    # Clean text first
    text = clean_text_for_extraction(text)

    # Strategy 1 (PRIORITY): Look for "Name:" pattern FIRST
    name_patterns = [
        # Handle "Name: First Last" with possible spaces in between
        r'(?:^|[\n\s])Name[\s:]+([A-Z][A-Za-z]+(?:\s+(?:Binte|Bin|S/O|D/O|s/o|d/o)?)?(?:\s+[A-Z][A-Za-z]+){1,4})',
        r'(?:Full Name|Candidate Name)[\s:]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})',
    ]

    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Skip if it's clearly metadata
            if any(x in name.lower() for x in ['file', 'type', 'docx', 'pdf', 'source']):
                continue
            # Clean up: remove anything after newline or common trailing words
            name = name.split('\n')[0].strip()
            name = re.split(r'\s+(?:Add|Address|Email|Phone|HP|NRIC)', name, flags=re.IGNORECASE)[0].strip()
            logger.info(f"👤 Found name via pattern: {name}")
            return name

    # Strategy 2: First line is often the name (fallback)
    lines = text.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()

        # Skip empty lines
        if not line or len(line) < 3:
            continue

        # Skip metadata lines
        if any(x in line.lower() for x in ['source:', 'file type:', 'images found:', 'extraction method:', '===', 'docx', 'pdf']):
            continue

        # Skip lines with contact info
        if any(x in line.lower() for x in ['@', 'email', 'phone', 'hp:', 'tel:', 'address', 'http']):
            continue

        # Skip lines with common headers
        if any(x in line.lower() for x in ['resume', 'curriculum vitae', 'cv', 'profile']):
            continue

        # Check if it looks like a name (2-5 words, title case or upper case)
        words = line.split()
        if 2 <= len(words) <= 5:
            # All words should be capitalized
            if all(word[0].isupper() for word in words if len(word) > 1):
                # Not too long (likely a sentence if > 50 chars)
                if len(line) <= 50:
                    logger.info(f"👤 Found name: {line}")
                    return line

    return None


def extract_email(text: str) -> Optional[str]:
    """
    📧 Extract email address - BULLETPROOF extraction!
    Handles malformed emails with spaces like "john doe @gmail.com"
    """
    # Search in the first part of the resume (header area)
    header_text = text[:3000] if len(text) > 3000 else text

    # Strategy 1: Look for Email: label pattern (handles spaces in address)
    email_label_patterns = [
        r'[Ee]mail[\s:]+([a-zA-Z0-9._%+-]+(?:\s+)?@(?:\s+)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'[Ee]-?mail[\s:]+([^\s@]+(?:\s+)?@(?:\s+)?[^\s]+\.[a-zA-Z]{2,})',
    ]

    for pattern in email_label_patterns:
        match = re.search(pattern, header_text)
        if match:
            email = match.group(1).strip()
            # Remove spaces around @
            email = re.sub(r'\s*@\s*', '@', email)
            # Remove spaces in username part
            email = re.sub(r'\s+', '', email)
            if '@' in email and '.' in email:
                logger.info(f"📧 Found email via label: {email}")
                return email

    # Strategy 2: Standard email regex pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    matches = re.findall(email_pattern, header_text)

    if matches:
        # Filter out common non-personal emails
        excluded_domains = ['example.com', 'test.com', 'email.com', 'domain.com']
        for email in matches:
            email_lower = email.lower()
            # Skip placeholder/example emails
            if any(domain in email_lower for domain in excluded_domains):
                continue
            # Skip if it looks like a company support email
            if 'support@' in email_lower or 'info@' in email_lower or 'noreply@' in email_lower:
                continue
            logger.info(f"📧 Found email: {email}")
            return email

    # Strategy 3: Try to find email with spaces and reconstruct
    # Pattern: "something @ domain.com" or "some thing@domain.com"
    spaced_email = re.search(r'([a-zA-Z0-9._%-]+(?:\s+[a-zA-Z0-9._%-]+)*)\s*@\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', header_text)
    if spaced_email:
        username = spaced_email.group(1).replace(' ', '')
        domain = spaced_email.group(2)
        email = f"{username}@{domain}"
        logger.info(f"📧 Found email (reconstructed): {email}")
        return email

    return None


def extract_phone(text: str) -> Optional[str]:
    """
    📱 Extract phone number - INTERNATIONAL support with Singapore focus!

    Handles:
    - Singapore: +65 9XXX XXXX, 9XXX XXXX (8 digits), HP: 91234567
    - Home/Mobile split: "91234567 (Mobile) / 67891234 (Home)"
    - Malformed: "Hp No : 90 927602" (spaces in number)
    """
    header_text = text[:2000] if len(text) > 2000 else text

    phone_patterns = [
        # 🆕 Pattern 0: "Hp No : 90 927602" or "Hp No: 90927602" (Singapore with spaces in number!)
        (r'(?:Hp\s*No|HP|H/P|Handphone|Mobile|Cell|Phone|Tel)[\s.:]+(\d[\d\s]{6,12}\d)', 'SG_SPACED'),

        # Pattern 1: Singapore HP with labels like "91234567 (Mobile) / 67891234 (Home)"
        (r'(?:HP|Phone|Tel|Mobile|Contact)[\s.:]*(\d{8}[\s]*\([^)]+\)(?:\s*/\s*\d{8}[\s]*\([^)]+\))?)', 'SG_LABELED'),

        # Pattern 2: HP: 91234567 (simple Singapore format)
        (r'(?:HP|H/P|Handphone|Mobile|Cell)[\s.:]*\+?65[\s.-]?([689]\d{3}[\s.-]?\d{4})', 'SG_WITH_CODE'),
        (r'(?:HP|H/P|Handphone|Mobile|Cell)[\s.:]*([689]\d{7})', 'SG_LOCAL'),

        # Pattern 3: +65 9123 4567 or +65 91234567
        (r'\+65[\s.-]?([689]\d{3}[\s.-]?\d{4})', 'SG_INTL'),

        # Pattern 4: Local SG without label (8 digits starting with 6, 8, or 9)
        (r'(?<!\d)([689]\d{3}[\s.-]?\d{4})(?!\d)', 'SG_PLAIN'),

        # Pattern 5: General labeled phone
        (r'(?:Phone|Tel|Mobile|Contact)[\s.:]*\+?([\d\s.()\-]{8,20})', 'GENERAL'),
    ]

    for pattern, pattern_type in phone_patterns:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            phone = match.group(1).strip()

            # Clean up - remove internal spaces for validation
            digits_only = re.sub(r'\D', '', phone)

            # Validate - must have exactly 8 digits for Singapore
            if len(digits_only) == 8 and digits_only[0] in '689':
                # Format nicely: XXXX XXXX
                formatted = f"{digits_only[:4]} {digits_only[4:]}"
                logger.info(f"📱 Found phone ({pattern_type}): {formatted}")
                return formatted
            elif len(digits_only) >= 8:
                logger.info(f"📱 Found phone ({pattern_type}): {phone}")
                return phone

    return None


def extract_date_of_birth(text: str) -> Optional[str]:
    """
    🎂 Extract date of birth - handles Singapore format!
    
    Formats supported:
    - "Date of Birth : 13 Oct 1989" (Singapore style with TAB!)
    - "DOB: 13/10/1989"
    - "Born: October 13, 1989"
    
    Returns: YYYY-MM-DD format
    """
    # Focus on header area
    header_text = text[:2000] if len(text) > 2000 else text
    
    # Current year for age validation
    current_year = datetime.now().year
    min_birth_year = current_year - 70  # Max age 70
    max_birth_year = current_year - 18  # Min age 18
    
    # Month mapping
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9, 
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
    }
    
    dob_patterns = [
        # 🆕 Pattern 1: "Date of Birth : 13 Oct 1989" (Singapore style with TAB or spaces!)
        (r'(?:Date\s*of\s*Birth|DOB|D\.O\.B\.?|Birth\s*Date)[\s:\t]+(\d{1,2})[\s\-/]+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\-/]+(\d{4})', 'dmy_month_name'),
        
        # Pattern 2: "13 Oct 1989" standalone
        (r'(?<!\d)(\d{1,2})[\s\-/]+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[\s\-/]+(\d{4})(?!\d)', 'dmy_month_name_plain'),
        
        # Pattern 3: With label - DD/MM/YYYY
        (r'(?:Date\s*of\s*Birth|DOB|D\.O\.B\.?|Birth\s*Date)[\s:\t]+(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', 'dmy_labeled'),
        
        # Pattern 4: YYYY-MM-DD (ISO format)
        (r'(?:Date\s*of\s*Birth|DOB|D\.O\.B\.?|Birth\s*Date)[\s:\t]+(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', 'ymd_labeled'),
        
        # Pattern 5: Month DD, YYYY (US style)
        (r'(?:Date\s*of\s*Birth|DOB|Born)[\s:\t]+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})', 'mdy_written'),
    ]
    
    for pattern, date_format in dob_patterns:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            try:
                if date_format in ['dmy_month_name', 'dmy_month_name_plain']:
                    day = int(match.group(1))
                    month_name = match.group(2).lower()[:3]
                    year = int(match.group(3))
                    month = month_map.get(month_name, 0)
                    
                elif date_format == 'dmy_labeled':
                    day = int(match.group(1))
                    month = int(match.group(2))
                    year = int(match.group(3))
                    
                elif date_format == 'ymd_labeled':
                    year = int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))
                    
                elif date_format == 'mdy_written':
                    month_name = match.group(1).lower()[:3]
                    day = int(match.group(2))
                    year = int(match.group(3))
                    month = month_map.get(month_name, 0)
                
                else:
                    continue
                
                # Validate year range
                if min_birth_year <= year <= max_birth_year:
                    # Validate date is real
                    dt = datetime(year, month, day)
                    formatted = dt.strftime('%Y-%m-%d')
                    logger.info(f"🎂 Found DOB ({date_format}): {formatted}")
                    return formatted
                    
            except (ValueError, OverflowError) as e:
                logger.debug(f"Date parsing failed: {e}")
                continue
    
    return None


def extract_language_from_header(text: str) -> Optional[str]:
    """
    🗣️ FAIRY CODEMOTHER'S LANGUAGE EXTRACTOR! 💬

    THE FIX: Language can be ANYWHERE in resume, not just header!

    Captures:
    - "Language : English & Chinese"
    - "Languages: English, Mandarin"
    - "languages Reads, writes & speaks fluent English & Malay"
    - "Language Proficiency: English (Native), Chinese (Fluent)"
    """
    # Search ENTIRE text, not just header
    language_patterns = [
        # Pattern 1: "languages Reads, writes & speaks fluent English & Malay"
        r'[Ll]anguages?\s+(?:Reads?,?\s*writes?\s*(?:&|and)\s*speaks?\s*(?:fluent\s+)?)?([A-Za-z]+(?:\s*(?:&|and|,)\s*[A-Za-z]+)*)',

        # Pattern 2: "Language : English & Chinese" (with TAB or colon!)
        r'(?:Language|Languages?)[\s:\t]+([A-Za-z]+(?:\s*(?:&|and|,)\s*[A-Za-z]+)+)',

        # Pattern 3: Language Proficiency: ...
        r'(?:Language\s+Proficiency|Spoken\s+Languages?|Languages?\s+Spoken)[\s:\t]+([^\n]+)',

        # Pattern 4: Multiple languages with proficiency levels
        r'(?:Language|Languages?)[\s:\t]+([A-Za-z]+(?:\s*[\(&][^)\n]+[\)&])?\s*(?:[,&]\s*[A-Za-z]+(?:\s*\([^)\n]+\))?)*)',
    ]

    for pattern in language_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            language = match.group(1).strip()

            # Clean up common suffixes and extra whitespace
            language = re.sub(r'\s+', ' ', language)
            language = language.rstrip(',;.')

            # Skip if it looks like a section header captured accidentally
            if language.lower() in ['work', 'education', 'skills', 'experience', 'others']:
                continue

            if len(language) > 2 and len(language) < 100:
                logger.info(f"🗣️ Found language: {language}")
                return language

    return None


def extract_nationality(text: str) -> Optional[str]:
    """
    🏳️ Extract nationality from header area

    Singapore resumes often have "Nationality : Singaporean"
    Handles broken words like "Singapo rean" -> "Singaporean"
    """
    header_text = text[:3000] if len(text) > 3000 else text

    # Known nationalities with possible OCR breaks
    nationality_fixes = {
        'singapo': 'Singaporean',
        'singapor': 'Singaporean',
        'singapo rean': 'Singaporean',
        'singaporean': 'Singaporean',
        'malaysia': 'Malaysian',
        'malaysian': 'Malaysian',
        'indonesian': 'Indonesian',
        'filipino': 'Filipino',
        'filipina': 'Filipino',
        'indian': 'Indian',
        'chinese': 'Chinese',
        'american': 'American',
        'british': 'British',
    }

    nationality_patterns = [
        # Pattern with possible broken words (up to 2 words after label)
        r'(?:Nationality|Citizen(?:ship)?)[\s:\t]+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
        r'(?:National\s+of)[\s:\t]?\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)',
    ]

    for pattern in nationality_patterns:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            nationality = match.group(1).strip().lower()

            # Try to fix broken nationality words
            for broken, fixed in nationality_fixes.items():
                if broken in nationality or nationality.startswith(broken[:6]):
                    logger.info(f"🏳️ Found nationality (fixed): {fixed}")
                    return fixed

            # If not a known fix, return as-is with title case
            if len(nationality) > 2 and len(nationality) < 50:
                logger.info(f"🏳️ Found nationality: {nationality.title()}")
                return nationality.title()

    return None


def extract_location(text: str) -> Optional[str]:
    """
    📍 Extract location/address

    Singapore format: "Address: Blk 123 Ang Mo Kio Ave 4 #12-345, Singapore 560123"
    Handles broken words like "Add ress" -> "Address"
    """
    header_text = text[:3000] if len(text) > 3000 else text

    location_patterns = [
        # Singapore HDB address with block number (handles broken "Add ress")
        r'(?:Add\s*ress|Address|Location|Residence)[\s:\t]+(Blk\s*\d+[^\n]*(?:#\d+-\d+)?[^\n]*S\s*\(\d+\s*\))',

        # Singapore address with postal code S(XXXXXX)
        r'(?:Add\s*ress|Address|Location|Residence)[\s:\t]+([^\n]*S\s*\(\s*\d{6}\s*\))',

        # Singapore address with block number
        r'(?:Add\s*ress|Address|Location|Residence)[\s:\t]+([^\n]+(?:Singapore|SG)[\s\d]*)',

        # Any labeled address with Blk
        r'(?:Add\s*ress|Address|Location)[\s:\t]+(Blk[^\n]+)',

        # Any labeled address
        r'(?:Add\s*ress|Address|Location)[\s:\t]+([^\n]+)',

        # City, Country pattern
        r'(?:City|Location)[\s:\t]+([A-Za-z\s]+,\s*[A-Za-z\s]+)',
    ]

    for pattern in location_patterns:
        match = re.search(pattern, header_text, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Clean up extra spaces
            location = re.sub(r'\s+', ' ', location)
            # Stop at first label-like word (Hp, Email, Phone, etc.)
            location = re.split(r'\s+(?:Hp|Email|Phone|Tel|NRIC|Race|Nationality)', location, flags=re.IGNORECASE)[0]
            location = location.strip()

            if len(location) > 5 and len(location) < 200:
                logger.info(f"📍 Found location: {location[:50]}...")
                return location

    # Fallback: Derive from nationality
    nationality = extract_nationality(text)
    if nationality:
        country_map = {
            'singaporean': 'Singapore',
            'malaysian': 'Malaysia',
            'indonesian': 'Indonesia',
            'filipino': 'Philippines',
            'indian': 'India',
            'chinese': 'China',
            'american': 'United States',
            'british': 'United Kingdom',
        }
        location = country_map.get(nationality.lower())
        if location:
            logger.info(f"📍 Derived location from nationality: {location}")
            return location

    return None


def extract_experience_date_first_format(text: str) -> List[Dict]:
    """
    💼 FAIRY CODEMOTHER'S FLEXIBLE EXPERIENCE EXTRACTOR! 💅

    Handles MULTIPLE formats:
    1. Date-first: "Feb 2016 to Present\tFinancial Consultant, Company"
    2. Role-first: "Full-Time Sales Assistant| Company June 2015 – Aug 2016"
    3. Inline blob: "Experience Full-Time Sales Assistant| Company June 2015 – Aug 2016 Description..."
    """
    jobs = []

    # Clean the text first
    text = clean_text_for_extraction(text)

    # Education keywords to filter out
    education_keywords = [
        'diploma', 'degree', 'bachelor', 'master', 'phd', 'certificate',
        'gce', 'o level', 'a level', 'n level', 'psle', 'nitec', 'ite',
        'polytechnic', 'university', 'college', 'school', 'institute',
        'secondary', 'primary', 'junior college', 'jc', 'retailing and the economy'
    ]

    # Try multiple extraction strategies

    # Strategy 1: Look for "Experience" markers followed by role|company date format
    # Pattern: "Experience Full-Time Sales Assistant| Company Month Year – Month Year"
    role_first_pattern = r'(?:[Ee]xperience\s+)?(?:Full-?Time|Part-?Time|Contract|Temp)?\s*([A-Za-z\s]+?)[\s|]+([A-Za-z][A-Za-z\s().,&]+?(?:Pte|Ltd|Inc|Corp|Company|Boutique)?[^A-Za-z]*?)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current)'

    matches = list(re.finditer(role_first_pattern, text, re.IGNORECASE))
    logger.info(f"💼 Found {len(matches)} role-first format entries")

    for match in matches:
        role = match.group(1).strip()
        company = match.group(2).strip()
        start_date = match.group(3).strip()
        end_date = match.group(4).strip()

        # Skip if looks like education
        combined = f"{role} {company}".lower()
        if any(edu_kw in combined for edu_kw in education_keywords):
            continue

        # Clean up role and company
        role = re.sub(r'\s+', ' ', role).strip()
        company = re.sub(r'\s+', ' ', company).strip()

        # Skip if role is too short or generic
        if len(role) < 3 or role.lower() in ['experience', 'full-time', 'part-time']:
            continue

        dates = f"{start_date} - {end_date}"

        jobs.append({
            "company": company[:200],
            "role": role[:200],
            "dates": dates,
            "description": "See resume for details"
        })
        logger.info(f"✅ Extracted (role-first): {role[:40]} at {company[:40]}")

    # Strategy 2: Try date-first pattern if no matches
    if not jobs:
        date_first_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})[\s\t]+(?:to|[-–—])[\s\t]+(Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})[\s\t]+([^\n]+)'

        matches = list(re.finditer(date_first_pattern, text, re.IGNORECASE | re.MULTILINE))
        logger.info(f"💼 Found {len(matches)} date-first format entries")

        for match in matches:
            start_date = match.group(1).strip()
            end_date = match.group(2).strip()
            role_company = match.group(3).strip()

            # Skip education entries
            if any(edu_kw in role_company.lower() for edu_kw in education_keywords):
                continue

            # Parse role and company
            if ',' in role_company:
                parts = role_company.split(',', 1)
                role = parts[0].strip()
                company = parts[1].strip()
            elif '|' in role_company:
                parts = role_company.split('|', 1)
                role = parts[0].strip()
                company = parts[1].strip()
            else:
                role = role_company
                company = "See description"

            dates = f"{start_date} - {end_date}"

            jobs.append({
                "company": company[:200],
                "role": role[:200],
                "dates": dates,
                "description": "See resume for details"
            })
            logger.info(f"✅ Extracted (date-first): {role[:40]} at {company[:40]}")

    # Remove duplicates based on company+role
    seen = set()
    unique_jobs = []
    for job in jobs:
        key = f"{job['company'].lower()}|{job['role'].lower()}"
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    return unique_jobs


def extract_education_date_first_format(text: str) -> List[Dict]:
    """
    🎓 FAIRY CODEMOTHER'S FLEXIBLE EDUCATION EXTRACTOR! 📚

    Handles MULTIPLE formats:
    1. Date-first: "Apr 2006 to Apr 2009\tDiploma in Banking, Singapore Polytechnic"
    2. Inline: "Education Singapore Institute of Retail Studies Diploma in Retail Management Oct 2015 – Sept 2016"
    3. Institution Degree Date: "Singapore Polytechnic Diploma in Business Oct 2015 – Sept 2016"
    """
    education = []

    # Clean text
    text = clean_text_for_extraction(text)

    # Singapore institutions for detection
    sg_institutions = [
        'singapore polytechnic', 'ngee ann polytechnic', 'temasek polytechnic',
        'republic polytechnic', 'nanyang polytechnic', 'ite college',
        'singapore institute', 'institute of technical education',
        'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss',
        'secondary school', 'junior college', 'jc',
        'national university', 'nanyang technological',
        'juying primary', 'jurong west secondary'
    ]

    # Degree keywords
    degree_keywords = [
        'diploma', 'degree', 'bachelor', 'master', 'phd', 'doctorate',
        'certificate', 'advanced cert', 'gce', 'o level', 'a level', 'n level',
        'psle', 'nitec', 'higher nitec', 'private candidate'
    ]

    # Strategy 1: Find education entries with dates (Month Year – Month Year format)
    # Pattern: institution/degree text Month Year – Month Year
    edu_with_dates_pattern = r'([A-Z][A-Za-z\s\'-]+(?:Institute|Polytechnic|University|College|School|Studies)?[^A-Za-z]*(?:Diploma|Degree|Certificate|Bachelor|Master|GCE|PSLE|Advanced Cert)[A-Za-z\s\'-]+?)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current)'

    matches = list(re.finditer(edu_with_dates_pattern, text, re.IGNORECASE))
    logger.info(f"🎓 Found {len(matches)} education entries with dates")

    for match in matches:
        edu_text_chunk = match.group(1).strip()
        start_date = match.group(2).strip()
        end_date = match.group(3).strip()

        dates = f"{start_date} - {end_date}"

        # Parse institution and degree from the text chunk
        institution = "Institution not specified"
        degree = "Qualification not specified"

        # Try to identify institution
        edu_lower = edu_text_chunk.lower()
        for inst in sg_institutions:
            if inst in edu_lower:
                # Find the institution name portion
                idx = edu_lower.find(inst)
                # Look for capitalized version
                inst_match = re.search(re.escape(inst), edu_text_chunk, re.IGNORECASE)
                if inst_match:
                    # Extract institution (grab full name)
                    inst_start = inst_match.start()
                    # Find where institution name ends (at degree keyword or end)
                    for deg_kw in degree_keywords:
                        deg_idx = edu_lower.find(deg_kw, inst_start)
                        if deg_idx > inst_start:
                            institution = edu_text_chunk[inst_start:deg_idx].strip()
                            degree = edu_text_chunk[deg_idx:].strip()
                            break
                    else:
                        institution = edu_text_chunk[inst_start:].strip()
                break

        # If no institution found, try degree first
        if institution == "Institution not specified":
            for deg_kw in degree_keywords:
                if deg_kw in edu_lower:
                    deg_idx = edu_lower.find(deg_kw)
                    institution = edu_text_chunk[:deg_idx].strip() if deg_idx > 0 else "See resume"
                    degree = edu_text_chunk[deg_idx:].strip()
                    break

        # Clean up
        institution = re.sub(r'\s+', ' ', institution).strip()
        degree = re.sub(r'\s+', ' ', degree).strip()

        if institution or degree != "Qualification not specified":
            education.append({
                "institution": institution[:250] if institution else "See resume",
                "degree": degree[:500] if degree else "See resume",
                "dates": dates
            })
            logger.info(f"🎓 Extracted: {degree[:50]} from {institution[:30]}")

    # Strategy 2: Try date-first pattern if needed
    if not education:
        date_first_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})[\s\t]+(?:to|[-–—])[\s\t]+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current)[\s\t]+([^\n]+)'

        matches = list(re.finditer(date_first_pattern, text, re.IGNORECASE))

        for match in matches:
            start_date = match.group(1).strip()
            end_date = match.group(2).strip()
            content = match.group(3).strip()

            # Check if it's education (has degree keywords)
            if any(deg_kw in content.lower() for deg_kw in degree_keywords):
                dates = f"{start_date} - {end_date}"
                education.append({
                    "institution": "See resume",
                    "degree": content[:500],
                    "dates": dates
                })
                logger.info(f"🎓 Extracted (date-first): {content[:50]}")

    # Remove duplicates
    seen = set()
    unique_education = []
    for edu in education:
        key = f"{edu['degree'].lower()[:50]}|{edu['dates']}"
        if key not in seen:
            seen.add(key)
            unique_education.append(edu)

    return unique_education


def extract_skills_from_nested_categories(text: str) -> Dict[str, List[str]]:
    """
    🎯 FAIRY CODEMOTHER'S NESTED SKILLS EXTRACTOR! 💅
    
    THE FIX: Handles skill sections with nested categories!
    
    Skills & Abilities
    Organisation Skills
      - Prepared and hosted seminars...
    Interpersonal Skills
      - Applied emotional competence...
    Leadership Skills
      - Motivated and helped...
    """
    hard_skills = []
    soft_skills = []
    
    # Clean text
    text = clean_text_for_extraction(text)
    
    # Find Skills section
    skills_patterns = [
        r'(?:^|\n)\s*(?:SKILLS?\s*(?:&|AND)\s*ABILITIES|SKILLS?)\s*\n(.*?)(?=\n\s*(?:ADDITIONAL\s+QUALIFICATIONS?|CERTIFICATIONS?|REFERENCES?|HOBBIES?|$))',
    ]
    
    skills_text = ""
    for pattern in skills_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            skills_text = match.group(1)
            break
    
    if not skills_text:
        logger.warning("⚠️ No Skills section found")
        return {"hard_skills": [], "soft_skills": []}
    
    logger.info(f"🎯 Skills section: {len(skills_text)} chars")
    
    # Detect nested categories like "Organisation Skills", "Interpersonal Skills"
    category_pattern = r'(?:^|\n)\s*([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)*\s+Skills?)\s*\n'
    categories = list(re.finditer(category_pattern, skills_text, re.IGNORECASE))
    
    if categories:
        logger.info(f"🎯 Found {len(categories)} nested skill categories!")
        
        for i, cat_match in enumerate(categories):
            category_name = cat_match.group(1).strip()
            
            # Get content for this category
            start = cat_match.end()
            if i + 1 < len(categories):
                end = categories[i + 1].start()
            else:
                end = len(skills_text)
            
            category_content = skills_text[start:end]
            
            # Extract skill descriptions
            skill_descriptions = []
            for line in category_content.split('\n'):
                line = line.strip()
                if len(line) > 10:
                    cleaned = re.sub(r'^[\*\-•·►➢○●]?\s*', '', line)
                    if cleaned:
                        skill_descriptions.append(cleaned)
            
            # Categorize based on category name
            category_lower = category_name.lower()
            
            # Soft skill categories
            soft_categories = ['interpersonal', 'leadership', 'organisation', 'organization', 
                               'communication', 'teamwork', 'management', 'personal', 'soft']
            
            # Technical/hard skill categories
            hard_categories = ['technical', 'programming', 'software', 'computer', 
                               'analytical', 'data', 'engineering', 'it', 'digital']
            
            if any(soft in category_lower for soft in soft_categories):
                soft_skills.append(category_name)
                soft_skills.extend(skill_descriptions[:2])
            elif any(hard in category_lower for hard in hard_categories):
                hard_skills.append(category_name)
                hard_skills.extend(skill_descriptions[:2])
            else:
                # Default to soft skills
                soft_skills.append(category_name)
                soft_skills.extend(skill_descriptions[:2])
    else:
        # No nested categories - try flat extraction
        logger.info("🎯 No nested categories, trying flat extraction...")
        
        lines = skills_text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 5:
                cleaned = re.sub(r'^[\*\-•·►➢○●]?\s*', '', line)
                if cleaned:
                    soft_skills.append(cleaned)
    
    # Remove duplicates while preserving order
    hard_skills = list(dict.fromkeys(hard_skills))
    soft_skills = list(dict.fromkeys(soft_skills))
    
    logger.info(f"📊 Skills extracted: {len(hard_skills)} hard, {len(soft_skills)} soft")
    
    return {
        "hard_skills": hard_skills,
        "soft_skills": soft_skills
    }


def extract_certifications(text: str) -> List[str]:
    """
    🏆 Extract certifications including CMFAS modules!
    
    Handles Singapore financial certifications:
    "CMFAS M5, M8, M8A, M9, M9A"
    "HI"
    "CGI – BCP, ComGI, PGI"
    """
    certifications = []
    
    # Find Additional Qualifications section
    cert_patterns = [
        r'(?:^|\n)\s*(?:Additional\s+Qualifications?|Certifications?|Licenses?|Professional\s+Qualifications?)\s*\n(.*?)(?=\n\s*(?:REFERENCES?|HOBBIES?|INTERESTS?|$)|$)',
    ]
    
    cert_text = ""
    for pattern in cert_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            cert_text = match.group(1)
            break
    
    if not cert_text:
        return certifications
    
    logger.info(f"🏆 Certifications section: {len(cert_text)} chars")
    
    # Split by newlines and clean
    lines = cert_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if len(line) < 2:
            continue
        
        # Expand CMFAS modules
        if 'CMFAS' in line.upper() or re.match(r'^M\d+', line):
            # This is a CMFAS line - expand it
            modules = re.findall(r'M\d+[A-Z]?', line, re.IGNORECASE)
            for module in modules:
                certifications.append(f"CMFAS {module.upper()}")
        
        # Known Singapore certifications
        elif any(cert in line.upper() for cert in ['HI', 'CGI', 'BCP', 'COMGI', 'PGI', 'PHI', 'CLI']):
            # Split by common delimiters and add CMFAS prefix
            parts = re.split(r'[,|–\-]', line)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 2:
                    if not part.upper().startswith('CMFAS'):
                        certifications.append(f"CMFAS {part}")
                    else:
                        certifications.append(part)
        else:
            # Other certifications
            if len(line) > 2:
                certifications.append(line)
    
    # Remove duplicates
    certifications = list(dict.fromkeys(certifications))
    
    logger.info(f"🏆 Found {len(certifications)} certifications")
    
    return certifications


def extract_achievements(text: str) -> List[str]:
    """
    🏅 Extract achievements/awards
    """
    achievements = []
    
    # Find Achievements section
    ach_pattern = r'(?:^|\n)\s*(?:Achievements?|Awards?|Honours?|Honors?)\s*\n(.*?)(?=\n\s*(?:Co-?Curricular|Skills|Additional|Education|Experience|$))'
    
    match = re.search(ach_pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return achievements
    
    ach_text = match.group(1)
    logger.info(f"🏅 Achievements section: {len(ach_text)} chars")
    
    # Extract achievements (skip table headers)
    lines = ach_text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) < 5:
            continue
        
        # Skip header lines
        if line.lower() in ['year', 'description', 'award', 'date']:
            continue
        
        # Skip if it's just a year
        if re.match(r'^\d{4}$', line):
            continue
        
        # Clean and add
        cleaned = re.sub(r'^\d{4}\s+', '', line)  # Remove leading year
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if len(cleaned) > 5:
            achievements.append(cleaned)
    
    logger.info(f"🏅 Found {len(achievements)} achievements")
    
    return achievements[:20]  # Limit to 20


# =============================================================================
# 🎭 MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_all_fields(text: str) -> Dict[str, Any]:
    """
    🎭 FAIRY CODEMOTHER'S ULTIMATE EXTRACTION! ✨
    
    Extracts ALL fields from a Singapore-style resume!
    """
    logger.info("=" * 60)
    logger.info("🧚‍♀️ FAIRY CODEMOTHER'S EXTRACTION BEGINS! 🧚‍♀️")
    logger.info("=" * 60)
    
    # Clean text first
    text = clean_text_for_extraction(text)
    
    result = {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Date_of_Birth": extract_date_of_birth(text),
        "Language": extract_language_from_header(text),
        "Nationality": extract_nationality(text),
        "Location": extract_location(text),
        "Working_Experience": None,
        "Education": None,
        "Skills": None,
        "Certifications": None,
        "Achievements": None,
    }
    
    # Extract structured fields
    experience = extract_experience_date_first_format(text)
    if experience:
        # Format for export
        exp_strings = []
        for job in experience:
            exp_strings.append(f"{job['company']} - {job['role']} ({job['dates']})")
        result["Working_Experience"] = " || ".join(exp_strings)
    
    education = extract_education_date_first_format(text)
    if education:
        edu_strings = []
        for edu in education:
            edu_strings.append(f"{edu['degree']} from {edu['institution']} ({edu['dates']})")
        result["Education"] = " || ".join(edu_strings)
    
    skills = extract_skills_from_nested_categories(text)
    if skills['hard_skills'] or skills['soft_skills']:
        all_skills = []
        if skills['soft_skills']:
            all_skills.append("SOFT: " + " | ".join(skills['soft_skills'][:10]))
        if skills['hard_skills']:
            all_skills.append("TECHNICAL: " + " | ".join(skills['hard_skills'][:10]))
        result["Skills"] = " || ".join(all_skills)
    
    certifications = extract_certifications(text)
    if certifications:
        result["Certifications"] = " | ".join(certifications)
    
    achievements = extract_achievements(text)
    if achievements:
        result["Achievements"] = " | ".join(achievements[:5])
    
    return result


def format_result_for_display(result: Dict) -> str:
    """
    🎨 Format the extraction result for pretty display
    """
    output = []
    output.append("\n" + "=" * 70)
    output.append("🎭 EXTRACTION RESULTS 🎭".center(70))
    output.append("=" * 70)
    
    # Define display order and emojis
    field_config = [
        ("Name", "👤"),
        ("Email", "📧"),
        ("Phone", "📱"),
        ("Date_of_Birth", "🎂"),
        ("Language", "🗣️"),
        ("Nationality", "🏳️"),
        ("Location", "📍"),
        ("Working_Experience", "💼"),
        ("Education", "🎓"),
        ("Skills", "🎯"),
        ("Certifications", "🏆"),
        ("Achievements", "🏅"),
    ]
    
    for field, emoji in field_config:
        value = result.get(field)
        if value:
            # Truncate long values for display
            display_value = str(value)
            if len(display_value) > 100:
                display_value = display_value[:97] + "..."
            output.append(f"\n{emoji} {field}:")
            output.append(f"   {display_value}")
        else:
            output.append(f"\n❌ {field}: NOT FOUND")
    
    output.append("\n" + "=" * 70)
    
    return "\n".join(output)


def calculate_extraction_score(result: Dict) -> Tuple[int, int, float]:
    """
    📊 Calculate extraction success score
    """
    critical_fields = ["Name", "Email", "Phone", "Date_of_Birth"]
    important_fields = ["Language", "Location", "Working_Experience", "Education", "Skills"]
    
    critical_found = sum(1 for f in critical_fields if result.get(f))
    important_found = sum(1 for f in important_fields if result.get(f))
    
    total_fields = len(critical_fields) + len(important_fields)
    total_found = critical_found + important_found
    
    score = (total_found / total_fields) * 100
    
    return critical_found, important_found, score


# =============================================================================
# 🏃 RUN THE TEST!
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🧚" * 35)
    print("🧚‍♀️✨ FAIRY CODEMOTHER'S EXTRACTION TEST SUITE ✨🧚‍♀️")
    print("🧚" * 35)
    
    # Run extraction
    result = extract_all_fields(TEST_RESUME_TEXT)
    
    # Display results
    print(format_result_for_display(result))
    
    # Calculate score
    critical, important, score = calculate_extraction_score(result)
    
    print("\n📊 EXTRACTION SCORE:")
    print(f"   Critical fields found: {critical}/4")
    print(f"   Important fields found: {important}/5")
    print(f"   Overall score: {score:.1f}%")
    
    if score >= 80:
        print("\n✨ FABULOUS! The extraction is working beautifully! 💅")
    elif score >= 60:
        print("\n👍 Good progress, but some fields need attention, sweetie!")
    else:
        print("\n⚠️ Uh oh honey, we need to fix some things!")
    
    # Export to JSON
    output_file = "extraction_test_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    print("\n" + "🧚" * 35)