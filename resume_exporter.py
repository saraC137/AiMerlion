"""
resume_exporter.py

💅✨ FAIRY CODEMOTHER'S RESUME DATA EXPORTER ✨💅

Exports candidate data from resume_extractions.db into your desired
JSON and CSV formats — ready for ATS integration, ML training,
or handoff to recruitment platforms! 🎭📤

This is the GRAND FINALE of the data pipeline — all that extraction
work gets packaged into a beautiful, standardized format like a 
contestant stepping onto the runway in their FINAL look! 👗✨

Output Formats:
  1. 📋 JSON  — Individual + batch export matching your exact schema
  2. 📊 CSV   — Flat table for Excel/Google Sheets (experience & education flattened)
  3. 🧠 ML    — Training-ready JSONL with structured labels for model fine-tuning

Field Mapping (DB → Your Schema):
  ┌─────────────────────────┬───────────────────────────┐
  │ YOUR FIELD              │ DB SOURCE                 │
  ├─────────────────────────┼───────────────────────────┤
  │ ID                      │ candidate_id              │
  │ Name                    │ name                      │
  │ Phone                   │ phone                     │
  │ Email                   │ email                     │
  │ Current Company         │ experience_json[0].company│
  │ Current Title           │ experience_json[0].role   │
  │ Current Location        │ location                  │
  │ Summary                 │ summary                   │
  │ Language Skills         │ languages                 │
  │ Work Experience[]       │ experience_json (parsed)  │
  │   .company              │   .company                │
  │   .title                │   .role                   │
  │   .from / .to           │   .dates (split)          │
  │   .responsibility[]     │   .description (split)    │
  │ Education[]             │ education_json (parsed)   │
  │   .school               │   .institution            │
  │   .major                │   .degree (parsed)        │
  │   .Degree               │   .degree                 │
  │ Project Experience      │ projects                  │
  │ tags[]                  │ skills_json (parsed)      │
  └─────────────────────────┴───────────────────────────┘

  Fields not in DB (exported as empty strings for your manual fill):
  Team, Expected Location, Gender, Last Contact

  Fields AUTO-CLASSIFIED from resume content:
  Function (17 categories), Industry (24 categories), Location (normalized)

Usage:
    python resume_exporter.py                          # Export all to JSON + CSV
    python resume_exporter.py --format json            # JSON only
    python resume_exporter.py --format csv             # CSV only
    python resume_exporter.py --format ml              # ML training JSONL
    python resume_exporter.py --id 12345               # Single candidate
    python resume_exporter.py --status Complete        # Filter by status
    python resume_exporter.py --reviewed-only          # Only reviewed candidates
    python resume_exporter.py --output exports/        # Custom output directory

Dependencies:
    pip install pandas --break-system-packages  (CSV export only)
"""

import sqlite3
import json
import csv
import os
import re
import sys
import datetime
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

# Default database path (override with --db flag or env var)
DATABASE_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")

# Default output directory
OUTPUT_DIR = "exports"

# System identifier for "Created By" field
SYSTEM_NAME = "AiMerlion Resume Extractor"


# =============================================================================
# 🧹 DATA PARSERS — Transform DB format → Your Schema format
# =============================================================================

def parse_dates_to_from_to(date_string: str) -> Tuple[str, str]:
    """
    📅 Split a date range string into 'from' and 'to' components.
    
    Your DB stores dates like "Feb 2016 to Present" or "2018 - 2022"
    in a single string. Your JSON schema needs separate from/to fields.
    
    Like splitting a dance duet into two solo performances! 💃🕺
    
    Handles:
      "Feb 2016 to Present"       → ("Feb 2016", "Present")
      "Jan 2020 – Dec 2023"       → ("Jan 2020", "Dec 2023")
      "2018 - 2022"               → ("2018", "2022")
      "Since 2019"                → ("2019", "Present")
      "Mar 2015 till Jul 2018"    → ("Mar 2015", "Jul 2018")
      "2020"                      → ("2020", "")
      "Dates not specified"       → ("", "")
      None / empty                → ("", "")
    """
    if not date_string or date_string.strip().lower() in ('n/a', 'dates not specified', 'unknown', ''):
        return "", ""
    
    date_string = date_string.strip()
    
    # Handle "Since XXXX" format
    since_match = re.match(r'^[Ss]ince\s+(.+)$', date_string)
    if since_match:
        return since_match.group(1).strip(), "Present"
    
    # Common separators between from and to dates
    separators = [
        r'\s+to\s+',           # "to"
        r'\s*–\s*',            # en-dash "–"
        r'\s*—\s*',            # em-dash "—"
        r'\s*-\s+',            # hyphen with space (avoid splitting "2020-01")
        r'\s+till\s+',         # "till"
        r'\s+until\s+',        # "until"
        r'\s+through\s+',      # "through"
    ]
    
    for sep in separators:
        parts = re.split(sep, date_string, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            from_date = parts[0].strip()
            to_date = parts[1].strip()
            return from_date, to_date
    
    # Single date (no range) — just the "from"
    # Check if it looks like a year or month+year
    if re.match(r'^(?:\w+\s+)?\d{4}$', date_string):
        return date_string, ""
    
    # Fallback: return as-is in the "from" field
    return date_string, ""


def parse_description_to_responsibilities(description: str) -> List[str]:
    """
    📝 Split a job description string into a list of individual responsibilities.
    
    Your DB stores descriptions as one big text block. Your JSON schema
    wants an array of responsibility strings.
    
    Like turning a monologue into a highlight reel! 🎭✂️
    
    Splitting strategy (in priority order):
      1. Split on bullet characters (•, ▪, ▸, ►, ◆, ■, ★, ✓, ➤)
      2. Split on dash bullets (lines starting with "- ")
      3. Split on asterisk bullets ("* ")
      4. Split on numbered items ("1. ", "2. ")
      5. Split on newlines (if lines are reasonably long)
      6. Return as single item if no clear structure
    """
    if not description or not description.strip():
        return []
    
    description = description.strip()
    
    # Strategy 1: Bullet characters
    bullet_chars = r'[•▪▸▹►◆◇○●■□★✓✔➤➢⁃·]'
    if re.search(bullet_chars, description):
        items = re.split(bullet_chars, description)
        items = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]
        if len(items) >= 2:
            return items
    
    # Strategy 2: Dash bullets at line start
    if re.search(r'(?:^|\n)\s*[-–—]\s+\S', description):
        items = re.split(r'(?:^|\n)\s*[-–—]\s+', description)
        items = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]
        if len(items) >= 2:
            return items
    
    # Strategy 3: Asterisk bullets
    if re.search(r'(?:^|\n)\s*\*\s+\S', description):
        items = re.split(r'(?:^|\n)\s*\*\s+', description)
        items = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]
        if len(items) >= 2:
            return items
    
    # Strategy 4: Numbered items
    if re.search(r'(?:^|\n)\s*\d+[.)]\s+\S', description):
        items = re.split(r'(?:^|\n)\s*\d+[.)]\s+', description)
        items = [item.strip() for item in items if item.strip() and len(item.strip()) > 5]
        if len(items) >= 2:
            return items
    
    # Strategy 5: Newlines (if resulting items are reasonably long)
    lines = [l.strip() for l in description.split('\n') if l.strip()]
    if len(lines) >= 2 and all(len(l) > 10 for l in lines):
        return lines
    
    # Strategy 6: Sentence splitting on period (for dense paragraphs)
    sentences = re.split(r'(?<=[.!])\s+(?=[A-Z])', description)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    if len(sentences) >= 2:
        return sentences
    
    # Fallback: return as single item
    return [description] if len(description) > 5 else []


def parse_education_degree_to_major(degree_string: str) -> Tuple[str, str]:
    """
    🎓 Extract major/field of study from a degree string.
    
    "Bachelor of Science in Computer Science" → ("Computer Science", "Bachelor of Science in Computer Science")
    "Diploma in Information Technology"       → ("Information Technology", "Diploma in Information Technology")
    "MBA"                                     → ("", "MBA")
    
    Like reading the fine print on a designer label! 🏷️
    """
    if not degree_string:
        return "", ""
    
    degree_string = degree_string.strip()
    
    # Pattern 1: "Degree of X in Major" (e.g., "Bachelor of Science in Computer Science")
    # Must check 'in' BEFORE 'of' for compound degrees
    in_of_match = re.search(
        r'(?:Bachelor|Master|Doctor|PhD|Diploma|Certificate|NITEC|Higher NITEC|GCE|Advanced Diploma)'
        r'\s+(?:of\s+[\w\s]+?\s+)?in\s+(.+?)(?:\s*[-–,;(]|$)',
        degree_string, re.IGNORECASE
    )
    if in_of_match:
        major = in_of_match.group(1).strip()
        major = re.sub(r'\s+(?:from|at|with|and|&)\s*$', '', major, flags=re.IGNORECASE)
        return major, degree_string

    # Pattern 2: "Degree of Major" without 'in' (e.g., "Master of Finance")
    of_match = re.search(
        r'(?:Bachelor|Master|Doctor|PhD|Diploma|Certificate|NITEC|Higher NITEC|GCE|Advanced Diploma)'
        r'\s+of\s+(.+?)(?:\s*[-–,;(]|$)',
        degree_string, re.IGNORECASE
    )
    if of_match:
        major = of_match.group(1).strip()
        # Clean up trailing prepositions/conjunctions
        major = re.sub(r'\s+(?:from|at|with|and|&)\s*$', '', major, flags=re.IGNORECASE)
        return major, degree_string
    
    # Pattern: "Major - Degree" or "Major, Degree"
    dash_match = re.match(r'^(.+?)\s*[-–]\s*(Bachelor|Master|Doctor|PhD|Diploma)', degree_string, re.IGNORECASE)
    if dash_match:
        return dash_match.group(1).strip(), degree_string
    
    # No clear major found
    return "", degree_string


def parse_skills_to_tags(skills_value: Any) -> List[str]:
    """
    🏷️ Convert skills data (various formats) into a clean tag list.
    
    Your DB stores skills as:
      - Pipe-separated: "Python | Java | SQL"
      - JSON array: ["Python", "Java", "SQL"]
      - JSON string of pipe-sep: "\"Python | Java | SQL\""
      - Nested dict: {"hard": ["Python"], "soft": ["Leadership"]}
      - Plain text: "Python, Java, SQL"
    
    Like sorting a messy jewelry box into a neat display! 💎📦
    """
    if not skills_value:
        return []
    
    tags = []
    
    # If it's a string, try to parse it
    if isinstance(skills_value, str):
        skills_str = skills_value.strip()
        
        # Try JSON parse first
        try:
            parsed = json.loads(skills_str)
            if isinstance(parsed, list):
                # JSON array: ["skill1", "skill2"]
                tags = [str(s).strip() for s in parsed if s]
            elif isinstance(parsed, dict):
                # Dict with categories: {"hard": [...], "soft": [...]}
                for category, items in parsed.items():
                    if isinstance(items, list):
                        tags.extend(str(s).strip() for s in items if s)
                    elif isinstance(items, str):
                        tags.append(items.strip())
            elif isinstance(parsed, str):
                # JSON-encoded string
                skills_str = parsed
                # Fall through to pipe/comma splitting
                if " | " in skills_str:
                    tags = [s.strip() for s in skills_str.split(" | ") if s.strip()]
                elif ", " in skills_str:
                    tags = [s.strip() for s in skills_str.split(", ") if s.strip()]
                else:
                    tags = [skills_str]
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON — try pipe or comma splitting
            if " | " in skills_str:
                tags = [s.strip() for s in skills_str.split(" | ") if s.strip()]
            elif ", " in skills_str:
                tags = [s.strip() for s in skills_str.split(", ") if s.strip()]
            else:
                tags = [skills_str] if skills_str else []
    
    elif isinstance(skills_value, list):
        tags = [str(s).strip() for s in skills_value if s]
    
    elif isinstance(skills_value, dict):
        for category, items in skills_value.items():
            if isinstance(items, list):
                tags.extend(str(s).strip() for s in items if s)
    
    # Deduplicate while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower not in seen and tag:
            seen.add(tag_lower)
            unique_tags.append(tag)
    
    return unique_tags


def parse_experience_entries(experience_value: Any) -> List[Dict]:
    """
    💼 Parse experience data into your schema's Work Experience format.
    
    DB format:  [{company, role, dates, description}, ...]
    Your format: [{company, title, from, to, responsibility: [...]}, ...]
    """
    if not experience_value:
        return []
    
    entries = []
    raw_entries = []
    
    # Parse the value from DB (could be JSON string, list, or plain text)
    if isinstance(experience_value, str):
        experience_value = experience_value.strip()
        try:
            parsed = json.loads(experience_value)
            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, dict):
                raw_entries = [parsed]
            elif isinstance(parsed, str):
                # It was a JSON-encoded string — try one more level
                try:
                    inner = json.loads(parsed)
                    if isinstance(inner, list):
                        raw_entries = inner
                except (json.JSONDecodeError, TypeError):
                    pass
        except (json.JSONDecodeError, TypeError):
            # Plain text — try to extract as single entry
            if len(experience_value) > 20:
                raw_entries = [{"company": "", "role": "", "dates": "", "description": experience_value}]
    elif isinstance(experience_value, list):
        raw_entries = experience_value
    
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        
        # Parse dates into from/to
        dates = str(entry.get("dates", "") or "")
        from_date, to_date = parse_dates_to_from_to(dates)
        
        # Parse description into responsibility list
        description = str(entry.get("description", "") or "")
        responsibilities = parse_description_to_responsibilities(description)
        
        entries.append({
            "company": str(entry.get("company", "") or "").strip(),
            "title": str(entry.get("role", "") or entry.get("title", "") or "").strip(),
            "from": from_date,
            "to": to_date,
            "responsibility": responsibilities
        })
    
    return entries


def parse_education_entries(education_value: Any) -> List[Dict]:
    """
    🎓 Parse education data into your schema's Education format.
    
    DB format:  [{institution, degree, dates, description}, ...]
    Your format: [{school, major, Degree}, ...]
    """
    if not education_value:
        return []
    
    entries = []
    raw_entries = []
    
    if isinstance(education_value, str):
        education_value = education_value.strip()
        try:
            parsed = json.loads(education_value)
            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, dict):
                raw_entries = [parsed]
            elif isinstance(parsed, str):
                try:
                    inner = json.loads(parsed)
                    if isinstance(inner, list):
                        raw_entries = inner
                except (json.JSONDecodeError, TypeError):
                    pass
        except (json.JSONDecodeError, TypeError):
            if len(education_value) > 10:
                raw_entries = [{"institution": education_value, "degree": "", "dates": ""}]
    elif isinstance(education_value, list):
        raw_entries = education_value
    
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        
        degree_str = str(entry.get("degree", "") or "").strip()
        major, full_degree = parse_education_degree_to_major(degree_str)
        
        entries.append({
            "school": str(entry.get("institution", "") or entry.get("school", "") or "").strip(),
            "major": major,
            "Degree": full_degree
        })
    
    return entries


def parse_languages(languages_value: Any) -> str:
    """
    🗣️ Parse language data into a comma-separated string.
    """
    if not languages_value:
        return ""
    
    if isinstance(languages_value, str):
        # Try JSON parse
        try:
            parsed = json.loads(languages_value)
            if isinstance(parsed, list):
                # Could be list of strings or list of dicts
                items = []
                for item in parsed:
                    if isinstance(item, dict):
                        lang = item.get("language", item.get("name", ""))
                        level = item.get("proficiency", item.get("level", ""))
                        if lang:
                            items.append(f"{lang} ({level})" if level else lang)
                    elif isinstance(item, str):
                        items.append(item)
                return ", ".join(items)
            elif isinstance(parsed, str):
                return parsed
        except (json.JSONDecodeError, TypeError):
            return languages_value.strip()
    
    return str(languages_value)


# =============================================================================
# 🎯 FUNCTION CLASSIFIER — "What department does this person belong to?"
# Reads job titles, skills, descriptions to assign the right Function category.
# Like a casting director who KNOWS which department fits! 🎭
# =============================================================================

# Maps keyword patterns → Function category.
# Order matters: FIRST match wins. More specific patterns come first.
# Each tuple: (category_name, set_of_keywords_to_match)
FUNCTION_CATEGORIES = [
    ("Accounting & Finance", {
        'accountant', 'accounting', 'finance', 'financial', 'auditor', 'audit',
        'tax', 'treasury', 'controller', 'bookkeeper', 'accounts payable',
        'accounts receivable', 'cpa', 'acca', 'cfa', 'financial analyst',
        'credit', 'billing', 'invoice', 'payroll', 'budgeting', 'revenue',
        'investment', 'fund', 'portfolio', 'compliance officer', 'risk analyst',
        'wealth management', 'actuarial', 'actuary', 'cmfas',
    }),
    ("IT", {
        'software', 'developer', 'programmer', 'engineer', 'devops', 'sre',
        'full stack', 'fullstack', 'frontend', 'backend', 'data scientist',
        'data engineer', 'data analyst', 'machine learning', 'ml engineer',
        'ai engineer', 'cloud', 'cybersecurity', 'security analyst',
        'network engineer', 'system administrator', 'sysadmin', 'dba',
        'database administrator', 'qa engineer', 'test engineer', 'tester',
        'web developer', 'mobile developer', 'ios', 'android developer',
        'scrum master', 'product owner', 'technical lead', 'tech lead',
        'python', 'java developer', 'solution architect', 'it manager',
        'it support', 'helpdesk', 'infrastructure', 'platform engineer',
        'ux designer', 'ui designer', 'ux/ui', 'information technology',
        'erp', 'sap consultant', 'it director', 'cto', 'cio',
    }),
    ("Engineering", {
        'mechanical engineer', 'electrical engineer', 'civil engineer',
        'chemical engineer', 'structural engineer', 'process engineer',
        'manufacturing engineer', 'industrial engineer', 'quality engineer',
        'reliability engineer', 'design engineer', 'project engineer',
        'maintenance engineer', 'field engineer', 'site engineer',
        'engineering manager', 'cad', 'autocad', 'solidworks',
        'plc', 'automation engineer', 'control engineer', 'safety engineer',
        'environmental engineer', 'surveyor', 'drafter', 'bim',
    }),
    ("Human Resources", {
        'human resources', 'hr', 'recruiter', 'recruiting', 'recruitment',
        'talent acquisition', 'hr manager', 'hr executive', 'hr generalist',
        'hr business partner', 'hrbp', 'compensation', 'benefits',
        'employee relations', 'people operations', 'people & culture',
        'organisational development', 'organizational development',
        'training and development', 'learning & development', 'l&d',
        'payroll specialist', 'hr admin', 'manpower', 'workforce',
    }),
    ("Sales", {
        'sales', 'account manager', 'account executive', 'business development',
        'sales manager', 'sales executive', 'sales director', 'sales engineer',
        'territory manager', 'key account', 'channel manager', 'pre-sales',
        'inside sales', 'field sales', 'telesales', 'sales coordinator',
        'revenue manager', 'commercial manager', 'client relationship',
    }),
    ("Marketing", {
        'marketing', 'brand manager', 'digital marketing', 'content',
        'seo', 'sem', 'social media', 'campaign', 'marketing manager',
        'marketing executive', 'communications', 'public relations', 'pr',
        'copywriter', 'creative director', 'media planner', 'advertising',
        'growth hacker', 'performance marketing', 'email marketing',
        'marketing coordinator', 'brand ambassador', 'event manager',
    }),
    ("Customer Service", {
        'customer service', 'customer support', 'call center', 'call centre',
        'contact center', 'service desk', 'customer experience', 'cx',
        'client services', 'customer success', 'guest relations',
        'front desk', 'receptionist', 'concierge', 'helpline',
        'complaint', 'after-sales', 'customer care', 'client support',
    }),
    ("Management", {
        'general manager', 'managing director', 'ceo', 'coo', 'cfo',
        'chief', 'vice president', 'vp', 'avp', 'svp', 'director',
        'head of', 'country manager', 'regional manager', 'group head',
        'president', 'managing partner', 'principal', 'executive director',
        'business unit head', 'division head', 'operations manager',
        'strategy', 'strategic planning', 'transformation',
    }),
    ("Administration", {
        'admin', 'administration', 'administrative', 'secretary', 'clerk',
        'office manager', 'executive assistant', 'personal assistant',
        'data entry', 'filing', 'office coordinator', 'office admin',
        'receptionist', 'general affairs', 'corporate services',
        'facilities coordinator', 'document controller',
    }),
    ("Legal", {
        'lawyer', 'legal', 'attorney', 'counsel', 'solicitor', 'paralegal',
        'legal executive', 'legal officer', 'compliance', 'regulatory',
        'contract', 'litigation', 'corporate secretary', 'governance',
        'intellectual property', 'patent', 'trademark', 'legal advisor',
    }),
    ("Procurement", {
        'procurement', 'purchasing', 'buyer', 'sourcing', 'vendor management',
        'supply chain', 'strategic sourcing', 'tender', 'contracts manager',
        'procurement manager', 'purchasing officer', 'category manager',
    }),
    ("Shipping & Logistics", {
        'logistics', 'shipping', 'warehouse', 'supply chain', 'freight',
        'import', 'export', 'customs', 'distribution', 'fleet',
        'transportation', 'delivery', 'inventory', 'materials planning',
        'logistics coordinator', 'warehouse supervisor', 'forklift',
        'cargo', 'container', 'port', 'forwarding', '3pl', 'courier',
    }),
    ("Production", {
        'production', 'manufacturing', 'assembly', 'fabrication',
        'production manager', 'production supervisor', 'machine operator',
        'cnc', 'technician', 'factory', 'plant manager', 'lean',
        'six sigma', 'quality control', 'qc', 'quality assurance',
        'process improvement', 'production planning', 'tool maker',
    }),
    ("Facilities Management", {
        'facilities', 'facility', 'building management', 'property',
        'estate', 'maintenance manager', 'building maintenance',
        'security manager', 'cleaning', 'landscaping', 'hvac',
        'fire safety', 'workplace', 'space planning', 'tenant',
    }),
    ("Fresh Graduates (Deg/Dip)", {
        # Matched by logic in classify_function() — not by keywords alone
    }),
    ("Fresh Graduates (ITE)", {
        # Matched by logic in classify_function() — not by keywords alone
    }),
]


def classify_function(
    current_title: str,
    all_titles: List[str],
    skills: List[str],
    education: List[Dict],
    experience: List[Dict],
) -> str:
    """
    🎯 Classify a candidate into a Function category.
    
    Algorithm (priority order):
    1. Match current job title against keyword sets
    2. Match ALL job titles (if current title is vague)
    3. Match skills against keyword sets
    4. Check for fresh graduate (no/minimal experience + recent education)
    5. Default to "Others"
    
    Fresh graduate detection:
    - 0-1 jobs AND education within last 2 years
    - ITE/NITEC keywords → "Fresh Graduates (ITE)"
    - Degree/Diploma keywords → "Fresh Graduates (Deg/Dip)"
    
    Args:
        current_title: Most recent job title.
        all_titles:    All job titles across experience entries.
        skills:        Skills tag list.
        education:     Parsed education entries.
        experience:    Parsed experience entries.
        
    Returns:
        Category name string.
    """
    # Build a combined text blob for matching (lowercase)
    title_blob = " ".join([current_title] + all_titles).lower()
    skills_blob = " ".join(skills).lower()
    # Include responsibility text for richer signal
    resp_blob = ""
    for job in experience:
        resp_blob += " ".join(job.get("responsibility", [])) + " "
    resp_blob = resp_blob.lower()
    
    combined = f"{title_blob} {skills_blob} {resp_blob}"
    
    # --- Pass 1: Match against keyword categories ---
    best_category = None
    best_score = 0
    
    for category, keywords in FUNCTION_CATEGORIES:
        if not keywords:
            continue  # Skip fresh grad categories (handled separately)
        
        score = 0
        for kw in keywords:
            # Title matches are worth more than skill/description matches
            if kw in title_blob:
                score += 3
            elif kw in skills_blob:
                score += 2
            elif kw in resp_blob:
                score += 1
        
        if score > best_score:
            best_score = score
            best_category = category
    
    # --- Pass 2: Fresh graduate detection ---
    # If low/no experience AND recent education → fresh grad
    if len(experience) <= 1:
        is_ite = False
        is_deg_dip = False
        
        for edu in education:
            degree_lower = edu.get("Degree", "").lower()
            school_lower = edu.get("school", "").lower()
            
            if any(kw in degree_lower or kw in school_lower for kw in
                   ['nitec', 'ite college', 'ite ', 'higher nitec']):
                is_ite = True
            elif any(kw in degree_lower for kw in
                     ['bachelor', 'master', 'degree', 'diploma', 'honours', 'honors',
                      'phd', 'doctor', 'graduate', 'bsc', 'ba ', 'beng', 'mba', 'msc']):
                is_deg_dip = True
        
        # Only classify as fresh grad if we couldn't find a strong function match
        if best_score < 3:
            if is_ite:
                return "Fresh Graduates (ITE)"
            elif is_deg_dip and len(experience) == 0:
                return "Fresh Graduates (Deg/Dip)"
    
    if best_category and best_score >= 2:
        return best_category
    
    return "Others"


# =============================================================================
# 🏭 INDUSTRY CLASSIFIER — "What industry is this candidate's experience in?"
# Reads company names, job descriptions, certifications, and skills.
# =============================================================================

INDUSTRY_CATEGORIES = [
    ("Banking & Finance", {
        'bank', 'banking', 'dbs', 'ocbc', 'uob', 'citibank', 'hsbc',
        'standard chartered', 'maybank', 'cimb', 'rhb', 'bnp paribas',
        'barclays', 'goldman', 'morgan stanley', 'jp morgan', 'credit suisse',
        'insurance', 'great eastern', 'prudential', 'aia', 'manulife',
        'zurich', 'aviva', 'ntuc income', 'wealth management', 'fintech',
        'remittance', 'securities', 'brokerage', 'forex', 'crypto',
        'mas regulated', 'cmfas', 'fund management', 'asset management',
    }),
    ("Semiconductor", {
        'semiconductor', 'wafer', 'fab', 'chip', 'integrated circuit',
        'micron', 'micron technology', 'globalfoundries', 'tsmc', 'intel',
        'broadcom', 'qualcomm', 'silicon', 'photolithography', 'etching',
        'diffusion', 'asm', 'applied materials', 'lam research', 'asml',
    }),
    ("Infocomm", {
        'software', 'it company', 'tech', 'technology', 'saas', 'cloud',
        'startup', 'digital', 'platform', 'app development', 'web development',
        'ai company', 'data analytics', 'cybersecurity firm', 'iot',
        'google', 'microsoft', 'amazon', 'meta', 'facebook', 'apple',
        'grab', 'sea group', 'shopee', 'lazada', 'gojek', 'bytedance',
        'tiktok', 'govtech', 'ncs', 'st engineering', 'singtel',
        'thoughtworks', 'accenture digital', 'infosys', 'tcs', 'wipro',
    }),
    ("Healthcare", {
        'hospital', 'clinic', 'healthcare', 'medical', 'health',
        'sgh', 'nuh', 'ttsh', 'cgh', 'ktph', 'skh', 'kkh',
        'parkway', 'raffles medical', 'mount elizabeth', 'gleneagles',
        'ihh', 'nursing', 'nurse', 'physiotherapy', 'dentist', 'dental',
        'pharmacy', 'pharmacist', 'doctor', 'physician', 'surgeon',
        'clinical', 'patient', 'allied health', 'radiographer',
        'speech therapy', 'occupational therapy', 'mental health',
    }),
    ("Education", {
        'school', 'university', 'polytechnic', 'college', 'academy',
        'moe', 'ministry of education', 'teacher', 'lecturer', 'tutor',
        'professor', 'principal', 'education', 'teaching', 'curriculum',
        'student', 'training provider', 'enrichment', 'tuition',
        'ite', 'nus', 'ntu', 'smu', 'sit', 'suss', 'sutd',
    }),
    ("Engineering - Aerospace", {
        'aerospace', 'aviation', 'aircraft', 'airline', 'airport',
        'sia', 'singapore airlines', 'changi airport', 'sats',
        'rolls royce', 'pratt & whitney', 'airbus', 'boeing',
        'mro', 'avionics', 'aero engine', 'caas', 'pilot',
        'flight', 'cabin crew', 'ground handling', 'air cargo',
    }),
    ("Engineering - Precision", {
        'precision engineering', 'precision', 'machining', 'tooling',
        'cnc', 'milling', 'grinding', 'edm', 'wire cut', 'lathe',
        'die casting', 'injection mould', 'injection mold', 'stamping',
        'jig', 'fixture', 'metrology', 'cmm', 'surface grinding',
    }),
    ("Marine & Offshore", {
        'marine', 'offshore', 'shipyard', 'ship', 'vessel', 'rig',
        'keppel', 'sembcorp', 'jurong shipyard', 'drydock',
        'subsea', 'fpso', 'naval', 'maritime', 'port', 'psa',
        'shipping line', 'container terminal', 'bunker',
    }),
    ("Energy (Oil & Gas)", {
        'oil', 'gas', 'petroleum', 'petrochemical', 'refinery',
        'shell', 'exxonmobil', 'chevron', 'bp', 'total energies',
        'lng', 'pipeline', 'drilling', 'upstream', 'downstream',
        'energy', 'renewable', 'solar', 'wind power', 'utilities',
    }),
    ("Pharmaceutical/Biotech", {
        'pharmaceutical', 'pharma', 'biotech', 'biotechnology',
        'gsk', 'pfizer', 'roche', 'novartis', 'sanofi', 'merck',
        'abbott', 'johnson & johnson', 'amgen', 'gilead', 'bayer',
        'clinical trial', 'drug', 'vaccine', 'biologics', 'gmp',
        'fda', 'hsa', 'regulatory affairs', 'pharmacovigilance',
    }),
    ("Medical Technology", {
        'medical device', 'medtech', 'medical technology', 'medtronic',
        'stryker', 'zimmer biomet', 'becton dickinson', 'bd',
        'siemens healthineers', 'ge healthcare', 'philips healthcare',
        'surgical', 'diagnostic', 'implant', 'prosthetics',
    }),
    ("Building/Construction", {
        'construction', 'building', 'contractor', 'developer',
        'property developer', 'real estate', 'architecture', 'architect',
        'quantity surveyor', 'site supervisor', 'project manager construction',
        'bca', 'hdb', 'jrc', 'capitaland', 'city developments',
        'concrete', 'steel structure', 'piling', 'scaffolding',
        'interior design', 'renovation', 'fit-out', 'landscape',
    }),
    ("Chemicals", {
        'chemical', 'chemicals', 'petrochemical', 'polymer', 'resin',
        'basf', 'dow', 'dupont', 'sabic', 'linde', 'air liquide',
        'specialty chemical', 'adhesive', 'coating', 'paint',
        'solvent', 'catalyst', 'laboratory', 'r&d chemical',
    }),
    ("Hospitality & Tourism", {
        'hotel', 'resort', 'hospitality', 'tourism', 'travel',
        'marriott', 'hilton', 'accor', 'hyatt', 'shangri-la',
        'marina bay sands', 'mbs', 'sentosa', 'rwg', 'cruise',
        'tour guide', 'travel agent', 'booking', 'concierge',
        'housekeeping', 'banquet', 'events management',
    }),
    ("F&B", {
        'food', 'beverage', 'f&b', 'fnb', 'restaurant', 'cafe',
        'catering', 'chef', 'cook', 'baker', 'pastry',
        'food manufacturing', 'food processing', 'food safety',
        'hawker', 'kopitiam', 'food court', 'barista', 'sommelier',
    }),
    ("FMCG", {
        'fmcg', 'consumer goods', 'p&g', 'procter', 'unilever',
        'nestle', 'colgate', 'coca-cola', 'pepsi', 'mondelez',
        'henkel', 'reckitt', 'l\'oreal', 'loreal', 'shiseido',
        'consumer product', 'personal care', 'household',
    }),
    ("Retail", {
        'retail', 'store', 'shop', 'merchandising', 'visual merchandiser',
        'store manager', 'retail assistant', 'cashier', 'point of sale',
        'e-commerce', 'ecommerce', 'marketplace', 'fashion retail',
        'luxury retail', 'department store', 'mall', 'outlet',
    }),
    ("Automotive", {
        'automotive', 'automobile', 'car', 'vehicle', 'motor',
        'toyota', 'honda', 'mercedes', 'bmw', 'volkswagen', 'hyundai',
        'bosch', 'continental', 'denso', 'aisin', 'ev', 'electric vehicle',
        'auto parts', 'workshop', 'mechanic', 'dealership',
    }),
    ("Telco", {
        'telco', 'telecom', 'telecommunications', 'singtel', 'starhub',
        'm1', 'simba', 'circles', 'digi', 'celcom', 'maxis',
        'network operator', 'mobile network', '5g', 'fibre',
        'broadband', 'iptv', 'tower', 'spectrum',
    }),
    ("Supply Chain Mgt & Logistics", {
        'supply chain', 'logistics', 'warehouse', 'distribution',
        'freight', 'forwarding', 'dhl', 'fedex', 'ups', 'maersk',
        'schenker', 'kuehne nagel', 'agility', 'ceva', 'nippon express',
        'inventory management', 'procurement', '3pl', 'last mile',
    }),
    ("Robotics", {
        'robotics', 'robot', 'automation', 'cobot', 'agv',
        'industrial robot', 'abb robotics', 'fanuc', 'kuka', 'yaskawa',
        'drone', 'uav', 'autonomous', 'plc programming',
    }),
    ("Environment & Water", {
        'environment', 'environmental', 'water', 'wastewater',
        'pub', 'nea', 'sustainability', 'waste management', 'recycling',
        'green building', 'carbon', 'esg', 'climate', 'hyflux',
        'keppel infrastructure', 'sembcorp utilities', 'pollution',
    }),
    ("Trading", {
        'trading', 'trader', 'commodity', 'import export',
        'trafigura', 'glencore', 'vitol', 'noble group', 'olam',
        'wilmar', 'adm', 'cargill', 'bunge', 'louis dreyfus',
        'general trading', 'wholesale', 'distributor',
    }),
]


def classify_industry(
    company_names: List[str],
    all_titles: List[str],
    skills: List[str],
    descriptions: List[str],
    certifications: str,
) -> str:
    """
    🏭 Classify a candidate's industry based on their work history.
    
    Algorithm:
    1. Score each industry category based on keyword matches in:
       - Company names (highest weight — most direct signal)
       - Job titles (high weight)
       - Skills (medium weight)
       - Job descriptions / responsibilities (low weight — noisy)
       - Certifications (medium weight)
    2. Highest scoring category wins
    3. Default to "Others" if no strong signal
    
    Args:
        company_names: List of company names from all experience entries.
        all_titles:    All job titles.
        skills:        Skills tag list.
        descriptions:  Combined responsibility text from all jobs.
        certifications: Raw certifications text.
        
    Returns:
        Industry category name string.
    """
    company_blob = " ".join(company_names).lower()
    title_blob = " ".join(all_titles).lower()
    skills_blob = " ".join(skills).lower()
    desc_blob = " ".join(descriptions).lower()
    cert_blob = (certifications or "").lower()
    
    combined_high = f"{company_blob} {title_blob} {cert_blob}"
    combined_low = f"{skills_blob} {desc_blob}"
    
    best_category = None
    best_score = 0
    
    for category, keywords in INDUSTRY_CATEGORIES:
        score = 0
        for kw in keywords:
            # Company name matches are the strongest signal
            if kw in company_blob:
                score += 4
            elif kw in title_blob:
                score += 3
            elif kw in cert_blob:
                score += 2
            elif kw in skills_blob:
                score += 2
            elif kw in desc_blob:
                score += 1
        
        if score > best_score:
            best_score = score
            best_category = category
    
    if best_category and best_score >= 3:
        return best_category
    
    return "Others"


def normalize_location(location_raw: str) -> str:
    """
    📍 Normalize location to a clean, standardized format.
    
    Singapore resumes often have full block addresses like:
    "Blk 123 Ang Mo Kio Ave 6 #12-345 Singapore 560123"
    
    We want just: "Singapore" or "Kuala Lumpur, Malaysia"
    
    Like trimming the trailing fabric after the final stitch — 
    clean and neat! ✂️📍
    """
    if not location_raw:
        return ""
    
    loc = location_raw.strip()
    
    # SG-specific: if it mentions Singapore postal code or "Blk", simplify
    if re.search(r'\b(?:Singapore|SG)\s*\d{6}\b', loc, re.IGNORECASE):
        return "Singapore"
    if re.match(r'(?:Blk|Block)\s+\d+', loc, re.IGNORECASE):
        return "Singapore"
    if re.match(r'#\d{1,3}-', loc):
        return "Singapore"
    
    # MY-specific: clean up state-level
    my_states = {
        # More specific patterns FIRST (before their parent regions)
        'johor bahru': 'Johor Bahru, Malaysia',
        'petaling jaya': 'Petaling Jaya, Malaysia',
        'subang jaya': 'Subang Jaya, Malaysia',
        'shah alam': 'Shah Alam, Malaysia',
        'kota kinabalu': 'Kota Kinabalu, Malaysia',
        'kuala lumpur': 'Kuala Lumpur, Malaysia',
        'selangor': 'Selangor, Malaysia',
        'penang': 'Penang, Malaysia',
        'johor': 'Johor, Malaysia',
        'putrajaya': 'Putrajaya, Malaysia',
        'ipoh': 'Ipoh, Malaysia',
        'malacca': 'Malacca, Malaysia',
        'melaka': 'Melaka, Malaysia',
        'kuching': 'Kuching, Malaysia',
        'kedah': 'Kedah, Malaysia',
        'sungai petani': 'Sungai Petani, Malaysia',
        'alor setar': 'Alor Setar, Malaysia',
    }
    
    loc_lower = loc.lower().strip()
    for pattern, normalized in my_states.items():
        if pattern in loc_lower:
            return normalized
    
    # If just "Singapore" or "Malaysia" alone
    if loc_lower in ('singapore', 'sg'):
        return "Singapore"
    if loc_lower in ('malaysia', 'my'):
        return "Malaysia"
    
    # Return as-is if we can't simplify
    return loc


# =============================================================================
# 🏗️ MAIN TRANSFORMER — DB Row → Your JSON Schema
# =============================================================================

def transform_candidate(row: Dict) -> Dict:
    """
    🏗️ Transform a single DB row into your exact JSON schema.
    
    This is THE metamorphosis — taking the caterpillar of raw DB data
    and turning it into the butterfly of your standardized format! 🦋✨
    
    Args:
        row: Dict from structured_extractions + raw_extractions join.
        
    Returns:
        Dict matching your exact JSON schema.
    """
    # Parse complex fields
    experience = parse_experience_entries(
        row.get("experience_json") or row.get("experience_raw")
    )
    education = parse_education_entries(
        row.get("education_json") or row.get("education_raw")
    )
    tags = parse_skills_to_tags(
        row.get("skills_json") or row.get("skills_raw")
    )
    languages = parse_languages(row.get("languages"))
    
    # Extract current company and title from FIRST (most recent) experience
    current_company = ""
    current_title = ""
    if experience:
        latest = experience[0]
        current_company = latest.get("company", "")
        current_title = latest.get("title", "")
        # Validate: if "to" date is "Present" or empty, it's truly current
        to_date = latest.get("to", "").lower()
        if to_date and to_date not in ("present", "current", "now", "ongoing", ""):
            # Not current — might be between jobs
            current_company = current_company + " (Last)"
    
    # Parse projects into a string
    projects = ""
    projects_raw = row.get("projects", "")
    if projects_raw:
        try:
            parsed_projects = json.loads(projects_raw) if isinstance(projects_raw, str) else projects_raw
            if isinstance(parsed_projects, list):
                project_items = []
                for p in parsed_projects:
                    if isinstance(p, dict):
                        project_items.append(
                            p.get("name", p.get("title", str(p)))
                        )
                    else:
                        project_items.append(str(p))
                projects = " | ".join(project_items)
            else:
                projects = str(parsed_projects)
        except (json.JSONDecodeError, TypeError):
            projects = str(projects_raw)
    
    # Build the final record in YOUR exact schema
    
    # --- Classify Function & Industry from resume content ---
    all_titles = [job.get("title", "") for job in experience]
    all_companies = [job.get("company", "") for job in experience]
    all_responsibilities = []
    for job in experience:
        all_responsibilities.extend(job.get("responsibility", []))
    
    classified_function = classify_function(
        current_title=current_title,
        all_titles=all_titles,
        skills=tags,
        education=education,
        experience=experience,
    )
    
    classified_industry = classify_industry(
        company_names=all_companies,
        all_titles=all_titles,
        skills=tags,
        descriptions=all_responsibilities,
        certifications=str(row.get("certifications", "") or ""),
    )

    # 🆕 ML-enhanced classification — uses RandomForest (72 roles, 97% accuracy)
    # Enhances keyword classification with ML prediction when confident.
    # Graceful degradation: if .pkl files are missing, keywords still work! 💅
    predicted_role = None
    role_confidence = 0.0
    try:
        from job_role_predictor import JobRolePredictor
        _predictor = JobRolePredictor()
        if _predictor.available:
            raw_text = str(row.get("raw_text", "") or "")
            if raw_text and len(raw_text.strip()) > 100:
                role_result = _predictor.predict(raw_text)
                predicted_role = role_result.get("predicted_role")
                role_confidence = role_result.get("confidence", 0.0)
                ml_function = role_result.get("function", "Others")

                # Override keyword Function if ML is confident
                if role_confidence >= 0.20 and ml_function != "Others":
                    classified_function = ml_function
                    logger.info(
                        f"🤖 ML: {predicted_role} ({role_confidence:.1%}) "
                        f"→ Function: {ml_function} for candidate {row.get('candidate_id')}"
                    )
    except ImportError:
        pass  # job_role_predictor.py not installed
    except Exception as e:
        logger.debug(f"ML prediction skipped in export: {e}")

    # --- Normalize location ---
    raw_location = str(row.get("location", "") or "").strip()
    normalized_location = normalize_location(raw_location)
    
    record = {
        "ID": row.get("candidate_id", ""),
        "Name": str(row.get("name", "") or "").strip(),
        "Phone": str(row.get("phone", "") or "").strip(),
        "Email": str(row.get("email", "") or "").strip(),
        "Current Company": current_company,
        "Current Title": current_title,
        "Team": "",                                         # Not in DB — placeholder
        "Current Location": normalized_location,
        "Expected Location": "",                            # Not in DB — placeholder
        "Gender": "",                                       # Not in DB — placeholder
        "Created By": SYSTEM_NAME,
        "Creation Date": str(row.get("created_at", "") or ""),
        "Last Contact": "",                                 # Not in DB — placeholder
        "Function": classified_function,
        "Industry": classified_industry,
        "Predicted Role": predicted_role or "",
        "Role Confidence": f"{role_confidence:.0%}" if predicted_role else "",
        "Summary": str(row.get("summary", "") or "").strip(),
        "Language Skills": languages,
        "Work Experience": experience,
        "Education": education,
        "Project Experience": projects,
        "tags": tags,
    }
    
    return record


# =============================================================================
# 📤 EXPORT FUNCTIONS
# =============================================================================

def export_json(
    records: List[Dict],
    output_path: str,
    pretty: bool = True
) -> str:
    """
    📋 Export all records to a single JSON file.
    
    Creates a JSON file with an array of candidate records,
    each matching your exact schema. Clean, organized, ready
    for any ATS or recruitment platform! 📋✨
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2 if pretty else None, ensure_ascii=False, default=str)
    
    logger.info(f"📋 JSON exported: {len(records)} records → {output_path}")
    return output_path


def export_individual_json(
    records: List[Dict],
    output_dir: str
) -> List[str]:
    """
    📋 Export each candidate as a SEPARATE JSON file.
    
    Useful for per-candidate processing or uploads.
    Files named: candidate_{ID}.json
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    
    for record in records:
        cid = record.get("ID", "unknown")
        path = os.path.join(output_dir, f"candidate_{cid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        paths.append(path)
    
    logger.info(f"📋 Individual JSON exported: {len(paths)} files → {output_dir}")
    return paths


def export_csv(
    records: List[Dict],
    output_path: str
) -> str:
    """
    📊 Export all records to a CSV file.
    
    Since CSV is flat (no nested arrays), we handle complex fields:
    - Work Experience → Flattened: "Company1 | Title1 | From1-To1; Company2 | ..."
    - Education → Flattened: "School1 | Degree1; School2 | ..."
    - tags → Pipe-separated: "Python | Java | SQL"
    - responsibility → Semicolons within each job
    
    Also creates a DETAILED CSV with one row per job entry for
    experience-level analysis. Like spreading out the dress pattern
    pieces so you can see every panel! 📐
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # ---- Main CSV (one row per candidate) ----
    flat_headers = [
        "ID", "Name", "Phone", "Email",
        "Current Company", "Current Title", "Team",
        "Current Location", "Expected Location",
        "Gender", "Created By", "Creation Date", "Last Contact",
        "Function", "Industry", "Summary", "Language Skills",
        "Work Experience", "Education", "Project Experience", "tags"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=flat_headers, extrasaction="ignore")
        writer.writeheader()
        
        for record in records:
            flat = dict(record)
            
            # Flatten Work Experience to a readable string
            exp_parts = []
            for job in record.get("Work Experience", []):
                resp_text = "; ".join(job.get("responsibility", []))
                exp_parts.append(
                    f"{job.get('company', '')} | {job.get('title', '')} | "
                    f"{job.get('from', '')}-{job.get('to', '')} | {resp_text}"
                )
            flat["Work Experience"] = " ;; ".join(exp_parts)
            
            # Flatten Education
            edu_parts = []
            for edu in record.get("Education", []):
                edu_parts.append(
                    f"{edu.get('school', '')} | {edu.get('Degree', '')} | {edu.get('major', '')}"
                )
            flat["Education"] = " ;; ".join(edu_parts)
            
            # Flatten tags
            flat["tags"] = " | ".join(record.get("tags", []))
            
            writer.writerow(flat)
    
    logger.info(f"📊 CSV exported: {len(records)} records → {output_path}")
    
    # ---- Detailed Experience CSV (one row per job) ----
    exp_path = output_path.replace(".csv", "_experience_detail.csv")
    exp_headers = [
        "Candidate_ID", "Name", "Job_Index",
        "Company", "Title", "From", "To",
        "Responsibility_1", "Responsibility_2", "Responsibility_3",
        "Responsibility_4", "Responsibility_5",
        "Responsibility_Full"
    ]
    
    with open(exp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=exp_headers, extrasaction="ignore")
        writer.writeheader()
        
        for record in records:
            for idx, job in enumerate(record.get("Work Experience", []), 1):
                resp = job.get("responsibility", [])
                row = {
                    "Candidate_ID": record["ID"],
                    "Name": record["Name"],
                    "Job_Index": idx,
                    "Company": job.get("company", ""),
                    "Title": job.get("title", ""),
                    "From": job.get("from", ""),
                    "To": job.get("to", ""),
                    "Responsibility_Full": " | ".join(resp),
                }
                # Fill individual responsibility columns (up to 5)
                for i in range(5):
                    row[f"Responsibility_{i+1}"] = resp[i] if i < len(resp) else ""
                writer.writerow(row)
    
    logger.info(f"📊 Experience detail CSV → {exp_path}")
    
    # ---- Detailed Education CSV ----
    edu_path = output_path.replace(".csv", "_education_detail.csv")
    edu_headers = [
        "Candidate_ID", "Name", "Edu_Index",
        "School", "Degree", "Major"
    ]
    
    with open(edu_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=edu_headers, extrasaction="ignore")
        writer.writeheader()
        
        for record in records:
            for idx, edu in enumerate(record.get("Education", []), 1):
                writer.writerow({
                    "Candidate_ID": record["ID"],
                    "Name": record["Name"],
                    "Edu_Index": idx,
                    "School": edu.get("school", ""),
                    "Degree": edu.get("Degree", ""),
                    "Major": edu.get("major", ""),
                })
    
    logger.info(f"📊 Education detail CSV → {edu_path}")
    
    return output_path


def export_ml_training(
    records: List[Dict],
    output_path: str
) -> str:
    """
    🧠 Export data in ML training-ready JSONL format.
    
    Each line is a self-contained training example with:
    - The structured fields as labels
    - Flattened features for model input
    - BIO-compatible entity spans
    
    This format works with:
    - Hugging Face Transformers (fine-tuning)
    - spaCy NER training
    - Custom PyTorch/TensorFlow pipelines
    
    Like preparing ingredients for a Michelin-star kitchen —
    everything pre-measured and ready to go! 🍽️✨
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            # Build ML training record with flattened features
            ml_record = {
                # ---- Identification ----
                "id": record["ID"],
                
                # ---- Structured Labels (ground truth for training) ----
                "labels": {
                    "name": record["Name"],
                    "phone": record["Phone"],
                    "email": record["Email"],
                    "current_company": record["Current Company"],
                    "current_title": record["Current Title"],
                    "location": record["Current Location"],
                    "summary": record["Summary"],
                    "language_skills": record["Language Skills"],
                },
                
                # ---- Experience as structured training data ----
                "experience_labels": [
                    {
                        "company": job["company"],
                        "title": job["title"],
                        "date_from": job["from"],
                        "date_to": job["to"],
                        "responsibilities": job["responsibility"],
                    }
                    for job in record.get("Work Experience", [])
                ],
                
                # ---- Education as structured training data ----
                "education_labels": [
                    {
                        "institution": edu["school"],
                        "degree": edu["Degree"],
                        "major": edu["major"],
                    }
                    for edu in record.get("Education", [])
                ],
                
                # ---- Skills as tag list ----
                "skill_tags": record.get("tags", []),
                
                # ---- Feature counts (for ML feature engineering) ----
                "features": {
                    "has_name": 1 if record["Name"] else 0,
                    "has_email": 1 if record["Email"] else 0,
                    "has_phone": 1 if record["Phone"] else 0,
                    "has_summary": 1 if record["Summary"] else 0,
                    "num_jobs": len(record.get("Work Experience", [])),
                    "num_education": len(record.get("Education", [])),
                    "num_skills": len(record.get("tags", [])),
                    "num_languages": len(record.get("Language Skills", "").split(","))
                        if record.get("Language Skills") else 0,
                    "has_projects": 1 if record.get("Project Experience") else 0,
                    "total_responsibilities": sum(
                        len(j.get("responsibility", []))
                        for j in record.get("Work Experience", [])
                    ),
                    "classified_function": record.get("Function", "Others"),
                    "classified_industry": record.get("Industry", "Others"),
                    "location": record.get("Current Location", ""),
                },
            }
            
            f.write(json.dumps(ml_record, ensure_ascii=False) + "\n")
    
    logger.info(f"🧠 ML training JSONL exported: {len(records)} records → {output_path}")
    return output_path


# =============================================================================
# 🗄️ DATABASE QUERY — Fetch candidates from DB
# =============================================================================

def fetch_candidates(
    db_path: str = DATABASE_PATH,
    candidate_ids: Optional[List[int]] = None,
    status_filter: Optional[str] = None,
    reviewed_only: bool = False,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    🗄️ Fetch candidate records from the database.
    
    Joins structured_extractions with raw_extractions to get
    all available data for each candidate.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Build WHERE clauses
        conditions = []
        params = []
        
        if candidate_ids:
            placeholders = ",".join("?" * len(candidate_ids))
            conditions.append(f"s.candidate_id IN ({placeholders})")
            params.extend(candidate_ids)
        
        if status_filter:
            conditions.append("s.extraction_status = ?")
            params.append(status_filter)
        
        if reviewed_only:
            conditions.append("s.reviewed = 1")
        
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT 
                s.*,
                r.raw_text,
                r.text_length,
                r.resume_language
            FROM structured_extractions s
            LEFT JOIN raw_extractions r 
                ON s.candidate_id = r.candidate_id
            {where}
            ORDER BY s.candidate_id
            {limit_clause}
        """
        
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
        
    except sqlite3.Error as e:
        logger.error(f"❌ Database query failed: {e}")
        return []
    finally:
        conn.close()


# =============================================================================
# 🎬 MAIN ORCHESTRATOR
# =============================================================================

def run_export(
    db_path: str = DATABASE_PATH,
    output_dir: str = OUTPUT_DIR,
    formats: Optional[List[str]] = None,
    candidate_ids: Optional[List[int]] = None,
    status_filter: Optional[str] = None,
    reviewed_only: bool = False,
    limit: Optional[int] = None,
    individual_json: bool = False,
    pretty_json: bool = True,
) -> Dict[str, Any]:
    """
    🎬 Run the full export pipeline.
    
    1. Fetch candidates from DB
    2. Transform each record to your schema
    3. Export to requested formats
    
    Returns dict with export results and file paths.
    """
    if formats is None:
        formats = ["json", "csv", "ml"]
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Fetch
    logger.info(f"🗄️ Fetching candidates from {db_path}...")
    raw_rows = fetch_candidates(
        db_path=db_path,
        candidate_ids=candidate_ids,
        status_filter=status_filter,
        reviewed_only=reviewed_only,
        limit=limit,
    )
    
    if not raw_rows:
        logger.warning("⚠️ No candidates found matching filters!")
        return {"error": "No candidates found", "records": 0}
    
    logger.info(f"📊 Found {len(raw_rows)} candidates")
    
    # Step 2: Transform
    logger.info("🏗️ Transforming records to target schema...")
    records = []
    errors = []
    
    for row in raw_rows:
        try:
            record = transform_candidate(row)
            records.append(record)
        except Exception as e:
            cid = row.get("candidate_id", "?")
            name = str(row.get("name", "") or "").strip() or f"ID:{cid}"
            logger.warning(f"⚠️ Transform failed for candidate {cid} ({name}): {e}")
            errors.append({"candidate_id": cid, "name": name, "error": str(e)})

    logger.info(f"✅ Transformed {len(records)} records ({len(errors)} errors)")

    # Step 3: Export
    results = {
        "records": len(records),
        "errors": len(errors),
        "failed_names": [e["name"] for e in errors],
        "timestamp": timestamp,
        "files": {},
    }
    
    if "json" in formats:
        path = os.path.join(output_dir, f"candidates_{timestamp}.json")
        export_json(records, path, pretty=pretty_json)
        results["files"]["json"] = path
    
    if individual_json:
        ind_dir = os.path.join(output_dir, f"individual_{timestamp}")
        paths = export_individual_json(records, ind_dir)
        results["files"]["individual_json"] = ind_dir
        results["files"]["individual_count"] = len(paths)
    
    if "csv" in formats:
        path = os.path.join(output_dir, f"candidates_{timestamp}.csv")
        export_csv(records, path)
        results["files"]["csv"] = path
        results["files"]["csv_experience"] = path.replace(".csv", "_experience_detail.csv")
        results["files"]["csv_education"] = path.replace(".csv", "_education_detail.csv")
    
    if "ml" in formats:
        path = os.path.join(output_dir, f"ml_training_{timestamp}.jsonl")
        export_ml_training(records, path)
        results["files"]["ml_jsonl"] = path
    
    # Summary report
    total_jobs = sum(len(r.get("Work Experience", [])) for r in records)
    total_edu = sum(len(r.get("Education", [])) for r in records)
    total_tags = sum(len(r.get("tags", [])) for r in records)
    
    results["summary"] = {
        "total_candidates": len(records),
        "total_jobs": total_jobs,
        "total_education": total_edu,
        "total_skills": total_tags,
        "avg_jobs_per_candidate": round(total_jobs / len(records), 1) if records else 0,
        "avg_skills_per_candidate": round(total_tags / len(records), 1) if records else 0,
        "fields_filled": {
            "name": sum(1 for r in records if r["Name"]),
            "email": sum(1 for r in records if r["Email"]),
            "phone": sum(1 for r in records if r["Phone"]),
            "summary": sum(1 for r in records if r["Summary"]),
            "experience": sum(1 for r in records if r["Work Experience"]),
            "education": sum(1 for r in records if r["Education"]),
            "skills": sum(1 for r in records if r["tags"]),
        }
    }
    
    return results


# =============================================================================
# 🖥️ CLI INTERFACE
# =============================================================================

def main():
    """
    🎬 Command-line interface for the resume exporter.
    
    Examples:
        python resume_exporter.py
        python resume_exporter.py --format json csv
        python resume_exporter.py --id 101 102 103
        python resume_exporter.py --status Complete --reviewed-only
        python resume_exporter.py --format ml --output ml_data/
    """
    parser = argparse.ArgumentParser(
        description="💅✨ Fairy Codemother's Resume Data Exporter ✨💅",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                             Export all → JSON + CSV + ML
  %(prog)s --format json               JSON only
  %(prog)s --format csv                CSV only (3 files: main + experience + education)
  %(prog)s --format ml                 ML training JSONL
  %(prog)s --id 101 102                Specific candidates
  %(prog)s --status Complete           Filter by extraction status
  %(prog)s --reviewed-only             Only reviewed candidates
  %(prog)s --individual                One JSON file per candidate
  %(prog)s --output my_exports/        Custom output directory
        """
    )
    
    parser.add_argument(
        "--db", default=DATABASE_PATH,
        help=f"Database path (default: {DATABASE_PATH})"
    )
    parser.add_argument(
        "--output", "-o", default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--format", "-f", nargs="+",
        choices=["json", "csv", "ml", "all"],
        default=["all"],
        help="Export format(s)"
    )
    parser.add_argument(
        "--id", nargs="+", type=int, default=None,
        help="Specific candidate IDs to export"
    )
    parser.add_argument(
        "--status", default=None,
        help="Filter by extraction status (e.g., 'Complete', 'Partial')"
    )
    parser.add_argument(
        "--reviewed-only", action="store_true",
        help="Only export reviewed candidates"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of candidates to export"
    )
    parser.add_argument(
        "--individual", action="store_true",
        help="Also export individual JSON files per candidate"
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Compact JSON (no pretty-printing)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - 📤 %(levelname)s - %(message)s"
    )
    
    # Validate database
    if not os.path.exists(args.db):
        print(f"\n❌ Database not found: {args.db}")
        print("💡 Run your extraction pipeline first!")
        sys.exit(1)
    
    # Resolve formats
    formats = args.format
    if "all" in formats:
        formats = ["json", "csv", "ml"]
    
    # Run export
    print("\n" + "=" * 60)
    print("  💅✨ FAIRY CODEMOTHER'S RESUME EXPORTER ✨💅")
    print("=" * 60)
    print(f"  📂 Database: {args.db}")
    print(f"  📤 Output: {args.output}")
    print(f"  📋 Formats: {', '.join(formats)}")
    if args.id:
        print(f"  🎯 Candidate IDs: {args.id}")
    if args.status:
        print(f"  🔍 Status filter: {args.status}")
    if args.reviewed_only:
        print(f"  ✅ Reviewed only: Yes")
    print("=" * 60 + "\n")
    
    results = run_export(
        db_path=args.db,
        output_dir=args.output,
        formats=formats,
        candidate_ids=args.id,
        status_filter=args.status,
        reviewed_only=args.reviewed_only,
        limit=args.limit,
        individual_json=args.individual,
        pretty_json=not args.compact,
    )
    
    # Print results
    if "error" in results:
        print(f"\n❌ {results['error']}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  📊 EXPORT RESULTS")
    print("=" * 60)
    
    summary = results.get("summary", {})
    print(f"  📋 Candidates exported: {results['records']}")
    if results['errors']:
        print(f"  ⚠️  Transform errors: {results['errors']}")
        print(f"  ❌ Failed names:")
        for name in results.get("failed_names", []):
            print(f"       • {name}")
    print(f"  💼 Total jobs: {summary.get('total_jobs', 0)}")
    print(f"  🎓 Total education: {summary.get('total_education', 0)}")
    print(f"  🏷️  Total skills: {summary.get('total_skills', 0)}")
    print(f"  📊 Avg jobs/candidate: {summary.get('avg_jobs_per_candidate', 0)}")
    print(f"  📊 Avg skills/candidate: {summary.get('avg_skills_per_candidate', 0)}")
    
    filled = summary.get("fields_filled", {})
    if filled:
        print(f"\n  📈 Field fill rates:")
        for field, count in filled.items():
            total = results['records']
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"      {field:12s} {bar} {pct:5.1f}% ({count}/{total})")
    
    print(f"\n  📁 Output files:")
    for label, path in results.get("files", {}).items():
        if isinstance(path, str) and os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
            print(f"      {label:20s} → {path} ({size_str})")
        elif isinstance(path, int):
            print(f"      {label:20s} → {path}")
    
    print("\n  ✨ Export complete! 💅\n")


if __name__ == "__main__":
    main()