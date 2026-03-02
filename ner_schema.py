"""
ner_schema.py

💅✨ FAIRY CODEMOTHER'S NER ANNOTATION ENGINE ✨💅

The FOUNDATION of our NLP-powered resume extraction! This is the DNA 🧬
of the entire system — it defines WHAT entities we extract, HOW they're
tagged, and handles all the messy edge cases that make resumes a nightmare.

Think of this as the costume design BLUEPRINT — before you sew a single
stitch, you need to know EXACTLY what you're building! 🎭👗

Architecture:
  Step A — ANNOTATION SCHEMA:  Entity type definitions with validation rules
  Step B — BIO TAGGING ENGINE: Tokenizer + BIO tag converter for NER training
  Step C — NESTED ENTITIES:    Multi-layer annotation for overlapping spans
  Step D — EDGE CASE RULES:    Singapore/Malaysia resume format handlers

Key Classes:
  EntitySchema      — The master catalog of all entity types
  ResumeTokenizer   — Smart tokenizer that preserves resume structure
  BIOTagger         — Converts span annotations to BIO-tagged sequences
  NestedEntityLayer — Handles overlapping entities across multiple layers
  EdgeCaseHandler   — Rules for "Present", nested ORG+LOC, bullet points
  PreAnnotator      — Auto-labels text using dictionaries (skills, cities, etc.)
  TrainingExporter  — Exports annotations to CoNLL, spaCy, HuggingFace formats
  ResumeClassifier  — Predicts Function & Industry from resume text + entities

Dependencies:
    pip install numpy --break-system-packages
    # Optional for advanced features:
    pip install spacy --break-system-packages

Usage:
    from ner_schema import EntitySchema, BIOTagger, PreAnnotator
    schema = EntitySchema()
    tagger = BIOTagger(schema)
    tokens, tags = tagger.tag_text(text, annotations)
"""

import re
import json
import os
import copy
import sqlite3
import datetime
import logging
import unicodedata
from typing import (
    Dict, List, Optional, Tuple, Any, Set,
    NamedTuple, Iterator
)
from dataclasses import dataclass, field, asdict
from collections import defaultdict, OrderedDict
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# 🧬 STEP A: ANNOTATION SCHEMA
# The blueprint that defines EVERY entity type we extract from resumes.
# =============================================================================

class ResumeClassifier:
    """
    🧠 Predicts Function and Industry based on resume text + NER annotations.
    Uses keyword matching with scoring; ties broken by order in the list.
    """

    FUNCTION_KEYWORDS = {
        "Administration": [
            "admin", "administration", "administrative", "office manager",
            "executive assistant", "secretary", "clerical", "receptionist",
            "personal assistant", "pa", "coordinator"
        ],
        "Accounting & Finance": [
            "accountant", "accounting", "finance", "financial", "audit",
            "tax", "ledger", "cpa", "cfa", "controller", "financial analyst",
            "accounts payable", "accounts receivable", "payroll", "bookkeeper"
        ],
        "Customer Service": [
            "customer service", "customer support", "service desk", "call center",
            "helpdesk", "client service", "customer care", "csa"
        ],
        "Engineering": [
            "engineer", "engineering", "mechanical", "electrical", "civil",
            "chemical", "structural", "process engineer", "project engineer",
            "design engineer", "r&d engineer"
        ],
        "Facilities Management": [
            "facilities", "facility manager", "maintenance", "building management",
            "estate management", "property management"
        ],
        "Fresh Graduates (Deg/Dip)": [
            "fresh graduate", "recent graduate", "entry level", "degree holder",
            "diploma holder", "class of", "bachelor", "master's", "graduate trainee"
        ],
        "Fresh Graduates (ITE)": [
            "ite graduate", "nitec", "higher nitec", "ite college", "vocational"
        ],
        "Human Resources": [
            "hr", "human resources", "talent acquisition", "recruiter", "payroll",
            "benefits", "hr generalist", "hr manager", "training", "l&d"
        ],
        "IT": [
            "it", "information technology", "software", "developer", "programmer",
            "network", "system admin", "cybersecurity", "data scientist",
            "cloud engineer", "devops", "database", "frontend", "backend",
            "full stack", "mobile developer", "web developer"
        ],
        "Legal": [
            "legal", "lawyer", "attorney", "paralegal", "legal counsel",
            "compliance", "contract", "litigation"
        ],
        "Management": [
            "manager", "management", "director", "head", "lead", "supervisor",
            "team lead", "vp", "senior management", "executive"
        ],
        "Marketing": [
            "marketing", "digital marketing", "social media", "brand manager",
            "product marketing", "market research", "seo", "sem", "content",
            "marketing communications", "marcom"
        ],
        "Procurement": [
            "procurement", "purchasing", "buyer", "sourcing", "supply chain",
            "procurement manager"
        ],
        "Production": [
            "production", "manufacturing", "operations", "assembly", "plant manager",
            "production planner", "process improvement"
        ],
        "Sales": [
            "sales", "account manager", "business development", "sales representative",
            "b2b", "b2c", "sales executive", "sales director", "channel sales"
        ],
        "Shipping & Logistics": [
            "logistics", "supply chain", "shipping", "warehouse", "distribution",
            "freight", "inventory", "transport", "3pl"
        ],
        "others": []
    }

    INDUSTRY_KEYWORDS = {
        "Automotive": ["automotive", "car", "vehicle", "auto parts", "automobile"],
        "Banking & Finance": [
            "bank", "banking", "finance", "financial services", "investment",
            "wealth management", "insurance", "dbs", "uob", "ocbc", "maybank",
            "private banking", "asset management"
        ],
        "Building/Construction": [
            "construction", "building", "civil engineering", "infrastructure",
            "contractor", "real estate development", "architect"
        ],
        "Chemicals": ["chemical", "petrochemical", "polymer", "specialty chemicals"],
        "Education": [
            "education", "school", "university", "training", "teaching", "lecturer",
            "tuition", "preschool", "international school"
        ],
        "Engineering - Aerospace": [
            "aerospace", "aviation", "aircraft", "airline", "sia engineering",
            "boeing", "airbus", "jet engine"
        ],
        "Engineering - Precision": [
            "precision engineering", "machining", "tooling", "mould", "precision parts",
            "cnc", "metal stamping"
        ],
        "Energy (Oil & Gas)": [
            "oil", "gas", "energy", "petroleum", "offshore", "upstream", "downstream",
            "refinery", "shell", "exxon", "chevron"
        ],
        "Environment & Water": [
            "environment", "water treatment", "waste management", "sustainability",
            "environmental consultant", "green energy"
        ],
        "Healthcare": [
            "healthcare", "hospital", "clinic", "medical", "nursing", "doctor",
            "pharmaceutical", "health services"
        ],
        "Hospitality & Tourism": [
            "hospitality", "hotel", "tourism", "travel", "restaurant", "resort",
            "marriott", "holiday inn", "airbnb"
        ],
        "F&B": ["food", "beverage", "f&b", "catering", "chef", "restaurant", "cafe"],
        "FMCG": [
            "fmcg", "fast-moving consumer goods", "consumer goods", "packaged goods",
            "procter & gamble", "unilever", "nestle"
        ],
        "Infocomm": ["infocomm", "telecom", "telco", "communications", "singtel", "starhub"],
        "Medical Technology": [
            "medtech", "medical devices", "healthcare technology", "medical equipment",
            "philips healthcare"
        ],
        "Marine & Offshore": [
            "marine", "offshore", "shipbuilding", "shipyard", "keppel", "sembcorp",
            "offshore engineering"
        ],
        "Retail": ["retail", "store", "merchandising", "e-commerce", "shopee", "lazada"],
        "Pharmaceutical/Biotech": [
            "pharmaceutical", "biotech", "pharma", "bioscience", "drug discovery",
            "clinical research", "gsk", "novartis"
        ],
        "Robotics": ["robotics", "robot", "automation", "robotic engineering", "industrial automation"],
        "Semiconductor": [
            "semiconductor", "wafer", "chip", "fab", "microelectronics", "foundry",
            "intel", "micron", "tsmc", "globalfoundries"
        ],
        "Supply Chain Mgt & Logistics": [
            "supply chain", "logistics", "warehouse", "distribution", "freight",
            "inventory management", "dhl", "fedex"
        ],
        "Telco": ["telecommunications", "telco", "mobile network", "5g", "broadband"],
        "Trading": ["trading", "commodity trading", "import export", "trader", "merchant"],
        "Others": []
    }

    @classmethod
    def classify(cls, text: str, entities: Optional[List["SpanAnnotation"]] = None) -> Dict[str, str]:
        """
        Returns predicted Function and Industry based on text and optional entity texts.
        """
        # Combine raw text with entity texts for richer matching
        full_text = text.lower()
        if entities:
            entity_texts = [e.text.lower() for e in entities if e.text]
            full_text += " " + " ".join(entity_texts)

        def _best_match(keywords_dict, default="others"):
            scores = {}
            for category, kw_list in keywords_dict.items():
                if not kw_list:  # skip empty lists (like 'others')
                    continue
                count = sum(full_text.count(kw) for kw in kw_list)
                if count > 0:
                    scores[category] = count
            if not scores:
                return default
            # Return category with highest score; ties keep first encountered
            return max(scores.items(), key=lambda x: x[1])[0]

        function = _best_match(cls.FUNCTION_KEYWORDS, default="others")
        industry = _best_match(cls.INDUSTRY_KEYWORDS, default="Others")
        return {"Function": function, "Industry": industry}


class EntityCategory(Enum):
    """
    📂 Top-level categories for entity types.
    
    Think of these as the departments in our fashion house:
    Personal is the face, Professional is the wardrobe,
    and Meta is the stage directions! 🎭
    """
    PERSONAL = auto()
    PROFESSIONAL = auto()
    EDUCATION = auto()
    SKILLS = auto()
    META = auto()


@dataclass
class EntityType:
    """
    🏷️ Definition of a single NER entity type.
    
    Each entity type has a name, category, validation rules,
    and instructions for annotators. Like a pattern card
    for a seamstress — tells you exactly how to cut the fabric! ✂️
    
    Attributes:
        name:             Unique identifier (e.g., "PERSON_NAME")
        label:            Human-readable label for the annotation UI
        category:         Which EntityCategory this belongs to
        description:      Annotation guideline — WHAT to label
        examples:         Example values annotators should look for
        boundary_rules:   HOW to decide where the entity starts/ends
        allows_nesting:   Can this entity contain other entities inside?
        nested_children:  Which entity types can be nested inside this one?
        color:            UI color for the annotation tool (hex)
        shortcut_key:     Keyboard shortcut for fast annotation
        validation_regex: Optional regex pattern for auto-validation
        is_required:      Must this entity be present in a valid resume?
        max_per_doc:      Maximum expected occurrences per resume
    """
    name: str
    label: str
    category: EntityCategory
    description: str
    examples: List[str] = field(default_factory=list)
    boundary_rules: str = ""
    allows_nesting: bool = False
    nested_children: List[str] = field(default_factory=list)
    color: str = "#7ee8fa"
    shortcut_key: str = ""
    validation_regex: Optional[str] = None
    is_required: bool = False
    max_per_doc: Optional[int] = None


class EntitySchema:
    """
    📋 THE MASTER ENTITY SCHEMA — Every entity type we extract!
    
    This is the BIBLE for annotators. It defines exactly what to label,
    how to handle boundaries, and what edge cases to watch for.
    
    Designed for Singapore/Malaysia resume formats with support for:
    - Multi-word entities (job titles, university names)
    - Date-first patterns (common in SG/MY resumes)
    - Bilingual content (English + Malay/Chinese)
    - CMFAS certifications, NRIC patterns, etc.
    
    Drama analogy: This is the casting sheet that tells you every
    character in the play, what they look like, and when they appear! 🎬
    """

    def __init__(self):
        """
        🎀 Initialize the schema with all entity types.
        
        Entity types are organized into 5 categories:
        1. PERSONAL  — Name, email, phone, DOB, location, nationality
        2. PROFESSIONAL — Job titles, companies, dates, descriptions
        3. EDUCATION — Degrees, institutions, fields of study
        4. SKILLS — Technical skills, soft skills, certifications
        5. META — Section headers, bullet markers, formatting tokens
        """
        self.entities: Dict[str, EntityType] = OrderedDict()
        self._build_schema()
    
    def _build_schema(self):
        """
        🏗️ Construct the full entity schema.
        
        ╔══════════════════════════════════════════════════════════╗
        ║  ANNOTATION GUIDELINE NOTES (for human annotators):     ║
        ║  - Label the MINIMUM span that captures the entity      ║
        ║  - Include titles (Mr., Dr.) in PERSON_NAME             ║
        ║  - Separate ORG from LOC even when adjacent             ║
        ║  - "Present" / "Current" → label as WORK_DATE           ║
        ║  - Entire bullet text → JOB_DESCRIPTION                 ║
        ║  - Skills in context → still label as SKILL             ║
        ╚══════════════════════════════════════════════════════════╝
        """
        
        # =================================================================
        # 👤 PERSONAL ENTITIES
        # =================================================================
        
        self._add(EntityType(
            name="PERSON_NAME",
            label="Person Name",
            category=EntityCategory.PERSONAL,
            description=(
                "Full name of the resume owner. Include honorifics (Mr., Mrs., Dr.) "
                "and suffixes (Jr., PhD) when directly attached. For Chinese/Malay names, "
                "include the FULL name as written (e.g., 'Tan Ah Kow', 'Muhammad bin Ali')."
            ),
            examples=["John Doe", "Tan Mei Ling", "Muhammad bin Ali", "Dr. Sarah Chen"],
            boundary_rules=(
                "START: First letter of name (or honorific if present). "
                "END: Last letter of surname/suffix. "
                "EXCLUDE: Contact info, job titles, addresses that follow the name."
            ),
            color="#7ee8fa",
            shortcut_key="1",
            is_required=True,
            max_per_doc=2  # Main name + possibly a reference name
        ))

        self._add(EntityType(
            name="EMAIL",
            label="Email Address",
            category=EntityCategory.PERSONAL,
            description="Email addresses. Include the complete address from first char to last.",
            examples=["john.doe@gmail.com", "tanml@company.com.sg"],
            boundary_rules="START: First character. END: Last character after TLD.",
            color="#56d364",
            shortcut_key="2",
            validation_regex=r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$',
            is_required=True,
            max_per_doc=3
        ))

        self._add(EntityType(
            name="PHONE",
            label="Phone Number",
            category=EntityCategory.PERSONAL,
            description=(
                "Phone numbers including country codes. Singapore: +65 XXXX XXXX (8 digits). "
                "Malaysia: +60 XX-XXXX XXXX (9-10 digits). Include the full number with "
                "any formatting (dashes, spaces, parentheses)."
            ),
            examples=["+65 9123 4567", "9123-4567", "+60 12-345 6789", "(65) 91234567"],
            boundary_rules=(
                "START: '+' or first digit. END: Last digit. "
                "INCLUDE: Country codes, area codes, formatting chars between digits. "
                "EXCLUDE: Labels like 'Phone:', 'Mobile:', 'H/P:'."
            ),
            color="#e3b341",
            shortcut_key="3",
            is_required=True,
            max_per_doc=3
        ))

        self._add(EntityType(
            name="DATE_OF_BIRTH",
            label="Date of Birth",
            category=EntityCategory.PERSONAL,
            description=(
                "Birth date in any format. Label the complete date expression. "
                "SG/MY resumes often use DD/MM/YYYY or DD Mon YYYY format."
            ),
            examples=["15/03/1990", "15 March 1990", "1990-03-15", "15-Mar-90"],
            boundary_rules="START: First char of date. END: Last char of date or year.",
            color="#f0883e",
            shortcut_key="4",
            max_per_doc=1
        ))

        self._add(EntityType(
            name="LOCATION",
            label="Location/Address",
            category=EntityCategory.PERSONAL,
            description=(
                "Physical addresses, city names, or location info. "
                "For 'Google, Mountain View' — label 'Mountain View' as LOCATION separately "
                "from 'Google' as ORGANIZATION. ALWAYS separate ORG from LOC."
            ),
            examples=["Singapore", "Kuala Lumpur", "Blk 123 Ang Mo Kio Ave 6 #12-345"],
            boundary_rules=(
                "START: First char of location. END: Last char (including postal code). "
                "SEPARATE from organization names even when adjacent."
            ),
            color="#a371f7",
            shortcut_key="5",
            allows_nesting=False
        ))

        self._add(EntityType(
            name="NATIONALITY",
            label="Nationality",
            category=EntityCategory.PERSONAL,
            description="Citizenship or nationality. Label the nationality word only.",
            examples=["Singaporean", "Malaysian", "Singapore Citizen", "PR"],
            boundary_rules="Label the complete nationality expression.",
            color="#bc8cff",
            max_per_doc=2
        ))

        self._add(EntityType(
            name="NRIC_ID",
            label="NRIC/ID Number",
            category=EntityCategory.PERSONAL,
            description=(
                "Singapore NRIC (S/T/F/G + 7 digits + letter) or "
                "Malaysian IC number (YYMMDD-PB-XXXX). Partially redacted IDs too."
            ),
            examples=["S1234567D", "T0012345G", "900315-14-5678", "S****567D"],
            validation_regex=r'^[STFGM]\d{7}[A-Z]$|^\d{6}-\d{2}-\d{4}$',
            color="#da3633",
            max_per_doc=1
        ))

        self._add(EntityType(
            name="GENDER",
            label="Gender",
            category=EntityCategory.PERSONAL,
            description=(
                "The gender of the resume owner. SG/MY resumes commonly combine "
                "gender and age in a single field — label the ENTIRE combined value "
                "as one GENDER span.\n"
                "  - 'Female/22'  → label the full string as GENDER\n"
                "  - 'Male / 35'  → label the full string as GENDER\n"
                "  - 'Female'     → label as GENDER (age may be elsewhere)\n"
                "  - 'M'          → label as GENDER\n"
                "  - 'Male, 28'   → label the full string as GENDER\n"
                "Do NOT split gender and age into separate spans — treat the "
                "combined 'Gender/Age' field value as a single entity."
            ),
            examples=[
                "Female", "Male", "Female/22", "Male / 35",
                "F", "M", "Female, 28", "Male, Age 30"
            ],
            boundary_rules=(
                "START: First character of the gender value (after any label like 'Gender:').\n"
                "END: Last character — include any age portion if written as 'Gender/Age'.\n"
                "EXCLUDE: The field label itself (e.g. 'Gender:', 'Gender/Age:').\n"
                "INCLUDE: The age number if it appears inline as part of 'Female/22' format."
            ),
            color="#f778ba",   # Pink — distinct from all existing personal entity colours
            max_per_doc=1
        ))

        self._add(EntityType(
            name="MARITAL_STATUS",
            label="Marital Status",
            category=EntityCategory.PERSONAL,
            description=(
                "The marital or relationship status of the resume owner. "
                "Common in SG/MY resumes as a dedicated field.\n"
                "  - 'Single'      → label as MARITAL_STATUS\n"
                "  - 'Married'     → label as MARITAL_STATUS\n"
                "  - 'Divorced'    → label as MARITAL_STATUS\n"
                "  - 'Widowed'     → label as MARITAL_STATUS\n"
                "  - 'Single / Not Married' → label the full phrase\n"
                "Label only the status VALUE, not the field label ('Status:')."
            ),
            examples=[
                "Single", "Married", "Divorced", "Widowed",
                "Single / Not Married", "Married with 2 children"
            ],
            boundary_rules=(
                "START: First character of the status value (after 'Status:' or 'Marital Status:').\n"
                "END: Last character of the status phrase.\n"
                "EXCLUDE: The field label (e.g. 'Status:', 'Marital Status:').\n"
                "INCLUDE: Any qualifiers written inline (e.g. 'Married with children')."
            ),
            color="#a5d6ff",   # Light blue — calm and distinct from GENDER pink
            max_per_doc=1
        ))

        self._add(EntityType(
            name="EXPECTED_LOCATION",
            label="Expected Location",
            category=EntityCategory.PERSONAL,
            description=(
                "Location the candidate is willing to relocate to, or their preferred "
                "work location. Appears in resume summaries, profile headers, or a "
                "dedicated \'Preferred Location\' / \'Willing to Relocate\' field.\n"
                "  - \'Willing to relocate to Singapore\'  → label \'Singapore\'\n"
                "  - \'Open to: KL, Singapore, Remote\'    → THREE separate spans\n"
                "  - \'Preferred location: Kuala Lumpur\'  → label \'Kuala Lumpur\'\n"
                "EXCLUDE the field label itself (\'Preferred location:\', "
                "\'Willing to relocate to:\')."
            ),
            examples=[
                "Singapore", "Kuala Lumpur", "Remote",
                "KL or Singapore", "Anywhere in Malaysia"
            ],
            boundary_rules=(
                "START: First character of the location value (not the label prefix).\n"
                "END: Last character of the location value.\n"
                "EXCLUDE: Prefix phrases like \'Willing to relocate to\', \'Preferred:\'."
            ),
            color="#58a6ff",   # Blue — distinct from LOCATION purple (#a371f7)
            shortcut_key="E",
            max_per_doc=5
        ))

        # =================================================================
        # 💼 PROFESSIONAL / WORK EXPERIENCE ENTITIES
        # =================================================================

        self._add(EntityType(
            name="JOB_TITLE",
            label="Job Title",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "Official job title or position. Include the FULL title even if long. "
                "Examples: 'Senior Software Engineer', 'Assistant Vice President, Operations'. "
                "For BIO tagging, multi-word titles are B-JOB_TITLE I-JOB_TITLE I-JOB_TITLE."
            ),
            examples=[
                "Software Engineer", "Financial Consultant",
                "Senior Vice President, Risk Management",
                "Assistant Manager (Operations)"
            ],
            boundary_rules=(
                "START: First word of title (include 'Senior', 'Lead', 'Junior'). "
                "END: Last word of title. "
                "INCLUDE: Seniority prefixes, department qualifiers in parentheses. "
                "EXCLUDE: Company name, dates, 'at', 'in'."
            ),
            color="#7ee8fa",
            shortcut_key="6",
            allows_nesting=False
        ))

        self._add(EntityType(
            name="ORGANIZATION",
            label="Company/Organization",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "Company, organization, or employer name. For 'Google, Mountain View', "
                "label 'Google' as ORGANIZATION and 'Mountain View' as LOCATION separately."
            ),
            examples=[
                "Google", "DBS Bank", "Ministry of Education",
                "National University Hospital", "Grab Holdings"
            ],
            boundary_rules=(
                "START: First word of organization name. "
                "END: Last word (include 'Pte Ltd', 'Sdn Bhd', 'Inc'). "
                "SEPARATE from location even when comma-separated."
            ),
            color="#eeb8ff",
            shortcut_key="7",
            allows_nesting=True,
            nested_children=["LOCATION"]  # "Google, Mountain View" → ORG contains LOC
        ))

        self._add(EntityType(
            name="WORK_DATE",
            label="Employment Date",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "Date or date range for employment. THIS IS CRITICAL for edge cases:\n"
                "  - 'Present' / 'Current' / 'Now' → label as WORK_DATE\n"
                "  - 'Feb 2016 to Present' → ONE span, entire range is WORK_DATE\n"
                "  - 'Jan 2020 – Dec 2023' → ONE span (include the separator)\n"
                "  - 'Since 2019' → label as WORK_DATE"
            ),
            examples=[
                "Feb 2016 to Present", "Jan 2020 – Dec 2023",
                "2019 - Current", "Since 2018", "Mar 2015 to Jul 2018"
            ],
            boundary_rules=(
                "START: First char of start date (or 'Since'). "
                "END: Last char of end date (including 'Present'/'Current'). "
                "INCLUDE: 'to', '–', '-', 'till', 'until' connectors. "
                "INCLUDE: 'Present', 'Current', 'Now', 'Ongoing' as valid end dates."
            ),
            color="#f0883e",
            shortcut_key="8"
        ))

        self._add(EntityType(
            name="WORK_LOCATION",
            label="Work Location",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "The geographic location associated with a specific job or role. "
                "Different from LOCATION (personal/home address) — this is WHERE "
                "the candidate WORKED, not where they live.\n"
                "  - 'Software Engineer at Google, Singapore' → 'Singapore' is WORK_LOCATION\n"
                "  - 'DBS Bank — Changi Business Park'        → 'Changi Business Park'\n"
                "  - 'Based in KL office'                     → 'KL'\n"
                "  - 'Remote (Singapore)'                     → 'Remote (Singapore)'\n"
                "Often appears next to company name or in job header lines. "
                "If a location appears in the work experience section and clearly "
                "belongs to a role, label it WORK_LOCATION instead of LOCATION."
            ),
            examples=[
                "Singapore", "Kuala Lumpur", "Changi Business Park",
                "Cyberjaya", "One North, Singapore", "Remote",
                "Penang", "Johor Bahru", "CBD Area"
            ],
            boundary_rules=(
                "START: First character of the work location.\n"
                "END: Last character (include area/district names).\n"
                "EXCLUDE: Company name — label separately as ORGANIZATION.\n"
                "EXCLUDE: Country codes or phone-style location markers.\n"
                "INCLUDE: 'Remote' if specified as work arrangement location."
            ),
            color="#d2a8ff",   # Soft purple — distinct from LOCATION (#a371f7) and ORG (#eeb8ff)
            shortcut_key="W",
            allows_nesting=False
        ))

        self._add(EntityType(
            name="JOB_DESCRIPTION",
            label="Job Description",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "Description of job responsibilities or achievements within a role. "
                "Label the ENTIRE bullet point or sentence as one JOB_DESCRIPTION span. "
                "Do NOT split into individual keywords — the full context matters.\n"
                "  - '• Managed team of 10 engineers' → entire bullet is JOB_DESCRIPTION\n"
                "  - 'Responsible for end-to-end testing' → entire phrase"
            ),
            examples=[
                "Managed a team of 10 software engineers across 3 time zones",
                "Developed RESTful APIs using Python Flask serving 1M+ requests/day",
                "Conducted risk assessments for derivatives trading desk"
            ],
            boundary_rules=(
                "START: First word AFTER the bullet marker (•, -, *, ▪). "
                "END: Last word before next bullet or section. "
                "INCLUDE: Metrics, percentages, technologies mentioned within. "
                "EXCLUDE: The bullet character itself."
            ),
            color="#8b949e",
            allows_nesting=True,  # 🆕 Allow children inside!
            nested_children=["SKILL", "SOFT_SKILL", "ORGANIZATION", "METRIC", "CERTIFICATION"]
        ))

        self._add(EntityType(
            name="METRIC",
            label="Quantitative Metric",
            category=EntityCategory.PROFESSIONAL,
            description="Numeric achievements or KPIs within job descriptions.",
            examples=["$2.5M revenue", "30% improvement", "10,000+ users", "team of 15"],
            color="#f778ba",
            allows_nesting=False
        ))

        self._add(EntityType(
            name="FUNCTION",
            label="Function / Dept",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "The broad functional area or department the candidate works in "
                "or is applying to. This is NOT the job title — it is the "
                "DEPARTMENT or FUNCTIONAL CATEGORY.\n"
                "  - \'Technology / IT\'        → FUNCTION\n"
                "  - \'Finance & Accounting\'   → FUNCTION\n"
                "  - \'Sales & Marketing\'      → FUNCTION\n"
                "  - \'Human Resources\'        → FUNCTION\n"
                "  - \'Risk Management\'        → FUNCTION\n"
                "Often found in profile headers, objective statements, or LinkedIn-style "
                "summaries. EXCLUDE job titles — those go in JOB_TITLE."
            ),
            examples=[
                "Technology", "Finance", "Sales", "Marketing",
                "Human Resources", "Operations", "Engineering",
                "Risk Management", "Compliance", "Product Management",
                "Finance & Accounting", "Sales & Marketing"
            ],
            boundary_rules=(
                "START: First word of the functional area.\n"
                "END: Last word (include connectors like \'&\', \'and\').\n"
                "INCLUDE: Compound names (\'Sales & Marketing\', \'Finance & Accounting\').\n"
                "EXCLUDE: Seniority levels (\'Head of\', \'Senior\') — those are in JOB_TITLE."
            ),
            color="#39d353",   # Bright green — distinct from all existing professional colours
            shortcut_key="F",
            max_per_doc=3
        ))

        self._add(EntityType(
            name="INDUSTRY",
            label="Industry",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "The industry sector the candidate has experience in or is targeting. "
                "This is the SECTOR — not the company name, not the job function.\n"
                "  - \'Banking & Finance\'           → INDUSTRY\n"
                "  - \'Information Technology\'      → INDUSTRY\n"
                "  - \'Healthcare / Pharmaceutical\'  → INDUSTRY\n"
                "  - \'Retail\'                       → INDUSTRY\n"
                "  - \'Manufacturing\'                → INDUSTRY\n"
                "Often appears in a resume objective (\'X years in [Industry]\') or "
                "in a candidate profile header. Can appear multiple times if the "
                "candidate has cross-sector experience."
            ),
            examples=[
                "Banking & Finance", "Information Technology", "Healthcare",
                "Retail", "Manufacturing", "Logistics & Supply Chain",
                "Insurance", "Real Estate", "Oil & Gas", "Education"
            ],
            boundary_rules=(
                "START: First word of the industry name.\n"
                "END: Last word (include connectors like \'&\', \'and\', \'/\').\n"
                "EXCLUDE: Company names — label those as ORGANIZATION.\n"
                "INCLUDE: Compound sector names (\'Banking & Finance\', \'Oil & Gas\')."
            ),
            color="#f1a340",   # Amber/orange — warm, distinct from all existing colours
            shortcut_key="I",
            max_per_doc=5
        ))

        self._add(EntityType(
            name="PROJECT_TITLE",
            label="Project Title",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "The name or title of a project listed in a \'Projects\' or "
                "\'Project Experience\' section. Label only the project name — "
                "not the description, dates, or technologies.\n"
                "  - \'E-Commerce Platform Redesign\'       → PROJECT_TITLE\n"
                "  - \'Predictive Analytics Dashboard\'     → PROJECT_TITLE\n"
                "  - \'Final Year Project: Smart Parking\'  → label the full title\n"
                "For academic final year projects, include \'Final Year Project\' "
                "as part of the title if it forms the display name."
            ),
            examples=[
                "E-Commerce Platform Redesign",
                "Predictive Analytics Dashboard",
                "Smart Parking System",
                "NUS FYP: AI Resume Screener",
                "Capstone: Supply Chain Optimisation"
            ],
            boundary_rules=(
                "START: First word of the project name.\n"
                "END: Last word of the project name.\n"
                "EXCLUDE: Technology stack, dates — label those as SKILL / WORK_DATE.\n"
                "EXCLUDE: The standalone label \'Project:\' if used as a prefix."
            ),
            color="#d29922",   # Gold — evokes achievement / trophy 🏆
            shortcut_key="P",
            max_per_doc=20
        ))

        self._add(EntityType(
            name="PROJECT_DESCRIPTION",
            label="Project Description",
            category=EntityCategory.PROFESSIONAL,
            description=(
                "A full description sentence or bullet point describing what was built, "
                "achieved, or contributed in a project. Same rule as JOB_DESCRIPTION: "
                "label the ENTIRE bullet as one span — do NOT split by keyword.\n"
                "  - \'Built a REST API serving 50K requests/day\'       → full sentence\n"
                "  - \'Reduced model inference time by 40%\'              → full sentence\n"
                "  - \'Led a team of 5 to deliver MVP in 3 months\'      → full sentence"
            ),
            examples=[
                "Built a REST API serving 50K daily requests using Python Flask",
                "Reduced model inference time by 40% through quantisation",
                "Led a cross-functional team of 5 to deliver MVP in 3 months",
                "Achieved 98% unit test coverage using pytest and GitHub Actions"
            ],
            boundary_rules=(
                "START: First word of the description (after any bullet marker •, -, *).\n"
                "END: Last word of the description sentence or bullet point.\n"
                "EXCLUDE: The bullet marker itself (•, -, *).\n"
                "INCLUDE: Technologies, metrics, team sizes inline — they are part of context."
            ),
            color="#b08800",   # Darker gold — visually paired with PROJECT_TITLE
            shortcut_key="D",
            allows_nesting=True,
            nested_children=["SKILL", "METRIC"],
            max_per_doc=50
        ))

        # =================================================================
        # 🎓 EDUCATION ENTITIES
        # =================================================================

        self._add(EntityType(
            name="DEGREE",
            label="Degree/Qualification",
            category=EntityCategory.EDUCATION,
            description=(
                "Academic degree or qualification name. Include the full degree title."
            ),
            examples=[
                "Bachelor of Science", "Master of Business Administration",
                "Diploma in Information Technology", "GCE 'O' Level",
                "PhD in Computer Science", "NITEC in Electronics"
            ],
            boundary_rules=(
                "START: First word of degree (Bachelor, Master, Diploma, etc.). "
                "END: Last word of degree name including field. "
                "INCLUDE: 'in', 'of' connectors within the degree title. "
                "EXCLUDE: Institution name, graduation date."
            ),
            color="#56d364",
            shortcut_key="9",
            allows_nesting=True,
            nested_children=["FIELD_OF_STUDY"]
        ))

        self._add(EntityType(
            name="INSTITUTION",
            label="Educational Institution",
            category=EntityCategory.EDUCATION,
            description="University, school, polytechnic, or training provider name.",
            examples=[
                "National University of Singapore", "NTU",
                "Singapore Polytechnic", "Universiti Malaya",
                "ITE College East", "Raffles Institution"
            ],
            color="#7ee8fa",
            allows_nesting=True,
            nested_children=["LOCATION"]
        ))

        self._add(EntityType(
            name="FIELD_OF_STUDY",
            label="Field of Study",
            category=EntityCategory.EDUCATION,
            description="Major, specialization, or field within a degree.",
            examples=["Computer Science", "Electrical Engineering", "Finance", "Nursing"],
            color="#a371f7"
        ))

        self._add(EntityType(
            name="EDU_DATE",
            label="Education Date",
            category=EntityCategory.EDUCATION,
            description=(
                "Date or date range for education. Same edge case rules as WORK_DATE:\n"
                "  - 'Expected 2025' or 'Expected Graduation: Dec 2025' → EDU_DATE\n"
                "  - '2018 - 2022' → entire range as one EDU_DATE span"
            ),
            examples=["2018 - 2022", "Expected Dec 2025", "Graduated 2019", "2015"],
            color="#f0883e"
        ))

        self._add(EntityType(
            name="GPA",
            label="GPA/Grade",
            category=EntityCategory.EDUCATION,
            description="Academic grades, GPA, class of honors, or CGPA.",
            examples=["3.85/4.0", "First Class Honours", "CGPA: 3.5", "Dean's List"],
            color="#e3b341"
        ))

        # =================================================================
        # 🛠️ SKILLS & CERTIFICATIONS
        # =================================================================

        self._add(EntityType(
            name="SKILL",
            label="Skill",
            category=EntityCategory.SKILLS,
            description=(
                "Technical or professional skill. Label each skill INDIVIDUALLY, "
                "even when in a comma-separated list.\n"
                "  - 'Python, Java, SQL' → THREE separate SKILL spans\n"
                "  - 'machine learning' → ONE SKILL span (2 tokens: B-SKILL I-SKILL)\n"
                "  - 'Microsoft Office Suite' → ONE SKILL span (3 tokens)"
            ),
            examples=[
                "Python", "Java", "SQL", "machine learning",
                "project management", "financial analysis",
                "Microsoft Office Suite", "Docker", "Kubernetes"
            ],
            boundary_rules=(
                "START: First word of the skill. "
                "END: Last word of the skill. "
                "SPLIT: Comma-separated skills into individual spans. "
                "KEEP TOGETHER: Multi-word skills (machine learning, data analysis). "
                "EXCLUDE: Proficiency levels ('Advanced', 'Intermediate')."
            ),
            color="#eeb8ff",
            shortcut_key="0"
        ))

        self._add(EntityType(
            name="SKILL_CATEGORY",
            label="Skill Category Header",
            category=EntityCategory.SKILLS,
            description="Category headers in skills sections (e.g., 'Programming Languages:').",
            examples=[
                "Programming Languages", "Frameworks", "Databases",
                "Soft Skills", "Tools & Technologies"
            ],
            color="#8b949e"
        ))

        self._add(EntityType(
            name="SOFT_SKILL",
            label="Soft Skill",
            category=EntityCategory.SKILLS,
            description="Non-technical, interpersonal, or transferable skills.",
            examples=["leadership", "communication", "team building", "problem solving"],
            color="#bc8cff"
        ))

        self._add(EntityType(
            name="CERTIFICATION",
            label="Certification",
            category=EntityCategory.SKILLS,
            description=(
                "Professional certifications, licenses, or accreditations. "
                "Singapore-specific: CMFAS modules, BCA certifications, WSQ."
            ),
            examples=[
                "AWS Certified Solutions Architect", "PMP",
                "CMFAS Module 5", "WSQ Advanced Certificate",
                "Certified Scrum Master", "CISSP"
            ],
            color="#56d364",
            allows_nesting=True,
            nested_children=["CERT_ISSUER", "CERT_DATE"]
        ))

        self._add(EntityType(
            name="CERT_ISSUER",
            label="Certification Issuer",
            category=EntityCategory.SKILLS,
            description="Organization that issued the certification.",
            examples=["AWS", "PMI", "MAS", "Scrum Alliance", "CompTIA"],
            color="#7ee8fa"
        ))

        self._add(EntityType(
            name="CERT_DATE",
            label="Certification Date",
            category=EntityCategory.SKILLS,
            description="Date the certification was earned or expires.",
            examples=["Issued Mar 2023", "Expires Dec 2026", "2022"],
            color="#f0883e"
        ))

        self._add(EntityType(
            name="LANGUAGE_SKILL",
            label="Language",
            category=EntityCategory.SKILLS,
            description="Spoken/written language abilities.",
            examples=["English", "Mandarin", "Malay", "Tamil", "Japanese"],
            color="#a371f7"
        ))

        self._add(EntityType(
            name="PROFICIENCY",
            label="Proficiency Level",
            category=EntityCategory.SKILLS,
            description="Proficiency or skill level descriptors.",
            examples=["Native", "Fluent", "Intermediate", "Advanced", "Beginner"],
            color="#8b949e"
        ))

        # =================================================================
        # 📋 META / STRUCTURAL ENTITIES
        # =================================================================

        self._add(EntityType(
            name="SECTION_HEADER",
            label="Section Header",
            category=EntityCategory.META,
            description=(
                "Section headers that divide resume into logical parts. "
                "Label the EXACT header text, not the content below it."
            ),
            examples=[
                "WORK EXPERIENCE", "Education", "SKILLS",
                "PROFESSIONAL SUMMARY", "CERTIFICATIONS", "REFERENCES"
            ],
            color="#6e7681",
            shortcut_key="H"
        ))

        self._add(EntityType(
            name="SUMMARY_TEXT",
            label="Summary/Objective",
            category=EntityCategory.META,
            description="Professional summary, objective, or profile statement text.",
            examples=["Results-driven software engineer with 5+ years experience..."],
            color="#8b949e"
        ))

    def _add(self, entity_type: EntityType):
        """Register an entity type in the schema."""
        self.entities[entity_type.name] = entity_type
    
    def get(self, name: str) -> Optional[EntityType]:
        """Get an entity type by name."""
        return self.entities.get(name)
    
    def get_all(self) -> List[EntityType]:
        """Get all entity types."""
        return list(self.entities.values())
    
    def get_by_category(self, category: EntityCategory) -> List[EntityType]:
        """Get all entity types in a specific category."""
        return [e for e in self.entities.values() if e.category == category]
    
    def get_nestable(self) -> List[EntityType]:
        """Get all entity types that allow nesting."""
        return [e for e in self.entities.values() if e.allows_nesting]
    
    def get_bio_tags(self) -> List[str]:
        """
        Generate the full BIO tag set from the schema.
        
        For each entity type X, generates: B-X, I-X
        Plus the universal O (Outside) tag.
        
        Returns:
            Sorted list of all valid BIO tags.
        """
        tags = ["O"]
        for name in self.entities:
            tags.append(f"B-{name}")
            tags.append(f"I-{name}")
        return sorted(tags)
    
    def get_tag_to_id(self) -> Dict[str, int]:
        """Map each BIO tag to a numeric ID (for model training)."""
        return {tag: i for i, tag in enumerate(self.get_bio_tags())}
    
    def get_id_to_tag(self) -> Dict[int, str]:
        """Map numeric IDs back to BIO tag strings."""
        return {i: tag for i, tag in enumerate(self.get_bio_tags())}
    
    def get_color_map(self) -> Dict[str, str]:
        """Map entity names to their UI colors."""
        return {name: e.color for name, e in self.entities.items()}
    
    def export_schema(self) -> Dict:
        """
        Export the full schema as a JSON-serializable dict.
        Useful for saving annotation guidelines or sending to the UI.
        """
        return {
            "version": "1.0",
            "total_entity_types": len(self.entities),
            "total_bio_tags": len(self.get_bio_tags()),
            "categories": [c.name for c in EntityCategory],
            "entities": {
                name: {
                    "label": e.label,
                    "category": e.category.name,
                    "description": e.description,
                    "examples": e.examples,
                    "boundary_rules": e.boundary_rules,
                    "allows_nesting": e.allows_nesting,
                    "nested_children": e.nested_children,
                    "color": e.color,
                    "shortcut_key": e.shortcut_key,
                    "is_required": e.is_required,
                    "max_per_doc": e.max_per_doc,
                }
                for name, e in self.entities.items()
            },
            "bio_tags": self.get_bio_tags()
        }


# =============================================================================
# ✂️ STEP B (Part 1): RESUME-AWARE TOKENIZER
# Splits text into tokens while preserving resume structure.
# =============================================================================

@dataclass
class Token:
    """
    A single token with its position in the original text.
    
    Preserving char_start/char_end is CRITICAL for mapping between
    the tokenized sequence and the original text spans! Like keeping
    the pattern markings when cutting fabric — you need to know
    where each piece came from! ✂️
    """
    text: str
    char_start: int      # Start position in original text
    char_end: int        # End position in original text (exclusive)
    line_number: int     # Which line this token is on
    is_punctuation: bool = False
    is_bullet: bool = False
    is_newline: bool = False


class ResumeTokenizer:
    """
    ✂️ Smart tokenizer designed for resume text.
    
    Unlike standard NLP tokenizers, this one:
    - Preserves line boundaries (critical for section detection)
    - Handles bullet point characters (•, ▪, -, *)
    - Keeps email addresses and URLs as single tokens
    - Preserves phone numbers with formatting as single tokens
    - Handles Singapore-specific patterns (Blk, #XX-XXX)
    - Preserves date ranges as individual components
    - Tracks exact character positions for span alignment
    
    Why not just split on whitespace? Because 'john.doe@email.com'
    would become 3 tokens, and '+65 9123 4567' would become 3 tokens.
    That DESTROYS entity boundaries! 🎭💔
    """
    
    # Patterns that should be kept as single tokens
    # Order matters — earlier patterns take priority
    ATOMIC_PATTERNS = [
        # Email addresses (keep as one token)
        (r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', 'email'),
        # URLs
        (r'https?://[^\s]+', 'url'),
        # Phone with country code: +65 9123 4567 or +60 12-345 6789
        (r'\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}', 'phone'),
        # NRIC: S1234567D
        (r'[STFGM]\d{7}[A-Z]', 'nric'),
        # Date patterns: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
        (r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', 'date'),
        (r'\d{4}[/\-]\d{1,2}[/\-]\d{1,2}', 'date'),
        # Singapore block numbers: Blk 123 or Block 123
        (r'(?:Blk|Block)\s*\d+', 'block'),
        # Unit numbers: #12-345 or #12-3456
        (r'#\d{1,3}-\d{2,4}', 'unit'),
        # Postal codes (Singapore 6-digit)
        (r'(?:Singapore|S)\s*\d{6}', 'postal'),
        # GPA: 3.85/4.0 or CGPA: 3.5
        (r'\d\.\d{1,2}/\d\.\d', 'gpa'),
    ]
    
    # Bullet point characters to detect
    BULLET_CHARS = set('•▪▸▹►◆◇○●■□–—·⁃➤➢✓✔☐☑★')
    
    def __init__(self):
        """Compile atomic patterns for efficiency."""
        self._compiled_atomics = [
            (re.compile(pattern), label)
            for pattern, label in self.ATOMIC_PATTERNS
        ]
    
    def tokenize(self, text: str) -> List[Token]:
        """
        ✂️ Tokenize resume text into a list of Token objects.
        
        Algorithm:
        1. Split text into lines (preserving line numbers)
        2. Within each line, find atomic patterns (emails, phones, etc.)
        3. Split remaining text on whitespace
        4. Track character positions for each token
        
        Args:
            text: Raw resume text.
            
        Returns:
            List of Token objects with preserved positions.
        """
        if not text:
            return []
        
        tokens = []
        lines = text.split('\n')
        char_offset = 0
        
        for line_num, line in enumerate(lines):
            if not line.strip():
                # Empty line — still track position
                char_offset += len(line) + 1  # +1 for the \n
                continue
            
            line_tokens = self._tokenize_line(line, char_offset, line_num)
            tokens.extend(line_tokens)
            char_offset += len(line) + 1  # +1 for the \n
        
        return tokens
    
    def _tokenize_line(
        self, line: str, char_offset: int, line_num: int
    ) -> List[Token]:
        """
        Tokenize a single line, handling atomic patterns first.
        
        Strategy: Find all atomic patterns, mark their positions as
        "protected", then split the unprotected regions on whitespace.
        """
        tokens = []
        protected_ranges = []  # (start, end) positions that are atomic tokens
        
        # Step 1: Find all atomic patterns in this line
        atomic_tokens = []
        for compiled_re, label in self._compiled_atomics:
            for match in compiled_re.finditer(line):
                start, end = match.start(), match.end()
                # Check for overlap with existing protected ranges
                overlaps = any(
                    not (end <= ps or start >= pe)
                    for ps, pe in protected_ranges
                )
                if not overlaps:
                    protected_ranges.append((start, end))
                    atomic_tokens.append((start, end, match.group(), label))
        
        # Sort atomic tokens by position
        atomic_tokens.sort(key=lambda x: x[0])
        
        # Step 2: Process the line, yielding atomic tokens at their positions
        # and splitting non-atomic regions on whitespace
        pos = 0
        atomic_idx = 0
        
        while pos < len(line):
            # Check if we're at an atomic token
            if atomic_idx < len(atomic_tokens) and pos == atomic_tokens[atomic_idx][0]:
                a_start, a_end, a_text, a_label = atomic_tokens[atomic_idx]
                tokens.append(Token(
                    text=a_text,
                    char_start=char_offset + a_start,
                    char_end=char_offset + a_end,
                    line_number=line_num,
                    is_punctuation=False,
                    is_bullet=False
                ))
                pos = a_end
                atomic_idx += 1
                continue
            
            # Skip to next atomic token or end of line
            next_atomic_start = (
                atomic_tokens[atomic_idx][0]
                if atomic_idx < len(atomic_tokens)
                else len(line)
            )
            
            # Process the gap between pos and next_atomic_start
            gap = line[pos:next_atomic_start]
            gap_offset = pos
            
            # Split gap on whitespace
            for word_match in re.finditer(r'\S+', gap):
                word = word_match.group()
                word_start = gap_offset + word_match.start()
                word_end = gap_offset + word_match.end()
                
                # Check if it's a bullet character
                is_bullet = len(word) == 1 and word in self.BULLET_CHARS
                
                # Check if it's punctuation-only
                is_punct = all(
                    unicodedata.category(c).startswith('P') or c in self.BULLET_CHARS
                    for c in word
                ) and not is_bullet
                
                tokens.append(Token(
                    text=word,
                    char_start=char_offset + word_start,
                    char_end=char_offset + word_end,
                    line_number=line_num,
                    is_punctuation=is_punct,
                    is_bullet=is_bullet
                ))
            
            pos = next_atomic_start
        
        return tokens


# =============================================================================
# 🏷️ STEP B (Part 2): BIO TAGGING ENGINE
# Converts span-based annotations to BIO-tagged token sequences.
# =============================================================================

@dataclass
class SpanAnnotation:
    """
    A single span-based annotation.
    
    Spans are the native format from the annotation tool — they say
    "characters 45 to 67 are a PERSON_NAME". The BIO tagger converts
    these to per-token tags for NER model training.
    
    Attributes:
        entity_type: Name of the entity type (must match schema)
        char_start:  Start character position in the original text
        char_end:    End character position (exclusive)
        text:        The annotated text (for display/validation)
        layer:       Annotation layer (0=primary, 1+=nested layers)
        confidence:  Auto-annotation confidence (1.0 for human annotations)
        annotator:   Who created this annotation (human/auto/model)
    """
    entity_type: str
    char_start: int
    char_end: int
    text: str = ""
    layer: int = 0
    confidence: float = 1.0
    annotator: str = "human"


@dataclass
class AnnotatedDocument:
    """
    A fully annotated resume document.
    
    Contains the raw text, token sequence, span annotations (possibly
    across multiple layers for nested entities), and metadata.
    """
    doc_id: str                              # Unique document identifier
    candidate_id: Optional[int] = None       # Links to resume_extractions.db
    raw_text: str = ""                       # The original resume text
    tokens: List[Token] = field(default_factory=list)
    annotations: List[SpanAnnotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotator: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = "pending"                  # pending, in_progress, completed, reviewed


class BIOTagger:
    """
    🏷️ Converts span annotations to BIO-tagged token sequences.
    
    The BIO (Beginning-Inside-Outside) tagging scheme is the standard
    for NER training. This class handles the conversion from
    character-level spans to token-level tags, including:
    
    - Proper B/I assignment for multi-word entities
    - Conflict resolution when spans overlap at token boundaries
    - Multiple annotation layers for nested entities
    - Validation against the entity schema
    
    Drama analogy: If the annotations are the costume sketches,
    the BIO tagger is the pattern maker who translates them into
    cutting instructions for each individual piece of fabric! ✂️🏷️
    
    BIO Tag Meanings:
        B-ENTITY  = This token is the BEGINNING of an entity
        I-ENTITY  = This token is INSIDE (continuation of) an entity
        O         = This token is OUTSIDE any entity (background)
    
    Example:
        Text:    "Senior Software Engineer at Google"
        Tokens:  [Senior, Software, Engineer, at, Google]
        Tags:    [B-JOB_TITLE, I-JOB_TITLE, I-JOB_TITLE, O, B-ORGANIZATION]
    """
    
    def __init__(self, schema: EntitySchema):
        self.schema = schema
        self.tokenizer = ResumeTokenizer()
        
        # Valid entity names for tag validation
        self._valid_entities = set(schema.entities.keys())
    
    def tag_document(
        self,
        doc: AnnotatedDocument,
        layer: int = 0
    ) -> Tuple[List[str], List[str]]:
        """
        🏷️ Convert an annotated document to BIO-tagged sequences.
        
        Args:
            doc:   The annotated document.
            layer: Which annotation layer to process (0=primary).
            
        Returns:
            Tuple of (token_texts, bio_tags) — parallel lists.
        """
        # Filter annotations to the requested layer
        layer_annotations = [
            a for a in doc.annotations if a.layer == layer
        ]
        
        return self.tag_text(doc.raw_text, layer_annotations, doc.tokens)
    
    def tag_text(
        self,
        text: str,
        annotations: List[SpanAnnotation],
        tokens: Optional[List[Token]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        🏷️ Convert raw text + span annotations into BIO-tagged token sequences.
        
        The CORE algorithm:
        1. Tokenize the text (or use pre-computed tokens)
        2. Sort annotations by position and length (longest first for conflicts)
        3. For each token, find which annotation(s) it belongs to
        4. Assign B- tag to the first token of each entity, I- to the rest
        5. Assign O to tokens outside any entity
        
        Conflict Resolution:
        - When a token falls in multiple annotations, the LONGEST span wins
        - When annotations share the same token boundary, the FIRST one wins
        - This matches standard NER training behavior
        
        Args:
            text:        Raw text to tag.
            annotations: List of span annotations.
            tokens:      Optional pre-computed tokens (avoids re-tokenizing).
            
        Returns:
            Tuple of (token_texts, bio_tags) — parallel lists of equal length.
        """
        if not text:
            return [], []
        
        # Step 1: Tokenize
        if tokens is None:
            tokens = self.tokenizer.tokenize(text)
        
        if not tokens:
            return [], []
        
        # Step 2: Sort annotations — longer spans first (for conflict resolution)
        sorted_anns = sorted(
            annotations,
            key=lambda a: (a.char_start, -(a.char_end - a.char_start))
        )
        
        # Step 3: Build token-to-annotation mapping
        # For each token index, track which annotation it belongs to
        # and whether it's the first token of that annotation (B) or not (I)
        token_tags = ["O"] * len(tokens)
        token_claimed = [False] * len(tokens)  # Prevent double-assignment
        
        for ann in sorted_anns:
            # Validate entity type
            if ann.entity_type not in self._valid_entities:
                logger.warning(f"⚠️ Unknown entity type: {ann.entity_type}")
                continue
            
            # Find tokens that overlap with this annotation span
            first_token_found = False
            
            for i, token in enumerate(tokens):
                # Skip already-claimed tokens (conflict resolution)
                if token_claimed[i]:
                    continue
                
                # Check if token overlaps with the annotation span
                # Token range: [token.char_start, token.char_end)
                # Annotation range: [ann.char_start, ann.char_end)
                if self._spans_overlap(
                    token.char_start, token.char_end,
                    ann.char_start, ann.char_end
                ):
                    if not first_token_found:
                        # First overlapping token → B tag
                        token_tags[i] = f"B-{ann.entity_type}"
                        first_token_found = True
                    else:
                        # Subsequent tokens → I tag
                        token_tags[i] = f"I-{ann.entity_type}"
                    
                    token_claimed[i] = True
        
        # Step 4: Extract parallel lists
        token_texts = [t.text for t in tokens]
        
        return token_texts, token_tags
    
    def _spans_overlap(
        self,
        start1: int, end1: int,
        start2: int, end2: int
    ) -> bool:
        """Check if two character spans overlap."""
        return start1 < end2 and start2 < end1
    
    def tags_to_spans(
        self,
        tokens: List[Token],
        tags: List[str]
    ) -> List[SpanAnnotation]:
        """
        🔄 Reverse operation: Convert BIO tags back to span annotations.
        
        Useful for converting model predictions back to human-readable
        annotations. Like translating pattern pieces back into the
        complete costume design! 🎭→👗
        
        Args:
            tokens: Token list (with character positions).
            tags:   Parallel BIO tag list.
            
        Returns:
            List of SpanAnnotation objects reconstructed from tags.
        """
        if not tokens or not tags or len(tokens) != len(tags):
            return []
        
        annotations = []
        current_entity = None
        current_start = None
        current_tokens = []
        
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith("B-"):
                # Close previous entity if any
                if current_entity is not None:
                    annotations.append(SpanAnnotation(
                        entity_type=current_entity,
                        char_start=current_start,
                        char_end=current_tokens[-1].char_end,
                        text=" ".join(t.text for t in current_tokens),
                        annotator="model"
                    ))
                
                # Start new entity
                current_entity = tag[2:]
                current_start = token.char_start
                current_tokens = [token]
                
            elif tag.startswith("I-"):
                entity_name = tag[2:]
                # Only continue if it matches the current entity
                # (handles malformed I- without preceding B-)
                if current_entity == entity_name:
                    current_tokens.append(token)
                else:
                    # Orphan I- tag — treat as B- (error recovery)
                    if current_entity is not None:
                        annotations.append(SpanAnnotation(
                            entity_type=current_entity,
                            char_start=current_start,
                            char_end=current_tokens[-1].char_end,
                            text=" ".join(t.text for t in current_tokens),
                            annotator="model"
                        ))
                    current_entity = entity_name
                    current_start = token.char_start
                    current_tokens = [token]
            else:
                # O tag — close any open entity
                if current_entity is not None:
                    annotations.append(SpanAnnotation(
                        entity_type=current_entity,
                        char_start=current_start,
                        char_end=current_tokens[-1].char_end,
                        text=" ".join(t.text for t in current_tokens),
                        annotator="model"
                    ))
                    current_entity = None
                    current_start = None
                    current_tokens = []
        
        # Close final entity if text ends mid-entity
        if current_entity is not None:
            annotations.append(SpanAnnotation(
                entity_type=current_entity,
                char_start=current_start,
                char_end=current_tokens[-1].char_end,
                text=" ".join(t.text for t in current_tokens),
                annotator="model"
            ))
        
        return annotations


# =============================================================================
# 🎭 STEP C: NESTED ENTITY HANDLER
# Manages overlapping annotations across multiple layers.
# =============================================================================

class NestedEntityLayer:
    """
    🎭 Multi-layer annotation system for nested entities.
    
    In resumes, entities OVERLAP constantly:
    - "Google, Mountain View" → ORGANIZATION + LOCATION
    - "Bachelor of Science in Computer Science from NUS" →
       DEGREE(contains FIELD_OF_STUDY) + INSTITUTION
    - "Led team of 15 engineers using Python and Docker" →
       JOB_DESCRIPTION(contains METRIC + SKILL + SKILL)
    
    Solution: Flatten into multiple annotation layers where:
    - Layer 0: Primary entities (the "outer" spans)
    - Layer 1: Nested entities (the "inner" spans)
    - Layer 2+: Deeper nesting (rare but supported)
    
    Each layer gets its own BIO tag sequence, and the model can
    be trained on each layer independently or jointly.
    
    Drama analogy: Like a costume with layers — the dress, the
    corset underneath, the petticoat below that. Each layer has
    its own pattern, but they all work together! 👗🎭
    """
    
    def __init__(self, schema: EntitySchema):
        self.schema = schema
        # Cache which entities can nest inside which
        self._nesting_rules = self._build_nesting_rules()
    
    def _build_nesting_rules(self) -> Dict[str, Set[str]]:
        """Build a map of parent → allowed children from the schema."""
        rules = {}
        for name, entity in self.schema.entities.items():
            if entity.allows_nesting and entity.nested_children:
                rules[name] = set(entity.nested_children)
        return rules
    
    def decompose_to_layers(
        self,
        annotations: List[SpanAnnotation]
    ) -> Dict[int, List[SpanAnnotation]]:
        """
        🎭 Decompose potentially overlapping annotations into flat layers.
        
        Algorithm:
        1. Sort annotations by span length (longest first)
        2. Try to place each annotation in the lowest available layer
        3. Two annotations conflict if they overlap but neither contains the other
        4. Parent-child relationships go to different layers automatically
        
        Args:
            annotations: List of span annotations (possibly overlapping).
            
        Returns:
            Dict mapping layer_number → list of non-overlapping annotations.
        """
        if not annotations:
            return {0: []}
        
        # Sort by length descending, then start position
        sorted_anns = sorted(
            annotations,
            key=lambda a: (-(a.char_end - a.char_start), a.char_start)
        )
        
        layers: Dict[int, List[SpanAnnotation]] = {}
        
        for ann in sorted_anns:
            placed = False
            
            # Try each layer starting from 0
            for layer_idx in range(10):  # Max 10 layers (safety limit)
                if layer_idx not in layers:
                    layers[layer_idx] = []
                
                # Check if this annotation conflicts with any in this layer
                conflicts = False
                for existing in layers[layer_idx]:
                    if self._annotations_conflict(ann, existing):
                        conflicts = True
                        break
                
                if not conflicts:
                    ann_copy = copy.copy(ann)
                    ann_copy.layer = layer_idx
                    layers[layer_idx].append(ann_copy)
                    placed = True
                    break
            
            if not placed:
                logger.warning(
                    f"⚠️ Could not place annotation: {ann.entity_type} "
                    f"[{ann.char_start}:{ann.char_end}] in any layer"
                )
        
        return layers
    
    def _annotations_conflict(
        self,
        a: SpanAnnotation,
        b: SpanAnnotation
    ) -> bool:
        """
        Check if two annotations CONFLICT for same-layer placement.
        
        For BIO tagging, ANY overlapping spans conflict — even if one
        fully contains the other (nesting). This is because a single
        BIO layer can only assign ONE tag per token.
        
        Two annotations conflict if they share ANY characters.
        They DON'T conflict only if they're completely non-overlapping.
        
        Nesting is handled by placing parent and child in DIFFERENT layers,
        each getting its own BIO tag sequence.
        """
        # Any overlap at all = conflict for same-layer BIO tagging
        if a.char_end <= b.char_start or b.char_end <= a.char_start:
            return False  # No overlap — safe for same layer
        
        return True  # Any overlap = must be in different layers
    
    def validate_nesting(
        self,
        annotations: List[SpanAnnotation]
    ) -> List[Dict[str, Any]]:
        """
        ✅ Validate that nested annotations follow schema rules.
        
        Checks:
        1. Child entities are actually inside parent spans
        2. Parent entity type allows the child type per schema
        3. No illegal cross-layer references
        
        Returns:
            List of validation warnings (empty = all valid).
        """
        warnings = []
        
        # Sort by span size (largest = parents, smallest = children)
        sorted_by_size = sorted(
            annotations,
            key=lambda a: -(a.char_end - a.char_start)
        )
        
        for i, potential_child in enumerate(sorted_by_size):
            for potential_parent in sorted_by_size[:i]:
                # Check if child is inside parent
                if (potential_parent.char_start <= potential_child.char_start and
                    potential_parent.char_end >= potential_child.char_end):
                    
                    # Check if this nesting is allowed by schema
                    parent_type = potential_parent.entity_type
                    child_type = potential_child.entity_type
                    
                    allowed_children = self._nesting_rules.get(parent_type, set())
                    
                    if child_type not in allowed_children and parent_type != child_type:
                        warnings.append({
                            "type": "illegal_nesting",
                            "parent": parent_type,
                            "child": child_type,
                            "parent_span": f"[{potential_parent.char_start}:{potential_parent.char_end}]",
                            "child_span": f"[{potential_child.char_start}:{potential_child.char_end}]",
                            "message": (
                                f"{child_type} nested inside {parent_type} is not "
                                f"allowed by schema. Allowed children: {allowed_children or 'none'}"
                            )
                        })
        
        return warnings


# =============================================================================
# 🧩 STEP D: EDGE CASE HANDLER
# Rules for all the messy, real-world resume patterns.
# =============================================================================

class EdgeCaseHandler:
    """
    🧩 Handles resume-specific edge cases for annotation.
    
    Resumes are the MESSIEST documents in all of NLP! This handler
    provides rules and normalizers for the common headaches:
    
    1. "Present" / "Current" as date values
    2. "Google, Mountain View" → separate ORG + LOC
    3. Bullet point text → full span as JOB_DESCRIPTION
    4. Partial / redacted data (S****567D)
    5. Singapore-specific formats (Blk, CMFAS, NRIC)
    6. Date format variations (DD/MM/YYYY vs MM/DD/YYYY)
    
    Drama analogy: The costume designer's emergency kit — for when
    a zipper breaks, a heel snaps, or a sequin falls off mid-show! 🧰✨
    """
    
    # Words that indicate "current employment" — should be tagged as WORK_DATE
    PRESENT_KEYWORDS = {
        'present', 'current', 'now', 'ongoing', 'today',
        'till date', 'to date', 'till now', 'up to now',
        'existing', 'presently'
    }
    
    # Bullet characters to strip from JOB_DESCRIPTION annotations
    BULLET_CHARS_STRIP = set('•▪▸▹►◆◇○●■□–—·⁃➤➢✓✔☐☑★*-')
    
    # Common organization suffixes (keep as part of ORG name)
    ORG_SUFFIXES = {
        'pte ltd', 'pte. ltd.', 'sdn bhd', 'sdn. bhd.',
        'inc', 'inc.', 'corp', 'corp.', 'llc', 'llp',
        'co.', 'company', 'group', 'holdings',
        'limited', 'ltd', 'ltd.'
    }
    
    # Location indicators that help separate ORG from LOC
    LOCATION_INDICATORS = {
        'singapore', 'malaysia', 'kuala lumpur', 'penang',
        'johor', 'selangor', 'japan', 'tokyo', 'hong kong',
        'mountain view', 'san francisco', 'new york', 'london',
        'remote', 'hybrid', 'onsite', 'on-site'
    }

    @classmethod
    def is_present_date(cls, text: str) -> bool:
        """
        Check if text represents "current" employment date.
        
        'Feb 2016 to Present' → the "Present" part should be
        INCLUDED in the WORK_DATE span, not excluded! 📅
        """
        return text.strip().lower() in cls.PRESENT_KEYWORDS
    
    @classmethod
    def split_org_location(
        cls, text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        🏢📍 Split "Organization, Location" into separate entities.
        
        This is the CLASSIC resume edge case:
        "Google, Mountain View" → ("Google", "Mountain View")
        "DBS Bank, Singapore"  → ("DBS Bank", "Singapore")
        "Grab Holdings Pte Ltd, Kuala Lumpur" → ("Grab Holdings Pte Ltd", "Kuala Lumpur")
        
        Algorithm:
        1. Split on the LAST comma
        2. Check if the part after comma matches known locations
        3. If yes → split into ORG + LOC
        4. If no → keep as single ORG
        
        Returns:
            Tuple of (org_text, loc_text) or (full_text, None).
        """
        if ',' not in text:
            return text.strip(), None
        
        # Try splitting on the last comma
        parts = text.rsplit(',', 1)
        if len(parts) != 2:
            return text.strip(), None
        
        potential_org = parts[0].strip()
        potential_loc = parts[1].strip()
        
        # Check if the part after comma looks like a location
        loc_lower = potential_loc.lower()
        
        if any(indicator in loc_lower for indicator in cls.LOCATION_INDICATORS):
            return potential_org, potential_loc
        
        # Check if it's a country name (simple heuristic: capitalized, 1-3 words)
        words = potential_loc.split()
        if len(words) <= 3 and all(w[0].isupper() for w in words if w):
            # Likely a location — but check it's not an ORG suffix
            if loc_lower not in cls.ORG_SUFFIXES:
                return potential_org, potential_loc
        
        return text.strip(), None
    
    @classmethod
    def normalize_bullet_text(cls, text: str) -> str:
        """
        🔫 Strip bullet characters from job description text.
        
        For annotation: the bullet character itself (•, -, *) should
        be EXCLUDED from the JOB_DESCRIPTION span. Label starts at
        the first word AFTER the bullet.
        
        "• Managed team of 10" → "Managed team of 10"
        """
        stripped = text.lstrip()
        
        # Remove leading bullet character
        if stripped and stripped[0] in cls.BULLET_CHARS_STRIP:
            stripped = stripped[1:].lstrip()
        
        return stripped
    
    @classmethod
    def classify_date_format(cls, date_str: str) -> Optional[str]:
        """
        📅 Classify a date string's format for consistent handling.
        
        Singapore/Malaysia commonly use DD/MM/YYYY, while the system
        internally uses YYYY-MM-DD. This identifies the format
        so we can convert correctly.
        
        Returns:
            Format string ('DD/MM/YYYY', 'YYYY-MM-DD', etc.) or None.
        """
        date_str = date_str.strip()
        
        patterns = [
            (r'^\d{1,2}/\d{1,2}/\d{4}$', 'DD/MM/YYYY'),
            (r'^\d{4}-\d{1,2}-\d{1,2}$', 'YYYY-MM-DD'),
            (r'^\d{1,2}-\d{1,2}-\d{4}$', 'DD-MM-YYYY'),
            (r'^\d{1,2}\s+\w+\s+\d{4}$', 'DD Mon YYYY'),
            (r'^\w+\s+\d{1,2},?\s+\d{4}$', 'Mon DD YYYY'),
            (r'^\w+\s+\d{4}$', 'Mon YYYY'),
            (r'^\d{4}$', 'YYYY'),
        ]
        
        for pattern, fmt in patterns:
            if re.match(pattern, date_str):
                return fmt
        
        return None
    
    @classmethod
    def is_date_range(cls, text: str) -> bool:
        """Check if text is a date range (contains a separator)."""
        separators = [' to ', ' – ', ' - ', ' till ', ' until ', ' — ']
        text_lower = text.lower()
        return any(sep in text_lower for sep in separators)


# =============================================================================
# 🤖 PRE-ANNOTATOR — Auto-label using dictionaries
# =============================================================================

class PreAnnotator:
    """
    🤖 Automatically pre-annotates resume text using dictionaries.
    
    Like UBIAI's auto-labeling feature — it scans the text for known
    patterns (skill names, city names, institution names) and creates
    preliminary annotations that a human reviewer can then correct.
    
    This is SPEED-BOOSTING, not replacement! Human review is still
    essential, but pre-annotation can save 60-70% of manual effort! 💅
    
    Dictionary sources:
    - Skills: From ai_extractor.py known skill lists
    - Institutions: SG/MY universities and polytechnics
    - Locations: Singapore/Malaysia city and district names
    - Certifications: CMFAS, AWS, PMP, etc.
    """
    
    def __init__(self, schema: EntitySchema):
        self.schema = schema
        self.edge_handler = EdgeCaseHandler()
        
        # Initialize dictionaries for each entity type
        self._skill_dict: Set[str] = set()
        self._institution_dict: Set[str] = set()
        self._location_dict: Set[str] = set()
        self._certification_dict: Set[str] = set()
        self._section_headers: Set[str] = set()
        
        self._load_default_dictionaries()
    
    def _load_default_dictionaries(self):
        """
        📚 Load built-in dictionaries for pre-annotation.
        
        These are curated for SG/MY resume formats. Additional terms
        can be loaded from files or the database.
        """
        # -- Skills (subset — expand from ai_extractor.py as needed) --
        self._skill_dict = {
            # Programming
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#',
            'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala',
            'html', 'css', 'sql', 'nosql', 'r', 'matlab', 'vba',
            # Frameworks
            'react', 'angular', 'vue', 'django', 'flask', 'spring',
            'node.js', 'express', 'fastapi', '.net', 'laravel',
            'next.js', 'nuxt.js', 'svelte', 'tailwind',
            # Data / AI
            'machine learning', 'deep learning', 'natural language processing',
            'computer vision', 'data science', 'data analysis',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'tableau', 'power bi', 'excel', 'spss', 'sas',
            # Cloud / DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions',
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'oracle', 'sql server', 'dynamodb', 'cassandra',
            # Other tech
            'git', 'linux', 'agile', 'scrum', 'jira', 'confluence',
            'figma', 'adobe photoshop', 'autocad', 'sap',
        }
        
        # -- Institutions (SG/MY focused) --
        self._institution_dict = {
            'national university of singapore', 'nus',
            'nanyang technological university', 'ntu',
            'singapore management university', 'smu',
            'singapore university of technology and design', 'sutd',
            'singapore institute of technology', 'sit',
            'singapore university of social sciences', 'suss',
            'singapore polytechnic', 'ngee ann polytechnic',
            'temasek polytechnic', 'republic polytechnic', 'nanyang polytechnic',
            'ite college east', 'ite college central', 'ite college west',
            'lasalle college of the arts', 'nafa',
            'raffles institution', 'hwa chong institution',
            # Malaysia
            'universiti malaya', 'um', 'universiti kebangsaan malaysia', 'ukm',
            'universiti putra malaysia', 'upm', 'universiti teknologi malaysia', 'utm',
            'universiti sains malaysia', 'usm', 'universiti teknologi mara', 'uitm',
            'monash university malaysia', 'taylor\'s university',
            'sunway university', 'help university', 'multimedia university',
        }
        
        # -- Locations (SG/MY regions) --
        self._location_dict = {
            'singapore', 'malaysia', 'kuala lumpur', 'penang', 'johor bahru',
            'selangor', 'petaling jaya', 'subang jaya', 'shah alam', 'ipoh',
            'malacca', 'melaka', 'kota kinabalu', 'kuching', 'putrajaya',
            # SG Areas
            'ang mo kio', 'bedok', 'bishan', 'bukit batok', 'bukit merah',
            'bukit timah', 'clementi', 'geylang', 'hougang', 'jurong east',
            'jurong west', 'kallang', 'marine parade', 'pasir ris', 'punggol',
            'queenstown', 'sembawang', 'sengkang', 'serangoon', 'tampines',
            'toa payoh', 'woodlands', 'yishun', 'orchard', 'tanjong pagar',
            'raffles place', 'marina bay', 'one-north', 'changi',
        }
        
        # -- Certifications --
        self._certification_dict = {
            'cmfas module 1', 'cmfas module 1a', 'cmfas module 5',
            'cmfas module 6', 'cmfas module 6a', 'cmfas module 8', 'cmfas module 9',
            'aws certified solutions architect', 'aws certified developer',
            'aws certified cloud practitioner', 'pmp', 'capm', 'prince2',
            'cissp', 'cism', 'ceh', 'comptia security+', 'comptia network+',
            'certified scrum master', 'csm', 'psm', 'safe agilist',
            'google cloud certified', 'azure fundamentals',
            'cfa level i', 'cfa level ii', 'cfa level iii',
            'acca', 'cpa', 'wsq advanced certificate',
        }
        
        # -- Section headers --
        self._section_headers = {
            'work experience', 'professional experience', 'employment history',
            'experience', 'career history', 'work history',
            'education', 'academic qualifications', 'qualifications',
            'skills', 'technical skills', 'core competencies', 'key skills',
            'professional summary', 'summary', 'objective', 'profile',
            'certifications', 'certificates', 'licenses',
            'languages', 'language proficiency',
            'projects', 'key projects', 'personal projects',
            'achievements', 'awards', 'honors', 'accomplishments',
            'references', 'referees',
            'hobbies', 'interests', 'personal interests',
            'volunteer', 'community service', 'extracurricular',
        }
    
    def pre_annotate(
        self,
        text: str,
        confidence_threshold: float = 0.8
    ) -> List[SpanAnnotation]:
        """
        🤖 Auto-annotate text using dictionary matching.
        
        Scans for known skills, institutions, locations, certifications,
        and section headers. Returns annotations with confidence scores
        so humans know which ones to review more carefully.
        
        Algorithm:
        1. Lowercase text for matching (but preserve original positions)
        2. Run regex-based entity detection (emails, phones, dates)
        3. Run dictionary matching for skills, institutions, etc.
        4. Resolve overlapping matches (longer match wins)
        5. Return annotations sorted by position
        
        Args:
            text:                 Raw resume text.
            confidence_threshold: Minimum confidence to include (0-1).
            
        Returns:
            List of SpanAnnotation objects (annotator="auto").
        """
        if not text:
            return []
        
        annotations = []
        text_lower = text.lower()
        
        # --- Regex-based detection (high confidence) ---
        annotations.extend(self._detect_emails(text))
        annotations.extend(self._detect_phones(text))
        annotations.extend(self._detect_dates(text))
        annotations.extend(self._detect_nric(text))
        
        # --- Dictionary matching ---
        annotations.extend(self._match_dictionary(
            text, text_lower, self._skill_dict, "SKILL", 0.75
        ))
        annotations.extend(self._match_dictionary(
            text, text_lower, self._institution_dict, "INSTITUTION", 0.85
        ))
        annotations.extend(self._match_dictionary(
            text, text_lower, self._location_dict, "LOCATION", 0.7
        ))
        annotations.extend(self._match_dictionary(
            text, text_lower, self._certification_dict, "CERTIFICATION", 0.9
        ))
        annotations.extend(self._match_dictionary(
            text, text_lower, self._section_headers, "SECTION_HEADER", 0.95
        ))
        
        # --- Resolve overlaps (longer match wins) ---
        annotations = self._resolve_overlaps(annotations)
        
        # --- Filter by confidence threshold ---
        annotations = [
            a for a in annotations if a.confidence >= confidence_threshold
        ]
        
        # Sort by position
        annotations.sort(key=lambda a: (a.char_start, a.char_end))
        
        return annotations
    
    def _detect_emails(self, text: str) -> List[SpanAnnotation]:
        """Detect email addresses using regex."""
        pattern = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
        return [
            SpanAnnotation(
                entity_type="EMAIL",
                char_start=m.start(),
                char_end=m.end(),
                text=m.group(),
                confidence=0.95,
                annotator="auto_regex"
            )
            for m in pattern.finditer(text)
        ]
    
    def _detect_phones(self, text: str) -> List[SpanAnnotation]:
        """Detect phone numbers (SG/MY formats)."""
        patterns = [
            r'\+65[\s\-]?\d{4}[\s\-]?\d{4}',           # SG: +65 XXXX XXXX
            r'\+60[\s\-]?\d{1,2}[\s\-]?\d{3,4}[\s\-]?\d{4}',  # MY: +60 XX XXXX XXXX
            r'\(\+?65\)\s?\d{4}\s?\d{4}',                # SG: (+65) XXXX XXXX
            r'\b[689]\d{3}[\s\-]?\d{4}\b',               # SG local: 9XXX XXXX
        ]
        
        annotations = []
        for pat in patterns:
            for m in re.finditer(pat, text):
                annotations.append(SpanAnnotation(
                    entity_type="PHONE",
                    char_start=m.start(),
                    char_end=m.end(),
                    text=m.group(),
                    confidence=0.9,
                    annotator="auto_regex"
                ))
        return annotations
    
    def _detect_dates(self, text: str) -> List[SpanAnnotation]:
        """Detect date patterns commonly found in resumes."""
        date_patterns = [
            # Date ranges: "Feb 2016 to Present", "Jan 2020 – Dec 2023"
            (r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*(?:to|–|-|—|till|until)\s*(?:Present|Current|Now|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})',
             "WORK_DATE", 0.9),
            # Year ranges: "2018 - 2022"
            (r'\b(?:19|20)\d{2}\s*(?:to|–|-|—)\s*(?:Present|Current|Now|(?:19|20)\d{2})\b',
             "WORK_DATE", 0.8),
            # Single dates: DD/MM/YYYY
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', "DATE_OF_BIRTH", 0.6),
        ]
        
        annotations = []
        for pat, etype, conf in date_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                annotations.append(SpanAnnotation(
                    entity_type=etype,
                    char_start=m.start(),
                    char_end=m.end(),
                    text=m.group(),
                    confidence=conf,
                    annotator="auto_regex"
                ))
        return annotations
    
    def _detect_nric(self, text: str) -> List[SpanAnnotation]:
        """Detect Singapore NRIC numbers."""
        pattern = re.compile(r'\b[STFGM]\d{7}[A-Z]\b')
        return [
            SpanAnnotation(
                entity_type="NRIC_ID",
                char_start=m.start(),
                char_end=m.end(),
                text=m.group(),
                confidence=0.95,
                annotator="auto_regex"
            )
            for m in pattern.finditer(text)
        ]
    
    def _match_dictionary(
        self,
        text: str,
        text_lower: str,
        dictionary: Set[str],
        entity_type: str,
        confidence: float
    ) -> List[SpanAnnotation]:
        """
        Match dictionary terms in text using case-insensitive search.
        
        Uses word-boundary matching to prevent partial matches
        (e.g., "java" shouldn't match inside "javascript").
        """
        annotations = []
        
        for term in dictionary:
            # Build regex with word boundaries
            # Escape special regex chars in the term
            escaped = re.escape(term)
            pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
            
            for m in pattern.finditer(text):
                annotations.append(SpanAnnotation(
                    entity_type=entity_type,
                    char_start=m.start(),
                    char_end=m.end(),
                    text=m.group(),
                    confidence=confidence,
                    annotator="auto_dict"
                ))
        
        return annotations
    
    def _resolve_overlaps(
        self,
        annotations: List[SpanAnnotation]
    ) -> List[SpanAnnotation]:
        """
        Resolve overlapping annotations — longer match wins.
        
        When 'machine learning' matches as SKILL and 'machine'
        matches something else, the longer 'machine learning' wins.
        """
        if not annotations:
            return []
        
        # Sort by span length descending
        sorted_anns = sorted(
            annotations,
            key=lambda a: -(a.char_end - a.char_start)
        )
        
        kept = []
        claimed_ranges = []
        
        for ann in sorted_anns:
            # Check if this annotation overlaps with any already-kept one
            overlaps = False
            for cs, ce in claimed_ranges:
                if not (ann.char_end <= cs or ann.char_start >= ce):
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(ann)
                claimed_ranges.append((ann.char_start, ann.char_end))
        
        return kept
    
    def add_terms(self, entity_type: str, terms: List[str]):
        """
        ➕ Add new terms to a dictionary for pre-annotation.
        
        Useful for expanding from the database or user corrections.
        """
        term_set = {t.lower() for t in terms}
        
        dict_map = {
            "SKILL": self._skill_dict,
            "SOFT_SKILL": self._skill_dict,
            "INSTITUTION": self._institution_dict,
            "LOCATION": self._location_dict,
            "CERTIFICATION": self._certification_dict,
            "SECTION_HEADER": self._section_headers,
        }
        
        target = dict_map.get(entity_type)
        if target is not None:
            target.update(term_set)
            logger.info(f"➕ Added {len(term_set)} terms to {entity_type} dictionary")


# =============================================================================
# 📤 TRAINING DATA EXPORTER
# Export annotations to standard NER training formats.
# =============================================================================

class TrainingExporter:
    """
    📤 Exports annotated data to standard NER training formats.
    
    Supports:
    - CoNLL format (industry standard for sequence labeling)
    - spaCy v3 training format (DocBin)
    - HuggingFace datasets format (JSON)
    - Custom JSON for our ML engine integration
    
    Drama analogy: The final step — packaging the costumes for
    shipment to the theater! Each theater (framework) needs a
    different shipping format! 📦🎭
    """
    
    def __init__(self, schema: EntitySchema):
        self.schema = schema
        self.tagger = BIOTagger(schema)
    
    def to_conll(
        self,
        documents: List[AnnotatedDocument],
        output_path: str,
        layer: int = 0
    ) -> str:
        """
        📤 Export to CoNLL-2003 format.
        
        CoNLL format is:  TOKEN<tab>TAG   (one per line, blank line between docs)
        
        This is the most widely supported NER training format,
        compatible with Flair, Hugging Face, and most NER frameworks.
        """
        lines = []
        
        for doc in documents:
            token_texts, bio_tags = self.tagger.tag_document(doc, layer=layer)
            
            # Add document separator comment
            lines.append(f"-DOCSTART- -X- -X- O")
            lines.append("")
            
            for text, tag in zip(token_texts, bio_tags):
                lines.append(f"{text}\t{tag}")
            
            lines.append("")  # Blank line between documents
        
        content = "\n".join(lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"📤 Exported {len(documents)} docs to CoNLL: {output_path}")
        
         # =================================================================
    # 🧠 CLASSIFICATION TRAINING EXPORT
    # Exports human-corrected Function/Industry labels for
    # training a text classification model to REPLACE the
    # keyword-based ResumeClassifier.
    #
    # Drama analogy: This is the diploma mill — it takes all the
    # lessons the human annotators taught and packages them into
    # a curriculum the ML model can study from! 🎓📦
    # =================================================================

    def to_classification_jsonl(
        self,
        db_path: str,
        output_path: str,
        min_text_length: int = 50,
        include_features: bool = True
    ) -> Dict[str, Any]:
        """
        📤 Export classification training data as JSONL.

        Reads human-corrected Function & Industry labels from the
        database and pairs them with resume text + extracted features.

        Each line is a JSON object:
        {
            "text":            "Full resume text...",
            "function_label":  "IT",
            "industry_label":  "Banking & Finance",
            "features": {              # Optional enrichment
                "job_titles":   ["Software Engineer", "Tech Lead"],
                "companies":    ["DBS Bank", "Grab"],
                "skills":       ["Python", "Java", "AWS"],
                "institutions": ["NUS"],
                "degrees":      ["Bachelor of Computing"]
            }
        }

        Why JSONL instead of plain JSON?
        ─────────────────────────────────
        JSONL (one JSON object per line) is the standard for ML training
        because it's streamable — you can load one record at a time
        without parsing the entire file into memory. HuggingFace
        datasets, PyTorch DataLoader, and pandas all read JSONL natively.
        Think of it as a buffet line vs a sit-down dinner — you grab
        one plate at a time! 🍽️

        Args:
            db_path:          Path to resume_extractions.db
            output_path:      Where to write the .jsonl file
            min_text_length:  Skip resumes shorter than this (noise filter)
            include_features: Whether to include extracted entity features

        Returns:
            Dict with export stats (total, skipped, label distribution)
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # ── Query: Join ner_documents (has human labels) with
        #    structured_extractions (has text + entity data)
        #    LEFT JOIN candidates to get raw_text if available ──────
        query = """
            SELECT
                nd.doc_id,
                nd.candidate_id,
                nd.function,
                nd.industry,
                se.name,
                se.experience_raw,
                se.education_raw,
                se.skills_raw,
                se.skills_json,
                se.experience_json,
                se.education_json,
                se.summary,
                c.raw_text
            FROM ner_documents nd
            JOIN structured_extractions se
                ON nd.candidate_id = se.candidate_id
            LEFT JOIN candidates c
                ON nd.candidate_id = c.id
            WHERE nd.function IS NOT NULL
              AND nd.function != ''
              AND nd.industry IS NOT NULL
              AND nd.industry != ''
        """

        try:
            rows = conn.execute(query).fetchall()
        except sqlite3.OperationalError as e:
            # ── Handle missing columns gracefully ─────────────────
            # If the candidates table doesn't have raw_text, or
            # ner_documents doesn't have function/industry yet,
            # degrade gracefully instead of crashing the export.
            logger.warning(f"⚠️ Classification export query failed: {e}")
            logger.warning("Trying fallback query without raw_text...")
            fallback_query = """
                SELECT
                    nd.doc_id,
                    nd.candidate_id,
                    nd.function,
                    nd.industry,
                    se.name,
                    se.experience_raw,
                    se.education_raw,
                    se.skills_raw,
                    se.skills_json,
                    se.experience_json,
                    se.education_json,
                    se.summary,
                    '' as raw_text
                FROM ner_documents nd
                JOIN structured_extractions se
                    ON nd.candidate_id = se.candidate_id
                WHERE nd.function IS NOT NULL
                  AND nd.function != ''
                  AND nd.industry IS NOT NULL
                  AND nd.industry != ''
            """
            rows = conn.execute(fallback_query).fetchall()
        finally:
            conn.close()

        # ── Build training records ────────────────────────────────
        records = []
        skipped = 0
        func_dist = defaultdict(int)   # Track label distribution
        ind_dist = defaultdict(int)

        for row in rows:
            # Build the full text from available sources
            # Priority: raw_text (complete) > reconstructed from fields
            text_parts = []
            if row["raw_text"]:
                text_parts.append(row["raw_text"])
            else:
                # Fallback: reconstruct from structured fields
                # This still gives the classifier enough signal!
                for field in ["summary", "experience_raw", "education_raw", "skills_raw"]:
                    if row[field]:
                        text_parts.append(str(row[field]))

            full_text = "\n".join(text_parts).strip()

            # ── Skip if text is too short ─────────────────────────
            # Resumes under 50 chars are usually parsing failures,
            # empty records, or test entries. Don't poison the
            # training data with garbage! 🗑️
            if len(full_text) < min_text_length:
                skipped += 1
                continue

            func_label = row["function"].strip()
            ind_label = row["industry"].strip()

            # ── Skip "others"/"Others" if desired ─────────────────
            # These are low-signal labels that can hurt classifier
            # performance. For now, we include them but log a warning.
            if func_label.lower() == "others" and ind_label.lower() == "others":
                logger.debug(
                    f"⚠️ Both function and industry are 'others' for "
                    f"candidate {row['candidate_id']} — including anyway"
                )

            record = {
                "text": full_text,
                "function_label": func_label,
                "industry_label": ind_label,
            }

            # ── Optional: add entity-based features ───────────────
            # These features let the classifier use STRUCTURED info
            # alongside raw text — like giving it reading glasses! 🤓
            if include_features:
                features = {}

                # Parse JSON fields safely
                def safe_json(raw, default):
                    if not raw:
                        return default
                    try:
                        return json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        return default

                # Extract job titles & companies from experience_json
                exp = safe_json(row["experience_json"], [])
                if isinstance(exp, list):
                    features["job_titles"] = [
                        e.get("title", "") for e in exp
                        if isinstance(e, dict) and e.get("title")
                    ]
                    features["companies"] = [
                        e.get("company", "") for e in exp
                        if isinstance(e, dict) and e.get("company")
                    ]
                else:
                    features["job_titles"] = []
                    features["companies"] = []

                # Extract skills
                skills = safe_json(row["skills_json"], [])
                features["skills"] = skills if isinstance(skills, list) else []

                # Extract education
                edu = safe_json(row["education_json"], [])
                if isinstance(edu, list):
                    features["institutions"] = [
                        e.get("institution", "") for e in edu
                        if isinstance(e, dict) and e.get("institution")
                    ]
                    features["degrees"] = [
                        e.get("degree", "") for e in edu
                        if isinstance(e, dict) and e.get("degree")
                    ]
                else:
                    features["institutions"] = []
                    features["degrees"] = []

                record["features"] = features

            records.append(record)
            func_dist[func_label] += 1
            ind_dist[ind_label] += 1

        # ── Write JSONL ──────────────────────────────────────────
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        stats = {
            "total_exported": len(records),
            "skipped": skipped,
            "function_distribution": dict(func_dist),
            "industry_distribution": dict(ind_dist),
        }

        logger.info(
            f"📤 Classification export: {len(records)} records → {output_path} | "
            f"Skipped: {skipped} | Functions: {len(func_dist)} | Industries: {len(ind_dist)}"
        )

        return stats
        
        return output_path
    
    def to_spacy_json(
        self,
        documents: List[AnnotatedDocument],
        output_path: str
    ) -> str:
        """
        📤 Export to spaCy v3 JSON training format.
        
        Format:
        [
            {"text": "...", "entities": [[start, end, "LABEL"], ...]}
        ]
        """
        training_data = []
        
        for doc in documents:
            entities = []
            for ann in doc.annotations:
                if ann.layer == 0:  # Primary layer only for spaCy
                    entities.append([ann.char_start, ann.char_end, ann.entity_type])
            
            training_data.append({
                "text": doc.raw_text,
                "entities": entities
            })
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📤 Exported {len(documents)} docs to spaCy JSON: {output_path}")
        return output_path
    
    def to_huggingface(
        self,
        documents: List[AnnotatedDocument],
        output_path: str,
        layer: int = 0
    ) -> str:
        """
        📤 Export to Hugging Face datasets format.
        
        Format per doc:
        {
            "id": "...",
            "tokens": ["word1", "word2", ...],
            "ner_tags": [0, 3, 4, ...]   ← numeric tag IDs
        }
        """
        tag_to_id = self.schema.get_tag_to_id()
        records = []
        
        for doc in documents:
            token_texts, bio_tags = self.tagger.tag_document(doc, layer=layer)
            
            tag_ids = [tag_to_id.get(tag, 0) for tag in bio_tags]
            
            records.append({
                "id": doc.doc_id,
                "tokens": token_texts,
                "ner_tags": tag_ids
            })
        
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"📤 Exported {len(documents)} docs to HF format: {output_path}")
        return output_path
    
    # =================================================================
    # 📋 CANDIDATE PROFILE EXPORT — The Talent Database Format
    # =================================================================

    def _group_work_experience(
        self,
        annotations: List["SpanAnnotation"],
        gap_threshold: int = 400
    ) -> List[Dict]:
        """
        🧩 Group work-related annotations into individual job blocks.

        IMPROVED VERSION — Uses STRUCTURAL BOUNDARY detection! 🎯

        The old version used only character-gap distance, which fails
        when resumes have dense formatting (Job 1's last bullet is
        only 50 chars away from Job 2's header).

        New strategy:
          1. Collect all work-related annotations sorted by char_start.
          2. Walk through them in text order.
          3. Start a NEW block when EITHER:
             a) A JOB_TITLE or ORGANIZATION appears AND the current
                block already has a JOB_TITLE or ORGANIZATION
                (= structural boundary — new job header detected!)
             b) The character gap exceeds gap_threshold AND we've
                seen at least one header entity (= physical gap)
          4. Within each block, pick first JOB_TITLE, ORGANIZATION,
             WORK_DATE, WORK_LOCATION and collect ALL JOB_DESCRIPTIONs.

        Think of it like reading a play script 🎭 — when you see a
        new character name in bold (JOB_TITLE), that's a new SCENE,
        even if the previous scene's dialogue (JOB_DESCRIPTION) was
        on the very line above! The character name IS the boundary!

        Args:
            annotations:    All annotations from the document (all types).
            gap_threshold:  Character gap that signals a new job block
                            (used as SECONDARY signal after structural
                            boundaries). 400 chars ≈ ~3–5 resume lines.

        Returns:
            List of job dicts matching the target export schema.
        """
        # ── Collect only work-relevant entity types ──────────────────────
        WORK_TYPES = {"JOB_TITLE", "ORGANIZATION", "WORK_DATE", "JOB_DESCRIPTION", "WORK_LOCATION"}
        HEADER_TYPES = {"JOB_TITLE", "ORGANIZATION", "WORK_DATE"}
        
        work_anns = sorted(
            [a for a in annotations if a.entity_type in WORK_TYPES and a.layer == 0],
            key=lambda a: a.char_start
        )

        if not work_anns:
            return []

        # ── Split into blocks using STRUCTURAL + GAP detection ───────────
        # Key insight: A new JOB_TITLE or ORGANIZATION after we've already
        # seen one = DEFINITE new job block boundary! 🎯
        blocks: List[List] = []
        current_block: List = [work_anns[0]]

        # Track what entity types the current block already contains
        current_block_types = {work_anns[0].entity_type}

        for ann in work_anns[1:]:
            prev_end = current_block[-1].char_end
            gap = ann.char_start - prev_end

            # ── STRUCTURAL BOUNDARY DETECTION ─────────────────────
            # If we see a new JOB_TITLE or ORGANIZATION, and the
            # current block already HAS a JOB_TITLE or ORGANIZATION,
            # this MUST be a new job — regardless of gap distance!
            #
            # But we're smart about it: if a JOB_TITLE and ORGANIZATION
            # appear right next to each other at the START of a block,
            # they're part of the SAME header. We only split when
            # we've already seen descriptions in this block.
            is_header = ann.entity_type in HEADER_TYPES
            block_has_header = bool(current_block_types & HEADER_TYPES)
            block_has_descriptions = "JOB_DESCRIPTION" in current_block_types

            # Condition 1: New header entity AFTER descriptions seen
            # = definite structural boundary
            structural_boundary = (
                is_header
                and block_has_header
                and block_has_descriptions
            )

            # Condition 2: New JOB_TITLE or ORG when current block
            # already has BOTH title and org (even without descriptions)
            # = two headers back-to-back = new job with no bullets yet
            double_header = (
                ann.entity_type in {"JOB_TITLE", "ORGANIZATION"}
                and "JOB_TITLE" in current_block_types
                and "ORGANIZATION" in current_block_types
                and ann.entity_type in current_block_types
            )

            # Condition 3: Character gap exceeds threshold
            # (fallback for unusual formatting)
            gap_boundary = gap > gap_threshold

            if structural_boundary or double_header or gap_boundary:
                # Start new block! 🆕
                blocks.append(current_block)
                current_block = [ann]
                current_block_types = {ann.entity_type}
            else:
                current_block.append(ann)
                current_block_types.add(ann.entity_type)

        blocks.append(current_block)  # Don't forget the last block!

        # ── Convert each block into the target dict format ───────────────
        jobs = []
        for block in blocks:
            job: Dict[str, Any] = {
                "company":        "",
                "title":          "",
                "location":       "",
                "from":           "",
                "to":             "",
                "responsibility": []
            }

            for ann in block:
                if ann.entity_type == "ORGANIZATION" and not job["company"]:
                    job["company"] = ann.text.strip()

                elif ann.entity_type == "JOB_TITLE" and not job["title"]:
                    job["title"] = ann.text.strip()

                elif ann.entity_type == "WORK_LOCATION" and not job["location"]:
                    job["location"] = ann.text.strip()

                elif ann.entity_type == "WORK_DATE" and not (job["from"] or job["to"]):
                    # Parse "Jan 2020 – Present" into from/to if possible
                    date_text = ann.text.strip()
                    # Common separators: –, -, to, till, until
                    sep_match = re.search(
                        r'\s*(?:–|—|-|to|till|until)\s*', date_text, re.IGNORECASE
                    )
                    if sep_match:
                        job["from"] = date_text[:sep_match.start()].strip()
                        job["to"]   = date_text[sep_match.end():].strip()
                    else:
                        # Single date (e.g. "Since 2018") → put in "from"
                        job["from"] = date_text

                elif ann.entity_type == "JOB_DESCRIPTION":
                    # Each description bullet becomes one list entry
                    resp = ann.text.strip()
                    if resp:
                        job["responsibility"].append(resp)

            # Only include blocks that have at least a title or company
            # (skip orphaned description-only fragments)
            if job["company"] or job["title"]:
                jobs.append(job)

        return jobs

    def _group_education(
        self,
        annotations: List["SpanAnnotation"],
        gap_threshold: int = 500
    ) -> List[Dict]:
        """
        🎓 Group education-related annotations into individual education blocks.

        Same proximity-grouping logic as _group_work_experience, applied to
        the education section. A diploma from one school should never end up
        mixed with a degree from another! 🏫

        Args:
            annotations:    All annotations from the document.
            gap_threshold:  Character gap that signals a new education block.

        Returns:
            List of education dicts matching the target export schema.
        """
        EDU_TYPES = {"INSTITUTION", "DEGREE", "FIELD_OF_STUDY", "EDU_DATE", "GPA"}
        edu_anns = sorted(
            [a for a in annotations if a.entity_type in EDU_TYPES and a.layer == 0],
            key=lambda a: a.char_start
        )

        if not edu_anns:
            return []

        # ── Split into blocks by gap ─────────────────────────────────────
        blocks: List[List] = []
        current_block: List = [edu_anns[0]]

        for ann in edu_anns[1:]:
            gap = ann.char_start - current_block[-1].char_end
            if gap > gap_threshold:
                blocks.append(current_block)
                current_block = [ann]
            else:
                current_block.append(ann)

        blocks.append(current_block)

        # ── Convert each block ───────────────────────────────────────────
        education = []
        for block in blocks:
            edu: Dict[str, Any] = {
                "school": "",
                "major":  "",
                "Degree": ""
            }

            for ann in block:
                if ann.entity_type == "INSTITUTION" and not edu["school"]:
                    edu["school"] = ann.text.strip()
                elif ann.entity_type == "DEGREE" and not edu["Degree"]:
                    edu["Degree"] = ann.text.strip()
                elif ann.entity_type == "FIELD_OF_STUDY" and not edu["major"]:
                    edu["major"] = ann.text.strip()
                # EDU_DATE and GPA are noted but not in the target schema —
                # they are captured in annotations but not surfaced here.

            if edu["school"] or edu["Degree"]:
                education.append(edu)

        return education

    def _group_projects(
        self,
        annotations: List["SpanAnnotation"],
        gap_threshold: int = 600
    ) -> List[Dict]:
        """
        🧩 Group project-related annotations into individual project blocks.

        Same proximity-grouping logic as _group_work_experience, but for
        the Projects / Project Experience section.

        Each project block becomes one entry in the "Project Experience" field
        of the candidate profile, formatted as:
            {
                "title":       "E-Commerce Platform Redesign",
                "description": ["Built a REST API...", "Achieved 98% test coverage..."]
            }

        Args:
            annotations:    All annotations from the document.
            gap_threshold:  Character gap that signals a new project block.
                            600 chars is larger than work experience (400) because
                            project sections often have longer descriptions.

        Returns:
            List of project dicts.
        """
        PROJECT_TYPES = {"PROJECT_TITLE", "PROJECT_DESCRIPTION"}
        proj_anns = sorted(
            [a for a in annotations if a.entity_type in PROJECT_TYPES and a.layer == 0],
            key=lambda a: a.char_start
        )

        if not proj_anns:
            return []

        # ── Split into blocks by character gap ──────────────────────────
        blocks: List[List] = []
        current_block: List = [proj_anns[0]]

        for ann in proj_anns[1:]:
            gap = ann.char_start - current_block[-1].char_end
            if gap > gap_threshold:
                blocks.append(current_block)
                current_block = [ann]
            else:
                current_block.append(ann)

        blocks.append(current_block)

        # ── Convert each block to the output dict ────────────────────────
        projects = []
        for block in blocks:
            proj: Dict[str, Any] = {
                "title":       "",
                "description": []
            }
            for ann in block:
                if ann.entity_type == "PROJECT_TITLE" and not proj["title"]:
                    proj["title"] = ann.text.strip()
                elif ann.entity_type == "PROJECT_DESCRIPTION":
                    desc = ann.text.strip()
                    if desc:
                        proj["description"].append(desc)

            # Only keep blocks that have at least a title or description
            if proj["title"] or proj["description"]:
                projects.append(proj)

        return projects

    def build_candidate_profile(
        self,
        doc: "AnnotatedDocument",
        structured_row: Optional[Dict] = None,
        created_by: str = "",
        creation_date: str = ""
    ) -> Dict:
        """
        🏗️ Assemble a single candidate profile dict from an AnnotatedDocument.

        Data priority
        ─────────────
        structured_extractions is the PRIMARY source for every field.
        It holds the latest human-reviewed data written back on every save.
        ner_annotations is the FALLBACK — used only when a field has no value
        in structured_extractions (e.g. the candidate was annotated before a
        column was added, or the save was interrupted).

        This means:
          structured_row["name"]          → Name   (falls back to PERSON_NAME ann)
          structured_row["email"]         → Email  (falls back to EMAIL ann)
          structured_row["phone"]         → Phone  (falls back to PHONE ann)
          structured_row["location"]      → Current Location
          structured_row["summary"]       → Summary
          structured_row["languages"]     → Language Skills
          structured_row["function"]      → Function
          structured_row["industry"]      → Industry
          structured_row["skills_json"]   → tags   (parsed JSON array)
          structured_row["experience_json"]→ Work Experience (parsed JSON)
          structured_row["education_json"]→ Education (parsed JSON)
          structured_row["projects"]      → Project Experience (raw text)

        Fields NOT in structured_extractions (always from annotations):
          Gender, Expected Location, Created By, Creation Date, Team

        Args:
            doc:            The annotated document (used for fallback annotation
                            values and for fields not stored in structured_extractions).
            structured_row: Row dict from structured_extractions (the primary source).
            created_by:     The annotator username.
            creation_date:  ISO timestamp of when annotation was saved.

        Returns:
            Dict matching the target candidate profile JSON schema.
        """
        anns = doc.annotations  # Used as fallback only
        fb   = structured_row or {}

        # ── Annotation helpers (fallback only) ────────────────────────────
        def first(entity_type: str) -> str:
            """First annotation text for this entity type, or ''."""
            for a in anns:
                if a.entity_type == entity_type and a.layer == 0:
                    return a.text.strip()
            return ""

        def all_of(entity_type: str) -> List[str]:
            """All unique annotation texts for this entity type."""
            seen: Set[str] = set()
            result = []
            for a in anns:
                if a.entity_type == entity_type and a.layer == 0:
                    val = a.text.strip()
                    if val and val not in seen:
                        seen.add(val)
                        result.append(val)
            return result

        # ── Helper: safely parse a JSON column, return default on failure ─
        def parse_json(raw: Optional[str], default):
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        # ══════════════════════════════════════════════════════════════════
        # SCALAR FIELDS — structured_extractions wins, annotations fallback
        # ══════════════════════════════════════════════════════════════════

        name     = fb.get("name",     "") or first("PERSON_NAME")
        email    = fb.get("email",    "") or first("EMAIL")
        phone    = fb.get("phone",    "") or first("PHONE")
        location = fb.get("location", "") or first("LOCATION")
        summary  = fb.get("summary",  "") or first("SUMMARY_TEXT")

        # Function / Industry: structured_extractions row → doc metadata → annotations
        stored_function = doc.metadata.get("function", "")
        stored_industry = doc.metadata.get("industry", "")
        function_value  = (fb.get("function", "")
                           or stored_function
                           or first("FUNCTION"))
        industry_value  = (fb.get("industry", "")
                           or stored_industry
                           or ", ".join(all_of("INDUSTRY")))

        # Language Skills: structured column already formatted "English (Native), ..."
        # Fall back to building it from annotation spans if the column is empty.
        language_skills = fb.get("languages", "")
        if not language_skills:
            lang_parts  = all_of("LANGUAGE_SKILL")
            proficiency = all_of("PROFICIENCY")
            if lang_parts and proficiency and len(lang_parts) == len(proficiency):
                language_skills = ", ".join(
                    f"{lang} ({prof})"
                    for lang, prof in zip(lang_parts, proficiency)
                )
            else:
                language_skills = ", ".join(lang_parts)

        # ══════════════════════════════════════════════════════════════════
        # TAGS (skills) — parse skills_json column; fallback to annotations
        # ══════════════════════════════════════════════════════════════════

        raw_skills = parse_json(fb.get("skills_json"), None)
        if isinstance(raw_skills, list) and raw_skills:
            # Deduplicate while preserving order
            tags_seen: Set[str] = set()
            tags: List[str] = []
            for val in raw_skills:
                val = str(val).strip()
                if val and val.lower() not in tags_seen:
                    tags_seen.add(val.lower())
                    tags.append(val)
        else:
            # Fallback: build from annotation spans
            tag_types = ["SKILL", "SOFT_SKILL", "CERTIFICATION"]
            tags_seen = set()
            tags = []
            for etype in tag_types:
                for a in anns:
                    if a.entity_type == etype and a.layer == 0:
                        val = a.text.strip()
                        if val and val.lower() not in tags_seen:
                            tags_seen.add(val.lower())
                            tags.append(val)

        # ══════════════════════════════════════════════════════════════════
        # WORK EXPERIENCE — parse experience_json; fallback to annotation groups
        #
        # structured_extractions stores experience as:
        #   {"positions": [{"title","organization","date","location"}, ...],
        #    "descriptions": [...], "metrics": [...]}
        #
        # build_candidate_profile expects:
        #   [{"company","title","dates","location","responsibilities"}, ...]
        # ══════════════════════════════════════════════════════════════════

        exp_data = parse_json(fb.get("experience_json"), None)
        if isinstance(exp_data, dict):
            # NEW: Use positions_with_responsibilities if available
            # (saved by the fixed update_structured_extraction)
            pwr = exp_data.get("positions_with_responsibilities")
            if pwr and isinstance(pwr, list):
                # 🎯 Best path: responsibilities are already grouped per job!
                work_experience = []
                for pos in pwr:
                    if not isinstance(pos, dict):
                        continue
                    work_experience.append({
                        "company":          pos.get("organization", ""),
                        "title":            pos.get("title",        ""),
                        "dates":            pos.get("date",         ""),
                        "location":         pos.get("location",     ""),
                        "responsibilities": pos.get("responsibilities", []),
                    })
            elif exp_data.get("positions"):
                # FALLBACK: Old format — positions + flat descriptions
                # Use proximity matching as best-effort
                positions     = exp_data["positions"]
                descriptions  = exp_data.get("descriptions", [])
                work_experience = []
                for i, pos in enumerate(positions):
                    if not isinstance(pos, dict):
                        continue
                    work_experience.append({
                        "company":          pos.get("organization", ""),
                        "title":            pos.get("title",        ""),
                        "dates":            pos.get("date",         ""),
                        "location":         pos.get("location",     ""),
                        "responsibilities": [descriptions[i]] if i < len(descriptions) else [],
                    })
            else:
                work_experience = self._group_work_experience(anns)
        else:
            # Fallback: derive from ner_annotations grouping
            work_experience = self._group_work_experience(anns)
        # ══════════════════════════════════════════════════════════════════
        # EDUCATION — parse education_json; fallback to annotation groups
        #
        # structured_extractions stores education as:
        #   [{"degree","institution","field_of_study","date","gpa"}, ...]
        #
        # build_candidate_profile expects:
        #   [{"institution","degree","field_of_study","dates","gpa"}, ...]
        # ══════════════════════════════════════════════════════════════════

        edu_data = parse_json(fb.get("education_json"), None)
        if isinstance(edu_data, list) and edu_data:
            education = []
            for entry in edu_data:
                if not isinstance(entry, dict):
                    continue
                education.append({
                    "institution":   entry.get("institution",   ""),
                    "degree":        entry.get("degree",        ""),
                    "field_of_study":entry.get("field_of_study",""),
                    "dates":         entry.get("date",          ""),
                    "gpa":           entry.get("gpa",           ""),
                })
        else:
            education = self._group_education(anns)

        # ══════════════════════════════════════════════════════════════════
        # PROJECT EXPERIENCE — structured_extractions stores as raw text
        # ("Title: Desc || Title2: Desc2").  Parse it back into structured
        # dicts for the profile; fall back to annotation grouping.
        # ══════════════════════════════════════════════════════════════════

        raw_projects = fb.get("projects", "")
        if raw_projects:
            project_experience = []
            for block in raw_projects.split(" || "):
                block = block.strip()
                if not block:
                    continue
                if ": " in block:
                    title, _, desc = block.partition(": ")
                    project_experience.append({
                        "title":       title.strip(),
                        "description": [desc.strip()] if desc.strip() else [],
                    })
                else:
                    project_experience.append({
                        "title":       block,
                        "description": [],
                    })
        else:
            project_experience = self._group_projects(anns)

        # ══════════════════════════════════════════════════════════════════
        # ANNOTATION-ONLY FIELDS (no structured_extractions column for these)
        # ══════════════════════════════════════════════════════════════════

        # Gender is not stored in structured_extractions — read from annotations
        gender = first("GENDER")

        # Expected Location is not stored — read from annotations
        expected_loc_value = first("EXPECTED_LOCATION")

        # Current Company / Title — first entry from work_experience list
        current_company = work_experience[0]["company"] if work_experience else ""
        current_title   = work_experience[0]["title"]   if work_experience else ""

        # ── Assemble the final profile ─────────────────────────────────────
        profile = {
            "ID":               str(doc.candidate_id),
            "Name":             name,
            "Phone":            phone,
            "Email":            email,
            "Current Company":  current_company,
            "Current Title":    current_title,
            "Team":             "",   # Recruiter-assigned — not on resume
            "Current Location": location,
            "Expected Location":expected_loc_value,
            "Gender":           gender,
            "Created By":       created_by or doc.annotator or "",
            "Creation Date":    creation_date or "",
            "Last Contact":     "",   # Recruiter-tracked — not on resume
            "Function":         function_value,
            "Industry":         industry_value,
            "Summary":          summary,
            "Language Skills":  language_skills,
            "Work Experience":  work_experience,
            "Education":        education,
            "Project Experience": project_experience,
            "tags":             tags,
        }

        return profile

    def to_candidate_json(
        self,
        documents: List["AnnotatedDocument"],
        output_path: str,
        structured_rows: Optional[Dict[int, Dict]] = None
    ) -> str:
        """
        📤 Export candidate profiles to JSON format.

        Produces a clean, recruiter-friendly JSON file — one profile per
        candidate, in the talent database schema. Like a beautifully
        formatted casting portfolio! 🎬

        Args:
            documents:       List of annotated documents to export.
            output_path:     Where to write the .json file.
            structured_rows: Optional dict of {candidate_id: row_dict} for
                             fallback values from structured_extractions.

        Returns:
            The output file path.
        """
        structured_rows = structured_rows or {}
        profiles = []

        for doc in documents:
            s_row = structured_rows.get(doc.candidate_id)

            # Use the doc's created_at from metadata if stored, else blank
            creation_date = doc.metadata.get("created_at", "")

            profile = self.build_candidate_profile(
                doc,
                structured_row=s_row,
                created_by=doc.annotator,
                creation_date=creation_date
            )
            profiles.append(profile)

        # Write as a JSON array
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)

        logger.info(
            f"📤 Exported {len(profiles)} candidate profiles to JSON: {output_path}"
        )
        return output_path

    def to_candidate_csv(
        self,
        documents: List["AnnotatedDocument"],
        output_path: str,
        structured_rows: Optional[Dict[int, Dict]] = None
    ) -> str:
        """
        📤 Export candidate profiles to CSV format.

        Flat fields are written as plain text columns. Nested structures
        (Work Experience, Education, tags) are JSON-serialized into their
        cell — this way the CSV stays importable while preserving all data.

        Think of it as packing a ballgown into a suitcase — you have to
        fold it carefully, but it all fits! 🧳👗

        Column order matches the target talent database schema exactly.

        Args:
            documents:       List of annotated documents to export.
            output_path:     Where to write the .csv file.
            structured_rows: Optional fallback data from structured_extractions.

        Returns:
            The output file path.
        """
        import csv

        structured_rows = structured_rows or {}

        # ── CSV column order — matches the target JSON schema exactly ─────
        FLAT_COLUMNS = [
            "ID", "Name", "Phone", "Email",
            "Current Company", "Current Title",
            "Team", "Current Location", "Expected Location",
            "Gender", "Created By", "Creation Date", "Last Contact",
            "Function", "Industry", "Summary", "Language Skills",
        ]
        # Nested columns are JSON-serialized in the CSV cell
        NESTED_COLUMNS = ["Work Experience", "Education", "Project Experience", "tags"]
        ALL_COLUMNS = FLAT_COLUMNS + NESTED_COLUMNS

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            # utf-8-sig BOM ensures Excel opens the file correctly (especially
            # for Malay/Chinese name characters — no mojibake drama! 🎭)
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()

            for doc in documents:
                s_row = structured_rows.get(doc.candidate_id)
                creation_date = doc.metadata.get("created_at", "")
                profile = self.build_candidate_profile(
                    doc,
                    structured_row=s_row,
                    created_by=doc.annotator,
                    creation_date=creation_date
                )

                row: Dict[str, Any] = {}
                for col in FLAT_COLUMNS:
                    row[col] = profile.get(col, "")

                # Serialize nested structures as compact JSON strings
                for col in NESTED_COLUMNS:
                    val = profile.get(col, "")
                    if isinstance(val, (list, dict)):
                        row[col] = json.dumps(val, ensure_ascii=False)
                    else:
                        row[col] = val or ""

                writer.writerow(row)

        logger.info(
            f"📤 Exported {len(documents)} candidate profiles to CSV: {output_path}"
        )
        return output_path

    def to_custom_json(
        self,
        documents: List[AnnotatedDocument],
        output_path: str
    ) -> str:
        """
        📤 Export to our custom JSON format with full metadata.
        
        Includes all layers, token positions, metadata — everything
        needed for our ML engine and annotation tool.
        """
        export_data = {
            "schema_version": "1.0",
            "exported_at": datetime.datetime.now().isoformat(),
            "total_documents": len(documents),
            "schema": self.schema.export_schema(),
            "documents": []
        }
        
        for doc in documents:
            doc_data = {
                "doc_id": doc.doc_id,
                "candidate_id": doc.candidate_id,
                "raw_text": doc.raw_text,
                "tokens": [
                    {
                        "text": t.text,
                        "char_start": t.char_start,
                        "char_end": t.char_end,
                        "line": t.line_number
                    }
                    for t in doc.tokens
                ],
                "annotations": [
                    {
                        "entity_type": a.entity_type,
                        "char_start": a.char_start,
                        "char_end": a.char_end,
                        "text": a.text,
                        "layer": a.layer,
                        "confidence": a.confidence,
                        "annotator": a.annotator
                    }
                    for a in doc.annotations
                ],
                "metadata": doc.metadata,
                "status": doc.status
            }
            export_data["documents"].append(doc_data)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📤 Exported {len(documents)} docs to custom JSON: {output_path}")
        return output_path


# =============================================================================
# 🗄️ ANNOTATION STORAGE — Save/Load from Database
# =============================================================================

class AnnotationStorage:
    """
    🗄️ Manages annotation storage in SQLite.
    
    Creates new tables in the existing resume_extractions.db to
    store annotation data alongside the extraction results.
    """
    
    def __init__(self, db_path: str = "resume_extractions.db"):
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create annotation tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ner_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL,
                    doc_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    text_content TEXT,
                    layer INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 1.0,
                    annotator TEXT DEFAULT 'human',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(doc_id, entity_type, char_start, char_end, layer)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ner_documents (
                    doc_id TEXT PRIMARY KEY,
                    candidate_id INTEGER,
                    status TEXT DEFAULT 'pending',
                    annotator TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    function TEXT DEFAULT '',
                    industry TEXT DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ner_ann_candidate 
                ON ner_annotations(candidate_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ner_ann_doc 
                ON ner_annotations(doc_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ner_ann_type 
                ON ner_annotations(entity_type)
            """)
            
            # Add function/industry columns if they don't exist (for existing databases)
            try:
                conn.execute("ALTER TABLE ner_documents ADD COLUMN function TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                conn.execute("ALTER TABLE ner_documents ADD COLUMN industry TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass

            # ----------------------------------------------------------
            # 🔄 MIGRATION: verified_candidates schema v1 → v2
            # 
            # v1 had single-value columns: email, phone, location, expected_location
            # v2 uses JSON arrays: emails_json, phones_json, locations_json, expected_locations_json
            #
            # If the old schema is detected, we drop and recreate the table.
            # Data loss is acceptable here because the table is always
            # rebuilt from ner_annotations on the next save. Like tearing
            # down last season's set to build a BETTER one! 🎭🔨
            # ----------------------------------------------------------
            try:
                cursor = conn.execute("PRAGMA table_info(verified_candidates)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if existing_cols and 'email' in existing_cols and 'emails_json' not in existing_cols:
                    logger.info("🔄 Migrating verified_candidates: v1 (single-value) → v2 (multi-value JSON)")
                    conn.execute("DROP TABLE IF EXISTS verified_candidates")
                    logger.info("   🗑️ Old table dropped — will be recreated with new schema")
            except sqlite3.OperationalError:
                pass  # Table doesn't exist yet — perfect, nothing to migrate!

            # ----------------------------------------------------------
            # ✅ TABLE 3: verified_candidates
            #
            # 🌟 THE STAR OF THE SHOW — Human-verified resume data! 🌟
            #
            # When an annotator edits data in Save Preview and hits Save,
            # this table captures the CORRECTED, VERIFIED version of
            # each candidate's profile — separate from the original
            # structured_extractions (the "before" photo 📸).
            #
            # Think of structured_extractions as the rough draft and
            # verified_candidates as the FINAL PRINT — polished,
            # reviewed, and ready for the runway! 💃✨
            #
            # Design decisions:
            #   • Single-value fields (name, DOB, gender) → TEXT columns
            #   • Multi-value fields (skills, jobs, degrees) → JSON arrays
            #   • INSERT OR REPLACE on candidate_id → always latest version
            #   • Tracks WHO verified and WHEN for audit trail
            #   • change_summary captures WHAT was edited (diff from original)
            # ----------------------------------------------------------
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verified_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER NOT NULL UNIQUE,
                    doc_id TEXT NOT NULL,

                    -- 👤 PERSONAL INFO
                    -- Single-value fields: only ONE per candidate (there is only ONE you, darling! 💅)
                    name TEXT,
                    date_of_birth TEXT,
                    gender TEXT,
                    marital_status TEXT,
                    nationality TEXT,
                    nric_id TEXT,

                    -- Multi-value personal fields: candidates can have MULTIPLE!
                    -- 🐛 BUG FIX: Previously email, phone, location were single TEXT
                    -- columns using first_val(). But the frontend allows multiple
                    -- entries (work + personal email, mobile + office phone, etc.).
                    -- Only the FIRST value was saved — the rest vanished! 😱
                    -- Now stored as JSON arrays: ["work@co.com", "personal@mail.com"]
                    emails_json TEXT DEFAULT '[]',
                    phones_json TEXT DEFAULT '[]',
                    locations_json TEXT DEFAULT '[]',
                    expected_locations_json TEXT DEFAULT '[]',

                    -- 💼 PROFESSIONAL EXPERIENCE (multi-value → JSON arrays)
                    -- Each JSON array contains strings, e.g. ["Software Engineer", "Team Lead"]
                    job_titles_json TEXT DEFAULT '[]',
                    organizations_json TEXT DEFAULT '[]',
                    work_dates_json TEXT DEFAULT '[]',
                    work_locations_json TEXT DEFAULT '[]',
                    job_descriptions_json TEXT DEFAULT '[]',
                    metrics_json TEXT DEFAULT '[]',
                    project_titles_json TEXT DEFAULT '[]',
                    project_descriptions_json TEXT DEFAULT '[]',
                    summary_text TEXT,

                    -- 🎓 EDUCATION (multi-value → JSON arrays)
                    degrees_json TEXT DEFAULT '[]',
                    institutions_json TEXT DEFAULT '[]',
                    fields_of_study_json TEXT DEFAULT '[]',
                    edu_dates_json TEXT DEFAULT '[]',
                    gpas_json TEXT DEFAULT '[]',

                    -- 🛠️ SKILLS & CERTIFICATIONS (multi-value → JSON arrays)
                    skills_json TEXT DEFAULT '[]',
                    soft_skills_json TEXT DEFAULT '[]',
                    certifications_json TEXT DEFAULT '[]',
                    language_skills_json TEXT DEFAULT '[]',
                    proficiencies_json TEXT DEFAULT '[]',
                    cert_issuers_json TEXT DEFAULT '[]',
                    cert_dates_json TEXT DEFAULT '[]',
                    skill_categories_json TEXT DEFAULT '[]',

                    -- 🧠 CLASSIFICATION
                    function TEXT DEFAULT '',
                    industry TEXT DEFAULT '',

                    -- 📊 VERIFICATION META
                    -- verification_status tracks the annotation workflow stage
                    --   'draft'     → saved mid-edit, not yet reviewed
                    --   'verified'  → human has reviewed and confirmed
                    verification_status TEXT DEFAULT 'draft',
                    verified_by TEXT DEFAULT 'annotator',
                    change_summary TEXT DEFAULT '',

                    -- 📅 AUDIT TRAIL
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_verified_candidate
                ON verified_candidates(candidate_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_verified_status
                ON verified_candidates(verification_status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_verified_function
                ON verified_candidates(function)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_verified_industry
                ON verified_candidates(industry)
            """)
            
            conn.commit()
            logger.info("🗄️ NER annotation tables initialized")
        except sqlite3.Error as e:
            logger.error(f"❌ Annotation table creation failed: {e}")
        finally:
            conn.close()
    
    def save_document(self, doc: AnnotatedDocument):
        """Save an annotated document and all its annotations."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.datetime.now().isoformat()
        
        try:
            # Save document metadata
            conn.execute("""
                INSERT OR REPLACE INTO ner_documents
                (doc_id, candidate_id, status, annotator, notes, function, industry, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.doc_id, doc.candidate_id, doc.status,
                doc.annotator, json.dumps(doc.metadata),
                doc.metadata.get("function", ""),
                doc.metadata.get("industry", ""),
                now
            ))
            
            # Delete existing annotations for this doc (replace strategy)
            conn.execute(
                "DELETE FROM ner_annotations WHERE doc_id = ?",
                (doc.doc_id,)
            )
            
            # Insert all annotations
            for ann in doc.annotations:
                conn.execute("""
                    INSERT OR REPLACE INTO ner_annotations 
                        (candidate_id, doc_id, entity_type, char_start, char_end,
                        text_content, layer, confidence, annotator, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    doc.candidate_id, doc.doc_id,
                    ann.entity_type, ann.char_start, ann.char_end,
                    ann.text, ann.layer, ann.confidence, ann.annotator
                ))
            
            conn.commit()
            logger.info(
                f"💾 Saved doc {doc.doc_id}: {len(doc.annotations)} annotations"
            )
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"❌ Failed to save annotations: {e}")
        finally:
            conn.close()
    
    def load_document(
        self,
        doc_id: str,
        db_path_override: Optional[str] = None
    ) -> Optional[AnnotatedDocument]:
        """Load an annotated document and its annotations."""
        db = db_path_override or self.db_path
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        
        try:
            # Load document metadata
            doc_row = conn.execute(
                "SELECT * FROM ner_documents WHERE doc_id = ?",
                (doc_id,)
            ).fetchone()
            
            if not doc_row:
                return None
            
            # Load raw text from raw_extractions
            raw_row = conn.execute("""
                SELECT raw_text FROM raw_extractions
                WHERE candidate_id = ?
                ORDER BY extraction_timestamp DESC LIMIT 1
            """, (doc_row["candidate_id"],)).fetchone()
            
            raw_text = raw_row["raw_text"] if raw_row else ""
            
            # Load annotations
            ann_rows = conn.execute("""
                SELECT * FROM ner_annotations
                WHERE doc_id = ?
                ORDER BY layer, char_start
            """, (doc_id,)).fetchall()
            
            annotations = [
                SpanAnnotation(
                    entity_type=row["entity_type"],
                    char_start=row["char_start"],
                    char_end=row["char_end"],
                    text=row["text_content"] or "",
                    layer=row["layer"],
                    confidence=row["confidence"],
                    annotator=row["annotator"]
                )
                for row in ann_rows
            ]
            
            # Build the document
            tokenizer = ResumeTokenizer()
            tokens = tokenizer.tokenize(raw_text)
            
            # Load metadata from notes, and include function/industry from dedicated columns
            metadata = json.loads(doc_row["notes"]) if doc_row["notes"] else {}
            metadata["function"] = doc_row["function"] or ""
            metadata["industry"] = doc_row["industry"] or ""
            
            doc = AnnotatedDocument(
                doc_id=doc_id,
                candidate_id=doc_row["candidate_id"],
                raw_text=raw_text,
                tokens=tokens,
                annotations=annotations,
                metadata=metadata,
                annotator=doc_row["annotator"],
                status=doc_row["status"]
            )
            
            return doc
            
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to load document {doc_id}: {e}")
            return None
        finally:
            conn.close()
    
    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get statistics about annotation progress."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            stats = {}
            
            # Document counts by status
            rows = conn.execute("""
                SELECT status, COUNT(*) as cnt 
                FROM ner_documents GROUP BY status
            """).fetchall()
            stats["doc_status"] = {r["status"]: r["cnt"] for r in rows}
            stats["total_docs"] = sum(stats["doc_status"].values())
            
            # Annotation counts by entity type
            rows = conn.execute("""
                SELECT entity_type, COUNT(*) as cnt
                FROM ner_annotations GROUP BY entity_type
                ORDER BY cnt DESC
            """).fetchall()
            stats["entity_counts"] = {r["entity_type"]: r["cnt"] for r in rows}
            stats["total_annotations"] = sum(stats["entity_counts"].values())
            
            # Auto vs human annotations
            rows = conn.execute("""
                SELECT 
                    SUM(CASE WHEN annotator LIKE 'auto%' THEN 1 ELSE 0 END) as auto_count,
                    SUM(CASE WHEN annotator = 'human' THEN 1 ELSE 0 END) as human_count
                FROM ner_annotations
            """).fetchone()
            stats["auto_annotations"] = rows["auto_count"] or 0
            stats["human_annotations"] = rows["human_count"] or 0
            
            return stats
            
        except sqlite3.Error as e:
            logger.error(f"❌ Stats query failed: {e}")
            return {}
        finally:
            conn.close()

    def update_document_metadata(self, doc_id: str, status: Optional[str] = None,
                             function: Optional[str] = None, industry: Optional[str] = None):
        """
        Update status and/or classification for a document.

        🐛 BUG FIX: Previously used a bare UPDATE which silently affected
        zero rows when the ner_documents record didn't exist yet (first save).
        Now uses INSERT OR IGNORE to ensure the row EXISTS before updating.
        Think of it as making sure the guest list is on the table BEFORE
        writing names on it! 📋✨
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # ── Step 1: Ensure the row exists (UPSERT guard) ──────────────
            # Extract candidate_id from doc_id format "doc_<candidate_id>"
            candidate_id = None
            if doc_id.startswith("doc_"):
                try:
                    candidate_id = int(doc_id[4:])
                except ValueError:
                    pass

            # INSERT OR IGNORE: creates the row with defaults if it doesn't
            # exist; does nothing if it already does. No data is overwritten.
            conn.execute("""
                INSERT OR IGNORE INTO ner_documents
                    (doc_id, candidate_id, status, function, industry, created_at, updated_at)
                VALUES (?, ?, 'pending', '', '', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (doc_id, candidate_id))

            # ── Step 2: Now UPDATE the row (guaranteed to exist) ──────────
            updates = []
            params = []
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if function is not None:
                updates.append("function = ?")
                params.append(function)
            if industry is not None:
                updates.append("industry = ?")
                params.append(industry)
            if not updates:
                conn.commit()  # Commit the INSERT OR IGNORE at minimum
                return
            updates.append("updated_at = CURRENT_TIMESTAMP")
            query = f"UPDATE ner_documents SET {', '.join(updates)} WHERE doc_id = ?"
            params.append(doc_id)
            conn.execute(query, params)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update document metadata for {doc_id}: {e}")
        finally:
            conn.close()

    def save_annotations(self, doc_id: str, annotations_list: List[Dict],
                     candidate_id: Optional[int] = None):
        """
        Replace all annotations for a document with a new list.
        Deduplicates based on the unique key (doc_id, entity_type, char_start, char_end, layer).
        """
        if not annotations_list and candidate_id is None:
            # If no annotations and no candidate_id, try to determine from doc_id
            if doc_id.startswith("doc_"):
                candidate_id = int(doc_id[4:])
            else:
                # Still no candidate_id – we can't insert, but we can still delete existing
                pass

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:  # auto-commit on success, rollback on exception
                # Delete all existing annotations for this document
                conn.execute("DELETE FROM ner_annotations WHERE doc_id = ?", (doc_id,))

                if not annotations_list:
                    # Nothing to insert, we're done
                    return

                # Retrieve or determine candidate_id
                if candidate_id is None:
                    row = conn.execute(
                        "SELECT candidate_id FROM ner_documents WHERE doc_id = ?",
                        (doc_id,)
                    ).fetchone()
                    if row:
                        candidate_id = row[0]
                    elif doc_id.startswith("doc_"):
                        candidate_id = int(doc_id[4:])
                    else:
                        raise ValueError(f"Could not determine candidate_id for {doc_id}")

                # Deduplicate annotations based on unique key
                # 🐛 BUG FIX: Manual annotations from Save Preview all have
                # char_start=-1, char_end=-1. Without including text_content in
                # the key, adding 3 SKILL entries manually would collapse to 1!
                # Now we include text for manual entries so each unique text survives.
                # Like giving each guest their OWN nametag instead of one per costume! 🎭
                seen_keys = set()
                unique_anns = []
                for ann in annotations_list:
                    cs = int(ann.get("char_start", -1))
                    ce = int(ann.get("char_end", -1))
                    # For manual entries (no char positions), include text in the key
                    # so different texts are treated as distinct annotations
                    text_key = ann.get("text", "").strip() if cs < 0 else ""
                    key = (
                        doc_id,
                        ann.get("entity_type"),
                        cs,
                        ce,
                        int(ann.get("layer", 0)),
                        text_key
                    )
                    if key in seen_keys:
                        continue  # skip duplicate
                    seen_keys.add(key)
                    unique_anns.append(ann)

                # Insert all unique annotations
                # 🐛 BUG FIX: Manual annotations (char_start < 0) all share -1/-1
                # which violates the UNIQUE constraint on ner_annotations table.
                # Assign each manual annotation a unique negative sentinel so the
                # SQL UNIQUE key stays happy. Like giving each guest a different
                # seat number even when they're all walk-ins! 💺✨
                manual_counter = -1  # Decreasing unique IDs for manual entries
                for ann in unique_anns:
                    cs = int(ann.get("char_start", 0))
                    ce = int(ann.get("char_end", 0))

                    # If this is a manual entry (negative positions), assign unique sentinels
                    if cs < 0 or ce < 0:
                        cs = manual_counter
                        ce = manual_counter  # Same value — marks it as manual
                        manual_counter -= 1

                    conn.execute("""
                        INSERT INTO ner_annotations 
                            (candidate_id, doc_id, entity_type, char_start, char_end,
                            text_content, layer, confidence, annotator, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        candidate_id,
                        doc_id,
                        ann.get("entity_type"),
                        cs,
                        ce,
                        ann.get("text", ""),
                        int(ann.get("layer", 0)),
                        float(ann.get("confidence", 1.0)),
                        ann.get("annotator", "human")
                    ))

                # If we reach here without exception, the with block commits
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error saving annotations for {doc_id}: {e}")
            # Re-raise so the API endpoint can handle it
            raise
        except Exception as e:
            logger.error(f"Failed to save annotations for {doc_id}: {e}")
            raise
        finally:
            conn.close()

    def update_structured_extraction(self, doc_id: str, annotations_list: List[Dict],
                                      function: str = '', industry: str = '',
                                      status: str = 'in_progress'):
        """
        💃 Update the structured_extractions table with human-corrected data!

        This is the REAL GLOW-UP, darling! 🎭✨ Instead of storing corrected
        data in a separate table, we update the ORIGINAL structured_extractions
        row — so the "after glam" table always reflects the LATEST truth,
        whether it came from AI or human review.

        Think of it like a wardrobe stylist fixing the outfit DIRECTLY on the
        mannequin instead of writing notes on a separate card! 👗✨

        Architecture:
          1. Group annotations by entity_type
          2. Map NER entities → structured_extractions columns:
             • Single-value fields (name, DOB) → first value
             • Multi-value personal (email, phone) → join with "; "
             • Skills → skills_raw (pipe-separated) + skills_json (JSON array)
             • Experience → experience_raw (formatted) + experience_json (JSON array)
             • Education → education_raw (formatted) + education_json (JSON array)
             • Certs/Languages/Projects → formatted text
          3. UPDATE the existing row (not INSERT — row already exists from extraction)
          4. Mark reviewed=1 and set review_notes with change summary

        Args:
            doc_id:           Document ID (format: "doc_<candidate_id>")
            annotations_list: List of annotation dicts from the frontend
            function:         Job function classification
            industry:         Industry sector classification
            status:           Annotation status ('in_progress' | 'completed')
        """
        # ── Extract candidate_id from doc_id ──────────────────────────
        candidate_id = None
        if doc_id.startswith("doc_"):
            try:
                candidate_id = int(doc_id[4:])
            except ValueError:
                logger.error(f"Cannot parse candidate_id from doc_id: {doc_id}")
                return
        if candidate_id is None:
            logger.error(f"No valid candidate_id for structured update: {doc_id}")
            return

        # ── Group annotations by entity_type ──────────────────────────
        # Collects all text values for each entity type into a list.
        # Like sorting your wardrobe by colour — everything in its place! 👗
        grouped = {}
        for ann in annotations_list:
            etype = ann.get("entity_type", "")
            text = ann.get("text", "").strip()
            if not etype or not text:
                continue
            if etype not in grouped:
                grouped[etype] = []
            grouped[etype].append(text)

        # ── Helpers ───────────────────────────────────────────────────
        def first_val(entity_type: str) -> str:
            """Return first non-empty value or empty string."""
            vals = grouped.get(entity_type, [])
            return vals[0] if vals else ""

        def all_vals(entity_type: str) -> list:
            """Return all values as a list."""
            return grouped.get(entity_type, [])

        def join_vals(entity_type: str, sep: str = "; ") -> str:
            """Join all values with separator."""
            vals = grouped.get(entity_type, [])
            return sep.join(vals) if vals else ""

        # ══════════════════════════════════════════════════════════════
        # 👤 PERSONAL INFO — Map single/multi-value fields
        # ══════════════════════════════════════════════════════════════
        name = first_val('PERSON_NAME')
        # Email & phone: join multiple with "; " since the column is TEXT
        # e.g. "work@co.com; personal@gmail.com"
        email = join_vals('EMAIL', '; ')
        phone = join_vals('PHONE', '; ')
        date_of_birth = first_val('DATE_OF_BIRTH')
        location = join_vals('LOCATION', '; ')
        nationality = first_val('NATIONALITY')

        # ══════════════════════════════════════════════════════════════
        # 🛠️ SKILLS — Dual storage: raw (pipe-separated) + JSON array
        # ══════════════════════════════════════════════════════════════
        # Combine SKILL + SOFT_SKILL + SKILL_CATEGORY into unified skills
        all_skills = (
            all_vals('SKILL') +
            all_vals('SOFT_SKILL') +
            all_vals('SKILL_CATEGORY')
        )
        skills_raw = ' | '.join(all_skills) if all_skills else ""
        skills_json_str = json.dumps(all_skills, ensure_ascii=False) if all_skills else "[]"

         # ══════════════════════════════════════════════════════════════
        # 💼 EXPERIENCE — Build structured work history
        #
        # IMPROVED: Position-aware grouping! 🎯
        #
        # The old approach used naive parallel-zipping of entity arrays:
        #   job_titles[0] + organizations[0] → Job 1
        #   job_titles[1] + organizations[1] → Job 2
        #   descriptions stored as flat array
        #
        # This FAILED because descriptions weren't associated with
        # their parent job. Now we use char_start positions to group
        # annotations into job blocks using structural boundaries.
        #
        # Think of it like organising a filing cabinet — each folder
        # (job block) gets ALL its documents (descriptions) inside it,
        # not randomly distributed across folders! 📁💅
        # ══════════════════════════════════════════════════════════════

        # Step 1: Collect work annotations WITH positions
        WORK_TYPES = {"JOB_TITLE", "ORGANIZATION", "WORK_DATE", "JOB_DESCRIPTION", "WORK_LOCATION"}
        HEADER_TYPES = {"JOB_TITLE", "ORGANIZATION", "WORK_DATE"}
        
        work_anns = sorted(
            [
                a for a in annotations_list
                if a.get("entity_type", "") in WORK_TYPES
                and a.get("text", "").strip()
                and a.get("layer", 0) == 0
            ],
            key=lambda a: a.get("char_start", 0)
        )

        # Step 2: Group into job blocks using structural boundaries
        experience_entries = []
        experience_raw_parts = []

        if work_anns:
            blocks = []
            current_block = [work_anns[0]]
            current_block_types = {work_anns[0].get("entity_type", "")}

            for ann in work_anns[1:]:
                etype = ann.get("entity_type", "")
                prev_end = current_block[-1].get("char_end", 0)
                gap = ann.get("char_start", 0) - prev_end

                is_header = etype in HEADER_TYPES
                block_has_header = bool(current_block_types & HEADER_TYPES)
                block_has_descriptions = "JOB_DESCRIPTION" in current_block_types

                # Structural boundary: new header after descriptions
                structural_boundary = (
                    is_header
                    and block_has_header
                    and block_has_descriptions
                )

                # Double header: same header type already in block
                # (e.g. two JOB_TITLEs = definitely two different jobs)
                double_header = (
                    etype in {"JOB_TITLE", "ORGANIZATION"}
                    and "JOB_TITLE" in current_block_types
                    and "ORGANIZATION" in current_block_types
                    and etype in current_block_types
                )

                # Gap boundary: fallback for unusual formatting
                gap_boundary = gap > 400

                if structural_boundary or double_header or gap_boundary:
                    blocks.append(current_block)
                    current_block = [ann]
                    current_block_types = {etype}
                else:
                    current_block.append(ann)
                    current_block_types.add(etype)

            blocks.append(current_block)

            # Step 3: Convert each block into structured format
            for block in blocks:
                title = ""
                org = ""
                date = ""
                loc = ""
                descriptions = []

                for ann in block:
                    etype = ann.get("entity_type", "")
                    text = ann.get("text", "").strip()

                    if etype == "JOB_TITLE" and not title:
                        title = text
                    elif etype == "ORGANIZATION" and not org:
                        org = text
                    elif etype == "WORK_DATE" and not date:
                        date = text
                    elif etype == "WORK_LOCATION" and not loc:
                        loc = text
                    elif etype == "JOB_DESCRIPTION":
                        if text:
                            descriptions.append(text)

                if not title and not org:
                    continue  # Skip orphaned description-only blocks

                entry = {
                    "title": title,
                    "organization": org,
                    "date": date,
                    "location": loc,
                    "responsibilities": descriptions,
                }
                experience_entries.append(entry)

                # Build raw string for display
                raw_part = f"{org} - {title}" if org and title else (org or title)
                if date:
                    raw_part += f" ({date})"
                experience_raw_parts.append(raw_part)

        # Build the final JSON structure
        # NOTE: New format includes responsibilities PER position!
        # Old format had flat "descriptions" array — new format
        # nests them correctly inside each position.
        experience_obj = {
            "positions": [
                {
                    "title": e["title"],
                    "organization": e["organization"],
                    "date": e["date"],
                    "location": e["location"],
                }
                for e in experience_entries
            ],
            "descriptions": [
                desc
                for e in experience_entries
                for desc in e.get("responsibilities", [])
            ],
            "metrics": all_vals('METRIC'),
            # NEW: Per-position responsibility grouping for accurate export
            "positions_with_responsibilities": experience_entries,
        }
        experience_raw = ' || '.join(experience_raw_parts) if experience_raw_parts else ""
        experience_json_str = json.dumps(experience_obj, ensure_ascii=False)


        # ══════════════════════════════════════════════════════════════
        # 🎓 EDUCATION — Build structured education history
        # Same zipping strategy as experience! 🎯
        # ══════════════════════════════════════════════════════════════
        degrees = all_vals('DEGREE')
        institutions = all_vals('INSTITUTION')
        fields_of_study = all_vals('FIELD_OF_STUDY')
        edu_dates = all_vals('EDU_DATE')
        gpas = all_vals('GPA')

        edu_count = max(len(degrees), len(institutions), 1)
        education_entries = []
        education_raw_parts = []

        for i in range(edu_count):
            deg = degrees[i] if i < len(degrees) else ""
            inst = institutions[i] if i < len(institutions) else ""
            field = fields_of_study[i] if i < len(fields_of_study) else ""
            date = edu_dates[i] if i < len(edu_dates) else ""
            gpa = gpas[i] if i < len(gpas) else ""

            if not deg and not inst:
                continue

            entry = {
                "degree": deg,
                "institution": inst,
                "field_of_study": field,
                "date": date,
                "gpa": gpa,
            }
            education_entries.append(entry)

            raw_part = f"{deg} from {inst}" if deg and inst else (deg or inst)
            if date:
                raw_part += f" ({date})"
            education_raw_parts.append(raw_part)

        education_raw = ' || '.join(education_raw_parts) if education_raw_parts else ""
        education_json_str = json.dumps(education_entries, ensure_ascii=False)

        # ══════════════════════════════════════════════════════════════
        # 📋 ADDITIONAL FIELDS — Simple text formatting
        # ══════════════════════════════════════════════════════════════
        summary = first_val('SUMMARY_TEXT')

        # Certifications: "CertName (Issuer, Date)"
        certs = all_vals('CERTIFICATION')
        cert_issuers = all_vals('CERT_ISSUER')
        cert_dates = all_vals('CERT_DATE')
        cert_parts = []
        for i, cert in enumerate(certs):
            issuer = cert_issuers[i] if i < len(cert_issuers) else ""
            date = cert_dates[i] if i < len(cert_dates) else ""
            detail = f"{cert}"
            extras = [x for x in [issuer, date] if x]
            if extras:
                detail += f" ({', '.join(extras)})"
            cert_parts.append(detail)
        certifications = ' | '.join(cert_parts) if cert_parts else ""

        # Languages: "Language (Proficiency)"
        lang_skills = all_vals('LANGUAGE_SKILL')
        profs = all_vals('PROFICIENCY')
        lang_parts = []
        for i, lang in enumerate(lang_skills):
            prof = profs[i] if i < len(profs) else ""
            lang_parts.append(f"{lang} ({prof})" if prof else lang)
        languages = ', '.join(lang_parts) if lang_parts else ""

        # Projects: "ProjectTitle: Description"
        proj_titles = all_vals('PROJECT_TITLE')
        proj_descs = all_vals('PROJECT_DESCRIPTION')
        proj_parts = []
        for i, title in enumerate(proj_titles):
            desc = proj_descs[i] if i < len(proj_descs) else ""
            proj_parts.append(f"{title}: {desc}" if desc else title)
        projects = ' || '.join(proj_parts) if proj_parts else ""

        # ══════════════════════════════════════════════════════════════
        # 📝 REVIEW NOTES — Track what was changed
        # ══════════════════════════════════════════════════════════════
        populated_types = [etype for etype, vals in grouped.items() if vals]
        review_notes = json.dumps({
            "updated_via": "annotation_tool_save_preview",
            "annotation_status": status,
            "function": function,
            "industry": industry,
            "fields_updated": populated_types,
            "total_annotations": len(annotations_list),
            "field_counts": {e: len(v) for e, v in grouped.items() if v}
        }, ensure_ascii=False)

        # ══════════════════════════════════════════════════════════════
        # 💾 UPDATE structured_extractions — The Main Event! 🎭
        # ══════════════════════════════════════════════════════════════
        conn = sqlite3.connect(self.db_path)
        try:
            # ── Ensure function/industry columns exist ────────────────
            # These may not be in the original schema — add them gracefully
            for col in ['function', 'industry']:
                try:
                    conn.execute(
                        f"ALTER TABLE structured_extractions ADD COLUMN {col} TEXT DEFAULT ''"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists — that's fine!

            # ── UPDATE the existing row for this candidate ────────────
            # We only update fields that have annotation data.
            # If a field has no annotations, we leave the original value
            # intact — don't overwrite good AI data with blanks!
            # Like a stylist: fix what needs fixing, keep what's already fabulous! 💅
            updates = []
            params = []

            # Helper: only add to UPDATE if we have data for this field
            def add_update(column: str, value: str):
                """Only update a column if the value is non-empty."""
                if value:
                    updates.append(f"{column} = ?")
                    params.append(value)

            # 👤 Personal Info
            add_update('name', name)
            add_update('email', email)
            add_update('phone', phone)
            add_update('date_of_birth', date_of_birth)
            add_update('location', location)
            add_update('nationality', nationality)

            # 🛠️ Skills
            add_update('skills_raw', skills_raw)
            add_update('skills_json', skills_json_str)

            # 💼 Experience
            add_update('experience_raw', experience_raw)
            add_update('experience_json', experience_json_str)

            # 🎓 Education
            add_update('education_raw', education_raw)
            add_update('education_json', education_json_str)

            # 📋 Additional
            add_update('summary', summary)
            add_update('certifications', certifications)
            add_update('languages', languages)
            add_update('projects', projects)

            # 🧠 Classification
            add_update('function', function)
            add_update('industry', industry)

            # 📊 Always update review metadata
            updates.append("reviewed = 1")
            updates.append("review_notes = ?")
            params.append(review_notes)
            updates.append("updated_at = CURRENT_TIMESTAMP")

            if not updates:
                logger.warning(f"⚠️ No data to update for candidate {candidate_id}")
                return

            # Build the final UPDATE query
            query = f"UPDATE structured_extractions SET {', '.join(updates)} WHERE candidate_id = ?"
            params.append(candidate_id)

            cursor = conn.execute(query, params)
            conn.commit()

            if cursor.rowcount == 0:
                logger.warning(
                    f"⚠️ No structured_extractions row found for candidate {candidate_id}. "
                    f"The UPDATE affected 0 rows — data was NOT saved. "
                    f"Ensure the extraction pipeline ran first!"
                )
            else:
                logger.info(
                    f"💃 structured_extractions UPDATED for candidate {candidate_id} — "
                    f"{len(populated_types)} field types, "
                    f"{len(annotations_list)} annotations, "
                    f"reviewed=1"
                )
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to update structured_extractions for {doc_id}: {e}")
            conn.rollback()
        finally:
            conn.close()

# =============================================================================
# 🖥️ CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for NER schema operations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="💅✨ Fairy Codemother's NER Schema Engine ✨💅"
    )
    parser.add_argument("--export-schema", type=str, help="Export schema to JSON file")
    parser.add_argument("--export-tags", action="store_true", help="Print all BIO tags")
    parser.add_argument("--test-tokenize", type=str, help="Test tokenizer on text")
    parser.add_argument("--test-preannotate", type=str, help="Test pre-annotator on text")
    parser.add_argument("--stats", type=str, help="Show annotation stats for a database")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    schema = EntitySchema()
    
    if args.export_schema:
        export = schema.export_schema()
        with open(args.export_schema, "w") as f:
            json.dump(export, f, indent=2)
        print(f"✅ Schema exported to {args.export_schema}")
        print(f"   {export['total_entity_types']} entity types, {export['total_bio_tags']} BIO tags")
    
    elif args.export_tags:
        tags = schema.get_bio_tags()
        print(f"\n🏷️ BIO TAGS ({len(tags)} total):\n")
        for tag in tags:
            print(f"  {tag}")
    
    elif args.test_tokenize:
        tokenizer = ResumeTokenizer()
        tokens = tokenizer.tokenize(args.test_tokenize)
        print(f"\n✂️ TOKENS ({len(tokens)}):\n")
        for t in tokens:
            print(f"  [{t.char_start:4d}:{t.char_end:4d}] L{t.line_number} '{t.text}'")
    
    elif args.test_preannotate:
        pre = PreAnnotator(schema)
        anns = pre.pre_annotate(args.test_preannotate, confidence_threshold=0.5)
        print(f"\n🤖 PRE-ANNOTATIONS ({len(anns)}):\n")
        for a in anns:
            print(f"  [{a.char_start:4d}:{a.char_end:4d}] {a.entity_type:20s} "
                  f"'{a.text}' (conf: {a.confidence:.0%}, by: {a.annotator})")
    
    elif args.stats:
        storage = AnnotationStorage(args.stats)
        stats = storage.get_annotation_stats()
        print(f"\n📊 ANNOTATION STATS:\n")
        print(f"  Documents: {stats.get('total_docs', 0)}")
        print(f"  Annotations: {stats.get('total_annotations', 0)}")
        print(f"  Auto: {stats.get('auto_annotations', 0)}")
        print(f"  Human: {stats.get('human_annotations', 0)}")
        if stats.get("entity_counts"):
            print(f"\n  Entity breakdown:")
            for etype, count in stats["entity_counts"].items():
                print(f"    {etype}: {count}")
    
    else:
        # Default: print schema summary
        print(f"\n💅✨ NER SCHEMA SUMMARY ✨💅\n")
        print(f"  Total entity types: {len(schema.entities)}")
        print(f"  Total BIO tags: {len(schema.get_bio_tags())}")
        print(f"\n  Categories:")
        for cat in EntityCategory:
            entities = schema.get_by_category(cat)
            print(f"    {cat.name}: {len(entities)} types")
            for e in entities:
                nested = " [NESTABLE]" if e.allows_nesting else ""
                required = " ★" if e.is_required else ""
                print(f"      {e.color} {e.name}{nested}{required}")


if __name__ == "__main__":
    main()