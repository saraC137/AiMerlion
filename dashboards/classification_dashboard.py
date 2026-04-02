"""
classification_dashboard.py

💅✨ FAIRY CODEMOTHER'S CLASSIFICATION & ACCURACY REVIEW DASHBOARD ✨💅

A standalone Flask web app (port 5001) that sits alongside the NER Annotation
Tool and gives HR users two superpowers:

  1. 🎯 AUTO-CLASSIFICATION
     Reads NER annotations (JOB_TITLE, SKILL, INDUSTRY, FUNCTION, SUMMARY…)
     and scores each candidate against predefined keyword maps to assign:
       - Function  (17 options — Administration, IT, Sales, etc.)
       - Industry  (24 options — Banking, Healthcare, Semiconductor, etc.)
     Confidence score (0–100%) calculated from weighted keyword matches.

  2. 🔬 ANNOTATION ACCURACY REVIEW
     For every completed annotation, shows:
       - Entity coverage score (which required entities are present?)
       - Entity chip display (colour-matched to the annotation tool palette)
       - Missing entity warnings with recommended actions
       - Field-by-field check against the target JSON schema

  3. ✍️ MANUAL OVERRIDE
     If the auto-classifier is wrong (it happens, darling! 💅),
     the HR user can correct Function + Industry via dropdowns
     and save. Every correction is logged for potential ML retraining.

Architecture:
  - Flask web server on port 5001
  - Reads from same DB as annotation_tool.py (ner_annotations.db)
  - Writes classifications to new `candidate_classifications` table
  - Zero extra dependencies beyond Flask + what annotation_tool already uses

Usage:
    python classification_dashboard.py
    # Then open http://localhost:5001 in your browser

Dependencies:
    pip install flask --break-system-packages
"""

import sqlite3
import json
import os
import re
import math
import datetime
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from flask import (
    Flask, render_template_string, request, jsonify,
    abort, redirect, url_for, g
)

# =============================================================================
# 🔧 LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 💅 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 🎛️ CONFIGURATION
# =============================================================================

# Same database as annotation_tool.py — we share the data! 💃
DATABASE_PATH = os.environ.get("NER_DB_PATH", "ner_annotations.db")

HOST  = "0.0.0.0"
PORT  = 5001
DEBUG = True

# Candidates per page in the list view
PER_PAGE = 30

# Minimum weighted score to call a match "confident" (not just a guess)
CONFIDENCE_THRESHOLD = 0.15


# =============================================================================
# 📋 TAXONOMY — THE SOURCE OF TRUTH FOR FUNCTION + INDUSTRY
# These are the EXACT labels that appear in dropdowns and badges.
# =============================================================================

FUNCTION_LIST: List[str] = [
    "Administration",
    "Accounting & Finance",
    "Customer Service",
    "Engineering",
    "Facilities Management",
    "Fresh Graduates (Deg/Dip)",
    "Fresh Graduates (ITE)",
    "Human Resources",
    "IT",
    "Legal",
    "Management",
    "Marketing",
    "Procurement",
    "Production",
    "Sales",
    "Shipping & Logistics",
    "Others",
]

INDUSTRY_LIST: List[str] = [
    "Automotive",
    "Banking & Finance",
    "Building/Construction",
    "Chemicals",
    "Education",
    "Engineering - Aerospace",
    "Engineering - Precision",
    "Energy (Oil & Gas)",
    "Environment & Water",
    "Healthcare",
    "Hospitality & Tourism",
    "F&B",
    "FMCG",
    "Infocomm",
    "Medical Technology",
    "Marine & Offshore",
    "Retail",
    "Pharmaceutical/Biotech",
    "Robotics",
    "Semiconductor",
    "Supply Chain Mgt & Logistics",
    "Telco",
    "Trading",
    "Others",
]


# =============================================================================
# 🗺️ KEYWORD MAPS — THE CLASSIFIER'S SECRET WARDROBE 👗
#
# Each entry is:
#   label_name: {
#       "title":    [...],   # keywords in JOB_TITLE  → weight 5 per hit
#       "skills":   [...],   # keywords in SKILL/SOFT_SKILL → weight 2
#       "summary":  [...],   # keywords in SUMMARY_TEXT / JOB_DESCRIPTION → weight 1
#       "org":      [...],   # keywords in ORGANIZATION → weight 2
#   }
#
# Matching is case-insensitive, whole-word or substring (configurable per key).
# All scores are summed, then divided by the MAXIMUM possible score for that
# label to produce a 0–1 confidence. The highest score wins!
# =============================================================================

FUNCTION_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "Administration": {
        "title":   [
            "admin", "administrative", "administrator", "office manager",
            "receptionist", "secretary", "coordinator", "executive assistant",
            "personal assistant", "pa", "clerical", "office assistant",
            "data entry", "document controller", "records",
        ],
        "skills":  [
            "microsoft office", "word", "excel", "outlook", "filing",
            "scheduling", "calendar management", "minute taking",
            "office management", "typing", "data entry",
        ],
        "summary": [
            "administrative", "front desk", "reception", "clerical",
            "office operations", "document management",
        ],
        "org":     [],
    },
    "Accounting & Finance": {
        "title":   [
            "accountant", "finance manager", "financial controller",
            "cfo", "chief financial officer", "treasurer", "auditor",
            "tax manager", "payroll", "bookkeeper", "accounts executive",
            "accounts manager", "financial analyst", "fp&a", "finance director",
            "credit analyst", "billing", "accounts receivable", "accounts payable",
            "internal auditor", "external auditor", "management accountant",
        ],
        "skills":  [
            "accounting", "bookkeeping", "financial reporting", "ifrs", "gaap",
            "taxation", "gst", "payroll", "sap", "oracle financials", "xero",
            "quickbooks", "audit", "budgeting", "forecasting", "variance analysis",
            "cash flow", "accounts receivable", "accounts payable", "balance sheet",
            "p&l", "income statement",
        ],
        "summary": [
            "accounting", "finance", "financial management", "audit",
            "tax compliance", "financial planning",
        ],
        "org":     ["kpmg", "pwc", "deloitte", "ey", "ernst", "bdo", "rsm",
                    "grant thornton"],
    },
    "Customer Service": {
        "title":   [
            "customer service", "customer support", "cs officer",
            "customer care", "service advisor", "client relations",
            "helpdesk", "help desk", "service desk", "contact center",
            "call center", "customer experience", "cx", "crm executive",
            "after-sales", "aftersales",
        ],
        "skills":  [
            "customer service", "crm", "salesforce", "zendesk",
            "freshdesk", "complaint handling", "conflict resolution",
            "customer satisfaction", "csat", "nps", "live chat",
            "call handling", "customer retention", "upselling",
        ],
        "summary": [
            "customer service", "customer satisfaction", "client support",
            "customer experience", "service excellence",
        ],
        "org":     [],
    },
    "Engineering": {
        "title":   [
            "engineer", "engineering manager", "r&d engineer",
            "design engineer", "process engineer", "electrical engineer",
            "mechanical engineer", "civil engineer", "chemical engineer",
            "structural engineer", "piping engineer", "instrument engineer",
            "validation engineer", "test engineer", "field engineer",
            "technical specialist", "systems engineer", "quality engineer",
        ],
        "skills":  [
            "autocad", "solidworks", "ansys", "matlab", "plc",
            "scada", "piping", "hvac", "fea", "cfd", "iso",
            "engineering drawings", "technical drawings", "p&id",
            "python", "c++",  # shared with IT but valid for engineering too
        ],
        "summary": [
            "engineering", "design and development", "research and development",
            "technical solutions", "product development",
        ],
        "org":     [],
    },
    "Facilities Management": {
        "title":   [
            "facilities manager", "facility manager", "fm manager",
            "building manager", "property manager", "estate manager",
            "maintenance manager", "maintenance supervisor",
            "building engineer", "facilities coordinator",
            "infrastructure manager", "asset manager",
        ],
        "skills":  [
            "facilities management", "building maintenance", "hvac",
            "m&e", "mechanical electrical", "preventive maintenance",
            "corrective maintenance", "vendor management", "property management",
            "lease management", "space planning", "bms", "building management system",
        ],
        "summary": [
            "facilities", "building management", "property operations",
            "maintenance operations", "infrastructure management",
        ],
        "org":     ["cbre", "jll", "savills", "colliers", "knight frank",
                    "ISS", "sodexo"],
    },
    "Fresh Graduates (Deg/Dip)": {
        "title":   [
            "fresh graduate", "graduate trainee", "management trainee",
            "intern", "internship", "attachment", "trainee",
            "associate (entry)", "junior associate", "entry level",
        ],
        "skills":  [],
        "summary": [
            "fresh graduate", "recent graduate", "newly graduated",
            "looking for first job", "entry level", "no experience",
            "just graduated", "degree holder", "diploma holder",
        ],
        "org":     [],
    },
    "Fresh Graduates (ITE)": {
        "title":   ["ite graduate", "nitec", "higher nitec"],
        "skills":  ["nitec", "higher nitec"],
        "summary": ["ite", "nitec", "higher nitec", "ite graduate",
                    "ite college"],
        "org":     ["ite college east", "ite college west", "ite college central"],
    },
    "Human Resources": {
        "title":   [
            "hr", "human resources", "hrbp", "hr business partner",
            "talent acquisition", "recruiter", "recruitment specialist",
            "talent management", "learning and development", "l&d",
            "compensation and benefits", "c&b", "hr manager",
            "hr director", "people operations", "org development",
            "hr generalist", "hr executive", "hr specialist",
        ],
        "skills":  [
            "recruitment", "talent acquisition", "onboarding",
            "performance management", "hris", "workday", "successfactors",
            "compensation", "benefits administration", "employee relations",
            "training", "learning development", "l&d", "manpower planning",
            "ir", "industrial relations",
        ],
        "summary": [
            "human resources", "hr function", "people management",
            "talent strategy", "workforce planning",
        ],
        "org":     [],
    },
    "IT": {
        "title":   [
            "software engineer", "software developer", "developer",
            "programmer", "it manager", "it director", "cto",
            "chief technology officer", "devops", "sre", "it support",
            "system administrator", "sysadmin", "network engineer",
            "security analyst", "cybersecurity", "data scientist",
            "data analyst", "data engineer", "ml engineer", "ai engineer",
            "cloud architect", "solutions architect", "backend developer",
            "frontend developer", "full stack developer", "mobile developer",
            "ios developer", "android developer", "qa engineer",
            "business analyst", "it consultant", "erp consultant",
            "database administrator", "dba",
        ],
        "skills":  [
            "python", "java", "javascript", "typescript", "c#", "c++",
            "go", "rust", "sql", "nosql", "react", "angular", "vue",
            "node", "django", "flask", "spring", "kubernetes", "docker",
            "aws", "azure", "gcp", "terraform", "ansible", "git",
            "linux", "networking", "firewall", "vpn", "cloud",
            "machine learning", "deep learning", "nlp", "data science",
            "tableau", "power bi", "spark", "hadoop", "kafka",
        ],
        "summary": [
            "software development", "information technology", "it solutions",
            "digital transformation", "cloud computing", "cybersecurity",
            "data analytics", "artificial intelligence",
        ],
        "org":     [],
    },
    "Legal": {
        "title":   [
            "lawyer", "attorney", "solicitor", "legal counsel",
            "in-house counsel", "legal executive", "paralegal",
            "compliance officer", "regulatory affairs", "legal manager",
            "legal director", "general counsel", "ip lawyer",
            "corporate counsel", "contract manager",
        ],
        "skills":  [
            "legal research", "contract drafting", "litigation",
            "compliance", "regulatory", "intellectual property", "ip",
            "corporate law", "employment law", "contract law",
            "due diligence", "mergers acquisitions", "m&a",
        ],
        "summary": [
            "legal practice", "legal advisory", "compliance management",
            "regulatory compliance", "contract management",
        ],
        "org":     [],
    },
    "Management": {
        "title":   [
            "ceo", "chief executive officer", "managing director", "md",
            "general manager", "gm", "director", "vp", "vice president",
            "head of", "department head", "senior manager", "group manager",
            "operations manager", "business unit head", "country manager",
            "regional manager", "president", "c-suite",
        ],
        "skills":  [
            "strategic planning", "p&l management", "budgeting",
            "team leadership", "stakeholder management", "business strategy",
            "change management", "board management", "executive leadership",
        ],
        "summary": [
            "strategic leadership", "business management", "executive management",
            "cross-functional leadership", "business transformation",
        ],
        "org":     [],
    },
    "Marketing": {
        "title":   [
            "marketing manager", "marketing director", "cmo",
            "brand manager", "digital marketing", "content marketing",
            "social media manager", "seo specialist", "sem specialist",
            "marketing executive", "communications manager", "pr manager",
            "public relations", "advertising manager", "campaign manager",
            "product marketing", "growth hacker", "demand generation",
        ],
        "skills":  [
            "digital marketing", "social media", "seo", "sem", "google ads",
            "facebook ads", "content marketing", "email marketing",
            "brand management", "market research", "marketing analytics",
            "hubspot", "mailchimp", "adobe creative", "copywriting",
            "content creation", "influencer marketing",
        ],
        "summary": [
            "marketing strategy", "brand building", "digital presence",
            "marketing campaigns", "go-to-market",
        ],
        "org":     [],
    },
    "Procurement": {
        "title":   [
            "procurement manager", "procurement executive", "buyer",
            "purchasing manager", "sourcing manager", "category manager",
            "supply chain manager", "vendor manager", "contracts manager",
            "tendering manager", "purchasing executive",
        ],
        "skills":  [
            "procurement", "purchasing", "sourcing", "supplier management",
            "vendor management", "contract negotiation", "tender management",
            "rfq", "rfp", "spend analysis", "sap mm", "oracle procurement",
            "supply chain", "inventory management",
        ],
        "summary": [
            "procurement strategy", "strategic sourcing", "vendor relations",
            "cost optimisation", "supply management",
        ],
        "org":     [],
    },
    "Production": {
        "title":   [
            "production manager", "production supervisor", "production executive",
            "manufacturing manager", "plant manager", "operations manager",
            "quality manager", "qc manager", "qa manager", "line leader",
            "shift supervisor", "production engineer", "process engineer",
            "machinist", "operator", "technician", "assembly supervisor",
        ],
        "skills":  [
            "production planning", "manufacturing", "quality control",
            "lean manufacturing", "six sigma", "kaizen", "5s", "tpm",
            "iso 9001", "sop", "gmp", "erp", "sap pp", "oee",
            "machine operation", "assembly", "process improvement",
        ],
        "summary": [
            "manufacturing operations", "production management",
            "quality assurance", "process optimisation", "plant operations",
        ],
        "org":     [],
    },
    "Sales": {
        "title":   [
            "sales manager", "sales director", "vp of sales",
            "account manager", "account executive", "sales executive",
            "business development manager", "business development",
            "bd manager", "key account manager", "national sales manager",
            "regional sales manager", "territory manager", "pre-sales",
            "inside sales", "field sales", "sales representative",
            "sales consultant", "relationship manager",
        ],
        "skills":  [
            "sales", "business development", "b2b", "b2c", "crm",
            "salesforce", "hubspot", "cold calling", "pipeline management",
            "revenue generation", "quota management", "negotiation",
            "client acquisition", "lead generation", "account management",
            "up-selling", "cross-selling", "tender", "proposal writing",
        ],
        "summary": [
            "sales growth", "revenue generation", "business development",
            "new business acquisition", "client relationship management",
        ],
        "org":     [],
    },
    "Shipping & Logistics": {
        "title":   [
            "logistics manager", "logistics executive", "warehouse manager",
            "warehouse supervisor", "supply chain manager",
            "freight manager", "shipping executive", "import export manager",
            "customs officer", "transport manager", "fleet manager",
            "distribution manager", "operations manager",
            "3pl manager", "fulfilment manager", "last mile",
        ],
        "skills":  [
            "logistics", "warehousing", "freight", "shipping", "customs",
            "import export", "incoterms", "supply chain", "erp",
            "sap", "wms", "inventory management", "demand planning",
            "3pl", "transport management", "cold chain", "last mile delivery",
        ],
        "summary": [
            "logistics operations", "supply chain management", "freight forwarding",
            "warehouse operations", "distribution management",
        ],
        "org":     ["dhl", "fedex", "ups", "maersk", "cma cgm", "ntn", "gdex",
                    "pos laju", "singpost"],
    },
}

INDUSTRY_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "Automotive": {
        "title":   ["automotive engineer", "vehicle engineer", "auto technician"],
        "skills":  ["automotive", "vehicle dynamics", "adas", "ev", "electric vehicle",
                    "can bus", "autosar", "engine calibration"],
        "summary": ["automotive", "vehicle", "automobile"],
        "org":     ["toyota", "honda", "bmw", "mercedes", "audi", "ford",
                    "hyundai", "kia", "volkswagen", "continental", "bosch",
                    "denso", "aisin", "magna", "delphi", "lear", "aptiv"],
    },
    "Banking & Finance": {
        "title":   ["banker", "relationship manager", "wealth manager",
                    "investment analyst", "fund manager", "credit analyst",
                    "risk analyst", "treasury analyst", "financial advisor"],
        "skills":  ["banking", "finance", "investment banking", "capital markets",
                    "equity", "fixed income", "derivatives", "fx", "forex",
                    "trade finance", "wealth management", "aml", "kyc",
                    "risk management", "credit risk", "market risk", "bloomberg"],
        "summary": ["banking", "financial services", "investment management",
                    "fintech", "wealth management"],
        "org":     ["dbs", "ocbc", "uob", "maybank", "cimb", "ubs", "jpmorgan",
                    "hsbc", "citibank", "standard chartered", "bnp paribas",
                    "morgan stanley", "goldman sachs", "blackrock",
                    "great eastern", "prudential", "aia", "manulife"],
    },
    "Building/Construction": {
        "title":   ["civil engineer", "structural engineer", "quantity surveyor",
                    "project manager", "site engineer", "architect",
                    "mep engineer", "construction manager"],
        "skills":  ["construction", "project management", "autocad", "revit",
                    "building information modelling", "bim", "quantity surveying",
                    "qs", "structural design", "foundation", "reinforced concrete",
                    "steel structure", "tender", "contract administration"],
        "summary": ["construction", "building", "infrastructure", "real estate development"],
        "org":     ["capitaland", "city developments", "far east", "keppel land",
                    "guocoland", "woh hup", "samsung c&t", "china construction",
                    "gamuda"],
    },
    "Chemicals": {
        "title":   ["chemical engineer", "process engineer", "r&d chemist",
                    "formulation chemist", "quality chemist"],
        "skills":  ["chemistry", "chemical engineering", "process engineering",
                    "hse", "sds", "reach", "polymer", "coating",
                    "formulation", "laboratory", "analytical chemistry",
                    "gc-ms", "hplc", "iso 14001"],
        "summary": ["chemical", "specialty chemicals", "petrochemical",
                    "industrial chemicals"],
        "org":     ["basf", "dow chemical", "dupont", "evonik", "arkema",
                    "shell chemicals", "exxonmobil chemical",
                    "lotte chemical", "sabic"],
    },
    "Education": {
        "title":   ["teacher", "lecturer", "professor", "educator",
                    "principal", "academic", "curriculum developer",
                    "instructional designer", "trainer", "tutor",
                    "education manager", "dean"],
        "skills":  ["teaching", "curriculum development", "e-learning",
                    "lesson planning", "pedagogy", "lms", "moodle",
                    "blended learning", "instructional design",
                    "classroom management", "student assessment"],
        "summary": ["education", "teaching", "academic", "learning",
                    "student development", "edtech"],
        "org":     ["moe", "ministry of education", "nus", "ntu", "smu",
                    "sim", "kaplan", "ite", "polytechnic", "republic polytechnic",
                    "temasek polytechnic", "ngee ann polytechnic"],
    },
    "Engineering - Aerospace": {
        "title":   ["aerospace engineer", "avionics engineer", "mro engineer",
                    "aircraft engineer", "flight engineer", "propulsion engineer"],
        "skills":  ["aerospace", "aviation", "avionics", "mro", "maintenance repair overhaul",
                    "aircraft structures", "propulsion", "do-178", "as9100",
                    "airworthiness", "faa", "easa", "caas"],
        "summary": ["aerospace", "aviation", "aircraft", "airline", "mro"],
        "org":     ["sia", "singapore airlines", "stse", "st engineering",
                    "rolls-royce", "pratt whitney", "safran", "airbus", "boeing",
                    "siaec", "haeco", "lufthansa technik"],
    },
    "Engineering - Precision": {
        "title":   ["precision engineer", "cnc machinist", "toolmaker",
                    "metrologist", "quality engineer", "process engineer"],
        "skills":  ["precision engineering", "cnc", "machining", "grinding",
                    "turning", "milling", "edm", "tooling", "jig fixture",
                    "cmm", "gd&t", "iso 9001", "anodising", "plating",
                    "stamping", "die casting"],
        "summary": ["precision engineering", "machining", "precision manufacturing"],
        "org":     ["hi-p", "sunningdale", "meiban", "beyonics"],
    },
    "Energy (Oil & Gas)": {
        "title":   ["oil gas engineer", "petroleum engineer", "drilling engineer",
                    "production engineer", "reservoir engineer", "process engineer",
                    "hse manager", "subsea engineer", "pipeline engineer"],
        "skills":  ["oil gas", "petroleum", "upstream", "downstream", "lng",
                    "refinery", "drilling", "well completion", "reservoir",
                    "process safety", "hse", "pipelines", "asset integrity",
                    "offshore", "fpso"],
        "summary": ["oil and gas", "energy sector", "petroleum", "upstream",
                    "lng", "refinery operations"],
        "org":     ["shell", "exxonmobil", "chevron", "bp", "total energies",
                    "sembcorp", "keppel offshore", "technip", "saipem", "schlumberger",
                    "halliburton", "baker hughes", "petroleum petronas"],
    },
    "Environment & Water": {
        "title":   ["environmental engineer", "water engineer",
                    "sustainability manager", "esg manager", "waste manager"],
        "skills":  ["environmental", "water treatment", "wastewater",
                    "sustainability", "esg", "carbon footprint", "iso 14001",
                    "environmental impact assessment", "eia", "green building",
                    "leed", "bca green mark"],
        "summary": ["environmental", "sustainability", "water management",
                    "green initiatives", "clean energy"],
        "org":     ["pub", "nea", "suez", "veolia", "sembcorp utilities",
                    "tuas power", "keppel seghers"],
    },
    "Healthcare": {
        "title":   ["nurse", "doctor", "physician", "surgeon", "pharmacist",
                    "radiographer", "physiotherapist", "occupational therapist",
                    "medical officer", "clinical research", "healthcare manager",
                    "hospital administrator", "health informatics"],
        "skills":  ["patient care", "clinical", "nursing", "ecg", "phlebotomy",
                    "medication management", "emr", "ehr", "healthcare",
                    "public health", "clinical trials", "gcp",
                    "infection control", "sdq", "jci accreditation"],
        "summary": ["healthcare", "clinical practice", "patient management",
                    "hospital administration", "medical care"],
        "org":     ["sgh", "ttsh", "nuh", "kk hospital", "cgh", "ah",
                    "raffles medical", "parkway pantai", "ihi", "mount elizabeth",
                    "kkh", "nsc", "nhcs", "ncc"],
    },
    "Hospitality & Tourism": {
        "title":   ["hotel manager", "front office manager", "f&b manager",
                    "housekeeping manager", "concierge", "revenue manager",
                    "event manager", "banquet manager", "guest relations",
                    "travel consultant", "tour guide"],
        "skills":  ["hotel management", "opera pms", "revenue management",
                    "food & beverage", "guest experience", "event management",
                    "hospitality", "tourism", "fohlio"],
        "summary": ["hospitality", "hotel operations", "tourism",
                    "guest services", "travel industry"],
        "org":     ["marriott", "hilton", "hyatt", "accor", "shangri-la",
                    "mandarin oriental", "four seasons", "intercontinental",
                    "singapore tourism board", "stb"],
    },
    "F&B": {
        "title":   ["chef", "sous chef", "kitchen manager", "f&b manager",
                    "restaurant manager", "food service manager",
                    "barista", "pastry chef", "baker"],
        "skills":  ["food and beverage", "f&b", "cooking", "haccp",
                    "food safety", "menu planning", "kitchen operations",
                    "restaurant management", "inventory control"],
        "summary": ["food and beverage", "restaurant", "food service",
                    "culinary", "catering"],
        "org":     ["mcdonalds", "kfc", "pizza hut", "breadtalk",
                    "old chang kee", "ya kun", "toast box", "starbucks",
                    "hawker", "catering"],
    },
    "FMCG": {
        "title":   ["brand manager", "trade marketing manager",
                    "product manager", "category manager", "key account manager"],
        "skills":  ["fmcg", "consumer goods", "trade marketing",
                    "category management", "shelf space", "planogram",
                    "shopper marketing", "neilsen", "iri", "kantar",
                    "brand management", "product development"],
        "summary": ["fmcg", "consumer goods", "fast moving consumer",
                    "brand portfolio", "retail channels"],
        "org":     ["unilever", "p&g", "nestle", "colgate", "johnson & johnson",
                    "heineken", "carlsberg", "tiger beer", "coca cola",
                    "pepsico", "mondelez", "kelloggs", "fonterra"],
    },
    "Infocomm": {
        "title":   ["ict consultant", "it project manager", "it manager",
                    "systems integrator", "solution architect",
                    "managed services", "it infrastructure"],
        "skills":  ["ict", "it services", "managed services", "system integration",
                    "it infrastructure", "cloud services", "it consulting",
                    "digital transformation"],
        "summary": ["infocomm", "it services company", "systems integrator",
                    "ict solutions", "technology services provider"],
        "org":     ["singtel", "starhub", "m1", "singtel optus", "NCS",
                    "cognizant", "infosys", "tcs", "accenture", "ibm",
                    "hewlett packard", "dell technologies", "fujitsu"],
    },
    "Medical Technology": {
        "title":   ["medtech engineer", "medical device specialist",
                    "field service engineer", "clinical specialist",
                    "regulatory affairs", "qra", "medical affairs"],
        "skills":  ["medical device", "fda", "mdr", "iso 13485",
                    "510k", "ce marking", "clinical evaluation",
                    "biocompatibility", "sterilisation", "medical imaging",
                    "diagnostic equipment", "surgical instruments"],
        "summary": ["medical technology", "medical device", "medtech",
                    "healthcare technology", "diagnostic"],
        "org":     ["siemens healthineers", "ge healthcare", "philips healthcare",
                    "medtronic", "becton dickinson", "bd", "stryker",
                    "zimmer biomet", "abbott", "hologic",
                    "roper technologies", "natus medical"],
    },
    "Marine & Offshore": {
        "title":   ["marine engineer", "naval architect", "offshore engineer",
                    "subsea engineer", "vessel superintendent",
                    "marine surveyor", "shipyard project manager"],
        "skills":  ["marine engineering", "naval architecture", "offshore",
                    "shipbuilding", "drydock", "hull", "propulsion systems",
                    "imo", "solas", "marpol", "classing society",
                    "dnv", "lloyd's register"],
        "summary": ["marine", "offshore", "shipbuilding", "maritime industry"],
        "org":     ["sembcorp marine", "keppel offshore", "seatrium",
                    "pan-united", "asm pacific", "damen shipyards",
                    "stx offshore", "cosco"],
    },
    "Retail": {
        "title":   ["retail manager", "store manager", "visual merchandiser",
                    "merchandiser", "buyer", "e-commerce manager",
                    "omnichannel manager", "loss prevention"],
        "skills":  ["retail", "merchandising", "visual merchandising",
                    "point of sale", "pos", "inventory management",
                    "e-commerce", "shopify", "magento", "woocommerce",
                    "customer service", "stock management", "replenishment"],
        "summary": ["retail", "consumer retail", "fashion retail",
                    "e-commerce", "omnichannel retail"],
        "org":     ["courts", "harvey norman", "challenger", "best denki",
                    "watsons", "guardian", "cold storage", "ntuc fairprice",
                    "robinsons", "zara", "h&m", "uniqlo", "charles & keith",
                    "pandora", "sephora", "luxasia", "tangs", "takashimaya"],
    },
    "Pharmaceutical/Biotech": {
        "title":   ["pharmacist", "pharmaceutical", "medical affairs",
                    "clinical research associate", "cra", "regulatory affairs",
                    "drug development", "formulation scientist",
                    "biotech scientist", "qp", "qualified person"],
        "skills":  ["gmp", "gcp", "glp", "fda", "hsa", "ema",
                    "clinical trials", "drug development", "pharmacovigilance",
                    "formulation", "analytical development", "lims",
                    "validation", "regulatory submission", "dossier",
                    "biochemistry", "cell culture", "chromatography"],
        "summary": ["pharmaceutical", "pharma", "biotech", "drug development",
                    "clinical research", "biopharmaceutical"],
        "org":     ["pfizer", "novartis", "roche", "astrazeneca", "gsk",
                    "johnson & johnson", "sanofi", "merck", "abbvie",
                    "agilent", "thermo fisher", "msd", "lilly", "amgen",
                    "hsa", "biopolis", "imcb"],
    },
    "Robotics": {
        "title":   ["robotics engineer", "automation engineer",
                    "rpa developer", "robot programmer", "cobot specialist"],
        "skills":  ["robotics", "automation", "rpa", "uipath", "automation anywhere",
                    "blue prism", "robot programming", "ros", "plc programming",
                    "industrial automation", "fanuc", "kuka", "abb robotics",
                    "computer vision", "machine vision", "cobots"],
        "summary": ["robotics", "industrial automation", "rpa", "autonomous systems"],
        "org":     ["universal robots", "fanuc", "abb", "kuka", "yaskawa",
                    "cognex", "keyence", "omron"],
    },
    "Semiconductor": {
        "title":   ["process engineer", "yield engineer", "device engineer",
                    "ic design engineer", "layout engineer", "fab engineer",
                    "equipment engineer", "test engineer"],
        "skills":  ["semiconductor", "wafer fabrication", "photolithography",
                    "cvd", "pvd", "etch", "cmp", "metrology",
                    "ic design", "vlsi", "spice", "verilog", "vhdl",
                    "analog design", "digital design", "soc", "yield analysis",
                    "defect analysis", "sem"],
        "summary": ["semiconductor", "wafer fab", "chip design",
                    "integrated circuit", "fab operations"],
        "org":     ["micron", "tsmc", "qualcomm", "intel", "ams", "ams-osram",
                    "infineon", "st microelectronics", "nxp", "renesas",
                    "siltronic", "globalfoundries", "umc"],
    },
    "Supply Chain Mgt & Logistics": {
        "title":   ["supply chain manager", "demand planner", "s&op manager",
                    "materials manager", "logistics manager",
                    "freight operations", "3pl manager"],
        "skills":  ["supply chain management", "demand planning", "s&op",
                    "inventory optimisation", "erp", "sap", "oracle scm",
                    "supplier collaboration", "integrated business planning",
                    "inbound logistics", "outbound logistics",
                    "freight management", "customs clearance"],
        "summary": ["supply chain", "logistics management",
                    "end-to-end supply chain", "supply chain transformation"],
        "org":     ["toll", "panalpina", "kn", "kuehne nagel", "expeditors",
                    "dsv", "geodis", "ceva", "db schenker", "bollore"],
    },
    "Telco": {
        "title":   ["network engineer", "rf engineer", "telecom engineer",
                    "5g engineer", "solutions architect", "presales engineer"],
        "skills":  ["telecommunications", "5g", "4g", "lte", "rf",
                    "core network", "ran", "voip", "sip", "mpls",
                    "sdwan", "sd-wan", "fiber optics", "wimax",
                    "network planning", "huawei", "ericsson", "nokia"],
        "summary": ["telecommunications", "telco", "5g", "mobile network",
                    "broadband services"],
        "org":     ["singtel", "starhub", "m1", "myrepublic", "viewqwest",
                    "ericsson", "huawei", "nokia", "samsung networks",
                    "celcom", "maxis", "digi", "u mobile"],
    },
    "Trading": {
        "title":   ["trader", "commodity trader", "trading analyst",
                    "import export executive", "business development manager",
                    "wholesale manager", "distribution manager"],
        "skills":  ["trading", "commodities", "import export", "distribution",
                    "wholesale", "trade finance", "letters of credit",
                    "incoterms", "commodity", "market making"],
        "summary": ["trading", "commodity trading", "import and export",
                    "wholesale distribution", "trading operations"],
        "org":     ["noble group", "trafigura", "vitol", "glencore",
                    "louis dreyfus", "cargill", "olam", "wilmar"],
    },
}


# =============================================================================
# 🏗️ FLASK APP
# =============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fairy-codemother-classify-2026")


# =============================================================================
# 🔌 DATABASE HELPERS
# =============================================================================

def get_db() -> sqlite3.Connection:
    """
    Get (or create) a per-request DB connection stored in Flask's g object.
    Thread-safe — each request gets its own connection, like its own dressing room! 🎬
    """
    if "db" not in g:
        if not os.path.exists(DATABASE_PATH):
            abort(500, description=f"Database not found: {DATABASE_PATH}")
        try:
            g.db = sqlite3.connect(DATABASE_PATH, timeout=15)
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA journal_mode=WAL")
            g.db.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as e:
            logger.error(f"❌ DB connection failed: {e}")
            abort(500, description=str(e))
    return g.db


@app.teardown_appcontext
def close_db(exc):
    """Close the DB connection when the request context ends."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def ensure_classification_table():
    """
    🏗️ Create the candidate_classifications table if it doesn't exist.

    This table stores both the AUTO-generated classification and the
    HUMAN-reviewed/corrected version. We keep both so we can:
      - Show the auto suggestion as a starting point
      - Track how often the auto-classifier needs correction (for future
        ML retraining when we have enough data!)
      - Preserve reviewer's identity and timestamp for auditability

    Schema:
        candidate_id        — FK to ner_documents.candidate_id
        auto_function       — Classifier's top Function pick
        auto_function_conf  — Confidence score 0.0–1.0
        auto_industry       — Classifier's top Industry pick
        auto_industry_conf  — Confidence score 0.0–1.0
        auto_reasoning      — JSON of top-3 scores per category
        reviewed_function   — Human-confirmed (or corrected) Function
        reviewed_industry   — Human-confirmed (or corrected) Industry
        is_function_correct — 1 if auto=human, 0 if corrected, NULL if not yet reviewed
        is_industry_correct — same for industry
        reviewed_by         — Reviewer's name / username
        reviewed_at         — ISO timestamp of last review
        classified_at       — ISO timestamp of last auto-classify run
    """
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS candidate_classifications (
            candidate_id        INTEGER PRIMARY KEY,
            auto_function       TEXT,
            auto_function_conf  REAL DEFAULT 0.0,
            auto_industry       TEXT,
            auto_industry_conf  REAL DEFAULT 0.0,
            auto_reasoning      TEXT,
            reviewed_function   TEXT,
            reviewed_industry   TEXT,
            is_function_correct INTEGER,
            is_industry_correct INTEGER,
            reviewed_by         TEXT,
            reviewed_at         TEXT,
            classified_at       TEXT
        )
    """)
    db.commit()


# =============================================================================
# 🧠 KEYWORD CLASSIFIER
# =============================================================================

class KeywordClassifier:
    """
    ✨ The Keyword Classifier — the brain behind the auto-grouping! 🧠

    Like a seasoned HR recruiter who scans a resume in 30 seconds and
    knows EXACTLY which department and sector this candidate belongs to —
    except she works 24/7 and never needs coffee! ☕💅

    Algorithm (for each candidate):
    1. Pull all their NER annotations from the DB
    2. Build weighted text signals:
         JOB_TITLE   → weight 5  (most reliable signal!)
         SKILL       → weight 2
         SOFT_SKILL  → weight 1
         ORGANIZATION→ weight 2
         SUMMARY_TEXT→ weight 1.5
         JOB_DESC    → weight 0.8
         FUNCTION    → weight 8  (annotator already labeled it!)
         INDUSTRY    → weight 8  (annotator already labeled it!)
    3. For each label in FUNCTION_LIST / INDUSTRY_LIST:
         Score = Σ (keyword_weight × signal_weight) for each match
    4. Normalize top score → confidence 0–1
    5. Return top pick + confidence + top-3 alternatives for reasoning display

    Direct annotation mapping:
    If the annotator labeled FUNCTION/INDUSTRY text, we FIRST try to map
    it directly to our taxonomy. If it matches (fuzzy), we skip keyword
    scoring entirely and use confidence=0.95.
    """

    # Entity weights: how much do we trust each entity type as a signal?
    ENTITY_WEIGHTS: Dict[str, float] = {
        "FUNCTION":          8.0,   # Annotator explicitly labeled this!
        "INDUSTRY":          8.0,   # Same — most reliable signal
        "JOB_TITLE":         5.0,   # Job title is the #2 strongest signal
        "ORGANIZATION":      2.0,   # Company name hints at industry
        "SKILL":             2.0,   # Skills reveal function
        "SOFT_SKILL":        1.0,
        "CERTIFICATION":     1.5,   # Certs hint at industry
        "SUMMARY_TEXT":      1.5,
        "JOB_DESCRIPTION":   0.8,
        "PROJECT_TITLE":     0.8,
    }

    def __init__(self):
        # Pre-compile keyword → label lookup tables for speed
        # Maps lowercase keyword → {label: keyword_tier_weight}
        self._func_lookup = self._build_lookup(FUNCTION_KEYWORDS)
        self._ind_lookup  = self._build_lookup(INDUSTRY_KEYWORDS)

    @staticmethod
    def _build_lookup(
        keyword_map: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a flat keyword → {label: weight} lookup.

        Keyword tier weights within the keyword map:
            "title" keys   → 3.0x  (JOB_TITLE keywords are most discriminative)
            "skills" keys  → 1.5x
            "org" keys     → 2.0x  (matching a known company is strong!)
            "summary" keys → 1.0x
        """
        tier_weights = {"title": 3.0, "skills": 1.5, "org": 2.0, "summary": 1.0}
        lookup: Dict[str, Dict[str, float]] = defaultdict(dict)

        for label, tiers in keyword_map.items():
            for tier, keywords in tiers.items():
                tier_w = tier_weights.get(tier, 1.0)
                for kw in keywords:
                    kw_lower = kw.lower()
                    # Keep the max weight if a keyword appears in multiple tiers
                    existing = lookup[kw_lower].get(label, 0.0)
                    lookup[kw_lower][label] = max(existing, tier_w)

        return dict(lookup)

    def _extract_signals(
        self, annotations: List[Dict]
    ) -> List[Tuple[str, float]]:
        """
        Convert annotation list → [(lowercase_text, signal_weight)] pairs.
        Each annotation contributes its text × entity_weight.
        """
        signals = []
        for ann in annotations:
            etype = ann.get("entity_type", "")
            text  = (ann.get("text") or "").strip().lower()
            if not text or ann.get("layer", 0) != 0:
                continue
            w = self.ENTITY_WEIGHTS.get(etype, 0.5)
            signals.append((text, w))
        return signals

    def _try_direct_map(
        self,
        signals: List[Tuple[str, float]],
        target_list: List[str],
        entity_type: str
    ) -> Optional[Tuple[str, float]]:
        """
        Try to directly map an annotated FUNCTION or INDUSTRY value to the
        standard taxonomy. Like checking if the label is already spelled
        correctly on the costume — saves a lot of sewing! 🧵

        Matching strategy (in order of confidence):
          1. Exact match (case-insensitive) → confidence 0.97
          2. One label contains the other → confidence 0.90
          3. Token overlap ≥ 60% → confidence 0.80

        Returns (matched_label, confidence) or None if no match.
        """
        # Only look at FUNCTION or INDUSTRY annotations
        func_signals = [
            text for text, w in signals
            # A rough check: signal entity type isn't available here;
            # we filter by checking if signal matches the keyword map label space
        ]
        _ = func_signals  # unused; we iterate all signals below

        # Pull only the raw texts that came from the target entity type
        # (We work from the full annotation dicts for direct mapping)
        return None  # Implemented in classify() with full ann dicts

    def _score_labels(
        self,
        signals: List[Tuple[str, float]],
        lookup: Dict[str, Dict[str, float]],
        label_list: List[str],
    ) -> Dict[str, float]:
        """
        Score every label in label_list against the text signals.

        For each signal (text, signal_weight):
          For each keyword in the lookup that is a SUBSTRING of signal_text:
            score[label] += keyword_weight × signal_weight

        Returns dict of {label: raw_score}.
        """
        scores: Dict[str, float] = {label: 0.0 for label in label_list}

        for signal_text, signal_w in signals:
            for keyword, label_weights in lookup.items():
                # Substring match — "machine learning" contains "machine"
                if keyword in signal_text or signal_text in keyword:
                    for label, kw_w in label_weights.items():
                        if label in scores:
                            scores[label] += kw_w * signal_w

        return scores

    def _normalize_scores(
        self, scores: Dict[str, float]
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Normalize raw scores to a top pick + confidence.

        Confidence = top_score / (top_score + second_score)
        This gives a relative confidence — if one label dominates, confidence
        is high. If two labels are neck-and-neck, confidence is ~0.5.

        Returns:
            (top_label, confidence_0_to_1, top3_alternatives)
        """
        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_label, top_score = ranked[0]

        if top_score == 0:
            # No signals matched at all — default to "Others"
            return "Others", 0.0, []

        # Relative confidence using softmax-style ratio
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total = top_score + second_score + 1e-9
        confidence = min(top_score / total, 0.97)   # cap at 97%

        # Also produce a 0-1 absolute confidence based on CONFIDENCE_THRESHOLD
        abs_conf = min(top_score / max(top_score * 3, CONFIDENCE_THRESHOLD * 100), 1.0)
        confidence = max(confidence, abs_conf) if top_score > 0 else 0.0

        # Top-3 for reasoning display (skip 0-score entries)
        top3 = [(l, s) for l, s in ranked[:3] if s > 0]

        return top_label, round(confidence, 3), top3

    def classify(
        self,
        annotations: List[Dict],
        candidate_id: int
    ) -> Dict[str, Any]:
        """
        Main entry point — classify a candidate's Function AND Industry.

        Args:
            annotations: List of annotation dicts from ner_annotations table.
            candidate_id: Used for logging only.

        Returns:
            {
                "function":          "IT",
                "function_conf":     0.82,
                "function_top3":     [("IT", 0.82), ("Engineering", 0.11)...],
                "industry":          "Banking & Finance",
                "industry_conf":     0.74,
                "industry_top3":     [("Banking & Finance", 0.74)...],
                "direct_function":   True/False,   # Was it from FUNCTION annotation?
                "direct_industry":   True/False,
            }
        """
        # ── Step 1: Try direct mapping from FUNCTION annotation ────────────
        direct_function: Optional[str] = None
        direct_industry: Optional[str] = None

        for ann in annotations:
            if ann.get("layer", 0) != 0:
                continue
            etype = ann.get("entity_type", "")
            text  = (ann.get("text") or "").strip()
            text_l = text.lower()

            if etype == "FUNCTION" and not direct_function:
                direct_function = self._map_to_taxonomy(text_l, FUNCTION_LIST)
            if etype == "INDUSTRY" and not direct_industry:
                direct_industry = self._map_to_taxonomy(text_l, INDUSTRY_LIST)

        # ── Step 2: Build text signals for keyword scoring ─────────────────
        signals = self._extract_signals(annotations)

        # ── Step 3: Score Function labels ──────────────────────────────────
        if direct_function:
            func_label    = direct_function
            func_conf     = 0.92
            func_top3     = [(direct_function, 0.92)]
            is_direct_fn  = True
        else:
            func_scores   = self._score_labels(signals, self._func_lookup, FUNCTION_LIST)
            func_label, func_conf, func_top3 = self._normalize_scores(func_scores)
            is_direct_fn  = False

        # ── Step 4: Score Industry labels ──────────────────────────────────
        if direct_industry:
            ind_label     = direct_industry
            ind_conf      = 0.92
            ind_top3      = [(direct_industry, 0.92)]
            is_direct_ind = True
        else:
            ind_scores    = self._score_labels(signals, self._ind_lookup, INDUSTRY_LIST)
            ind_label, ind_conf, ind_top3 = self._normalize_scores(ind_scores)
            is_direct_ind = False

        logger.debug(
            f"Candidate {candidate_id}: fn={func_label}({func_conf:.0%})"
            f" ind={ind_label}({ind_conf:.0%})"
        )

        return {
            "function":        func_label,
            "function_conf":   func_conf,
            "function_top3":   func_top3,
            "industry":        ind_label,
            "industry_conf":   ind_conf,
            "industry_top3":   ind_top3,
            "direct_function": is_direct_fn,
            "direct_industry": is_direct_ind,
        }

    @staticmethod
    def _map_to_taxonomy(text: str, taxonomy: List[str]) -> Optional[str]:
        """
        Fuzzy-map an annotated text value to the closest taxonomy label.

        Matching tiers (descending preference):
          1. Exact match (case-insensitive)
          2. Taxonomy label fully contained in text
          3. Text fully contained in taxonomy label
          4. Token overlap ≥ 50%

        Returns matched label string, or None if no reasonable match.
        """
        text_tokens = set(text.lower().split())

        for label in taxonomy:
            label_l = label.lower()
            if text == label_l:
                return label  # Perfect match
            if label_l in text or text in label_l:
                return label  # Substring match
            # Token overlap
            label_tokens = set(label_l.split())
            overlap = text_tokens & label_tokens
            if overlap and len(overlap) / max(len(label_tokens), 1) >= 0.5:
                return label

        return None


# Module-level singleton — built once at startup
classifier = KeywordClassifier()


# =============================================================================
# 🔬 ANNOTATION ACCURACY ANALYZER
# =============================================================================

# Required entity types for a "complete" candidate profile, with display info
REQUIRED_ENTITIES = {
    "PERSON_NAME":    {"label": "Name",            "weight": 2},
    "EMAIL":          {"label": "Email",            "weight": 2},
    "PHONE":          {"label": "Phone",            "weight": 1},
    "LOCATION":       {"label": "Current Location", "weight": 1},
    "JOB_TITLE":      {"label": "Current Title",    "weight": 2},
    "ORGANIZATION":   {"label": "Current Company",  "weight": 2},
    "SKILL":          {"label": "Skills (tags)",    "weight": 2},
    "DEGREE":         {"label": "Degree",           "weight": 1},
    "INSTITUTION":    {"label": "Institution",      "weight": 1},
    "SUMMARY_TEXT":   {"label": "Summary",          "weight": 1},
}

# Entity type → hex color (matches annotation_tool palette)
ENTITY_COLORS: Dict[str, str] = {
    "PERSON_NAME":          "#7ee8fa",
    "EMAIL":                "#56d364",
    "PHONE":                "#e3b341",
    "DATE_OF_BIRTH":        "#f0883e",
    "LOCATION":             "#a371f7",
    "NATIONALITY":          "#bc8cff",
    "NRIC_ID":              "#da3633",
    "GENDER":               "#f778ba",
    "MARITAL_STATUS":       "#a5d6ff",
    "EXPECTED_LOCATION":    "#58a6ff",
    "JOB_TITLE":            "#7ee8fa",
    "ORGANIZATION":         "#eeb8ff",
    "WORK_DATE":            "#f0883e",
    "JOB_DESCRIPTION":      "#8b949e",
    "METRIC":               "#f778ba",
    "FUNCTION":             "#39d353",
    "INDUSTRY":             "#f1a340",
    "PROJECT_TITLE":        "#d29922",
    "PROJECT_DESCRIPTION":  "#b08800",
    "DEGREE":               "#56d364",
    "INSTITUTION":          "#7ee8fa",
    "FIELD_OF_STUDY":       "#a371f7",
    "EDU_DATE":             "#f0883e",
    "GPA":                  "#e3b341",
    "SKILL":                "#eeb8ff",
    "SKILL_CATEGORY":       "#8b949e",
    "SOFT_SKILL":           "#bc8cff",
    "CERTIFICATION":        "#56d364",
    "CERT_ISSUER":          "#7ee8fa",
    "CERT_DATE":            "#f0883e",
    "LANGUAGE_SKILL":       "#a371f7",
    "PROFICIENCY":          "#8b949e",
    "SECTION_HEADER":       "#6e7681",
    "SUMMARY_TEXT":         "#8b949e",
}


def compute_accuracy(annotations: List[Dict]) -> Dict[str, Any]:
    """
    🔬 Compute annotation accuracy metrics for a single candidate.

    Accuracy is measured as a WEIGHTED score based on which required
    entity types are present vs missing. Think of it like a beauty
    scorecard at a pageant — every category matters, but some score more! 👑

    Args:
        annotations: List of annotation dicts from ner_annotations table.

    Returns:
        {
            "score":         85,           # 0–100 integer
            "grade":         "B",          # A/B/C/D/F
            "present":       {"PERSON_NAME": 1, "SKILL": 7, ...},
            "missing":       ["DEGREE", "INSTITUTION"],
            "warnings":      ["Possible duplicate PHONE spans", ...],
            "entity_counts": {"PERSON_NAME": 1, "SKILL": 7, ...}
                             (all entities, not just required)
        }
    """
    # Count annotations per entity type (layer 0 only)
    entity_counts: Dict[str, int] = defaultdict(int)
    for ann in annotations:
        if ann.get("layer", 0) == 0:
            entity_counts[ann["entity_type"]] += 1

    # Score against required entities
    total_weight = sum(v["weight"] for v in REQUIRED_ENTITIES.values())
    earned_weight = 0.0
    present: Dict[str, int] = {}
    missing: List[str] = []

    for etype, info in REQUIRED_ENTITIES.items():
        count = entity_counts.get(etype, 0)
        if count > 0:
            present[etype] = count
            earned_weight += info["weight"]
        else:
            missing.append(etype)

    score = int(round(earned_weight / total_weight * 100)) if total_weight > 0 else 0

    # Grade thresholds
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 55:
        grade = "C"
    elif score >= 35:
        grade = "D"
    else:
        grade = "F"

    # Warnings — common annotation quality issues
    warnings: List[str] = []

    # Warning: Too many PERSON_NAME annotations (possible mis-annotation)
    if entity_counts.get("PERSON_NAME", 0) > 2:
        warnings.append(f"Unusually high PERSON_NAME count ({entity_counts['PERSON_NAME']}) — check for mis-labeled names")

    # Warning: No SKILL labels — common miss for experienced candidates
    if entity_counts.get("SKILL", 0) == 0 and entity_counts.get("JOB_TITLE", 0) > 0:
        warnings.append("No SKILL annotations — did you label the skills section?")

    # Warning: JOB_DESCRIPTION but no JOB_TITLE — possibly incomplete
    if entity_counts.get("JOB_DESCRIPTION", 0) > 0 and entity_counts.get("JOB_TITLE", 0) == 0:
        warnings.append("JOB_DESCRIPTION found but no JOB_TITLE — please label the job title")

    # Warning: No WORK_DATE on a candidate who has JOB_TITLE
    if entity_counts.get("JOB_TITLE", 0) > 0 and entity_counts.get("WORK_DATE", 0) == 0:
        warnings.append("JOB_TITLE found but no WORK_DATE — please label employment date ranges")

    # Warning: Many more ORGANIZATION than JOB_TITLE (unbalanced work experience)
    orgs = entity_counts.get("ORGANIZATION", 0)
    titles = entity_counts.get("JOB_TITLE", 0)
    if orgs > 0 and titles > 0 and abs(orgs - titles) > 2:
        warnings.append(f"Imbalanced work experience: {orgs} ORGANIZATION vs {titles} JOB_TITLE spans")

    return {
        "score":         score,
        "grade":         grade,
        "present":       dict(present),
        "missing":       missing,
        "warnings":      warnings,
        "entity_counts": dict(entity_counts),
    }


# =============================================================================
# 📦 DATA ACCESS FUNCTIONS
# =============================================================================

def get_candidates_list(
    page: int = 1,
    search: str = "",
    status_filter: str = "",
    function_filter: str = "",
    industry_filter: str = "",
    review_filter: str = "",
    sort_by: str = "candidate_id",
    sort_order: str = "DESC",
) -> Tuple[List[Dict], int]:
    """
    Paginated list of candidates with their annotation status,
    classification results, and accuracy scores.

    Joins ner_documents with candidate_classifications so we can show
    the Function + Industry badges on every row of the list.

    Args:
        page:            Current page number (1-indexed).
        search:          Text to search in name / email / candidate_id.
        status_filter:   'completed', 'in_progress', 'pending', or ''.
        function_filter: Filter by exact Function label, or ''.
        industry_filter: Filter by exact Industry label, or ''.
        review_filter:   'reviewed', 'unreviewed', or ''.
        sort_by:         Column name (whitelisted).
        sort_order:      'ASC' or 'DESC'.

    Returns:
        (list_of_candidate_dicts, total_count)
    """
    db = get_db()
    ensure_classification_table()

    # ── Whitelist sortable columns to prevent SQL injection ────────────
    sort_map = {
        "candidate_id":   "nd.candidate_id",
        "name":           "se.name",
        "status":         "nd.status",
        "score":          "acc_score",
        "function":       "COALESCE(cc.reviewed_function, cc.auto_function)",
        "industry":       "COALESCE(cc.reviewed_industry, cc.auto_industry)",
        "updated_at":     "nd.updated_at",
    }
    sort_col = sort_map.get(sort_by, "nd.candidate_id")
    sort_dir = "DESC" if sort_order.upper() == "DESC" else "ASC"

    # ── Build WHERE clause ─────────────────────────────────────────────
    conditions = ["nd.status IS NOT NULL"]
    params: List[Any] = []

    if search:
        conditions.append(
            "(se.name LIKE ? OR se.email LIKE ? OR CAST(nd.candidate_id AS TEXT) LIKE ?)"
        )
        like = f"%{search}%"
        params.extend([like, like, like])

    if status_filter:
        conditions.append("nd.status = ?")
        params.append(status_filter)

    if function_filter:
        conditions.append(
            "(cc.reviewed_function = ? OR cc.auto_function = ?)"
        )
        params.extend([function_filter, function_filter])

    if industry_filter:
        conditions.append(
            "(cc.reviewed_industry = ? OR cc.auto_industry = ?)"
        )
        params.extend([industry_filter, industry_filter])

    if review_filter == "reviewed":
        conditions.append("cc.reviewed_by IS NOT NULL")
    elif review_filter == "unreviewed":
        conditions.append("(cc.reviewed_by IS NULL OR cc.reviewed_at IS NULL)")

    where = "WHERE " + " AND ".join(conditions)

    # ── Count total for pagination ─────────────────────────────────────
    count_sql = f"""
        SELECT COUNT(*) as total
        FROM ner_documents nd
        LEFT JOIN structured_extractions se ON nd.candidate_id = se.candidate_id
        LEFT JOIN candidate_classifications cc ON nd.candidate_id = cc.candidate_id
        {where}
    """
    total = db.execute(count_sql, params).fetchone()["total"]

    # ── Fetch page data ────────────────────────────────────────────────
    offset = (page - 1) * PER_PAGE
    data_sql = f"""
        SELECT
            nd.candidate_id,
            nd.status,
            nd.annotator,
            nd.updated_at,
            se.name,
            se.email,
            se.extraction_status,
            cc.auto_function,
            cc.auto_function_conf,
            cc.auto_industry,
            cc.auto_industry_conf,
            cc.reviewed_function,
            cc.reviewed_industry,
            cc.reviewed_by,
            cc.reviewed_at,
            cc.is_function_correct,
            cc.is_industry_correct,
            -- ⚡ Pre-aggregated JOIN instead of correlated subquery
            COALESCE(acc.entity_count, 0) * 10 as acc_score
        FROM ner_documents nd
        LEFT JOIN structured_extractions se ON nd.candidate_id = se.candidate_id
        LEFT JOIN candidate_classifications cc ON nd.candidate_id = cc.candidate_id
        LEFT JOIN (
            SELECT doc_id, COUNT(DISTINCT entity_type) as entity_count
            FROM ner_annotations
            WHERE layer = 0
              AND entity_type IN (
                'PERSON_NAME','EMAIL','PHONE','LOCATION',
                'JOB_TITLE','ORGANIZATION','SKILL','DEGREE',
                'INSTITUTION','SUMMARY_TEXT'
              )
            GROUP BY doc_id
        ) acc ON acc.doc_id = nd.doc_id
        {where}
        ORDER BY {sort_col} {sort_dir}
        LIMIT ? OFFSET ?
    """
    params.extend([PER_PAGE, offset])
    rows = db.execute(data_sql, params).fetchall()
    return [dict(row) for row in rows], total


def get_candidate_detail(candidate_id: int) -> Optional[Dict]:
    """
    Retrieve full detail for one candidate: document, annotations, classification,
    and structured extraction fallback data.
    """
    db = get_db()
    ensure_classification_table()

    # ── Document row ───────────────────────────────────────────────────
    doc = db.execute("""
        SELECT nd.*, se.name, se.email, se.phone, se.extraction_status
        FROM ner_documents nd
        LEFT JOIN structured_extractions se ON nd.candidate_id = se.candidate_id
        WHERE nd.candidate_id = ?
        ORDER BY nd.created_at DESC
        LIMIT 1
    """, (candidate_id,)).fetchone()

    if not doc:
        return None

    doc_dict = dict(doc)
    doc_id = doc_dict["doc_id"]

    # ── All annotations for this document ──────────────────────────────
    anns = db.execute("""
        SELECT entity_type, char_start, char_end, text, layer,
               confidence, annotator, updated_at
        FROM ner_annotations
        WHERE doc_id = ?
        ORDER BY char_start ASC
    """, (doc_id,)).fetchall()
    annotations = [dict(a) for a in anns]

    # ── Raw resume text ────────────────────────────────────────────────
    raw_row = db.execute("""
        SELECT raw_text FROM raw_extractions
        WHERE candidate_id = ?
        ORDER BY extraction_timestamp DESC
        LIMIT 1
    """, (candidate_id,)).fetchone()
    raw_text = raw_row["raw_text"] if raw_row else doc_dict.get("raw_text", "")

    # ── Classification row ─────────────────────────────────────────────
    cls_row = db.execute("""
        SELECT * FROM candidate_classifications WHERE candidate_id = ?
    """, (candidate_id,)).fetchone()
    classification = dict(cls_row) if cls_row else {}

    # ── Accuracy metrics ───────────────────────────────────────────────
    accuracy = compute_accuracy(annotations)

    return {
        "doc":            doc_dict,
        "annotations":    annotations,
        "raw_text":       raw_text or "",
        "classification": classification,
        "accuracy":       accuracy,
    }


def get_annotations_for_candidate(candidate_id: int) -> List[Dict]:
    """
    Fetch all layer-0 annotations for a candidate (by candidate_id).
    Used by the auto-classifier.
    """
    db = get_db()
    doc = db.execute("""
        SELECT doc_id FROM ner_documents
        WHERE candidate_id = ?
        ORDER BY created_at DESC LIMIT 1
    """, (candidate_id,)).fetchone()
    if not doc:
        return []

    rows = db.execute("""
        SELECT entity_type, text, layer, char_start, char_end
        FROM ner_annotations
        WHERE doc_id = ? AND layer = 0
    """, (doc["doc_id"],)).fetchall()
    return [dict(r) for r in rows]


def save_classification(
    candidate_id: int,
    reviewed_function: str,
    reviewed_industry: str,
    reviewed_by: str,
    auto_data: Optional[Dict] = None
) -> bool:
    """
    💾 Upsert a classification review into candidate_classifications.

    Compares the human review against the auto result to track accuracy.
    Every correction is a GIFT — it tells us where the classifier needs work! 🎁

    Args:
        candidate_id:       The candidate being reviewed.
        reviewed_function:  Human-selected Function label.
        reviewed_industry:  Human-selected Industry label.
        reviewed_by:        Reviewer's name / username.
        auto_data:          The auto-classification dict (for logging is_correct).

    Returns:
        True if saved successfully, False on DB error.
    """
    db = get_db()
    ensure_classification_table()
    now = datetime.datetime.now().isoformat()

    # Determine correctness flags
    is_fn_correct  = None
    is_ind_correct = None
    if auto_data:
        auto_fn  = auto_data.get("auto_function")
        auto_ind = auto_data.get("auto_industry")
        if auto_fn:
            is_fn_correct  = 1 if reviewed_function  == auto_fn  else 0
        if auto_ind:
            is_ind_correct = 1 if reviewed_industry == auto_ind else 0

    try:
        db.execute("""
            INSERT INTO candidate_classifications
                (candidate_id, reviewed_function, reviewed_industry,
                 is_function_correct, is_industry_correct,
                 reviewed_by, reviewed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                reviewed_function   = excluded.reviewed_function,
                reviewed_industry   = excluded.reviewed_industry,
                is_function_correct = excluded.is_function_correct,
                is_industry_correct = excluded.is_industry_correct,
                reviewed_by         = excluded.reviewed_by,
                reviewed_at         = excluded.reviewed_at
        """, (
            candidate_id, reviewed_function, reviewed_industry,
            is_fn_correct, is_ind_correct,
            reviewed_by, now
        ))
        db.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"❌ save_classification failed for {candidate_id}: {e}")
        db.rollback()
        return False


def auto_classify_candidate(candidate_id: int) -> Optional[Dict]:
    """
    Run the classifier on one candidate and persist the result.

    Returns the classification dict, or None if the candidate has no doc.
    """
    db = get_db()
    ensure_classification_table()
    anns = get_annotations_for_candidate(candidate_id)

    if not anns and not db.execute(
        "SELECT 1 FROM ner_documents WHERE candidate_id = ?", (candidate_id,)
    ).fetchone():
        return None

    result = classifier.classify(anns, candidate_id)
    now = datetime.datetime.now().isoformat()

    reasoning = json.dumps({
        "function_top3": result["function_top3"],
        "industry_top3": result["industry_top3"],
    })

    try:
        db.execute("""
            INSERT INTO candidate_classifications
                (candidate_id, auto_function, auto_function_conf,
                 auto_industry, auto_industry_conf, auto_reasoning, classified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                auto_function      = excluded.auto_function,
                auto_function_conf = excluded.auto_function_conf,
                auto_industry      = excluded.auto_industry,
                auto_industry_conf = excluded.auto_industry_conf,
                auto_reasoning     = excluded.auto_reasoning,
                classified_at      = excluded.classified_at
        """, (
            candidate_id,
            result["function"],  result["function_conf"],
            result["industry"],  result["industry_conf"],
            reasoning, now
        ))
        db.commit()
    except sqlite3.Error as e:
        logger.error(f"❌ auto_classify upsert failed: {e}")
        db.rollback()
        return None

    return result


# =============================================================================
# 🌐 ROUTES
# =============================================================================

@app.route("/")
def index():
    """
    📋 Main list view — paginated table of all candidates with badges for
    annotation status, accuracy score, Function, and Industry.
    """
    ensure_classification_table()

    page           = max(1, request.args.get("page", 1, type=int))
    search         = request.args.get("q", "").strip()
    status_filter  = request.args.get("status", "")
    func_filter    = request.args.get("function", "")
    ind_filter     = request.args.get("industry", "")
    review_filter  = request.args.get("review", "")
    sort_by        = request.args.get("sort", "candidate_id")
    sort_order     = request.args.get("order", "DESC")

    candidates, total = get_candidates_list(
        page=page, search=search,
        status_filter=status_filter,
        function_filter=func_filter,
        industry_filter=ind_filter,
        review_filter=review_filter,
        sort_by=sort_by, sort_order=sort_order,
    )

    total_pages = max(1, math.ceil(total / PER_PAGE))

    # ── Windowed pagination (same logic as annotation_tool) ────────────
    WINDOW = 2
    page_window = []
    for p in range(1, total_pages + 1):
        if (p == 1 or p == total_pages
                or abs(p - page) <= WINDOW):
            page_window.append(p)
        elif page_window and page_window[-1] is not None:
            page_window.append(None)   # Ellipsis sentinel

    # ── Dashboard summary stats ────────────────────────────────────────
    db = get_db()
    stats = {}
    try:
        stats["total"] = db.execute(
            "SELECT COUNT(*) as c FROM ner_documents"
        ).fetchone()["c"]
        stats["completed"] = db.execute(
            "SELECT COUNT(*) as c FROM ner_documents WHERE status='completed'"
        ).fetchone()["c"]
        stats["classified"] = db.execute(
            "SELECT COUNT(*) as c FROM candidate_classifications WHERE auto_function IS NOT NULL"
        ).fetchone()["c"]
        stats["reviewed"] = db.execute(
            "SELECT COUNT(*) as c FROM candidate_classifications WHERE reviewed_by IS NOT NULL"
        ).fetchone()["c"]
    except sqlite3.OperationalError:
        stats = {"total": 0, "completed": 0, "classified": 0, "reviewed": 0}

    return render_template_string(
        INDEX_TEMPLATE,
        candidates=candidates,
        total=total,
        page=page,
        total_pages=total_pages,
        page_window=page_window,
        search=search,
        status_filter=status_filter,
        func_filter=func_filter,
        ind_filter=ind_filter,
        review_filter=review_filter,
        sort_by=sort_by,
        sort_order=sort_order,
        function_list=FUNCTION_LIST,
        industry_list=INDUSTRY_LIST,
        stats=stats,
    )


@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id: int):
    """
    🔬 Detail view — side-by-side annotation accuracy panel + classification panel.
    Shows all entity chips, completeness warnings, and the classification dropdowns.
    """
    detail = get_candidate_detail(candidate_id)
    if not detail:
        abort(404, description=f"Candidate {candidate_id} not found")

    return render_template_string(
        DETAIL_TEMPLATE,
        candidate_id=candidate_id,
        doc=detail["doc"],
        annotations=detail["annotations"],
        raw_text=detail["raw_text"],
        classification=detail["classification"],
        accuracy=detail["accuracy"],
        function_list=FUNCTION_LIST,
        industry_list=INDUSTRY_LIST,
        entity_colors=ENTITY_COLORS,
        required_entities=REQUIRED_ENTITIES,
    )


@app.route("/api/classify/<int:candidate_id>", methods=["POST"])
def api_classify_one(candidate_id: int):
    """
    🤖 Auto-classify a single candidate and return the result as JSON.
    Called by the "Auto-Classify" button on the detail page.
    """
    result = auto_classify_candidate(candidate_id)
    if result is None:
        return jsonify({"error": "Candidate not found or has no annotations"}), 404
    return jsonify({"success": True, "result": result})


@app.route("/api/classify-all", methods=["POST"])
def api_classify_all():
    """
    🚀 Batch auto-classify ALL candidates that have completed annotations.

    Runs the classifier on every completed document and updates the
    candidate_classifications table. Returns a summary.

    This is the GRAND REVEAL — classify the whole wardrobe at once! 👗✨
    """
    db = get_db()
    ensure_classification_table()

    # Fetch all candidate_ids that have at least one ner_document
    rows = db.execute("""
        SELECT DISTINCT candidate_id FROM ner_documents
        WHERE status = 'completed'
    """).fetchall()

    total      = len(rows)
    classified = 0
    failed     = 0

    for row in rows:
        cid = row["candidate_id"]
        result = auto_classify_candidate(cid)
        if result is not None:
            classified += 1
        else:
            failed += 1

    logger.info(f"🤖 Batch classify: {classified}/{total} done, {failed} failed")

    return jsonify({
        "success":    True,
        "total":      total,
        "classified": classified,
        "failed":     failed,
    })


@app.route("/api/save-classification/<int:candidate_id>", methods=["POST"])
def api_save_classification(candidate_id: int):
    """
    💾 Save a human-reviewed Function + Industry for a candidate.

    Accepts JSON body:
        {
            "function":  "IT",
            "industry":  "Banking & Finance",
            "reviewer":  "sarah_hr"    (optional, defaults to 'reviewer')
        }

    Validates that function and industry are in the allowed taxonomies.
    Logs whether the auto-classification was correct (for future retraining).
    """
    data = request.get_json() or {}
    func  = (data.get("function") or "").strip()
    ind   = (data.get("industry") or "").strip()
    rev   = (data.get("reviewer") or "reviewer").strip()

    # ── Validate against taxonomy ──────────────────────────────────────
    if func and func not in FUNCTION_LIST:
        return jsonify({"error": f"Unknown function: {func}"}), 400
    if ind and ind not in INDUSTRY_LIST:
        return jsonify({"error": f"Unknown industry: {ind}"}), 400
    if not func and not ind:
        return jsonify({"error": "At least one of function/industry must be provided"}), 400

    # Load auto result for correctness tracking
    db = get_db()
    cls_row = db.execute(
        "SELECT auto_function, auto_industry FROM candidate_classifications WHERE candidate_id = ?",
        (candidate_id,)
    ).fetchone()
    auto_data = dict(cls_row) if cls_row else None

    ok = save_classification(candidate_id, func, ind, rev, auto_data)
    if ok:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Database save failed"}), 500


@app.route("/api/stats")
def api_stats():
    """
    📊 Return classification + annotation stats for the dashboard header.
    """
    db = get_db()
    ensure_classification_table()
    stats: Dict[str, Any] = {}

    try:
        stats["total_candidates"] = db.execute(
            "SELECT COUNT(DISTINCT candidate_id) as c FROM ner_documents"
        ).fetchone()["c"]

        stats["completed"] = db.execute(
            "SELECT COUNT(*) as c FROM ner_documents WHERE status='completed'"
        ).fetchone()["c"]

        stats["classified"] = db.execute(
            "SELECT COUNT(*) as c FROM candidate_classifications "
            "WHERE auto_function IS NOT NULL"
        ).fetchone()["c"]

        stats["reviewed"] = db.execute(
            "SELECT COUNT(*) as c FROM candidate_classifications "
            "WHERE reviewed_by IS NOT NULL"
        ).fetchone()["c"]

        # Auto-classifier accuracy (where reviewed)
        acc_row = db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_function_correct = 1 THEN 1 ELSE 0 END) as fn_correct,
                SUM(CASE WHEN is_industry_correct = 1 THEN 1 ELSE 0 END) as ind_correct
            FROM candidate_classifications
            WHERE reviewed_by IS NOT NULL
        """).fetchone()
        if acc_row and acc_row["total"] > 0:
            stats["fn_accuracy"]  = round(acc_row["fn_correct"]  / acc_row["total"] * 100)
            stats["ind_accuracy"] = round(acc_row["ind_correct"] / acc_row["total"] * 100)
        else:
            stats["fn_accuracy"]  = None
            stats["ind_accuracy"] = None

        # Function distribution
        fn_rows = db.execute("""
            SELECT COALESCE(reviewed_function, auto_function) as fn, COUNT(*) as cnt
            FROM candidate_classifications
            WHERE fn IS NOT NULL
            GROUP BY fn ORDER BY cnt DESC LIMIT 10
        """).fetchall()
        stats["function_dist"] = [dict(r) for r in fn_rows]

        # Industry distribution
        ind_rows = db.execute("""
            SELECT COALESCE(reviewed_industry, auto_industry) as ind, COUNT(*) as cnt
            FROM candidate_classifications
            WHERE ind IS NOT NULL
            GROUP BY ind ORDER BY cnt DESC LIMIT 10
        """).fetchall()
        stats["industry_dist"] = [dict(r) for r in ind_rows]

    except sqlite3.OperationalError as e:
        logger.warning(f"Stats query failed: {e}")

    return jsonify(stats)


# =============================================================================
# 🎨 SHARED CSS — Dark theme matching annotation_tool.py palette
# =============================================================================

SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-deep:       #080b12;
    --bg-primary:    #0e1117;
    --bg-secondary:  #161b22;
    --bg-elevated:   #1c2333;
    --bg-hover:      #252d3a;
    --border:        #2a3140;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #545d68;
    --accent-cyan:   #7ee8fa;
    --accent-purple: #eeb8ff;
    --accent-green:  #56d364;
    --accent-orange: #f0883e;
    --accent-red:    #da3633;
    --accent-yellow: #e3b341;
    --accent-pink:   #f778ba;
    --accent-blue:   #a371f7;
    --font-body:     'Outfit', system-ui, sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
    --font-display:  'Syne', system-ui, sans-serif;
    --radius:        8px;
    --shadow:        0 4px 24px rgba(0,0,0,0.4);
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
}
a { color: var(--accent-cyan); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Header ── */
.header {
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4);
}
.header h1 {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.3px;
}
.header h1 span { color: var(--accent-cyan); }
.header-actions { display: flex; gap: 8px; align-items: center; }

/* ── Buttons ── */
.btn {
    padding: 7px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-elevated);
    color: var(--text-primary);
    font-size: 0.82rem;
    font-family: var(--font-body);
    cursor: pointer;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}
.btn:hover { background: var(--bg-hover); border-color: var(--accent-cyan); color: var(--accent-cyan); }
.btn-primary { background: var(--accent-cyan); color: var(--bg-deep); border-color: var(--accent-cyan); font-weight: 600; }
.btn-primary:hover { opacity: 0.9; color: var(--bg-deep); }
.btn-success { background: var(--accent-green); color: var(--bg-deep); border-color: var(--accent-green); font-weight: 600; }
.btn-success:hover { opacity: 0.9; color: var(--bg-deep); }
.btn-warning { background: var(--accent-yellow); color: var(--bg-deep); border-color: var(--accent-yellow); font-weight: 600; }
.btn-sm { padding: 4px 10px; font-size: 0.75rem; }

/* ── KPI stat row ── */
.kpi-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border);
}
.kpi-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.15s;
}
.kpi-card:hover { border-color: var(--accent-cyan); }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: -12px; right: -12px;
    width: 60px; height: 60px;
    background: radial-gradient(circle, var(--kpi-glow, rgba(126,232,250,0.08)), transparent 70%);
    pointer-events: none;
}
.kpi-val {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--kpi-color, var(--accent-cyan));
    line-height: 1;
    margin-bottom: 3px;
}
.kpi-lbl {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

/* ── Filter bar ── */
.filter-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
}
.filter-bar input, .filter-bar select {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-primary);
    padding: 6px 10px;
    font-size: 0.82rem;
    font-family: var(--font-body);
    transition: border-color 0.15s;
}
.filter-bar input { width: 220px; }
.filter-bar select { min-width: 160px; }
.filter-bar input:focus, .filter-bar select:focus {
    outline: none;
    border-color: var(--accent-cyan);
}
.filter-bar label {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.filter-sep {
    width: 1px;
    height: 24px;
    background: var(--border);
    flex-shrink: 0;
}

/* ── Candidate table ── */
.table-wrap {
    padding: 16px 24px 100px;
    overflow-x: auto;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
thead th {
    background: var(--bg-secondary);
    color: var(--text-muted);
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
thead th a { color: var(--text-muted); }
thead th a:hover { color: var(--accent-cyan); text-decoration: none; }
tbody tr {
    border-bottom: 1px solid rgba(42,49,64,0.5);
    transition: background 0.1s;
}
tbody tr:hover { background: var(--bg-elevated); }
tbody td {
    padding: 10px 12px;
    vertical-align: middle;
}
.td-id {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
}
.td-name { font-weight: 600; color: var(--text-primary); }
.td-email { color: var(--text-secondary); font-size: 0.8rem; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    white-space: nowrap;
}
.badge-completed  { background: rgba(86,211,100,0.15);   color: var(--accent-green); }
.badge-progress   { background: rgba(126,232,250,0.12);  color: var(--accent-cyan); }
.badge-pending    { background: rgba(227,179,65,0.12);   color: var(--accent-yellow); }
.badge-fn  { background: rgba(57,211,83,0.12);  color: #39d353; }
.badge-ind { background: rgba(241,163,64,0.12); color: #f1a340; }
.badge-auto   { background: rgba(238,184,255,0.1);  color: var(--accent-purple); font-style: italic; }
.badge-human  { background: rgba(86,211,100,0.15);  color: var(--accent-green); }
.badge-others { background: rgba(84,93,104,0.2);    color: var(--text-muted); }

/* Accuracy grade badges */
.grade { 
    display: inline-flex;
    width: 28px; height: 28px;
    align-items: center; justify-content: center;
    border-radius: 50%;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 700;
}
.grade-A { background: rgba(86,211,100,0.2);   color: var(--accent-green); border: 1px solid rgba(86,211,100,0.4); }
.grade-B { background: rgba(126,232,250,0.15); color: var(--accent-cyan);  border: 1px solid rgba(126,232,250,0.3); }
.grade-C { background: rgba(227,179,65,0.15);  color: var(--accent-yellow);border: 1px solid rgba(227,179,65,0.3); }
.grade-D { background: rgba(240,136,62,0.15);  color: var(--accent-orange);border: 1px solid rgba(240,136,62,0.3); }
.grade-F { background: rgba(218,54,51,0.15);   color: var(--accent-red);   border: 1px solid rgba(218,54,51,0.3); }

/* Confidence bar (mini horizontal bar) */
.conf-bar {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 80px;
}
.conf-track {
    flex: 1;
    height: 5px;
    background: var(--bg-elevated);
    border-radius: 3px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--accent-cyan);
    transition: width 0.5s ease;
}
.conf-pct {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    width: 28px;
    text-align: right;
}

/* ── Pagination ── */
.pagination-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 4px;
    padding: 20px;
    flex-wrap: wrap;
}
.pagination-wrap a {
    padding: 5px 11px;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 0.78rem;
    transition: all 0.12s;
}
.pagination-wrap a:hover,
.pagination-wrap a.active {
    background: var(--accent-cyan);
    color: var(--bg-deep);
    border-color: var(--accent-cyan);
    text-decoration: none;
}
.pagination-wrap .pg-arrow { color: var(--accent-cyan); font-weight: 700; }
.pagination-wrap .pg-arrow.disabled { opacity: 0.3; pointer-events: none; }
.pagination-wrap .ellipsis { color: var(--text-muted); padding: 5px 4px; font-size: 0.78rem; }
.pagination-info { text-align: center; font-size: 0.72rem; color: var(--text-muted); padding-bottom: 6px; }

/* ── Toast notification ── */
.toast {
    position: fixed;
    bottom: 24px; right: 24px;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.84rem;
    box-shadow: var(--shadow);
    transform: translateY(80px);
    opacity: 0;
    transition: all 0.25s ease;
    z-index: 999;
    max-width: 340px;
}
.toast.show { transform: translateY(0); opacity: 1; }
.toast.success { border-color: var(--accent-green); }
.toast.error   { border-color: var(--accent-red); }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 80px 40px;
    color: var(--text-muted);
}
.empty-state .big-emoji { font-size: 3rem; margin-bottom: 12px; display: block; }
.empty-state h3 { color: var(--text-secondary); margin-bottom: 6px; }
"""


# =============================================================================
# 📄 INDEX TEMPLATE — Main list view
# =============================================================================

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🏷️ Classification & Accuracy Review</title>
  <style>""" + SHARED_CSS + """</style>
</head>
<body>

<!-- ── Header ── -->
<div class="header">
  <div style="display:flex; align-items:center; gap:14px;">
    <h1>🏷️ Classification &amp; <span>Accuracy Review</span></h1>
    <span style="font-size:0.72rem; color:var(--text-muted); background:var(--bg-elevated); padding:3px 9px; border-radius:4px; border:1px solid var(--border);">
      Port 5001
    </span>
  </div>
  <div class="header-actions">
    <a href="http://localhost:5000" class="btn" target="_blank">🏷️ Annotation Tool</a>
    <button class="btn btn-warning" onclick="classifyAll()" id="classifyAllBtn">
      🤖 Auto-Classify All
    </button>
  </div>
</div>

<!-- ── KPI Bar ── -->
<div class="kpi-bar">
  <div class="kpi-card" style="--kpi-color:var(--accent-cyan); --kpi-glow:rgba(126,232,250,0.1)">
    <div class="kpi-val">{{ stats.total }}</div>
    <div class="kpi-lbl">Total in Queue</div>
  </div>
  <div class="kpi-card" style="--kpi-color:var(--accent-green); --kpi-glow:rgba(86,211,100,0.1)">
    <div class="kpi-val">{{ stats.completed }}</div>
    <div class="kpi-lbl">Annotated ✅</div>
  </div>
  <div class="kpi-card" style="--kpi-color:#39d353; --kpi-glow:rgba(57,211,83,0.1)">
    <div class="kpi-val">{{ stats.classified }}</div>
    <div class="kpi-lbl">Auto-Classified 🤖</div>
  </div>
  <div class="kpi-card" style="--kpi-color:var(--accent-yellow); --kpi-glow:rgba(227,179,65,0.1)">
    <div class="kpi-val">{{ stats.reviewed }}</div>
    <div class="kpi-lbl">Human Reviewed ✍️</div>
  </div>
</div>

<!-- ── Filter Bar ── -->
<form method="GET" action="/">
  <div class="filter-bar">
    <!-- Search -->
    <input type="text" name="q" placeholder="🔍  Search name / email / ID…"
           value="{{ search }}" onchange="this.form.submit()">

    <div class="filter-sep"></div>
    <label>Status</label>
    <select name="status" onchange="this.form.submit()">
      <option value="">All</option>
      <option value="completed"  {{ 'selected' if status_filter=='completed' }}>✅ Completed</option>
      <option value="in_progress"{{ 'selected' if status_filter=='in_progress' }}>🔵 In Progress</option>
    </select>

    <label>Function</label>
    <select name="function" onchange="this.form.submit()">
      <option value="">All Functions</option>
      {% for fn in function_list %}
      <option value="{{ fn }}" {{ 'selected' if func_filter==fn }}>{{ fn }}</option>
      {% endfor %}
    </select>

    <label>Industry</label>
    <select name="industry" onchange="this.form.submit()">
      <option value="">All Industries</option>
      {% for ind in industry_list %}
      <option value="{{ ind }}" {{ 'selected' if ind_filter==ind }}>{{ ind }}</option>
      {% endfor %}
    </select>

    <div class="filter-sep"></div>
    <label>Review</label>
    <select name="review" onchange="this.form.submit()">
      <option value="">All</option>
      <option value="reviewed"   {{ 'selected' if review_filter=='reviewed' }}>✍️ Reviewed</option>
      <option value="unreviewed" {{ 'selected' if review_filter=='unreviewed' }}>⏳ Unreviewed</option>
    </select>

    {% if search or status_filter or func_filter or ind_filter or review_filter %}
    <a href="/" class="btn btn-sm" style="margin-left:auto">✕ Clear Filters</a>
    {% endif %}
  </div>
</form>

<!-- ── Table ── -->
<div class="table-wrap">
  {% if candidates %}
  <table>
    <thead>
      <tr>
        <th><a href="?sort=candidate_id&order={{ 'ASC' if sort_order=='DESC' else 'DESC' }}&q={{ search }}">ID</a></th>
        <th><a href="?sort=name&order={{ 'ASC' if sort_order=='DESC' else 'DESC' }}&q={{ search }}">Name</a></th>
        <th>Email</th>
        <th><a href="?sort=status&order={{ 'ASC' if sort_order=='DESC' else 'DESC' }}&q={{ search }}">Annotation</a></th>
        <th><a href="?sort=score&order={{ 'ASC' if sort_order=='DESC' else 'DESC' }}&q={{ search }}">Accuracy</a></th>
        <th><a href="?sort=function&order={{ 'ASC' if sort_order=='DESC' else 'DESC' }}&q={{ search }}">Function</a></th>
        <th><a href="?sort=industry&order={{ 'ASC' if sort_order=='DESC' else 'DESC' }}&q={{ search }}">Industry</a></th>
        <th>Reviewed</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody>
      {% for c in candidates %}
      <tr>
        <!-- ID -->
        <td class="td-id">#{{ c.candidate_id }}</td>

        <!-- Name -->
        <td>
          <div class="td-name">{{ c.name or '—' }}</div>
        </td>

        <!-- Email -->
        <td class="td-email">{{ c.email or '—' }}</td>

        <!-- Annotation Status -->
        <td>
          {% if c.status == 'completed' %}
            <span class="badge badge-completed">✅ Done</span>
          {% elif c.status == 'in_progress' %}
            <span class="badge badge-progress">🔵 In Progress</span>
          {% else %}
            <span class="badge badge-pending">⏳ {{ c.status or 'Pending' }}</span>
          {% endif %}
        </td>

        <!-- Accuracy Score -->
        <td>
          {% set score = c.acc_score or 0 %}
          {% if score >= 90 %}
            <span class="grade grade-A">A</span>
          {% elif score >= 75 %}
            <span class="grade grade-B">B</span>
          {% elif score >= 55 %}
            <span class="grade grade-C">C</span>
          {% elif score >= 35 %}
            <span class="grade grade-D">D</span>
          {% else %}
            <span class="grade grade-F">F</span>
          {% endif %}
          <span style="font-size:0.68rem; color:var(--text-muted); margin-left:4px;">{{ score }}%</span>
        </td>

        <!-- Function badge -->
        <td>
          {% set fn = c.reviewed_function or c.auto_function %}
          {% if fn %}
            <span class="badge {{ 'badge-human' if c.reviewed_function else 'badge-auto' }} badge-fn"
                  title="{{ '✍️ Human reviewed' if c.reviewed_function else '🤖 Auto-classified (' + (c.auto_function_conf or 0)|round(0)|string + '%)' }}">
              {{ fn }}
            </span>
            {% if c.auto_function and not c.reviewed_function %}
            <div class="conf-bar" style="margin-top:3px">
              <div class="conf-track">
                <div class="conf-fill" style="width:{{ ((c.auto_function_conf or 0)*100)|int }}%"></div>
              </div>
              <span class="conf-pct">{{ ((c.auto_function_conf or 0)*100)|int }}%</span>
            </div>
            {% endif %}
          {% else %}
            <span style="color:var(--text-muted); font-size:0.75rem">—</span>
          {% endif %}
        </td>

        <!-- Industry badge -->
        <td>
          {% set ind = c.reviewed_industry or c.auto_industry %}
          {% if ind %}
            <span class="badge {{ 'badge-human' if c.reviewed_industry else 'badge-auto' }} badge-ind"
                  title="{{ '✍️ Human reviewed' if c.reviewed_industry else '🤖 Auto-classified' }}">
              {{ ind }}
            </span>
            {% if c.auto_industry and not c.reviewed_industry %}
            <div class="conf-bar" style="margin-top:3px">
              <div class="conf-track">
                <div class="conf-fill" style="width:{{ ((c.auto_industry_conf or 0)*100)|int }}%; background:var(--accent-orange)"></div>
              </div>
              <span class="conf-pct">{{ ((c.auto_industry_conf or 0)*100)|int }}%</span>
            </div>
            {% endif %}
          {% else %}
            <span style="color:var(--text-muted); font-size:0.75rem">—</span>
          {% endif %}
        </td>

        <!-- Reviewed by -->
        <td>
          {% if c.reviewed_by %}
            <span style="font-size:0.75rem; color:var(--accent-green)">✍️ {{ c.reviewed_by }}</span>
            {% if c.reviewed_at %}
            <div style="font-size:0.65rem; color:var(--text-muted)">{{ c.reviewed_at[:10] }}</div>
            {% endif %}
          {% else %}
            <span style="color:var(--text-muted); font-size:0.75rem">—</span>
          {% endif %}
        </td>

        <!-- Action -->
        <td>
          <a href="/candidate/{{ c.candidate_id }}" class="btn btn-sm btn-primary">
            🔬 Review
          </a>
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <!-- Pagination -->
  <div class="pagination-info">
    Page {{ page }} of {{ total_pages }} &middot; {{ total }} candidates
  </div>
  <div class="pagination-wrap">
    <a href="?page={{ page-1 }}&q={{ search }}&status={{ status_filter }}&function={{ func_filter }}&industry={{ ind_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}"
       class="pg-arrow {{ 'disabled' if page <= 1 }}">« Prev</a>
    {% for p in page_window %}
      {% if p is none %}
        <span class="ellipsis">…</span>
      {% else %}
        <a href="?page={{ p }}&q={{ search }}&status={{ status_filter }}&function={{ func_filter }}&industry={{ ind_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}"
           class="{{ 'active' if p == page }}">{{ p }}</a>
      {% endif %}
    {% endfor %}
    <a href="?page={{ page+1 }}&q={{ search }}&status={{ status_filter }}&function={{ func_filter }}&industry={{ ind_filter }}&review={{ review_filter }}&sort={{ sort_by }}&order={{ sort_order }}"
       class="pg-arrow {{ 'disabled' if page >= total_pages }}">Next »</a>
  </div>

  {% else %}
  <!-- Empty state -->
  <div class="empty-state">
    <span class="big-emoji">🔍</span>
    <h3>No candidates found</h3>
    <p>Try adjusting your filters, or run the annotation tool first.</p>
    <div style="margin-top:16px; display:flex; gap:10px; justify-content:center">
      <a href="/" class="btn">Clear Filters</a>
      <button class="btn btn-warning" onclick="classifyAll()">🤖 Auto-Classify All</button>
    </div>
  </div>
  {% endif %}
</div>

<!-- ── Toast ── -->
<div class="toast" id="toast"></div>

<script>
// ── Toast helper ────────────────────────────────────────────────────────────
function showToast(msg, type='success') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = `toast show ${type}`;
    setTimeout(() => t.className = 'toast', 3500);
}

// ── Auto-classify ALL ───────────────────────────────────────────────────────
async function classifyAll() {
    const btn = document.getElementById('classifyAllBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Classifying…';

    try {
        const r = await fetch('/api/classify-all', { method: 'POST' });
        const d = await r.json();
        if (d.success) {
            showToast(`🤖 Done! ${d.classified}/${d.total} candidates classified`, 'success');
            setTimeout(() => location.reload(), 1500);
        } else {
            showToast('❌ Classification failed', 'error');
        }
    } catch(err) {
        showToast('❌ Network error: ' + err, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🤖 Auto-Classify All';
    }
}
</script>
</body>
</html>
"""


# =============================================================================
# 🔬 DETAIL TEMPLATE — Annotation accuracy + classification panel
# =============================================================================

DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Review — Candidate #{{ candidate_id }}</title>
  <style>""" + SHARED_CSS + """

/* ── Detail layout: two-column ── */
.detail-grid {
    display: grid;
    grid-template-columns: 1fr 400px;
    gap: 0;
    height: calc(100vh - 57px);
    overflow: hidden;
}

/* ── LEFT panel: annotation accuracy ── */
.acc-panel {
    overflow-y: auto;
    padding: 20px 24px;
    border-right: 1px solid var(--border);
}

/* ── RIGHT panel: classification form ── */
.cls-panel {
    overflow-y: auto;
    padding: 20px;
    background: var(--bg-secondary);
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* ── Section card ── */
.card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
}
.card h3 {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Accuracy score ring ── */
.score-ring-wrap {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 14px;
}
.score-ring {
    position: relative;
    width: 88px;
    height: 88px;
    flex-shrink: 0;
}
.score-ring svg { transform: rotate(-90deg); }
.ring-track { fill: none; stroke: var(--bg-secondary); stroke-width: 10; }
.ring-fill  {
    fill: none;
    stroke-width: 10;
    stroke-linecap: round;
    transition: stroke-dashoffset 1s ease;
}
.score-center {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.score-number {
    font-family: var(--font-mono);
    font-size: 1.2rem;
    font-weight: 700;
    line-height: 1;
}
.score-pct {
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.score-meta h4 {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.score-meta p {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* ── Required entity checklist ── */
.entity-checklist { display: flex; flex-direction: column; gap: 6px; }
.entity-check-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 6px;
    font-size: 0.8rem;
    transition: background 0.1s;
}
.entity-check-row:hover { background: var(--bg-secondary); }
.entity-check-row.present { }
.entity-check-row.missing { opacity: 0.55; }
.check-icon { font-size: 0.85rem; flex-shrink: 0; }
.check-label { flex: 1; color: var(--text-secondary); }
.check-label.present { color: var(--text-primary); font-weight: 600; }
.check-count {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
}

/* ── Entity chip cloud ── */
.entity-cloud { display: flex; flex-wrap: wrap; gap: 6px; }
.entity-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid transparent;
    cursor: default;
    transition: transform 0.1s;
}
.entity-chip:hover { transform: translateY(-1px); }
.entity-chip .chip-count {
    background: rgba(0,0,0,0.25);
    border-radius: 10px;
    padding: 0 5px;
    font-size: 0.62rem;
    margin-left: 2px;
}

/* ── Warning alerts ── */
.warning-list { display: flex; flex-direction: column; gap: 6px; }
.warning-item {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    background: rgba(227,179,65,0.06);
    border: 1px solid rgba(227,179,65,0.2);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.warning-item .warn-icon { color: var(--accent-yellow); flex-shrink: 0; margin-top: 1px; }

/* ── Raw text viewer ── */
.raw-text-box {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    line-height: 1.7;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-secondary);
}
/* Inline annotation highlights in raw text */
.raw-highlight {
    border-radius: 3px;
    padding: 1px 0;
    cursor: help;
    position: relative;
}

/* ── Classification form (right panel) ── */
.cls-header {
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.cls-header h2 {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 4px;
}
.cls-header .sub {
    font-size: 0.75rem;
    color: var(--text-muted);
}
.cls-section { display: flex; flex-direction: column; gap: 8px; }
.cls-section label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
}
.cls-section select {
    width: 100%;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-primary);
    padding: 9px 12px;
    font-size: 0.85rem;
    font-family: var(--font-body);
    cursor: pointer;
    transition: border-color 0.15s;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%238b949e'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 12px center;
}
.cls-section select:focus { outline: none; border-color: var(--accent-cyan); }
.cls-section select option { background: var(--bg-elevated); }

/* Auto suggestion pill */
.auto-suggestion {
    background: rgba(238,184,255,0.08);
    border: 1px solid rgba(238,184,255,0.2);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.auto-suggestion .sug-label {
    color: var(--accent-purple);
    font-weight: 700;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.auto-suggestion .sug-value { color: var(--text-primary); font-weight: 600; }
.auto-suggestion .sug-conf  { color: var(--text-muted); font-size: 0.7rem; }

/* Top-3 alternatives */
.alternatives { margin-top: 6px; }
.alt-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
    font-size: 0.72rem;
}
.alt-name { color: var(--text-secondary); }
.alt-bar-track {
    flex: 1;
    height: 4px;
    background: var(--bg-elevated);
    border-radius: 2px;
    margin: 0 8px;
    overflow: hidden;
}
.alt-bar-fill { height: 100%; background: var(--accent-purple); border-radius: 2px; }
.alt-pct { font-family: var(--font-mono); color: var(--text-muted); width: 30px; text-align: right; }

/* Reviewer input */
.reviewer-wrap {
    display: flex;
    gap: 8px;
    align-items: center;
}
.reviewer-wrap input {
    flex: 1;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-primary);
    padding: 8px 12px;
    font-size: 0.84rem;
    font-family: var(--font-body);
    transition: border-color 0.15s;
}
.reviewer-wrap input:focus { outline: none; border-color: var(--accent-cyan); }

/* Save feedback */
.save-feedback {
    font-size: 0.8rem;
    text-align: center;
    padding: 8px;
    border-radius: 6px;
    display: none;
}
.save-feedback.success { background: rgba(86,211,100,0.12); color: var(--accent-green); }
.save-feedback.error   { background: rgba(218,54,51,0.12);  color: var(--accent-red); }

/* Previous review history */
.review-history {
    background: rgba(126,232,250,0.04);
    border: 1px solid rgba(126,232,250,0.15);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.78rem;
}
.review-history .rh-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--accent-cyan);
    font-weight: 700;
    margin-bottom: 6px;
}

  </style>
</head>
<body>

<!-- ── Header ── -->
<div class="header">
  <div style="display:flex; align-items:center; gap:12px;">
    <a href="/" class="btn btn-sm">← Back</a>
    <h1 style="font-size:1rem">
      🔬 Candidate <span>#{{ candidate_id }}</span>
      {% if doc.name %} — {{ doc.name }}{% endif %}
    </h1>
    <span class="badge {{ 'badge-completed' if doc.status=='completed' else 'badge-progress' }}">
      {{ doc.status }}
    </span>
  </div>
  <div class="header-actions">
    <button class="btn" onclick="runAutoClassify()" id="autoBtn">
      🤖 Auto-Classify This
    </button>
    <button class="btn btn-primary" onclick="saveClassification()" id="saveBtn">
      💾 Save Review
    </button>
  </div>
</div>

<!-- ── Two-column layout ── -->
<div class="detail-grid">

  <!-- ════════════════════════════════════════════
       LEFT PANEL: Annotation Accuracy
       ════════════════════════════════════════════ -->
  <div class="acc-panel">

    <!-- Accuracy Score Ring -->
    <div class="card">
      <h3>📊 Annotation Accuracy</h3>
      <div class="score-ring-wrap">
        {% set score = accuracy.score %}
        {% set grade = accuracy.grade %}
        {% set ring_color = {
            'A': '#56d364', 'B': '#7ee8fa',
            'C': '#e3b341', 'D': '#f0883e', 'F': '#da3633'
           }[grade] %}
        {% set circ = 2 * 3.14159 * 34 %}
        {% set dash = (score / 100) * circ %}

        <div class="score-ring">
          <svg width="88" height="88" viewBox="0 0 88 88">
            <circle class="ring-track" cx="44" cy="44" r="34"/>
            <circle class="ring-fill" cx="44" cy="44" r="34"
              stroke="{{ ring_color }}"
              stroke-dasharray="{{ dash | round(1) }} {{ circ | round(1) }}"
              stroke-dashoffset="0"/>
          </svg>
          <div class="score-center">
            <div class="score-number" style="color:{{ ring_color }}">{{ score }}</div>
            <div class="score-pct">%</div>
          </div>
        </div>

        <div class="score-meta">
          <h4 style="color:{{ ring_color }}">Grade {{ grade }}
            — {{ accuracy.present|length }} / {{ (accuracy.present|length + accuracy.missing|length) }} required entities
          </h4>
          <p>
            {{ annotations|length }} total annotation spans
            · {{ accuracy.entity_counts|length }} distinct entity types
          </p>
          {% if accuracy.missing %}
          <p style="color:var(--accent-yellow); margin-top:4px; font-size:0.75rem">
            ⚠️ Missing: {{ accuracy.missing|join(', ') }}
          </p>
          {% endif %}
        </div>
      </div>

      <!-- Required entity checklist -->
      <div class="entity-checklist">
        {% for etype, info in required_entities.items() %}
        {% set count = accuracy.entity_counts.get(etype, 0) %}
        <div class="entity-check-row {{ 'present' if count > 0 else 'missing' }}">
          <span class="check-icon">{{ '✅' if count > 0 else '⬜' }}</span>
          <span class="check-label {{ 'present' if count > 0 else '' }}"
                style="{% if count > 0 %}color:{{ entity_colors.get(etype, '#8b949e') }}{% endif %}">
            {{ info.label }}
            <span style="font-size:0.65rem; color:var(--text-muted); font-weight:400">({{ etype }})</span>
          </span>
          {% if count > 0 %}
          <span class="check-count">× {{ count }}</span>
          {% else %}
          <span style="font-size:0.65rem; color:var(--text-muted)">missing</span>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </div>

    <!-- Warnings -->
    {% if accuracy.warnings %}
    <div class="card">
      <h3>⚠️ Annotation Warnings</h3>
      <div class="warning-list">
        {% for w in accuracy.warnings %}
        <div class="warning-item">
          <span class="warn-icon">⚠️</span>
          <span>{{ w }}</span>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    <!-- All entity chips -->
    <div class="card">
      <h3>🏷️ All Annotation Spans
        <span style="font-weight:400; color:var(--text-muted)">({{ annotations|length }} total)</span>
      </h3>

      <!-- Group chips by entity type -->
      {% set grouped = {} %}
      {% for ann in annotations %}
        {% if ann.layer == 0 %}
          {% if ann.entity_type not in grouped %}
            {% set _ = grouped.__setitem__(ann.entity_type, []) %}
          {% endif %}
          {% set _ = grouped[ann.entity_type].append(ann) %}
        {% endif %}
      {% endfor %}

      {% if grouped %}
      <div class="entity-cloud">
        {% for etype, etype_anns in grouped.items() | sort %}
        {% set color = entity_colors.get(etype, '#8b949e') %}
        <!-- One chip per unique text value (show first 3, then count) -->
        {% for ann in etype_anns[:1] %}
        <div class="entity-chip"
             style="background:{{ color }}18; color:{{ color }}; border-color:{{ color }}30;"
             title="{{ etype }}: {{ etype_anns|map(attribute='text')|list|join(', ') }}">
          <span style="font-size:0.6rem; opacity:0.7">{{ etype }}</span>
          <span style="margin: 0 2px;">·</span>
          <span>{{ ann.text[:28] }}{{ '…' if ann.text|length > 28 else '' }}</span>
          {% if etype_anns|length > 1 %}
          <span class="chip-count">+{{ etype_anns|length - 1 }}</span>
          {% endif %}
        </div>
        {% endfor %}
        {% endfor %}
      </div>
      {% else %}
      <p style="color:var(--text-muted); font-size:0.82rem">No annotations yet.</p>
      {% endif %}
    </div>

    <!-- Raw text preview with annotation highlights -->
    {% if raw_text %}
    <div class="card">
      <h3>📄 Resume Text (read-only reference)</h3>
      <div class="raw-text-box" id="rawTextBox">{{ raw_text | e }}</div>
    </div>
    {% endif %}

  </div>  <!-- /acc-panel -->


  <!-- ════════════════════════════════════════════
       RIGHT PANEL: Classification
       ════════════════════════════════════════════ -->
  <div class="cls-panel">

    <div class="cls-header">
      <h2>🏷️ Classification</h2>
      <div class="sub">
        Review or correct the auto-classified Function &amp; Industry labels.
        <br>Human review always overrides auto-classification in exports.
      </div>
    </div>

    <!-- Previous review (if any) -->
    {% if classification.reviewed_by %}
    <div class="review-history">
      <div class="rh-label">✍️ Last Reviewed</div>
      <div>By <strong>{{ classification.reviewed_by }}</strong>
        {% if classification.reviewed_at %}
          on {{ classification.reviewed_at[:10] }}
        {% endif %}
      </div>
      <div style="margin-top:4px; font-size:0.8rem; color:var(--text-secondary)">
        Function: <strong style="color:#39d353">{{ classification.reviewed_function or '—' }}</strong>
        &nbsp;·&nbsp;
        Industry: <strong style="color:#f1a340">{{ classification.reviewed_industry or '—' }}</strong>
      </div>
    </div>
    {% endif %}

    <!-- ── Function Section ── -->
    <div class="card" style="margin-bottom:0">
      <h3>🗂️ Function / Department</h3>
      <div class="cls-section">

        <!-- Auto suggestion -->
        {% if classification.auto_function %}
        <div class="auto-suggestion">
          <div class="sug-label">🤖 Auto Suggestion</div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px">
            <span class="sug-value">{{ classification.auto_function }}</span>
            <span class="sug-conf">{{ (classification.auto_function_conf * 100)|int }}% confidence</span>
          </div>
          <!-- Top-3 alternatives from reasoning -->
          {% if classification.auto_reasoning %}
          {% set reasoning = classification.auto_reasoning | from_json %}
          {% if reasoning.function_top3 and reasoning.function_top3|length > 1 %}
          <div class="alternatives">
            {% set max_score = reasoning.function_top3[0][1] %}
            {% for label, score in reasoning.function_top3 %}
            <div class="alt-row">
              <span class="alt-name">{{ label }}</span>
              <div class="alt-bar-track">
                <div class="alt-bar-fill"
                     style="width:{{ (score / [max_score, 0.01]|max * 100)|int }}%"></div>
              </div>
              <span class="alt-pct">{{ (score)|round(1) }}</span>
            </div>
            {% endfor %}
          </div>
          {% endif %}
          {% endif %}
        </div>
        {% endif %}

        <label for="fnSelect">✍️ Your Classification</label>
        <select id="fnSelect">
          <option value="">— Select Function —</option>
          {% for fn in function_list %}
          <option value="{{ fn }}"
            {{ 'selected' if (classification.reviewed_function == fn
                              or (not classification.reviewed_function
                                  and classification.auto_function == fn)) }}>
            {{ fn }}
          </option>
          {% endfor %}
        </select>

        <!-- Quick-select chips for top 6 common functions -->
        <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top:4px">
          {% for fn in ['IT', 'Sales', 'Accounting & Finance', 'Engineering', 'Human Resources', 'Customer Service'] %}
          <button class="btn btn-sm" onclick="document.getElementById('fnSelect').value='{{ fn }}'"
                  style="font-size:0.7rem; padding:3px 8px">{{ fn }}</button>
          {% endfor %}
        </div>
      </div>
    </div>

    <!-- ── Industry Section ── -->
    <div class="card" style="margin-bottom:0">
      <h3>🏭 Industry / Sector</h3>
      <div class="cls-section">

        <!-- Auto suggestion -->
        {% if classification.auto_industry %}
        <div class="auto-suggestion" style="border-color:rgba(241,163,64,0.25); background:rgba(241,163,64,0.06)">
          <div class="sug-label" style="color:#f1a340">🤖 Auto Suggestion</div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px">
            <span class="sug-value">{{ classification.auto_industry }}</span>
            <span class="sug-conf">{{ (classification.auto_industry_conf * 100)|int }}% confidence</span>
          </div>
          {% if classification.auto_reasoning %}
          {% set reasoning = classification.auto_reasoning | from_json %}
          {% if reasoning.industry_top3 and reasoning.industry_top3|length > 1 %}
          <div class="alternatives">
            {% set max_score = reasoning.industry_top3[0][1] %}
            {% for label, score in reasoning.industry_top3 %}
            <div class="alt-row">
              <span class="alt-name">{{ label }}</span>
              <div class="alt-bar-track">
                <div class="alt-bar-fill"
                     style="width:{{ (score / [max_score, 0.01]|max * 100)|int }}%; background:#f1a340"></div>
              </div>
              <span class="alt-pct">{{ (score)|round(1) }}</span>
            </div>
            {% endfor %}
          </div>
          {% endif %}
          {% endif %}
        </div>
        {% endif %}

        <label for="indSelect">✍️ Your Classification</label>
        <select id="indSelect">
          <option value="">— Select Industry —</option>
          {% for ind in industry_list %}
          <option value="{{ ind }}"
            {{ 'selected' if (classification.reviewed_industry == ind
                              or (not classification.reviewed_industry
                                  and classification.auto_industry == ind)) }}>
            {{ ind }}
          </option>
          {% endfor %}
        </select>

        <!-- Quick-select chips for top 6 common industries -->
        <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top:4px">
          {% for ind in ['Banking & Finance', 'Infocomm', 'Healthcare', 'Retail', 'Manufacturing', 'Semiconductor'] %}
          <button class="btn btn-sm" onclick="document.getElementById('indSelect').value='{{ ind }}'"
                  style="font-size:0.7rem; padding:3px 8px">{{ ind }}</button>
          {% endfor %}
        </div>
      </div>
    </div>

    <!-- ── Reviewer name + Save ── -->
    <div class="card" style="margin-bottom:0">
      <h3>✍️ Reviewer</h3>
      <div class="reviewer-wrap">
        <input type="text" id="reviewerInput"
               placeholder="Your name or username"
               value="{{ classification.reviewed_by or '' }}">
      </div>
    </div>

    <button class="btn btn-success" onclick="saveClassification()" style="width:100%; padding:11px">
      💾 Save Classification
    </button>
    <div class="save-feedback" id="saveFeedback"></div>

    <!-- ── Nav buttons ── -->
    <div style="display:flex; gap:8px">
      <a href="/?status=completed" class="btn" style="flex:1; justify-content:center">
        ← Back to List
      </a>
    </div>

  </div>  <!-- /cls-panel -->
</div>  <!-- /detail-grid -->


<script>
const CANDIDATE_ID = {{ candidate_id }};
const CLASSIFICATION = {{ classification | tojson }};

// ── Auto-classify this candidate ────────────────────────────────────────────
async function runAutoClassify() {
    const btn = document.getElementById('autoBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Classifying…';

    try {
        const r = await fetch(`/api/classify/${CANDIDATE_ID}`, { method: 'POST' });
        const d = await r.json();
        if (d.success) {
            // Update dropdowns with auto result
            const res = d.result;
            document.getElementById('fnSelect').value = res.function  || '';
            document.getElementById('indSelect').value = res.industry || '';
            showFeedback(`🤖 Auto-classified: ${res.function} / ${res.industry} (fn:${Math.round(res.function_conf*100)}%, ind:${Math.round(res.industry_conf*100)}%)`, 'success');
        } else {
            showFeedback('❌ ' + (d.error || 'Classification failed'), 'error');
        }
    } catch(err) {
        showFeedback('❌ Network error: ' + err, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🤖 Auto-Classify This';
    }
}

// ── Save classification ─────────────────────────────────────────────────────
async function saveClassification() {
    const fn  = document.getElementById('fnSelect').value;
    const ind = document.getElementById('indSelect').value;
    const rev = document.getElementById('reviewerInput').value.trim() || 'reviewer';

    if (!fn && !ind) {
        showFeedback('⚠️ Please select at least a Function or Industry', 'error');
        return;
    }

    const btn = document.getElementById('saveBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Saving…';

    try {
        const r = await fetch(`/api/save-classification/${CANDIDATE_ID}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ function: fn, industry: ind, reviewer: rev })
        });
        const d = await r.json();
        if (d.success) {
            showFeedback('✅ Classification saved successfully!', 'success');
        } else {
            showFeedback('❌ Save failed: ' + (d.error || 'Unknown error'), 'error');
        }
    } catch(err) {
        showFeedback('❌ Network error: ' + err, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '💾 Save Review';
    }
}

// ── Feedback helper ─────────────────────────────────────────────────────────
function showFeedback(msg, type) {
    const el = document.getElementById('saveFeedback');
    el.textContent = msg;
    el.className = `save-feedback ${type}`;
    el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 5000);
}

// ── Keyboard shortcut: Ctrl+S to save ──────────────────────────────────────
document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveClassification();
    }
});
</script>
</body>
</html>
"""


# =============================================================================
# 🔧 JINJA2 CUSTOM FILTER
# =============================================================================

@app.template_filter('from_json')
def from_json_filter(value):
    """
    Jinja2 filter to parse a JSON string inside a template.
    Used to parse auto_reasoning stored as JSON in the DB.

    Usage in template: {{ some_json_string | from_json }}
    Returns empty dict on parse failure (never crashes the template!).
    """
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


# =============================================================================
# 🚀 APP STARTUP
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  💅✨ FAIRY CODEMOTHER'S CLASSIFICATION DASHBOARD ✨💅")
    print("=" * 62)
    print(f"  📂 Database : {DATABASE_PATH}")
    print(f"  🌐 URL      : http://localhost:{PORT}")
    print(f"  🔧 Debug    : {DEBUG}")
    print(f"  🏷️  Functions : {len(FUNCTION_LIST)}")
    print(f"  🏭 Industries: {len(INDUSTRY_LIST)}")
    print("=" * 62)

    # ── Verify database exists ─────────────────────────────────────────
    if not os.path.exists(DATABASE_PATH):
        print(f"\n  ❌ Database not found: {DATABASE_PATH}")
        print(f"  💡 Run annotation_tool.py first to create the DB,")
        print(f"     or set NER_DB_PATH environment variable.\n")
        exit(1)

    # ── Quick DB health check + create classifications table ───────────
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row

        # Verify ner_documents table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}

        if "ner_documents" not in table_names:
            print(f"\n  ❌ ner_documents table not found in {DATABASE_PATH}")
            print(f"  💡 Please run annotation_tool.py first to set up the schema.\n")
            conn.close()
            exit(1)

        count = conn.execute(
            "SELECT COUNT(*) as c FROM ner_documents"
        ).fetchone()["c"]

        completed = conn.execute(
            "SELECT COUNT(*) as c FROM ner_documents WHERE status='completed'"
        ).fetchone()["c"]

        # Ensure our classifications table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_classifications (
                candidate_id        INTEGER PRIMARY KEY,
                auto_function       TEXT,
                auto_function_conf  REAL DEFAULT 0.0,
                auto_industry       TEXT,
                auto_industry_conf  REAL DEFAULT 0.0,
                auto_reasoning      TEXT,
                reviewed_function   TEXT,
                reviewed_industry   TEXT,
                is_function_correct INTEGER,
                is_industry_correct INTEGER,
                reviewed_by         TEXT,
                reviewed_at         TEXT,
                classified_at       TEXT
            )
        """)
        conn.commit()
        conn.close()

        print(f"  ✅ Database OK — {count} candidates ({completed} completed)\n")

    except sqlite3.Error as e:
        print(f"\n  ❌ Database health check failed: {e}\n")
        exit(1)

    app.run(host=HOST, port=PORT, debug=DEBUG)