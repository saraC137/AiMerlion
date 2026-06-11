"""
stage0_lite_generate.py

💎✨ FAIRY CODEMOTHER'S STAGE 0 LITE GENERATOR ✨💎

Generates 100 synthetic SG/MY resumes for fine-tuning sanity check.
- 80 via local Qwen 2.5 32B (Ollama)
- 20 hard cases via Gemini API
- Auto-validates against canonical schema
- Auto-checks for leakage against golden test set
- Saves progress incrementally (resumable on crash)

Usage:
    cd C:\\Users\\user\\github\\AiMerlion
    python stage0_lite_generate.py

Output:
    data_gen/synthetic_lite.jsonl           — all 100 resumes
    data_gen/synthetic_lite_progress.json   — checkpoint (for resume on crash)
    data_gen/synthetic_lite_rejected.jsonl  — failed generations (for inspection)
"""

import os
import sys
import json
import time
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# ─── Load .env ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Ollama client (for local Qwen 32B) ─────────────────────────────────────
import requests
OLLAMA_URL = "http://127.0.0.1:11434"
LOCAL_MODEL = "qwen2.5:32b-instruct-q4_K_M"

# ─── Content banks loader (built by build_content_banks.py) ────────────────
BANKS_DIR = Path("content_banks")
BANKS = {}

def load_content_banks(banks_dir: Path = BANKS_DIR) -> Dict:
    """Load all JSON banks. Returns dict of bank_name → bank_data."""
    banks = {}
    if not banks_dir.exists():
        print(f"⚠️  No content_banks/ directory found.")
        print(f"⚠️  Run `python build_content_banks.py` first, OR continue with Qwen fallback only.")
        return banks
    for json_file in banks_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                banks[json_file.stem] = json.load(f)
            print(f"   📚 Loaded {json_file.name}")
        except json.JSONDecodeError as e:
            print(f"   ⚠️  Could not parse {json_file.name}: {e}")
    return banks


# ════════════════════════════════════════════════════════════════════════════
# 🎨 DIVERSITY MATRIX — 100 cells for Stage 0 LITE
# ════════════════════════════════════════════════════════════════════════════

DIVERSITY_MATRIX = [
    # (ethnicity, industry, seniority, education_path, count, hard?)
    ("malay_sg",       "finance",        "mid_5yr",       "polytechnic_then_uni", 8,  False),
    ("malay_sg",       "finance",        "senior_10yr",   "polytechnic_then_uni", 4,  False),
    ("malay_sg",       "finance",        "junior_2yr",    "uni_local",            3,  False),
    ("malay_sg",       "tech",           "mid_5yr",       "uni_local",            5,  False),
    ("malay_sg",       "tech",           "junior_2yr",    "polytechnic_then_uni", 3,  False),
    ("malay_sg",       "education",      "mid_5yr",       "uni_local",            2,  False),

    ("chinese_sg",     "finance",        "mid_5yr",       "uni_local",            8,  False),
    ("chinese_sg",     "finance",        "senior_10yr",   "uni_local",            5,  False),
    ("chinese_sg",     "finance",        "fresh_grad",    "uni_local",            2,  False),
    ("chinese_sg",     "tech",           "mid_5yr",       "uni_local",            6,  False),
    ("chinese_sg",     "tech",           "senior_10yr",   "uni_overseas",         3,  False),
    ("chinese_sg",     "healthcare",     "mid_5yr",       "uni_local",            3,  False),
    ("chinese_sg",     "hospitality",    "junior_2yr",    "polytechnic_then_uni", 3,  False),

    ("indian_sg",      "tech",           "mid_5yr",       "uni_local",            4,  False),
    ("indian_sg",      "finance",        "senior_10yr",   "uni_local",            3,  False),
    ("indian_sg",      "healthcare",     "mid_5yr",       "uni_local",            3,  False),

    ("chinese_my",     "manufacturing",  "mid_5yr",       "uni_local",            5,  False),
    ("chinese_my",     "tech",           "senior_10yr",   "uni_overseas",         3,  False),
    ("malay_my",       "government",     "mid_5yr",       "uni_local",            4,  False),
    ("malay_my",       "manufacturing",  "junior_2yr",    "uni_local",            3,  False),

    # Edge cases (hard) — sent to Gemini API for higher quality
    ("malay_sg",       "finance",        "lead_15yr",     "ITE_then_uni",         3,  True),
    ("chinese_sg",     "tech",           "senior_10yr",   "diploma_only",         3,  True),
    ("indian_sg",      "education",      "senior_10yr",   "uni_overseas",         3,  True),
    ("malay_my",       "logistics",      "lead_15yr",     "professional_cert_only", 3, True),
    ("chinese_sg",     "fmcg",           "mid_5yr",       "uni_local",            4,  True),
    ("chinese_my",     "finance",        "senior_10yr",   "uni_overseas",         4,  True),
]


# ════════════════════════════════════════════════════════════════════════════
# 📋 CANONICAL SCHEMA (must match annotation tool + golden test set)
# ════════════════════════════════════════════════════════════════════════════

CANONICAL_SCHEMA_KEYS = [
    "ID", "Name", "Page", "Phone", "Email",
    "Current Company", "Current Title", "Team",
    "Current Location", "Expected Location",
    "Gender", "Created By", "Creation Date", "Last Contact",
    "Function", "Industry", "Summary", "Language Skills",
    "Work Experience", "Project Experience", "Education",
    "Certifications", "hard_skills/tags", "soft_skills/skills",
    "Achievements", "References", "Hobbies"
]


# ════════════════════════════════════════════════════════════════════════════
# 🎭 PROMPT BUILDER
# ════════════════════════════════════════════════════════════════════════════

ETHNICITY_GUIDANCE = {
    "malay_sg": {
        "name_pattern": "Use Malay name with bin (for males) or binti/binte (for females) patronymic. Example: 'Ahmad bin Abdullah', 'Siti binti Mohamed'.",
        "phone_format": "Use the MANDATORY PHONE provided above. Do not invent a different phone.",
        "country_context": "Singapore — use Pte Ltd suffix for companies, Singapore polytechnic/university names."
    },
    "chinese_sg": {
        "name_pattern": "Chinese Singaporean name, typically 3 characters (e.g., 'Tan Wei Ming', 'Lim Hui Ling') or with English first name ('Jeremy Tan', 'Vanessa Chua').",
        "phone_format": "Use the MANDATORY PHONE provided above. Do not invent a different phone.",
        "country_context": "Singapore — use Pte Ltd suffix."
    },
    "indian_sg": {
        "name_pattern": "Indian Singaporean name often with s/o (son of) or d/o (daughter of) patronymic. Example: 'Rajesh s/o Kumar', 'Priya d/o Selvam'. Or simpler form like 'Shivani Iyer'.",
        "phone_format": "Use the MANDATORY PHONE provided above. Do not invent a different phone.",
        "country_context": "Singapore — Pte Ltd suffix."
    },
    "chinese_my": {
        "name_pattern": "Chinese Malaysian name, similar to SG but may use 'Wong', 'Lee', 'Chong' surnames more commonly.",
        "phone_format": "Use the MANDATORY PHONE provided above. Do not invent a different phone.",
        "country_context": "Malaysia — use Sdn Bhd suffix, Malaysian university names (UM, UKM, UTM, USM)."
    },
    "malay_my": {
        "name_pattern": "Malay Malaysian name with bin/binti patronymic, e.g., 'Mohd Faiz bin Rahman', 'Aisyah binti Yusof'.",
        "phone_format": "Use the MANDATORY PHONE provided above. Do not invent a different phone.",
        "country_context": "Malaysia — Sdn Bhd suffix."
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 💎 NAME POOLS — Prevent mode collapse (no more "all Ahmads"!)
# ════════════════════════════════════════════════════════════════════════════

NAME_POOLS = {
    "malay_sg_male": [
        "Ahmad bin Abdullah", "Mohd Faiz bin Rahman", "Hakim bin Yusof",
        "Ismail bin Hassan", "Rashid bin Othman", "Khairul bin Anwar",
        "Zulfikar bin Ibrahim", "Faisal bin Mahmud", "Hafiz bin Latif",
        "Idris bin Suleiman", "Nazri bin Ariffin", "Syafiq bin Kamarudin",
        "Iskandar bin Rosli", "Razak bin Salleh", "Amir bin Zainal",
        "Hisham bin Daud", "Asyraf bin Ramli", "Danial bin Hashim",
        "Imran bin Jamal", "Luqman bin Saad",
        # Additional names
        "Rizwan bin Karim", "Faris bin Sulaiman", "Adli bin Ahmad",
        "Iqbal bin Mansor", "Aziz bin Halim", "Shafiq bin Razali",
        "Hairul bin Bakar", "Zulhilmi bin Yusoff", "Ridhwan bin Talib",
        "Aiman bin Salim",
    ],
    "malay_sg_female": [
        "Siti Aishah binti Mohamed", "Nurul Huda binti Ibrahim",
        "Fadhilah binti Othman", "Aida binti Razak", "Suhana binte Salim",
        "Khairany binte Abu Bakar", "Nur Shafiqah binti Yusoff",
        "Aisyah binti Selamat", "Haszelina binte Mohamed",
        "Nadia binti Ismail", "Farhana binti Rosli", "Zarina binti Ali",
        "Shamsiah binte Latif", "Norizan binti Hashim",
        "Roziana binti Ahmad", "Liyana binti Kamal", "Hidayah binti Hamid",
        # Additional names
        "Nuraini binti Mansor", "Suria binte Yaakob", "Zalina binti Hassan",
        "Sakinah binti Bakri", "Hidayu binte Razali", "Rohaya binti Daud",
        "Mariam binti Yusuf", "Sharifah binti Omar", "Norhanim binti Sulaiman",
        "Fauziah binti Anwar",
    ],
    "chinese_sg": [
        "Tan Wei Ming", "Lim Hui Ling", "Wong Kah Seng", "Lee Jia Min",
        "Chua Boon Heng", "Ng Si Hui", "Goh Yong Hao", "Teo Mei Ling",
        "Ong Jin Hao", "Sim Pei Lin", "Khoo Wen Jie", "Foo Hui Min",
        "Jeremy Tan Kah Wei", "Vanessa Chua Li Ying", "Marcus Lim Jun Wei",
        "Cheryl Ng Hui Yi", "Benjamin Goh Wei Liang", "Stephanie Lee Mei Hui",
        "Daryl Wong Kai Xuan", "Rachel Teo Shi Min", "Kelvin Tan Yong Sheng",
        "Melissa Chua Wei Ting", "Nicholas Ong Jia Hao", "Janice Sim Hui Min",
        # Additional names for diversity (expanded pool prevents collisions)
        "Aaron Chong Jun Hao", "Bryan Tan Yi Jie", "Clarissa Ng Hui Min",
        "Darryl Goh Cheng Wei", "Eunice Lim Pei Shan", "Fiona Lee Wei Ling",
        "Gabriel Chua Jun Kai", "Hannah Teo Jia Hui", "Ivan Wong Boon Keng",
        "Jolene Khoo Yan Ling", "Kenneth Sim Wei Jun", "Lydia Ang Hui Xuan",
        "Mervyn Goh Kah Yong", "Natalie Foo Jia Yi", "Oliver Tan Wei Hao",
        "Patricia Chen Mei Yi", "Quinn Lim Jia Wei", "Reuben Ng Cheng Kang",
        "Serena Wong Hui Qi", "Terrence Tay Jun Wei", "Ursula Liew Pei Ling",
        "Vincent Chua Yong Sheng", "Wendy Ong Hui Min", "Xavier Goh Jun Yu",
        "Yvonne Lim Wei Ting", "Zachary Tan Kah Hao",
        "Adeline Yeo Hui Lin", "Brendan Sim Jia Wei", "Charlene Toh Mei Xin",
        "Donovan Heng Kai Wen", "Eileen Quek Pei Yi",
    ],
    "indian_sg": [
        "Rajesh s/o Kumar", "Priya d/o Selvam", "Shivani Iyer",
        "Karthik Murugesan", "Lakshmi d/o Krishnan", "Arjun s/o Ramesh",
        "Deepa Pillai", "Vijay s/o Subramaniam", "Anjali Nair",
        "Suresh Kumar", "Meera d/o Raman", "Ashwin s/o Velu",
        "Divya Menon", "Rohit Sharma", "Kavitha d/o Devaraj",
        "Pradeep s/o Annamalai", "Lalitha D/O Thamara Selvan",
        "Murugesan Manikandan", "Durkeswari d/o Shanmugam",
    ],
    "chinese_my": [
        "Wong Yin Mun", "Lee Chee Keong", "Chong Hui Min", "Lim Boon Hock",
        "Tan Wei Loon", "Chan Mei Fang", "Goh Kok Wai", "Ng Siew Lin",
        "Ooi Jia Hui", "Yap Cheng Hoe", "Sia Hui Yen", "Khoo Beng Hooi",
        "Lim Mei Xuan", "Chew Wai Keat", "Loke Yi Ling", "Teo Boon Seng",
    ],
    "malay_my_male": [
        "Mohd Hafiz bin Rahim", "Azizul bin Sharif", "Hakim bin Rosli",
        "Faisal bin Mahmud", "Iskandar bin Daud", "Khairul bin Sulaiman",
        "Imran bin Halim", "Syamsul bin Razali", "Zaki bin Aziz",
        "Hamzah bin Yusof", "Adam bin Ridzuan",
    ],
    "malay_my_female": [
        "Aisyah binti Yusof", "Nur Liyana binti Kamal", "Farah binti Hassan",
        "Sakinah binti Latif", "Hidayah binti Mansor", "Aina binti Razak",
        "Nadia binti Salim", "Fatimah binti Ibrahim",
    ],
}


def pick_random_name(ethnicity: str, rng: random.Random) -> Tuple[str, str]:
    """Pick a random name from the pool. Returns (full_name, gender_hint)."""
    if ethnicity == "malay_sg":
        is_male = rng.random() < 0.5
        pool = NAME_POOLS["malay_sg_male"] if is_male else NAME_POOLS["malay_sg_female"]
        return rng.choice(pool), "male" if is_male else "female"
    elif ethnicity == "malay_my":
        is_male = rng.random() < 0.5
        pool = NAME_POOLS["malay_my_male"] if is_male else NAME_POOLS["malay_my_female"]
        return rng.choice(pool), "male" if is_male else "female"
    elif ethnicity in ("chinese_sg", "chinese_my", "indian_sg"):
        return rng.choice(NAME_POOLS[ethnicity]), "any"
    return "", "any"


# ════════════════════════════════════════════════════════════════════════════
# 📱 PHONE GENERATORS — prevent Qwen from copying example phones from prompt
# ════════════════════════════════════════════════════════════════════════════

def generate_phone(country: str, rng: random.Random) -> str:
    """Generate a realistic, UNIQUE phone number for SG or MY.

    SG format: +65 8XXX XXXX or +65 9XXX XXXX (always 8 digits after +65)
    MY format: +60 1X-XXX XXXX (X = 0-9, mobile prefix)
    """
    if country == "sg":
        # SG mobile: starts with 8 or 9
        prefix = rng.choice(["8", "9"])
        # Build 7 more random digits, broken into XXX XXXX
        rest = "".join(str(rng.randint(0, 9)) for _ in range(7))
        # Avoid the dreaded "123 4567" pattern that Qwen kept copying
        while rest in ("1234567", "0000000", "1111111", "9999999"):
            rest = "".join(str(rng.randint(0, 9)) for _ in range(7))
        return f"+65 {prefix}{rest[:3]} {rest[3:]}"
    elif country == "my":
        # MY mobile: +60 1X-YYY YYYY (X = 0-9 for celcom/digi/maxis/u mobile)
        x = rng.randint(0, 9)
        rest = "".join(str(rng.randint(0, 9)) for _ in range(7))
        while rest in ("3456789", "0000000", "1234567"):
            rest = "".join(str(rng.randint(0, 9)) for _ in range(7))
        return f"+60 1{x}-{rest[:3]} {rest[3:]}"
    return ""


def generate_email(name: str, rng: random.Random) -> str:
    """Generate a plausible email from the assigned name."""
    if not name:
        return ""
    # Take 1-2 words from the name
    # Strip patronymics like "bin", "binti", "s/o", "d/o"
    clean = name.lower()
    for skip in ["bin ", "binti ", "binte ", "s/o ", "d/o ", "s/o", "d/o"]:
        clean = clean.replace(skip, " ")
    words = [w for w in clean.split() if len(w) >= 2 and w.isalpha()]
    if not words:
        return f"candidate{rng.randint(100, 9999)}@gmail.com"

    # Combine first 1-2 names
    style = rng.choice(["dotted", "joined", "first_only", "numbered"])
    if style == "dotted":
        local = ".".join(words[:2]) if len(words) >= 2 else words[0]
    elif style == "joined":
        local = "".join(words[:2]) if len(words) >= 2 else words[0]
    elif style == "first_only":
        local = words[0]
    else:  # numbered
        local = words[0] + str(rng.randint(1, 99))

    domain = rng.choice(["gmail.com", "outlook.com", "yahoo.com.sg", "hotmail.com"])
    return f"{local}@{domain}"


# ════════════════════════════════════════════════════════════════════════════
# 🏦 BANK PICKERS — Pull specific items from content banks
# ════════════════════════════════════════════════════════════════════════════

# Track which fields had to fall back to Qwen invention
FALLBACK_LOG = {"companies": 0, "schools": 0, "certs": 0, "skills": 0, "industry": 0, "function": 0}


def pick_companies(country: str, industry: str, n: int, rng: random.Random) -> List[str]:
    """Pull n company names from the appropriate bank."""
    bank_key = "sg_companies" if country == "sg" else "my_companies"
    bank = BANKS.get(bank_key, {})
    pool = bank.get(industry, [])
    if not pool:
        FALLBACK_LOG["companies"] += 1
        return []  # Empty signal: let Qwen invent
    return rng.sample(pool, min(n, len(pool)))


def pick_schools(country: str, education_path: str, rng: random.Random) -> List[str]:
    """Pick schools matching the education path."""
    bank_key = "sg_schools" if country == "sg" else "my_schools"
    bank = BANKS.get(bank_key, {})
    if not bank:
        FALLBACK_LOG["schools"] += 1
        return []

    schools = []
    if country == "sg":
        if education_path == "polytechnic_then_uni":
            polys = bank.get("polytechnics", [])
            unis = bank.get("universities", [])
            if polys:
                schools.append(rng.choice(polys))
            if unis:
                schools.append(rng.choice(unis))
        elif education_path == "uni_local":
            unis = bank.get("universities", [])
            if unis:
                schools.append(rng.choice(unis))
        elif education_path == "ITE_then_uni":
            ites = bank.get("ITE", [])
            polys = bank.get("polytechnics", [])
            unis = bank.get("universities", [])
            if ites:
                schools.append(rng.choice(ites))
            if polys:
                schools.append(rng.choice(polys))
            if unis:
                schools.append(rng.choice(unis))
        elif education_path == "diploma_only":
            polys = bank.get("polytechnics", [])
            if polys:
                schools.append(rng.choice(polys))
    else:  # my
        if education_path in ("uni_local", "uni_overseas"):
            unis = bank.get("public_universities", []) + bank.get("private_universities", [])
            if unis:
                schools.append(rng.choice(unis))
        elif education_path == "diploma_only":
            polys = bank.get("polytechnics", [])
            if polys:
                schools.append(rng.choice(polys))

    if not schools:
        FALLBACK_LOG["schools"] += 1
    return schools


def pick_certifications(industry: str, n: int, rng: random.Random) -> List[str]:
    """Pick relevant certifications from the cert bank.
    Matches the actual bank structure: flat keys like 'finance_sg_cmfas', 'tech_aws', etc.
    """
    bank = BANKS.get("certifications", {})
    if not bank:
        FALLBACK_LOG["certs"] += 1
        return []

    # 💎 Match the bank's flat key structure (e.g. finance_sg_cmfas, tech_aws, tech_azure)
    industry_prefixes = {
        "finance": ["finance_sg_cmfas", "finance_sg_ibf", "finance_general"],
        "tech": ["tech_aws", "tech_azure", "tech_gcp", "tech_other"],
        "healthcare": ["healthcare_general", "healthcare"],
    }
    relevant_keys = industry_prefixes.get(industry, [])

    pool = []
    for key in relevant_keys:
        items = bank.get(key, [])
        if isinstance(items, list):
            pool.extend(items)

    # Add a sprinkle of generals (PMP, ITIL, etc. apply to many roles)
    pool.extend(bank.get("general", []))

    if not pool:
        FALLBACK_LOG["certs"] += 1
        return []

    # 💎 BIAS: for finance, ensure CMFAS gets picked (it's the SG-critical cert)
    # For tech, ensure at least 1 AWS cert
    must_include = []
    if industry == "finance":
        cmfas_pool = bank.get("finance_sg_cmfas", [])
        if cmfas_pool:
            # Pick 1-2 CMFAS modules MANDATORY
            must_include.extend(rng.sample(cmfas_pool, min(2, len(cmfas_pool))))
    elif industry == "tech":
        aws_pool = bank.get("tech_aws", [])
        if aws_pool:
            must_include.append(rng.choice(aws_pool))

    # Build final list: must-includes first, then random from remaining pool
    remaining_pool = [c for c in pool if c not in must_include]
    extras_needed = max(0, n - len(must_include))
    extras = rng.sample(remaining_pool, min(extras_needed, len(remaining_pool))) if remaining_pool else []
    result = must_include + extras
    return result[:n]


def pick_skills(industry: str, hard_n: int, soft_n: int,
                rng: random.Random) -> Tuple[List[str], List[str]]:
    """Pick hard + soft skills for an industry."""
    bank = BANKS.get("skills_by_industry", {})
    industry_bank = bank.get(industry, {})
    hard_pool = industry_bank.get("hard", [])
    soft_pool = industry_bank.get("soft", [])

    if not hard_pool and not soft_pool:
        FALLBACK_LOG["skills"] += 1
        return [], []

    hard = rng.sample(hard_pool, min(hard_n, len(hard_pool))) if hard_pool else []
    soft = rng.sample(soft_pool, min(soft_n, len(soft_pool))) if soft_pool else []
    return hard, soft


def pick_industry_label(industry: str, rng: random.Random) -> List[str]:
    """Pick the formal Industry label(s) from the industries bank."""
    bank = BANKS.get("industries", {})
    industries_list = bank.get("industries", [])
    if not industries_list:
        FALLBACK_LOG["industry"] += 1
        return []

    # Map our internal industry key to formal industry name (best match)
    name_map = {
        "finance": "Banking & Finance",
        "tech": "Technology",
        "healthcare": "Healthcare",
        "hospitality": "Hospitality",
        "education": "Education",
        "manufacturing": "Manufacturing",
        "logistics": "Logistics",
        "government": "Government",
        "fmcg": "FMCG",
    }
    target = name_map.get(industry, industry)
    # Find best matching industry entry
    for entry in industries_list:
        if isinstance(entry, dict):
            name = entry.get("name", "")
            if target.lower() in name.lower() or name.lower() in target.lower():
                subcats = entry.get("subcategories", [])
                # Return formal industry name + 1 subcategory if exists
                if subcats:
                    return [name, rng.choice(subcats)]
                return [name]
    FALLBACK_LOG["industry"] += 1
    return []


def pick_function_label(industry: str, role_hint: str, rng: random.Random) -> str:
    """Pick a Function label that matches the industry/role."""
    bank = BANKS.get("functions", {})
    functions_list = bank.get("functions", [])
    if not functions_list:
        FALLBACK_LOG["function"] += 1
        return ""

    # Heuristic mapping industry → function
    industry_to_function = {
        "finance": ["Finance & Accounting", "Sales & Business Development"],
        "tech": ["Engineering", "Product Management", "Data & Analytics"],
        "healthcare": ["Operations", "Research & Development"],
        "hospitality": ["Operations", "Customer Success"],
        "education": ["Research & Development", "Administration"],
        "manufacturing": ["Engineering", "Operations"],
        "logistics": ["Operations"],
        "government": ["Administration", "Legal & Compliance"],
        "fmcg": ["Marketing & Communications", "Sales & Business Development"],
    }
    candidate_names = industry_to_function.get(industry, [])
    if not candidate_names:
        FALLBACK_LOG["function"] += 1
        return ""

    chosen_name = rng.choice(candidate_names)
    # Verify it exists in the bank
    for entry in functions_list:
        if isinstance(entry, dict) and entry.get("name", "").lower() == chosen_name.lower():
            return chosen_name
    return chosen_name  # use it anyway even if not exact in bank



SENIORITY_GUIDANCE = {
    "fresh_grad":    "0-1 year experience. 1 job (internship or first role). Recently graduated.",
    "junior_2yr":    "2-3 years experience. 1-2 jobs.",
    "mid_5yr":       "4-6 years experience. 2-3 jobs. Some leadership of small projects.",
    "senior_10yr":   "8-12 years experience. 3-4 jobs. Team lead or manager.",
    "lead_15yr":     "15+ years experience. 4-6 jobs. Senior leadership, principal engineer, or director level.",
}

INDUSTRY_GUIDANCE = {
    "finance":       "Banking, insurance, investment. Mention CMFAS modules (5, 9, 9A, HI, M9, M5) if Singapore.",
    "tech":          "Software, cloud, data, AI. Mention specific technologies (Python, Kubernetes, AWS, etc.).",
    "healthcare":    "Hospitals, clinics, medical devices, pharma.",
    "hospitality":   "Hotels, F&B, tourism, events.",
    "education":     "Schools, tuition centers, universities, edtech.",
    "manufacturing": "Electronics, semiconductors, food production, automotive.",
    "logistics":     "Shipping, freight, supply chain, warehousing.",
    "government":    "Civil service, statutory boards, public sector.",
    "fmcg":          "Consumer goods, retail, e-commerce.",
}

EDUCATION_GUIDANCE = {
    "polytechnic_then_uni": "Diploma from a Singapore polytechnic (Nanyang, Singapore, Temasek, Republic, or Ngee Ann Polytechnic), then bachelor's from local university (NUS, NTU, SMU, SUSS, SIT, SUTD).",
    "uni_local":            "Local university degree only (NUS, NTU, SMU, SUSS, SIT, SUTD for SG; UM, UKM, UTM, USM for MY).",
    "uni_overseas":         "University degree from overseas (Australia, UK, US — e.g., University of Melbourne, Imperial College, MIT).",
    "ITE_then_uni":         "ITE (Institute of Technical Education) Nitec/Higher Nitec, then polytechnic diploma, then bachelor's.",
    "diploma_only":         "Polytechnic diploma only, no university degree.",
    "professional_cert_only": "No university — only professional certifications (CFA, CPA, PMP, AWS certs, etc.).",
}


def build_generation_prompt(ethnicity: str, industry: str, seniority: str,
                            education_path: str, is_hard: bool = False,
                            assigned_name: str = "", gender_hint: str = "",
                            assigned_phone: str = "", assigned_email: str = "",
                            bank_companies: Optional[List[str]] = None,
                            bank_schools: Optional[List[str]] = None,
                            bank_certs: Optional[List[str]] = None,
                            bank_hard_skills: Optional[List[str]] = None,
                            bank_soft_skills: Optional[List[str]] = None,
                            bank_industry: Optional[List[str]] = None,
                            bank_function: str = "") -> str:
    """Build the prompt sent to the teacher LLM."""
    eth_info = ETHNICITY_GUIDANCE.get(ethnicity, {})
    sen_info = SENIORITY_GUIDANCE.get(seniority, "")
    ind_info = INDUSTRY_GUIDANCE.get(industry, "")
    edu_info = EDUCATION_GUIDANCE.get(education_path, "")

    schema_json = json.dumps({
        "Name": "",
        "Phone": "",
        "Email": "",
        "Current Location": "",
        "Current Company": "",
        "Current Title": "",
        "Summary": "",
        "Function": "",
        "Industry": [],
        "Work Experience": [{"company": "", "title": "", "from": "", "to": "", "responsibility": []}],
        "Education": [{"school": "", "major": "", "degree": "", "dates": ""}],
        "Certifications": [],
        "hard_skills/tags": [],
        "soft_skills/skills": [],
        "Language Skills": [],
    }, indent=2)

    hard_case_extra = ""
    if is_hard:
        hard_case_extra = """
EDGE CASE: This is a HARD case. Include at least one of:
- A career break (gap year, parental leave, sabbatical)
- Concurrent freelance + full-time roles
- A role at an NGO or volunteer organization
- Multiple rapid promotions in same company
- An unusual layout (bilingual EN+Mandarin or EN+Malay code-switching in summary)
"""

    # 💎 INJECT THE ASSIGNED NAME + PHONE + EMAIL — prevents Qwen defaults
    name_directive = ""
    if assigned_name:
        gender_note = f" (gender: {gender_hint})" if gender_hint and gender_hint != "any" else ""
        name_directive = f"""
MANDATORY IDENTITY (use EXACTLY these values, do not modify):
  Name:  {assigned_name}{gender_note}
  Phone: {assigned_phone}
  Email: {assigned_email}
"""

    # 💎 INJECT BANK INGREDIENTS — Qwen uses these REAL items
    bank_directive = ""
    bank_lines = []
    if bank_companies:
        bank_lines.append(f"COMPANIES (use 2-3 of these, in chronological order, latest first):")
        for c in bank_companies:
            bank_lines.append(f"  - {c}")
    if bank_schools:
        bank_lines.append(f"\nSCHOOLS (use these in your Education section):")
        for s in bank_schools:
            bank_lines.append(f"  - {s}")
    if bank_certs:
        bank_lines.append(f"\nCERTIFICATIONS (use 3-5 of these, with FULL official names):")
        for c in bank_certs:
            bank_lines.append(f"  - {c}")
    if bank_hard_skills:
        bank_lines.append(f"\nHARD SKILLS (use 8-12 of these in the skills section):")
        for s in bank_hard_skills:
            bank_lines.append(f"  - {s}")
    if bank_soft_skills:
        bank_lines.append(f"\nSOFT SKILLS (use 4-6 of these):")
        for s in bank_soft_skills:
            bank_lines.append(f"  - {s}")
    if bank_industry:
        bank_lines.append(f"\nINDUSTRY LABEL (set the Industry JSON field to this list):")
        bank_lines.append(f"  {bank_industry}")
    if bank_function:
        bank_lines.append(f"\nFUNCTION LABEL (set the Function JSON field to this string):")
        bank_lines.append(f"  {bank_function}")

    if bank_lines:
        bank_directive = "\n💎 USE THESE REAL INGREDIENTS (don't invent SG/MY names if these are provided):\n" + "\n".join(bank_lines) + "\n"

    return f"""You are generating a SYNTHETIC RESUME for training a Singapore/Malaysia resume extraction model.
{name_directive}{bank_directive}
ETHNICITY/NAME PATTERN: {eth_info.get('name_pattern', '')}
PHONE FORMAT: {eth_info.get('phone_format', '')}
COUNTRY CONTEXT: {eth_info.get('country_context', '')}

SENIORITY: {sen_info}
INDUSTRY: {industry} — {ind_info}
EDUCATION PATH: {edu_info}
{hard_case_extra}

INSTRUCTIONS:
1. Use the MANDATORY IDENTITY (name + phone + email) EXACTLY as given above. Do not modify them.
2. The phone number MUST appear in the resume header verbatim. The email MUST appear verbatim.
3. If REAL INGREDIENTS are provided above, USE them (don't invent your own SG/MY companies/schools/certs/skills).
4. If a category has no ingredients listed, you may invent realistic SG/MY equivalents (with proper Pte Ltd / Sdn Bhd suffix).
5. Include 2-5 work experiences matching the seniority level.
6. Include 1-3 education entries matching the education path.
7. Set the "Industry" JSON field as a list (e.g., ["Banking & Finance", "Private Banking"]).
8. Set the "Function" JSON field as a single string (e.g., "Finance & Accounting").
9. Use the EXACT Industry label provided above — do not change "Banking & Finance" to just "Banking".

OUTPUT FORMAT (CRITICAL):
Return TWO sections separated by ===SPLIT===

SECTION 1: The raw resume text (as if pasted from PDF). Plain text, no markdown. Include sections like SUMMARY, EXPERIENCE, EDUCATION, CERTIFICATIONS, SKILLS.

===SPLIT===

SECTION 2: The ground truth JSON extraction matching this schema EXACTLY:
{schema_json}

Use the canonical schema keys EXACTLY as shown (Title Case, with spaces).
Do not add any text before or after the two sections.
"""


# ════════════════════════════════════════════════════════════════════════════
# 🦙 OLLAMA CALL
# ════════════════════════════════════════════════════════════════════════════

def call_ollama(prompt: str, model: str = LOCAL_MODEL, timeout: int = 300) -> Optional[str]:
    """Call local Ollama for generation."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 3000,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"   ⚠️  Ollama error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# 💎 NOTE: Gemini polish step removed.
# Gemini is now used ONCE to build content banks via build_content_banks.py
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# 🔍 PARSING & VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def parse_llm_output(raw: str) -> Optional[Tuple[str, Dict]]:
    """Parse the LLM output into (resume_text, ground_truth_dict).
    Returns None if parsing fails."""
    if not raw or "===SPLIT===" not in raw:
        return None

    parts = raw.split("===SPLIT===", maxsplit=1)
    if len(parts) != 2:
        return None

    resume_text = parts[0].strip()
    json_part = parts[1].strip()

    # Try to find a JSON block (might be wrapped in ```json...```)
    json_match = re.search(r'\{[\s\S]*\}', json_part)
    if not json_match:
        return None

    try:
        ground_truth = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(ground_truth, dict):
        return None

    if len(resume_text) < 200:
        return None

    return resume_text, ground_truth


def validate_against_schema(gt: Dict) -> Tuple[bool, str]:
    """Validate that ground truth has the canonical schema fields."""
    required_fields = ["Name", "Phone", "Email",
                       "Work Experience", "Education"]
    for f in required_fields:
        if f not in gt:
            return False, f"Missing required field: {f}"

    # Work Experience must be a list of dicts with the right sub-fields
    we = gt.get("Work Experience", [])
    if not isinstance(we, list) or len(we) == 0:
        return False, "Work Experience must be non-empty list"
    for i, job in enumerate(we):
        if not isinstance(job, dict):
            return False, f"Work Experience[{i}] not a dict"
        for sub in ["company", "title"]:
            if sub not in job:
                return False, f"Work Experience[{i}] missing {sub}"

    edu = gt.get("Education", [])
    if not isinstance(edu, list) or len(edu) == 0:
        return False, "Education must be non-empty list"

    return True, ""


def normalize_to_canonical(gt: Dict) -> Dict:
    """Ensure all canonical schema keys exist, defaults if missing."""
    canonical = {}
    for key in CANONICAL_SCHEMA_KEYS:
        if key in gt:
            canonical[key] = gt[key]
        elif key in ["Industry", "Language Skills", "Work Experience",
                     "Project Experience", "Education", "Certifications",
                     "hard_skills/tags", "soft_skills/skills", "Achievements"]:
            canonical[key] = gt.get(key, []) or []
        else:
            canonical[key] = gt.get(key, "") or ""
    return canonical


# ════════════════════════════════════════════════════════════════════════════
# 🔒 LEAKAGE CHECK
# ════════════════════════════════════════════════════════════════════════════

def load_golden_blacklist(golden_path: Path) -> set:
    """Load names/phones/emails from golden test set to PREVENT leakage."""
    blacklist = set()
    if not golden_path.exists():
        print(f"⚠️  No golden test set found at {golden_path} — leakage check disabled")
        return blacklist

    with open(golden_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                gt = record.get("ground_truth", {})
                for key in ["name", "phone", "email", "Name", "Phone", "Email"]:
                    v = gt.get(key, "")
                    if v and isinstance(v, str) and len(v) >= 5:
                        blacklist.add(v.lower().strip())
            except json.JSONDecodeError:
                continue
    print(f"🔒 Loaded {len(blacklist)} blacklisted identifiers from golden set")
    return blacklist


def is_leaked(gt: Dict, blacklist: set) -> bool:
    """Check if any identifying field in generated resume matches golden set."""
    for key in ["Name", "Phone", "Email"]:
        v = gt.get(key, "")
        if v and isinstance(v, str):
            if v.lower().strip() in blacklist:
                return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# 💾 PROGRESS PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════

def load_progress(progress_path: Path) -> Dict:
    """Load progress checkpoint for resume on crash."""
    if progress_path.exists():
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed_cells": [], "generated_count": 0, "rejected_count": 0}


def save_progress(progress_path: Path, progress: Dict):
    """Save progress checkpoint."""
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# 🎬 MAIN GENERATION LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    global BANKS
    print("╔" + "═" * 68 + "╗")
    print("║" + "  💎  STAGE 0 LITE — 100 SYNTHETIC SG/MY RESUMES  💎  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print("\n📚 Loading content banks...")
    BANKS = load_content_banks()
    if BANKS:
        print(f"   ✨ {len(BANKS)} banks loaded\n")
    else:
        print(f"   ⚠️  No banks — Qwen will invent everything (lower quality)\n")

    output_dir = Path("data_gen")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "synthetic_lite.jsonl"
    progress_path = output_dir / "synthetic_lite_progress.json"
    rejected_path = output_dir / "synthetic_lite_rejected.jsonl"

    # Load leakage blacklist
    golden_path = Path("validation/golden_sg_my.jsonl")
    blacklist = load_golden_blacklist(golden_path)

    # Resume from checkpoint
    progress = load_progress(progress_path)
    completed_cells = set(tuple(c) for c in progress.get("completed_cells", []))
    if completed_cells:
        print(f"📂 Resuming: {len(completed_cells)} cells already done")

    total_target = sum(count for _, _, _, _, count, _ in DIVERSITY_MATRIX)
    print(f"🎯 Target: {total_target} synthetic resumes\n")

    output_file = open(output_path, 'a', encoding='utf-8')
    rejected_file = open(rejected_path, 'a', encoding='utf-8')

    # 💎 Track used names to ensure diversity (no repeats!)
    used_names = set()
    rng = random.Random(42)  # reproducible randomness

    try:
        generated_so_far = progress.get("generated_count", 0)
        rejected_so_far = progress.get("rejected_count", 0)

        for cell_idx, (ethnicity, industry, seniority, edu_path, count, is_hard) in enumerate(DIVERSITY_MATRIX):
            cell_key = (ethnicity, industry, seniority, edu_path)
            if cell_key in completed_cells:
                continue

            engine = "Qwen-32B + Banks" if BANKS else "Qwen-32B (no banks)"
            print(f"🎨 Cell {cell_idx+1}/{len(DIVERSITY_MATRIX)}: "
                  f"{ethnicity} | {industry} | {seniority} | {edu_path} "
                  f"→ {count} resumes via {engine}")

            # Resolve country from ethnicity tag
            country = "sg" if ethnicity.endswith("_sg") else "my"

            for i in range(count):
                # 💎 Pick a unique name from the pool (no duplicates!)
                assigned_name, gender_hint = pick_random_name(ethnicity, rng)
                attempts = 0
                while assigned_name in used_names and attempts < 20:
                    assigned_name, gender_hint = pick_random_name(ethnicity, rng)
                    attempts += 1
                used_names.add(assigned_name)

                # 💎 Generate a unique phone (no more "+65 8123 4567" copies!)
                assigned_phone = generate_phone(country, rng)
                # 💎 Generate an email tied to the name
                assigned_email = generate_email(assigned_name, rng)

                # 💎 Pull ingredients from banks
                bank_companies = pick_companies(country, industry, n=3, rng=rng)
                bank_schools = pick_schools(country, edu_path, rng=rng)
                bank_certs = pick_certifications(industry, n=5, rng=rng)
                bank_hard, bank_soft = pick_skills(industry, hard_n=12, soft_n=6, rng=rng)
                bank_industry = pick_industry_label(industry, rng=rng)
                bank_function = pick_function_label(industry, "", rng=rng)

                prompt = build_generation_prompt(
                    ethnicity, industry, seniority, edu_path,
                    is_hard=is_hard,
                    assigned_name=assigned_name,
                    gender_hint=gender_hint,
                    assigned_phone=assigned_phone,
                    assigned_email=assigned_email,
                    bank_companies=bank_companies,
                    bank_schools=bank_schools,
                    bank_certs=bank_certs,
                    bank_hard_skills=bank_hard,
                    bank_soft_skills=bank_soft,
                    bank_industry=bank_industry,
                    bank_function=bank_function,
                )
                start_time = time.time()

                # 🦙 Generate with local Qwen 32B (no Gemini polish — using banks instead)
                raw = call_ollama(prompt)

                elapsed = time.time() - start_time

                if raw is None:
                    rejected_so_far += 1
                    rejected_file.write(json.dumps({
                        "reason": "LLM call failed",
                        "cell": cell_key,
                    }) + "\n")
                    rejected_file.flush()
                    print(f"   ✗ Failed ({elapsed:.1f}s)")
                    continue

                parsed = parse_llm_output(raw)
                if parsed is None:
                    rejected_so_far += 1
                    rejected_file.write(json.dumps({
                        "reason": "Parse failed",
                        "cell": cell_key,
                        "raw_preview": raw[:500] if raw else "",
                    }) + "\n")
                    rejected_file.flush()
                    print(f"   ✗ Parse fail ({elapsed:.1f}s)")
                    continue

                resume_text, ground_truth = parsed

                # Schema validation
                valid, err = validate_against_schema(ground_truth)
                if not valid:
                    rejected_so_far += 1
                    rejected_file.write(json.dumps({
                        "reason": f"Schema invalid: {err}",
                        "cell": cell_key,
                        "ground_truth_preview": {k: str(v)[:100] for k, v in ground_truth.items()},
                    }) + "\n")
                    rejected_file.flush()
                    print(f"   ✗ Schema fail: {err} ({elapsed:.1f}s)")
                    continue

                # Leakage check
                if is_leaked(ground_truth, blacklist):
                    rejected_so_far += 1
                    rejected_file.write(json.dumps({
                        "reason": "LEAKAGE — matched golden test set identifier",
                        "cell": cell_key,
                        "name": ground_truth.get("Name", ""),
                    }) + "\n")
                    rejected_file.flush()
                    print(f"   ✗ Leakage! ({elapsed:.1f}s)")
                    continue

                # Normalize and save
                ground_truth = normalize_to_canonical(ground_truth)

                record = {
                    "id": f"synth_lite_{generated_so_far + 1:05d}",
                    "metadata": {
                        "ethnicity_tag": ethnicity,
                        "industry": industry,
                        "seniority": seniority,
                        "education_path": edu_path,
                        "is_hard_case": is_hard,
                        "source": engine,
                        "generated_at": datetime.now().isoformat(),
                    },
                    "raw_text": resume_text,
                    "ground_truth": ground_truth,
                }

                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                generated_so_far += 1

                name_preview = ground_truth.get("Name", "?")[:30]
                print(f"   ✓ #{generated_so_far:3d}: {name_preview:30s} ({elapsed:.1f}s)")

                # 💎 Every 10 resumes, show name diversity stats
                if generated_so_far % 10 == 0:
                    print(f"\n   📊 Progress: {generated_so_far} generated, {len(used_names)} unique names used")
                    if generated_so_far - len(used_names) > 3:
                        print(f"   ⚠️  WARNING: name diversity dropping! Consider stopping.\n")
                    else:
                        print(f"   ✨ Name diversity looking good!\n")

            # Mark cell complete
            completed_cells.add(cell_key)
            progress["completed_cells"] = [list(c) for c in completed_cells]
            progress["generated_count"] = generated_so_far
            progress["rejected_count"] = rejected_so_far
            save_progress(progress_path, progress)

    finally:
        output_file.close()
        rejected_file.close()

    # ─── Final summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  ✨ GENERATION COMPLETE ✨".center(70))
    print("═" * 70)
    print(f"   Generated:  {generated_so_far}")
    print(f"   Rejected:   {rejected_so_far}")
    print(f"   Output:     {output_path}")
    print(f"   Rejected log: {rejected_path}")
    print(f"\n📊 Bank fallback log (how often Qwen had to invent items):")
    for key, count in FALLBACK_LOG.items():
        if count > 0:
            print(f"   - {key:12s}: {count} times")
    print(f"\n💋 Next step: Hand-inspect 10 random samples before fine-tuning!")


if __name__ == "__main__":
    main()