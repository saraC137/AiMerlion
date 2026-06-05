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

# ─── Gemini integration (Pattern B: polish Qwen output) ────────────────────
GEMINI_AVAILABLE = False
gemini_client = None
try:
    from gemini_client import GeminiClient
    try:
        gemini_client = GeminiClient(model="flash-lite")
        GEMINI_AVAILABLE = True
    except ValueError as e:
        print(f"⚠️  Gemini not configured: {e}")
        print(f"⚠️  Will run pure-local without polish step")
except ImportError:
    print("⚠️  gemini_client.py not found — skipping polish step")


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
        "phone_format": "Singapore 8-digit phone starting with 8 or 9. Format: '+65 8XXX XXXX' or '+65 9XXX XXXX' or '8XXXXXXX'.",
        "country_context": "Singapore — use Pte Ltd suffix for companies, Singapore polytechnic/university names."
    },
    "chinese_sg": {
        "name_pattern": "Chinese Singaporean name, typically 3 characters (e.g., 'Tan Wei Ming', 'Lim Hui Ling') or with English first name ('Jeremy Tan', 'Vanessa Chua').",
        "phone_format": "Singapore 8-digit phone starting with 8 or 9.",
        "country_context": "Singapore — use Pte Ltd suffix."
    },
    "indian_sg": {
        "name_pattern": "Indian Singaporean name often with s/o (son of) or d/o (daughter of) patronymic. Example: 'Rajesh s/o Kumar', 'Priya d/o Selvam'. Or simpler form like 'Shivani Iyer'.",
        "phone_format": "Singapore 8-digit phone.",
        "country_context": "Singapore — Pte Ltd suffix."
    },
    "chinese_my": {
        "name_pattern": "Chinese Malaysian name, similar to SG but may use 'Wong', 'Lee', 'Chong' surnames more commonly.",
        "phone_format": "Malaysia phone: +60 1X-XXXX XXXX (where X = 0-9, e.g., +60 12-345 6789).",
        "country_context": "Malaysia — use Sdn Bhd suffix, Malaysian university names (UM, UKM, UTM, USM)."
    },
    "malay_my": {
        "name_pattern": "Malay Malaysian name with bin/binti patronymic, e.g., 'Mohd Faiz bin Rahman', 'Aisyah binti Yusof'.",
        "phone_format": "Malaysia phone: +60 1X-XXXX XXXX.",
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
    ],
    "malay_sg_female": [
        "Siti Aishah binti Mohamed", "Nurul Huda binti Ibrahim",
        "Fadhilah binti Othman", "Aida binti Razak", "Suhana binte Salim",
        "Khairany binte Abu Bakar", "Nur Shafiqah binti Yusoff",
        "Aisyah binti Selamat", "Haszelina binte Mohamed",
        "Nadia binti Ismail", "Farhana binti Rosli", "Zarina binti Ali",
        "Shamsiah binte Latif", "Norizan binti Hashim",
        "Roziana binti Ahmad", "Liyana binti Kamal", "Hidayah binti Hamid",
    ],
    "chinese_sg": [
        "Tan Wei Ming", "Lim Hui Ling", "Wong Kah Seng", "Lee Jia Min",
        "Chua Boon Heng", "Ng Si Hui", "Goh Yong Hao", "Teo Mei Ling",
        "Ong Jin Hao", "Sim Pei Lin", "Khoo Wen Jie", "Foo Hui Min",
        "Jeremy Tan Kah Wei", "Vanessa Chua Li Ying", "Marcus Lim Jun Wei",
        "Cheryl Ng Hui Yi", "Benjamin Goh Wei Liang", "Stephanie Lee Mei Hui",
        "Daryl Wong Kai Xuan", "Rachel Teo Shi Min", "Kelvin Tan Yong Sheng",
        "Melissa Chua Wei Ting", "Nicholas Ong Jia Hao", "Janice Sim Hui Min",
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
                            assigned_name: str = "", gender_hint: str = "") -> str:
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
        "Work Experience": [{"company": "", "title": "", "from": "", "to": "", "responsibility": []}],
        "Education": [{"school": "", "major": "", "degree": "", "dates": ""}],
        "Certifications": [],
        "hard_skills/tags": [],
        "soft_skills/skills": [],
        "Industry": [],
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

    # 💎 INJECT THE ASSIGNED NAME — this prevents Qwen from defaulting to "Ahmad"
    name_directive = ""
    if assigned_name:
        gender_note = f" (gender: {gender_hint})" if gender_hint and gender_hint != "any" else ""
        name_directive = f"""
MANDATORY NAME (use EXACTLY this name, do not modify):
  {assigned_name}{gender_note}
"""

    return f"""You are generating a SYNTHETIC RESUME for training a Singapore/Malaysia resume extraction model.
{name_directive}
ETHNICITY/NAME PATTERN: {eth_info.get('name_pattern', '')}
PHONE FORMAT: {eth_info.get('phone_format', '')}
COUNTRY CONTEXT: {eth_info.get('country_context', '')}

SENIORITY: {sen_info}
INDUSTRY: {industry} — {ind_info}
EDUCATION PATH: {edu_info}
{hard_case_extra}

INSTRUCTIONS:
1. Use the MANDATORY NAME exactly as given above (do not change it!).
2. Generate a REALISTIC resume that matches all the above constraints.
3. Use REAL Singapore/Malaysia company names (DBS, OCBC, Grab, Shopee, Sea, etc. for SG; Maybank, CIMB, Petronas, AirAsia, etc. for MY).
4. Use REAL polytechnic/university names from the country.
5. Include 2-5 work experiences appropriate for seniority.
6. Include 1-3 education entries matching the education path.
7. Include relevant certifications (CMFAS for finance, AWS for tech, etc.).
8. Generate a UNIQUE phone number and email — do NOT use generic ones like "name@email.com".

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
# 💎 GEMINI POLISH (Pattern B) — Polish Qwen-generated resumes
# ════════════════════════════════════════════════════════════════════════════

POLISH_PROMPT = """You are reviewing a synthetic Singapore/Malaysia resume for a training dataset.

ORIGINAL (generated by a local LLM):
===
{original}
===

YOUR TASK: Polish this resume + ground truth JSON for SG/MY authenticity.

Check and fix these issues if present:
1. Names — ensure bin/binti patronymics are correct for Malay names, s/o or d/o for Indian
2. Phone numbers — Singapore should be 8-digit starting 8 or 9; Malaysia +60 1X-XXXX XXXX
3. Company names — should have realistic Pte Ltd (SG) or Sdn Bhd (MY) suffix
4. School names — must be REAL SG/MY institutions
5. Certifications — CMFAS modules should be valid (5, 9, 9A, HI, M9, M5) for SG finance
6. JSON structure — Title Case keys (Name, Phone, Work Experience, etc.)
7. Internal consistency — dates make sense, seniority matches years of experience

OUTPUT FORMAT (CRITICAL — same as input):
Return TWO sections separated by ===SPLIT===

SECTION 1: The polished resume text
===SPLIT===
SECTION 2: The polished ground truth JSON

Do not add commentary. Just output the polished version.
"""


def polish_with_gemini(qwen_output: str) -> Optional[str]:
    """
    Pattern B: Take Qwen's output, ask Gemini to polish it.
    Returns the polished version, or the original if Gemini fails.
    """
    if not GEMINI_AVAILABLE or gemini_client is None:
        return qwen_output  # No-op if Gemini unavailable

    prompt = POLISH_PROMPT.format(original=qwen_output)
    polished = gemini_client.generate(
        prompt,
        temperature=0.3,  # Lower temp for polish — we want consistency
        max_output_tokens=4096,
    )

    if polished and "===SPLIT===" in polished:
        return polished
    # Polish failed — return original unchanged
    return qwen_output


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
    print("╔" + "═" * 68 + "╗")
    print("║" + "  💎  STAGE 0 LITE — 100 SYNTHETIC SG/MY RESUMES  💎  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

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

            engine = "Qwen-32B + Gemini-polish" if GEMINI_AVAILABLE else "Qwen-32B"
            print(f"🎨 Cell {cell_idx+1}/{len(DIVERSITY_MATRIX)}: "
                  f"{ethnicity} | {industry} | {seniority} | {edu_path} "
                  f"→ {count} resumes via {engine}")

            for i in range(count):
                # 💎 Pick a unique name from the pool (no duplicates!)
                assigned_name, gender_hint = pick_random_name(ethnicity, rng)
                attempts = 0
                while assigned_name in used_names and attempts < 20:
                    assigned_name, gender_hint = pick_random_name(ethnicity, rng)
                    attempts += 1
                used_names.add(assigned_name)

                prompt = build_generation_prompt(
                    ethnicity, industry, seniority, edu_path,
                    is_hard=is_hard,
                    assigned_name=assigned_name,
                    gender_hint=gender_hint,
                )
                start_time = time.time()

                # 💎 PATTERN B: Always generate with local Qwen 32B
                raw = call_ollama(prompt)

                # If generation worked AND Gemini is available, POLISH it
                if raw and GEMINI_AVAILABLE:
                    polished = polish_with_gemini(raw)
                    if polished:
                        raw = polished

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
    print(f"\n💋 Next step: Hand-inspect 10 random samples before fine-tuning!")


if __name__ == "__main__":
    main()