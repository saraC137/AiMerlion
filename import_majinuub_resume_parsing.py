"""
import_majinuub_resume_parsing.py

💅✨ FAIRY CODEMOTHER'S MAJINUUB RESUME PARSING IMPORTER ✨💅

Downloads and converts the Majinuub/Resume_Parsing dataset (305 resumes)
from HuggingFace into AiMerlion's ShareGPT JSONL training format —
ready to feed straight into finetune_unsloth.py! 🚀

Dataset source: https://huggingface.co/datasets/Majinuub/Resume_Parsing
License: Apache-2.0 (free for commercial use! 💃)

The Majinuub dataset is a GOLDMINE for resume extraction training:
  - 305 diverse resumes across 40+ countries
  - Rich structured output: names, skills, workplaces, education, certs
  - Chain-of-Thought reasoning included (bonus for CoT training!)
  - Clean JSON output with consistent schema

BUT — it has ZERO Singapore/Malaysia resumes! So this supplements
(not replaces) your precious SG/MY annotated data. Think of it as
hiring backup dancers — they make the show bigger and better, but
YOUR star performers (SG/MY data) still lead the choreography! 💃🌟

Field Mapping (Majinuub → AiMerlion):
  ┌────────────────────────┬────────────────────────────────┐
  │ MAJINUUB FIELD         │ AIMERLION FIELD                │
  ├────────────────────────┼────────────────────────────────┤
  │ FirstName + LastName   │ name                           │
  │ Email                  │ email                          │
  │ Phone                  │ phone                          │
  │ Address/City/Country   │ location                       │
  │ Skill[]                │ hard_skills[]                  │
  │ Workplaces[]           │ experience[]                   │
  │   .JobTitle            │   .title                       │
  │   .Company_Name        │   .company                     │
  │   .From_Date/.To_Date  │   .duration                    │
  │ Study[]                │ education[]                    │
  │   .Degree              │   .degree                      │
  │   .Institution         │   .institution                 │
  │ Certifications[]       │ certifications[]               │
  │ Summary                │ summary                        │
  │ Designation            │ (used for current title)       │
  │ Experience (int)       │ (years of experience)          │
  └────────────────────────┴────────────────────────────────┘

Usage:
    # Download + convert (default: outputs to training_data/ directory)
    python import_majinuub_resume_parsing.py

    # Custom output path
    python import_majinuub_resume_parsing.py --output my_training_data.jsonl

    # Include Chain-of-Thought in system prompt (experimental)
    python import_majinuub_resume_parsing.py --include-cot

    # Preview first N examples without writing files
    python import_majinuub_resume_parsing.py --preview 3

    # Merge with existing training data from your annotations
    python import_majinuub_resume_parsing.py --merge-with train_data.jsonl

    # Use a local file instead of downloading
    python import_majinuub_resume_parsing.py --local-file "cleaned_file (3).json"

Dependencies:
    pip install requests --break-system-packages
    (requests is usually pre-installed, but just in case!)
"""

import os
import sys
import json
import argparse
import random
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import Counter, defaultdict

# ── Optional: requests for downloading ────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# =============================================================================
# 🎨 CONSOLE — Pretty output (matching AiMerlion style)
# =============================================================================

class Console:
    """Pretty console output matching the AiMerlion pipeline style."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def banner(text: str):
        width = 60
        print(f"\n{'=' * width}")
        print(f"  {Console.BOLD}{Console.HEADER}{text}{Console.RESET}")
        print(f"{'=' * width}")

    @staticmethod
    def success(msg: str):
        print(f"  {Console.GREEN}✅ {msg}{Console.RESET}")

    @staticmethod
    def warning(msg: str):
        print(f"  {Console.YELLOW}⚠️  {msg}{Console.RESET}")

    @staticmethod
    def error(msg: str):
        print(f"  {Console.RED}❌ {msg}{Console.RESET}")

    @staticmethod
    def info(msg: str):
        print(f"  {Console.CYAN}💡 {msg}{Console.RESET}")

    @staticmethod
    def stat(label: str, value: Any, indent: int = 2):
        spaces = "  " * indent
        print(f"{spaces}{Console.DIM}{label}:{Console.RESET} {Console.BOLD}{value}{Console.RESET}")

    @staticmethod
    def progress_bar(current: int, total: int, width: int = 30):
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        pct = (current / total * 100) if total > 0 else 0
        print(f"\r  [{bar}] {pct:.0f}% ({current}/{total})", end="", flush=True)
        if current >= total:
            print()  # newline when done


# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 🎓 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HuggingFace download URL for the dataset
DATASET_URL = (
    "https://huggingface.co/datasets/Majinuub/Resume_Parsing/resolve/main/"
    "cleaned_file%20(3).json"
)

# Default output directory (matches AiMerlion convention)
DEFAULT_OUTPUT_DIR = "training_data"

# Candidate ID offset for imported records (avoid collision with real data)
# Your existing imports use 900,000+; we use 950,000+ for Majinuub
CANDIDATE_ID_OFFSET = 950_000

# ── AiMerlion system prompt (MUST match what's in prepare_training_data.py
#    and your Ollama Modelfile — consistency is CRITICAL for training!) ──────
SYSTEM_PROMPT = (
    "You are a precise resume data extraction assistant specializing in "
    "Singapore and Malaysia resumes. Extract the requested information "
    "and return valid JSON only. Handle 8-digit phone numbers (+65/+60), "
    "date-first experience formats, local company names, and multilingual "
    "content accurately."
)

# ── Instruction template (MUST match prepare_training_data.py) ─────────────
INSTRUCTION_TEMPLATE = (
    "Extract structured data from this resume. "
    "Return valid JSON with these fields where available: "
    "name, email, phone, date_of_birth, location, summary, "
    "hard_skills (array), soft_skills (array), "
    "experience (array of {{title, company, duration}}), "
    "education (array of {{degree, institution}}), "
    "certifications (array), languages (array), "
    "function (job function category), industry (industry category).\n\n"
    "Resume:\n{resume_text}"
)


# =============================================================================
# 📥 PHASE 1: DOWNLOAD / LOAD DATASET
# =============================================================================

def download_dataset(output_path: str = None) -> str:
    """
    📥 Download the Majinuub Resume_Parsing dataset from HuggingFace.

    Think of this as placing an order for fabric from the supplier —
    we need the raw material before we can start sewing! 🧵

    If the file already exists locally, we skip the download (caching!).
    The file is ~1.5MB so it downloads in seconds even on slow connections.

    Args:
        output_path: Where to save the downloaded JSON file.
                     Defaults to training_data/majinuub_raw.json

    Returns:
        Path to the downloaded/cached file
    """
    if output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, "majinuub_raw.json")

    # ── Check if already downloaded (cache hit!) ──────────────────────
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 100_000:  # Sanity check: should be ~1.5MB
            Console.info(f"Dataset already cached at: {output_path} ({file_size:,} bytes)")
            Console.info("Delete the file to force re-download.")
            return output_path
        else:
            Console.warning(f"Cached file seems too small ({file_size} bytes), re-downloading...")

    # ── Download from HuggingFace ─────────────────────────────────────
    if not HAS_REQUESTS:
        Console.error(
            "The 'requests' library is not installed!\n"
            "  Install it:  pip install requests --break-system-packages\n"
            "  Or download manually from:\n"
            f"  {DATASET_URL}\n"
            "  Then use: --local-file path/to/downloaded.json"
        )
        sys.exit(1)

    Console.info(f"Downloading from HuggingFace...")
    Console.info(f"URL: {DATASET_URL}")

    try:
        response = requests.get(DATASET_URL, timeout=60, stream=True)
        response.raise_for_status()

        # ── Stream to file with progress ──────────────────────────────
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        Console.progress_bar(downloaded, total_size)

        Console.success(f"Downloaded: {output_path} ({os.path.getsize(output_path):,} bytes)")
        return output_path

    except requests.exceptions.ConnectionError:
        Console.error(
            "Cannot connect to HuggingFace! Check your internet connection.\n"
            "  Alternatively, download manually and use --local-file"
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        Console.error(f"HTTP error downloading dataset: {e}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        Console.error("Download timed out after 60 seconds. Try again or use --local-file")
        sys.exit(1)


def load_dataset(file_path: str) -> List[Dict]:
    """
    📂 Load the Majinuub dataset from a local JSON file.

    Validates the structure and reports basic stats.
    Like unpacking the delivery boxes and checking everything arrived! 📦

    Args:
        file_path: Path to the JSON file (downloaded or local)

    Returns:
        List of raw dataset records
    """
    Console.banner("📂 PHASE 1: Loading Dataset")

    if not os.path.exists(file_path):
        Console.error(f"File not found: {file_path}")
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        Console.error(f"Invalid JSON in {file_path}: {e}")
        sys.exit(1)
    except UnicodeDecodeError:
        # ── Fallback: try latin-1 encoding ────────────────────────────
        Console.warning("UTF-8 decode failed, trying latin-1...")
        with open(file_path, "r", encoding="latin-1") as f:
            data = json.load(f)

    # ── Validate structure ────────────────────────────────────────────
    if not isinstance(data, list):
        Console.error(f"Expected a JSON array, got {type(data).__name__}")
        sys.exit(1)

    if len(data) == 0:
        Console.error("Dataset is empty!")
        sys.exit(1)

    # ── Check required fields ─────────────────────────────────────────
    required_fields = {"Input", "Output"}
    first_keys = set(data[0].keys())
    missing = required_fields - first_keys
    if missing:
        Console.error(
            f"Missing required fields in dataset: {missing}\n"
            f"  Found fields: {first_keys}\n"
            f"  Expected at minimum: {required_fields}"
        )
        sys.exit(1)

    Console.success(f"Loaded {len(data)} records from {file_path}")
    Console.stat("Fields per record", list(first_keys))
    Console.stat("File size", f"{os.path.getsize(file_path):,} bytes")

    return data


# =============================================================================
# 🔄 PHASE 2: FIELD MAPPING — Majinuub → AiMerlion Schema
# =============================================================================

def map_majinuub_to_aimerlion(majinuub_output: Dict) -> Dict:
    """
    🔄 Map Majinuub's output schema to AiMerlion's extraction schema.

    This is the COSTUME CHANGE, darling! 👗→💃 Same performer (data),
    different outfit (schema). We need the output to match EXACTLY what
    your ai_extractor.py produces, so the model learns the right format.

    Majinuub schema:                    AiMerlion schema:
    ┌──────────────────┐               ┌──────────────────────┐
    │ FirstName         │ ──────────→  │ name                  │
    │ LastName          │ ──┘          │                       │
    │ Email             │ ──────────→  │ email                 │
    │ Phone             │ ──────────→  │ phone                 │
    │ Address/City/     │ ──────────→  │ location              │
    │   Country         │              │                       │
    │ Skill[]           │ ──────────→  │ hard_skills[]         │
    │ Workplaces[]      │ ──────────→  │ experience[]          │
    │ Study[]           │ ──────────→  │ education[]           │
    │ Certifications[]  │ ──────────→  │ certifications[]      │
    │ Summary           │ ──────────→  │ summary               │
    └──────────────────┘               └──────────────────────┘

    Args:
        majinuub_output: Parsed JSON from the Majinuub Output field

    Returns:
        Dict in AiMerlion's extraction output format
    """
    result = {}

    # ── Name: combine FirstName + LastName ────────────────────────────
    # Edge case: some records might have empty LastName or compound names
    first_name = (majinuub_output.get("FirstName") or "").strip()
    last_name = (majinuub_output.get("LastName") or "").strip()
    if first_name and last_name:
        result["name"] = f"{first_name} {last_name}"
    elif first_name:
        result["name"] = first_name
    elif last_name:
        result["name"] = last_name
    # else: omit — missing name is better than empty string for training

    # ── Email ─────────────────────────────────────────────────────────
    email = (majinuub_output.get("Email") or "").strip()
    if email and "@" in email:
        result["email"] = email

    # ── Phone ─────────────────────────────────────────────────────────
    phone = (majinuub_output.get("Phone") or "").strip()
    if phone:
        result["phone"] = phone

    # ── Location: build from Address/City/State/Country ───────────────
    # Strategy: use the most specific info available, avoid duplicates
    address = (majinuub_output.get("Address") or "").strip()
    city = (majinuub_output.get("City") or "").strip()
    state = (majinuub_output.get("State") or "").strip()
    country = (majinuub_output.get("Country") or "").strip()

    # If Address already contains the city/country, just use Address
    # Otherwise, build a composite location string
    if address and city and city.lower() in address.lower():
        result["location"] = address
    elif address:
        result["location"] = address
    else:
        # Build from components
        parts = [p for p in [city, state, country] if p]
        if parts:
            result["location"] = ", ".join(parts)

    # ── Summary ───────────────────────────────────────────────────────
    summary = (majinuub_output.get("Summary") or "").strip()
    if summary:
        result["summary"] = summary

    # ── Skills → hard_skills ──────────────────────────────────────────
    # Majinuub stores all skills in one array; AiMerlion splits hard/soft.
    # Since we can't reliably auto-classify, we put them all in hard_skills
    # and leave soft_skills empty. The model will learn from YOUR annotated
    # SG/MY data what constitutes a soft skill!
    skills = majinuub_output.get("Skill", [])
    if isinstance(skills, list) and skills:
        # Deduplicate while preserving order
        seen = set()
        unique_skills = []
        for s in skills:
            s_clean = str(s).strip()
            if s_clean and s_clean.lower() not in seen:
                seen.add(s_clean.lower())
                unique_skills.append(s_clean)
        if unique_skills:
            result["hard_skills"] = unique_skills
            # Provide empty soft_skills so model learns the field exists
            result["soft_skills"] = []

    # ── Workplaces → experience ───────────────────────────────────────
    # Map Majinuub's detailed workplace entries to AiMerlion's format
    workplaces = majinuub_output.get("Workplaces", [])
    if isinstance(workplaces, list) and workplaces:
        experience = []
        for wp in workplaces:
            if not isinstance(wp, dict):
                continue

            exp_entry = {}

            # Job title
            title = (wp.get("JobTitle") or "").strip()
            if title:
                exp_entry["title"] = title

            # Company name
            company = (wp.get("Company_Name") or "").strip()
            if company:
                exp_entry["company"] = company

            # Duration: combine From_Date and To_Date into a readable string
            # Majinuub uses "YYYY/MM/DD" or "current" format
            from_date = (wp.get("From_Date") or "").strip()
            to_date = (wp.get("To_Date") or "").strip()
            duration = _format_duration(from_date, to_date)
            if duration:
                exp_entry["duration"] = duration

            # Only add if we have meaningful data (at least title OR company)
            if exp_entry.get("title") or exp_entry.get("company"):
                experience.append(exp_entry)

        if experience:
            result["experience"] = experience

    # ── Study → education ─────────────────────────────────────────────
    studies = majinuub_output.get("Study", [])
    if isinstance(studies, list) and studies:
        education = []
        for study in studies:
            if not isinstance(study, dict):
                continue

            edu_entry = {}

            degree = (study.get("Degree") or "").strip()
            if degree:
                edu_entry["degree"] = degree

            institution = (study.get("Institution") or "").strip()
            if institution:
                edu_entry["institution"] = institution

            if edu_entry:
                education.append(edu_entry)

        if education:
            result["education"] = education

    # ── Certifications ────────────────────────────────────────────────
    certs = majinuub_output.get("Certifications", [])
    if isinstance(certs, list) and certs:
        cert_list = []
        for cert in certs:
            if isinstance(cert, dict):
                cert_title = (cert.get("Certificate Title") or "").strip()
                if cert_title:
                    cert_list.append(cert_title)
            elif isinstance(cert, str):
                cert_list.append(cert.strip())

        if cert_list:
            result["certifications"] = cert_list

    # ── Languages (not in Majinuub, but include empty array) ──────────
    # This teaches the model that the field EXISTS even when not found
    result["languages"] = []

    return result


def _format_duration(from_date: str, to_date: str) -> str:
    """
    📅 Format Majinuub's date fields into a human-readable duration string.

    Majinuub uses "YYYY/MM/DD" format. AiMerlion uses human-readable
    strings like "January 2020 to Present" (matching SG/MY resume style).

    Examples:
        "2020/01/01", "current"    → "January 2020 to Present"
        "2017/07/01", "2019/12/31" → "July 2017 to December 2019"
        "2020", "current"          → "2020 to Present"

    Args:
        from_date: Start date string
        to_date:   End date string

    Returns:
        Formatted duration string, or empty string if both are empty
    """
    MONTH_NAMES = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    def parse_date_str(date_str: str) -> str:
        """Convert YYYY/MM/DD to 'Month Year' format."""
        if not date_str:
            return ""

        # Handle "current" / "present" / "now" variants
        if date_str.lower() in ("current", "present", "now", "ongoing"):
            return "Present"

        # Try YYYY/MM/DD format
        parts = date_str.replace("-", "/").split("/")
        try:
            year = int(parts[0])
            if len(parts) >= 2:
                month = int(parts[1])
                if 1 <= month <= 12:
                    return f"{MONTH_NAMES[month]} {year}"
            return str(year)
        except (ValueError, IndexError):
            # If all parsing fails, return the raw string
            return date_str

    from_str = parse_date_str(from_date)
    to_str = parse_date_str(to_date)

    if from_str and to_str:
        return f"{from_str} to {to_str}"
    elif from_str:
        return f"{from_str} to Present"
    elif to_str:
        return f"to {to_str}"
    return ""


# =============================================================================
# 🎭 PHASE 3: FORMAT AS SHAREGPT — Ready for finetune_unsloth.py
# =============================================================================

def format_as_sharegpt(
    resume_text: str,
    aimerlion_output: Dict,
    include_cot: bool = False,
    chain_of_thought: str = ""
) -> Optional[Dict]:
    """
    🎭 Format one training example as ShareGPT conversation.

    This produces the EXACT format that finetune_unsloth.py expects:
    {
        "conversations": [
            {"from": "system",  "value": "<system prompt>"},
            {"from": "human",   "value": "Extract data from: <resume>"},
            {"from": "gpt",     "value": '{"name": "...", ...}'}
        ]
    }

    The system prompt, instruction template, and output format MUST match
    what's in prepare_training_data.py and your Modelfile — otherwise the
    model learns conflicting formats and gets confused! It's like teaching
    a dancer ballet AND hip-hop at the SAME TIME — pick ONE style per
    rehearsal, darling! 💃

    Args:
        resume_text:       Raw resume text from the Input field
        aimerlion_output:  Mapped output from map_majinuub_to_aimerlion()
        include_cot:       Whether to include Chain-of-Thought reasoning
        chain_of_thought:  The CoT text from Majinuub dataset

    Returns:
        ShareGPT formatted dict, or None if the data is invalid
    """
    # ── Validate: skip if output is essentially empty ─────────────────
    if not aimerlion_output or len(aimerlion_output) < 2:
        return None

    # ── Validate: skip if resume text is too short ────────────────────
    if not resume_text or len(resume_text.strip()) < 50:
        return None

    # ── Truncate very long resumes (saves VRAM during training) ───────
    # Most Majinuub resumes are 250-780 chars (short), but safety first!
    truncated_text = resume_text.strip()[:4000]

    # ── Build the instruction (matching prepare_training_data.py) ─────
    instruction = INSTRUCTION_TEMPLATE.format(resume_text=truncated_text)

    # ── Build the response JSON ───────────────────────────────────────
    # Use compact JSON (no indent) to save tokens during training
    response = json.dumps(aimerlion_output, ensure_ascii=False, indent=None)

    # ── Build the conversation ────────────────────────────────────────
    conversations = [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "human", "value": instruction},
        {"from": "gpt", "value": response},
    ]

    # ── Optional: Include Chain-of-Thought as a reasoning step ────────
    # This is EXPERIMENTAL. CoT can help the model "think through" its
    # extraction process, but it also makes training examples ~2x longer
    # (more VRAM, longer training). Use with caution! ⚠️
    if include_cot and chain_of_thought and chain_of_thought.strip():
        # Insert CoT BEFORE the extraction response
        # This teaches the model to reason first, then extract
        cot_text = chain_of_thought.strip()[:2000]  # Cap CoT length
        conversations = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": instruction},
            {
                "from": "gpt",
                "value": (
                    f"Let me analyze this resume step by step:\n\n"
                    f"{cot_text}\n\n"
                    f"Based on this analysis, here is the extracted data:\n\n"
                    f"{response}"
                )
            },
        ]

    return {"conversations": conversations}


# =============================================================================
# 📊 PHASE 4: QUALITY ANALYSIS & REPORTING
# =============================================================================

def analyze_dataset(data: List[Dict]) -> Dict:
    """
    📊 Analyze the dataset for quality issues before conversion.

    Like a dress rehearsal inspection — check for ripped seams, missing
    buttons, and wrong sizes BEFORE the show! 👗🔍

    Returns a stats dict with coverage info, country distribution, etc.
    """
    Console.banner("📊 PHASE 2: Dataset Analysis")

    stats = {
        "total": len(data),
        "valid": 0,
        "invalid": 0,
        "parse_errors": 0,
        "country_dist": Counter(),
        "field_coverage": defaultdict(int),
        "text_lengths": [],
        "output_field_counts": [],
        "has_cot": 0,
    }

    for i, record in enumerate(data):
        # ── Parse the Output field (it's a JSON string) ───────────────
        try:
            output = json.loads(record.get("Output", "{}"))
            # Edge case: some records wrap the dict in a list (e.g., record 79)
            if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                output = output[0]
            elif not isinstance(output, dict):
                stats["parse_errors"] += 1
                continue
        except (json.JSONDecodeError, TypeError):
            stats["parse_errors"] += 1
            continue

        # ── Validate minimum viable data ──────────────────────────────
        has_name = bool(output.get("FirstName") or output.get("LastName"))
        has_input = bool(record.get("Input", "").strip())

        if has_name and has_input:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

        # ── Track field coverage ──────────────────────────────────────
        for key in output:
            value = output[key]
            # Count only non-empty fields
            if value and (not isinstance(value, (list, dict)) or len(value) > 0):
                stats["field_coverage"][key] += 1

        # ── Track country distribution ────────────────────────────────
        country = output.get("Country", "Unknown")
        stats["country_dist"][country] += 1

        # ── Track text lengths ────────────────────────────────────────
        stats["text_lengths"].append(len(record.get("Input", "")))

        # ── Track output richness ─────────────────────────────────────
        stats["output_field_counts"].append(len(output))

        # ── Track Chain-of-Thought presence ───────────────────────────
        if record.get("Chain_of_Thoughts", "").strip():
            stats["has_cot"] += 1

    # ── Print report ──────────────────────────────────────────────────
    Console.success(f"Total records: {stats['total']}")
    Console.stat("Valid (have name + input text)", stats["valid"])
    Console.stat("Invalid (missing name or text)", stats["invalid"])
    Console.stat("JSON parse errors", stats["parse_errors"])
    Console.stat("Have Chain-of-Thought", stats["has_cot"])

    if stats["text_lengths"]:
        avg_len = sum(stats["text_lengths"]) // len(stats["text_lengths"])
        Console.stat("Avg input text length", f"{avg_len} chars")
        Console.stat("Min/Max text length",
                     f"{min(stats['text_lengths'])} / {max(stats['text_lengths'])} chars")

    print()
    print("  📊 FIELD COVERAGE (out of {})".format(stats["total"]))
    print("  " + "─" * 45)
    for field, count in sorted(stats["field_coverage"].items(), key=lambda x: -x[1]):
        pct = count / stats["total"] * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {field:<20s} [{bar}] {count:>3d} ({pct:.0f}%)")

    print()
    print("  🌍 TOP 10 COUNTRIES")
    print("  " + "─" * 45)
    for country, count in stats["country_dist"].most_common(10):
        pct = count / stats["total"] * 100
        print(f"    {country:<20s} {count:>3d} ({pct:.0f}%)")

    # ── SG/MY warning ─────────────────────────────────────────────────
    sg_my = stats["country_dist"].get("Singapore", 0) + stats["country_dist"].get("Malaysia", 0)
    if sg_my == 0:
        print()
        Console.warning(
            "⚠️  NO Singapore/Malaysia resumes in this dataset!\n"
            "     This data SUPPLEMENTS your SG/MY annotations — it does NOT replace them.\n"
            "     Keep annotating SG/MY resumes for domain-specific accuracy! 💪"
        )

    return stats


# =============================================================================
# 🚀 PHASE 5: CONVERT & EXPORT
# =============================================================================

def convert_dataset(
    data: List[Dict],
    include_cot: bool = False,
    stats: Optional[Dict] = None
) -> List[Dict]:
    """
    🚀 Convert the entire Majinuub dataset to AiMerlion ShareGPT format.

    This is the MAIN EVENT, darling! The grand transformation where
    305 Majinuub resumes become training-ready JSONL entries! 🎭→🎯

    Args:
        data:        Raw dataset records
        include_cot: Include Chain-of-Thought in training examples
        stats:       Pre-computed stats (optional, for filtering)

    Returns:
        List of ShareGPT formatted training examples
    """
    Console.banner("🚀 PHASE 3: Converting to AiMerlion Format")

    converted = []
    skipped = 0
    errors = 0

    for i, record in enumerate(data):
        Console.progress_bar(i + 1, len(data))

        try:
            # ── Parse the Output JSON string ──────────────────────────
            raw_output = record.get("Output", "{}")
            if isinstance(raw_output, str):
                majinuub_output = json.loads(raw_output)
            elif isinstance(raw_output, dict):
                majinuub_output = raw_output
            elif isinstance(raw_output, list):
                majinuub_output = raw_output
            else:
                skipped += 1
                continue

            # Edge case: some records wrap the dict in a list (e.g., record 79)
            if isinstance(majinuub_output, list):
                if len(majinuub_output) > 0 and isinstance(majinuub_output[0], dict):
                    majinuub_output = majinuub_output[0]
                else:
                    skipped += 1
                    continue

            # ── Map fields to AiMerlion schema ────────────────────────
            aimerlion_output = map_majinuub_to_aimerlion(majinuub_output)

            # ── Format as ShareGPT ────────────────────────────────────
            resume_text = record.get("Input", "").strip()
            cot_text = record.get("Chain_of_Thoughts", "")

            sharegpt_entry = format_as_sharegpt(
                resume_text=resume_text,
                aimerlion_output=aimerlion_output,
                include_cot=include_cot,
                chain_of_thought=cot_text,
            )

            if sharegpt_entry:
                converted.append(sharegpt_entry)
            else:
                skipped += 1

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            errors += 1
            logger.debug(f"Error converting record {i}: {e}")

    print()  # newline after progress bar
    Console.success(f"Converted: {len(converted)} examples")
    if skipped > 0:
        Console.warning(f"Skipped (insufficient data): {skipped}")
    if errors > 0:
        Console.warning(f"Errors (parse failures): {errors}")

    return converted


def write_jsonl(
    examples: List[Dict],
    output_path: str,
    split_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[str, str, int, int]:
    """
    📁 Write training examples to JSONL files with train/validation split.

    Creates TWO files:
      - {output_path}           → training data (90% by default)
      - {output_path}_val.jsonl → validation data (10% by default)

    Args:
        examples:    List of ShareGPT formatted dicts
        output_path: Path for the training JSONL file
        split_ratio: Fraction to hold out for validation (default: 10%)
        seed:        Random seed for reproducible splits

    Returns:
        Tuple of (train_path, val_path, train_count, val_count)
    """
    Console.banner("📁 PHASE 4: Writing Training Files")

    # ── Ensure output directory exists ────────────────────────────────
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Shuffle and split ─────────────────────────────────────────────
    random.seed(seed)
    shuffled = list(examples)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - split_ratio))
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    # ── Write training file ───────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    Console.success(f"Training data: {len(train_data)} examples → {output_path}")

    # ── Write validation file ─────────────────────────────────────────
    # Build val path: training_data/majinuub_train.jsonl → training_data/majinuub_val.jsonl
    base, ext = os.path.splitext(output_path)
    if base.endswith("_train"):
        val_path = base.replace("_train", "_val") + ext
    else:
        val_path = base + "_val" + ext

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    Console.success(f"Validation data: {len(val_data)} examples → {val_path}")

    return output_path, val_path, len(train_data), len(val_data)


# =============================================================================
# 🤝 PHASE 6: MERGE WITH EXISTING DATA (OPTIONAL)
# =============================================================================

def merge_with_existing(
    new_examples: List[Dict],
    existing_path: str,
    output_path: str,
    seed: int = 42
) -> Tuple[int, int]:
    """
    🤝 Merge Majinuub data with your existing AiMerlion training data.

    This is like combining your star performers with the backup dancers —
    the COMBINED troupe is stronger than either alone! 💃🕺

    IMPORTANT: We interleave (shuffle together) rather than concatenate,
    so the model sees a MIX during training instead of all-Majinuub
    followed by all-AiMerlion. This prevents "catastrophic forgetting"
    where the model over-fits to whichever dataset it sees last!

    Also adds a deduplication check based on the first 100 chars of
    each resume's input text — no point training on the same resume twice.

    Args:
        new_examples:  Majinuub converted examples
        existing_path: Path to your existing training JSONL
        output_path:   Path for the merged output

    Returns:
        Tuple of (total_examples, duplicates_removed)
    """
    Console.banner("🤝 Merging with Existing Training Data")

    if not os.path.exists(existing_path):
        Console.error(f"Existing training file not found: {existing_path}")
        Console.info("Skipping merge — writing Majinuub data only.")
        return 0, 0

    # ── Load existing data ────────────────────────────────────────────
    existing = []
    with open(existing_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    Console.info(f"Existing data: {len(existing)} examples from {existing_path}")
    Console.info(f"New data: {len(new_examples)} examples from Majinuub")

    # ── Deduplication using resume text fingerprint ───────────────────
    # Extract the first 100 chars of the "human" message as a fingerprint.
    # This catches exact duplicates without being too aggressive.
    def get_fingerprint(example: Dict) -> str:
        """Extract a fingerprint from the human message for dedup."""
        for conv in example.get("conversations", []):
            if conv.get("from") == "human":
                # Hash the first 200 chars of the resume portion
                text = conv.get("value", "")
                # Find "Resume:\n" and take text after it
                resume_start = text.find("Resume:\n")
                if resume_start >= 0:
                    resume_text = text[resume_start + 8:resume_start + 208]
                else:
                    resume_text = text[:200]
                return hashlib.md5(resume_text.encode()).hexdigest()
        return ""

    # Build fingerprint set from existing data
    existing_fingerprints = set()
    for ex in existing:
        fp = get_fingerprint(ex)
        if fp:
            existing_fingerprints.add(fp)

    # Filter out duplicates from new data
    unique_new = []
    duplicates = 0
    for ex in new_examples:
        fp = get_fingerprint(ex)
        if fp and fp in existing_fingerprints:
            duplicates += 1
        else:
            unique_new.append(ex)
            if fp:
                existing_fingerprints.add(fp)

    if duplicates > 0:
        Console.warning(f"Removed {duplicates} duplicate resumes")

    # ── Merge and shuffle ─────────────────────────────────────────────
    combined = existing + unique_new
    random.seed(seed)
    random.shuffle(combined)

    # ── Write merged file ─────────────────────────────────────────────
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    Console.success(f"Merged data: {len(combined)} examples → {output_path}")
    Console.stat("From existing", len(existing))
    Console.stat("From Majinuub (new)", len(unique_new))
    Console.stat("Duplicates removed", duplicates)

    return len(combined), duplicates


# =============================================================================
# 👀 PREVIEW MODE
# =============================================================================

def preview_examples(data: List[Dict], num_examples: int = 3, include_cot: bool = False):
    """
    👀 Preview converted examples without writing files.

    Like a fitting room try-on — see how the data looks before committing! 👗
    """
    Console.banner(f"👀 Preview: First {num_examples} Examples")

    for i, record in enumerate(data[:num_examples]):
        try:
            raw_output = record.get("Output", "{}")
            if isinstance(raw_output, str):
                majinuub_output = json.loads(raw_output)
            else:
                majinuub_output = raw_output

            # Edge case: list-wrapped dict
            if isinstance(majinuub_output, list) and majinuub_output:
                majinuub_output = majinuub_output[0]

            aimerlion_output = map_majinuub_to_aimerlion(majinuub_output)
            resume_text = record.get("Input", "").strip()
            cot_text = record.get("Chain_of_Thoughts", "")

            sharegpt = format_as_sharegpt(
                resume_text=resume_text,
                aimerlion_output=aimerlion_output,
                include_cot=include_cot,
                chain_of_thought=cot_text,
            )

            if sharegpt:
                print(f"\n  ────── Example {i + 1} ──────")
                for conv in sharegpt["conversations"]:
                    role = conv["from"].upper()
                    value = conv["value"]
                    # Truncate for display
                    if len(value) > 300:
                        value = value[:300] + "... [truncated]"
                    print(f"  [{role}]:")
                    for line in value.split("\n")[:8]:
                        print(f"    {line}")
                    if value.count("\n") > 8:
                        print(f"    ... ({value.count(chr(10))} total lines)")
                print()

        except Exception as e:
            print(f"  ❌ Error previewing record {i}: {e}")


# =============================================================================
# 🎬 MAIN — THE GRAND PRODUCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "💅✨ Fairy Codemother's Majinuub Resume Parsing Importer ✨💅\n"
            "Downloads and converts the Majinuub/Resume_Parsing dataset\n"
            "into AiMerlion's ShareGPT JSONL training format."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, "majinuub_train.jsonl"),
        help="Output JSONL file path (default: training_data/majinuub_train.jsonl)"
    )
    parser.add_argument(
        "--local-file", type=str, default=None,
        help="Use a local JSON file instead of downloading from HuggingFace"
    )
    parser.add_argument(
        "--include-cot", action="store_true",
        help="Include Chain-of-Thought reasoning in training examples (experimental, uses more VRAM)"
    )
    parser.add_argument(
        "--preview", type=int, default=0, metavar="N",
        help="Preview first N converted examples without writing files"
    )
    parser.add_argument(
        "--merge-with", type=str, default=None,
        help="Path to existing training JSONL to merge with (e.g., train_data.jsonl)"
    )
    parser.add_argument(
        "--split", type=float, default=0.1,
        help="Validation split ratio (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip download, use cached file only"
    )

    args = parser.parse_args()

    # ── Grand opening banner ──────────────────────────────────────────
    print()
    print("═" * 60)
    print("  💅✨ MAJINUUB RESUME PARSING IMPORTER ✨💅")
    print("  HuggingFace → AiMerlion Training Data Pipeline")
    print("═" * 60)
    print(f"  📂 Output:     {args.output}")
    print(f"  🧪 CoT mode:   {'ON (experimental)' if args.include_cot else 'OFF'}")
    print(f"  ✂️  Val split:  {args.split * 100:.0f}%")
    if args.merge_with:
        print(f"  🤝 Merge with: {args.merge_with}")
    print("═" * 60)

    # ── Step 1: Get the data ──────────────────────────────────────────
    if args.local_file:
        file_path = args.local_file
    elif args.no_download:
        file_path = os.path.join(DEFAULT_OUTPUT_DIR, "majinuub_raw.json")
        if not os.path.exists(file_path):
            Console.error(f"No cached file found at {file_path}. Remove --no-download to fetch it.")
            sys.exit(1)
    else:
        file_path = download_dataset()

    # ── Step 2: Load and validate ─────────────────────────────────────
    data = load_dataset(file_path)

    # ── Step 3: Analyze ───────────────────────────────────────────────
    stats = analyze_dataset(data)

    # ── Step 4: Preview mode (exit early) ─────────────────────────────
    if args.preview > 0:
        preview_examples(data, num_examples=args.preview, include_cot=args.include_cot)
        print("  💡 Preview complete! Remove --preview to generate files.")
        return

    # ── Step 5: Convert ───────────────────────────────────────────────
    converted = convert_dataset(
        data,
        include_cot=args.include_cot,
        stats=stats,
    )

    if not converted:
        Console.error("No examples were converted! Check the dataset format.")
        sys.exit(1)

    # ── Step 6: Merge or write standalone ─────────────────────────────
    if args.merge_with:
        # Merge mode: combine with existing training data
        merged_output = args.output.replace("majinuub_train", "merged_train")
        total, dupes = merge_with_existing(
            new_examples=converted,
            existing_path=args.merge_with,
            output_path=merged_output,
            seed=args.seed,
        )

        # Also write standalone Majinuub files for reference
        train_path, val_path, train_n, val_n = write_jsonl(
            converted, args.output, args.split, args.seed
        )

        Console.banner("🎯 NEXT STEPS")
        print(f"  Option A — Train on MERGED data (recommended):")
        print(f"    python finetune_unsloth.py \\")
        print(f"      --train-file {merged_output}")
        print()
        print(f"  Option B — Train on Majinuub ONLY:")
        print(f"    python finetune_unsloth.py \\")
        print(f"      --train-file {train_path} \\")
        print(f"      --val-file {val_path}")

    else:
        # Standalone mode: write Majinuub data only
        train_path, val_path, train_n, val_n = write_jsonl(
            converted, args.output, args.split, args.seed
        )

        Console.banner("🎯 NEXT STEPS")
        print(f"  1. Fine-tune your model:")
        print(f"     python finetune_unsloth.py \\")
        print(f"       --train-file {train_path} \\")
        print(f"       --val-file {val_path}")
        print()
        print(f"  2. Or merge with your SG/MY data first:")
        print(f"     python {os.path.basename(__file__)} \\")
        print(f"       --merge-with train_data.jsonl")
        print()
        print(f"  3. Deploy to Ollama after training:")
        print(f"     ollama create aimerlion-resume -f resume_model_finetuned/Modelfile")

    # ── Final summary ─────────────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  💅 {len(converted)} resumes converted and READY for the runway!")
    print(f"  📊 Training: {train_n} | Validation: {val_n}")
    if args.include_cot:
        print(f"  🧠 Chain-of-Thought: INCLUDED (longer examples, more VRAM)")
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()