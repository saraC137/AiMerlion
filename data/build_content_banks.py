"""
build_content_banks.py

💎✨ FAIRY CODEMOTHER'S CONTENT BANK BUILDER ✨💎

Uses Gemini sparingly (~6-8 calls total) to build reusable JSON banks
of SG/MY-specific resume ingredients. Run ONCE, then reuse forever.

Output files (all in content_banks/ folder):
    sg_companies.json     — Real SG companies grouped by industry
    my_companies.json     — Real MY companies grouped by industry
    sg_schools.json       — Polytechnics, universities, ITE
    my_schools.json       — Universities, IPTA/IPTS
    certifications.json   — CMFAS, IBF, AWS, professional certs
    skills_by_industry.json — Hard + soft skills per industry
    industries.json       — Industry taxonomy
    functions.json        — Job function taxonomy

Usage:
    python build_content_banks.py
    python build_content_banks.py --only certifications   # rebuild just one
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the Gemini client (must be in same directory)
sys.path.insert(0, str(Path(__file__).parent))
from gemini_client import GeminiClient


# ════════════════════════════════════════════════════════════════════════════
# 🎨 PROMPT TEMPLATES — One per bank
# ════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "sg_companies": """List 30 REAL Singapore companies organized by industry.

Output format: PURE JSON (no markdown, no commentary).
{
  "finance": ["DBS Bank Pte Ltd", "OCBC Bank Pte Ltd", ...],
  "tech": ["Grab Holdings Pte Ltd", "Shopee Singapore Pte Ltd", ...],
  "healthcare": [...],
  "hospitality": [...],
  "education": [...],
  "manufacturing": [...],
  "logistics": [...],
  "government": [...],
  "fmcg": [...]
}

Rules:
- 3-4 companies per industry
- Use real legal entity names with "Pte Ltd" / "Pte. Ltd." suffix where applicable
- Include both MNCs (e.g., Standard Chartered Bank Singapore) AND local champions (e.g., Sea Group, Razer)
- For government: include actual statutory boards (MAS, IRAS, HDB, EDB, etc.)
""",

    "my_companies": """List 30 REAL Malaysian companies organized by industry.

Output format: PURE JSON (no markdown, no commentary).
{
  "finance": ["Maybank Berhad", "CIMB Bank Berhad", ...],
  "tech": ["Grab Malaysia Sdn Bhd", "Axiata Group Berhad", ...],
  "healthcare": [...],
  "hospitality": [...],
  "education": [...],
  "manufacturing": [...],
  "logistics": [...],
  "government": [...],
  "fmcg": [...]
}

Rules:
- 3-4 companies per industry
- Use real legal names with "Sdn Bhd" or "Berhad" suffix
- Include both major Bursa-listed companies AND well-known SMEs
- For government: include actual ministries and GLCs (PETRONAS, Khazanah, etc.)
""",

    "sg_schools": """List all REAL Singapore educational institutions.

Output format: PURE JSON (no markdown, no commentary).
{
  "polytechnics": ["Singapore Polytechnic", "Ngee Ann Polytechnic", ...],
  "universities": ["National University of Singapore", "Nanyang Technological University", ...],
  "ITE": ["Institute of Technical Education (ITE)", "ITE College Central", ...],
  "international_schools": ["Singapore American School", ...],
  "secondary_schools": ["Raffles Institution", "Hwa Chong Institution", ...]
}

Rules:
- Use FULL official names
- Include all 5 SG polytechnics
- Include autonomous universities (NUS, NTU, SMU, SUTD, SIT, SUSS, NIE)
- ITE: all 3 colleges
""",

    "my_schools": """List REAL Malaysian educational institutions.

Output format: PURE JSON (no markdown, no commentary).
{
  "public_universities": ["Universiti Malaya (UM)", "Universiti Kebangsaan Malaysia (UKM)", ...],
  "private_universities": ["Taylor's University", "Monash University Malaysia", ...],
  "polytechnics": ["Politeknik Sultan Salahuddin Abdul Aziz Shah", ...],
  "colleges": ["Sunway College", "Methodist College Kuala Lumpur", ...]
}

Rules:
- Use both Malay AND English names where applicable (e.g., "Universiti Malaya (UM)")
- Include all major IPTA (public) universities
- Include top IPTS (private) universities
""",

    "certifications": """List REAL professional certifications relevant to SG/MY resumes.

Output format: PURE JSON (no markdown, no commentary).
{
  "finance_sg": {
    "CMFAS": ["CMFAS Module 1A", "CMFAS Module 5", "CMFAS Module 6", "CMFAS Module 6A", "CMFAS Module 8", "CMFAS Module 8A", "CMFAS Module 9", "CMFAS Module 9A", "CMFAS HI"],
    "IBF": ["IBF Standards Level 1 - Wealth Management", "IBF Advanced Level 2 - Private Banking", ...],
    "other": ["CFA Charter", "CPA Singapore", "ACCA", ...]
  },
  "tech": {
    "AWS": ["AWS Certified Solutions Architect Associate", "AWS Certified Developer Associate", "AWS Certified DevOps Engineer Professional", "AWS Certified Security Specialty", ...],
    "Azure": ["Microsoft Certified: Azure Solutions Architect Expert", ...],
    "GCP": ["Google Cloud Professional Cloud Architect", ...],
    "other": ["Certified Kubernetes Administrator (CKA)", "PMP", "PMI-ACP", "Certified ScrumMaster (CSM)", ...]
  },
  "healthcare": {
    "general": ["Singapore Medical Council Registration", "BCLS Certification", "ACLS Certification", ...]
  },
  "general": ["TOEFL", "IELTS", "PMP", "Six Sigma Green Belt", "Six Sigma Black Belt", "PRINCE2 Foundation", "PRINCE2 Practitioner", "ITIL v4 Foundation", ...]
}

Rules:
- Use OFFICIAL cert names with full level qualifiers
- CMFAS module names must be REAL (verify against MAS website knowledge)
- Include specialty/level qualifiers as PART of the cert name
""",

    "skills_by_industry": """List skills (both hard and soft) by industry for SG/MY resumes.

Output format: PURE JSON (no markdown, no commentary).
{
  "finance": {
    "hard": ["Financial Modeling", "Bloomberg Terminal", "Wealth Management", "Risk Profiling", "Investment Advisory", "Anti-Money Laundering (AML)", "Know Your Customer (KYC)", "MAS Notice 626 Compliance", ...],
    "soft": ["Client Relationship Management", "Cross-Selling", "Stakeholder Management", ...]
  },
  "tech": {
    "hard": ["Python", "Java", "Kubernetes", "AWS", "Docker", "Terraform", "React", "TypeScript", ...],
    "soft": ["Code Review", "Mentoring", "Technical Leadership", "Agile/Scrum", ...]
  },
  "healthcare": { "hard": [...], "soft": [...] },
  "hospitality": { "hard": [...], "soft": [...] },
  "education": { "hard": [...], "soft": [...] },
  "manufacturing": { "hard": [...], "soft": [...] },
  "logistics": { "hard": [...], "soft": [...] },
  "government": { "hard": [...], "soft": [...] },
  "fmcg": { "hard": [...], "soft": [...] }
}

Rules:
- 12-15 hard skills per industry, 6-8 soft skills
- For finance: include SG-specific (CMFAS topics, MAS regulations)
- For tech: include modern stack items
- Avoid generic non-skills like "communication" alone — be specific
""",

    "industries": """List industry taxonomy for SG/MY recruitment.

Output format: PURE JSON (no markdown, no commentary).
{
  "industries": [
    {"name": "Banking & Finance", "subcategories": ["Retail Banking", "Private Banking", "Investment Banking", "Insurance", "Asset Management", "Fintech"]},
    {"name": "Technology", "subcategories": ["Software Development", "Cloud & DevOps", "Data Science", "Cybersecurity", "Product Management"]},
    {"name": "Healthcare", "subcategories": [...]},
    ...
  ]
}

Rules:
- 12-15 main industries
- 4-8 subcategories each
- Match SG/MY market reality (heavy finance, tech, manufacturing, logistics)
""",

    "functions": """List job function taxonomy for SG/MY recruitment.

Output format: PURE JSON (no markdown, no commentary).
{
  "functions": [
    {"name": "Engineering", "examples": ["Software Engineer", "DevOps Engineer", "Site Reliability Engineer", ...]},
    {"name": "Finance & Accounting", "examples": ["Financial Analyst", "Accountant", "Auditor", ...]},
    {"name": "Sales & Business Development", "examples": [...]},
    {"name": "Marketing & Communications", "examples": [...]},
    {"name": "Operations", "examples": [...]},
    {"name": "Human Resources", "examples": [...]},
    {"name": "Customer Success", "examples": [...]},
    {"name": "Legal & Compliance", "examples": [...]},
    {"name": "Product Management", "examples": [...]},
    {"name": "Data & Analytics", "examples": [...]},
    {"name": "Design", "examples": [...]},
    {"name": "Research & Development", "examples": [...]},
    {"name": "Administration", "examples": [...]}
  ]
}

Rules:
- 10-15 functions covering typical SG/MY job roles
- 5-10 example titles per function
"""
}


# ════════════════════════════════════════════════════════════════════════════
# 🔧 BANK BUILDER LOGIC
# ════════════════════════════════════════════════════════════════════════════

def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON object from LLM response, handling markdown fencing."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (``` or ```json) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # Find the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON decode failed: {e}")
        return None


def build_bank(name: str, prompt: str, gemini: GeminiClient,
               output_path: Path, force: bool = False) -> bool:
    """Build a single content bank file."""
    if output_path.exists() and not force:
        print(f"   ⏭️  {output_path.name} already exists, skipping (use --force to rebuild)")
        return True

    print(f"   🎨 Building {name}...")
    response = gemini.generate(
        prompt,
        temperature=0.3,  # Low temp for factual/list content
        max_output_tokens=4096,
    )

    if not response:
        print(f"   ❌ Gemini returned nothing for {name}")
        return False

    data = extract_json_from_response(response)
    if not data:
        print(f"   ❌ Could not parse JSON for {name}")
        # Save raw response for debugging
        debug_path = output_path.with_suffix(".raw.txt")
        debug_path.write_text(response, encoding="utf-8")
        print(f"   📄 Raw response saved to {debug_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"   ✅ Saved {output_path}")
    return True


# ════════════════════════════════════════════════════════════════════════════
# 🎬 MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build SG/MY content banks for synthetic resume generation")
    parser.add_argument("--only", help="Build only ONE bank (e.g., 'certifications')")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if file already exists")
    parser.add_argument("--output-dir", default="content_banks",
                        help="Output directory (default: content_banks/)")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + "  💎  SG/MY CONTENT BANK BUILDER  💎  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Init Gemini
    try:
        gemini = GeminiClient(model="flash-lite")
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # Test connection
    print("\n🧪 Testing Gemini connection...")
    if not gemini.test_connection():
        print("\n💔 Cannot proceed without Gemini access.")
        sys.exit(1)
    print()

    output_dir = Path(args.output_dir)
    banks_to_build = [args.only] if args.only else list(PROMPTS.keys())

    successes, failures = 0, 0
    for bank_name in banks_to_build:
        if bank_name not in PROMPTS:
            print(f"   ⚠️  Unknown bank: {bank_name}. Skipping.")
            continue
        output_path = output_dir / f"{bank_name}.json"
        if build_bank(bank_name, PROMPTS[bank_name], gemini, output_path, force=args.force):
            successes += 1
        else:
            failures += 1

    print("\n" + "═" * 70)
    print(f"  ✨ DONE: {successes} succeeded, {failures} failed")
    print("═" * 70)
    print(f"\n💋 Next step: Run stage0_lite_generate.py — Qwen will pull from these banks!")


if __name__ == "__main__":
    main()