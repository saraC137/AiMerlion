"""
fix_failed_banks.py

💎 Quick rebuild for sg_schools and certifications with SIMPLER prompts
that won't trigger Gemini's runaway hallucination.

Run this after build_content_banks.py if any banks failed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gemini_client import GeminiClient
from build_content_banks import build_bank, extract_json_from_response


# ════════════════════════════════════════════════════════════════════════════
# 🎯 TIGHTER PROMPTS — Constrained scope, no runaway hallucination
# ════════════════════════════════════════════════════════════════════════════

FIXED_PROMPTS = {
    "sg_schools": """List Singapore tertiary educational institutions only (NO secondary/primary schools).

Output ONLY valid JSON. Stop after the JSON closes. No commentary.

{
  "polytechnics": ["Singapore Polytechnic", "Ngee Ann Polytechnic", "Temasek Polytechnic", "Nanyang Polytechnic", "Republic Polytechnic"],
  "universities": ["National University of Singapore", "Nanyang Technological University", "Singapore Management University", "Singapore University of Technology and Design", "Singapore Institute of Technology", "Singapore University of Social Sciences", "Nanyang Academy of Fine Arts", "LASALLE College of the Arts"],
  "ITE": ["ITE College Central", "ITE College East", "ITE College West"],
  "private": ["Kaplan Higher Education Singapore", "James Cook University Singapore", "PSB Academy", "MDIS Singapore", "Curtin University Singapore"]
}

That's the EXACT output. Just copy it. Do not add anything else.""",

    "certifications": """List the most common professional certifications used in Singapore/Malaysia resumes.

Output ONLY valid JSON. Stop after the JSON closes. No commentary. KEEP IT SHORT.

{
  "finance_sg_cmfas": [
    "CMFAS Module 1A",
    "CMFAS Module 5",
    "CMFAS Module 6",
    "CMFAS Module 6A",
    "CMFAS Module 8",
    "CMFAS Module 8A",
    "CMFAS Module 9",
    "CMFAS Module 9A",
    "CMFAS HI"
  ],
  "finance_sg_ibf": [
    "IBF Standards Level 1 - Wealth Management",
    "IBF Standards Level 2 - Private Banking",
    "IBF Standards Level 3 - Specialist"
  ],
  "finance_general": [
    "CFA Charter",
    "CPA Singapore",
    "ACCA",
    "CA (Singapore)",
    "FRM (Financial Risk Manager)"
  ],
  "tech_aws": [
    "AWS Certified Solutions Architect Associate",
    "AWS Certified Solutions Architect Professional",
    "AWS Certified Developer Associate",
    "AWS Certified SysOps Administrator Associate",
    "AWS Certified DevOps Engineer Professional",
    "AWS Certified Security Specialty",
    "AWS Certified Cloud Practitioner"
  ],
  "tech_azure": [
    "Microsoft Certified: Azure Fundamentals (AZ-900)",
    "Microsoft Certified: Azure Administrator Associate (AZ-104)",
    "Microsoft Certified: Azure Solutions Architect Expert (AZ-305)",
    "Microsoft Certified: Azure DevOps Engineer Expert (AZ-400)"
  ],
  "tech_gcp": [
    "Google Cloud Associate Cloud Engineer",
    "Google Cloud Professional Cloud Architect",
    "Google Cloud Professional Data Engineer"
  ],
  "tech_other": [
    "Certified Kubernetes Administrator (CKA)",
    "Certified Kubernetes Application Developer (CKAD)",
    "HashiCorp Certified: Terraform Associate",
    "Docker Certified Associate"
  ],
  "general": [
    "PMP (Project Management Professional)",
    "PRINCE2 Foundation",
    "PRINCE2 Practitioner",
    "ITIL v4 Foundation",
    "Certified ScrumMaster (CSM)",
    "PMI-ACP",
    "Six Sigma Green Belt",
    "Six Sigma Black Belt",
    "TOGAF 9 Certified"
  ]
}

That's the EXACT output. Just copy it. Do not add anything else."""
}


def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + "  🔧  FIXING FAILED BANKS  🔧  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    output_dir = Path("content_banks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which banks to fix (check which raw files exist)
    to_fix = []
    for bank_name in FIXED_PROMPTS:
        raw_file = output_dir / f"{bank_name}.raw.txt"
        json_file = output_dir / f"{bank_name}.json"
        if raw_file.exists() and not json_file.exists():
            to_fix.append(bank_name)
        elif not json_file.exists():
            to_fix.append(bank_name)

    if not to_fix:
        print("✅ No failed banks detected — all good!")
        return

    print(f"\n🔧 Will rebuild: {to_fix}\n")

    # Init Gemini
    try:
        gemini = GeminiClient(model="flash-lite")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    successes = 0
    for bank_name in to_fix:
        output_path = output_dir / f"{bank_name}.json"
        if build_bank(bank_name, FIXED_PROMPTS[bank_name], gemini, output_path, force=True):
            successes += 1
            # Clean up the .raw.txt file since we succeeded
            raw_file = output_dir / f"{bank_name}.raw.txt"
            if raw_file.exists():
                raw_file.unlink()
                print(f"   🧹 Removed stale {raw_file.name}")

    print(f"\n✨ Fixed {successes}/{len(to_fix)} banks")
    print("💋 Next step: python stage0_lite_generate.py")


if __name__ == "__main__":
    main()