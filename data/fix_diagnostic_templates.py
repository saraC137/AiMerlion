"""
fix_diagnostic_templates.py

🛠️✨ FAIRY CODEMOTHER'S TEMPLATE SCHEMA FIXER ✨🛠️

Regenerates your diagnostic/case_XX_template.json files with the canonical
schema (model_output, ground_truth, and field_evaluation all having the same
26 fields). Preserves any work you've already done in existing templates.

Usage:
    python fix_diagnostic_templates.py
    python fix_diagnostic_templates.py --dir diagnostic --backup
"""

import json
import shutil
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime


# =============================================================================
# 🎨 THE CANONICAL SCHEMA — single source of truth
# =============================================================================

CANONICAL_FLAT_FIELDS = [
    "ID", "Name", "Page", "Phone", "Email",
    "Current Company", "Current Title", "Team",
    "Current Location", "Expected Location",
    "Gender", "Created By", "Creation Date", "Last Contact",
    "Function", "Summary",
    "References", "Hobbies"
]

CANONICAL_ARRAY_FIELDS = [
    "Industry", "Language Skills", "Certifications",
    "hard_skills/tags", "soft_skills/skills", "Achievements"
]

# Nested fields and their sub-schemas
CANONICAL_NESTED_FIELDS = {
    "Work Experience": ["company", "title", "from", "to", "responsibility"],
    "Project Experience": ["name", "description", "date"],
    "Education": ["school", "major", "degree", "dates"],
}


def build_blank_extraction():
    """Build an empty extraction matching the canonical schema."""
    blank = {}
    for f in CANONICAL_FLAT_FIELDS:
        blank[f] = ""
    for f in CANONICAL_ARRAY_FIELDS:
        blank[f] = []
    for parent, subs in CANONICAL_NESTED_FIELDS.items():
        sub_template = {}
        for s in subs:
            sub_template[s] = [] if s == "responsibility" else ""
        blank[parent] = [sub_template.copy()]
    return blank


def build_blank_field_eval_entry():
    """Build an empty per-field evaluation entry."""
    return {
        "correct": None,
        "failure_type": "",
        "root_cause": "",
        "notes": ""
    }


def build_blank_field_evaluation():
    """Build the full field_evaluation block."""
    eval_block = {}
    # Flat fields
    for f in CANONICAL_FLAT_FIELDS + CANONICAL_ARRAY_FIELDS:
        eval_block[f] = build_blank_field_eval_entry()
    # Nested fields — each gets _overall + per-sub-field
    for parent, subs in CANONICAL_NESTED_FIELDS.items():
        eval_block[parent] = {"_overall": build_blank_field_eval_entry()}
        for s in subs:
            eval_block[parent][s] = build_blank_field_eval_entry()
    return eval_block


def build_canonical_template():
    """Build a complete fresh template matching the canonical schema."""
    return {
        "case_id": "case_XX",
        "candidate_id": "",
        "metadata": {
            "ethnicity_tag": "",
            "industry_hint": "",
            "raw_text_length": 0,
            "_notes": ""
        },
        "input_text": "<see case_XX_raw.txt - too long to inline here>",

        "_section_separator_1": "=========== MODEL OUTPUT (paste from ai_extractor) ===========",
        "model_output": build_blank_extraction(),

        "_section_separator_2": "=========== GROUND TRUTH (what SHOULD have been extracted) ===========",
        "ground_truth": build_blank_extraction(),

        "_section_separator_3": "=========== FIELD EVALUATION (your judgment per field) ===========",
        "_eval_instructions": {
            "correct": "true / false / null (null = not applicable, field doesn't exist in source)",
            "failure_type": "ONE of: hallucination / truncation / missing / wrong_format / wrong_assignment / partial / not_applicable",
            "root_cause": "ONE of: llm_weakness / post_processing / schema_issue / prompt_issue / ocr_issue / not_applicable",
            "notes": "Free-text observations"
        },
        "field_evaluation": build_blank_field_evaluation(),

        "_section_separator_4": "=========== SUMMARY (auto-fill after evaluation) ===========",
        "summary": {
            "total_fields_evaluated": 0,
            "correct": 0,
            "incorrect": 0,
            "not_applicable": 0,
            "field_accuracy": 0.0,
            "failures_by_root_cause": {
                "llm_weakness": 0,
                "post_processing": 0,
                "schema_issue": 0,
                "prompt_issue": 0,
                "ocr_issue": 0
            }
        }
    }


# =============================================================================
# 🔄 MIGRATION LOGIC — preserve user work
# =============================================================================

def deep_merge_preserve_user_data(canonical, existing, path=""):
    """Recursively merge existing data into canonical structure.
    Preserves any non-empty values the user has already entered."""
    if not isinstance(existing, dict) or not isinstance(canonical, dict):
        return canonical

    result = deepcopy(canonical)
    for key in result:
        if key in existing:
            existing_val = existing[key]
            canonical_val = result[key]

            # Both dicts: recurse
            if isinstance(canonical_val, dict) and isinstance(existing_val, dict):
                result[key] = deep_merge_preserve_user_data(canonical_val, existing_val,
                                                              f"{path}.{key}")
            # Both lists: keep existing if non-empty
            elif isinstance(canonical_val, list):
                if existing_val and existing_val != [{}]:
                    result[key] = existing_val
            # Scalar: keep existing if non-empty
            else:
                if existing_val not in (None, "", 0, 0.0):
                    result[key] = existing_val
    return result


def migrate_template(existing_path: Path, output_path: Path = None,
                      backup: bool = True) -> dict:
    """Migrate an existing template to canonical schema. Preserves user work."""
    if output_path is None:
        output_path = existing_path

    # Backup if requested
    if backup and existing_path.exists():
        backup_path = existing_path.with_suffix(
            f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        shutil.copy2(existing_path, backup_path)

    # Load existing (if exists)
    existing_data = {}
    if existing_path.exists():
        try:
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"   ⚠️  Could not parse {existing_path.name}: {e}")
            print(f"   ⚠️  Writing fresh canonical template (no merge)")
            existing_data = {}

    # Build canonical structure
    canonical = build_canonical_template()

    # Preserve case_id if it was set
    if existing_data.get("case_id") and existing_data["case_id"] != "case_XX":
        canonical["case_id"] = existing_data["case_id"]
    else:
        # Derive from filename: case_01_template.json -> case_01
        stem = existing_path.stem.replace("_template", "")
        canonical["case_id"] = stem

    # Preserve candidate_id, metadata, input_text references
    for top_key in ["candidate_id", "metadata", "input_text"]:
        if top_key in existing_data:
            canonical[top_key] = existing_data[top_key]

    # Deep merge model_output, ground_truth, field_evaluation
    for section in ["model_output", "ground_truth", "field_evaluation"]:
        if section in existing_data:
            canonical[section] = deep_merge_preserve_user_data(
                canonical[section], existing_data[section], section
            )

    # Write back
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)

    return canonical


# =============================================================================
# 🎬 MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fix diagnostic templates to canonical schema")
    parser.add_argument('--dir', default='diagnostic',
                        help='Directory containing case_XX_template.json files (default: diagnostic)')
    parser.add_argument('--backup', action='store_true', default=True,
                        help='Backup existing files before overwriting (default: True)')
    parser.add_argument('--no-backup', dest='backup', action='store_false',
                        help='Skip creating backup files')
    parser.add_argument('--fresh', action='store_true',
                        help='Create fresh templates (do NOT preserve existing data)')
    args = parser.parse_args()

    template_dir = Path(args.dir)
    if not template_dir.exists():
        print(f"❌ Directory not found: {template_dir}")
        print(f"   Run select_diagnostic_resumes.py first to create diagnostic/ folder")
        return

    print("╔" + "═" * 68 + "╗")
    print("║" + "  🛠️  TEMPLATE SCHEMA FIXER  🛠️  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Find all case templates
    templates = sorted(template_dir.glob("case_*_template.json"))
    if not templates:
        print(f"\n⚠️  No case_*_template.json files found in {template_dir}/")
        print(f"   Creating ONE example template at {template_dir}/case_template_example.json")
        example_path = template_dir / "case_template_example.json"
        template_dir.mkdir(parents=True, exist_ok=True)
        with open(example_path, 'w', encoding='utf-8') as f:
            json.dump(build_canonical_template(), f, indent=2, ensure_ascii=False)
        print(f"   ✅ Done")
        return

    print(f"\n📋 Found {len(templates)} template files to migrate")
    print(f"🔒 Backup mode: {'ON' if args.backup else 'OFF'}")
    print(f"🆕 Fresh mode: {'ON (existing data will be lost)' if args.fresh else 'OFF (existing data preserved)'}")
    print()

    for template_path in templates:
        case_id = template_path.stem.replace("_template", "")
        if args.fresh:
            # Just write a fresh canonical template
            canonical = build_canonical_template()
            canonical["case_id"] = case_id
            if args.backup and template_path.exists():
                backup_path = template_path.with_suffix(
                    f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )
                shutil.copy2(template_path, backup_path)
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(canonical, f, indent=2, ensure_ascii=False)
            print(f"   ✅ {template_path.name} -> fresh canonical schema")
        else:
            migrate_template(template_path, backup=args.backup)
            print(f"   ✅ {template_path.name} -> canonical schema (existing data preserved)")

    print(f"\n🎉 Done! {len(templates)} templates now have consistent schema.")
    if args.backup:
        print(f"📦 Backups saved with .backup_TIMESTAMP.json suffix")
    print(f"\n💋 — Fairy Codemother")


if __name__ == '__main__':
    main()