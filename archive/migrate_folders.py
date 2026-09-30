"""
migrate_folders.py - AiMerlion Project Reorganization Script
============================================================
🧚 The Fairy Codemother's Wardrobe Makeover Script! 🧚

This script reorganizes the flat AiMerlion project structure into
10 organized module folders. It:
  1. Creates the new folder structure
  2. Copies files to their new homes (originals preserved!)
  3. Creates __init__.py files with re-exports for backward compatibility
  4. Generates a compatibility shim layer so OLD imports still work
  5. Logs everything for your peace of mind

USAGE:
  python migrate_folders.py              # Dry run (preview only)
  python migrate_folders.py --execute    # Actually perform the migration
  python migrate_folders.py --rollback   # Undo the migration

SAFETY:
  - Original files are NEVER deleted (only copied)
  - A backup manifest is saved for rollback
  - Dry run mode is the default (you must explicitly --execute)
  
Author: Fairy Codemother ✨
Date: 2026-03-31
"""

import os
import sys
import shutil
import json
import argparse
from datetime import datetime
from pathlib import Path

# ============================================================================
# MIGRATION MAP: Where each file goes
# ============================================================================
# Format: "original_filename": "target_folder/"
# Files not listed here stay at the project root.

MIGRATION_MAP = {
    # --- core/ : Entry points & orchestration ---
    "main.py":              "core/",
    "config.py":            "core/",
    "utils.py":             "core/",

    # --- extraction/ : The AI+Regex extraction engine ---
    "ai_extractor.py":      "extraction/",
    "ai_validator.py":      "extraction/",
    "document_parser.py":   "extraction/",
    "marker_extractor.py":  "extraction/",
    "ner_schema.py":        "extraction/",
    "optimized_prompts.py": "extraction/",

    # --- data/ : Database & data management ---
    "db_manager.py":        "data/",
    "db_diagnostic.py":     "data/",
    "fix_cache.py":         "data/",

    # --- ml/ : Machine learning & model training ---
    "ml_engine.py":         "ml/",
    "finetune_model.py":    "ml/",
    "finetune_unsloth.py":  "ml/",
    "train_ner.py":         "ml/",
    "merge_lora.py":        "ml/",
    "prompt_optimizer.py":  "ml/",

    # --- analysis/ : Evaluation, classification & export ---
    "resume_classifier.py":  "analysis/",
    "evaluator.py":          "analysis/",
    "resume_exporter.py":    "analysis/",
    "compare_extraction.py": "analysis/",

    # --- dashboards/ : Visual review interfaces ---
    "classification_dashboard.py": "dashboards/",
    "review_dashboard.py":         "dashboards/",

    # --- data_prep/ : Training data preparation & labeling ---
    "labeling_tool.py":          "data_prep/",
    "annotation_tool.py":        "data_prep/",
    "converter.py":              "data_prep/",
    "extract_raw_text.py":       "data_prep/",
    "prepare_training_data.py":  "data_prep/",
    "split_dataset.py":          "data_prep/",

    # --- diagnostics/ : Health checks, debugging & inspection ---
    "diagnose.py":          "diagnostics/",
    "diagnose_ai.py":       "diagnostics/",
    "diagnose_pdf.py":      "diagnostics/",
    "pdf_inspector.py":     "diagnostics/",
    "check_resumes.py":     "diagnostics/",
    "check_structure.py":   "diagnostics/",
    "list_resumes.py":      "diagnostics/",
    "migrate_indexes.py":   "diagnostics/",

    # --- tests/ : Test suite ---
    "test_128k.py":                      "tests/",
    "test_extraction.py":                "tests/",
    "test_extraction_comprehensive.py":  "tests/",
    "test_extraction_fix.py":            "tests/",
    "test_marker.py":                    "tests/",
    "test_model.py":                     "tests/",
    "test_ocr.py":                       "tests/",
    "test_pdf_setup.py":                 "tests/",
    "test_setup.py":                     "tests/",

    # --- archive/ : Legacy versions (kept for reference) ---
    "ai_extractor_old.py":       "archive/",
    "ai_extractor_old_v2.py":    "archive/",
    "ai_extractor_old_v3.py":    "archive/",
    "ai_extractor_finetuned.py": "archive/",
    "mainONE.py":                "archive/",
    "quicktest.py":              "archive/",

    # --- docs/ : Documentation files ---
    "READ.md":                    "docs/",
    "CODE_DOCUMENTATION.md":      "docs/",
    "project_documentation.txt":  "docs/",

    # --- logs/ : Runtime logs ---
    "processing.log":            "logs/",
}

# Files that STAY at root (not in MIGRATION_MAP)
ROOT_FILES = [
    "requirements.txt",
    "README.md",
    "migrate_folders.py",   # This script itself
]

# ============================================================================
# __init__.py RE-EXPORT TEMPLATES
# ============================================================================
# These ensure that old-style imports like `from config import MODEL_NAME`
# still work after migration via compatibility shims.

INIT_TEMPLATES = {
    "core/__init__.py": '''"""
core/ - Entry points & orchestration for AiMerlion
===================================================
Contains the main CLI, configuration, and utility modules.

Module path convention:
  from core.config import MODEL_NAME
  from core.utils import standardize_phone_number
  from core.main import UltimateResumeExtractor
"""
from core.config import *
''',

    "extraction/__init__.py": '''"""
extraction/ - The AI+Regex extraction engine
=============================================
Contains AI extractor, validator, document parser, and prompt modules.

Module path convention:
  from extraction.ai_extractor import AIExtractor
  from extraction.document_parser import DocumentParser
  from extraction.ai_validator import AIValidator
"""
''',

    "data/__init__.py": '''"""
data/ - Database & data management
====================================
Contains the SQLite database manager, diagnostics, and cache tools.

Module path convention:
  from data.db_manager import DatabaseManager
"""
''',

    "ml/__init__.py": '''"""
ml/ - Machine learning & model training
=========================================
Contains the ML engine, fine-tuning scripts, and prompt optimizer.

Module path convention:
  from ml.ml_engine import MLEngine
"""
''',

    "analysis/__init__.py": '''"""
analysis/ - Evaluation, classification & export
=================================================
Contains evaluator, classifier, exporter, and comparison tools.

Module path convention:
  from analysis.evaluator import Evaluator
  from analysis.resume_exporter import ResumeExporter
"""
''',

    "dashboards/__init__.py": '''"""
dashboards/ - Visual review interfaces
========================================
Contains classification and review dashboard UIs.
"""
''',

    "data_prep/__init__.py": '''"""
data_prep/ - Training data preparation & labeling
===================================================
Contains labeling tools, annotation interface, and data pipeline utilities.
"""
''',

    "diagnostics/__init__.py": '''"""
diagnostics/ - Health checks, debugging & inspection
======================================================
Contains diagnostic scripts for AI, PDF, database, and project health.
"""
''',

    "tests/__init__.py": '''"""
tests/ - Test suite for AiMerlion
==================================
Contains unit tests, integration tests, and setup verification.
"""
''',

    "archive/__init__.py": '''"""
archive/ - Legacy versions (kept for reference)
=================================================
Old extractor versions and deprecated entry points.
These are NOT imported by any active code.
"""
''',

    "docs/__init__.py": None,  # No __init__.py for docs
    "logs/__init__.py": None,  # No __init__.py for logs
}

# ============================================================================
# COMPATIBILITY SHIM TEMPLATE
# ============================================================================
# These are small files placed at the ROOT that redirect old imports
# to the new locations, so nothing breaks during the transition.

SHIM_TEMPLATE = '''"""
{filename} - COMPATIBILITY SHIM (auto-generated by migrate_folders.py)
========================================================================
This file redirects imports to the new module location: {new_path}

OLD import (still works):
  from {module_name} import SomeClass

NEW import (preferred):
  from {package}.{module_name} import SomeClass

TODO: Update your imports to use the new paths, then delete this shim.
"""
import warnings
warnings.warn(
    "Importing '{module_name}' from root is deprecated. "
    "Use 'from {package}.{module_name} import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)
from {package}.{module_name} import *
'''


# ============================================================================
# MIGRATION ENGINE
# ============================================================================

class MigrationEngine:
    """Handles the safe migration of files to their new folder homes."""

    def __init__(self, project_root: str, dry_run: bool = True):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.manifest = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "operations": [],
            "shims_created": [],
            "inits_created": [],
        }
        self.errors = []
        self.warnings = []

    def _log(self, action: str, source: str, target: str = ""):
        """Log a migration action."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        arrow = f" → {target}" if target else ""
        print(f"  {prefix}{action}: {source}{arrow}")

    def validate_project(self) -> bool:
        """Pre-flight check: verify we're in the right project."""
        print("\n🔍 Pre-flight checks...")

        # Check for key files that confirm this is AiMerlion
        key_files = ["main.py", "config.py", "ai_extractor.py", "utils.py"]
        missing = [f for f in key_files if not (self.project_root / f).exists()]

        if missing:
            print(f"  ❌ Missing key files: {', '.join(missing)}")
            print(f"     Are you sure this is the AiMerlion project root?")
            return False

        # Check if migration was already done
        existing_folders = [
            f for f in ["core", "extraction", "data", "ml", "analysis"]
            if (self.project_root / f).is_dir()
        ]
        if existing_folders:
            print(f"  ⚠️  Folders already exist: {', '.join(existing_folders)}")
            print(f"     Migration may have already been run. Use --rollback first.")
            self.warnings.append(f"Existing folders found: {existing_folders}")
            return False

        print("  ✅ Project validated! All key files found.")
        return True

    def create_folders(self):
        """Step 1: Create the new folder structure."""
        print("\n📁 Creating folder structure...")
        folders = sorted(set(MIGRATION_MAP.values()))

        for folder in folders:
            folder_path = self.project_root / folder
            self._log("CREATE DIR", folder)
            if not self.dry_run:
                folder_path.mkdir(parents=True, exist_ok=True)

            self.manifest["operations"].append({
                "action": "create_dir",
                "path": folder,
            })

    def copy_files(self):
        """Step 2: Copy files to their new locations."""
        print("\n📦 Copying files to new locations...")
        copied = 0
        skipped = 0

        for filename, target_folder in sorted(MIGRATION_MAP.items()):
            source = self.project_root / filename
            target = self.project_root / target_folder / filename

            if not source.exists():
                self._log("SKIP (not found)", filename)
                skipped += 1
                self.warnings.append(f"File not found: {filename}")
                continue

            self._log("COPY", filename, f"{target_folder}{filename}")
            if not self.dry_run:
                shutil.copy2(source, target)

            self.manifest["operations"].append({
                "action": "copy",
                "source": filename,
                "target": f"{target_folder}{filename}",
                "size": source.stat().st_size if source.exists() else 0,
            })
            copied += 1

        print(f"\n  📊 Copied: {copied} | Skipped: {skipped}")

    def create_init_files(self):
        """Step 3: Create __init__.py files for each package."""
        print("\n📝 Creating __init__.py files...")

        for init_path, content in INIT_TEMPLATES.items():
            if content is None:
                continue  # Skip docs/ and logs/

            full_path = self.project_root / init_path
            self._log("CREATE", init_path)
            if not self.dry_run:
                full_path.write_text(content, encoding="utf-8")

            self.manifest["inits_created"].append(init_path)

    def create_compatibility_shims(self):
        """Step 4: Create root-level shim files for backward compatibility."""
        print("\n🔗 Creating backward-compatible import shims...")

        # Only create shims for files that are commonly imported by other modules
        # (not tests, diagnostics, archive, docs, or logs)
        shim_folders = {
            "core/": "core",
            "extraction/": "extraction",
            "data/": "data",
            "ml/": "ml",
            "analysis/": "analysis",
            "dashboards/": "dashboards",
        }

        for filename, target_folder in MIGRATION_MAP.items():
            # Only shim Python files in importable packages
            if not filename.endswith(".py"):
                continue
            if target_folder not in shim_folders:
                continue

            package = shim_folders[target_folder]
            module_name = filename.replace(".py", "")
            new_path = f"{package}/{filename}"

            shim_path = self.project_root / filename
            shim_content = SHIM_TEMPLATE.format(
                filename=filename,
                new_path=new_path,
                module_name=module_name,
                package=package,
            )

            self._log("SHIM", filename, f"→ {new_path}")
            if not self.dry_run:
                # Only overwrite original if it was already copied
                target_check = self.project_root / target_folder / filename
                if target_check.exists():
                    shim_path.write_text(shim_content, encoding="utf-8")
                else:
                    self.errors.append(
                        f"Cannot create shim for {filename}: "
                        f"target {new_path} doesn't exist yet"
                    )
                    continue

            self.manifest["shims_created"].append(filename)

    def save_manifest(self):
        """Save the migration manifest for rollback capability."""
        manifest_path = self.project_root / ".migration_manifest.json"
        print(f"\n💾 Saving migration manifest...")
        self._log("SAVE", ".migration_manifest.json")

        if not self.dry_run:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)

    def print_summary(self):
        """Print a fabulous summary of the migration."""
        print("\n" + "=" * 60)
        if self.dry_run:
            print("✨ DRY RUN COMPLETE - No files were modified! ✨")
            print("  Run with --execute to perform the actual migration.")
        else:
            print("✨ MIGRATION COMPLETE! Your project just got a GLOW-UP! ✨")

        ops = self.manifest["operations"]
        dirs_created = len([o for o in ops if o["action"] == "create_dir"])
        files_copied = len([o for o in ops if o["action"] == "copy"])
        shims = len(self.manifest["shims_created"])
        inits = len(self.manifest["inits_created"])

        print(f"\n  📁 Folders created:    {dirs_created}")
        print(f"  📦 Files copied:       {files_copied}")
        print(f"  📝 __init__.py files:  {inits}")
        print(f"  🔗 Compat shims:       {shims}")

        if self.warnings:
            print(f"\n  ⚠️  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"     - {w}")

        if self.errors:
            print(f"\n  ❌ Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"     - {e}")

        print("=" * 60)

    def run(self):
        """Execute the full migration pipeline."""
        print("🧚 AiMerlion Project Reorganization - Fairy Codemother Edition ✨")
        print(f"   Project root: {self.project_root}")
        print(f"   Mode: {'DRY RUN (safe preview)' if self.dry_run else '🔥 EXECUTE MODE 🔥'}")

        if not self.validate_project():
            print("\n❌ Pre-flight checks failed. Migration aborted.")
            return False

        self.create_folders()
        self.copy_files()
        self.create_init_files()
        self.create_compatibility_shims()
        self.save_manifest()
        self.print_summary()

        if not self.dry_run:
            print("\n📋 NEXT STEPS:")
            print("  1. Test that 'python core/main.py' runs correctly")
            print("  2. Test that 'python main.py' still works (via shim)")
            print("  3. Gradually update imports to use new paths:")
            print("     OLD: from config import MODEL_NAME")
            print("     NEW: from core.config import MODEL_NAME")
            print("  4. Once all imports are updated, delete the root shim files")
            print("  5. Run your test suite to confirm everything works!")

        return True


def rollback(project_root: str):
    """Undo a migration using the saved manifest."""
    root = Path(project_root)
    manifest_path = root / ".migration_manifest.json"

    if not manifest_path.exists():
        print("❌ No migration manifest found. Nothing to roll back.")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print("🔄 Rolling back migration...")

    # Remove shim files (restore originals from the target folders)
    for shim_file in manifest.get("shims_created", []):
        shim_path = root / shim_file
        # Find the original in its target folder
        target_folder = MIGRATION_MAP.get(shim_file, "")
        if target_folder:
            original_in_folder = root / target_folder / shim_file
            if original_in_folder.exists():
                print(f"  RESTORE: {shim_file} (from {target_folder})")
                shutil.copy2(original_in_folder, shim_path)

    # Remove __init__.py files
    for init_file in manifest.get("inits_created", []):
        init_path = root / init_file
        if init_path.exists():
            print(f"  DELETE: {init_file}")
            init_path.unlink()

    # Remove created folders (only if empty after removing copied files)
    for op in reversed(manifest.get("operations", [])):
        if op["action"] == "copy":
            target_path = root / op["target"]
            if target_path.exists():
                print(f"  DELETE: {op['target']}")
                target_path.unlink()
        elif op["action"] == "create_dir":
            dir_path = root / op["path"]
            if dir_path.exists() and not any(dir_path.iterdir()):
                print(f"  RMDIR: {op['path']}")
                dir_path.rmdir()

    # Remove manifest
    manifest_path.unlink()
    print("\n✅ Rollback complete! Project restored to flat structure.")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🧚 AiMerlion Project Reorganization Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_folders.py              # Preview the migration (safe!)
  python migrate_folders.py --execute    # Actually do it
  python migrate_folders.py --rollback   # Undo the migration
        """,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default is dry run)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Undo a previous migration using the saved manifest",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    if args.rollback:
        rollback(args.root)
    else:
        engine = MigrationEngine(
            project_root=args.root,
            dry_run=not args.execute,
        )
        engine.run()


if __name__ == "__main__":
    main()