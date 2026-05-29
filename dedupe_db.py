"""
dedupe_db.py
============
One-time cleanup for duplicate rows in structured_extractions.

Older runs of main.py (before the idempotent-save fix in
data/db_manager.py::save_structured_extraction) inserted a NEW row every
time a candidate folder was re-processed. This script collapses each
candidate down to ONE row — keeping the most recent extraction and
salvaging any human review work (reviewed flag / review_notes).

Usage
-----
  python dedupe_db.py            # preview only (dry run) — changes nothing
  python dedupe_db.py --apply    # actually delete the duplicate rows

After applying once, re-running main.py no longer creates duplicates
(raw_extractions already de-duplicated; structured_extractions now does too).
"""

import sys
import logging
import coloredlogs

import config
from data.db_manager import DatabaseManager

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger,
                    fmt="%(asctime)s - 🧹 %(levelname)s - %(message)s")


def main() -> int:
    apply = "--apply" in sys.argv
    dry_run = not apply

    db_path = getattr(config, "DATABASE_FILE", "resume_extractions.db")
    logger.info(f"Opening database: {db_path}")
    logger.info("MODE: %s", "APPLY (will delete rows)" if apply else "DRY RUN (no changes)")

    db = DatabaseManager(db_path)
    try:
        result = db.dedupe_structured_extractions(dry_run=dry_run)
    finally:
        db.close()

    print("\n" + "=" * 60)
    print("  STRUCTURED_EXTRACTIONS DEDUPE SUMMARY")
    print("=" * 60)
    print(f"  Candidates with duplicates : {result['candidates_with_dupes']}")
    print(f"  Duplicate rows {'to remove' if dry_run else 'removed'}     : {result['rows_removed']}")
    print(f"  Review flags salvaged      : {result['review_flags_salvaged']}")
    print("=" * 60)

    if dry_run and result["rows_removed"] > 0:
        print("\n  This was a DRY RUN — nothing changed.")
        print("  Re-run with:  python dedupe_db.py --apply\n")
    elif not dry_run:
        print("\n  ✅ Cleanup complete. structured_extractions is now de-duplicated.\n")
    else:
        print("\n  ✅ No duplicates found — database is already clean.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
