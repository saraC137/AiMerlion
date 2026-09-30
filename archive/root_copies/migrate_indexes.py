"""
migrate_indexes.py — 💅✨ Fairy Codemother's Database Glow-Up Script ✨💅

One-time index creation for AiMerlion DB optimization.
Creates all missing indexes identified in the performance audit.

Run:
    python migrate_indexes.py

Safe to run multiple times (uses IF NOT EXISTS).
"""
import sqlite3
import os
import time

DB_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")

# ─── New indexes to create ─────────────────────────────────────────────
# Format: (index_name, table_name, column_expression)
NEW_INDEXES = [
    # 🔴 CRITICAL — JOIN performance for annotation queue + classification dashboard
    ("idx_ner_doc_candidate",            "ner_documents",          "candidate_id"),
    ("idx_ner_doc_status",               "ner_documents",          "status"),

    # 🔴 CRITICAL — Composite covering index for acc_score aggregation
    ("idx_ner_ann_doc_layer_type",       "ner_annotations",        "doc_id, layer, entity_type"),

    # 🟡 MODERATE — Activity dashboard date-based grouping
    ("idx_ner_ann_updated",              "ner_annotations",        "updated_at"),

    # 🟡 MODERATE — Composite index for IAA span lookups (replaces two singles)
    ("idx_iaa_doc_annotator",            "iaa_annotations",        "doc_id, annotator_name"),

    # 🟡 MODERATE — IAA JOIN on candidate_id (replaces string-concat join)
    ("idx_iaa_candidate",                "iaa_annotations",        "candidate_id"),

    # 🟢 LOW — "Get latest extraction" pattern optimization
    ("idx_structured_candidate_created", "structured_extractions", "candidate_id, created_at DESC"),
    ("idx_raw_candidate_timestamp",      "raw_extractions",        "candidate_id, extraction_timestamp DESC"),
]


def get_existing_indexes(conn):
    """Return a set of existing index names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {row[0] for row in rows}


def get_table_stats(conn):
    """Return row counts for key tables (for context in the report)."""
    tables = [
        "structured_extractions", "raw_extractions", "ner_documents",
        "ner_annotations", "iaa_annotations", "candidate_classifications",
        "verified_candidates", "annotators"
    ]
    stats = {}
    for table in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
            stats[table] = row[0]
        except Exception:
            stats[table] = "N/A"
    return stats


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"\n  ❌ Database not found: {DB_PATH}")
        print(f"  💡 Set RESUME_DB_PATH environment variable or run from the project directory.\n")
        return

    conn = sqlite3.connect(DB_PATH)
    print(f"\n💅 AiMerlion Database Performance Migration")
    print(f"   Database: {DB_PATH}")
    print(f"   Size: {os.path.getsize(DB_PATH) / (1024*1024):.1f} MB")

    # ── Show current table stats ──────────────────────────────────────
    stats = get_table_stats(conn)
    print(f"\n📊 Table Row Counts:")
    for table, count in stats.items():
        print(f"   {table:30s} → {count}")

    # ── Show existing indexes ─────────────────────────────────────────
    existing = get_existing_indexes(conn)
    print(f"\n🔎 Existing indexes: {len(existing)}")
    for idx in sorted(existing):
        if not idx.startswith("sqlite_"):  # Skip internal auto-indexes
            print(f"   ✅ {idx}")

    # ── Create new indexes ────────────────────────────────────────────
    print(f"\n⚡ Creating {len(NEW_INDEXES)} optimized indexes...\n")
    created = 0
    skipped = 0
    failed = 0

    for idx_name, table, columns in NEW_INDEXES:
        sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})"

        if idx_name in existing:
            print(f"   ⏭️  {idx_name} — already exists, skipping")
            skipped += 1
            continue

        try:
            start = time.time()
            conn.execute(sql)
            elapsed = time.time() - start
            print(f"   ✅ {idx_name} on {table}({columns}) [{elapsed:.2f}s]")
            created += 1
        except Exception as e:
            print(f"   ❌ {idx_name} FAILED: {e}")
            failed += 1

    conn.commit()

    # ── Run ANALYZE to update query planner statistics ─────────────────
    print(f"\n📈 Running ANALYZE to update query planner statistics...")
    start = time.time()
    conn.execute("ANALYZE")
    conn.commit()
    elapsed = time.time() - start
    print(f"   ✅ ANALYZE complete [{elapsed:.2f}s]")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ✨ Migration Summary:")
    print(f"     Created:  {created}")
    print(f"     Skipped:  {skipped} (already existed)")
    print(f"     Failed:   {failed}")
    print(f"     New size: {os.path.getsize(DB_PATH) / (1024*1024):.1f} MB")
    print(f"{'='*55}")
    print(f"\n  💅 Your database is SNATCHED, darling! 👑✨\n")

    conn.close()


if __name__ == "__main__":
    migrate()