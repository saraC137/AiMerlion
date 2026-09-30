"""
annotation/app.py — NER annotation tool: Flask app, config, shared NER components.

Run from the repo root:
    python -m annotation            # or: python annotation_tool.py
    # Open http://localhost:5055

Layout:
    app.py          Flask app, config, NER components, request-scoped DB
    queries.py      read-only DB helpers
    routes/         one module per feature (annotate, annotators, export, classify, iaa)
    templates/      index.html + annotate.html, split into partials under index/ and annotate/

Env: RESUME_DB_PATH (default resume_extractions.db), SECRET_KEY.
"""

import sqlite3
import os
import logging

from flask import Flask, g

# Import NER schema components
from extraction.ner_schema import (
    EntitySchema, ResumeTokenizer, BIOTagger,
    NestedEntityLayer, EdgeCaseHandler, PreAnnotator, TrainingExporter,
    AnnotationStorage,
    InterAnnotatorAgreement  # 📏 IAA computation engine
)

# =============================================================================
# 🔧 CONFIGURATION
# =============================================================================

DATABASE_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")
HOST = "0.0.0.0"
PORT = 5055
DEBUG = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - 🏷️ %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 🏗️ FLASK APP + NER COMPONENTS
# =============================================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fairy-ner-sparkle-2026")

# Initialize NER components
schema = EntitySchema()
tokenizer = ResumeTokenizer()
tagger = BIOTagger(schema)
pre_annotator = PreAnnotator(schema)
# 🧠 Auto-expand dictionaries from existing completed annotations!
# Every annotated resume makes the pre-annotator smarter for the next one.
# This is safe to call even if no annotations exist yet — it simply skips.
pre_annotator.load_from_annotations(DATABASE_PATH)
logger.info(f"📊 PreAnnotator dict stats: {pre_annotator.get_dict_stats()}")
nested_handler = NestedEntityLayer(schema)
edge_handler = EdgeCaseHandler()
exporter = TrainingExporter(schema)
storage = AnnotationStorage(DATABASE_PATH)

iaa_engine = InterAnnotatorAgreement(DATABASE_PATH, schema)  # 📏 IAA metrics calculator


# =============================================================================
# 🛠️ HELPER FUNCTIONS
# =============================================================================


def get_db():
    """
    Get a request-scoped database connection.
    ⚡ Reuses the same connection within a single HTTP request,
    avoiding the overhead of opening/closing per helper function.
    Like keeping your backstage pass for the whole show! 🎭✨
    """
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
        # ⚡ Performance PRAGMAs — like giving your DB a double espresso! ☕
        g.db.execute("PRAGMA journal_mode=WAL")        # Concurrent reads
        g.db.execute("PRAGMA synchronous=NORMAL")       # Faster writes (safe with WAL)
        g.db.execute("PRAGMA cache_size=-8000")          # 8MB page cache (default is 2MB)
        g.db.execute("PRAGMA temp_store=MEMORY")         # Temp tables in RAM
        g.db.execute("PRAGMA mmap_size=268435456")       # 256MB memory-mapped I/O
    return g.db

@app.teardown_appcontext
def close_db(exception):
    """Auto-close DB connection when the request ends."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Partials are plain .js/.css includes: keep them autoescaped like the page and
# keep their trailing newlines so the rendered page matches the old single string.
app.jinja_env.autoescape = True
app.jinja_env.keep_trailing_newline = True

# Register routes (modules import app from here, so this must come last).
from annotation.routes import annotate, annotators, export, classify, iaa  # noqa: E402,F401


def main():
    print("\n" + "=" * 60)
    print("  🏷️✨ FAIRY CODEMOTHER'S NER ANNOTATION TOOL ✨🏷️")
    print("=" * 60)
    print(f"  📂 Database: {DATABASE_PATH}")
    print(f"  🌐 URL: http://localhost:{PORT}")
    print(f"  📋 Entity types: {len(schema.entities)}")
    print(f"  🏷️ BIO tags: {len(schema.get_bio_tags())}")
    print("=" * 60)

    if not os.path.exists(DATABASE_PATH):
        print(f"\n  ❌ Database not found: {DATABASE_PATH}")
        print("  💡 Run your extraction pipeline first!")
        exit(1)

    app.run(host=HOST, port=PORT, debug=DEBUG)
