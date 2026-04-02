"""
vector_search.py

💅✨ FAIRY CODEMOTHER'S VECTOR SEARCH ENGINE ✨💅

Semantic candidate matching powered by vector embeddings!
Paste a job description → get the most relevant candidates
ranked by semantic similarity. Think of it as a talent matchmaker
who actually UNDERSTANDS what the job needs! 💘🎯

Architecture:
  - Qdrant     → High-performance vector database (ChromaDB's stable, drama-free cousin!)
  - Ollama     → Local embedding model (nomic-embed-text, 768 dimensions)
  - SQLite     → Reads candidates from your existing resume_extractions.db

How It Works:
  1. INDEX:  Reads structured_extractions → builds a "candidate profile"
             text for each person → embeds it → stores in Qdrant
  2. SEARCH: Takes a job description → embeds it → finds top-N most
             similar candidate profiles using cosine similarity
  3. FILTER: Optional skill-based keyword filtering on top of semantic search

Data Flow:
  resume_extractions.db → candidate profiles → Ollama embeddings → Qdrant
                                                                      ↑
  Job Description text → Ollama embedding ─────────────────── search ─┘

Setup (one-time):
    pip install qdrant-client --break-system-packages
    ollama pull nomic-embed-text

Usage:
    from vector_search import VectorSearchEngine

    engine = VectorSearchEngine()
    engine.index_all_candidates()                         # Build the index
    results = engine.search("Senior Python developer...")  # Search!

Dependencies:
    pip install qdrant-client --break-system-packages
    ollama pull nomic-embed-text

Migration note (ChromaDB → Qdrant):
    💅 WHY WE SWITCHED — The Great Migration of 2026! 🎭
    ChromaDB kept changing her EmbeddingFunction protocol across versions
    (0.x → 1.x → 1.5.x), breaking our code every time she had a mood swing.
    The final straw was the `.name()` AttributeError on `get_or_create_collection`!
    Qdrant has a STABLE API, native cosine similarity, proper payload filtering,
    and never once asked our embedding function for its name! She's the reliable
    backup dancer who always hits her mark! 💃✨

    To migrate:
      1. pip uninstall chromadb   (bye, drama queen! 👋)
      2. pip install qdrant-client --break-system-packages
      3. Delete the old chroma_db/ folder (no longer needed)
      4. Run "Rebuild All" to re-index into Qdrant — she handles the rest! ✨
"""

import sqlite3
import json
import os
import re
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# 🔧 LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - 🔍 %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 🎛️ CONFIGURATION
# =============================================================================

# Database paths
RESUME_DB_PATH = os.environ.get("RESUME_DB_PATH", "resume_extractions.db")
QDRANT_PERSIST_DIR = os.environ.get("QDRANT_DIR", "qdrant_db")

# Embedding model — nomic-embed-text is free, local, and good quality!
# Other options: mxbai-embed-large, all-minilm (via sentence-transformers)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# Collection name in Qdrant
COLLECTION_NAME = "candidate_profiles"

# Search defaults
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# Embedding dimensions — nomic-embed-text = 768, mxbai-embed-large = 1024
# Auto-detected at runtime from the test embedding, but this is the fallback.
DEFAULT_EMBEDDING_DIM = 768


# =============================================================================
# 📦 DEPENDENCY CHECKS
# =============================================================================

# ── Qdrant ─────────────────────────────────────────────────────────
# Qdrant is the STABLE QUEEN of vector databases! 👑
# Unlike ChromaDB's protocol drama (name() AttributeError, Settings import
# breakage, where filter migration, embedding function protocol changes...),
# Qdrant's API has been consistent across versions. She handles vectors
# as raw lists — no embedding function protocol nonsense! 🎉
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchText,
        TextIndexParams,
        TokenizerType,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning(
        "⚠️ qdrant-client not installed! Run: pip install qdrant-client --break-system-packages"
    )

# ── Ollama (for embeddings) ───────────────────────────────────────
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("⚠️ ollama not installed! Run: pip install ollama --break-system-packages")


# =============================================================================
# 🧠 OLLAMA EMBEDDING FUNCTION
# =============================================================================

class OllamaEmbeddingFunction:
    """
    Plain Ollama embedding helper — standalone, no DB protocol needed!

    💅 DRAMATIC BACKSTORY: ChromaDB 1.5.x kept changing its EmbeddingFunction
    protocol (first needs `name` attr, then needs it callable, then needs it
    as a class method...). Instead of chasing her drama, we QUIT the protocol
    entirely and handle embeddings MANUALLY.

    With Qdrant, we STILL use this helper — but now it's by CHOICE, not
    because the DB keeps breaking! Qdrant never asked for a .name() method.
    She accepts raw vectors and minds her own business! 👸✨

    Usage (internal):
        embed_fn = OllamaEmbeddingFunction()
        vectors = embed_fn(["text one", "text two"])  # → List[List[float]]
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.dimensions = DEFAULT_EMBEDDING_DIM  # Updated by _test_model()
        self._test_model()

    def _test_model(self):
        """
        Verify the Ollama embedding model is live and ready.

        This is our pre-show soundcheck, darling! 🎤
        We send one test word, confirm we get a valid vector back,
        and log the dimensionality so we know what we're working with.
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "Ollama Python client not installed!\n"
                "Run: pip install ollama --break-system-packages"
            )
        try:
            result = ollama.embed(model=self.model_name, input="test")

            # ── Validate response shape ────────────────────────────
            # Ollama's embed API returns {"embeddings": [[float, ...]]}
            # Guard against unexpected shapes before accessing nested keys.
            if not result or "embeddings" not in result or not result["embeddings"]:
                raise ValueError(
                    f"Unexpected response from Ollama embed API: {result}"
                )

            dims = len(result["embeddings"][0])
            if dims == 0:
                raise ValueError("Embedding model returned zero-dimensional vector!")

            # ── Store actual dimension for collection creation ──────
            # This ensures Qdrant's VectorParams always matches the
            # real model output, even if you switch embedding models!
            # Like measuring the stage before building the set! 📐✨
            self.dimensions = dims

            logger.info(
                f"✨ Embedding model '{self.model_name}' ready! ({dims} dimensions)"
            )

        except RuntimeError:
            raise  # Re-raise our own errors as-is
        except Exception as e:
            raise RuntimeError(
                f"Embedding model '{self.model_name}' not available.\n"
                f"Run: ollama pull {self.model_name}\n"
                f"Error: {e}"
            )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using Ollama. Returns raw float vectors.

        Called manually by VectorSearchEngine — we pre-compute vectors
        and feed them directly to Qdrant's upsert() and query_points().
        Clean separation: Ollama does the math, Qdrant stores the results! 🎯

        Args:
            input: List of strings to embed. Can be 1 or many.

        Returns:
            List of float vectors, one per input string.

        Raises:
            RuntimeError: On Ollama API failure or response mismatch.
        """
        # ── Guard: empty input list ────────────────────────────────
        # Without this, Ollama returns an empty list and the DB
        # throws a cryptic downstream error. Better to catch it here!
        if not input:
            logger.warning("⚠️ OllamaEmbeddingFunction received empty input — returning []")
            return []

        # ── Sanitize: replace empty strings with a neutral placeholder ─
        # Empty strings produce garbage embeddings that poison similarity
        # scores. The placeholder keeps batch size consistent since
        # the DB expects len(embeddings) == len(input) exactly.
        sanitized = [
            text.strip() if (text and text.strip()) else "[empty]"
            for text in input
        ]

        try:
            result = ollama.embed(model=self.model_name, input=sanitized)

            # ── Validate response structure ────────────────────────
            if not result or "embeddings" not in result:
                raise RuntimeError(
                    f"Ollama returned unexpected response shape: {result}"
                )

            embeddings = result["embeddings"]

            # ── Validate count matches input ───────────────────────
            # Mismatch here would cause a silent data corruption in the
            # vector DB — IDs would be paired with wrong vectors! 😱
            if len(embeddings) != len(sanitized):
                raise RuntimeError(
                    f"Embedding count mismatch! "
                    f"Expected {len(sanitized)}, got {len(embeddings)}"
                )

            return embeddings

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"❌ Embedding failed for {len(input)} texts: {e}")
            raise RuntimeError(f"Ollama embedding call failed: {e}") from e


# =============================================================================
# 🏗️ CANDIDATE PROFILE BUILDER
# =============================================================================

def build_candidate_profile(structured: Dict) -> str:
    """
    Build a rich text profile from structured extraction data.

    This is the SECRET SAUCE, darling! 💅 We combine all the
    meaningful fields into one cohesive text that captures
    WHO this candidate is professionally. The embedding model
    then turns this into a vector that represents their
    professional DNA! 🧬✨

    Fields combined (in order of importance):
      1. Summary/Objective  → What they WANT to do
      2. Skills             → What they CAN do
      3. Experience         → What they HAVE done
      4. Education          → Where they LEARNED
      5. Certifications     → Official proof of ability
      6. Languages          → Communication abilities

    ⚠️ NULL SAFETY NOTE — The Great None Trap! 🪤
    SQLite NULL columns come back as Python None, NOT as empty string.
    dict.get("key", "") only uses the default when the KEY is absent.
    When the key EXISTS but holds None (NULL), you get None back —
    and None.strip() throws AttributeError and ruins the whole party! 💀

    We use a helper `_safe(value)` to collapse both None AND empty
    strings into "" before calling .strip() — one bouncer for all
    the NULL troublemakers at the door! 🚪✨

    Args:
        structured: Dict from structured_extractions table row.
                    Values may be None (SQLite NULL) — handled safely.

    Returns:
        A clean text string combining all relevant fields.
        Returns empty string if candidate has no meaningful data.
    """

    def _safe(value: Any, max_len: int = 0) -> str:
        """
        Collapse None, non-string, or whitespace-only values to "".

        This is our NULL bodyguard, honey! 💪 She intercepts any
        troublemaker (None, int leftovers, whatever SQLite throws)
        and converts them to a safe, strippable empty string.

        Args:
            value:   The raw value from the SQLite row dict.
            max_len: If > 0, truncate to this many characters AFTER
                     stripping. Useful for very long text fields.

        Returns:
            Clean stripped string, never None, never raises.
        """
        if value is None:
            return ""
        # Guard against unexpected non-string types from SQLite
        # (e.g. integer candidate_id accidentally passed in)
        text = str(value).strip()
        if max_len > 0 and len(text) > max_len:
            text = text[:max_len]
        return text

    parts = []

    # ── Candidate name ─────────────────────────────────────────────
    name = _safe(structured.get("name"))
    if name:
        parts.append(f"Candidate: {name}")

    # ── Summary / Objective ────────────────────────────────────────
    summary = _safe(structured.get("summary"))
    if summary:
        parts.append(f"Professional Summary: {summary}")

    # ── Skills (highest weight for job matching!) ──────────────────
    skills_raw = _safe(structured.get("skills_raw"))
    if skills_raw:
        parts.append(f"Skills: {skills_raw}")

    # ── Work Experience ────────────────────────────────────────────
    # Truncate very long experience sections to avoid noise
    # but keep enough for semantic understanding (2000 chars ≈ ~400 words)
    exp_raw = _safe(structured.get("experience_raw"), max_len=2000)
    if exp_raw:
        parts.append(f"Work Experience: {exp_raw}")

    # ── Education ──────────────────────────────────────────────────
    edu_raw = _safe(structured.get("education_raw"), max_len=1000)
    if edu_raw:
        parts.append(f"Education: {edu_raw}")

    # ── Certifications ─────────────────────────────────────────────
    certs = _safe(structured.get("certifications"))
    if certs:
        parts.append(f"Certifications: {certs}")

    # ── Languages ──────────────────────────────────────────────────
    languages = _safe(structured.get("languages"))
    if languages:
        parts.append(f"Languages: {languages}")

    # ── Location ───────────────────────────────────────────────────
    location = _safe(structured.get("location"))
    if location:
        parts.append(f"Location: {location}")

    profile = "\n".join(parts)

    # ── Clean up whitespace ────────────────────────────────────────
    profile = re.sub(r'\n{3,}', '\n\n', profile)   # Max 2 consecutive newlines
    profile = re.sub(r' {2,}', ' ', profile)        # Collapse multiple spaces
    profile = profile.strip()

    # ── HARD CHARACTER CAP — The Context Window Bouncer! 🚪 ────────
    # nomic-embed-text has an 8192 token limit (~1 token ≈ 4 chars).
    # That's ~32K chars theoretically, BUT Ollama applies a stricter
    # limit in practice — very long resumes (walls of skills, 10-year
    # experience dumps) blow past it and get a 400 Bad Request.
    #
    # Think of it as the bouncer at the embedding club: "Sorry honey,
    # the venue only holds 5000 characters. Anything more and we're
    # shutting the doors!" 🚫✨
    #
    # 5000 chars ≈ ~1250 tokens — well within the safe zone for
    # nomic-embed-text, and still captures the most important
    # semantic content of a candidate's profile. Quality over
    # quantity, darling! 💅
    MAX_PROFILE_CHARS = 5000
    if len(profile) > MAX_PROFILE_CHARS:
        profile = profile[:MAX_PROFILE_CHARS]
        # ── Snap to last complete line to avoid cutting mid-sentence ─
        # Cutting mid-word produces garbage tokens at the boundary.
        # Snapping to the last newline keeps sentences whole! ✂️✨
        last_newline = profile.rfind('\n')
        if last_newline > MAX_PROFILE_CHARS * 0.7:  # Only snap if cut isn't too drastic
            profile = profile[:last_newline]
        logger.debug(f"✂️ Profile truncated to {len(profile)} chars (was over {MAX_PROFILE_CHARS})")

    return profile


def build_candidate_metadata(structured: Dict) -> Dict[str, Any]:
    """
    Build metadata dict for Qdrant payload storage.

    Qdrant stores payload (metadata) alongside vectors for filtering.
    We store key fields as payload so we can filter results
    (e.g., "only show candidates with Python skills").

    💅 UPGRADE NOTE from ChromaDB:
    Qdrant payloads are MORE flexible than ChromaDB metadata!
    ChromaDB restricted values to str/int/float/bool only.
    Qdrant accepts nested dicts, lists, and arbitrary JSON! 🎉
    But we keep it flat for simplicity and compatibility.
    """
    return {
        "candidate_id": int(structured.get("candidate_id", 0)),
        "name": (structured.get("name") or "Unknown")[:100],
        "email": (structured.get("email") or "")[:100],
        "phone": (structured.get("phone") or "")[:50],
        "location": (structured.get("location") or "")[:100],
        "skills": (structured.get("skills_raw") or "")[:500],
        "extraction_status": (structured.get("extraction_status") or "")[:50],
        "ai_assisted": bool(structured.get("ai_assisted")),
    }


def _candidate_id_to_point_id(candidate_id: Any) -> str:
    """
    Convert a candidate_id to a Qdrant-compatible point ID string.

    💅 WHY NOT JUST USE INTEGERS? — The UUID Convention! 🎭
    Qdrant supports both integer and UUID string point IDs.
    We use a deterministic UUID-like hash so that:
      1. Re-indexing the SAME candidate always produces the SAME point ID
         (idempotent upserts — no duplicates, ever! 🔒)
      2. We can reconstruct the point ID from candidate_id alone
         (for single-candidate re-indexing and lookups)
      3. No collisions even if candidate_ids have gaps or weird numbering

    Think of it as giving each candidate a permanent backstage pass
    that never changes, no matter how many times they audition! 🎫✨

    Args:
        candidate_id: The raw candidate_id from SQLite (int or str).

    Returns:
        A deterministic UUID-format string for Qdrant.
    """
    # Create a deterministic hash from the candidate_id
    # Using MD5 for speed (not security) — truncated to UUID format
    raw = f"cand_{candidate_id}"
    hex_hash = hashlib.md5(raw.encode()).hexdigest()
    # Format as UUID: 8-4-4-4-12
    return f"{hex_hash[:8]}-{hex_hash[8:12]}-{hex_hash[12:16]}-{hex_hash[16:20]}-{hex_hash[20:32]}"


# =============================================================================
# 🔍 VECTOR SEARCH ENGINE
# =============================================================================

class VectorSearchEngine:
    """
    💅✨ THE VECTOR SEARCH DIVA! ✨💅

    She takes job descriptions and finds the most matching candidates
    using semantic similarity. Not just keyword matching, honey —
    she UNDERSTANDS what the job needs! 🧠💘

    Now powered by Qdrant — the STABLE, drama-free vector database
    that never breaks her API protocol between versions! 👑

    Usage:
        engine = VectorSearchEngine()
        engine.index_all_candidates()
        results = engine.search("Looking for a senior Python developer...")

    The engine maintains a Qdrant collection that persists to disk,
    so you only need to index once (and re-index when new candidates
    are added to the database).
    """

    def __init__(
        self,
        db_path: str = RESUME_DB_PATH,
        qdrant_dir: str = QDRANT_PERSIST_DIR,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        """
        Initialize the Vector Search Engine.

        Args:
            db_path:         Path to resume_extractions.db
            qdrant_dir:      Directory for Qdrant persistent storage
            embedding_model: Ollama model name for embeddings
        """
        self.db_path = db_path
        self.qdrant_dir = qdrant_dir
        self.embedding_model = embedding_model

        # ── Validate dependencies ──────────────────────────────────
        if not QDRANT_AVAILABLE:
            raise RuntimeError(
                "Qdrant client not installed!\n"
                "Run: pip install qdrant-client --break-system-packages"
            )
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "Ollama Python client not installed!\n"
                "Run: pip install ollama --break-system-packages"
            )

        # ── Initialize embedding function ──────────────────────────
        # Plain helper that calls Ollama — NOT tied to any DB protocol.
        # We call this manually to get vectors, then pass them directly
        # to Qdrant's upsert/query methods. Clean separation! 💅
        self.embed_fn = OllamaEmbeddingFunction(model_name=self.embedding_model)

        # ── Initialize Qdrant client (local persistent mode) ───────
        # 💅 QDRANT LOCAL MODE — The Backstage Storage Room! 🗄️
        # QdrantClient(path=...) stores everything on disk, just like
        # SQLite does for relational data. No server needed! ✨
        # When you're ready to scale, just switch to:
        #   QdrantClient(url="http://localhost:6333")
        # and your code stays EXACTLY the same. That's class, darling! 👑
        self.client = QdrantClient(path=self.qdrant_dir)

        # ── Ensure collection exists with correct vector config ────
        # Qdrant needs to know the vector dimensions AND distance
        # metric at collection creation time. Unlike ChromaDB (which
        # let you change hnsw:space and then silently gave wrong results),
        # Qdrant locks this in and TELLS YOU if there's a mismatch! 🔒✨
        self._ensure_collection()

        # ── Create text index for skill/location filtering ─────────
        # 💅 TEXT INDEX — The Smart Filing System! 📁
        # This lets Qdrant do fast full-text search on the profile_text
        # field, so skill_filter and location_filter work efficiently
        # even with thousands of candidates. Without this index,
        # every filter would scan ALL documents — like reading every
        # resume in the filing cabinet instead of checking the labels! 🏷️
        self._ensure_text_index()

        count = self._get_count()
        logger.info(
            f"✨ VectorSearchEngine initialized! "
            f"Collection '{COLLECTION_NAME}' has {count} documents."
        )

    def _ensure_collection(self):
        """
        Create the Qdrant collection if it doesn't exist yet.

        Uses cosine distance — the GOLD STANDARD for NLP embeddings! 🥇
        Cosine measures the ANGLE between vectors (direction = meaning),
        not raw distance (length = intensity). Like judging if two people
        are facing the same career destination, not whether they're the
        same height! 🧭✨
        """
        try:
            if not self.client.collection_exists(COLLECTION_NAME):
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.embed_fn.dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    f"📦 Created Qdrant collection '{COLLECTION_NAME}' "
                    f"({self.embed_fn.dimensions}D, cosine distance)"
                )
            else:
                # ── Validate existing collection dimensions match ───
                # If someone switched embedding models without rebuilding,
                # the dimensions won't match and everything silently breaks.
                # Better to catch it NOW with a clear error message! 🚨
                info = self.client.get_collection(COLLECTION_NAME)
                existing_dim = info.config.params.vectors.size
                if existing_dim != self.embed_fn.dimensions:
                    logger.warning(
                        f"⚠️ Dimension mismatch! Collection has {existing_dim}D "
                        f"but embedding model produces {self.embed_fn.dimensions}D. "
                        f"Run rebuild_index() to fix this!"
                    )
        except Exception as e:
            logger.error(f"❌ Failed to ensure collection: {e}")
            raise

    def _ensure_text_index(self):
        """
        Create a full-text index on the profile_text payload field.

        This enables fast $contains-style filtering for skill_filter
        and location_filter. Without it, text search would be a
        brute-force scan — like searching every page of every resume
        instead of using the index tabs! 📑✨

        Safe to call multiple times — Qdrant ignores duplicate index
        creation (idempotent). No drama! 🎭
        """
        try:
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="profile_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=40,
                    lowercase=True,
                ),
            )
        except Exception:
            # Qdrant may raise if index already exists (version-dependent).
            # Either way, not fatal — filtering still works, just slower
            # on first queries until the index is built. 🏗️
            pass

    def _get_count(self) -> int:
        """Get the number of points in the collection — safely."""
        try:
            return self.client.count(COLLECTION_NAME).count
        except Exception:
            return 0

    # ─────────────────────────────────────────────────────────────────
    # 📥 INDEXING — Load candidates into the vector database
    # ─────────────────────────────────────────────────────────────────

    def index_all_candidates(self, batch_size: int = 50) -> Dict[str, Any]:
        """
        Index ALL candidates from resume_extractions.db into Qdrant.

        This reads every candidate's structured extraction, builds a
        profile text, embeds it, and stores it in Qdrant.

        Think of this as the casting call — every candidate gets
        their headshot and resume filed in the talent database! 📸💼

        Args:
            batch_size: How many candidates to embed at once.
                        Larger = faster but uses more memory.

        Returns:
            Dict with stats: {indexed, skipped, errors, total_time}
        """
        start_time = time.time()
        stats = {"indexed": 0, "skipped": 0, "errors": 0, "already_indexed": 0}

        # ── Get existing IDs to avoid re-indexing ──────────────────
        # 🚀 PERFORMANCE: Use scroll() with no payload/vectors to get
        # just the point IDs. Like asking the filing clerk for folder
        # labels only, not the entire contents! 📁✨
        existing_ids = set()
        try:
            offset = None
            while True:
                # Qdrant scroll returns (points, next_offset)
                # next_offset is None when there are no more results
                points, next_offset = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=1000,  # Fetch in chunks of 1000
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                existing_ids.update(str(p.id) for p in points)
                if next_offset is None:
                    break
                offset = next_offset

            stats["already_indexed"] = len(existing_ids)
            if existing_ids:
                logger.info(f"📋 Found {len(existing_ids)} already-indexed candidates — skipping them!")
        except Exception as e:
            # Non-fatal: if we can't fetch existing IDs, we'll just
            # upsert everything (Qdrant handles duplicates gracefully).
            logger.warning(f"⚠️ Could not fetch existing IDs (non-fatal): {e}")

        # ── Read candidates from SQLite ────────────────────────────
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row

        # ── Read ONE row per candidate from SQLite ─────────────────
        # 🚨 THE DUPLICATE ID ROOT CAUSE, DARLING! 💀
        # structured_extractions can have MULTIPLE rows per candidate
        # (each re-extraction adds a new row). A plain SELECT * returns
        # ALL of them — so cand_43300 shows up twice.
        #
        # Fix: use GROUP BY candidate_id + MAX(created_at) to get only
        # the MOST RECENT extraction per candidate. Like choosing the
        # latest headshot, not every photo ever taken! 📸✨
        cursor = conn.execute("""
            SELECT s.*
            FROM structured_extractions s
            INNER JOIN (
                SELECT candidate_id, MAX(created_at) AS latest
                FROM structured_extractions
                GROUP BY candidate_id
            ) latest_only
              ON s.candidate_id = latest_only.candidate_id
             AND s.created_at   = latest_only.latest
            ORDER BY s.candidate_id ASC
        """)

        batch_ids = []
        batch_docs = []
        batch_metas = []
        batch_vectors = []

        for row in cursor:
            candidate = dict(row)
            cid = str(candidate.get("candidate_id", ""))
            point_id = _candidate_id_to_point_id(cid)

            # Skip if already indexed in a previous run
            if point_id in existing_ids:
                continue

            # ── Skip if already queued in THIS batch run ───────────
            # Edge case: if two rows somehow share the same candidate_id
            # AND the same created_at timestamp (shouldn't happen, but
            # defensive programming is a virtue, baby! 💪), this guard
            # prevents us from adding the same point_id twice to one batch.
            if point_id in batch_ids:
                logger.warning(f"⚠️ Skipping duplicate point_id in batch: {point_id}")
                continue

            # Build profile text
            profile = build_candidate_profile(candidate)
            if not profile or len(profile) < 20:
                stats["skipped"] += 1
                logger.debug(f"⏭️ Skipping candidate {cid}: profile too short")
                continue

            # Build metadata (Qdrant calls it "payload")
            metadata = build_candidate_metadata(candidate)
            # ── Store profile text in payload for text filtering ────
            # This lets us use MatchText filters on skill_filter and
            # location_filter without needing a separate index table! 📝
            metadata["profile_text"] = profile

            batch_ids.append(point_id)
            batch_docs.append(profile)
            batch_metas.append(metadata)

            # ── Process batch when full ────────────────────────────
            if len(batch_ids) >= batch_size:
                try:
                    batch_embeddings = self.embed_fn(batch_docs)

                    # ── Build Qdrant PointStruct objects ───────────
                    # Each point = (id, vector, payload). Simple and clean!
                    points = [
                        PointStruct(
                            id=batch_ids[i],
                            vector=batch_embeddings[i],
                            payload=batch_metas[i],
                        )
                        for i in range(len(batch_ids))
                    ]

                    self.client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )
                    stats["indexed"] += len(batch_ids)
                    # ── Track newly indexed IDs ────────────────────
                    # Add to existing_ids so if this function is ever
                    # called in a loop, we don't re-add them! 🔒
                    existing_ids.update(batch_ids)
                    logger.info(
                        f"📥 Indexed batch: {stats['indexed']} candidates so far..."
                    )
                except Exception as e:
                    stats["errors"] += len(batch_ids)
                    logger.error(f"❌ Batch indexing error: {e}")
                finally:
                    # ── ALWAYS clear the batch, success OR failure! ─
                    # 🚨 CRITICAL: Without `finally`, a failed batch
                    # carries its IDs into the NEXT batch, causing
                    # duplicate errors! `finally` runs whether try
                    # succeeded or except fired.
                    # Like clearing the stage between acts — ALWAYS! 🎭✨
                    batch_ids, batch_docs, batch_metas = [], [], []

        # ── Process remaining final batch ──────────────────────────
        if batch_ids:
            try:
                batch_embeddings = self.embed_fn(batch_docs)
                points = [
                    PointStruct(
                        id=batch_ids[i],
                        vector=batch_embeddings[i],
                        payload=batch_metas[i],
                    )
                    for i in range(len(batch_ids))
                ]
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                )
                stats["indexed"] += len(batch_ids)
                existing_ids.update(batch_ids)
            except Exception as e:
                stats["errors"] += len(batch_ids)
                logger.error(f"❌ Final batch error: {e}")
            finally:
                batch_ids, batch_docs, batch_metas = [], [], []

        conn.close()

        stats["total_time"] = round(time.time() - start_time, 2)
        stats["total_in_collection"] = self._get_count()

        logger.info(
            f"✅ Indexing complete! "
            f"{stats['indexed']} new + {stats['already_indexed']} existing = "
            f"{stats['total_in_collection']} total candidates in vector DB "
            f"({stats['total_time']}s)"
        )

        return stats

    def index_single_candidate(self, candidate_id: int) -> bool:
        """
        Index (or re-index) a single candidate.

        Useful after corrections or re-extraction. Like updating
        someone's headshot after a makeover! 💅📸

        Args:
            candidate_id: The candidate's ID.

        Returns:
            True if indexed successfully, False otherwise.
        """
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row

        row = conn.execute("""
            SELECT * FROM structured_extractions
            WHERE candidate_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (candidate_id,)).fetchone()

        conn.close()

        if not row:
            logger.warning(f"⚠️ Candidate {candidate_id} not found")
            return False

        candidate = dict(row)
        profile = build_candidate_profile(candidate)
        if not profile or len(profile) < 20:
            logger.warning(f"⚠️ Candidate {candidate_id} profile too short")
            return False

        metadata = build_candidate_metadata(candidate)
        # Store profile text in payload for text filtering
        metadata["profile_text"] = profile
        point_id = _candidate_id_to_point_id(candidate_id)

        try:
            # Pre-compute embedding manually
            embedding = self.embed_fn([profile])

            # Upsert = insert if new, update if already exists ✨
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding[0],
                        payload=metadata,
                    )
                ],
            )
            logger.info(f"📥 Indexed candidate {candidate_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to index candidate {candidate_id}: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────
    # 🔍 SEARCH — Find candidates matching a job description
    # ─────────────────────────────────────────────────────────────────

    def search(
        self,
        job_description: str,
        top_k: int = DEFAULT_TOP_K,
        skill_filter: Optional[List[str]] = None,
        location_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for candidates matching a job description.

        This is the MAIN EVENT, darling! 🌟 You paste a job description,
        and the engine finds the most relevant candidates using semantic
        similarity — not just keyword matching!

        Args:
            job_description: The job description text to match against.
            top_k:           Number of top results to return (default 10).
            skill_filter:    Optional list of must-have skills (keyword filter).
            location_filter: Optional location string to filter by.

        Returns:
            Dict with:
              - results: List of matched candidates with scores
              - query_info: Metadata about the search
              - total_candidates: How many are in the DB
        """
        collection_count = self._get_count()

        if not job_description or len(job_description.strip()) < 10:
            return {
                "results": [],
                "query_info": {"error": "Job description too short (min 10 chars)"},
                "total_candidates": collection_count,
            }

        top_k = min(top_k, MAX_TOP_K)
        start_time = time.time()

        # ── Build Qdrant filter ────────────────────────────────────
        # 💅 QDRANT'S FILTERING — Clean and Drama-Free! 🎭
        #
        # Unlike ChromaDB's confusing where vs where_document split
        # (and the $contains eviction of 1.x), Qdrant has ONE filter
        # system that works on payload fields. We use MatchText on
        # the profile_text field — same effect as ChromaDB's old
        # where_document={$contains: ...}, but with a stable API! 💪✨
        #
        # MatchText does tokenized full-text search:
        #   "Python" matches "Python developer", "python scripting", etc.
        # This is actually BETTER than ChromaDB's substring match
        # because it handles case and word boundaries properly! 🎯
        query_filter = None
        filter_conditions = []

        if skill_filter:
            # Each skill must appear somewhere in the candidate profile.
            # We limit to 5 to avoid over-constraining the search — more
            # filters = fewer results, which can be TOO restrictive! 🎯
            for skill in skill_filter[:5]:
                skill_clean = skill.strip()
                if skill_clean:
                    filter_conditions.append(
                        FieldCondition(
                            key="profile_text",
                            match=MatchText(text=skill_clean),
                        )
                    )

        if location_filter:
            # Location text is embedded in the profile as "Location: ..."
            # so MatchText will find it naturally! 📍
            location_clean = location_filter.strip()
            if location_clean:
                filter_conditions.append(
                    FieldCondition(
                        key="profile_text",
                        match=MatchText(text=location_clean),
                    )
                )

        # ── Assemble the filter ────────────────────────────────────
        if filter_conditions:
            # ALL conditions must match (AND logic) — the candidate must
            # have ALL required skills AND be in the right location! 💼
            query_filter = Filter(must=filter_conditions)

        # ── Safety check: empty collection ─────────────────────────
        if collection_count == 0:
            logger.warning("⚠️ Vector collection is empty! Run index_all_candidates() first.")
            return {
                "results": [],
                "query_info": {
                    "error": "Collection is empty. Please index candidates first.",
                    "top_k": top_k,
                    "skill_filter": skill_filter,
                    "location_filter": location_filter,
                    "search_time_seconds": round(time.time() - start_time, 3),
                    "jd_length": len(job_description),
                },
                "total_candidates": 0,
            }

        # ── Pre-compute query embedding ────────────────────────────
        # We embed the job description ourselves and pass the raw vector
        # to Qdrant's query. Same cosine space as indexed docs! 🍎
        try:
            query_vectors = self.embed_fn([job_description])
            if not query_vectors or len(query_vectors) == 0:
                raise RuntimeError("Ollama returned empty embedding for query!")
        except Exception as e:
            logger.error(f"❌ Failed to embed job description: {e}")
            return {
                "results": [],
                "query_info": {"error": f"Query embedding failed: {e}"},
                "total_candidates": collection_count,
            }

        # ── Query Qdrant with pre-computed vector ──────────────────
        try:
            # ── Clamp top_k to collection size ─────────────────────
            # Qdrant is more forgiving than ChromaDB here (it just
            # returns fewer results instead of erroring), but let's
            # be explicit about it for cleaner logging! 🎯
            safe_limit = min(top_k, collection_count)

            search_results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vectors[0],  # Single query vector
                limit=safe_limit,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {
                "results": [],
                "query_info": {"error": str(e)},
                "total_candidates": collection_count,
            }

        # ── Format results ─────────────────────────────────────────
        results = []
        if search_results and search_results.points:
            for i, point in enumerate(search_results.points):
                payload = point.payload or {}

                # ── Qdrant cosine similarity score ─────────────────
                # Qdrant returns SIMILARITY directly (0.0 → 1.0 for cosine):
                #   1.0 = identical vectors (perfect match! 💯)
                #   0.0 = orthogonal (completely unrelated)
                #  <0.0 = opposite (theoretically possible but rare)
                #
                # Convert to 0-100% for human-friendly display:
                similarity = max(0.0, min(100.0, point.score * 100.0))

                # ── Extract profile preview from payload ───────────
                profile_text = payload.get("profile_text", "")
                preview = profile_text[:300] if profile_text else ""

                results.append({
                    "rank": i + 1,
                    "candidate_id": int(payload.get("candidate_id", 0)),
                    "name": payload.get("name", "Unknown"),
                    "email": payload.get("email", ""),
                    "phone": payload.get("phone", ""),
                    "location": payload.get("location", ""),
                    "skills": payload.get("skills", ""),
                    "similarity_score": round(similarity, 1),
                    "distance": round(1.0 - point.score, 4),  # For backward compat
                    "profile_preview": preview,
                })

        search_time = round(time.time() - start_time, 3)

        logger.info(
            f"🔍 Search complete: {len(results)} results in {search_time}s "
            f"(top score: {results[0]['similarity_score'] if results else 0}%)"
        )

        return {
            "results": results,
            "query_info": {
                "top_k": top_k,
                "skill_filter": skill_filter,
                "location_filter": location_filter,
                "search_time_seconds": search_time,
                "jd_length": len(job_description),
            },
            "total_candidates": self._get_count(),
        }

    # ─────────────────────────────────────────────────────────────────
    # 📊 MANAGEMENT — Stats, rebuild, clear
    # ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get stats about the vector database."""
        return {
            "collection_name": COLLECTION_NAME,
            "total_documents": self._get_count(),
            "embedding_model": self.embedding_model,
            "qdrant_dir": self.qdrant_dir,
            "resume_db": self.db_path,
            "resume_db_exists": os.path.exists(self.db_path),
        }

    def rebuild_index(self) -> Dict[str, Any]:
        """
        Completely rebuild the vector index from scratch.

        Drops the existing collection and re-indexes everything.
        Use after major changes to the extraction pipeline.
        Like a complete wardrobe overhaul! 👗🔄

        💅 QDRANT UPGRADE NOTE:
        Unlike ChromaDB's rebuild (which triggered the `.name()` error
        on get_or_create_collection with a custom embedding function),
        Qdrant's rebuild is clean: delete collection → create fresh →
        re-index. No protocol drama, no AttributeError ambush! ✨
        """
        logger.info("🔄 Rebuilding vector index from scratch...")
        try:
            if self.client.collection_exists(COLLECTION_NAME):
                self.client.delete_collection(COLLECTION_NAME)
                logger.info(f"🗑️ Deleted old collection '{COLLECTION_NAME}'")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete old collection (non-fatal): {e}")

        # ── Recreate collection with fresh config ──────────────────
        self._ensure_collection()
        self._ensure_text_index()

        return self.index_all_candidates()

    def clear_index(self):
        """Delete all documents from the vector database."""
        try:
            if self.client.collection_exists(COLLECTION_NAME):
                self.client.delete_collection(COLLECTION_NAME)
            self._ensure_collection()
            self._ensure_text_index()
            logger.info("🗑️ Vector index cleared!")
        except Exception as e:
            logger.error(f"❌ Failed to clear index: {e}")


# =============================================================================
# 🚀 CLI ENTRY POINT (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " 🧚‍♀️✨ VECTOR SEARCH ENGINE — CLI ✨🧚‍♀️ ".center(60) + "║")
    print("╠" + "═" * 60 + "╣")
    print("║" + f"  📂 Resume DB: {RESUME_DB_PATH}".ljust(60) + "║")
    print("║" + f"  🧠 Embedding: {EMBEDDING_MODEL}".ljust(60) + "║")
    print("║" + f"  📦 Qdrant:    {QDRANT_PERSIST_DIR}/".ljust(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()

    # ── Initialize engine ──────────────────────────────────────────
    try:
        engine = VectorSearchEngine()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # ── CLI Menu ───────────────────────────────────────────────────
    while True:
        print("\n" + "─" * 50)
        print("  1. Index all candidates")
        print("  2. Rebuild index (from scratch)")
        print("  3. Search by job description")
        print("  4. View stats")
        print("  5. Exit")
        print("─" * 50)

        choice = input("Choose (1-5): ").strip()

        if choice == "1":
            stats = engine.index_all_candidates()
            print(f"\n✅ Indexed: {stats['indexed']} new, "
                  f"{stats['already_indexed']} existing, "
                  f"{stats['errors']} errors "
                  f"({stats['total_time']}s)")

        elif choice == "2":
            stats = engine.rebuild_index()
            print(f"\n✅ Rebuilt: {stats['indexed']} candidates ({stats['total_time']}s)")

        elif choice == "3":
            print("\nPaste your job description (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            jd = "\n".join(lines)

            if jd.strip():
                top_k = input("How many results? (default 10): ").strip()
                top_k = int(top_k) if top_k.isdigit() else 10

                results = engine.search(jd, top_k=top_k)
                print(f"\n🔍 Found {len(results['results'])} matches:")
                for r in results["results"]:
                    print(f"  #{r['rank']} [{r['similarity_score']}%] "
                          f"{r['name']} — {r['skills'][:60]}...")

        elif choice == "4":
            stats = engine.get_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")

        elif choice == "5":
            print("\n👋 Bye, darling! ✨")
            break