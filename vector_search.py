"""
vector_search.py

💅✨ FAIRY CODEMOTHER'S VECTOR SEARCH ENGINE ✨💅

Semantic candidate matching powered by vector embeddings!
Paste a job description → get the most relevant candidates
ranked by semantic similarity. Think of it as a talent matchmaker
who actually UNDERSTANDS what the job needs! 💘🎯

Architecture:
  - Qdrant     → High-performance vector database (ChromaDB's stable, drama-free cousin!)
  - Ollama     → Local embedding model (mxbai-embed-large, 1024 dimensions)
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
    ollama pull mxbai-embed-large

Usage:
    from vector_search import VectorSearchEngine

    engine = VectorSearchEngine()
    engine.index_all_candidates()                         # Build the index
    results = engine.search("Senior Python developer...")  # Search!

Dependencies:
    pip install qdrant-client --break-system-packages
    ollama pull mxbai-embed-large

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

# 🆕 Traditional Method: Hybrid Role Predictor (.pkl + Keywords)
try:
    from job_role_predictor import HybridClassifier
    JOB_ROLE_PREDICTOR_AVAILABLE = True
except ImportError:
    JOB_ROLE_PREDICTOR_AVAILABLE = False

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

# Embedding model — mxbai-embed-large is high quality and supports embeddings!
# Other options: nomic-embed-text (768d), all-minilm
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")

# Collection name in Qdrant
COLLECTION_NAME = "candidate_profiles"

# Search defaults
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# Embedding dimensions — mxbai-embed-large = 1024, nomic-embed-text = 768.
# Auto-detected at runtime from the test embedding, but this is the fallback.
DEFAULT_EMBEDDING_DIM = 1024


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
    # mxbai-embed-large has a 512 token limit (~1 token ≈ 4 chars).
    # 1500 chars ≈ ~375 tokens — safely below the 512-token ceiling.
    # Previously at 2000 chars some profiles hit the limit (dense text,
    # multi-byte chars, or short words all inflate token count vs chars).
    MAX_PROFILE_CHARS = 1500
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


def build_candidate_metadata(structured: Dict, predictor: Optional[Any] = None) -> Dict[str, Any]:
    """
    Build metadata dict for Qdrant payload storage.

    Qdrant stores payload (metadata) alongside vectors for filtering.
    We store key fields as payload so we can filter results
    (e.g., "only show candidates with Python skills").

    💅 TRADITIONAL METHOD INTEGRATION:
    We use the .pkl-powered JobRolePredictor to assign a Role and Function
    to the candidate based on their profile text.
    """
    profile_text = build_candidate_profile(structured)
    
    metadata = {
        "candidate_id": int(structured.get("candidate_id", 0)),
        "name": (structured.get("name") or "Unknown")[:100],
        "email": (structured.get("email") or "")[:100],
        "phone": (structured.get("phone") or "")[:50],
        "location": (structured.get("location") or "")[:100],
        "skills": (structured.get("skills_raw") or "")[:500],
        "extraction_status": (structured.get("extraction_status") or "")[:50],
        "ai_assisted": bool(structured.get("ai_assisted")),
        "profile_text": profile_text,
    }

    # ── Enrich with Traditional ML Role Prediction ────────────────
    if predictor and profile_text:
        try:
            # Using Hybrid prediction for better accuracy! ✨
            prediction = predictor.classify_text_only(profile_text)
            metadata["predicted_role"] = prediction.get("predicted_role") or "Unknown"
            metadata["function"] = prediction.get("function") or "Others"
        except Exception as e:
            logger.warning(f"⚠️ Metadata role prediction failed: {e}")
            metadata["predicted_role"] = "Unknown"
            metadata["function"] = "Others"
    else:
        metadata["predicted_role"] = "Unknown"
        metadata["function"] = "Others"

    return metadata


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

        # ── Initialize Traditional Role Predictor ──────────────────
        self.predictor = None
        if JOB_ROLE_PREDICTOR_AVAILABLE:
            try:
                # Use absolute paths for the .pkl files to ensure they load
                # regardless of where the script is called from. 📂✨
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "job_role_prediction_model.pkl")
                vectorizer_path = os.path.join(base_dir, "combined_tfidf_vectorizer1__1_.pkl")
                
                # Using HybridClassifier for better accuracy (ML + Keywords!) 🤝✨
                self.predictor = HybridClassifier(
                    model_path=model_path,
                    vectorizer_path=vectorizer_path
                )
                logger.info("✨ Hybrid Role Predictor (ML + Keywords) loaded successfully!")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Role Predictor: {e}")

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

        # ── Create role index for fast filtering ───────────────────
        self._ensure_payload_indexes()

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
        self._dim_mismatch = False  # reset flag each call

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
                info = self.client.get_collection(COLLECTION_NAME)
                existing_dim = info.config.params.vectors.size
                if existing_dim != self.embed_fn.dimensions:
                    self._dim_mismatch = True
                    logger.error(
                        f"❌ Dimension mismatch! Collection has {existing_dim}D "
                        f"but '{self.embedding_model}' produces {self.embed_fn.dimensions}D. "
                        f"Use 'Rebuild All' to fix."
                    )
        except Exception as e:
            logger.error(f"❌ Failed to ensure collection: {e}")
            raise

    def _ensure_payload_indexes(self):
        """Create indexes for frequently filtered fields."""
        try:
            # Predicted Role (Keyword match)
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="predicted_role",
                field_schema="keyword",
            )
            # Function (Keyword match)
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="function",
                field_schema="keyword",
            )
        except Exception:
            pass

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

    def _upsert_batch(
        self,
        ids: List[str],
        docs: List[str],
        metas: List[Dict],
        existing_ids: set,
    ) -> Tuple[int, int]:
        """
        Embed and upsert one batch. Returns (indexed_count, error_count).

        On context-length errors the batch is retried one document at a time
        so only the genuinely oversized profiles are skipped, not the whole
        batch of 50.
        """
        try:
            embeddings = self.embed_fn(docs)
            points = [
                PointStruct(id=ids[i], vector=embeddings[i], payload=metas[i])
                for i in range(len(ids))
            ]
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            existing_ids.update(ids)
            return len(ids), 0
        except RuntimeError as e:
            # context-length errors come through as RuntimeError from OllamaEmbeddingFunction
            if "context length" in str(e).lower() or "input length" in str(e).lower():
                logger.warning(
                    f"⚠️ Batch of {len(ids)} hit context limit — retrying one-by-one to preserve quality..."
                )
                indexed, errors = 0, 0
                for i in range(len(ids)):
                    try:
                        # ── Step 1: Try at FULL length first ───────────────────
                        # Most items in a failed batch are actually fine; only
                        # one or two usually cause the batch overflow.
                        emb = self.embed_fn([docs[i]])
                        self.client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[PointStruct(
                                id=ids[i], vector=emb[0], payload=metas[i]
                            )],
                        )
                        existing_ids.add(ids[i])
                        indexed += 1
                    except RuntimeError as inner_e:
                        # ── Step 2: Fallback to truncation ONLY if needed ──────
                        if "context length" in str(inner_e).lower() or "input length" in str(inner_e).lower():
                            logger.info(f"✂️ Truncating oversized profile for candidate {metas[i].get('candidate_id')}")
                            try:
                                # mxbai-embed-large supports 512 tokens (~2000 chars)
                                # 1200 is very safe and much better than 800! ✨
                                short_doc = docs[i][:1200]
                                emb = self.embed_fn([short_doc])
                                self.client.upsert(
                                    collection_name=COLLECTION_NAME,
                                    points=[PointStruct(
                                        id=ids[i], vector=emb[0], payload=metas[i]
                                    )],
                                )
                                existing_ids.add(ids[i])
                                indexed += 1
                            except Exception as final_e:
                                logger.error(f"❌ Failed even after truncation for {metas[i].get('candidate_id')}: {final_e}")
                                errors += 1
                        else:
                            logger.error(f"❌ Individual indexing error for {metas[i].get('candidate_id')}: {inner_e}")
                            errors += 1
                    except Exception as inner_e:
                        logger.error(
                            f"❌ Skipping candidate {metas[i].get('candidate_id')}: {inner_e}"
                        )
                        errors += 1
                return indexed, errors
            # Other errors — skip the whole batch
            logger.error(f"❌ Batch indexing error: {e}")
            return 0, len(ids)
        except Exception as e:
            logger.error(f"❌ Batch indexing error: {e}")
            return 0, len(ids)

    def index_all_candidates(self, batch_size: int = 50, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Index ALL candidates from resume_extractions.db into Qdrant.

        Args:
            batch_size:    How many candidates to embed at once.
            force_refresh: If True, re-embed and update all candidates even if they exist.

        Returns:
            Dict with stats: {indexed, skipped, errors, total_time}
        """
        if getattr(self, "_dim_mismatch", False):
            raise RuntimeError(
                f"Dimension mismatch: the Qdrant collection was built with a different "
                f"embedding model. Use 'Rebuild All' (POST /api/search/index with "
                f"{{\"rebuild\": true}}) to drop and re-index from scratch."
            )

        start_time = time.time()
        stats = {"indexed": 0, "skipped": 0, "errors": 0, "already_indexed": 0}

        # ── Get existing IDs ───────────────────────────────────────
        existing_ids = set()
        if not force_refresh:
            try:
                offset = None
                while True:
                    points, next_offset = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        limit=1000,
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
                logger.warning(f"⚠️ Could not fetch existing IDs (non-fatal): {e}")
        else:
            logger.info("🔄 Force refresh enabled — re-indexing all candidates!")

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

            # 💅 REFRESH LOGIC: We no longer skip existing IDs! 
            # This ensures that when you run indexing, the new Hybrid Role 
            # logic is applied to every candidate, updating their "Unknown" pills.
            # Qdrant handles the 'upsert' (update if exists) automatically. 🔄✨
            
            # (Deleted the skip check that was here)

            # ── Skip if already queued in THIS batch run ───────────
            # Edge case: if two rows somehow share the same candidate_id
            # AND the same created_at timestamp (shouldn't happen, but
            # defensive programming is a virtue, baby! 💪), this guard
            # prevents us from adding the same point_id twice to one batch.
            if point_id in batch_ids:
                logger.warning(f"⚠️ Skipping duplicate point_id in batch: {point_id}")
                continue

            # Build metadata (Qdrant calls it "payload")
            # This now includes the .pkl role prediction! ✨
            metadata = build_candidate_metadata(candidate, predictor=self.predictor)
            profile = metadata.get("profile_text", "")

            if not profile or len(profile) < 20:
                stats["skipped"] += 1
                logger.debug(f"⏭️ Skipping candidate {cid}: profile too short")
                continue

            batch_ids.append(point_id)
            batch_docs.append(profile)
            batch_metas.append(metadata)

            # ── Process batch when full ────────────────────────────
            if len(batch_ids) >= batch_size:
                n, e = self._upsert_batch(batch_ids, batch_docs, batch_metas, existing_ids)
                stats["indexed"] += n
                stats["errors"] += e
                if n:
                    logger.info(f"📥 Indexed batch: {stats['indexed']} candidates so far...")
                batch_ids, batch_docs, batch_metas = [], [], []

        # ── Process remaining final batch ──────────────────────────
        if batch_ids:
            n, e = self._upsert_batch(batch_ids, batch_docs, batch_metas, existing_ids)
            stats["indexed"] += n
            stats["errors"] += e
            batch_ids, batch_docs, batch_metas = [], [], []

        conn.close()

        stats["total_time"] = round(time.time() - start_time, 2)
        stats["total_in_collection"] = self._get_count()
        
        # ── Calculate real math for the user ───────────────────────
        # We need to distinguish between truly NEW candidates and those
        # we just UPDATED with new metadata (like Hybrid Roles).
        # Math: total_now - total_at_start = truly_new
        # Math: total_processed - truly_new = updated
        total_at_start = stats["already_indexed"]
        total_now = stats["total_in_collection"]
        truly_new = max(0, total_now - total_at_start)
        updated = max(0, stats["indexed"] - truly_new)

        logger.info(
            f"✅ Indexing complete! "
            f"{truly_new} new + {updated} updated = "
            f"{total_now} total candidates in vector DB "
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
        metadata = build_candidate_metadata(candidate, predictor=self.predictor)
        profile = metadata.get("profile_text", "")
        
        if not profile or len(profile) < 20:
            logger.warning(f"⚠️ Candidate {candidate_id} profile too short")
            return False

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
        use_role_filter: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for candidates matching a job description.

        💅✨ HYBRID SEARCH UPGRADE ✨💅
        1. VECTOR (Ollama): Finds candidates who "feel" like the JD semantically.
        2. TRADITIONAL (.pkl): Analyzes the JD to predict the intended Role/Function.
        3. FILTER (Optional): Narrow results to candidates who match the predicted Role.

        Args:
            job_description: The job description text to match against.
            top_k:           Number of top results to return (default 10).
            skill_filter:    Optional list of must-have skills (keyword filter).
            location_filter: Optional location string to filter by.
            use_role_filter: If True, automatically filter by detected Job Role.

        Returns:
            Dict with results and detected job role info.
        """
        collection_count = self._get_count()

        if getattr(self, "_dim_mismatch", False):
            return {
                "results": [],
                "query_info": {
                    "error": (
                        f"Dimension mismatch: the Qdrant collection was built with a different "
                        f"embedding model ({self.embedding_model} produces "
                        f"{self.embed_fn.dimensions}D but the collection expects a different size). "
                        f"Use 'Rebuild All' to fix."
                    )
                },
                "total_candidates": collection_count,
            }

        if not job_description or len(job_description.strip()) < 10:
            return {
                "results": [],
                "query_info": {"error": "Job description too short (min 10 chars)"},
                "total_candidates": collection_count,
            }

        top_k = min(top_k, MAX_TOP_K)
        start_time = time.time()

        # ── Detect Role of JD (Traditional Method) ────────────────
        jd_prediction = {}
        if self.predictor:
            try:
                # Use classify_text_only() for HybridClassifier! ✨
                jd_prediction = self.predictor.classify_text_only(job_description)
                logger.info(f"🎯 JD Role Detected: {jd_prediction.get('predicted_role')}")
            except Exception as e:
                logger.warning(f"⚠️ JD role prediction failed: {e}")

        # ── Build Qdrant filter ────────────────────────────────────
        query_filter = None
        filter_conditions = []

        if skill_filter:
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
            location_clean = location_filter.strip()
            if location_clean:
                filter_conditions.append(
                    FieldCondition(
                        key="profile_text",
                        match=MatchText(text=location_clean),
                    )
                )

        # ── New: Traditional Role Filter ──────────────────────────
        if use_role_filter and jd_prediction.get("predicted_role"):
            from qdrant_client.models import MatchValue
            filter_conditions.append(
                FieldCondition(
                    key="predicted_role",
                    match=MatchValue(value=jd_prediction["predicted_role"])
                )
            )

        if filter_conditions:
            query_filter = Filter(must=filter_conditions)

        # ── Safety check: empty collection ─────────────────────────
        if collection_count == 0:
            return {
                "results": [],
                "query_info": {
                    "error": "Collection is empty. Please index candidates first.",
                    "jd_role": jd_prediction,
                },
                "total_candidates": 0,
            }

        # ── Pre-compute query embedding ────────────────────────────
        try:
            query_vectors = self.embed_fn([job_description])
        except Exception as e:
            return {
                "results": [],
                "query_info": {"error": f"Query embedding failed: {e}"},
                "total_candidates": collection_count,
            }

        # ── Query Qdrant ───────────────────────────────────────────
        try:
            safe_limit = min(top_k, collection_count)
            search_results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vectors[0],
                limit=safe_limit,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {
                "results": [],
                "query_info": {"error": str(e), "jd_role": jd_prediction},
                "total_candidates": collection_count,
            }

        # ── Format results ─────────────────────────────────────────
        results = []
        if search_results and search_results.points:
            for i, point in enumerate(search_results.points):
                payload = point.payload or {}
                similarity = max(0.0, min(100.0, point.score * 100.0))
                profile_text = payload.get("profile_text", "")

                results.append({
                    "rank": i + 1,
                    "candidate_id": int(payload.get("candidate_id", 0)),
                    "name": payload.get("name", "Unknown"),
                    "email": payload.get("email", ""),
                    "location": payload.get("location", ""),
                    "skills": payload.get("skills", ""),
                    "predicted_role": payload.get("predicted_role", "Unknown"),
                    "function": payload.get("function", "Others"),
                    "similarity_score": round(similarity, 1),
                    "profile_preview": profile_text[:300] if profile_text else "",
                })

        search_time = round(time.time() - start_time, 3)

        return {
            "results": results,
            "query_info": {
                "top_k": top_k,
                "jd_role": jd_prediction,
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
        Use after major changes to the extraction pipeline or after
        switching embedding models (e.g. 768D → 1024D).

        Unlike delete_collection(), we nuke the entire qdrant_dir on disk
        because Qdrant local mode can leave stale HNSW segment files behind
        after a collection delete. Those stale files keep the old dimension
        (e.g. 768D) and cause numpy broadcast errors when the new model
        produces 1024D vectors. Wiping the folder guarantees a clean slate.
        """
        import shutil

        logger.info("🔄 Rebuilding vector index from scratch...")

        # ── Close the client before touching the filesystem ────────
        try:
            self.client.close()
        except Exception:
            pass

        # ── Nuke the entire storage folder ────────────────────────
        # delete_collection() alone leaves stale HNSW segment files on disk
        # which causes "(1024,) into shape (768,)" numpy errors on upsert.
        if os.path.exists(self.qdrant_dir):
            max_retries = 5
            deletion_successful = False
            for i in range(max_retries):
                try:
                    shutil.rmtree(self.qdrant_dir)
                    logger.info(f"🗑️ Deleted Qdrant storage dir '{self.qdrant_dir}'")
                    deletion_successful = True
                    break
                except PermissionError as e:
                    if i < max_retries - 1:
                        logger.warning(f"⚠️ PermissionError while deleting {self.qdrant_dir} (attempt {i+1}/{max_retries}). Retrying in 1s...")
                        time.sleep(1)
                    else:
                        logger.error(f"❌ Failed to delete Qdrant storage dir after {max_retries} attempts: {e}")
                        # Don't raise yet, we want to re-init the client so the engine isn't broken
                except Exception as e:
                    logger.error(f"❌ Unexpected error deleting Qdrant storage dir: {e}")
                    # Re-init client before raising
                    self.client = QdrantClient(path=self.qdrant_dir)
                    raise

        # ── Reinitialize client + collection ──────────────────────
        # We MUST re-init even if deletion failed, otherwise the engine
        # stays in a "closed" state and breaks all subsequent requests!
        self.client = QdrantClient(path=self.qdrant_dir)
        self._ensure_collection()
        self._ensure_text_index()
        self._ensure_payload_indexes()

        if os.path.exists(self.qdrant_dir) and not deletion_successful:
             logger.warning("⚠️ Re-initialized client on existing directory because deletion failed.")

        # ── Force refresh to re-index everything ───────────────────
        return self.index_all_candidates(force_refresh=True)

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
            total_at_start = stats.get("already_indexed", 0)
            total_now = stats.get("total_in_collection", 0)
            truly_new = max(0, total_now - total_at_start)
            updated = max(0, stats.get("indexed", 0) - truly_new)
            
            print(f"\n✅ Indexing complete: {truly_new} new, "
                  f"{updated} updated, "
                  f"{stats['errors']} errors "
                  f"({stats['total_time']}s)")

        elif choice == "2":
            stats = engine.rebuild_index()
            print(f"\n✅ Rebuilt: {stats['indexed']} total candidates ({stats['total_time']}s)")

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
