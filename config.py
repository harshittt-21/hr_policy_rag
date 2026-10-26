import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project root, rather than relying on the current directory.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back if invalid."""
    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        logger.warning(
            "[config] Invalid integer for %s=%r. Using default: %s",
            name, raw_value, default,
        )
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back if invalid."""
    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return float(raw_value)
    except ValueError:
        logger.warning(
            "[config] Invalid float for %s=%r. Using default: %s",
            name, raw_value, default,
        )
        return default


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"

# FAISS is an optional local alternative.
FAISS_DIR = Path(
    os.getenv("FAISS_DIR", str(VECTORSTORE_DIR / "faiss"))
).expanduser()


# -------------------------------------------------------------------
# LLM: Groq
# -------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.0)


# -------------------------------------------------------------------
# Embeddings: Jina API
# -------------------------------------------------------------------

JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
JINA_EMBEDDING_MODEL = os.getenv(
    "JINA_EMBEDDING_MODEL",
    "jina-embeddings-v3",
).strip()

# Sent to Jina as "dimensions" and used for the Qdrant collection size,
# so the two can never drift apart. jina-embeddings-v3 defaults to 1024
# and supports Matryoshka truncation to smaller sizes.
EMBEDDING_DIM = _get_int("EMBEDDING_DIM", 1024)


# -------------------------------------------------------------------
# Vector database: Qdrant Cloud or local FAISS
# -------------------------------------------------------------------

VECTOR_DB = os.getenv("VECTOR_DB", "qdrant").strip().lower()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "hr_policy_cluster",
).strip()


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------

PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)

CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# -------------------------------------------------------------------
# Retrieval pipeline
# -------------------------------------------------------------------

NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)

VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)

RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)

# "simple" avoids PyTorch and cross-encoder dependencies.
SUPPORTED_RERANKERS = ("simple", "cross-encoder")
RERANKER = os.getenv("RERANKER", "simple").strip().lower()

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
).strip()

# The score threshold depends on the reranker's score scale.
# cross-encoder: raw ms-marco logits, so -4.0 is a sensible floor.
# simple: scale depends on your implementation; the default of 0.0
#         assumes non-negative scores, so adjust it to match your scorer.
# MIN_RERANK_SCORE in the environment overrides either default.
_DEFAULT_MIN_RERANK_SCORE = {
    "simple": 0.0,
    "cross-encoder": -4.0,
}
MIN_RERANK_SCORE = _get_float(
    "MIN_RERANK_SCORE",
    _DEFAULT_MIN_RERANK_SCORE.get(RERANKER, 0.0),
)


# -------------------------------------------------------------------
# Conversation memory
# -------------------------------------------------------------------

MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)


# -------------------------------------------------------------------
# Configuration validation
# -------------------------------------------------------------------

def check_llm_config() -> None:
    """Validate the Groq configuration."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env.")

    if not GROQ_MODEL:
        raise ValueError("GROQ_MODEL cannot be empty.")


def check_embedding_config() -> None:
    """Validate the Jina API configuration."""
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set in the environment or .env.")

    if not JINA_EMBEDDING_MODEL:
        raise ValueError("JINA_EMBEDDING_MODEL cannot be empty.")

    if EMBEDDING_DIM <= 0:
        raise ValueError(f"EMBEDDING_DIM must be positive, got {EMBEDDING_DIM}.")


def check_vector_store_config() -> None:
    """Validate the selected vector database."""
    if VECTOR_DB == "qdrant":
        if not QDRANT_URL:
            raise ValueError("VECTOR_DB=qdrant but QDRANT_URL is empty.")
        if not QDRANT_API_KEY:
            raise ValueError("VECTOR_DB=qdrant but QDRANT_API_KEY is empty.")
        if not QDRANT_COLLECTION_NAME:
            raise ValueError("QDRANT_COLLECTION_NAME cannot be empty.")

    elif VECTOR_DB == "faiss":
        pass  # Local persistence; no remote credentials required.

    else:
        raise ValueError(
            f"Unsupported VECTOR_DB={VECTOR_DB!r}. "
            "Choose 'qdrant' or 'faiss'."
        )


def check_chunking_config() -> None:
    """Validate chunk sizes and overlaps."""
    sizes = {
        "PARENT_CHUNK_SIZE": PARENT_CHUNK_SIZE,
        "CHILD_CHUNK_SIZE": CHILD_CHUNK_SIZE,
    }
    for name, value in sizes.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    for name, value in {
        "PARENT_CHUNK_OVERLAP": PARENT_CHUNK_OVERLAP,
        "CHILD_CHUNK_OVERLAP": CHILD_CHUNK_OVERLAP,
    }.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative, got {value}.")

    if PARENT_CHUNK_OVERLAP >= PARENT_CHUNK_SIZE:
        raise ValueError(
            "PARENT_CHUNK_OVERLAP must be smaller than PARENT_CHUNK_SIZE."
        )

    if CHILD_CHUNK_OVERLAP >= CHILD_CHUNK_SIZE:
        raise ValueError(
            "CHILD_CHUNK_OVERLAP must be smaller than CHILD_CHUNK_SIZE."
        )

    if CHILD_CHUNK_SIZE >= PARENT_CHUNK_SIZE:
        raise ValueError(
            "CHILD_CHUNK_SIZE must be smaller than PARENT_CHUNK_SIZE."
        )


def check_retrieval_config() -> None:
    """Validate retrieval and reranking settings."""
    if RERANKER not in SUPPORTED_RERANKERS:
        raise ValueError(
            f"Unsupported RERANKER={RERANKER!r}. "
            f"Choose one of: {', '.join(SUPPORTED_RERANKERS)}."
        )

    if RERANKER == "cross-encoder" and not CROSS_ENCODER_MODEL:
        raise ValueError(
            "RERANKER=cross-encoder but CROSS_ENCODER_MODEL is empty."
        )

    for name, value in {
        "NUM_QUERY_VARIANTS": NUM_QUERY_VARIANTS,
        "VECTOR_TOP_K": VECTOR_TOP_K,
        "BM25_TOP_K": BM25_TOP_K,
        "HYBRID_TOP_K": HYBRID_TOP_K,
        "RERANK_TOP_K": RERANK_TOP_K,
        "MAX_CONTEXT_CHARS": MAX_CONTEXT_CHARS,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    if RERANK_TOP_K > HYBRID_TOP_K:
        raise ValueError("RERANK_TOP_K cannot exceed HYBRID_TOP_K.")

    if MEMORY_WINDOW < 0:
        raise ValueError(f"MEMORY_WINDOW cannot be negative, got {MEMORY_WINDOW}.")


def check_config() -> None:
    """Validate all required configuration for the selected setup."""
    check_llm_config()
    check_embedding_config()
    check_vector_store_config()
    check_chunking_config()
    check_retrieval_config()


def describe_settings() -> dict:
    """Return safe-to-display settings without exposing credentials."""
    return {
        "LLM Provider": "Groq",
        "Chat Model": GROQ_MODEL,
        "Embedding Provider": "Jina",
        "Embedding Model": JINA_EMBEDDING_MODEL,
        "Embedding Dimensions": EMBEDDING_DIM,
        "Vector Store": VECTOR_DB,
        "Qdrant Collection": (
            QDRANT_COLLECTION_NAME if VECTOR_DB == "qdrant" else "N/A"
        ),
        "Reranker": RERANKER,
        "Min Rerank Score": MIN_RERANK_SCORE,
        "Parent Chunk Size": PARENT_CHUNK_SIZE,
        "Child Chunk Size": CHILD_CHUNK_SIZE,
        "Query Variants": NUM_QUERY_VARIANTS,
    }