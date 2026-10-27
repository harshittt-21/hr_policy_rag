import logging
import math
import re
from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

# Score scale used if the cross-encoder fails and we fall back to "simple".
SIMPLE_FALLBACK_MIN_SCORE = 0.0

_TOKEN_RE = re.compile(r"\w+")

# Question words and glue words that say nothing about the topic.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "what", "which", "who", "whom", "how", "when", "where", "why",
    "can", "could", "should", "would", "will", "i", "me", "my", "we", "our",
    "you", "your", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "about", "it", "its", "this", "that", "there", "any", "as",
    "if", "not", "have", "has", "get",
}


def _terms(text: str) -> List[str]:
    """Lowercase, drop stopwords, and strip a plural 's' ("days" -> "day")."""
    terms = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        terms.append(token)
    return terms


# -------------------------------------------------------------------
# Scorers (higher = more relevant)
# -------------------------------------------------------------------

def _simple_scores(query: str, documents: List[Document]) -> List[float]:
    """Keyword-overlap score between 0 and 1, with no extra dependencies.

    Each query term is weighted by how rare it is among the candidates, so a
    distinctive word like "sick" counts for more than "policy", which appears
    in most documents. The score is the weighted share of query terms found
    in the document. A document sharing no query terms scores 0.
    """
    query_terms = set(_terms(query))
    if not query_terms:
        return [0.0] * len(documents)

    doc_terms = [set(_terms(doc.page_content)) for doc in documents]
    n = len(documents)

    weights = {}
    for term in query_terms:
        df = sum(1 for terms in doc_terms if term in terms)
        weights[term] = math.log(1 + n / (df + 0.5))

    total_weight = sum(weights.values())
    return [
        sum(weights[t] for t in query_terms if t in terms) / total_weight
        for terms in doc_terms
    ]


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Load the cross-encoder once. Needs sentence-transformers + PyTorch.

    The model is told explicitly to return raw logits (no sigmoid), so that
    MIN_RERANK_SCORE=-4.0 means the same thing on every library version.
    The constructor argument was renamed between versions (older releases:
    ``default_activation_function``, newer: ``activation_fn``), so we pick
    whichever this installation supports.
    """
    import inspect

    import torch
    from sentence_transformers import CrossEncoder

    kwargs = {}
    params = inspect.signature(CrossEncoder.__init__).parameters
    if "activation_fn" in params:
        kwargs["activation_fn"] = torch.nn.Identity()
    elif "default_activation_function" in params:
        kwargs["default_activation_function"] = torch.nn.Identity()
    else:
        logger.warning(
            "Could not set the cross-encoder activation; score scale is "
            "whatever the model's default is. Run check_cross_encoder.py."
        )

    logger.info("Loading cross-encoder model %s", config.CROSS_ENCODER_MODEL)
    return CrossEncoder(config.CROSS_ENCODER_MODEL, **kwargs)


def _cross_encoder_scores(query: str, documents: List[Document]) -> List[float]:
    """Score each (query, document) pair with the cross-encoder.

    For cross-encoder/ms-marco-MiniLM-L-6-v2 these are raw logits (roughly
    -10 to +10), which is the scale MIN_RERANK_SCORE=-4.0 is meant for.
    """
    model = _load_cross_encoder()
    pairs = [(query, doc.page_content) for doc in documents]
    return [float(score) for score in model.predict(pairs)]


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[Document]:
    """Re-order candidate documents by relevance and drop weak ones.

    Meant to be run on PARENT chunks (what the LLM will actually read), after
    ``record_parent``. Retrieval casts a wide net; this step keeps only the
    parents that really answer the question.

    Args:
        query: The user's original question (not the rewritten variants).
        documents: Candidate parent documents.
        top_k: How many to keep. Defaults to ``config.RERANK_TOP_K``.
        min_score: Documents scoring at or below this are dropped. Defaults to
            ``config.MIN_RERANK_SCORE``. The scale depends on the reranker.

    Returns:
        Up to ``top_k`` copies of the documents, best first, each with a
        ``rerank_score`` in its metadata. The input documents are not
        modified, because they are shared with the parent lookup. The list
        can be empty, which means nothing relevant was found.
    """
    if not documents:
        return []

    top_k = config.RERANK_TOP_K if top_k is None else top_k
    min_score = config.MIN_RERANK_SCORE if min_score is None else min_score

    if config.RERANKER == "cross-encoder":
        try:
            scores = _cross_encoder_scores(query, documents)
        except Exception:
            logger.exception(
                "Cross-encoder reranking failed; falling back to the simple reranker."
            )
            scores = _simple_scores(query, documents)
            min_score = SIMPLE_FALLBACK_MIN_SCORE
    else:
        scores = _simple_scores(query, documents)

    # sorted() is stable, so equal scores keep their retrieval order.
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)


    kept: List[Document] = []
    for doc, score in ranked:
        if score <= min_score:
            continue
        kept.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(score, 4)},
            )
        )
        if len(kept) == top_k:
            break

    logger.info(
        "Reranked %d candidates -> kept %d (top_k=%d, min_score=%s)",
        len(documents), len(kept), top_k, min_score,
    )
    return kept