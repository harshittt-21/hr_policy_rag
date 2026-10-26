import logging
from typing import Dict, List, Set
from langsmith import traceable
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

import config
from llm_client import get_chat_model

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Prompt for generating query variants
# -------------------------------------------------------------------
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries for a document search engine.\n"
            "Given a user question, write {num_variants} alternative versions "
            "of it that use different wording or synonyms but keep the same "
            "meaning.\n"
            "If conversation history is given, make each version standalone by "
            "replacing pronouns like 'it' or 'that policy' with the real subject.\n"
            "Return one query per line. No numbering, no extra text.",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\nUser question: {question}",
        ),
    ]
)


def _run_llm_chain(question: str, history: str) -> List[str]:
    """Internal helper that asks the chat model to rewrite a question.

    Returns a list of raw lines (including the original question). The caller
    is responsible for de‑duplicating and trimming.
    """
    chat = get_chat_model()
    chain = (
        MULTI_QUERY_PROMPT
        | chat
        | StrOutputParser()
    )
    try:
        raw_output: str = chain.invoke({"question": question, "history": history, "num_variants": config.NUM_QUERY_VARIANTS})
    except Exception as exc:
        logger.exception("LLM failed while generating query variants")
        return [question]
    # Split into lines, strip empties, keep original question as first entry
    lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
    if not lines:
        return [question]
    # Ensure the original question is present at the front
    if lines[0].lower() != question.lower():
        lines.insert(0, question)
    else:
        # make sure we don't have duplicates
        lines = [question] + [ln for ln in lines if ln.lower() != question.lower()]
    # Limit to the configured number of variants (including the original)
    return lines[: config.NUM_QUERY_VARIANTS]


def generate_query_variants(question: str, history: str = "") -> List[str]:
    """Public wrapper that returns a list of query variants.

    The first element is always the original *question*; subsequent items are
    LLM‑generated rewrites. If the LLM call fails we fall back to a single‑item
    list containing only the original question.
    """
    return _run_llm_chain(question, history)


def build_keyword_retriever(children: List[Document]) -> BM25Retriever:
    """Create a BM25 keyword retriever over the child chunks.

    ``children`` are the small chunks stored in the JSON doc store. The returned
    retriever is ready to be called with ``search_keywords``.
    """
    retriever = BM25Retriever.from_documents(children)
    # ``k`` is applied later; we just store the default config for reference.
    retriever.k = config.BM25_TOP_K
    return retriever


def search_vector_store(store: VectorStore, query: str, k: int) -> List[Document]:
    """Semantic (embedding) search against the configured vector store.

    Returns up to ``k`` child documents ordered by similarity. Any exception is
    logged and results in an empty list so the pipeline can continue.
    """
    try:
        return store.similarity_search(query, k=k)
    except Exception as exc:  # pragma: no cover – defensive
        logger.exception("Vector store search failed for query %r", query)
        return []


def search_keywords(retriever: BM25Retriever, query: str) -> List[Document]:
    """Keyword (BM25) search over the child chunks.

    The BM25 retriever respects its ``k`` attribute; we honour the global
    ``BM25_TOP_K`` constant for consistency.
    """
    try:
        # ``retrieve`` returns a list of Documents already limited to ``k``
        return retriever.invoke(query)[: config.BM25_TOP_K]
    except Exception as exc:  # pragma: no cover – defensive
        logger.exception("BM25 search failed for query %r", query)
        return []


def fuse_results(ranked_lists: List[List[Document]], top_k: int) -> List[Document]:
    """Merge several result lists using Reciprocal Rank Fusion (RRF).

    The algorithm assigns a score of ``1 / (60 + rank)`` for each appearance of a
    document (rank is zero‑based). Documents are de‑duplicated by their
    ``chunk_id`` metadata; if that key is missing we fall back to the full page
    content hash.
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst):
            key = doc.metadata.get("chunk_id") or doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank)
            if key not in doc_lookup:
                doc_lookup[key] = doc
    ordered_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    fused = [doc_lookup[k] for k in ordered_keys[:top_k]]
    return fused


@traceable(name="hybrid_retrieve")
def hybrid_retrieve(
    queries: List[str],
    store: VectorStore,
    keyword_retriever: BM25Retriever,
) -> List[Document]:
    """Run every query through both semantic and keyword search and fuse results.

    The final list size respects ``config.HYBRID_TOP_K``.
    """
    all_ranked: List[List[Document]] = []
    for q in queries:
        all_ranked.append(search_vector_store(store, q, config.VECTOR_TOP_K))
        all_ranked.append(search_keywords(keyword_retriever, q))
    return fuse_results(all_ranked, config.HYBRID_TOP_K)


def record_parent(
    children: List[Document],
    parent_lookup: Dict[str, Document],
) -> List[Document]:
    """Replace child chunks with their parent documents, deduplicating.

    ``parent_lookup`` maps ``parent_id`` strings to the full parent Document.
    The returned list preserves the order of first appearance of each parent.
    """
    seen: Set[str] = set()
    parents: List[Document] = []
    for child in children:
        parent_id = child.metadata.get("parent_id")
        if not parent_id:
            continue
        if parent_id in seen:
            continue
        parent_doc = parent_lookup.get(parent_id)
        if parent_doc:
            parents.append(parent_doc)
            seen.add(parent_id)
    return parents
