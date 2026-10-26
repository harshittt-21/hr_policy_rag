import logging
import re
from typing import Dict, List, Set

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langsmith import traceable

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

# Models sometimes add bullets or numbering despite being told not to.
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def _clean_variant(line: str) -> str:
    """Strip list markers and wrapping quotes from one line of LLM output."""
    line = _LIST_MARKER.sub("", line).strip()
    return line.strip("\"'` ").strip()


def _run_llm_chain(question: str, history: str) -> List[str]:
    """Ask the chat model to rewrite a question.

    NUM_QUERY_VARIANTS is the TOTAL number of queries searched, including the
    original question, so the model is asked for one fewer rewrite. With a
    value of 1 the LLM call is skipped entirely.

    Returns the original question first, followed by unique rewrites.
    """
    question = question.strip()
    extra = config.NUM_QUERY_VARIANTS - 1
    if extra <= 0:
        return [question]

    chain = MULTI_QUERY_PROMPT | get_chat_model() | StrOutputParser()
    try:
        raw_output: str = chain.invoke(
            {
                "question": question,
                "history": history.strip() or "(none)",
                "num_variants": extra,
            }
        )
    except Exception:
        logger.exception("LLM failed while generating query variants")
        return [question]

    variants = [question]
    seen = {question.lower()}
    for line in raw_output.splitlines():
        cleaned = _clean_variant(line)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        variants.append(cleaned)

    return variants[: config.NUM_QUERY_VARIANTS]


def generate_query_variants(question: str, history: str = "") -> List[str]:
    """Return a list of query variants.

    The first element is always the original question; subsequent items are
    LLM-generated rewrites. If the LLM call fails we fall back to a
    single-item list containing only the original question.
    """
    return _run_llm_chain(question, history)


# -------------------------------------------------------------------
# Keyword search (BM25)
# -------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lowercase and split on word characters.

    BM25Retriever's default just splits on whitespace, so "Password",
    "password" and "password?" would all count as different words.
    """
    return _TOKEN_RE.findall(text.lower())


def build_keyword_retriever(children: List[Document]) -> BM25Retriever:
    """Create a BM25 keyword retriever over the child chunks.

    Building the index takes a moment for large corpora, so build it once and
    reuse it (for example with a cache in the UI) rather than per question.
    Requires the ``rank_bm25`` package.
    """
    if not children:
        raise ValueError("Cannot build a keyword index without child chunks.")

    return BM25Retriever.from_documents(
        children,
        preprocess_func=_tokenize,
        k=config.BM25_TOP_K,
    )


def search_keywords(retriever: BM25Retriever, query: str) -> List[Document]:
    """Keyword (BM25) search over the child chunks.

    Only chunks that share at least one word with the query are returned.
    BM25Retriever on its own always returns k chunks, even when every score
    is zero, which would feed arbitrary chunks into the fusion step.
    """
    tokens = retriever.preprocess_func(query)
    if not tokens:
        return []

    scores = retriever.vectorizer.get_scores(tokens)
    top_indexes = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[: config.BM25_TOP_K]

    return [retriever.docs[i] for i in top_indexes if scores[i] > 0]


# -------------------------------------------------------------------
# Semantic search, fusion and hybrid retrieval
# -------------------------------------------------------------------

def search_vector_store(store: VectorStore, query: str, k: int) -> List[Document]:
    """Semantic (embedding) search against the configured vector store.

    Returns up to ``k`` child documents ordered by similarity. Errors are not
    swallowed here; ``hybrid_retrieve`` decides how to handle them.
    """
    return store.similarity_search(query, k=k)


def fuse_results(ranked_lists: List[List[Document]], top_k: int) -> List[Document]:
    """Merge several result lists using Reciprocal Rank Fusion (RRF).

    Each appearance of a document scores ``1 / (60 + rank)`` (rank is
    zero-based). Documents are de-duplicated by their ``chunk_id`` metadata;
    if that key is missing, the page content is used instead.
    """
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.metadata.get("chunk_id") or doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (60.0 + rank)
            doc_lookup.setdefault(key, doc)

    ordered_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_lookup[k] for k in ordered_keys[:top_k]]


# process_inputs keeps the trace small: without it, LangSmith would try to
# serialise the whole vector store and BM25 index as inputs on every call.
@traceable(
    name="hybrid_retrieve",
    process_inputs=lambda inputs: {"queries": inputs.get("queries")},
)
def hybrid_retrieve(
    queries: List[str],
    store: VectorStore,
    keyword_retriever: BM25Retriever,
) -> List[Document]:
    """Run every query through semantic and keyword search and fuse results.

    The final list size respects ``config.HYBRID_TOP_K``. If semantic search
    fails for every query (bad API key, network down) the pipeline carries on
    with keyword results only and logs a warning. If every single search
    fails, it raises instead of pretending the documents contain no answer.
    """
    ranked: List[List[Document]] = []
    vector_failures = 0
    keyword_failures = 0

    for query in queries:
        try:
            ranked.append(search_vector_store(store, query, config.VECTOR_TOP_K))
        except Exception:
            vector_failures += 1
            logger.exception("Vector search failed for query %r", query)

        try:
            ranked.append(search_keywords(keyword_retriever, query))
        except Exception:
            keyword_failures += 1
            logger.exception("Keyword search failed for query %r", query)

    if queries and vector_failures + keyword_failures == 2 * len(queries):
        raise RuntimeError(
            "All retrieval searches failed. Check the logs for the underlying errors."
        )

    if queries and vector_failures == len(queries):
        logger.warning(
            "Semantic search failed for every query; results are keyword-only."
        )

    return fuse_results(ranked, config.HYBRID_TOP_K)


# -------------------------------------------------------------------
# Child -> parent
# -------------------------------------------------------------------

def record_parent(
    children: List[Document],
    parent_lookup: Dict[str, Document],
) -> List[Document]:
    """Replace child chunks with their parent documents, deduplicating.

    ``parent_lookup`` maps ``parent_id`` strings to the full parent Document.
    The result keeps the order in which each parent first appears. If a
    child's parent cannot be found (for example a stale index), the child
    itself is used so that relevant text is never silently dropped.
    """
    seen: Set[str] = set()
    results: List[Document] = []

    for child in children:
        parent_id = child.metadata.get("parent_id")
        parent_doc = parent_lookup.get(parent_id) if parent_id else None

        if parent_doc is None:
            logger.warning(
                "No parent found for chunk %s; using the chunk itself.",
                child.metadata.get("chunk_id", "?"),
            )
            parent_doc = child
            key = child.metadata.get("chunk_id") or child.page_content
        else:
            key = parent_id

        if key in seen:
            continue
        seen.add(key)
        results.append(parent_doc)

    return results