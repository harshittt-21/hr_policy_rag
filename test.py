"""
End-to-end smoke test: retrieval -> parents -> reranking -> context.

Run from the project root:

    python test_retrieval.py                      # build index if missing, then query
    python test_retrieval.py --rebuild            # force a fresh index build
    python test_retrieval.py --query "How many days of annual leave do I get?" --expect "annual leave"

Tip: set NUM_QUERY_VARIANTS=1 in .env for repeatable runs. The LLM rewrites
differ slightly from run to run, which changes what retrieval returns.

Exits with code 1 if a required check fails.
"""
import argparse
import logging
import sys
from pathlib import Path

import config
from src.chunking import build_parent_lookup
from src.indexing import build_index, index_exists, load_doc_store, load_vector_store
from src.ingestion import load_documents_from_folder
from src.refinement import refine_context
from src.reranking import rerank_documents
from src.retrieval import (
    build_keyword_retriever,
    generate_query_variants,
    hybrid_retrieve,
    record_parent,
    search_keywords,
    search_vector_store,
)

DEFAULT_QUERY = "What is the sick leave policy?"
DEFAULT_EXPECT = "sick leave"


def preview(text: str, limit: int = 200) -> str:
    """One-line preview of a chunk."""
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def describe_chunk(doc) -> str:
    meta = doc.metadata
    page = meta.get("page_number")
    where = f"{meta.get('filename', '?')}" + (f" p.{page}" if page else "")
    return f"[{meta.get('chunk_id', '?')}] {where}"


def print_chunks(title: str, docs, limit: int = 5) -> None:
    print(f"\n{title} ({len(docs)} found)")
    for doc in docs[:limit]:
        print(f"  - {describe_chunk(doc)}")
        print(f"      {preview(doc.page_content, 150)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the RAG pipeline.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question to search for.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index first.")
    parser.add_argument(
        "--expect",
        default=None,
        help="Text the top reranked parent must contain (case-insensitive).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Hide the per-request HTTP lines from Qdrant / Groq / Jina.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    expect = args.expect
    if expect is None and args.query == DEFAULT_QUERY:
        expect = DEFAULT_EXPECT

    # 1. Fail fast on bad configuration -----------------------------------
    config.check_config()
    print("Settings:", config.describe_settings())

    # 2. Build the index if needed ----------------------------------------
    if args.rebuild or not index_exists():
        docs = load_documents_from_folder(Path(config.DOCUMENTS_DIR))
        if not docs:
            print(f"\nNo readable documents found in {config.DOCUMENTS_DIR}.")
            return 1
        stats = build_index(docs)
        print(f"\nBuilt fresh index: {stats}")
    else:
        print("\nIndex already present - skipping build (use --rebuild to force).")

    # 3. Load persisted stores --------------------------------------------
    parents, children = load_doc_store()
    vector_store = load_vector_store()
    parent_lookup = build_parent_lookup(parents)
    keyword_retriever = build_keyword_retriever(children)
    print(f"Loaded {len(parents)} parents and {len(children)} children.")

    # 4. Query variants ---------------------------------------------------
    query = args.query
    queries = generate_query_variants(query)
    print(f"\nQuery: {query!r}")
    print("Variants searched:")
    for q in queries:
        print(f"  - {q}")

    # 5. Each search method on its own ------------------------------------
    vector_hits = search_vector_store(vector_store, query, config.VECTOR_TOP_K)
    keyword_hits = search_keywords(keyword_retriever, query)
    print_chunks("Vector search (original query)", vector_hits, limit=3)
    print_chunks("Keyword search (original query)", keyword_hits, limit=3)

    # 6. Full hybrid pipeline ---------------------------------------------
    hybrid_hits = hybrid_retrieve(
        queries=queries,
        store=vector_store,
        keyword_retriever=keyword_retriever,
    )
    print_chunks("Hybrid result (fused child chunks)", hybrid_hits, limit=config.HYBRID_TOP_K)

    # 7. Children -> parents (BEFORE reranking) ---------------------------
    candidates = record_parent(hybrid_hits, parent_lookup)
    before_chars = sum(len(d.page_content) for d in candidates)
    print(
        f"\nCandidate parents before reranking: {len(candidates)} "
        f"({before_chars} chars, limit {config.MAX_CONTEXT_CHARS})"
    )

    # 8. Rerank: first show EVERY candidate's score, then the real result --
    scored_all = rerank_documents(
        query, candidates, top_k=len(candidates), min_score=float("-inf")
    )
    print(f"\n=== Rerank scores for all candidates (reranker: {config.RERANKER}) ===")
    print(f"Threshold MIN_RERANK_SCORE = {config.MIN_RERANK_SCORE}, keep top {config.RERANK_TOP_K}")
    for doc in scored_all:
        kept_marker = "keep" if doc.metadata["rerank_score"] > config.MIN_RERANK_SCORE else "drop"
        print(
            f"  {doc.metadata['rerank_score']:>9.4f}  {kept_marker}  "
            f"{doc.metadata.get('filename')}  |  {doc.metadata.get('section') or '-'}  |  "
            f"{preview(doc.page_content, 60)}"
        )

    reranked = rerank_documents(query, candidates)

    # 9. Build the final context ------------------------------------------
    used, context = refine_context(reranked)
    print(f"\n=== Final context: {len(used)} parents, {len(context)} chars ===")
    for i, doc in enumerate(used, start=1):
        page = doc.metadata.get("page_number")
        print(
            f"  [{i}] {doc.metadata.get('filename')}"
            + (f" p.{page}" if page else "")
            + f"  | Section: {doc.metadata.get('section') or '-'}"
            + f"  | score {doc.metadata.get('rerank_score')}"
        )

    # 10. Checks ----------------------------------------------------------
    # (name, passed, required)
    top_text = used[0].page_content.lower() if used else ""
    checks = [
        ("index contains child chunks", len(children) > 0, True),
        ("vector search returned results", len(vector_hits) > 0, True),
        ("keyword search returned results", len(keyword_hits) > 0, False),
        ("hybrid retrieval returned results", len(hybrid_hits) > 0, True),
        ("hybrid results within HYBRID_TOP_K", len(hybrid_hits) <= config.HYBRID_TOP_K, True),
        (
            "every hit has a parent in the doc store",
            all(h.metadata.get("parent_id") in parent_lookup for h in hybrid_hits),
            True,
        ),
        (
            "parents are unique",
            len({d.metadata.get("parent_id") for d in candidates}) == len(candidates),
            True,
        ),
        ("reranking kept at least one parent", len(reranked) > 0, True),
        ("reranking kept at most RERANK_TOP_K", len(reranked) <= config.RERANK_TOP_K, True),
        (
            "reranked parents are sorted best-first",
            [d.metadata["rerank_score"] for d in reranked]
            == sorted((d.metadata["rerank_score"] for d in reranked), reverse=True),
            True,
        ),
        (
            "final context fits MAX_CONTEXT_CHARS",
            len(context) <= config.MAX_CONTEXT_CHARS,
            True,
        ),
        (
            "reranking reduced the context size",
            len(context) < before_chars,
            False,
        ),
    ]
    if expect:
        checks.append(
            (f"top reranked parent mentions {expect!r}", expect.lower() in top_text, True)
        )

    print("\n=== Checks ===")
    failed_required = False
    for name, passed, required in checks:
        status = "PASS" if passed else ("FAIL" if required else "WARN")
        print(f"[{status}] {name}")
        if not passed and required:
            failed_required = True

    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())