
"""
Retrieval + guardrail evaluation for the TechNova RAG project.

Usage:
    python run_eval.py --stage dense_baseline
    python run_eval.py --stage hybrid
    python run_eval.py --stage hybrid_rerank
    python run_eval.py --stage hybrid_rerank_pc
    python run_eval.py --stage full
    python run_eval.py --stage hybrid --guardrails
"""
import logging
import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache

from src.chunking import build_parent_lookup
from src.guardrails import check_input, filter_unsafe_documents
from src.indexing import load_doc_store, load_vector_store
from src.refinement import refine_context
from src.reranking import rerank_documents
from src.retrieval import (
    build_keyword_retriever,
    generate_query_variants,
    hybrid_retrieve,
    record_parent,
    search_vector_store,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

KS = (1, 3, 5)

STAGES = (
    "dense_baseline",
    "hybrid",
    "hybrid_rerank",
    "hybrid_rerank_pc",
    "full",
)


def norm(text: str) -> str:
    """Normalize text for evidence matching."""
    return re.sub(r"\s+", " ", text.replace("\u2019", "'")).strip().lower()


# ------------------------------------------------------------------
# Retrieval setup


@lru_cache(maxsize=1)
def _load_retrieval_components():
    """Load the existing index and retrievers once per evaluation run."""
    parents, children = load_doc_store()

    if not children:
        raise RuntimeError(
            "No indexed documents found. Build the vector database first."
        )

    store = load_vector_store()
    parent_lookup = build_parent_lookup(parents)
    keyword_retriever = build_keyword_retriever(children)

    return store, parent_lookup, keyword_retriever


# ------------------------------------------------------------------
# TODO 1: Retrieval


def retrieve(question: str, k: int, stage: str) -> list[str]:
    """Return the top-k retrieved texts, ranked best first."""

    if stage not in STAGES:
        raise ValueError(
            f"Unknown stage {stage!r}. Choose from {STAGES}."
        )

    store, parent_lookup, keyword_retriever = (
        _load_retrieval_components()
    )

    # Stage 1: Dense vector retrieval only.
    if stage == "dense_baseline":
        documents = search_vector_store(store, question, k)
        return [doc.page_content for doc in documents[:k]]

    # Query refinement is enabled only in the full pipeline.
    if stage == "full":
        input_result = check_input(question)
        if not input_result.allowed:
            return []

        clean_question = input_result.text
        queries = generate_query_variants(clean_question)
    else:
        clean_question = question
        queries = [question]

    # Stage 2 onward: hybrid dense + BM25 retrieval and fusion.
    hits = hybrid_retrieve(
        queries=queries,
        store=store,
        keyword_retriever=keyword_retriever,
    )

    if stage == "hybrid":
        return [doc.page_content for doc in hits[:k]]

    # Stage 3: Rerank the retrieved child chunks.
    if stage == "hybrid_rerank":
        reranked = rerank_documents(
            clean_question,
            hits,
            top_k=k,
        )
        return [doc.page_content for doc in reranked[:k]]

    # Stages 4 and 5: Map child hits to parent documents.
    candidates = record_parent(hits, parent_lookup)
    candidates = filter_unsafe_documents(candidates)

    rerank_query = clean_question
    if stage == "full" and len(queries) > 1:
        rerank_query = queries[1]

    reranked = rerank_documents(
        rerank_query,
        candidates,
        top_k=k,
    )

    if stage == "hybrid_rerank_pc":
        return [doc.page_content for doc in reranked[:k]]

    # Stage 5: Apply the production context-refinement step.
    used, _context = refine_context(reranked)
    return [doc.page_content for doc in used[:k]]


# ------------------------------------------------------------------
# TODO 2: Input guardrail evaluation


def guardrail_blocked(prompt: str) -> bool:
    """Return True when the real input guardrail blocks the prompt."""
    return not check_input(prompt).allowed


# ------------------------------------------------------------------
# Evaluation data


def load(path: str) -> list[dict]:
    """Load a JSONL evaluation dataset."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# ------------------------------------------------------------------
# Retrieval scoring


def eval_retrieval(stage: str, path: str = "eval_retrieval.jsonl"):
    if stage not in STAGES:
        raise ValueError(
            f"Unknown stage {stage!r}. Choose from {STAGES}."
        )

    rows = [row for row in load(path) if row["answerable"]]

    if not rows:
        raise ValueError(
            f"No answerable questions found in {path!r}."
        )

    hit = {k: 0 for k in KS}
    coverage = {k: 0.0 for k in KS}
    reciprocal_rank = 0.0

    by_type = defaultdict(
        lambda: {"n": 0, "hit5": 0, "cov5": 0.0}
    )

    kmax = max(KS)

    for row in rows:
        retrieved = retrieve(row["question"], kmax, stage)
        chunks = [norm(chunk) for chunk in retrieved]
        snippets = [
            norm(item["snippet"])
            for item in row["evidence"]
            if item.get("snippet", "").strip()
        ]

        if not snippets:
            raise ValueError(
                "An answerable evaluation row has no evidence snippets: "
                f"{row['question']!r}"
            )

        # Reciprocal rank of the first chunk matching any evidence.
        first = next(
            (
                index + 1
                for index, chunk in enumerate(chunks)
                if any(snippet in chunk for snippet in snippets)
            ),
            None,
        )
        reciprocal_rank += 1 / first if first else 0

        for k in KS:
            top = chunks[:k]
            found = [
                any(snippet in chunk for chunk in top)
                for snippet in snippets
            ]

            hit[k] += int(any(found))
            coverage[k] += sum(found) / len(found)

        top5 = chunks[:5]
        found5 = [
            any(snippet in chunk for chunk in top5)
            for snippet in snippets
        ]

        question_type = by_type[row["type"]]
        question_type["n"] += 1
        question_type["hit5"] += int(any(found5))
        question_type["cov5"] += sum(found5) / len(found5)

    n = len(rows)

    print(f"\n=== Retrieval | stage={stage} | {n} answerable questions ===")

    for k in KS:
        print(
            f"Hit Rate@{k}: {hit[k] / n:.1%}    "
            f"Coverage@{k}: {coverage[k] / n:.1%}"
        )

    print(f"MRR: {reciprocal_rank / n:.3f}")

    print("\nBy question type (Hit@5 / Coverage@5):")
    for question_type, stats in sorted(by_type.items()):
        count = stats["n"]
        print(
            f"  {question_type:14s} n={count:2d}  "
            f"{stats['hit5'] / count:.1%} / "
            f"{stats['cov5'] / count:.1%}"
        )


# ------------------------------------------------------------------
# Guardrail scoring


def eval_guardrails(path: str = "eval_guardrails.jsonl"):
    rows = load(path)

    tp = fp = fn = tn = 0

    for row in rows:
        blocked = guardrail_blocked(row["prompt"])

        if row["expected"] == "block":
            tp += int(blocked)
            fn += int(not blocked)
        else:
            fp += int(blocked)
            tn += int(not blocked)

    print("\n=== Guardrails ===")
    print(f"Blocked {tp} of {tp + fn} injection/off-topic prompts")
    print(
        f"False positives: {fp} of {fp + tn} legitimate prompts wrongly blocked"
    )


# ------------------------------------------------------------------
# Entry point


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TechNova RAG retrieval and input guardrails."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGES,
        help="Retrieval pipeline stage to evaluate.",
    )
    parser.add_argument(
        "--guardrails",
        action="store_true",
        help="Also run the input guardrail evaluation.",
    )
    args = parser.parse_args()

    eval_retrieval(args.stage)

    if args.guardrails:
        eval_guardrails()


if __name__ == "__main__":
    main()
