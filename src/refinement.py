"""
Step 6 of the pipeline: decide what actually gets sent to the LLM.

Reranking gave us an ordered list. Context refinement answers the follow-up
question: "should all of these really go into the prompt?"

We do three simple things:

1. Drop documents the reranker scored as clearly irrelevant.
2. Stop once we hit a character budget, so the prompt stays small and cheap.
3. Format the survivors with their source labels, so the LLM can cite them.

If nothing survives, the result is empty on purpose. The caller should then
reply "I couldn't find this in the documents" without calling the LLM at all.

That is the whole technique. There is no need for a complicated compression
framework to teach the idea.
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

# Blocks in the final context are joined by a blank line.
BLOCK_SEPARATOR = "\n\n"
TRUNCATION_MARK = "..."


def describe_source(document: Document) -> str:
    """
    Build a short, human-readable label describing where a chunk came from.

    This label is shown under every answer, and is also written into the
    prompt so the model knows which document it is reading.

    Args:
        document: A chunk with metadata from the ingestion step.

    Returns:
        A string such as "technova_employee_handbook.pdf - Page 4, Section:
        Sick Leave". Page and section are each included when known; with
        neither, just the filename.
    """
    filename = document.metadata.get("filename", "unknown file")

    details = []
    page_number = document.metadata.get("page_number")
    if page_number:
        details.append(f"Page {page_number}")

    section = document.metadata.get("section")
    if section:
        details.append(f"Section: {section}")

    if not details:
        return filename

    return f"{filename} - {', '.join(details)}"


def format_block(number: int, document: Document) -> str:
    """Format one numbered, source-labelled block of context."""
    return f"[{number}] Source: {describe_source(document)}\n{document.page_content}"


def drop_irrelevant(documents: List[Document]) -> List[Document]:
    """
    Remove chunks the reranker scored at or below the relevance threshold.

    Retrieval always returns its top results, even when the documents contain
    nothing about the question. Feeding those to the LLM invites made-up
    answers, so we throw away anything that scored too low.

    `rerank_documents` already applies this threshold, so this is normally a
    no-op. It stays as a safety net for documents that reach this step by
    another route. Documents with no `rerank_score` at all (reranking was
    skipped) are kept rather than silently discarded.

    Args:
        documents: Reranked documents, usually carrying a `rerank_score`.

    Returns:
        Only the documents worth reading. This can be an empty list, and that
        is a meaningful answer: the documents do not cover the question.
        Keeping a low-scoring "best of a bad bunch" would send irrelevant
        text to the LLM and show a misleading citation under the answer.
    """
    relevant = []
    for document in documents:
        score = document.metadata.get("rerank_score")
        if score is None or score > config.MIN_RERANK_SCORE:
            relevant.append(document)

    dropped = len(documents) - len(relevant)
    if dropped:
        logger.info("Dropped %d chunk(s) below the relevance threshold.", dropped)

    return relevant


def apply_context_budget(documents: List[Document]) -> List[Document]:
    """
    Keep adding documents until the character budget runs out.

    Long prompts cost more, are slower, and actually make answers worse: a
    model asked to read ten pages often misses the one sentence that matters.
    Because the list is already sorted best-first, stopping at the first
    document that does not fit removes the least useful text and never lets a
    lower-ranked document jump ahead of a higher-ranked one.

    The budget counts the final formatted text (source labels and blank lines
    included), so the context string never exceeds config.MAX_CONTEXT_CHARS.

    Args:
        documents: Relevant documents, best first.

    Returns:
        The documents that fit inside config.MAX_CONTEXT_CHARS.

    Example:
        With a 8000-character budget and 2500-character parents, the 4th
        document is dropped (3 x 2500 plus labels fits, 4 x 2500 does not).
    """
    selected: List[Document] = []
    used_characters = 0

    for document in documents:
        block = format_block(len(selected) + 1, document)
        separator = len(BLOCK_SEPARATOR) if selected else 0
        cost = separator + len(block)

        if used_characters + cost > config.MAX_CONTEXT_CHARS:
            break

        selected.append(document)
        used_characters += cost

    # Never return nothing when there is something relevant: if even the first
    # document is larger than the whole budget, send a trimmed copy of it.
    if not selected and documents:
        first = documents[0]
        header_length = len(format_block(1, first)) - len(first.page_content)
        room = max(config.MAX_CONTEXT_CHARS - header_length - len(TRUNCATION_MARK), 0)

        # Cut at a word boundary, not in the middle of a word.
        text = first.page_content[:room].rsplit(" ", 1)[0].rstrip()
        selected.append(
            Document(
                page_content=text + TRUNCATION_MARK,
                metadata=dict(first.metadata),  # copy: do not share with the parent lookup
            )
        )

    return selected


def format_context(documents: List[Document]) -> str:
    """
    Turn the selected documents into the single text block the LLM reads.

    Each chunk is numbered and labelled with its source. That numbering is
    what lets the prompt say "cite the source numbers you used".

    Args:
        documents: The final selected documents.

    Returns:
        A formatted string, for example:

            [1] Source: handbook.pdf - Page 2, Section: Annual Leave
            Employees receive 24 days of annual leave...

            [2] Source: support_policy.txt - Section: Refunds
            Refunds are issued within 30 days...
    """
    return BLOCK_SEPARATOR.join(
        format_block(number, document)
        for number, document in enumerate(documents, start=1)
    )


def refine_context(documents: List[Document]) -> Tuple[List[Document], str]:
    """
    Run the full refinement step: filter, budget, then format.

    This is the single function the pipeline calls. Keeping the three smaller
    steps separate makes each one easy to read, test and explain.

    Args:
        documents: Reranked documents.

    Returns:
        A tuple `(selected_documents, context_text)`. `context_text` is ready
        to drop straight into the prompt, and the source numbers in it match
        the order of `selected_documents`, so use that list for citations.
        Both are empty when nothing relevant was found.
    """
    relevant = drop_irrelevant(documents)
    selected = apply_context_budget(relevant)
    return selected, format_context(selected)