import re
import uuid
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# Words that may stay lowercase inside a Title Case heading ("Use of AI").
_SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in",
    "of", "on", "or", "the", "to", "with",
}


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build a text splitter that tries to cut text at meaningful boundaries.

    "Meaning-aware" splitting here means the splitter prefers to break at a
    paragraph break, then at a line break, then at the end of a sentence, and
    only chops mid-word as a last resort. That keeps ideas intact inside a
    chunk instead of slicing a sentence in half.

    Args:
        chunk_size: Target chunk length in characters.
        chunk_overlap: How many characters neighbouring chunks share, so a
            sentence sitting on a boundary is not lost.

    Returns:
        A configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        # Keep the punctuation with the sentence it ends, instead of
        # letting the next chunk start with ". ".
        keep_separator="end",
        length_function=len,
    )


def _is_title_case(text: str) -> bool:
    """True if every word is capitalised, apart from small words like 'of'."""
    words = text.split()
    if not words or not words[0][0].isupper():
        return False
    return all(
        w[0].isupper() or w.lower() in _SMALL_WORDS or not w[0].isalpha()
        for w in words
    )


def guess_section(text: str) -> str:
    """
    Guess the section heading a piece of text belongs to.

    Policy documents are written as "Password Policy", "Refund Policy" and so
    on. Capturing that heading lets a citation say "Section: Password Policy",
    which is far more useful to a reader than a filename alone.

    Args:
        text: A chunk of document text.

    Returns:
        The heading we found, or an empty string if the text has no obvious
        heading. We deliberately keep this simple: a wrong guess only affects
        the citation label, never the answer itself.
    """
    number_prefix = r"^\d+(\.\d+)*[.)]?\s+"

    for raw_line in text.split("\n")[:15]:
        line = raw_line.strip()

        # A heading is short, has a handful of words and does not end like a
        # sentence. "4. PASSWORD POLICY" passes; "We reset passwords." fails.
        if not (3 < len(line) <= 70) or line.endswith((".", ",", ";", ":")):
            continue
        if len(line.split()) > 8:
            continue

        # Strip a leading number so "4. Password Policy" becomes
        # "Password Policy", then judge the heading text itself. Judging the
        # text (not just the number) stops list items such as
        # "3. Use strong passwords" from being mistaken for headings.
        heading = re.sub(number_prefix, "", line).strip()
        if not heading:
            continue

        if heading.isupper() or _is_title_case(heading):
            return heading

    return ""


def make_point_id(chunk_id: str) -> str:
    """
    Turn a readable chunk id into a stable UUID.

    Qdrant only accepts unsigned integers or UUIDs as point ids, so
    "handbook.pdf::3::2" cannot be used directly. The same chunk_id always
    produces the same UUID, which keeps re-indexing idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def split_documents(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    """
    Turn loaded documents into large parent chunks and small child chunks.

    This is the core of parent-child retrieval. Each child carries a
    `parent_id` in its metadata, which is the link we follow later to recover
    the bigger context.

    Args:
        documents: Cleaned documents from `src.ingestion`.

    Returns:
        A tuple `(parents, children)`:
            parents  - large chunks, used as the context given to the LLM.
            children - small chunks, the ones we actually embed and search.

    Example:
        A 5-page handbook may produce 12 parents and 48 children. Searching
        finds child #37; the LLM reads parent "handbook.pdf::8" instead.
    """
    parent_splitter = make_splitter(config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
    child_splitter = make_splitter(config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP)

    parents: List[Document] = []
    children: List[Document] = []

    # We count parents separately for each file. Numbering per file (rather
    # than across the whole batch) means a chunk always gets the same id no
    # matter which other documents were processed alongside it - which is what
    # lets `update_index` recognise and skip a file it has already indexed.
    parents_per_file: Dict[str, int] = {}

    # The last heading seen in each file. A parent that starts mid-section
    # has no heading of its own, so it inherits the previous one.
    current_section: Dict[str, str] = {}

    for document in documents:
        filename = document.metadata.get("filename", "unknown")

        for parent_text in parent_splitter.split_text(document.page_content):
            parent_number = parents_per_file.get(filename, 0)
            parents_per_file[filename] = parent_number + 1
            parent_id = f"{filename}::{parent_number}"

            section = guess_section(parent_text) or current_section.get(filename, "")
            current_section[filename] = section

            parent_metadata = dict(document.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["section"] = section

            parents.append(Document(page_content=parent_text, metadata=parent_metadata))

            for child_index, child_text in enumerate(child_splitter.split_text(parent_text)):
                chunk_id = f"{parent_id}::{child_index}"
                child_metadata = dict(parent_metadata)
                child_metadata["chunk_id"] = chunk_id
                child_metadata["point_id"] = make_point_id(chunk_id)
                children.append(Document(page_content=child_text, metadata=child_metadata))

    return parents, children


def build_parent_lookup(parents: List[Document]) -> Dict[str, Document]:
    """
    Build a dictionary that maps a parent_id to its parent Document.

    Retrieval gives us child chunks; we need an instant way to fetch the
    matching parent. A dictionary lookup is O(1), which is as simple and fast
    as this problem needs to be.

    Args:
        parents: The list of parent chunks.

    Returns:
        A dictionary like {"handbook.pdf::3": Document(...), ...}.
    """
    return {parent.metadata["parent_id"]: parent for parent in parents}