
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


def make_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """
    Build a text splitter that tries to cut text at meaningful boundaries.

    The splitter prefers paragraph breaks, then line breaks, sentence
    boundaries, and finally spaces or individual characters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        keep_separator="end",
        length_function=len,
    )


def _is_title_case(text: str) -> bool:
    """True if every word is capitalised, apart from small words like 'of'."""
    words = text.split()

    if not words or not words[0][0].isupper():
        return False

    return all(
        word[0].isupper()
        or word.lower() in _SMALL_WORDS
        or not word[0].isalpha()
        for word in words
    )


def _heading_of(line: str) -> str:
    """
    Return a heading if a line looks like one; otherwise return an empty string.

    Recognises uppercase headings, Title Case headings, and numbered headings
    such as '4. Password Policy' or '4.2 Account Security'.
    """
    line = line.strip()

    # Headings should be reasonably short and should not look like sentences.
    if not (3 < len(line) <= 70):
        return ""

    if line.endswith((".", ",", ";", ":")):
        return ""

    if len(line.split()) > 8:
        return ""

    # Remove a leading section number before checking the heading text.
    number_prefix = r"^\d+(?:\.\d+)*[.)]?\s+"
    heading = re.sub(number_prefix, "", line).strip()

    if not heading:
        return ""

    if heading.isupper() or _is_title_case(heading):
        return heading

    return ""


def find_headings(text: str) -> List[Tuple[int, str]]:
    """
    Find headings and their character offsets in a text block.

    Returns:
        A list of (offset, heading) pairs, ordered by position.
    """
    headings: List[Tuple[int, str]] = []
    offset = 0

    for raw_line in text.split("\n"):
        heading = _heading_of(raw_line)

        if heading:
            headings.append((offset, heading))

        # Include the newline character in the offset calculation.
        offset += len(raw_line) + 1

    return headings


def _is_document_title(heading: str, filename: str) -> bool:
    """
    Avoid treating a document's filename or year-bearing title as a section.

    The year check is intentionally conservative in implementation scope:
    headings containing a four-digit year are excluded from section labels.
    """
    filename_stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    filename_stem = re.sub(r"\.[^.]+$", "", filename_stem)

    normalise = lambda value: re.sub(
        r"[^a-z0-9]+", " ", value.lower()
    ).strip()

    if filename_stem and normalise(heading) == normalise(filename_stem):
        return True

    if re.search(r"\b(?:19|20)\d{2}\b", heading):
        return True

    return False


def make_point_id(chunk_id: str) -> str:
    """
    Turn a readable chunk ID into a stable UUID.

    Qdrant accepts UUIDs as point IDs. The same chunk ID always produces
    the same UUID, keeping re-indexing idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def split_documents(
    documents: List[Document],
) -> Tuple[List[Document], List[Document]]:
    """
    Turn loaded documents into large parent chunks and small child chunks.

    Each child carries a parent_id linking it to its parent. Section labels
    are assigned per child using the last heading before that child's
    position in the parent text.

    Returns:
        A tuple of (parents, children).
    """
    parent_splitter = make_splitter(
        config.PARENT_CHUNK_SIZE,
        config.PARENT_CHUNK_OVERLAP,
    )
    child_splitter = make_splitter(
        config.CHILD_CHUNK_SIZE,
        config.CHILD_CHUNK_OVERLAP,
    )

    parents: List[Document] = []
    children: List[Document] = []

    # Number parents independently for each file to keep IDs stable.
    parents_per_file: Dict[str, int] = {}

    # The latest section heading seen in each file.
    current_section: Dict[str, str] = {}

    for document in documents:
        filename = document.metadata.get("filename", "unknown")

        for parent_text in parent_splitter.split_text(document.page_content):
            parent_number = parents_per_file.get(filename, 0)
            parents_per_file[filename] = parent_number + 1

            parent_id = f"{filename}::{parent_number}"

            inherited = current_section.get(filename, "")

            # Find headings and their positions within this parent.
            headings = [
                (offset, heading)
                for offset, heading in find_headings(parent_text)
                if not _is_document_title(heading, filename)
            ]

            # A parent inherits the previous section if it has no headings.
            # Otherwise, its metadata records the last heading in the parent.
            section = headings[-1][1] if headings else inherited

            parent_metadata = dict(document.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["section"] = section

            parents.append(
                Document(
                    page_content=parent_text,
                    metadata=parent_metadata,
                )
            )

            child_texts = child_splitter.split_text(parent_text)

            # Child chunks can overlap. Search from the previous match's
            # starting position + 1 rather than its ending position, so
            # overlapping chunks can still be located.
            search_from = 0

            for child_index, child_text in enumerate(child_texts):
                pos = parent_text.find(child_text, search_from)

                if pos == -1:
                    # Defensive fallback for text altered by a splitter.
                    pos = max(search_from, 0)
                else:
                    search_from = pos + 1

                # Start with the inherited section, then apply every heading
                # that occurs before this child's starting position.
                child_section = inherited

                for offset, heading in headings:
                    if offset > pos:
                        break

                    child_section = heading

                chunk_id = f"{parent_id}::{child_index}"

                child_metadata = dict(parent_metadata)
                child_metadata["chunk_id"] = chunk_id
                child_metadata["point_id"] = make_point_id(chunk_id)
                child_metadata["section"] = child_section

                children.append(
                    Document(
                        page_content=child_text,
                        metadata=child_metadata,
                    )
                )

            # Carry the last heading forward to the next parent.
            current_section[filename] = section

    return parents, children


def build_parent_lookup(
    parents: List[Document],
) -> Dict[str, Document]:
    """
    Build a dictionary mapping each parent_id to its parent Document.

    Dictionary lookup is O(1), making it efficient to retrieve parent
    context after searching child chunks.
    """
    return {
        parent.metadata["parent_id"]: parent
        for parent in parents
    }
