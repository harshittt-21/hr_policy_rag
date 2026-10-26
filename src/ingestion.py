import hashlib
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_document(file_path: str | Path) -> List[Document]:
    """
    Load a single PDF, TXT or DOCX file into LangChain Documents.

    Different file formats need different readers, so this function picks the
    right LangChain loader based on the file extension. Keeping that choice in
    one place means the rest of the project never worries about file formats.

    Args:
        file_path: Path to the file on disk.

    Returns:
        A list of Documents. PDFs return one Document per page; TXT and DOCX
        files return a single Document for the whole file.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    extension = path.suffix.lower()  # "security_policy.pdf" -> ".pdf"

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        # utf-8-sig also handles the BOM that Windows Notepad adds.
        loader = TextLoader(str(path), encoding="utf-8-sig")

    documents = loader.load()

    # Page headers/footers repeat on most pages; drop them before cleaning.
    if extension == ".pdf":
        documents = strip_repeated_headers_footers(documents)

    cleaned_documents = []

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

        if doc.page_content.strip():
            cleaned_documents.append(doc)

    if not cleaned_documents:
        logger.warning("No readable text found in %s. Skipping it.", path.name)
        return []

    return add_metadata(cleaned_documents, path, compute_file_hash(path))


def load_documents_from_folder(folder_path: str | Path) -> List[Document]:
    """
    Load every supported document inside a folder.

    This is what the "Build Vector DB" button and the build script use to read
    the whole `data/documents` folder in one go.

    Args:
        folder_path: Folder to scan (not recursive, one level only).

    Returns:
        A combined list of Documents from all readable files. Files that fail
        to load are skipped with a warning instead of stopping the whole run.
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning("Document folder does not exist: %s", folder)
        return []

    all_documents: List[Document] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue
        # Skip hidden files and Word lock files such as "~$policy.docx".
        if file_path.name.startswith(("~$", ".")):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            all_documents.extend(load_document(file_path))
        except Exception as e:
            logger.warning("Could not load %s: %s", file_path.name, e)

    logger.info("Loaded %d document pages from %s", len(all_documents), folder)
    return all_documents


def _normalize_line(line: str) -> str:
    """Lowercase a line and mask digits so 'Page 4' and 'Page 5' match."""
    return re.sub(r"\d+", "#", line.strip().lower())


def strip_repeated_headers_footers(
    documents: List[Document],
    edge_lines: int = 2,
    min_ratio: float = 0.5,
) -> List[Document]:
    """
    Remove running headers and footers from a multi-page document.

    Only the first and last few non-empty lines of each page are considered.
    A line (with digits masked) that shows up on at least `min_ratio` of the
    pages is treated as a header or footer and removed. Page numbers are
    caught the same way.

    Args:
        documents: One Document per page, as returned by PyPDFLoader.
        edge_lines: How many lines at the top and bottom of a page to check.
        min_ratio: Fraction of pages a line must appear on to count as repeated.

    Returns:
        The same documents with repeated edge lines removed. Documents with
        fewer than 3 pages are returned unchanged, because there is not enough
        evidence to tell a header from real content.
    """
    if len(documents) < 3:
        return documents

    page_edges = []
    counts: Counter = Counter()

    for doc in documents:
        lines = doc.page_content.split("\n")
        non_empty = [i for i, line in enumerate(lines) if line.strip()]
        edge_indexes = set(non_empty[:edge_lines] + non_empty[-edge_lines:])
        page_edges.append(edge_indexes)

        # A set per page, so a line repeated within one page counts once.
        counts.update({_normalize_line(lines[i]) for i in edge_indexes})

    threshold = max(2, math.ceil(len(documents) * min_ratio))
    repeated = {line for line, n in counts.items() if n >= threshold and line}

    for doc, edge_indexes in zip(documents, page_edges):
        lines = doc.page_content.split("\n")
        kept = [
            line
            for i, line in enumerate(lines)
            if not (i in edge_indexes and _normalize_line(line) in repeated)
        ]
        doc.page_content = "\n".join(kept)

    return documents


def clean_text(text: str) -> str:
    """
    Remove the noise that PDF and DOCX extraction usually leaves behind.

    Raw extracted text is full of repeated spaces, runs of blank lines and
    words hyphenated across line breaks. Cleaning it first means the chunks we
    embed are closer to real sentences, which improves retrieval quality.
    Single line breaks are kept on purpose, because the splitter uses them as
    natural boundaries.

    Note that a real compound word split across lines ("well-\\nknown") is
    also joined ("wellknown"). That is an accepted trade-off.

    Args:
        text: Raw text straight out of a document loader.

    Returns:
        The same text, tidied up.
    """
    if not text:
        return ""

    # Ex: "secu-\nrity" -> "security"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def compute_file_hash(path: Path) -> str:
    """
    Return a SHA-256 hash of a file's bytes.

    Stored in each chunk's metadata so the indexer can tell whether a file
    with the same name has changed since it was last indexed.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_metadata(documents: List[Document], path: Path, file_hash: str = "") -> List[Document]:
    """
    Attach source information to each Document so we can cite it later.

    Retrieval returns a piece of text with no memory of where it came from.
    By storing the filename, file type and page number now, the final answer
    can say "technova_employee_handbook.pdf - Page 4".

    Only the keys the project uses are kept. Loaders add extras (producer,
    creation date, absolute path and so on) that would just bloat every
    chunk stored in the vector database.

    Args:
        documents: Documents produced by a loader.
        path: Path of the file they were loaded from.
        file_hash: Content hash of the file, used to detect edited files.

    Returns:
        The same documents, with clean metadata. `page_number` is only set for
        formats that have pages (PDF), so read it with `.get("page_number")`.
    """
    for document in documents:
        page = document.metadata.get("page")

        metadata = {
            "filename": path.name,
            "doc_type": path.suffix.lower().lstrip("."),
            "file_hash": file_hash,
        }
        if page is not None:
            metadata["page_number"] = int(page) + 1

        document.metadata = metadata

    return documents