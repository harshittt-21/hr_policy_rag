"""
Find where words get glued together ("toTechNova", "kind.Concerns").

Prints the raw text from the PDF loader next to the same text after
clean_text(), so you can see which stage removes the space or line break.

Run from the project root:

    python check_text.py
    python check_text.py data/documents/technova_employee_handbook.pdf
"""
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

import config
from src.ingestion import clean_text

NEEDLES = ["Welcome to", "any kind."]  # the two spots seen in the test output
CONTEXT = 30


def snippet(text: str, needle: str) -> str:
    index = text.find(needle)
    if index == -1:
        return "(not found)"
    return repr(text[index : index + len(needle) + CONTEXT])


def main() -> int:
    default = Path(config.DOCUMENTS_DIR) / "technova_employee_handbook.pdf"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    if not path.exists():
        print(f"File not found: {path}")
        return 1

    pages = PyPDFLoader(str(path)).load()
    print(f"{path.name}: {len(pages)} pages\n")

    for needle in NEEDLES:
        for page_number, page in enumerate(pages, start=1):
            raw = page.page_content
            if needle not in raw:
                continue
            print(f"--- {needle!r} on page {page_number} ---")
            print("raw    :", snippet(raw, needle))
            print("cleaned:", snippet(clean_text(raw), needle))
            print()

    print(
        "How to read this:\n"
        "  raw already glued ('toTechNova')      -> the PDF extraction does it.\n"
        "      Try PyPDFLoader(path, extraction_mode='layout') or PyMuPDFLoader.\n"
        "  raw has '\\n' but cleaned does not     -> clean_text removes line breaks\n"
        "      without putting a space in their place.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())