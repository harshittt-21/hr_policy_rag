# test_retrieval.py
from src.ingestion import load_documents_from_folder
import logging
from pathlib import Path
# ---- Project imports -------------------------------------------------
import config
from src.ingestion import load_document   # helper that reads files from data/documents
from src.chunking import split_documents,build_parent_lookup
from src.indexing import build_index, index_exists, load_vector_store, load_doc_store
from src.retrieval import hybrid_retrieve, build_keyword_retriever, record_parent
# --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
def main() -> None:
    # 1️⃣  Ensure an index exists – build a fresh one if needed
    if not index_exists():
        # Load raw PDFs/markdown files from the data folder
        docs = load_documents_from_folder(Path(config.DOCUMENTS_DIR))
        # Build the whole index (parents + children + vector store)
        stats = build_index(docs)
        logging.info("Built fresh index: %s", stats)
    else:
        logging.info("Index already present – skipping build")
    # 2️⃣  Load the persisted stores
    parents, children = load_doc_store()
    vector_store = load_vector_store()
    parent_lookup = build_parent_lookup(parents)
    # 3️⃣  Run a very small query through the hybrid pipeline
    query = "What is the sick leave policy ?"
    # Build keyword retriever over child chunks
    keyword_retriever = build_keyword_retriever(children)
    # Perform hybrid retrieval (list of queries)
    results = hybrid_retrieve(
        queries=[query],
        store=vector_store,
        keyword_retriever=keyword_retriever,
    )
    # Map child results back to parent documents
    results = record_parent(results, parent_lookup)
    # 4️⃣  Show the retrieved parent documents
    print("\n=== Retrieved parent chunks (most relevant) ===")
    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Source file: {doc.metadata.get('filename')}")
        print(f"Section: {doc.metadata.get('section')}")
        print(f"Content preview (first 300 chars):")
        print(doc.page_content[:300].replace('\n', ' '))
if __name__ == "__main__":
    main()