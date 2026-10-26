import json
import logging
import shutil
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

import config
from llm_client import get_embedding_model
from src.chunking import make_point_id, split_documents

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Document store (parents + children; used by parent lookup and BM25)
# -------------------------------------------------------------------

def _documents_to_dicts(documents: List[Document]) -> List[dict]:
    """Convert LangChain Documents into JSON-serializable dictionaries."""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]


def _dicts_to_documents(items: List[dict]) -> List[Document]:
    """Convert dictionaries back into LangChain Documents."""
    return [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in items
    ]


def save_doc_store(parents: List[Document], children: List[Document]) -> None:
    """Persist parent and child chunks using an atomic file replacement."""
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    config.DOC_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "parents": _documents_to_dicts(parents),
        "children": _documents_to_dicts(children),
    }
    tmp_path = config.DOC_STORE_PATH.with_suffix(
        config.DOC_STORE_PATH.suffix + ".tmp"
    )
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(config.DOC_STORE_PATH)
    finally:
        # If writing/replacing failed, do not leave a stale temporary file.
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info(
        "Saved %d parents and %d children to %s",
        len(parents),
        len(children),
        config.DOC_STORE_PATH,
    )


def load_doc_store() -> Tuple[List[Document], List[Document]]:
    """Load parent and child chunks from the JSON document store."""
    if not config.DOC_STORE_PATH.exists():
        return [], []

    try:
        payload = json.loads(config.DOC_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read document store at {config.DOC_STORE_PATH}. "
            "Restore a valid backup or rebuild the index."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Invalid document store format at {config.DOC_STORE_PATH}: "
            "expected a JSON object."
        )

    parents = _dicts_to_documents(payload.get("parents", []))
    children = _dicts_to_documents(payload.get("children", []))
    return parents, children


# -------------------------------------------------------------------
# Backend helpers
# -------------------------------------------------------------------

def _faiss_class():
    """Import FAISS only when the FAISS backend is selected."""
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError as exc:
        raise ImportError(
            "FAISS requires faiss-cpu and langchain-community. "
            "Install them in your active environment."
        ) from exc
    return FAISS


def _qdrant_class():
    """Import Qdrant only when the Qdrant backend is selected."""
    try:
        from langchain_qdrant import QdrantVectorStore
    except ImportError as exc:
        raise ImportError(
            "Qdrant requires langchain-qdrant and qdrant-client. "
            "Install them in your active environment."
        ) from exc
    return QdrantVectorStore


def _qdrant_client():
    """Create a Qdrant client for the configured cloud instance."""
    from qdrant_client import QdrantClient

    config.check_vector_store_config()
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def _validate_vector_db() -> str:
    """Validate and return the configured vector-store backend name."""
    backend = config.VECTOR_DB.strip().lower()
    if backend not in {"qdrant", "faiss"}:
        raise ValueError(
            f"Unknown VECTOR_DB '{config.VECTOR_DB}'. Choose 'qdrant' or 'faiss'."
        )
    return backend


def _point_ids(documents: List[Document]) -> List[str]:
    """Return stable UUIDs for child chunks."""
    ids = []
    for doc in documents:
        point_id = doc.metadata.get("point_id")
        chunk_id = doc.metadata.get("chunk_id")
        if point_id:
            ids.append(str(point_id))
        elif chunk_id:
            ids.append(make_point_id(chunk_id))
        else:
            raise ValueError(
                "A child chunk is missing both 'point_id' and 'chunk_id' metadata."
            )
    return ids


def _filename(doc: Document) -> str:
    """Return the source filename recorded in chunk metadata."""
    return str(doc.metadata.get("filename", "unknown"))


def _by_filename(documents: List[Document]) -> Dict[str, str]:
    """Map each source filename to its content hash."""
    hashes: Dict[str, str] = {}
    for doc in documents:
        name = _filename(doc)
        file_hash = str(doc.metadata.get("file_hash", ""))
        # All chunks from one source should have the same hash.
        previous = hashes.get(name)
        if previous is not None and previous != file_hash:
            raise ValueError(
                f"Inconsistent file hashes found for {name!r} in the document store."
            )
        hashes[name] = file_hash
    return hashes


def _delete_ids(store: VectorStore, ids: List[str], backend: str) -> None:
    """Delete IDs that actually exist in the selected vector store."""
    if not ids:
        return

    if backend == "faiss":
        # FAISS deletion expects IDs in its docstore mapping.
        present = set(store.index_to_docstore_id.values())
        ids = [point_id for point_id in ids if point_id in present]

    if ids:
        store.delete(ids=ids)


# -------------------------------------------------------------------
# Vector store
# -------------------------------------------------------------------

def create_vector_store(children: List[Document]) -> VectorStore:
    """Create a vector index from child chunks."""
    if not children:
        raise ValueError("There are no chunks to index. Add documents before building.")

    backend = _validate_vector_db()
    embeddings = get_embedding_model()
    ids = _point_ids(children)

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate child point IDs detected. Check source identity and chunk IDs."
        )

    if backend == "faiss":
        config.FAISS_DIR.mkdir(parents=True, exist_ok=True)
        # Embedding completes before save_local is called. Note that writing the
        # multiple FAISS files is not a fully atomic directory-level operation.
        store = _faiss_class().from_documents(children, embeddings, ids=ids)
        store.save_local(str(config.FAISS_DIR))
        return store

    config.check_vector_store_config()
    return _qdrant_class().from_documents(
        documents=children,
        embedding=embeddings,
        ids=ids,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        force_recreate=True,
    )


def load_vector_store() -> VectorStore:
    """Load an existing index without rebuilding it."""
    backend = _validate_vector_db()
    embeddings = get_embedding_model()

    if backend == "faiss":
        index_file = config.FAISS_DIR / "index.faiss"
        pickle_file = config.FAISS_DIR / "index.pkl"
        if not index_file.exists() or not pickle_file.exists():
            raise FileNotFoundError("No complete FAISS index found. Build the index first.")

        # Only load FAISS files generated by this trusted application.
        return _faiss_class().load_local(
            str(config.FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    client = _qdrant_client()
    if not client.collection_exists(config.QDRANT_COLLECTION_NAME):
        raise FileNotFoundError(
            f"Qdrant collection '{config.QDRANT_COLLECTION_NAME}' does not exist. "
            "Build the index first."
        )
    return _qdrant_class()(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )


# -------------------------------------------------------------------
# Build / update
# -------------------------------------------------------------------

def build_index(documents: List[Document]) -> Dict[str, int]:
    """Build a fresh index from the supplied source documents.

    Warning: the Qdrant backend recreates its configured collection. If you
    need zero-downtime rebuilds, use a versioned collection + alias swap.
    """
    if not documents:
        raise ValueError("No documents to index. Add files to data/documents first.")

    parents, children = split_documents(documents)
    if not children:
        raise ValueError("Chunking produced no child chunks. Check the source documents.")

    create_vector_store(children)
    save_doc_store(parents, children)
    return {
        "documents": len(documents),
        "parents": len(parents),
        "children": len(children),
    }


def update_index(
    documents: List[Document],
    *,
    remove_missing: bool = False,
) -> Dict[str, int]:
    """Add new/changed documents and optionally remove sources absent from input.

    If ``remove_missing=True``, ``documents`` MUST represent a complete snapshot
    of the source folder. This mode removes indexed files not present in that
    snapshot. Leave it False when passing only a subset of documents.

    The vector store and JSON store are separate persistence systems; a process
    failure between vector operations and save_doc_store can still require a
    rebuild to restore consistency.
    """
    existing_parents, existing_children = load_doc_store()

    if not documents:
        if not remove_missing or not existing_children:
            return {
                "added_children": 0,
                "removed_children": 0,
                "skipped_children": 0,
            }

        # An explicit empty full snapshot means remove all indexed sources.
        store = load_vector_store()
        backend = _validate_vector_db()
        _delete_ids(store, _point_ids(existing_children), backend)
        if backend == "faiss":
            store.save_local(str(config.FAISS_DIR))
        save_doc_store([], [])
        return {
            "added_children": 0,
            "removed_children": len(existing_children),
            "skipped_children": 0,
        }

    if not existing_children:
        result = build_index(documents)
        return {
            "added_children": result["children"],
            "removed_children": 0,
            "skipped_children": 0,
        }

    new_parents, new_children = split_documents(documents)

    old_hashes = _by_filename(existing_children)
    new_hashes = _by_filename(new_children)

    unchanged = {
        name
        for name, file_hash in new_hashes.items()
        if file_hash and old_hashes.get(name) == file_hash
    }
    changed = {
        name
        for name in new_hashes
        if name in old_hashes and name not in unchanged
    }
    missing = (set(old_hashes) - set(new_hashes)) if remove_missing else set()
    to_remove = changed | missing

    stale_children = [doc for doc in existing_children if _filename(doc) in to_remove]
    children_to_add = [
        doc for doc in new_children if _filename(doc) not in unchanged
    ]
    parents_to_add = [
        doc for doc in new_parents if _filename(doc) not in unchanged
    ]
    skipped = len(new_children) - len(children_to_add)

    if not children_to_add and not stale_children:
        return {
            "added_children": 0,
            "removed_children": 0,
            "skipped_children": skipped,
        }

    store = load_vector_store()
    backend = _validate_vector_db()

    # Delete changed/missing chunks first so old content cannot remain searchable.
    # This sequence is not transactional; rebuild the index if a later operation
    # fails before the JSON store is updated.
    if stale_children:
        _delete_ids(store, _point_ids(stale_children), backend)

    if children_to_add:
        new_ids = _point_ids(children_to_add)
        if len(new_ids) != len(set(new_ids)):
            raise ValueError(
                "Duplicate child point IDs detected while updating the index. "
                "Check source identity and chunk IDs."
            )
        store.add_documents(children_to_add, ids=new_ids)

    if backend == "faiss":
        store.save_local(str(config.FAISS_DIR))

    kept_parents = [
        doc for doc in existing_parents if _filename(doc) not in to_remove
    ]
    kept_children = [
        doc for doc in existing_children if _filename(doc) not in to_remove
    ]
    save_doc_store(
        kept_parents + parents_to_add,
        kept_children + children_to_add,
    )

    return {
        "added_children": len(children_to_add),
        "removed_children": len(stale_children),
        "skipped_children": skipped,
    }


# -------------------------------------------------------------------
# Maintenance
# -------------------------------------------------------------------

def clear_index() -> None:
    """Delete the selected vector index and the shared local JSON document store.

    The JSON document store is shared by both backends. Therefore this function
    intentionally clears it too; it cannot preserve metadata for the other
    backend independently.
    """
    backend = _validate_vector_db()

    if backend == "faiss":
        if config.FAISS_DIR.exists():
            shutil.rmtree(config.FAISS_DIR)
    else:
        client = _qdrant_client()
        if client.collection_exists(config.QDRANT_COLLECTION_NAME):
            client.delete_collection(config.QDRANT_COLLECTION_NAME)

    if config.DOC_STORE_PATH.exists():
        config.DOC_STORE_PATH.unlink()

    logger.info("Index cleared for backend: %s", backend)


def index_exists() -> bool:
    """Return True if the JSON store and selected vector index both exist."""
    if not config.DOC_STORE_PATH.exists():
        return False

    backend = _validate_vector_db()
    if backend == "faiss":
        return (
            (config.FAISS_DIR / "index.faiss").exists()
            and (config.FAISS_DIR / "index.pkl").exists()
        )

    try:
        client = _qdrant_client()
        return client.collection_exists(config.QDRANT_COLLECTION_NAME)
    except Exception:
        logger.exception("Could not check Qdrant index existence")
        return False


def index_summary() -> Dict[str, object]:
    """Return parent/child counts and indexed source filenames."""
    parents, children = load_doc_store()
    filenames = sorted({_filename(doc) for doc in parents})
    return {
        "parents": len(parents),
        "children": len(children),
        "files": filenames,
    }
