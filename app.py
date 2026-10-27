"""
Streamlit chat UI for the RAG document-question-answering pipeline.

Run from the project root:

    python -m streamlit run app.py
"""

import html
import logging
import threading

import streamlit as st

# set_page_config must be the first Streamlit call.
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Import the backend. Show a friendly error if a dependency is missing.
# -------------------------------------------------------------------
try:
    import config
    import src.pipeline as pipeline_module
    import src.retrieval as retrieval_module
    from src.memory import clear_memory, create_memory
    from src.pipeline import (
        ANSWERED,
        BLOCKED_INPUT,
        BLOCKED_OUTPUT,
        ERROR,
        NOT_FOUND,
        RagPipeline,
    )
    from src.reranking import SIMPLE_FALLBACK_MIN_SCORE
except Exception as import_error:  # noqa: BLE001
    st.error(
        "Could not import the RAG backend. Make sure you started the app from "
        "the project root and that all dependencies are installed."
    )
    st.caption(f"Import problem: {type(import_error).__name__}: {import_error}")
    st.stop()

# describe_source is only used by the backend's own terminal chat; import it
# the same way pipeline.main() does, and fall back gracefully if unavailable.
try:
    from src.refinement import describe_source as describe_source_fn
except Exception:  # noqa: BLE001

    def describe_source_fn(doc):  # type: ignore[misc]
        return "Document excerpt"


# -------------------------------------------------------------------
# Read-only probes (the backend files are NOT edited)
#
# RagPipeline.ask() looks up hybrid_retrieve, record_parent,
# filter_unsafe_documents and rerank_documents in the src.pipeline module,
# and hybrid_retrieve looks up search_vector_store and search_keywords in
# src.retrieval. We replace those names with wrappers that record what went
# in and out, call the original function unchanged, and return its result
# untouched. A failure inside a wrapper's recording never affects the answer.
#
# The capture object lives in st.cache_resource, because Streamlit re-runs
# this whole script on every interaction and a plain module-level object
# would be recreated each time. It is thread-local, so concurrent browser
# sessions do not mix.
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_capture():
    return threading.local()


_capture = get_capture()

RRF_K = 60.0  # same constant as fuse_results() in src/retrieval.py


def _store(key: str, value) -> None:
    data = getattr(_capture, "data", None)
    if isinstance(data, dict):
        data[key] = value


def _chunk_key(doc) -> str:
    """Same de-duplication key that fuse_results() uses."""
    return doc.metadata.get("chunk_id") or doc.page_content


def _describe_chunk(doc) -> dict:
    metadata = getattr(doc, "metadata", None) or {}
    try:
        title = str(describe_source_fn(doc))
    except Exception:  # noqa: BLE001
        title = "Document excerpt"
    text = " ".join(str(getattr(doc, "page_content", "")).split())
    return {
        "title": title,
        "chunk_id": metadata.get("chunk_id"),
        "parent_id": metadata.get("parent_id"),
        "score": metadata.get("rerank_score"),
        "preview": text[:100],
        "text": text,
    }


class _FallbackFilter(logging.Filter):
    """Flags the thread when the reranker logs its cross-encoder fallback.

    A logger filter (not a handler) is used on purpose: adding a handler
    would stop Python's default stderr output for that logger.
    """

    _rag_ui = True

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if "falling back to the simple reranker" in record.getMessage():
                _capture.fallback = True
        except Exception:  # noqa: BLE001
            pass
        return True


# ---- search wrappers (inside src.retrieval) -----------------------
def _make_vector_probe(original):
    def probe(*args, **kwargs):
        query = args[1] if len(args) > 1 else kwargs.get("query")
        log = getattr(_capture, "searches", None)
        try:
            docs = original(*args, **kwargs)
        except Exception as exc:
            if log is not None:
                log.append(
                    {"kind": "vector", "query": query, "docs": [],
                     "error": type(exc).__name__, "bm25": []}
                )
            raise
        if log is not None:
            docs_list = list(docs)
            log.append(
                {"kind": "vector", "query": query, "docs": docs_list,
                 "error": None, "bm25": [None] * len(docs_list)}
            )
        return docs

    return probe


def _make_keyword_probe(original):
    def probe(*args, **kwargs):
        retriever = args[0] if args else kwargs.get("retriever")
        query = args[1] if len(args) > 1 else kwargs.get("query")
        log = getattr(_capture, "searches", None)
        try:
            docs = original(*args, **kwargs)
        except Exception as exc:
            if log is not None:
                log.append(
                    {"kind": "keyword", "query": query, "docs": [],
                     "error": type(exc).__name__, "bm25": []}
                )
            raise
        if log is not None:
            docs_list = list(docs)
            by_id = {}
            try:
                # Same calls search_keywords() itself makes.
                tokens = retriever.preprocess_func(query)
                scores = retriever.vectorizer.get_scores(tokens)
                index_of = {id(d): i for i, d in enumerate(retriever.docs)}
                for d in docs_list:
                    i = index_of.get(id(d))
                    if i is not None:
                        by_id[id(d)] = float(scores[i])
            except Exception:  # noqa: BLE001
                by_id = {}
            log.append(
                {"kind": "keyword", "query": query, "docs": docs_list,
                 "error": None, "bm25": [by_id.get(id(d)) for d in docs_list]}
            )
        return docs

    return probe


# ---- hybrid_retrieve ----------------------------------------------
def _summarise_hybrid(queries, searches, fused) -> dict:
    rrf, appearances, search_rows = {}, {}, []
    vector_hits = keyword_hits = 0

    for s in searches:
        q = s["query"]
        label = f"Q{queries.index(q) + 1}" if q in queries else "Q?"
        search_rows.append(
            {
                "Query": label,
                "Retriever": s["kind"],
                "Hits": len(s["docs"]),
                "Status": f"failed ({s['error']})" if s["error"] else "ok",
                "Text": q,
            }
        )
        if s["kind"] == "vector":
            vector_hits += len(s["docs"])
        else:
            keyword_hits += len(s["docs"])
        for rank, doc in enumerate(s["docs"]):
            key = _chunk_key(doc)
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank)
            appearances.setdefault(key, []).append(
                {
                    "kind": s["kind"],
                    "label": label,
                    "rank": rank + 1,
                    "bm25": s["bm25"][rank] if rank < len(s["bm25"]) else None,
                }
            )

    chunks, scores = [], []
    for position, doc in enumerate(fused, start=1):
        key = _chunk_key(doc)
        apps = appearances.get(key, [])
        kinds = sorted({a["kind"] for a in apps})
        bm25_values = [a["bm25"] for a in apps if a["bm25"] is not None]
        info = _describe_chunk(doc)
        info.update(
            {
                "position": position,
                "found_by": " + ".join(kinds) if kinds else "unknown",
                "appearances": ", ".join(
                    f"{a['kind']} {a['label']} #{a['rank']}" for a in apps
                ),
                "rrf": rrf.get(key),
                "bm25": max(bm25_values) if bm25_values else None,
            }
        )
        scores.append(info["rrf"])
        chunks.append(info)

    order_ok = bool(scores) and all(s is not None for s in scores) and all(
        scores[i] >= scores[i + 1] - 1e-9 for i in range(len(scores) - 1)
    )

    return {
        "queries": list(queries),
        "searches": search_rows,
        "vector_hits": vector_hits,
        "keyword_hits": keyword_hits,
        "unique": len(rrf),
        "chunks": chunks,
        "order_ok": order_ok if scores else True,
        "settings": {
            "vector_top_k": getattr(config, "VECTOR_TOP_K", None),
            "bm25_top_k": getattr(config, "BM25_TOP_K", None),
            "hybrid_top_k": getattr(config, "HYBRID_TOP_K", None),
        },
    }


def _make_hybrid_probe(original):
    def probe(*args, **kwargs):
        queries = kwargs.get("queries", args[0] if args else [])
        _capture.searches = []
        try:
            fused = original(*args, **kwargs)
        except Exception:
            _capture.searches = None
            raise
        searches = _capture.searches or []
        _capture.searches = None
        try:
            _store("hybrid", _summarise_hybrid(list(queries), searches, list(fused)))
        except Exception:  # noqa: BLE001
            logger.exception("Hybrid probe failed")
        return fused

    return probe


# ---- record_parent -------------------------------------------------
def _make_parent_probe(original):
    def probe(*args, **kwargs):
        children = args[0] if args else kwargs.get("children")
        lookup = args[1] if len(args) > 1 else kwargs.get("parent_lookup")
        result = original(*args, **kwargs)
        try:
            children_list = list(children)
            parents_out = list(result)

            # Mirror record_parent()'s grouping to count child hits per parent.
            groups, order = {}, []
            for rank, child in enumerate(children_list, start=1):
                pid = child.metadata.get("parent_id")
                parent_doc = lookup.get(pid) if (pid and lookup is not None) else None
                if parent_doc is None:
                    key = child.metadata.get("chunk_id") or child.page_content
                    missing = True
                else:
                    key = pid
                    missing = False
                if key not in groups:
                    groups[key] = {"ranks": [], "missing": missing, "parent_id": pid}
                    order.append(key)
                groups[key]["ranks"].append(rank)

            rows = None
            if len(order) == len(parents_out):
                rows = []
                for key, parent in zip(order, parents_out):
                    group = groups[key]
                    info = _describe_chunk(parent)
                    rows.append(
                        {
                            "Parent": info["title"],
                            "parent_id": group["parent_id"],
                            "Child hits": len(group["ranks"]),
                            "Child fused ranks": ", ".join(f"#{r}" for r in group["ranks"]),
                            "Note": "parent not found, child used" if group["missing"] else "",
                        }
                    )
            _store(
                "parents",
                {
                    "children_in": len(children_list),
                    "parents_out": len(parents_out),
                    "rows": rows,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("Parent probe failed")
        return result

    return probe


# ---- filter_unsafe_documents --------------------------------------
def _make_guard_probe(original):
    def probe(documents, *args, **kwargs):
        docs_in = list(documents)
        result = original(docs_in, *args, **kwargs)
        try:
            docs_out = list(result)
            kept_texts = {d.page_content for d in docs_out}
            removed = [d for d in docs_in if d.page_content not in kept_texts]
            _store(
                "guardrail",
                {
                    "before": len(docs_in),
                    "after": len(docs_out),
                    "removed": [_describe_chunk(d) for d in removed],
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("Guardrail probe failed")
        return result

    return probe


# ---- rerank_documents ----------------------------------------------
def _make_rerank_probe(original):
    def probe(query, documents, *args, **kwargs):
        docs_in = list(documents)
        _capture.fallback = False
        result = original(query, docs_in, *args, **kwargs)
        try:
            kept = list(result)

            # Effective settings, mirroring rerank_documents' own defaults.
            top_k = args[0] if len(args) > 0 else kwargs.get("top_k")
            min_score = args[1] if len(args) > 1 else kwargs.get("min_score")
            top_k = config.RERANK_TOP_K if top_k is None else top_k
            min_score = config.MIN_RERANK_SCORE if min_score is None else min_score

            fallback = bool(getattr(_capture, "fallback", False))
            if fallback:
                min_score = SIMPLE_FALLBACK_MIN_SCORE  # what the backend switches to
                scorer = "keyword overlap, 0 to 1 (cross-encoder failed, fallback used)"
            elif config.RERANKER == "cross-encoder":
                scorer = f"cross-encoder, raw logits ({config.CROSS_ENCODER_MODEL})"
            else:
                scorer = "keyword overlap, 0 to 1 (simple reranker)"

            _store(
                "rerank",
                {
                    "query": query,
                    "candidates": [_describe_chunk(d) for d in docs_in],
                    "kept": [_describe_chunk(d) for d in kept],
                    "top_k": top_k,
                    "min_score": min_score,
                    "scorer": scorer,
                    "fallback": fallback,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("Rerank probe failed")
        return result

    return probe


# ---- installation --------------------------------------------------
def _unwrap(func):
    while getattr(func, "_is_probe", False):
        func = func._original
    return func


def _patch(module, name, factory):
    """Wrap module.<name>. Re-wraps cleanly on every Streamlit rerun."""
    original = _unwrap(getattr(module, name))
    probe = factory(original)
    probe._is_probe = True  # type: ignore[attr-defined]
    probe._original = original  # type: ignore[attr-defined]
    setattr(module, name, probe)
    return original


def _install_probes() -> None:
    _patch(retrieval_module, "search_vector_store", _make_vector_probe)
    _patch(retrieval_module, "search_keywords", _make_keyword_probe)
    _patch(pipeline_module, "hybrid_retrieve", _make_hybrid_probe)
    _patch(pipeline_module, "record_parent", _make_parent_probe)
    _patch(pipeline_module, "filter_unsafe_documents", _make_guard_probe)
    rerank_original = _patch(pipeline_module, "rerank_documents", _make_rerank_probe)

    rerank_logger = logging.getLogger(rerank_original.__module__)
    for existing in list(rerank_logger.filters):
        if getattr(existing, "_rag_ui", False):
            rerank_logger.removeFilter(existing)
    rerank_logger.addFilter(_FallbackFilter())


_install_probes()

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 900px; }
        .status-badge {
            display: inline-block;
            padding: 0.1rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #ffffff;
            margin-right: 0.5rem;
        }
        .meta-text { font-size: 0.8rem; opacity: 0.7; }
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# status value -> (label, colour)
STATUS_STYLE = {
    ANSWERED: ("Answered", "#1a7f37"),
    NOT_FOUND: ("Not found in documents", "#9a6700"),
    BLOCKED_INPUT: ("Question blocked", "#cf222e"),
    BLOCKED_OUTPUT: ("Answer blocked", "#cf222e"),
    ERROR: ("Error", "#6e7781"),
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading index and models...")
def load_pipeline() -> RagPipeline:
    """
    Create ONE RagPipeline for the whole server process.

    Streamlit does not cache a call that raises, so after fixing a problem
    (for example building the index) the next attempt tries again.
    """
    return RagPipeline()


def safe_markdown(text: str) -> str:
    """Escape '$' so Streamlit does not treat prices as LaTeX."""
    return str(text).replace("$", "\\$")


def fmt_score(score) -> str:
    return f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"


def render_badge(status: str, elapsed: float) -> None:
    label, colour = STATUS_STYLE.get(status, (status, "#6e7781"))
    st.markdown(
        f'<span class="status-badge" style="background:{colour}">'
        f"{html.escape(label)}</span>"
        f'<span class="meta-text">{elapsed:.2f}s</span>',
        unsafe_allow_html=True,
    )


def render_funnel(entry: dict) -> None:
    """One line: how many chunks survive each stage."""
    trace = entry.get("trace")
    if not trace:
        st.caption("Retrieval stats: retrieval was not reached for this question.")
        return

    parts = []
    hybrid = trace.get("hybrid")
    if hybrid:
        parts.append(
            f"unique chunks found: **{hybrid['unique']}** → after fusion: "
            f"**{len(hybrid['chunks'])}**"
        )
    parents = trace.get("parents")
    if parents:
        parts.append(f"parents: **{parents['parents_out']}**")
    guard = trace.get("guardrail")
    if guard:
        parts.append(f"after guardrail: **{guard['after']}**")
    rerank = trace.get("rerank")
    if rerank:
        parts.append(f"kept by reranker: **{len(rerank['kept'])}**")
    parts.append(f"used as sources: **{len(entry.get('sources') or [])}**")
    st.caption(" → ".join(parts))


def render_sources(sources) -> None:
    """One expander per source, numbered to match the [n] markers in the answer."""
    if not sources:
        return
    st.markdown("**Sources**")
    for number, doc in enumerate(sources, start=1):
        try:
            title = str(describe_source_fn(doc))
        except Exception:  # noqa: BLE001
            title = "Document excerpt"
        metadata = getattr(doc, "metadata", None) or {}
        header = f"[{number}] {title}"
        if "rerank_score" in metadata:
            header += f"  (score {fmt_score(metadata['rerank_score'])})"
        with st.expander(header):
            st.markdown(safe_markdown(doc.page_content))
            if metadata:
                st.divider()
                for key, value in metadata.items():
                    st.caption(f"{key}: {value}")


# ---- retrieval trace tabs -----------------------------------------
def render_hybrid_tab(hybrid) -> None:
    if not hybrid:
        st.caption("Hybrid search data was not captured for this question.")
        return

    settings = hybrid["settings"]
    st.caption(
        f"Per query: vector top-k **{settings['vector_top_k']}**, BM25 top-k "
        f"**{settings['bm25_top_k']}**. Fusion (RRF) keeps the top "
        f"**{settings['hybrid_top_k']}** chunks. Q1 is your question as typed; "
        "later queries are the LLM's rewrites."
    )

    failed = [s for s in hybrid["searches"] if s["Status"] != "ok"]
    if failed:
        st.warning(
            f"{len(failed)} search(es) failed; the pipeline continued with the rest. "
            "See the terminal log for the underlying error."
        )

    st.markdown("**Searches run**")
    if hybrid["searches"]:
        st.dataframe(hybrid["searches"], hide_index=True)
    st.caption(
        f"Total hits: vector **{hybrid['vector_hits']}**, keyword "
        f"**{hybrid['keyword_hits']}** → unique chunks **{hybrid['unique']}** → "
        f"kept after fusion **{len(hybrid['chunks'])}** "
        f"({max(hybrid['unique'] - len(hybrid['chunks']), 0)} cut by the fusion cap)."
    )

    st.markdown("**Candidate chunks after fusion** (child chunks, best first)")
    if not hybrid["chunks"]:
        st.caption("No chunks were returned by hybrid search.")
    else:
        rows = []
        for item in hybrid["chunks"]:
            rows.append(
                {
                    "Rank": item["position"],
                    "Source": item["title"],
                    "chunk_id": item["chunk_id"],
                    "parent_id": item["parent_id"],
                    "Found by": item["found_by"],
                    "RRF score": round(item["rrf"], 5) if item["rrf"] is not None else None,
                    "Best BM25": round(item["bm25"], 3) if item["bm25"] is not None else None,
                    "Appearances": item["appearances"],
                    "Text": item["text"],
                }
            )
        st.dataframe(rows, hide_index=True)

    order_note = (
        "The recomputed order matches the order the backend returned."
        if hybrid["order_ok"]
        else "WARNING: the recomputed RRF order does not match the backend's order, "
        "so treat the RRF column with caution."
    )
    st.caption(
        "RRF score is recomputed here with the backend's formula, 1/(60 + rank) "
        "summed over every search that returned the chunk, because fuse_results() "
        f"discards it. {order_note} Best BM25 is the raw BM25 score from the same "
        "keyword index. Vector similarity scores are not shown: the backend's "
        "similarity_search returns documents without scores."
    )


def render_parents_tab(parents, guard) -> None:
    if not parents:
        st.caption("Parent expansion data was not captured for this question.")
    else:
        st.markdown(
            f"**Child to parent:** {parents['children_in']} child chunks → "
            f"{parents['parents_out']} unique parents"
        )
        if parents["rows"]:
            st.dataframe(parents["rows"], hide_index=True)
        else:
            st.caption("Could not align parents with child hits for this question.")

    st.markdown("**Context guardrail**")
    if not guard:
        st.caption("Guardrail data was not captured for this question.")
        return
    removed = guard["removed"]
    st.caption(
        f"{guard['before']} parents in → {guard['after']} out "
        f"({guard['before'] - guard['after']} removed)."
    )
    if removed:
        st.dataframe(
            [
                {"Source": r["title"], "parent_id": r["parent_id"], "Text": r["preview"]}
                for r in removed
            ],
            hide_index=True,
        )
        st.caption("Removed chunks are matched by text, so this list is best-effort.")


def render_rerank_tab(rerank) -> None:
    if not rerank:
        st.caption("Reranking data was not captured for this question.")
        return

    before = len(rerank["candidates"])
    kept = rerank["kept"]
    dropped = before - len(kept)
    cutoff = rerank["min_score"]

    if rerank["fallback"]:
        st.warning(
            "The cross-encoder failed for this question, so the backend used "
            "the simple keyword reranker. Scores below are on a 0 to 1 scale."
        )
    st.caption(f"Scorer: {rerank['scorer']}")
    st.caption(
        f"Cutoff: chunks scoring at or below **{cutoff}** are dropped. "
        f"Top-k: **{rerank['top_k']}**."
    )
    st.caption(f"Reranked against: {safe_markdown(rerank['query'])}")

    st.markdown(f"**Kept by the reranker ({len(kept)} of {before})**")
    if kept:
        rows = []
        for position, item in enumerate(kept, start=1):
            score = item["score"]
            margin = round(score - cutoff, 4) if isinstance(score, (int, float)) else None
            rows.append(
                {
                    "Rank": position,
                    "Source": item["title"],
                    "Score": score,
                    "Above cutoff by": margin,
                }
            )
        st.dataframe(rows, hide_index=True)
    else:
        st.caption("The reranker kept no chunks.")

    st.caption(
        f"Dropped: **{dropped}** chunk(s), either at or below the cutoff or "
        "beyond top-k. The backend does not expose scores for dropped chunks."
    )

    st.markdown("**Candidates before reranking** (retrieval order)")
    if rerank["candidates"]:
        st.dataframe(
            [
                {"Position": i, "Source": c["title"], "Preview": c["preview"]}
                for i, c in enumerate(rerank["candidates"], start=1)
            ],
            hide_index=True,
        )


def render_trace(entry: dict) -> None:
    trace = entry.get("trace")
    if not trace:
        return
    with st.expander("Retrieval trace (hybrid search, parents, reranking)"):
        tab_hybrid, tab_parents, tab_rerank = st.tabs(
            ["1. Hybrid search", "2. Parents and guardrail", "3. Reranking"]
        )
        with tab_hybrid:
            render_hybrid_tab(trace.get("hybrid"))
        with tab_parents:
            render_parents_tab(trace.get("parents"), trace.get("guardrail"))
        with tab_rerank:
            render_rerank_tab(trace.get("rerank"))


def render_details(entry: dict) -> None:
    """Optional debug info: what was searched and which guardrail flags fired."""
    if not st.session_state.show_details:
        return
    with st.expander("Details (queries and guardrail flags)"):
        queries = entry.get("queries") or []
        flags = entry.get("flags") or []
        st.markdown("**Queries searched**")
        if queries:
            for query in queries:
                st.markdown(f"- {safe_markdown(query)}")
        else:
            st.caption("None")
        st.markdown("**Flags**")
        if flags:
            for flag in flags:
                st.markdown(f"- `{flag}`")
        else:
            st.caption("None")


def render_assistant(entry: dict) -> None:
    st.markdown(safe_markdown(entry["content"]))
    render_badge(entry["status"], entry["elapsed"])
    render_funnel(entry)
    if entry["status"] == ANSWERED:
        render_sources(entry["sources"])
    render_trace(entry)
    render_details(entry)


def render_entry(entry: dict) -> None:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.markdown(safe_markdown(entry["content"]))
    else:
        with st.chat_message("assistant"):
            render_assistant(entry)


# -------------------------------------------------------------------
# Session state: one memory + one display transcript per browser session
# -------------------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = create_memory()
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "show_details" not in st.session_state:
    st.session_state.show_details = False

# -------------------------------------------------------------------
# Load the pipeline (cached). Handle a missing index gracefully.
# -------------------------------------------------------------------
try:
    pipeline = load_pipeline()
except RuntimeError as error:
    # reload() raises RuntimeError when no index exists.
    st.title("📄 Document Assistant")
    st.error(str(error))
    st.info(
        "Build your vector database with your existing indexing step, "
        "then click the button below."
    )
    if st.button("Try again"):
        st.rerun()
    st.stop()
except Exception as error:  # noqa: BLE001
    # Do not print the message: configuration errors could mention settings.
    logger.exception("Pipeline failed to start")
    st.title("📄 Document Assistant")
    st.error(
        f"The pipeline could not start ({type(error).__name__}). "
        "Check your configuration and the terminal output for details."
    )
    if st.button("Try again"):
        st.rerun()
    st.stop()

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Document Assistant")
    st.caption("Answers come only from your indexed documents, with numbered sources.")

    st.divider()
    questions_asked = sum(1 for e in st.session_state.transcript if e["role"] == "user")
    st.metric("Questions this session", questions_asked)
    st.caption(f"Memory window: last {config.MEMORY_WINDOW} exchange(s)")

    if st.button("🗑️ Clear conversation"):
        clear_memory(st.session_state.memory)
        st.session_state.transcript = []
        st.rerun()

    st.divider()
    st.caption("Retrieval settings (from config)")
    st.markdown(
        f"- Query variants: `{getattr(config, 'NUM_QUERY_VARIANTS', '?')}`\n"
        f"- Vector top-k: `{getattr(config, 'VECTOR_TOP_K', '?')}`\n"
        f"- BM25 top-k: `{getattr(config, 'BM25_TOP_K', '?')}`\n"
        f"- Hybrid top-k: `{getattr(config, 'HYBRID_TOP_K', '?')}`\n"
        f"- Reranker: `{config.RERANKER}`\n"
        f"- Rerank top-k: `{config.RERANK_TOP_K}`\n"
        f"- Min rerank score: `{config.MIN_RERANK_SCORE}`"
    )

    st.session_state.show_details = st.toggle(
        "Show query and guardrail details",
        value=st.session_state.show_details,
    )

    if st.button("🔄 Reload index from disk"):
        try:
            with st.spinner("Reloading index..."):
                pipeline.reload()
            st.success("Index reloaded.")
        except RuntimeError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001
            logger.exception("Reload failed")
            st.error(f"Reload failed ({type(error).__name__}). See the terminal.")

    st.divider()
    st.caption("Status legend")
    for label, colour in STATUS_STYLE.values():
        st.markdown(
            f'<span class="status-badge" style="background:{colour}">'
            f"{html.escape(label)}</span>",
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------------
# Main chat area
# -------------------------------------------------------------------
st.title("Ask your documents")

if not st.session_state.transcript:
    st.info("Type a question below to get started. Follow-up questions are understood.")

for entry in st.session_state.transcript:
    render_entry(entry)

prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    user_entry = {"role": "user", "content": prompt}
    st.session_state.transcript.append(user_entry)
    render_entry(user_entry)

    with st.chat_message("assistant"):
        try:
            # Fresh capture for this question, so stale data is never shown.
            _capture.data = {}
            _capture.searches = None
            _capture.fallback = False

            with st.spinner("Searching your documents..."):
                result = pipeline.ask(prompt, st.session_state.memory)

            captured = getattr(_capture, "data", None)
            assistant_entry = {
                "role": "assistant",
                "content": result.answer,
                "status": result.status,
                "elapsed": result.elapsed_seconds,
                "sources": list(result.sources),
                "queries": list(result.queries),
                "flags": list(result.flags),
                "trace": dict(captured) if captured else None,
            }
        except Exception:  # noqa: BLE001
            # ask() already catches its own errors; this is a last safety net.
            logger.exception("Unexpected failure in ask()")
            assistant_entry = {
                "role": "assistant",
                "content": "Sorry, something went wrong while answering. Please try again.",
                "status": ERROR,
                "elapsed": 0.0,
                "sources": [],
                "queries": [],
                "flags": [],
                "trace": None,
            }

        render_assistant(assistant_entry)

    st.session_state.transcript.append(assistant_entry)