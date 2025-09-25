# app.py — Regulatory Response Assistant (Streamlit + LlamaIndex <= 0.12.53)
# -------------------------------------------------------------------------
# Upload Company Policy docs and Prior Regulator Response emails (PDF/DOCX/TXT)
# Dynamically add questions (textarea or table)
# Retrieve evidence from both corpora, draft regulator-ready reply, judge & cite
#
# This version removes all tokenization/truncation logic.
# It relies purely on LlamaIndex chunking (SentenceSplitter) to keep chunks manageable.

import os
import io
import re
import json
import time
import sqlite3
import hashlib
from html import escape
from typing import List, Tuple, Dict, Optional

import streamlit as st
import docx2txt
from PyPDF2 import PdfReader

# Optional preview backends (visual PDF previews)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

# ---- LlamaIndex (<=0.12.53) ----
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import VectorStoreQuery

# ---------------------------
# Page / Header
# ---------------------------
st.set_page_config(page_title="Regulatory Response Assistant", layout="wide")
st.title("📄 Regulatory Response Assistant (LlamaIndex)")
st.caption("Upload policy docs + prior regulator responses → Ask regulator questions → Get evidence-backed drafts with citations.")

# ---------------------------
# Sidebar: Settings
# ---------------------------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    base_url = st.text_input("OPENAI_BASE_URL (optional)", value=os.getenv("OPENAI_BASE_URL", ""))

    chat_model = st.text_input("Chat Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    embed_model = st.text_input("Embedding Model", value=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    top_k = st.slider("Top-K per corpus", min_value=2, max_value=15, value=5, step=1)
    temperature = st.slider("Draft temperature", 0.0, 1.0, 0.2, 0.1)

    # Embedding performance controls
    st.markdown("### Embedding Performance")
    embed_page_batch = st.slider("Embed page batch (pages → nodes)", 2, 64, 16, 2)
    embed_api_batch  = st.slider("Embed API batch (texts per request)", 4, 128, 32, 4)

    # NOTE: kept for UI continuity; ignored in chunking-only mode
    embed_max_tokens = st.slider("Max tokens per node (ignored with chunking-only)", 256, 4096, 768, 64)

    # Toggle-only debugger (persisted; used inside build function)
    st.toggle("Enable embedding debugger", value=False, key="embed_debug")

    st.toggle("Show Judge Raw Output (debug)", value=False, key="show_debug")

    st.markdown("---")
    if fitz and Image and ImageDraw:
        st.info("PDF visual preview: **Enabled** (pymupdf + Pillow detected)")
    else:
        st.info("PDF visual preview: **Text-only fallback** (install `pymupdf` and `Pillow` to enable images)")
    st.caption("Evidence-driven flow only. No deterministic or hard-coded validations.")

    # Clear/reset button
    if st.button("🧹 Clear loaded docs (reset)"):
        st.session_state.policy_docs = []
        st.session_state.email_docs = []
        st.session_state.policy_loaded_hashes = set()
        st.session_state.email_loaded_hashes = set()
        st.session_state.policy_files_set = set()
        st.session_state.email_files_set = set()
        st.session_state.file_bytes_map = {}
        st.session_state.pdf_page_texts = {}
        st.session_state.policy_index = None
        st.session_state.email_index = None
        st.session_state.vector_ready = False
        st.success("Cleared all loaded documents and indexes.")

# ---------------------------
# Configure LlamaIndex defaults
# ---------------------------
def configure_settings():
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url

    Settings.llm = OpenAI(
        model=chat_model,
        temperature=temperature,
        api_key=api_key or None,
        **kwargs,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=embed_model,
        api_key=api_key or None,
        base_url=base_url or None
    )
    # Chunking controls the node size; truncation removed.
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

configure_settings()

# ---------------------------
# File readers & PDF helpers
# ---------------------------
def read_pdf_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i + 1, txt))
    return out

def read_docx(file_bytes: bytes) -> str:
    tmp_path = "tmp_doc.docx"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    text = docx2txt.process(tmp_path) or ""
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return text

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def highlight_html(text: str, needle: str, window: int = 400) -> str:
    text_norm = re.sub(r"\s+", " ", text or "")
    n = (needle or "").strip()

    def esc(s: str) -> str:
        return escape(s, quote=False)

    if not n:
        return f"<pre>{esc(text_norm[:window])}...</pre>"

    idx = text_norm.lower().find(n.lower())
    if idx == -1:
        return f"<pre>{esc(text_norm[:window])}...</pre>"

    start = max(0, idx - window // 2)
    end = min(len(text_norm), idx + len(n) + window // 2)
    before = esc(text_norm[start:idx])
    match = esc(text_norm[idx:idx + len(n)])
    after = esc(text_norm[idx + len(n):end])

    return f"<pre>{before}<mark>{match}</mark>{after}</pre>"

def render_pdf_page_with_highlight(file_bytes: bytes, page_num: int, query_text: str):
    if not (fitz and Image and ImageDraw):
        return None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[page_num - 1]
        q = (query_text or "").strip()
        if len(q) > 120:
            q = q[:120]
        rects = page.search_for(q) if q else []
        pix = page.get_pixmap(dpi=144)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if rects:
            draw = ImageDraw.Draw(img)
            for r in rects[:5]:
                draw.rectangle([(r.x0, r.y0), (r.x1, r.y1)], outline=(255, 0, 0), width=4)
        return img
    except Exception:
        return None

# ---------------------------
# Session state
# ---------------------------
if "policy_docs" not in st.session_state:
    st.session_state.policy_docs: List[Document] = []
if "email_docs" not in st.session_state:
    st.session_state.email_docs: List[Document] = []

if "policy_index" not in st.session_state:
    st.session_state.policy_index = None
if "email_index" not in st.session_state:
    st.session_state.email_index = None

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "policy_files_set" not in st.session_state:
    st.session_state.policy_files_set = set()
if "email_files_set" not in st.session_state:
    st.session_state.email_files_set = set()

if "policy_loaded_hashes" not in st.session_state:
    st.session_state.policy_loaded_hashes = set()
if "email_loaded_hashes" not in st.session_state:
    st.session_state.email_loaded_hashes = set()

if "file_bytes_map" not in st.session_state:
    st.session_state.file_bytes_map: Dict[str, bytes] = {}
if "pdf_page_texts" not in st.session_state:
    st.session_state.pdf_page_texts: Dict[str, Dict[int, str]] = {}

if "embed_cache_path" not in st.session_state:
    st.session_state.embed_cache_path = "embed_cache.sqlite3"

def _init_embed_cache(path: str):
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emb_cache (
                  h TEXT PRIMARY KEY,
                  v TEXT NOT NULL
                )
            """)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
    finally:
        conn.close()

_init_embed_cache(st.session_state.embed_cache_path)

# ---------------------------
# Uploaders
# ---------------------------
policy_files = st.file_uploader(
    "Upload Company Policy documents (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)
email_files = st.file_uploader(
    "Upload Prior Regulator Response emails (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

def add_to_docs(files, target_list, corpus_name: str):
    for f in files:
        name = f.name
        data = f.read()
        f.seek(0)
        digest = fingerprint_bytes(data)

        seen_set = (
            st.session_state.policy_loaded_hashes
            if corpus_name == "policy"
            else st.session_state.email_loaded_hashes
        )
        if digest in seen_set:
            continue

        lower = name.lower()
        if lower.endswith(".pdf"):
            pages = read_pdf_pages(data)
            if not any(txt.strip() for _, txt in pages):
                st.warning(f"Empty/unsupported PDF: {name}")
                continue
            for page_num, txt in pages:
                target_list.append(Document(text=txt, metadata={"source": name, "page": page_num}))
            st.session_state.file_bytes_map[name] = data
            st.session_state.pdf_page_texts[name] = {p: t for p, t in pages}
        elif lower.endswith(".docx"):
            txt = read_docx(data)
            if not txt.strip():
                st.warning(f"Empty/unsupported DOCX: {name}")
                continue
            target_list.append(Document(text=txt, metadata={"source": name}))
            st.session_state.file_bytes_map[name] = data
        elif lower.endswith(".txt"):
            txt = read_txt(data)
            if not txt.strip():
                st.warning(f"Empty/unsupported TXT: {name}")
                continue
            target_list.append(Document(text=txt, metadata={"source": name}))
            st.session_state.file_bytes_map[name] = data
        else:
            st.warning(f"Unsupported file type: {name}")
            continue

        seen_set.add(digest)
        if corpus_name == "policy":
            st.session_state.policy_files_set.add(name)
        else:
            st.session_state.email_files_set.add(name)

if policy_files:
    add_to_docs(policy_files, st.session_state.policy_docs, "policy")
if email_files:
    add_to_docs(email_files, st.session_state.email_docs, "email")

# ---------------------------
# Unique counters (no creep)
# ---------------------------
colA, colB = st.columns(2)
with colA:
    if st.session_state.policy_loaded_hashes:
        st.success(f"Policy docs loaded: {len(st.session_state.policy_loaded_hashes)}")
with colB:
    if st.session_state.email_loaded_hashes:
        st.success(f"Prior response docs loaded: {len(st.session_state.email_loaded_hashes)}")

# ---------------------------
# Embedding cache helpers (hashes use normalized full chunk text)
# ---------------------------
def _cache_get_many(conn, hashes: List[str]) -> Dict[str, List[float]]:
    if not hashes:
        return {}
    qmarks = ",".join(["?"] * len(hashes))
    rows = conn.execute(f"SELECT h, v FROM emb_cache WHERE h IN ({qmarks})", tuple(hashes)).fetchall()
    out: Dict[str, List[float]] = {}
    for h, v_json in rows:
        try:
            out[h] = json.loads(v_json)
        except Exception:
            pass
    return out

def _cache_set_many(conn, items: Dict[str, List[float]]):
    if not items:
        return
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO emb_cache (h, v) VALUES (?, ?)",
            [(h, json.dumps(vec)) for h, vec in items.items()],
        )

def _normalize_for_hash(s: str) -> str:
    # minimal normalization so identical text yields identical hashes
    return re.sub(r"\s+", " ", (s or "").strip())

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ---------------------------
# Build indexes — pre-embedded + cached (FAST) with docstore-first write
# ---------------------------
def build_index_incremental_preembedded_cached(
    docs: List[Document],
    label: str,
    page_batch_size: int,
    api_batch_size: int,
    max_tokens_unused: int,   # kept for signature compatibility; ignored
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from documents:
      - Page-by-page nodes via SentenceSplitter (chunking-only; no truncation)
      - Hash full normalized chunk text for caching
      - Cache hits/misses + optional per-batch debug timings
      - Store nodes in DOCSTORE first, then vector store (prevents KeyError)
    """
    # Prepare explicit docstore + vector store
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
    if not docs:
        return VectorStoreIndex([], storage_context=storage_context)

    def _dummy_text() -> str:
        return ("lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50).strip()

    parser = Settings.node_parser
    total_pages = len(docs)
    db_path = st.session_state.embed_cache_path

    DBG_PING_BATCHES = 3
    DBG_PING_BATCH_SIZE = 16

    with st.status(f"Embedding {label} ({total_pages} pages)…", state="running", expanded=True) as status:
        outer_pb = st.progress(0.0)
        t0_all = time.time()
        total_nodes = total_hits = total_miss = 0

        if st.session_state.get("embed_debug", False):
            status.write(f"🔎 Debugger: baseline ping on {DBG_PING_BATCHES}×{DBG_PING_BATCH_SIZE} dummy items…")
            dummy = _dummy_text()
            ping_times = []
            t_ping_all = time.time()
            for b in range(DBG_PING_BATCHES):
                batch_payload = [dummy] * DBG_PING_BATCH_SIZE
                t0 = time.time()
                _ = Settings.embed_model.get_text_embedding_batch(batch_payload)
                dt = time.time() - t0
                ping_times.append(dt)
                status.write(f"• Baseline batch {b+1}/{DBG_PING_BATCHES}: {dt:.2f}s (~{dt/DBG_PING_BATCH_SIZE:.2f}s/item)")
            dt_ping_all = time.time() - t_ping_all
            status.write(f"✅ Baseline avg/batch {sum(ping_times)/len(ping_times):.2f}s, avg/item {dt_ping_all/(DBG_PING_BATCHES*DBG_PING_BATCH_SIZE):.2f}s")

        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            for i in range(0, total_pages, page_batch_size):
                batch_docs = docs[i:i+page_batch_size]
                nodes: List[TextNode] = parser.get_nodes_from_documents(batch_docs)

                texts, hashes = [], []
                for n in nodes:
                    raw = n.get_content(metadata_mode="none")
                    norm = _normalize_for_hash(raw)
                    texts.append(norm)
                    hashes.append(_sha(norm))

                cache_map = _cache_get_many(conn, hashes)
                hits = sum(1 for h in hashes if h in cache_map)
                miss_idx = [idx for idx, h in enumerate(hashes) if h not in cache_map]
                miss_texts = [texts[idx] for idx in miss_idx]

                new_items: Dict[str, List[float]] = {}
                if miss_texts:
                    t0 = time.time()
                    embed_items = 0
                    for j in range(0, len(miss_texts), api_batch_size):
                        sub = miss_texts[j:j+api_batch_size]
                        t_sub = time.time()
                        embs = Settings.embed_model.get_text_embedding_batch(sub)
                        sub_dt = time.time() - t_sub
                        if st.session_state.get("embed_debug", False):
                            status.write(
                                f"• Miss-embed sub-batch {j//api_batch_size+1}/"
                                f"{(len(miss_texts)+api_batch_size-1)//api_batch_size}: "
                                f"{sub_dt:.2f}s (~{sub_dt/max(1,len(sub)):.2f}s/item) size={len(sub)}"
                            )
                        for k, e in enumerate(embs):
                            new_items[hashes[miss_idx[j+k]]] = e
                        embed_items += len(sub)
                    _cache_set_many(conn, new_items)
                    dt = time.time() - t0
                    status.write(f"• Embedded {embed_items} new nodes in {dt:.2f}s (~{dt/max(1,embed_items):.2f}s/item)")
                else:
                    status.write("• All nodes from this batch served from cache")

                # attach embeddings
                for n, h in zip(nodes, hashes):
                    vec = cache_map.get(h) or new_items.get(h)
                    n.embedding = vec

                # ✅ write to DOCSTORE first, then vector store (prevents KeyError on retrieve)
                storage_context.docstore.add_documents(nodes)
                vector_store.add(nodes)

                total_nodes += len(nodes)
                total_hits  += hits
                total_miss  += len(miss_texts)

                pages_done = min(total_pages, i + len(batch_docs))
                outer_pb.progress(min(1.0, pages_done / total_pages))
                status.write(
                    f"Progress: pages {pages_done}/{total_pages} | nodes: {total_nodes} | "
                    f"cache hits: {total_hits} | misses: {total_miss}"
                )
        finally:
            conn.close()

        dt_all = time.time() - t0_all
        status.update(
            label=f"{label} embedded — nodes: {total_nodes}, hits: {total_hits}, misses: {total_miss}, time: {dt_all:.1f}s",
            state="complete",
            expanded=False
        )

    return VectorStoreIndex([], storage_context=storage_context)

# ---------------------------
# Helper: ensure indexes exist (lazy build)
# ---------------------------
def ensure_indexes_built() -> bool:
    """Build indexes if missing and docs are loaded. Return True if at least one index exists."""
    need_policy = st.session_state.policy_index is None and bool(st.session_state.policy_docs)
    need_email  = st.session_state.email_index  is None and bool(st.session_state.email_docs)

    if need_policy or need_email:
        with st.spinner("Building indexes on demand…"):
            configure_settings()
            if need_policy:
                st.session_state.policy_index = build_index_incremental_preembedded_cached(
                    st.session_state.policy_docs,
                    label="Policy corpus",
                    page_batch_size=embed_page_batch,
                    api_batch_size=embed_api_batch,
                    max_tokens_unused=0,  # ignored
                )
            if need_email:
                st.session_state.email_index = build_index_incremental_preembedded_cached(
                    st.session_state.email_docs,
                    label="Prior responses corpus",
                    page_batch_size=embed_page_batch,
                    api_batch_size=embed_api_batch,
                    max_tokens_unused=0,  # ignored
                )

    st.session_state.vector_ready = bool(st.session_state.policy_index or st.session_state.email_index)
    return st.session_state.vector_ready

# ---------------------------
# Build / Rebuild Indexes button
# ---------------------------
if st.button("🔧 Build / Rebuild Indexes"):
    if not st.session_state.policy_docs and not st.session_state.email_docs:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Preparing to build indexes…"):
            configure_settings()

        if st.session_state.policy_docs:
            st.session_state.policy_index = build_index_incremental_preembedded_cached(
                st.session_state.policy_docs,
                label="Policy corpus",
                page_batch_size=embed_page_batch,
                api_batch_size=embed_api_batch,
                max_tokens_unused=0,  # ignored
            )
        else:
            st.session_state.policy_index = None

        if st.session_state.email_docs:
            st.session_state.email_index = build_index_incremental_preembedded_cached(
                st.session_state.email_docs,
                label="Prior responses corpus",
                page_batch_size=embed_page_batch,
                api_batch_size=embed_api_batch,
                max_tokens_unused=0,  # ignored
            )
        else:
            st.session_state.email_index = None

        st.session_state.vector_ready = bool(st.session_state.policy_index or st.session_state.email_index)
        if st.session_state.vector_ready:
            st.success("Indexes built.")
        else:
            st.warning("No indexes available (no documents loaded).")

# ---------------------------
# Prompts
# ---------------------------
DRAFT_PROMPT = """You are a compliance assistant drafting replies to financial regulators.
ONLY use the EVIDENCE provided. If evidence is insufficient or conflicting, state this clearly.

Return a concise regulator-appropriate reply (1–3 short paragraphs) and include inline bracket citations
like [Doc: <filename>, Node <n>] for each supported statement.

Question:
{question}

Evidence:
{evidence}

Constraints:
- No fabrication beyond evidence.
- Formal, neutral tone suitable for regulators.
"""

JUDGE_PROMPT = """You are validating a regulator-facing draft.

Rules:
- Return ONLY a single JSON object. No commentary, no markdown, no code fences.
- JSON must be minified on one line.
- Use these exact keys: source_type, confidence, keep_citations.
- source_type ∈ {{"Policy","PriorEmail","Blended","Insufficient"}}.
- confidence is a float 0.0–1.0 (use one decimal place).
- keep_citations is an array of strings; each string must exactly match a citation tag from Evidence.

Question: {question}

Draft:
{draft}

Evidence:
{evidence}

Return:
{{"source_type":"Policy","confidence":0.8,"keep_citations":["[Doc: sample.pdf, Node 0]","[Doc: prior_email.txt, Node 2]"]}}
"""

# ---------------------------
# Robust JSON extraction + fallback helpers
# ---------------------------
def extract_first_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    s_stripped = s.strip()
    if s_stripped.startswith("{") and s_stripped.endswith("}"):
        try:
            return json.loads(s_stripped)
        except Exception:
            pass
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    block = m.group(0)
    for end in range(len(block), 1, -1):
        try:
            return json.loads(block[:end])
        except Exception:
            continue
    return None

def infer_source_type_from_tags(tags: List[str], tag_to_file: Dict[str, str]) -> str:
    if not tags:
        return "Insufficient"
    policy = sum(1 for t in tags if tag_to_file.get(t) in st.session_state.policy_files_set)
    email  = sum(1 for t in tags if tag_to_file.get(t) in st.session_state.email_files_set)
    if policy and email:
        return "Blended"
    if policy:
        return "Policy"
    if email:
        return "PriorEmail"
    return "Insufficient"

def confidence_from_scores(tag_to_score: Dict[str, float], kept_tags: List[str]) -> float:
    use_tags = kept_tags or list(tag_to_score.keys())
    scores = [tag_to_score.get(t, 0.0) for t in use_tags]
    if not scores:
        return 0.0
    clipped = [min(1.0, max(0.0, float(s))) for s in scores]
    avg = sum(clipped) / len(clipped)
    return round(max(0.05, min(0.99, avg)), 3)

# ---------------------------
# Resilient retrieval with direct-query fallback + auto-repair
# ---------------------------
def _repair_index(kind: str) -> Optional[VectorStoreIndex]:
    """Rebuild a single index (policy|email) from loaded docs, using cached embeddings."""
    if kind == "policy" and st.session_state.policy_docs:
        st.info("Repairing policy index from cache…")
        idx = build_index_incremental_preembedded_cached(
            st.session_state.policy_docs,
            label="Policy corpus (repair)",
            page_batch_size=embed_page_batch,
            api_batch_size=embed_api_batch,
            max_tokens_unused=0,
        )
        st.session_state.policy_index = idx
        return idx
    if kind == "email" and st.session_state.email_docs:
        st.info("Repairing prior responses index from cache…")
        idx = build_index_incremental_preembedded_cached(
            st.session_state.email_docs,
            label="Prior responses corpus (repair)",
            page_batch_size=embed_page_batch,
            api_batch_size=embed_api_batch,
            max_tokens_unused=0,
        )
        st.session_state.email_index = idx
        return idx
    return None

def _direct_query_nodes(index: VectorStoreIndex, q: str, k: int) -> List[NodeWithScore]:
    """Query the vector store directly and safely build NodeWithScore list, skipping missing docstore IDs."""
    try:
        q_emb = Settings.embed_model.get_text_embedding(q)
        vs = getattr(index, "_vector_store", None) or getattr(index, "vector_store", None)
        ds = getattr(index, "_docstore", None) or index.storage_context.docstore
        if vs is None or ds is None:
            return []
        res = vs.query(VectorStoreQuery(query_embedding=q_emb, similarity_top_k=k))
        nodes_ws: List[NodeWithScore] = []
        if not res or not getattr(res, "ids", None):
            return nodes_ws
        sims = res.similarities or [0.0] * len(res.ids)
        for nid, score in zip(res.ids, sims):
            node = ds.get_document(nid, raise_error=False)
            if node is not None:
                nodes_ws.append(NodeWithScore(node=node, score=score))
        return nodes_ws
    except Exception as e:
        st.warning(f"Direct vector query failed: {e}")
        return []

def _safe_retrieve_with_fallback(index: Optional[VectorStoreIndex], kind: str, q: str, k: int) -> List[NodeWithScore]:
    """Try standard retriever; on KeyError use direct-query fallback; if still empty, repair index and retry."""
    if not index:
        return []
    try:
        r = VectorIndexRetriever(index=index, similarity_top_k=k)
        return r.retrieve(q)
    except KeyError:
        nodes = _direct_query_nodes(index, q, k)
        if nodes:
            return nodes
        idx2 = _repair_index(kind)
        if not idx2:
            st.warning(f"Unable to repair {kind} index (no docs).")
            return []
        try:
            r2 = VectorIndexRetriever(index=idx2, similarity_top_k=k)
            return r2.retrieve(q)
        except KeyError:
            return _direct_query_nodes(idx2, q, k)

def retrieve_across(q: str, k: int) -> List[NodeWithScore]:
    out: List[NodeWithScore] = []
    out += _safe_retrieve_with_fallback(st.session_state.policy_index, "policy", q, k)
    out += _safe_retrieve_with_fallback(st.session_state.email_index,  "email",  q, k)
    return out

def nodes_to_evidence(nodes_ws: List[NodeWithScore]) -> Tuple[str, Dict[str, str], Dict[str, float], Dict[str, Optional[int]]]:
    evidence_lines: List[str] = []
    tag_to_file: Dict[str, str] = {}
    tag_to_score: Dict[str, float] = {}
    tag_to_page: Dict[str, Optional[int]] = {}

    for i, nws in enumerate(nodes_ws):
        node = nws.node
        src = (node.metadata or {}).get("source", "unknown")
        page = (node.metadata or {}).get("page")
        if page is not None:
            tag = f"[Doc: {src}, Page {page}, Node {i}]"
        else:
            tag = f"[Doc: {src}, Node {i}]"
        text = re.sub(r"\s+", " ", node.get_text()).strip()
        evidence_lines.append(f"{tag} {text}")
        tag_to_file[tag] = src
        tag_to_page[tag] = page
        try:
            tag_to_score[tag] = float(nws.score or 0.0)
        except Exception:
            tag_to_score[tag] = 0.0

    return "\n".join(evidence_lines), tag_to_file, tag_to_score, tag_to_page

# ---------------------------
# Questions UI
# ---------------------------
st.markdown("### Ask Regulator Questions")

qs_text = st.text_area(
    "Paste questions (one per line):",
    height=120,
    placeholder="e.g., Describe our data retention policy for customer PII.\n"
                "e.g., Summarize the remediation timeline for the 2023 key management audit finding."
)

st.markdown("Or build them interactively:")
q_table = st.data_editor(
    st.session_state.get("q_table", [{"question": ""}]),
    num_rows="dynamic",
    key="q_editor",
    column_config={"question": st.column_config.TextColumn("Question", required=False, width="large")},
)

# ---------------------------
# Generate Responses (lazy-build if needed)
# ---------------------------
if st.button("🧠 Generate Responses"):
    ready = ensure_indexes_built()
    if not ready:
        st.error("No indexes available. Upload documents and build indexes first (or re-upload so I can rebuild from cache).")
        st.stop()

    configure_settings()
    llm = Settings.llm

    typed = [q.strip() for q in qs_text.split("\n") if q.strip()]
    edited = [row.get("question", "").strip() for row in q_table if row.get("question", "").strip()]
    questions, seen = [], set()
    for q in (typed + edited):
        if q not in seen:
            questions.append(q)
            seen.add(q)

    if not questions:
        st.warning("Please add at least one question.")
    else:
        results = []
        with st.status("Generating responses…", state="running", expanded=True) as status:
            total = len(questions)
            for idx, q in enumerate(questions, start=1):
                status.write(f"Processing **{q}** ({idx}/{total})")

                nodes_ws = retrieve_across(q, top_k)
                if nodes_ws:
                    ev_str, tag2file, tag2score, tag2page = nodes_to_evidence(nodes_ws)
                else:
                    ev_str, tag2file, tag2score, tag2page = "(no evidence found)", {}, {}, {}

                draft = llm.complete(DRAFT_PROMPT.format(question=q, evidence=ev_str)).text

                judge_raw = llm.complete(JUDGE_PROMPT.format(question=q, draft=draft, evidence=ev_str)).text
                parsed = extract_first_json_obj(judge_raw)

                if parsed:
                    source_type = parsed.get("source_type", "Insufficient")
                    keep = [str(x) for x in parsed.get("keep_citations", [])]
                    try:
                        confidence = float(parsed.get("confidence", 0.0))
                    except Exception:
                        confidence = 0.0
                else:
                    sorted_tags = sorted(tag2score.items(), key=lambda kv: kv[1], reverse=True)
                    keep = [t for (t, _) in sorted_tags[:3]]
                    source_type = infer_source_type_from_tags(keep, tag2file)
                    confidence = confidence_from_scores(tag2score, keep)

                ev_snips, kept_docs = [], []
                if ev_str != "(no evidence found)":
                    lines = ev_str.splitlines()
                    tag_set = set(keep) if keep else set()
                    if tag_set:
                        for line in lines:
                            tag = line.split("]")[0] + "]"
                            if tag in tag_set:
                                ev_snips.append(line)
                                kept_docs.append(tag2file.get(tag, "unknown"))
                    else:
                        for line in lines[:4]:
                            ev_snips.append(line)
                            m = re.search(r"\[Doc: (.*?)(?:, Page \d+)?, Node \d+\]", line)
                            if m:
                                kept_docs.append(m.group(1))
                kept_docs = list(dict.fromkeys(kept_docs))[:5]

                results.append({
                    "question": q,
                    "draft": draft,
                    "source_type": source_type,
                    "confidence": round(confidence, 3),
                    "kept_docs": kept_docs,
                    "ev_snips": ev_snips[:6],
                    "judge_raw": judge_raw,
                    "tag2file": tag2file,
                    "tag2page": tag2page,
                })

            status.update(label="Responses generated", state="complete", expanded=False)

        show_debug = st.session_state.get("show_debug", False)

        for r in results:
            with st.expander(f"❓ {r['question']}", expanded=False):
                left, right = st.columns([2, 1])
                with left:
                    st.subheader("Suggested Response")
                    st.write(r["draft"])
                with right:
                    st.subheader("Assessment")
                    st.write(f"**Source Type:** {r['source_type']}")
                    st.write(f"**Confidence:** {r['confidence']}")
                    st.write("**Source Docs:**")
                    if r["kept_docs"]:
                        for d in r["kept_docs"]:
                            st.write(f"- {d}")
                    else:
                        st.write("- (none)")

                st.subheader("Evidence Snippets")
                if r["ev_snips"]:
                    for sn in r["ev_snips"]:
                        st.code(sn, language="text")
                        tag = sn.split("]")[0] + "]"
                        fname = r["tag2file"].get(tag)
                        page = r["tag2page"].get(tag)
                        query_text = sn[len(tag):].strip()
                        if len(query_text) > 500:
                            query_text = query_text[:500]

                        with st.expander("Show evidence"):
                            if fname and page and fname in st.session_state.file_bytes_map:
                                file_bytes = st.session_state.file_bytes_map[fname]
                                img = render_pdf_page_with_highlight(file_bytes, page, query_text)
                                if img is not None:
                                    st.image(img, caption=f"{fname} — Page {page}", use_column_width=True)
                                else:
                                    page_text = ""
                                    if fname in st.session_state.pdf_page_texts:
                                        page_text = st.session_state.pdf_page_texts[fname].get(page, "")
                                    if page_text:
                                        safe_html = highlight_html(page_text, query_text, window=600)
                                        st.markdown(f"**{fname} — Page {page} (text preview)**", help="Install pymupdf+Pillow for image previews.")
                                        st.markdown(safe_html, unsafe_allow_html=True)
                                    else:
                                        st.info("Preview unavailable for this PDF page.")
                            else:
                                src_file = fname or "(unknown)"
                                data = st.session_state.file_bytes_map.get(src_file)
                                preview_text = ""
                                if data:
                                    lower = src_file.lower()
                                    try:
                                        if lower.endswith(".pdf"):
                                            if src_file in st.session_state.pdf_page_texts:
                                                preview_text = " ".join(st.session_state.pdf_page_texts[src_file].values())
                                        elif lower.endswith(".docx"):
                                            preview_text = read_docx(data)
                                        elif lower.endswith(".txt"):
                                            preview_text = read_txt(data)
                                    except Exception:
                                        preview_text = ""
                                if preview_text:
                                    safe_html = highlight_html(preview_text, query_text, window=600)
                                    st.markdown(f"**{src_file} (text preview)**")
                                    st.markdown(safe_html, unsafe_allow_html=True)
                                else:
                                    st.info("Preview unavailable for this evidence.")
                else:
                    st.write("_No supporting evidence found._")

                if show_debug:
                    st.subheader("Debug: Judge Raw Output")
                    st.code(r["judge_raw"][:4000], language="json")
