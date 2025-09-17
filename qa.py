# app.py — Regulatory Response Assistant (Streamlit + LlamaIndex <= 0.12.53)
# -------------------------------------------------------------------------
# Upload Company Policy docs and Prior Regulator Response emails (PDF/DOCX/TXT)
# Dynamically add questions (textarea or table)
# Retrieve evidence from both corpora, draft regulator-ready reply, judge & cite
#
# Features:
# - Strict JSON-only judge prompt (escaped braces) + robust JSON extractor
# - Fallback inference for source_type + confidence when judge JSON is messy
# - SHA-256 fingerprint dedupe for uploaded files + accurate counters
# - "Clear loaded docs" reset button
# - Evidence previewer (PDF page image highlighting if PyMuPDF + Pillow; otherwise text preview with <mark>)
# - Generate Responses uses a global loader with progress
# - FAST indexing:
#     * One Document per PDF page (keeps page metadata)
#     * Page-by-page pre-embedding with batching
#     * SQLite embedding cache keyed by hash of token-truncated text
#     * Configurable token truncation (to shrink payload) + API batch sizes
# - Thread-safe SQLite usage (connection created/closed inside build function)

import os
import io
import re
import json
import time
import sqlite3
import hashlib
from html import escape  # for safe HTML in text preview
from typing import List, Tuple, Dict, Optional

import streamlit as st
import docx2txt
import tiktoken
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
    embed_max_tokens = st.slider("Embedding max tokens per node", 256, 4096, 768, 64)

    show_debug = st.toggle("Show Judge Raw Output (debug)", value=False)

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
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

configure_settings()


# ---------------------------
# File readers & PDF helpers
# ---------------------------
def read_pdf_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Return list of (1-based page_number, text) for a PDF."""
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
    """
    Text-based fallback preview: returns a small HTML slice around the first occurrence of needle,
    with <mark>...</mark> highlighting. If needle not found, returns a truncated preview.
    """
    text_norm = re.sub(r"\s+", " ", text or "")
    n = (needle or "").strip()

    def esc(s: str) -> str:
        # quote=False keeps quotes as-is; safe inside <pre> block
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
    """Render a PDF page image and draw highlight rectangles for query_text (best-effort)."""
    if not (fitz and Image and ImageDraw):
        return None  # visual backend not available
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
            for r in rects[:5]:  # cap highlights
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

# sets to identify which filenames belong to which corpus (used for fallback)
if "policy_files_set" not in st.session_state:
    st.session_state.policy_files_set = set()
if "email_files_set" not in st.session_state:
    st.session_state.email_files_set = set()

# sets of file hashes to dedupe uploads across reruns
if "policy_loaded_hashes" not in st.session_state:
    st.session_state.policy_loaded_hashes = set()
if "email_loaded_hashes" not in st.session_state:
    st.session_state.email_loaded_hashes = set()

# map of filename -> raw bytes (for previews)
if "file_bytes_map" not in st.session_state:
    st.session_state.file_bytes_map: Dict[str, bytes] = {}

# PDF page text cache: filename -> {page_num: text}
if "pdf_page_texts" not in st.session_state:
    st.session_state.pdf_page_texts: Dict[str, Dict[int, str]] = {}

# embedding cache path (thread-safe handling done during build)
if "embed_cache_path" not in st.session_state:
    st.session_state.embed_cache_path = "embed_cache.sqlite3"

def _init_embed_cache(path: str):
    """Create DB & table if needed, then close immediately (no long-lived connection)."""
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emb_cache (
                  h TEXT PRIMARY KEY,
                  v TEXT NOT NULL
                )
            """)
            # Optional pragmas for speed
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
    finally:
        conn.close()

# Ensure cache DB exists
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
            continue  # already loaded this exact file in this session

        lower = name.lower()
        if lower.endswith(".pdf"):
            pages = read_pdf_pages(data)
            if not any(txt.strip() for _, txt in pages):
                st.warning(f"Empty/unsupported PDF: {name}")
                continue
            # One Document per PDF page, so we can keep page metadata
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
# Embedding cache helpers
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
    return re.sub(r"\s+", " ", (s or "").strip())

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])


# ---------------------------
# Build indexes — pre-embedded + cached (FAST) with thread-safe SQLite
# ---------------------------
def build_index_incremental_preembedded_cached(
    docs: List[Document],
    label: str,
    page_batch_size: int,
    api_batch_size: int,
    max_tokens: int,
) -> VectorStoreIndex:
    """
    1) Pages -> nodes (keeps page metadata)
    2) For each node: normalize + token-truncate + hash
    3) Pull cached embeddings by hash; embed only misses (batched)
    4) Attach embeddings to nodes; add to SimpleVectorStore
    SQLite connection is opened/closed inside this function (thread-safe).
    """
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context)
    if not docs:
        return index

    parser = Settings.node_parser
    total_pages = len(docs)
    db_path = st.session_state.embed_cache_path

    with st.status(f"Embedding {label} ({total_pages} pages)…", state="running", expanded=True) as status:
        outer_pb = st.progress(0.0)
        t0_all = time.time()
        total_nodes = 0
        total_hits = 0
        total_miss = 0

        # Open a fresh connection for this run/thread
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            for i in range(0, total_pages, page_batch_size):
                batch_docs = docs[i:i+page_batch_size]

                # 1) pages -> nodes
                nodes: List[TextNode] = parser.get_nodes_from_documents(batch_docs)

                # 2) normalize + truncate + hash
                texts: List[str] = []
                hashes: List[str] = []
                for n in nodes:
                    raw = n.get_content(metadata_mode="none")
                    norm = _normalize_for_hash(raw)
                    trunc = _truncate_to_tokens(norm, max_tokens=max_tokens)
                    texts.append(trunc)
                    hashes.append(_sha(trunc))

                # 3) cache lookup
                cache_map = _cache_get_many(conn, hashes)
                hits = sum(1 for h in hashes if h in cache_map)
                miss_idx = [idx for idx, h in enumerate(hashes) if h not in cache_map]
                miss_texts = [texts[idx] for idx in miss_idx]

                # embed misses in API sub-batches
                new_items: Dict[str, List[float]] = {}
                if miss_texts:
                    t0 = time.time()
                    for j in range(0, len(miss_texts), api_batch_size):
                        sub = miss_texts[j:j+api_batch_size]
                        embs = Settings.embed_model.get_text_embedding_batch(sub)
                        for k, e in enumerate(embs):
                            new_items[hashes[miss_idx[j+k]]] = e
                    _cache_set_many(conn, new_items)
                    dt = time.time() - t0
                    status.write(f"• Embedded {len(miss_texts)} new nodes in {dt:.1f}s ({dt/max(1,len(miss_texts)):.2f}s/item)")
                else:
                    status.write("• All nodes from this batch served from cache")

                # 4) attach embeddings and add to vector store
                for n, h in zip(nodes, hashes):
                    vec = cache_map.get(h) or new_items.get(h)
                    n.embedding = vec  # prevents re-embedding
                vector_store.add(nodes)

                total_nodes += len(nodes)
                total_hits  += hits
                total_miss  += len(miss_texts)

                pages_done = min(total_pages, i + len(batch_docs))
                outer_pb.progress(min(1.0, pages_done / total_pages))
                status.write(f"Progress: pages {pages_done}/{total_pages} | nodes: {total_nodes} | cache hits: {total_hits} | misses: {total_miss}")
        finally:
            conn.close()

        dt_all = time.time() - t0_all
        status.update(
            label=f"{label} embedded — nodes: {total_nodes}, hits: {total_hits}, misses: {total_miss}, time: {dt_all:.1f}s",
            state="complete",
            expanded=False
        )

    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


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
                max_tokens=embed_max_tokens,
            )
        else:
            st.session_state.policy_index = None

        if st.session_state.email_docs:
            st.session_state.email_index = build_index_incremental_preembedded_cached(
                st.session_state.email_docs,
                label="Prior responses corpus",
                page_batch_size=embed_page_batch,
                api_batch_size=embed_api_batch,
                max_tokens=embed_max_tokens,
            )
        else:
            st.session_state.email_index = None

        st.session_state.vector_ready = True
        st.success("Indexes built.")


# ---------------------------
# Prompts (strict JSON judge; braces escaped)
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
    """Extract the first valid top-level {...} JSON object from a string."""
    if not s:
        return None
    s_stripped = s.strip()
    # Common case: pure JSON
    if s_stripped.startswith("{") and s_stripped.endswith("}"):
        try:
            return json.loads(s_stripped)
        except Exception:
            pass
    # Remove code fences if present
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # Greedy search for first {...}
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    block = m.group(0)
    # Try progressively trimming to valid JSON
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
# Retrieval helpers (keep NodeWithScore for fallback)
# ---------------------------
def retrieve_across(q: str, k: int) -> List[NodeWithScore]:
    out: List[NodeWithScore] = []
    if st.session_state.policy_index:
        r = VectorIndexRetriever(index=st.session_state.policy_index, similarity_top_k=k)
        out += r.retrieve(q)
    if st.session_state.email_index:
        r = VectorIndexRetriever(index=st.session_state.email_index, similarity_top_k=k)
        out += r.retrieve(q)
    return out

def nodes_to_evidence(nodes_ws: List[NodeWithScore]) -> Tuple[str, Dict[str, str], Dict[str, float], Dict[str, Optional[int]]]:
    """
    Returns:
      evidence_text: str
      tag_to_file: {tag -> filename}
      tag_to_score: {tag -> similarity_score}
      tag_to_page: {tag -> page_number or None}
    """
    evidence_lines: List[str] = []
    tag_to_file: Dict[str, str] = {}
    tag_to_score: Dict[str, float] = {}
    tag_to_page: Dict[str, Optional[int]] = {}

    for i, nws in enumerate(nodes_ws):
        node = nws.node
        src = (node.metadata or {}).get("source", "unknown")
        page = (node.metadata or {}).get("page")  # int or None
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
# Questions UI (text + dynamic editor)
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
# Generate Responses (global loader + progress)
# ---------------------------
if st.button("🧠 Generate Responses"):
    if not st.session_state.vector_ready:
        st.error("Please build indexes first.")
    else:
        configure_settings()
        llm = Settings.llm

        # Collate deduped questions from both inputs
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

                    # Retrieve (with scores + pages)
                    nodes_ws = retrieve_across(q, top_k)
                    if nodes_ws:
                        ev_str, tag2file, tag2score, tag2page = nodes_to_evidence(nodes_ws)
                    else:
                        ev_str, tag2file, tag2score, tag2page = "(no evidence found)", {}, {}, {}

                    # Draft
                    draft = llm.complete(DRAFT_PROMPT.format(question=q, evidence=ev_str)).text

                    # Judge (robust JSON parse)
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
                        # Fallback: pick top-3 by similarity, infer source_type & confidence
                        sorted_tags = sorted(tag2score.items(), key=lambda kv: kv[1], reverse=True)
                        keep = [t for (t, _) in sorted_tags[:3]]
                        source_type = infer_source_type_from_tags(keep, tag2file)
                        confidence = confidence_from_scores(tag2score, keep)

                    # Derive kept docs + snippets
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

            # ---- Render AFTER work completes ----
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

                            # Inline previewer per snippet
                            tag = sn.split("]")[0] + "]"  # "[Doc: ..., Page X, Node i]"
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
