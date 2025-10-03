# app.py — Regulatory Response Assistant (Streamlit + LlamaIndex <= 0.12.53)
# -------------------------------------------------------------------------
# Upload Policy docs & Prior Regulator Responses (PDF/DOCX/TXT/XLSX/CSV)
# Ask regulator questions → evidence-backed drafts + citations
# Excel/CSV: ONE ROW = ONE NODE/EMBEDDING. PDFs/DOCX/TXT: chunked.
# JSON-safe embed cache. Persistent indexes & files. Visual PDF & row previews.
# Hybrid retrieval: dense + BM25 + fuzzy (general, domain-agnostic).
# Confidence scoring: multi-signal (similarity, coverage, diversity, alignment via rapidfuzz, corpus weight),
# blended with judge confidence.
# Includes: cache rebuild on persisted index load + node/chunk inspector.

import os
import io
import re
import json
import time
import math
import sqlite3
import hashlib
from html import escape
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import docx2txt
from PyPDF2 import PdfReader
from rapidfuzz import fuzz  # for alignment + fuzzy

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
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import VectorStoreQuery

# ---------------------------
# Persistent storage roots
# ---------------------------
PERSIST_ROOT = ".storage"
POLICY_DIR = os.path.join(PERSIST_ROOT, "policy")
EMAIL_DIR  = os.path.join(PERSIST_ROOT, "email")
FILES_ROOT = os.path.join(PERSIST_ROOT, "files")
os.makedirs(POLICY_DIR, exist_ok=True)
os.makedirs(EMAIL_DIR,  exist_ok=True)
os.makedirs(FILES_ROOT, exist_ok=True)

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
    embed_page_batch = st.slider("Embed page batch (pages/rows → nodes)", 2, 64, 16, 2)
    embed_api_batch  = st.slider("Embed API batch (texts per request)", 4, 128, 32, 4)

    # kept only for UI continuity; unused in chunking-only mode
    st.slider("Max tokens per node (ignored here)", 256, 4096, 768, 64, key="embed_max_tokens_ignored")

    st.toggle("Enable embedding debugger", value=False, key="embed_debug")
    st.toggle("Show Judge Raw Output (debug)", value=False, key="show_debug")

    st.markdown("---")
    if fitz and Image and ImageDraw:
        st.info("PDF visual preview: **Enabled** (pymupdf + Pillow detected)")
    else:
        st.info("PDF visual preview: **Text-only fallback** (install `pymupdf` and `Pillow` to enable images)")

    # Clear/reset button
    if st.button("🧹 Clear loaded docs (reset)"):
        st.session_state.clear()
        # preserve persisted storage on disk; only clear in-memory session state
        st.success("Cleared in-memory state. Persisted storage on disk remains.")

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
    # Chunking controls node size; no truncation used for non-row docs.
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

configure_settings()

# ---------------------------
# File readers & helpers
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

def read_excel_rows(file_bytes: bytes) -> List[Tuple[str, int, Dict[str, str]]]:
    """
    Return list of (sheet_name, row_index_1based, ordered_row_dict).
    Reads ALL sheets at once; preserves column order; trims whitespace;
    includes a row if ANY cell is non-empty after trimming.
    """
    out: List[Tuple[str, int, Dict[str, str]]] = []
    bio = io.BytesIO(file_bytes)
    all_sheets = pd.read_excel(
        bio,
        sheet_name=None,
        dtype=str,
        engine=None,
        keep_default_na=False
    )
    for sheet_name, df in (all_sheets or {}).items():
        if df is None or df.empty:
            continue
        df = df.fillna("")
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        cols = [str(c) for c in df.columns]
        mask_non_empty = df.apply(
            lambda r: any((isinstance(v, str) and v.strip() != "") or (not isinstance(v, str) and str(v).strip() != "")
                          for v in r.values),
            axis=1
        )
        if not mask_non_empty.any():
            continue
        df = df.loc[mask_non_empty]
        for ridx, row in df.iterrows():
            row_dict: Dict[str, str] = {}
            for c in cols:
                v = str(row[c]).strip()
                if v != "":
                    row_dict[c] = v
            if not row_dict:
                continue
            out.append((str(sheet_name), int(ridx) + 1, row_dict))
    return out

def read_csv_rows(file_bytes: bytes) -> List[Tuple[str, int, Dict[str, str]]]:
    """
    Treat whole CSV as a single sheet 'CSV'.
    Keeps original column order; includes only non-empty cells.
    """
    out: List[Tuple[str, int, Dict[str, str]] ] = []
    bio = io.BytesIO(file_bytes)
    df = pd.read_csv(bio, dtype=str)
    if df.empty:
        return out
    df = df.fillna("")
    cols = [str(c) for c in df.columns]
    for ridx, row in df.iterrows():
        row_dict: Dict[str, str] = {}
        for c in cols:
            v = str(row[c]).strip()
            if v != "":
                row_dict[c] = v
        if not row_dict:
            continue
        out.append(("CSV", int(ridx) + 1, row_dict))
    return out

def row_dict_to_text(row: Dict[str, str]) -> str:
    """Serialize an ordered row dict to text with no assumptions about headers."""
    if not row:
        return ""
    return "\n".join(f"{k}: {v}" for k, v in row.items())

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
# Tokenization / BM25 / fuzzy helpers (general-purpose)
# ---------------------------
STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","at","by","with",
    "is","are","was","were","be","been","being","as","that","this","these",
    "those","from","it","its","we","you","your","our","they","their","them",
    "about","into","over","under","per","via","within","without","not","no",
    "but","if","then","than","so","such","any","all","each","every"
}
_word_re = re.compile(r"\b\w+\b", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t for t in (_word_re.findall(text.lower()) if text else []) if t not in STOPWORDS]

def bm25_score_row(query_tokens: List[str], tf: Counter, dl: int, df_map: Dict[str,int], N: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_tokens or not tf or dl <= 0 or N <= 0 or avgdl <= 0:
        return 0.0
    score = 0.0
    for term in query_tokens:
        df = df_map.get(term, 0)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        f = tf.get(term, 0)
        if f <= 0:
            continue
        denom = f + k1 * (1 - b + b * (dl / avgdl))
        score += idf * (f * (k1 + 1) / denom)
    # squash to ~[0,1]
    return 1 - math.exp(-score / 5.0)

def fuzzy_partial(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return max(0.0, min(1.0, fuzz.partial_ratio(a, b) / 100.0))

def norm_sim_dense(s: float) -> float:
    try:
        return max(0.0, min(1.0, (float(s) - 0.4) / (0.85 - 0.4)))
    except Exception:
        return 0.0

# ---------------------------
# Session state
# ---------------------------
def ensure_state_defaults():
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

    # mapping original filename -> saved disk path for previews
    if "saved_file_paths" not in st.session_state:
        st.session_state.saved_file_paths = {}
        try:
            for fn in os.listdir(FILES_ROOT):
                parts = fn.split("__", 1)
                if len(parts) == 2:
                    original = parts[1]
                    st.session_state.saved_file_paths[original] = os.path.join(FILES_ROOT, fn)
        except Exception:
            pass

    # Row caches (node_id -> dict{text, meta, tf, dl})
    if "policy_row_cache" not in st.session_state:
        st.session_state.policy_row_cache = {}
    if "email_row_cache" not in st.session_state:
        st.session_state.email_row_cache = {}

    # BM25 corpus stats per row-corpus
    if "policy_bm25" not in st.session_state:
        st.session_state.policy_bm25 = {"df": {}, "N": 0, "avgdl": 0.0}
    if "email_bm25" not in st.session_state:
        st.session_state.email_bm25 = {"df": {}, "N": 0, "avgdl": 0.0}

ensure_state_defaults()

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
    "Upload Company Policy documents (PDF/DOCX/TXT/XLSX/CSV)",
    type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
    accept_multiple_files=True
)
email_files = st.file_uploader(
    "Upload Prior Regulator Response emails (PDF/DOCX/TXT/XLSX/CSV)",
    type=["pdf", "docx", "txt", "xlsx", "xls", "csv"],
    accept_multiple_files=True
)

def _persist_uploaded_file(name: str, data: bytes):
    safe_name = f"{hashlib.sha1(name.encode()).hexdigest()}__{name}"
    save_path = os.path.join(FILES_ROOT, safe_name)
    try:
        with open(save_path, "wb") as f:
            f.write(data)
        st.session_state.saved_file_paths[name] = save_path
    except Exception:
        pass

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
            _persist_uploaded_file(name, data)

        elif lower.endswith(".docx"):
            txt = read_docx(data)
            if not txt.strip():
                st.warning(f"Empty/unsupported DOCX: {name}")
                continue
            target_list.append(Document(text=txt, metadata={"source": name}))
            st.session_state.file_bytes_map[name] = data
            _persist_uploaded_file(name, data)

        elif lower.endswith(".txt"):
            txt = read_txt(data)
            if not txt.strip():
                st.warning(f"Empty/unsupported TXT: {name}")
                continue
            target_list.append(Document(text=txt, metadata={"source": name}))
            st.session_state.file_bytes_map[name] = data
            _persist_uploaded_file(name, data)

        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            try:
                rows = read_excel_rows(data)
            except Exception as e:
                st.warning(f"Failed to read Excel {name}: {e}")
                continue
            if not rows:
                st.warning(f"No usable rows found in {name}")
                continue
            # DEBUG summary per-sheet
            sheet_counts = {}
            for sheet, rownum, row_dict in rows:
                sheet_counts.setdefault(sheet, 0)
                sheet_counts[sheet] += 1
            with st.expander(f"📊 Ingestion summary for {name}", expanded=False):
                st.json({"corpus": corpus_name, "total_rows_to_ingest": sum(sheet_counts.values()), "per_sheet": sheet_counts})

            for sheet, rownum, row_dict in rows:
                doc_text = row_dict_to_text(row_dict)
                row_key = hashlib.sha1(("||".join(f"{k}={v}" for k, v in row_dict.items())).encode()).hexdigest()[:12]
                meta = {
                    "source": name, "sheet": sheet, "row": rownum, "row_key": row_key,
                    "row_node": True, "headers": list(row_dict.keys())
                }
                target_list.append(Document(text=doc_text, metadata=meta))
            st.session_state.file_bytes_map[name] = data
            _persist_uploaded_file(name, data)

        elif lower.endswith(".csv"):
            try:
                rows = read_csv_rows(data)
            except Exception as e:
                st.warning(f"Failed to read CSV {name}: {e}")
                continue
            if not rows:
                st.warning(f"No usable rows found in {name}")
                continue

            # DEBUG summary for CSV
            with st.expander(f"📊 Ingestion summary for {name}", expanded=False):
                st.json({"corpus": corpus_name, "total_rows_to_ingest": len(rows), "per_sheet": {"CSV": len(rows)}})

            for sheet, rownum, row_dict in rows:
                doc_text = row_dict_to_text(row_dict)
                row_key = hashlib.sha1(("||".join(f"{k}={v}" for k, v in row_dict.items())).encode()).hexdigest()[:12]
                meta = {
                    "source": name, "sheet": sheet, "row": rownum, "row_key": row_key,
                    "row_node": True, "headers": list(row_dict.keys())
                }
                target_list.append(Document(text=doc_text, metadata=meta))
            st.session_state.file_bytes_map[name] = data
            _persist_uploaded_file(name, data)

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
# Storage status
# ---------------------------
if st.session_state.policy_index or st.session_state.email_index:
    st.success("✅ Loaded existing indexes from local storage (or current session). You can ask questions immediately.")
else:
    st.info("ℹ️ No indexes found in local storage yet. Upload documents and build indexes to get started.")

# ---------------------------
# Unique counters (no creep)
# ---------------------------
colA, colB = st.columns(2)
with colA:
    if st.session_state.policy_loaded_hashes:
        st.success(f"Policy docs loaded (this session): {len(st.session_state.policy_loaded_hashes)}")
with colB:
    if st.session_state.email_loaded_hashes:
        st.success(f"Prior response docs loaded (this session): {len(st.session_state.email_loaded_hashes)}")

# ---------------------------
# Embedding cache helpers (JSON-safe; ndarray -> list[float])
# ---------------------------
def _as_float_list(vec):
    """Return a plain Python list[float] from numpy arrays, lists, tuples, etc."""
    try:
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
    except Exception:
        pass
    if isinstance(vec, (list, tuple)):
        try:
            return [float(x) for x in vec]
        except Exception:
            return list(vec)
    try:
        return [float(x) for x in list(vec)]
    except Exception:
        return []

def _cache_get_many(conn, hashes: List[str]) -> Dict[str, List[float]]:
    if not hashes:
        return {}
    qmarks = ",".join(["?"] * len(hashes))
    rows = conn.execute(f"SELECT h, v FROM emb_cache WHERE h IN ({qmarks})", tuple(hashes)).fetchall()
    out: Dict[str, List[float]] = {}
    for h, v_json in rows:
        try:
            arr = json.loads(v_json)
            out[h] = [float(x) for x in arr] if isinstance(arr, list) else []
        except Exception:
            out[h] = []
    return out

def _cache_set_many(conn, items: Dict[str, List[float]]):
    if not items:
        return
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO emb_cache (h, v) VALUES (?, ?)",
            [(h, json.dumps(_as_float_list(vec))) for h, vec in items.items()],
        )

def _normalize_for_hash(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ---------------------------
# Build indexes — cached embeddings, docstore-first, persist
# - Excel/CSV: ONE ROW = ONE NODE (no splitting)
# - Other docs: chunk via SentenceSplitter
# - Build BM25 stats for row nodes
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
      - Excel/CSV rows become individual TextNodes (one embedding per row)
      - Other docs chunked via SentenceSplitter
      - Embeddings cached in SQLite by hash of normalized text
      - Docstore-first write, then vector store
      - Persist storage to disk (policy/email)
      - BM25 stats for row nodes (df, N, avgdl)
    """
    # Prepare explicit docstore + vector store
    docstore = SimpleDocumentStore()
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

    # Decide target cache/stat bucket
    target_cache = "policy" if "policy" in label.lower() else "email"
    # Reset caches/stats for this corpus
    if target_cache == "policy":
        st.session_state.policy_row_cache = {}
    else:
        st.session_state.email_row_cache = {}
    bm_df_local = defaultdict(int); bm_N_local = 0; bm_sumdl_local = 0

    if not docs:
        index = VectorStoreIndex([], storage_context=storage_context)
        persist_dir = POLICY_DIR if target_cache == "policy" else EMAIL_DIR
        index.storage_context.persist(persist_dir=persist_dir)
        # finalize stats
        if target_cache == "policy":
            st.session_state.policy_bm25 = {"df": {}, "N": 0, "avgdl": 0.0}
        else:
            st.session_state.email_bm25 = {"df": {}, "N": 0, "avgdl": 0.0}
        return index

    def _dummy_text() -> str:
        return ("lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50).strip()

    parser = Settings.node_parser
    total_items = len(docs)
    db_path = st.session_state.embed_cache_path

    DBG_PING_BATCHES = 3
    DBG_PING_BATCH_SIZE = 16

    with st.status(f"Embedding {label} ({total_items} docs/pages/rows)…", state="running", expanded=True) as status:
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

            for i in range(0, total_items, page_batch_size):
                batch_docs = docs[i:i+page_batch_size]

                # Build nodes:
                #   - Excel/CSV rows (metadata.row_node=True): one node per doc, with stable id
                #   - Others: chunk via parser
                nodes: List[TextNode] = []
                row_docs = [d for d in batch_docs if (d.metadata or {}).get("row_node")]
                other_docs = [d for d in batch_docs if not (d.metadata or {}).get("row_node")]

                for d in row_docs:
                    txt = d.text or ""
                    md = d.metadata or {}
                    node_id = hashlib.sha1(
                        (md.get("source","") + "|" + str(md.get("sheet","")) + "|" + str(md.get("row","")) + "|" + txt).encode("utf-8")
                    ).hexdigest()[:32]
                    n = TextNode(text=txt, metadata=md, id_=node_id)
                    nodes.append(n)

                    # --- Row cache + BM25 stats
                    toks = tokenize(txt)
                    tf = Counter(toks)
                    dl = len(toks)
                    if target_cache == "policy":
                        st.session_state.policy_row_cache[node_id] = {
                            "text": txt.lower(), "meta": md, "tf": tf, "dl": dl
                        }
                    else:
                        st.session_state.email_row_cache[node_id] = {
                            "text": txt.lower(), "meta": md, "tf": tf, "dl": dl
                        }
                    for term in tf.keys():
                        bm_df_local[term] += 1
                    bm_N_local += 1
                    bm_sumdl_local += max(1, dl)

                if other_docs:
                    nodes.extend(parser.get_nodes_from_documents(other_docs))

                # Embed with cache
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
                            new_items[hashes[miss_idx[j+k]]] = _as_float_list(e)
                        embed_items += len(sub)
                    _cache_set_many(conn, new_items)
                    dt = time.time() - t0
                    status.write(f"• Embedded {embed_items} new nodes in {dt:.2f}s (~{dt/max(1,embed_items):.2f}s/item)")
                else:
                    status.write("• All nodes from this batch served from cache")

                # attach embeddings
                for n, h in zip(nodes, hashes):
                    vec = cache_map.get(h) or new_items.get(h)
                    n.embedding = _as_float_list(vec) if vec is not None else None

                # ✅ write to DOCSTORE first, then vector store
                storage_context.docstore.add_documents(nodes)
                vector_store.add(nodes)

                total_nodes += len(nodes)
                total_hits  += hits
                total_miss  += len(miss_texts)

                items_done = min(total_items, i + len(batch_docs))
                outer_pb.progress(min(1.0, items_done / total_items))
                status.write(
                    f"Progress: items {items_done}/{total_items} | nodes: {total_nodes} | "
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

    # Finalize BM25 stats for this corpus
    stats = {
        "df": dict(bm_df_local),
        "N": bm_N_local,
        "avgdl": (bm_sumdl_local / bm_N_local) if bm_N_local > 0 else 0.0,
    }
    if target_cache == "policy":
        st.session_state.policy_bm25 = stats
    else:
        st.session_state.email_bm25 = stats

    # Bind index to storage and persist to disk
    index = VectorStoreIndex([], storage_context=storage_context)
    persist_dir = POLICY_DIR if target_cache == "policy" else EMAIL_DIR
    index.storage_context.persist(persist_dir=persist_dir)
    return index

# ---------------------------
# Rebuild row caches/BM25 from persisted indexes
# ---------------------------
def rebuild_row_caches_from_index(index, corpus: str):
    """
    Rebuild row caches (node_id -> text/meta/tf/dl) and BM25 stats (df, N, avgdl)
    by scanning the docstore of a loaded/persisted index.
    """
    if not index:
        return

    ds = getattr(index, "_docstore", None) or index.storage_context.docstore
    try:
        ids = ds.get_all_document_ids()
    except Exception:
        ids = list(getattr(ds, "docs", {}).keys())

    # Reset targets
    if corpus == "policy":
        st.session_state.policy_row_cache = {}
        bm_df_local = defaultdict(int); bm_N_local = 0; bm_sumdl_local = 0
    else:
        st.session_state.email_row_cache = {}
        bm_df_local = defaultdict(int); bm_N_local = 0; bm_sumdl_local = 0

    for nid in ids:
        node = ds.get_document(nid, raise_error=False)
        if node is None:
            continue
        md = node.metadata or {}
        if not md.get("row_node", False):
            continue
        try:
            txt = node.get_text()
        except Exception:
            txt = getattr(node, "text", "") or ""

        toks = tokenize(txt)
        tf = Counter(toks)
        dl = max(1, len(toks))

        if corpus == "policy":
            st.session_state.policy_row_cache[nid] = {"text": (txt or "").lower(), "meta": md, "tf": tf, "dl": dl}
        else:
            st.session_state.email_row_cache[nid] = {"text": (txt or "").lower(), "meta": md, "tf": tf, "dl": dl}

        for term in tf.keys():
            bm_df_local[term] += 1
        bm_N_local += 1
        bm_sumdl_local += dl

    stats = {
        "df": dict(bm_df_local),
        "N": bm_N_local,
        "avgdl": (bm_sumdl_local / bm_N_local) if bm_N_local > 0 else 0.0,
    }
    if corpus == "policy":
        st.session_state.policy_bm25 = stats
    else:
        st.session_state.email_bm25 = stats

# ---------------------------
# Load persisted indexes (and rebuild caches) on startup/refresh
# ---------------------------
def try_load_persisted_indexes():
    if os.path.isdir(POLICY_DIR) and not st.session_state.get("policy_index"):
        try:
            sc = StorageContext.from_defaults(persist_dir=POLICY_DIR)
            st.session_state.policy_index = load_index_from_storage(sc)
            rebuild_row_caches_from_index(st.session_state.policy_index, "policy")  # NEW
        except Exception as e:
            st.warning(f"Could not load persisted policy index: {e}")

    if os.path.isdir(EMAIL_DIR) and not st.session_state.get("email_index"):
        try:
            sc = StorageContext.from_defaults(persist_dir=EMAIL_DIR)
            st.session_state.email_index = load_index_from_storage(sc)
            rebuild_row_caches_from_index(st.session_state.email_index, "email")  # NEW
        except Exception as e:
            st.warning(f"Could not load persisted prior responses index: {e}")

    st.session_state.vector_ready = bool(st.session_state.policy_index or st.session_state.email_index)

try_load_persisted_indexes()

# ---------------------------
# Build / Rebuild Indexes button
# ---------------------------
if st.button("🔧 Build / Rebuild Indexes"):
    if (not st.session_state.policy_docs and not st.session_state.email_docs) and not st.session_state.vector_ready:
        st.error("Please upload at least one document (or rely on persisted indexes already loaded).")
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

        if st.session_state.email_docs:
            st.session_state.email_index = build_index_incremental_preembedded_cached(
                st.session_state.email_docs,
                label="Prior responses corpus",
                page_batch_size=embed_page_batch,
                api_batch_size=embed_api_batch,
                max_tokens_unused=0,  # ignored
            )

        st.session_state.vector_ready = bool(st.session_state.policy_index or st.session_state.email_index)
        if st.session_state.vector_ready:
            st.success("Indexes built and persisted.")
        else:
            st.info("No new documents to build. Using existing persisted indexes if available.")

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
- source_type ∈ {"Policy","PriorEmail","Blended","Insufficient"}.
- confidence is a float 0.0–1.0 (use one decimal place).
- keep_citations is an array of strings; each string must exactly match a citation tag from Evidence.

Question: {question}

Draft:
{draft}

Evidence:
{evidence}

Return:
{"source_type":"Policy","confidence":0.8,"keep_citations":["[Doc: sample.pdf, Node 0]","[Doc: prior_email.txt, Node 2]"]}
"""

# ---------------------------
# JSON extraction + source typing
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

# ---------------------------
# Rich, blended confidence scoring
# ---------------------------
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def compute_confidence(
    question: str,
    draft: str,
    tag_to_score: Dict[str, float],
    kept_tags: List[str],
    tag_to_file: Dict[str, str],
    ev_str: str,
    source_type: str,
) -> float:
    """
    Heuristic confidence in [0,1] combining:
      - similarity strength (weighted top scores)
      - citation coverage (#kept)
      - source diversity (distinct docs)
      - alignment: draft↔evidence & question↔evidence (rapidfuzz)
      - corpus weight (Policy > Blended > PriorEmail > Insufficient)
    """
    if not ev_str or ev_str.strip() == "(no evidence found)":
        return 0.0

    pairs = sorted(tag_to_score.items(), key=lambda kv: kv[1], reverse=True)
    if kept_tags:
        kept_scored = [(t, tag_to_score.get(t, 0.0)) for t in kept_tags]
        extra = [p for p in pairs if p[0] not in kept_tags]
        pairs = kept_scored + extra

    # similarity strength: normalize scores roughly and weight top-3
    top = [p[1] for p in pairs[:3]] if pairs else []
    def norm_sim(s):
        return _clamp01((s - 0.4) / (0.85 - 0.4)) if isinstance(s, (int, float)) else 0.0
    top_n = [norm_sim(s) for s in top]
    weights = [1.0, 0.7, 0.5][:len(top_n)]
    sim_strength = _clamp01(sum(w*s for w, s in zip(weights, top_n)) / (sum(weights) or 1.0))

    # coverage & diversity
    k = len(kept_tags)
    coverage = _clamp01(k / 4.0)  # saturate at 4 citations
    kept_docs = {tag_to_file.get(t, "") for t in kept_tags if t in tag_to_file}
    diversity = _clamp01(len([d for d in kept_docs if d]) / max(1.0, k)) if k else 0.0

    # alignment via rapidfuzz (use kept evidence text)
    ev_lines = ev_str.splitlines()
    ev_map = {}
    for line in ev_lines:
        if "]" in line:
            tag = line.split("]")[0] + "]"
            ev_map[tag] = line[len(tag):].strip()
    if kept_tags:
        kept_text = "\n".join(ev_map.get(t, "") for t in kept_tags)
    else:
        kept_text = "\n".join((ev_lines[i] if i < len(ev_lines) else "") for i in range(min(3, len(ev_lines))))

    def rf(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        return _clamp01(fuzz.partial_ratio(s1, s2) / 100.0)

    align_draft = rf(draft, kept_text)
    align_question = rf(question, kept_text)
    alignment = 0.6 * align_draft + 0.4 * align_question

    # corpus weight
    corpus_weight = {
        "Policy": 0.95,
        "Blended": 0.9,
        "PriorEmail": 0.8,
        "Insufficient": 0.2,
    }.get(source_type or "Insufficient", 0.2)

    raw = (
        0.45 * sim_strength +
        0.30 * alignment +
        0.15 * coverage +
        0.10 * diversity
    )
    raw = raw * (0.85 + 0.15 * corpus_weight)

    if k > 0 and raw > 0.02:
        return round(_clamp01(max(0.05, min(0.99, raw))), 3)
    if tag_to_score and raw > 0.02:
        return round(max(0.03, min(0.4, raw * 0.6)), 3)
    return 0.0

# ---------------------------
# Resilient retrieval plumbing
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

# ---------------------------
# GENERAL HYBRID RETRIEVAL (dense + BM25 + fuzzy)
# ---------------------------
def retrieve_across(q: str, k: int) -> List[NodeWithScore]:
    """
    General hybrid retrieval:
      1) Dense candidates from both corpora (larger pool)
      2) Rerank each candidate by combined score:
         score = 0.60*dense + 0.25*BM25(row-only) + 0.15*fuzzy
         (BM25 applies when the node is an Excel/CSV row we cached.)
      3) If still weak, do a row-only sweep via BM25+fuzzy across all row nodes.
    """
    q = (q or "").strip()
    if not q:
        return []
    q_tokens = tokenize(q)

    # 1) dense pool
    pool_k = max(k * 5, 20)
    cand: List[NodeWithScore] = []
    cand += _safe_retrieve_with_fallback(st.session_state.policy_index, "policy", q, pool_k)
    cand += _safe_retrieve_with_fallback(st.session_state.email_index,  "email",  q, pool_k)

    # refs
    pol_cache = st.session_state.policy_row_cache
    eml_cache = st.session_state.email_row_cache
    pol_bm = st.session_state.policy_bm25
    eml_bm = st.session_state.email_bm25

    def bm25_for_node(node_id: str) -> float:
        if node_id in pol_cache:
            rec = pol_cache[node_id]
            return bm25_score_row(q_tokens, rec["tf"], rec["dl"], pol_bm["df"], pol_bm["N"], pol_bm["avgdl"])
        if node_id in eml_cache:
            rec = eml_cache[node_id]
            return bm25_score_row(q_tokens, rec["tf"], rec["dl"], eml_bm["df"], eml_bm["N"], eml_bm["avgdl"])
        return 0.0

    def fuzzy_for_node(node: TextNode) -> float:
        try:
            txt = (node.get_text() or "").lower()
        except Exception:
            txt = ""
        return fuzzy_partial(q, txt)

    if cand:
        reranked = []
        for nws in cand:
            dense = norm_sim_dense(nws.score or 0.0)
            node = nws.node
            nid = getattr(node, "node_id", None) or getattr(node, "id_", None) or ""
            bm = bm25_for_node(nid)
            fz = fuzzy_for_node(node)
            score = 0.60 * dense + 0.25 * bm + 0.15 * fz
            reranked.append((score, nws))
        reranked.sort(key=lambda x: x[0], reverse=True)
        top = [NodeWithScore(node=pair[1].node, score=pair[0]) for pair in reranked[:k]]
        if top and top[0].score >= 0.35:
            return top

    # 3) Row-only sweep if dense is inconclusive
    row_hits: List[Tuple[float, str, str]] = []  # (score, node_id, corpus)
    def sweep(cache, bm, corpus):
        for nid, rec in cache.items():
            bm_s = bm25_score_row(q_tokens, rec["tf"], rec["dl"], bm["df"], bm["N"], bm["avgdl"])
            fz_s = fuzzy_partial(q, rec["text"])
            s = 0.70 * bm_s + 0.30 * fz_s
            if s > 0:
                row_hits.append((s, nid, corpus))

    sweep(pol_cache, pol_bm, "policy")
    sweep(eml_cache, eml_bm, "email")

    if not row_hits:
        return []

    row_hits.sort(key=lambda x: x[0], reverse=True)
    row_hits = row_hits[:k]

    out: List[NodeWithScore] = []
    for score, nid, corpus in row_hits:
        index = st.session_state.policy_index if corpus == "policy" else st.session_state.email_index
        if not index:
            continue
        ds = getattr(index, "_docstore", None) or index.storage_context.docstore
        node = ds.get_document(nid, raise_error=False)
        if node is not None:
            out.append(NodeWithScore(node=node, score=score))
    return out

# ---------------------------
# Evidence formatting
# ---------------------------
def nodes_to_evidence(nodes_ws: List[NodeWithScore]) -> Tuple[str, Dict[str, str], Dict[str, float], Dict[str, Optional[int]]]:
    evidence_lines: List[str] = []
    tag_to_file: Dict[str, str] = {}
    tag_to_score: Dict[str, float] = {}
    tag_to_page: Dict[str, Optional[int]] = {}

    for i, nws in enumerate(nodes_ws):
        node = nws.node
        md = node.metadata or {}
        src  = md.get("source", "unknown")
        page = md.get("page")
        sheet = md.get("sheet")
        row = md.get("row")
        row_key = md.get("row_key")
        if page is not None:
            tag = f"[Doc: {src}, Page {page}, Node {i}]"
        elif sheet is not None and row is not None:
            if row_key:
                tag = f"[Doc: {src}, Sheet {sheet}, Row {row}, Key {row_key}, Node {i}]"
            else:
                tag = f"[Doc: {src}, Sheet {sheet}, Row {row}, Node {i}]"
        else:
            tag = f"[Doc: {src}, Node {i}]"

        # Optional: add a tiny header hint (first 3 columns) to the snippet text (not the tag)
        headers = md.get("headers")
        header_hint = f" | Columns: {', '.join(headers[:3])}" if isinstance(headers, list) and headers else ""

        text = re.sub(r"\s+", " ", node.get_text()).strip()
        evidence_lines.append(f"{tag} {text}{header_hint}")
        tag_to_file[tag] = src
        tag_to_page[tag] = page  # may be None for Excel/CSV
        try:
            tag_to_score[tag] = float(nws.score or 0.0)
        except Exception:
            tag_to_score[tag] = 0.0

    return "\n".join(evidence_lines), tag_to_file, tag_to_score, tag_to_page

# ---------------------------
# Ensure indexes (lazy load/build), now that rebuild is defined
# ---------------------------
def ensure_indexes_built() -> bool:
    # reload persisted if available (also rebuilds caches)
    try_load_persisted_indexes()
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
# 🔍 Inspect index contents (nodes/chunks)
# ---------------------------
with st.expander("🔍 Inspect index contents (nodes/chunks)", expanded=False):
    corpus_choice = st.selectbox("Corpus", ["Policy", "Prior Responses"])
    index = st.session_state.policy_index if corpus_choice == "Policy" else st.session_state.email_index

    if not index:
        st.info("No index loaded for this corpus yet.")
    else:
        ds = getattr(index, "_docstore", None) or index.storage_context.docstore
        try:
            ids = ds.get_all_document_ids()
        except Exception:
            ids = list(getattr(ds, "docs", {}).keys())

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            kw = st.text_input("Filter by keyword (substring)")
        with col2:
            src_filter = st.text_input("Filter by source filename contains")
        with col3:
            sheet_filter = st.text_input("Filter by sheet name contains (Excel only)")

        show_rows_only = st.checkbox("Show ONLY Excel/CSV row nodes", value=False)
        max_show = st.slider("Show up to N entries", 50, 2000, 500, 50)

        rows = []
        total_rows = 0
        for nid in ids:
            node = ds.get_document(nid, raise_error=False)
            if node is None:
                continue
            md = node.metadata or {}
            try:
                text = node.get_text()
            except Exception:
                text = getattr(node, "text", "") or ""
            src = md.get("source", "")
            page = md.get("page")
            sheet = md.get("sheet")
            rowno = md.get("row")
            is_row = md.get("row_node", False)

            if show_rows_only and not is_row:
                continue
            if src_filter and (src_filter.lower() not in str(src).lower()):
                continue
            if sheet_filter and (sheet is None or sheet_filter.lower() not in str(sheet).lower()):
                continue
            if kw and (kw.lower() not in text.lower()):
                continue

            total_rows += 1
            if len(rows) < max_show:
                rows.append({
                    "node_id": nid,
                    "source": src,
                    "page": page,
                    "sheet": sheet,
                    "row": rowno,
                    "row_node": is_row,
                    "chars": len(text),
                    "preview": re.sub(r"\s+", " ", text)[:300]
                })

        st.write(f"Showing {len(rows)} / {total_rows} matching nodes.")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No nodes matched your filters.")

# ---------------------------
# Generate Responses (lazy-build if needed)
# ---------------------------
if st.button("🧠 Generate Responses"):
    ready = ensure_indexes_built()
    if not ready:
        st.error("No indexes available. Upload documents and build indexes first (or rely on persisted indexes).")
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
                        judge_conf = float(parsed.get("confidence", 0.0))
                    except Exception:
                        judge_conf = 0.0
                else:
                    sorted_tags = sorted(tag2score.items(), key=lambda kv: kv[1], reverse=True)
                    keep = [t for (t, _) in sorted_tags[:3]]
                    source_type = infer_source_type_from_tags(keep, tag2file)
                    judge_conf = 0.0

                # Our computed confidence from multiple signals
                comp_conf = compute_confidence(q, draft, tag2score, keep, tag2file, ev_str, source_type)

                # Blend judge with computed
                if judge_conf <= 0.0:
                    confidence = comp_conf
                else:
                    hi = max(judge_conf, comp_conf)
                    lo = min(judge_conf, comp_conf)
                    confidence = round(_clamp01(0.65 * hi + 0.35 * lo), 3)

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
                            m = re.search(r"\[Doc: (.*?)(?:, Page \d+|, Sheet .*?, Row \d+(?:, Key [^,]+)?)*, Node \d+\]", line)
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
                            # PDF preview
                            if fname and page and fname.lower().endswith(".pdf"):
                                data = st.session_state.file_bytes_map.get(fname)
                                if data is None:
                                    p = st.session_state.saved_file_paths.get(fname)
                                    if p and os.path.isfile(p):
                                        with open(p, "rb") as fh:
                                            data = fh.read()
                                if data:
                                    img = render_pdf_page_with_highlight(data, page, query_text)
                                    if img is not None:
                                        st.image(img, caption=f"{fname} — Page {page}", use_column_width=True)
                                    else:
                                        page_text = ""
                                        if fname in st.session_state.pdf_page_texts:
                                            page_text = st.session_state.pdf_page_texts[fname].get(page, "")
                                        if not page_text and data:
                                            try:
                                                pages = read_pdf_pages(data)
                                                st.session_state.pdf_page_texts[fname] = {p: t for p, t in pages}
                                                page_text = st.session_state.pdf_page_texts[fname].get(page, "")
                                            except Exception:
                                                page_text = ""
                                        if page_text:
                                            safe_html = highlight_html(page_text, query_text, window=600)
                                            st.markdown(f"**{fname} — Page {page} (text preview)**")
                                            st.markdown(safe_html, unsafe_allow_html=True)
                                        else:
                                            st.info("Preview unavailable for this PDF page.")
                                else:
                                    st.info("Preview unavailable for this evidence.")

                            # Excel/CSV preview (exact row)
                            elif fname and (fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls") or fname.lower().endswith(".csv")):
                                m = re.search(r"Sheet (.*?), Row (\d+)", tag)
                                sheet_name = None
                                row_num = None
                                if m:
                                    sheet_name = m.group(1).strip()
                                    try:
                                        row_num = int(m.group(2))
                                    except Exception:
                                        row_num = None

                                data = st.session_state.file_bytes_map.get(fname)
                                if data is None:
                                    p = st.session_state.saved_file_paths.get(fname)
                                    if p and os.path.isfile(p):
                                        with open(p, "rb") as fh:
                                            data = fh.read()

                                if data:
                                    try:
                                        if fname.lower().endswith(".csv"):
                                            df = pd.read_csv(io.BytesIO(data), dtype=str).fillna("")
                                        else:
                                            if sheet_name:
                                                df = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name, dtype=str).fillna("")
                                            else:
                                                xl = pd.ExcelFile(io.BytesIO(data))
                                                df = pd.read_excel(xl, sheet_name=xl.sheet_names[0], dtype=str).fillna("")
                                        if row_num and 1 <= row_num <= len(df):
                                            st.markdown(f"**{fname} — {('Sheet ' + sheet_name + ', ') if sheet_name else ''}Row {row_num}**")
                                            st.dataframe(df.iloc[[row_num - 1]], use_container_width=True)
                                        else:
                                            st.markdown(f"**{fname} — preview**")
                                            st.dataframe(df.head(10), use_container_width=True)
                                    except Exception as e:
                                        st.info(f"Preview unavailable for this table: {e}")
                                else:
                                    st.info("Preview unavailable for this evidence.")

                            # DOCX/TXT or unknown → text preview
                            else:
                                src_file = fname or "(unknown)"
                                data = st.session_state.file_bytes_map.get(src_file)
                                if data is None:
                                    p = st.session_state.saved_file_paths.get(src_file)
                                    if p and os.path.isfile(p):
                                        with open(p, "rb") as fh:
                                            data = fh.read()

                                preview_text = ""
                                if data and src_file:
                                    lower = src_file.lower()
                                    try:
                                        if lower.endswith(".pdf"):
                                            if src_file in st.session_state.pdf_page_texts:
                                                preview_text = " ".join(st.session_state.pdf_page_texts[src_file].values())
                                            else:
                                                pages = read_pdf_pages(data)
                                                st.session_state.pdf_page_texts[src_file] = {p: t for p, t in pages}
                                                preview_text = " ".join(t for _, t in pages)
                                        elif lower.endswith(".docx"):
                                            preview_text = read_docx(data)
                                        elif lower.endswith(".txt"):
                                            preview_text = read_txt(data)
                                        else:
                                            # excel/csv without sheet/row info in tag
                                            if lower.endswith(".csv"):
                                                df = pd.read_csv(io.BytesIO(data), dtype=str).fillna("")
                                                preview_text = df.head(10).to_csv(index=False)
                                            elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                                                xl = pd.ExcelFile(io.BytesIO(data))
                                                df = pd.read_excel(xl, sheet_name=xl.sheet_names[0], dtype=str).fillna("")
                                                preview_text = df.head(10).to_csv(index=False)
                                    except Exception:
                                        preview_text = ""

                                if preview_text:
                                    safe_html = highlight_html(preview_text, query_text, window=600)
                                    st.markdown(f"**{src_file} (text/table preview)**")
                                    st.markdown(safe_html, unsafe_allow_html=True)
                                else:
                                    st.info("Preview unavailable for this evidence.")
                else:
                    st.write("_No supporting evidence found._")

                if show_debug:
                    st.subheader("Debug: Judge Raw Output")
                    st.code(r["judge_raw"][:4000], language="json")
