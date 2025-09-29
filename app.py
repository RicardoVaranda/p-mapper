# app.py — Vendor Q&A Policy Checker (Streamlit + LlamaIndex 0.12.x)
# ------------------------------------------------------------------
# Purpose
#  - Ask vendors a set of prewritten questions (or add your own).
#  - For each question, the vendor provides:
#       1) a plain‑English Answer,
#       2) supporting Evidence documents (per‑question uploads).
#  - Company policy documents are uploaded once (global corpus).
#  - The app:
#       a) maps each Question to the best‑matching internal policy passage(s),
#       b) judges whether the Vendor Answer meets/exceeds policy,
#       c) extracts evidence snippets from the vendor's uploaded docs that support the answer,
#       d) returns a Suggestion: ✅ Acceptable / ❌ Not acceptable / ⚠️ Insufficient evidence.
#
# Domain‑agnostic. Evidence‑driven. No hard‑coded validations.
#
# Quickstart
# ----------
# 1) pip install -r requirements.txt
# 2) streamlit run app.py
# 3) In the sidebar, set your OpenAI‑compatible API key and models.
#
# requirements.txt (example)
# --------------------------
# streamlit>=1.36.0
# llama-index>=0.12.50
# llama-index-llms-openai>=0.2.3
# llama-index-embeddings-openai>=0.2.3
# PyPDF2>=3.0.1
# docx2txt>=0.8
# pandas>=2.2.2
# tiktoken>=0.7.0

import os
import io
import re
import json
import uuid
import shutil
import hashlib
import typing as t
from dataclasses import dataclass

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import docx2txt

# ---- LlamaIndex 0.12.x ----
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ---------------------------
# Persistent storage roots
# ---------------------------
PERSIST_ROOT = ".storage_vendors"
POLICY_DIR   = os.path.join(PERSIST_ROOT, "policy")
os.makedirs(POLICY_DIR, exist_ok=True)

# ---------------------------
# Streamlit page
# ---------------------------
st.set_page_config(page_title="Vendor Q&A Policy Checker", layout="wide")
st.title("📋 Vendor Q&A Policy Checker — Evidence‑Driven")
st.caption("Upload global policy docs once • For each question: Vendor answer + evidence docs • Get policy mapping, judgement, and supporting citations.")

# ---------------------------
# Sidebar — Settings & Cache controls
# ---------------------------
with st.sidebar:
    st.header("LLM Settings")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    base_url = st.text_input("OPENAI_BASE_URL (optional)", value=os.getenv("OPENAI_BASE_URL", ""))
    chat_model = st.text_input("Chat Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    embed_model = st.text_input("Embedding Model", value=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    st.markdown("### Retrieval")
    k_policy = st.slider("Policy Top‑K", 1, 10, 3)
    k_evidence_per_claim = st.slider("Evidence per claim", 1, 5, 2)

    st.markdown("### Index persistence")
    auto_build = st.checkbox("Auto‑build policy index on upload", value=True)
    cache_root = st.text_input("Index cache directory", value=PERSIST_ROOT)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear policy index cache"):
            try:
                shutil.rmtree(POLICY_DIR, ignore_errors=True)
                st.session_state.pop("policy_index", None)
                st.success("Cleared policy index cache.")
            except Exception as e:
                st.warning(f"Couldn't clear: {e}")
    with c2:
        st.caption("Indexes persist to disk and are reloaded automatically.")

# ---------------------------
# Configure LlamaIndex defaults
# ---------------------------

def configure_settings():
    kw = {"temperature": 0.0}
    if base_url:
        kw["base_url"] = base_url
    Settings.llm = OpenAI(model=chat_model, api_key=api_key or None, **kw)
    Settings.embed_model = OpenAIEmbedding(model=embed_model, api_key=api_key or None, base_url=base_url or None)
    # deterministic chunking for non-row docs; rows become single TextNodes
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

configure_settings()

# ---------------------------
# Read files → Documents (page‑aware for PDFs, row‑aware for Excel/CSV)
# ---------------------------

def read_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for i, p in enumerate(reader.pages, start=1):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i, txt))
    return out


def read_docx(file_bytes: bytes) -> str:
    tmp = f"/tmp/{uuid.uuid4()}.docx"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    try:
        txt = docx2txt.process(tmp) or ""
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass
    return txt


def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def read_excel_rows(file_bytes: bytes) -> list[tuple[str, int, dict]]:
    """Return list of (sheet_name, row_index_1based, ordered_row_dict)."""
    out: list[tuple[str, int, dict]] = []
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
        if df.empty:
            continue
        df = df.fillna("")
        cols = [str(c) for c in df.columns]
        for ridx, row in df.iterrows():
            row_dict: dict[str, str] = {}
            for c in cols:
                v = str(row[c]).strip()
                if v != "":
                    row_dict[c] = v
            if not row_dict:
                continue
            out.append((sheet, int(ridx) + 1, row_dict))
    return out


def read_csv_rows(file_bytes: bytes) -> list[tuple[str, int, dict]]:
    out: list[tuple[str, int, dict]] = []
    bio = io.BytesIO(file_bytes)
    df = pd.read_csv(bio, dtype=str)
    if df.empty:
        return out
    df = df.fillna("")
    cols = [str(c) for c in df.columns]
    for ridx, row in df.iterrows():
        row_dict: dict[str, str] = {}
        for c in cols:
            v = str(row[c]).strip()
            if v != "":
                row_dict[c] = v
        if not row_dict:
            continue
        out.append(("CSV", int(ridx) + 1, row_dict))
    return out


def row_dict_to_text(row: dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in row.items())


def files_to_docs(files: list, kind: str) -> list[dict]:
    docs: list[dict] = []
    for f in files or []:
        name = f.name
        data = f.read()
        f.seek(0)
        lower = name.lower()
        if lower.endswith(".pdf"):
            pages = read_pdf_pages(data)
            for page_no, txt in pages:
                if txt.strip():
                    docs.append({"text": txt, "metadata": {"source": name, "page": page_no, "type": kind}})
        elif lower.endswith(".docx"):
            txt = read_docx(data)
            if txt.strip():
                docs.append({"text": txt, "metadata": {"source": name, "type": kind}})
        elif lower.endswith(".txt") or lower.endswith(".md"):
            txt = read_txt(data)
            if txt.strip():
                docs.append({"text": txt, "metadata": {"source": name, "type": kind}})
        elif lower.endswith(".xlsx") or lower.endswith(".xls"):
            try:
                rows = read_excel_rows(data)
            except Exception as e:
                st.warning(f"Failed to read Excel {name}: {e}")
                rows = []
            for sheet, rownum, row_dict in rows:
                txt = row_dict_to_text(row_dict)
                if not txt.strip():
                    continue
                row_key = hashlib.sha1(("||".join(f"{k}={v}" for k, v in row_dict.items())).encode()).hexdigest()[:12]
                meta = {"source": name, "sheet": sheet, "row": rownum, "row_key": row_key, "headers": list(row_dict.keys()), "type": kind, "row_node": True}
                docs.append({"text": txt, "metadata": meta})
        elif lower.endswith(".csv"):
            try:
                rows = read_csv_rows(data)
            except Exception as e:
                st.warning(f"Failed to read CSV {name}: {e}")
                rows = []
            for sheet, rownum, row_dict in rows:
                txt = row_dict_to_text(row_dict)
                if not txt.strip():
                    continue
                row_key = hashlib.sha1(("||".join(f"{k}={v}" for k, v in row_dict.items())).encode()).hexdigest()[:12]
                meta = {"source": name, "sheet": sheet, "row": rownum, "row_key": row_key, "headers": list(row_dict.keys()), "type": kind, "row_node": True}
                docs.append({"text": txt, "metadata": meta})
        else:
            st.warning(f"Unsupported file type: {name}")
    return docs

# ---------------------------
# Persistent index cache (policy global; vendor per‑question)
# ---------------------------

@dataclass
class IndexedCorpus:
    index: t.Any | None
    raw_docs: list[Document]
    loaded_from_cache: bool


def _fingerprint_docs(docs: list[dict]) -> str:
    h = hashlib.sha1()
    for d in docs:
        md = d.get("metadata", {}) or {}
        src = md.get("source", "")
        pg  = str(md.get("page", ""))
        sheet = md.get("sheet", "")
        row = str(md.get("row", ""))
        h.update(f"{src}|{pg}|{sheet}|{row}|SEP|".encode("utf-8", "ignore"))
        h.update(((d.get("text", "") or "") + "|SEP|TEXT").encode("utf-8", "ignore"))
    h.update(("embed=" + (embed_model or "default")).encode("utf-8"))
    return h.hexdigest()


def build_or_load_index_persisted(
    docs: list[dict], *, prefix: str, persist_root: str
) -> IndexedCorpus:
    """Load a VectorStoreIndex from disk if present; else build and persist.
       Ensures: Excel/CSV rows become **single TextNodes** (one chunk per row).
    """
    li_docs = [Document(text=d.get("text", ""), metadata=d.get("metadata", {})) for d in docs]
    if not docs:
        return IndexedCorpus(index=None, raw_docs=li_docs, loaded_from_cache=False)

    key = f"{prefix}-{_fingerprint_docs(docs)}"
    cache_dir = os.path.join(persist_root, key)

    # Try load
    try:
        if os.path.isdir(cache_dir) and os.listdir(cache_dir):
            sc = StorageContext.from_defaults(persist_dir=cache_dir)
            idx = load_index_from_storage(sc)
            return IndexedCorpus(index=idx, raw_docs=li_docs, loaded_from_cache=True)
    except Exception:
        pass

    # Build nodes explicitly: rows → TextNode (no re‑chunk), others → parser chunks
    configure_settings()
    row_nodes: list[TextNode] = []
    other_docs: list[Document] = []
    for d in li_docs:
        md = d.metadata or {}
        if md.get("row_node"):
            # stable id from (source|sheet|row|text)
            node_id = hashlib.sha1((md.get("source","") + "|" + str(md.get("sheet","")) + "|" + str(md.get("row","")) + "|" + (d.text or "")).encode()).hexdigest()[:32]
            row_nodes.append(TextNode(text=d.text, metadata=md, id_=node_id))
        else:
            other_docs.append(d)

    # Parse non‑row docs to nodes
    other_nodes: list[TextNode] = []
    if other_docs:
        other_nodes = Settings.node_parser.get_nodes_from_documents(other_docs)

    all_nodes = row_nodes + other_nodes
    idx = VectorStoreIndex(nodes=all_nodes)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        idx.storage_context.persist(persist_dir=cache_dir)
    except Exception:
        pass
    return IndexedCorpus(index=idx, raw_docs=li_docs, loaded_from_cache=False)

# ---------------------------
# Retrieval & LLM helpers
# ---------------------------

def retrieve_answer(indexed: IndexedCorpus, question: str, k: int) -> tuple[str, list[dict]]:
    """Return (answer_text, sources) where sources carry doc, page OR sheet/row and a snippet."""
    def _pack_src(meta: dict, content: str) -> dict:
        return {
            "source": meta.get("source", meta.get("file_name", "unknown")),
            "page": meta.get("page"),
            "sheet": meta.get("sheet"),
            "row": meta.get("row"),
            "snippet": content[:500],
            "headers": meta.get("headers"),
        }

    if indexed.index is None:
        # naive skim over raw docs
        toks = [t for t in re.split(r"\W+", (question or "").lower()) if t]
        scored: list[tuple[int, dict]] = []
        for d in indexed.raw_docs:
            txt = (d.text or "").lower()
            score = sum(txt.count(tok) for tok in toks)
            scored.append((score, {"text": d.text, "metadata": d.metadata}))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        seen = set()
        for _, d in scored[:k]:
            meta = d.get("metadata", {}) or {}
            key = (meta.get("source"), meta.get("page"), meta.get("sheet"), meta.get("row"))
            if key in seen: continue
            seen.add(key)
            out.append(_pack_src(meta, d.get("text", "") or ""))
        return "", out

    try:
        qe = indexed.index.as_query_engine(similarity_top_k=k)
        resp = qe.query(question)
        out = []
        seen = set()
        for n in getattr(resp, "source_nodes", [])[:k]:
            meta = n.node.metadata or {}
            key = (meta.get("source"), meta.get("page"), meta.get("sheet"), meta.get("row"))
            if key in seen: continue
            seen.add(key)
            out.append(_pack_src(meta, n.node.get_content()))
        return getattr(resp, "response", str(resp)), out
    except Exception as e:
        return f"[query error] {e}", []


def extract_json_block(text: str) -> str | None:
    s = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}$", s)
    if m: return m.group(0)
    m = re.search(r"\{[\s\S]*?\}", s)
    return m.group(0) if m else None


def try_parse_json(text: str) -> t.Any | None:
    blk = extract_json_block(text)
    if not blk: return None
    try:
        return json.loads(blk)
    except Exception:
        blk2 = re.sub(r",(\s*[}\]])", r"\1", blk)
        try:
            return json.loads(blk2)
        except Exception:
            return None


def extract_vendor_claims(answer_text: str) -> list[str]:
    answer_text = (answer_text or "").strip()
    if not answer_text:
        return []
    try:
        configure_settings()
        prompt = (
            "Extract atomic, testable CLAIMS from the vendor's answer.\n"
            "Return JSON: {claims: [\"short claim\", ...]}\n\n"
            f"VendorAnswer:\n{answer_text}\n"
        )
        resp = Settings.llm.complete(prompt)
        obj = try_parse_json(getattr(resp, "text", str(resp)))
        if obj and isinstance(obj, dict) and isinstance(obj.get("claims"), list):
            claims = [str(c).strip() for c in obj["claims"] if str(c).strip()]
            # de‑dup
            seen = set(); out = []
            for c in claims:
                k = c.lower()
                if k not in seen:
                    seen.add(k); out.append(c)
            return out[:8]
    except Exception:
        pass
    # Fallback: sentence split
    sents = re.split(r"(?<=[.!?])\s+", answer_text)
    return [s.strip() for s in sents if 20 <= len(s) <= 220][:5]


def gather_evidence(indexed: IndexedCorpus, claims: list[str], k_per: int) -> list[dict]:
    cites: list[dict] = []
    for cl in claims or []:
        _ans, srcs = retrieve_answer(indexed, cl, k=k_per)
        for s in srcs:
            cites.append({"claim": cl, **s})
    # de‑dup
    seen = set(); out = []
    for c in cites:
        key = (c["source"], c.get("page"), c.get("sheet"), c.get("row"), (c.get("snippet", "")[:120]))
        if key in seen: continue
        seen.add(key); out.append(c)
    return out[:12]


def llm_judge(policy_answer: str, vendor_answer: str) -> tuple[bool, str]:
    try:
        configure_settings()
        prompt = (
            "You are a compliance analyst. Compare the company policy answer and the vendor answer. "
            "Decide if the vendor meets or exceeds the policy (binary yes/no). "
            "Return a single JSON object with keys: meets(bool), rationale(str).\n\n"
            f"CompanyPolicyAnswer:\n{policy_answer}\n\nVendorAnswer:\n{vendor_answer}\n"
        )
        resp = Settings.llm.complete(prompt)
        text = getattr(resp, "text", str(resp)).strip()
        obj = try_parse_json(text)
        if obj and isinstance(obj, dict):
            return bool(obj.get("meets", False)), str(obj.get("rationale", ""))
        # fallback: keyword
        meets = bool(re.search(r"\b(yes|meets|compliant)\b", text.lower()))
        return meets, text
    except Exception as e:
        return False, f"[judge error] {e}"

# ---------------------------
# Helpers: label/heading extraction for policy mapping rows
# ---------------------------

def _extract_policy_label_from_snippet(snippet: str) -> str | None:
    if not snippet:
        return None
    # Look for lines like "Control Name: X", "Policy: Y", "Requirement: Z", etc.
    candidates = [
        r"^control\s*name\s*:\s*(.+)$",
        r"^control\s*id\s*:\s*(.+)$",
        r"^policy\s*name\s*:\s*(.+)$",
        r"^policy\s*:\s*(.+)$",
        r"^requirement\s*:\s*(.+)$",
        r"^title\s*:\s*(.+)$",
        r"^name\s*:\s*(.+)$",
        r"^id\s*:\s*(.+)$",
    ]
    lines = [l.strip() for l in snippet.splitlines() if l.strip()]
    for line in lines[:12]:  # scan first dozen lines
        for pat in candidates:
            m = re.match(pat, line, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                return (val[:120] + "…") if len(val) > 120 else val
    # fallback: first short-ish line
    for line in lines:
        if 3 <= len(line) <= 80:
            return line
    # ultimate fallback: first 10 words of snippet
    words = re.split(r"\s+", snippet)
    return " ".join(words[:10]) + ("…" if len(words) > 10 else "")


# ---------------------------
# Session state containers
# ---------------------------
if "policy_index" not in st.session_state:
    st.session_state.policy_index: IndexedCorpus | None = None
if "q_items" not in st.session_state:
    st.session_state.q_items = []  # [{id,text,answer}]

# ---------------------------
# Upload global policy docs (built ahead of questions) — ACCEPTS Excel/CSV
# ---------------------------

st.subheader("1) Upload Company Policy Documents (global)")
pol_files = st.file_uploader(
    "Upload .pdf / .docx / .txt / .md / .xlsx / .xls / .csv",
    type=["pdf","docx","txt","md","xlsx","xls","csv"],
    accept_multiple_files=True
)
policy_docs = files_to_docs(pol_files, "policy") if pol_files else []

if policy_docs:
    st.success(f"Loaded {len(policy_docs)} policy doc parts (pages/rows/files).")
    if auto_build:
        with st.spinner("Building/Loading policy index…"):
            st.session_state.policy_index = build_or_load_index_persisted(policy_docs, prefix="policy", persist_root=PERSIST_ROOT)
            st.info("Policy index " + ("loaded from cache." if st.session_state.policy_index.loaded_from_cache else "built and cached."))

if not st.session_state.policy_index and os.path.isdir(POLICY_DIR) and os.listdir(POLICY_DIR):
    try:
        sc = StorageContext.from_defaults(persist_dir=POLICY_DIR)
        idx = load_index_from_storage(sc)
        st.session_state.policy_index = IndexedCorpus(index=idx, raw_docs=[], loaded_from_cache=True)
        st.info("Policy index loaded from existing cache.")
    except Exception:
        pass

st.divider()

# ---------------------------
# 2) Questions: prewritten + dynamic rows (answer + evidence per question)
# ---------------------------

st.subheader("2) Vendor Questions, Answers, & Evidence (per question)")

PREWRITTEN = [
    "Does the vendor's password policy meet our minimum requirements?",
    "Is customer data retained for at least 7 years as per policy?",
    "Is encryption at rest implemented for all customer data?",
    "Is multi‑factor authentication enforced for privileged access?",
]

row_ctrl = st.columns([1,1,2,4])
if row_ctrl[0].button("➕ Add question", use_container_width=True):
    st.session_state.q_items.append({"id": str(uuid.uuid4())[:8], "text": "", "answer": ""})
if row_ctrl[1].button("➖ Remove last", use_container_width=True) and st.session_state.q_items:
    st.session_state.q_items.pop()
if row_ctrl[2].button("Load prewritten set", use_container_width=True):
    st.session_state.q_items = [{"id": str(uuid.uuid4())[:8], "text": q, "answer": ""} for q in PREWRITTEN]

if not st.session_state.q_items:
    st.info("Add a question to begin, or load the prewritten set.")

# For each question: text, vendor answer, vendor evidence upload (per‑question index built on the fly)
per_question_vendor_indexes: dict[str, IndexedCorpus] = {}

for i, item in enumerate(st.session_state.q_items, start=1):
    qid = item["id"]
    with st.container(border=True):
        st.markdown(f"**Question {i}**")
        c1, c2 = st.columns([2,2])
        with c1:
            qtext = st.text_input("Question", value=item.get("text",""), key=f"qtext_{qid}")
            st.session_state.q_items[i-1]["text"] = qtext
            ans = st.text_area("Vendor Answer", value=item.get("answer",""), height=120, key=f"ans_{qid}", placeholder="Vendor's plain‑English answer…")
            st.session_state.q_items[i-1]["answer"] = ans
        with c2:
            ev_files = st.file_uploader("Evidence docs", type=["pdf","docx","txt","md","xlsx","xls","csv"], accept_multiple_files=True, key=f"ev_{qid}")
            ev_docs = files_to_docs(ev_files, "third") if ev_files else []
            if ev_docs:
                per_question_vendor_indexes[qid] = build_or_load_index_persisted(ev_docs, prefix=f"third-{qid}", persist_root=PERSIST_ROOT)
                st.caption("Evidence index " + ("loaded from cache." if per_question_vendor_indexes[qid].loaded_from_cache else "built and cached."))
            else:
                per_question_vendor_indexes[qid] = IndexedCorpus(index=None, raw_docs=[], loaded_from_cache=False)

st.divider()

# ---------------------------
# 3) Run validation
# ---------------------------
run = st.button("Run Validation", type="primary", use_container_width=True,
                disabled=not st.session_state.q_items or not st.session_state.policy_index)

if run:
    results_rows: list[dict] = []
    with st.spinner("Analyzing questions…"):
        policy_corpus = st.session_state.policy_index
        for idx, item in enumerate(st.session_state.q_items, start=1):
            q = (item.get("text") or "").strip()
            a = (item.get("answer") or "").strip()
            if not q:
                continue

            # 1) Map question → policy (best passages)
            pol_answer, pol_sources = retrieve_answer(policy_corpus, q, k=k_policy)
            # Build a human-friendly mapping label: <Control/Policy> — <Doc> (loc)
            if pol_sources:
                src0 = pol_sources[0]
                control = _extract_policy_label_from_snippet(src0.get("snippet", "")) or "(unlabeled)"
                src_name = src0.get("source", "(unknown)")
                if src0.get("page"):
                    loc = f"p.{src0['page']}"
                elif src0.get("sheet") and src0.get("row"):
                    loc = f"{src0['sheet']} r.{src0['row']}"
                else:
                    loc = "(location n/a)"
                mapping_label = f"{control} — {src_name} ({loc})"
            else:
                mapping_label = "(no policy match)"

            # 2) Judge vendor answer vs policy
            meets, rationale = llm_judge(pol_answer, a)

            # 3) Evidence from vendor docs (supports answer, not the question)
            vend_index = per_question_vendor_indexes.get(item["id"]) or IndexedCorpus(index=None, raw_docs=[], loaded_from_cache=False)
            claims = extract_vendor_claims(a)
            ev_cites = gather_evidence(vend_index, claims, k_per=k_evidence_per_claim) if claims else []

            # 4) Suggestion
            if meets and ev_cites:
                suggestion = "✅ Acceptable"
            elif not meets:
                suggestion = "❌ Not acceptable"
            else:
                suggestion = "⚠️ Insufficient evidence"

            def fmt_source(s: dict) -> str:
                where = (
                    f" p.{s['page']}" if s.get('page') else (
                        f" {s['sheet']} r.{s['row']}" if s.get('sheet') and s.get('row') else ""
                    )
                )
                sn = (s.get('snippet','') or '')
                return f"• {s['source']}{where}: {sn[:260]}…"

            ev_display = "\n\n".join(fmt_source(s) for s in ev_cites[:6]) or "(no evidence)"

            results_rows.append({
                "#": idx,
                "Question": q,
                "Vendor Answer": a[:900] if a else "(none)",
                "Internal Policy (best match)": mapping_label,
                "Policy Answer": pol_answer[:800],
                "Evidence (from vendor docs)": ev_display,
                "Suggestion": suggestion,
                "Rationale": (rationale or "").strip()[:900],
                "Evidence cites": len(ev_cites),
            })

    st.success("Validation complete.")

    # Results table
    df = pd.DataFrame(results_rows)
    m1, m2, m3 = st.columns(3)
    m1.metric("Questions", len(df))
    m2.metric("Acceptable", int((df["Suggestion"] == "✅ Acceptable").sum()))
    m3.metric("Not acceptable", int((df["Suggestion"] == "❌ Not acceptable").sum()))

    st.dataframe(df, use_container_width=True, height=560)

    # Downloads
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results (CSV)", csv, "vendor_policy_validation.csv", "text/csv", use_container_width=True)
    json_bytes = json.dumps(results_rows, indent=2).encode("utf-8")
    st.download_button("Download Results (JSON)", json_bytes, "vendor_policy_validation.json", "application/json", use_container_width=True)

st.caption("For each question: we map to the best matching internal policy (with doc + page or sheet/row), judge the vendor's plain‑English answer against policy, mine supporting evidence from the vendor's uploaded docs, and recommend Acceptable / Not acceptable / Insufficient.")
