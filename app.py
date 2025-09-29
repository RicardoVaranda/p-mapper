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
import time
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
    # deterministic chunking
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

configure_settings()

# ---------------------------
# Read files → Documents (page‑aware for PDFs)
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
        h.update((src + "|" + pg + "\x1e").encode("utf-8", "ignore"))
        h.update(((d.get("text", "") or "") + "\x1e").encode("utf-8", "ignore"))
    h.update(("embed=" + (embed_model or "default")).encode("utf-8"))
    return h.hexdigest()


def build_or_load_index_persisted(
    docs: list[dict], *, prefix: str, persist_root: str
) -> IndexedCorpus:
    """Load a VectorStoreIndex from disk if present; else build and persist."""
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

    # Build & persist
    configure_settings()
    idx = VectorStoreIndex.from_documents(li_docs)
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
    if indexed.index is None:
        # naive skim over raw docs
        toks = [t for t in re.split(r"\W+", (question or "").lower()) if t]
        scores: list[tuple[int, dict]] = []
        for d in indexed.raw_docs:
            txt = (d.text or "").lower()
            score = sum(txt.count(tok) for tok in toks)
            scores.append((score, {"text": d.text, "metadata": d.metadata}))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:k]
        srcs = []
        seen = set()
        for _, d in top:
            meta = d.get("metadata", {}) or {}
            src = meta.get("source", "unknown")
            pg  = meta.get("page")
            snip = (d.get("text", "") or "")[:500]
            key = (src, pg, snip[:100])
            if key in seen: continue
            seen.add(key)
            srcs.append({"source": src, "page": pg, "snippet": snip})
        return "", srcs

    try:
        qe = indexed.index.as_query_engine(similarity_top_k=k)
        resp = qe.query(question)
        srcs = []
        seen = set()
        for n in getattr(resp, "source_nodes", [])[:k]:
            meta = n.node.metadata or {}
            src = meta.get("source", meta.get("file_name", "unknown"))
            pg  = meta.get("page")
            snip = n.node.get_content()[:500]
            key = (src, pg, snip[:100])
            if key in seen: continue
            seen.add(key)
            srcs.append({"source": src, "page": pg, "snippet": snip})
        return getattr(resp, "response", str(resp)), srcs
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
        key = (c["source"], c.get("page"), (c.get("snippet", "")[:120]))
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
# Session state containers
# ---------------------------
if "policy_index" not in st.session_state:
    st.session_state.policy_index: IndexedCorpus | None = None
if "q_items" not in st.session_state:
    st.session_state.q_items = []  # [{id,text,answer}]

# ---------------------------
# Upload global policy docs (built ahead of questions)
# ---------------------------
st.subheader("1) Upload Company Policy Documents (global)")
pol_files = st.file_uploader("Upload .pdf / .docx / .txt / .md", type=["pdf","docx","txt","md"], accept_multiple_files=True)
policy_docs = files_to_docs(pol_files, "policy") if pol_files else []

if policy_docs:
    st.success(f"Loaded {len(policy_docs)} policy doc parts (pages or files).")
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
            ev_files = st.file_uploader("Evidence docs", type=["pdf","docx","txt","md"], accept_multiple_files=True, key=f"ev_{qid}")
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
            best_policy_doc = pol_sources[0]["source"] if pol_sources else "(none)"

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
                pg = f" p.{s['page']}" if s.get('page') else ""
                sn = (s.get('snippet','') or '')
                return f"• {s['source']}{pg}: {sn[:260]}…"

            ev_display = "\n\n".join(fmt_source(s) for s in ev_cites[:6]) or "(no evidence)"

            results_rows.append({
                "#": idx,
                "Question": q,
                "Vendor Answer": a[:900] if a else "(none)",
                "Internal Policy (best match)": best_policy_doc,
                "Policy Answer": pol_answer[:800],
                "Evidence (from vendor docs)": ev_display,
                "Suggestion": suggestion,
                "Rationale": (rationale or "").strip()[:900],
                "Evidence docs count": len(ev_cites),
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

st.caption("For each question: we map to the best matching internal policy, judge the vendor's plain‑English answer against policy, mine supporting evidence from the vendor's uploaded docs, and recommend Acceptable / Not acceptable / Insufficient.")
