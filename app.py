# app.py — Policy vs Third‑Party Compliance Checker (Streamlit)
# -------------------------------------------------------------
# Updated: Third‑party docs are now uploaded **per question**.
# Company policy docs remain **global** and are reused for all questions.
# The app stays domain‑agnostic and evidence‑driven (LLM + citations).
#
# Quickstart
# ----------
# 1) pip install -r requirements.txt
# 2) streamlit run app.py
# 3) In the sidebar, set your OpenAI‑compatible API key and model.
#
# requirements.txt
# ----------------
# streamlit>=1.36.0
# llama-index>=0.10.50
# llama-index-llms-openai>=0.2.3
# llama-index-embeddings-openai>=0.2.3
# PyPDF2>=3.0.1
# docx2txt>=0.8
# tiktoken>=0.7.0
# pydantic>=2.8.2
# pandas>=2.2.2
# numpy>=1.26.4
# # optional OCR (not required here)
# pdf2image>=1.17.0
# pytesseract>=0.3.13
# pillow>=10.3.0

import io
import os
import re
import json
import uuid
import hashlib
import typing as t
from dataclasses import dataclass

import streamlit as st
import pandas as pd

from PyPDF2 import PdfReader
import docx2txt

# ------------------------- File parsing helpers ------------------------------
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}
MAX_FILE_MB = 50


def _size_ok(uploaded_file) -> bool:
    try:
        uploaded_file.seek(0, os.SEEK_END)
        bytes_len = uploaded_file.tell()
        uploaded_file.seek(0)
        return bytes_len <= MAX_FILE_MB * 1024 * 1024
    except Exception:
        return True


def _read_pdf(file: io.BytesIO) -> str:
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        return f"[PDF parse error] {e}"


def _read_docx(file: io.BytesIO) -> str:
    try:
        data = file.read()
        tmp = f"/tmp/{uuid.uuid4()}.docx"
        with open(tmp, "wb") as f:
            f.write(data)
        text = docx2txt.process(tmp) or ""
        try:
            os.remove(tmp)
        except Exception:
            pass
        return text
    except Exception as e:
        return f"[DOCX parse error] {e}"


def read_file(uploaded_file) -> str:
    name = uploaded_file.name
    lower = name.lower()
    if lower.endswith(".pdf"):
        return _read_pdf(uploaded_file)
    elif lower.endswith(".docx"):
        return _read_docx(uploaded_file)
    else:
        try:
            data = uploaded_file.read()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1", errors="ignore")
        except Exception as e:
            return f"[TXT parse error] {e}"


def files_to_docs(files: list, kind: str) -> list[dict]:
    docs: list[dict] = []
    for f in files or []:
        if not _size_ok(f):
            st.warning(f"Skipping {f.name}: exceeds {MAX_FILE_MB} MB limit")
            continue
        text = read_file(f)
        if text and text.strip() and not (text.startswith("[") and "parse error" in text.lower()):
            docs.append({"text": text, "metadata": {"source": f.name, "type": kind}})
    return docs


# ------------------------- LlamaIndex plumbing -------------------------------
LLM_AVAILABLE = True
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
except Exception:
    LLM_AVAILABLE = False


@dataclass
class IndexedCorpus:
    index: t.Any | None
    raw_docs: list[Document]


# JSON parsing helpers

def extract_json_block(text: str) -> str | None:
    s = text.strip()
    m = re.search(r"\{[\s\S]*\}$", s)
    if m:
        return m.group(0)
    m = re.search(r"\{[\s\S]*?\}", text)
    return m.group(0) if m else None


def try_parse_json(text: str) -> t.Any | None:
    blk = extract_json_block(text)
    if not blk:
        return None
    try:
        return json.loads(blk)
    except Exception:
        try:
            blk2 = re.sub(r",(\s*[}\]])", r"\1", blk)
            return json.loads(blk2)
        except Exception:
            return None


def configure_llm(api_key: str, model_name: str, base_url: str = "", embed_model: str | None = None):
    if not LLM_AVAILABLE:
        return
    Settings.llm = OpenAI(model=model_name, api_key=api_key, base_url=base_url or None, temperature=0.0, timeout=60)
    if embed_model:
        Settings.embed_model = OpenAIEmbedding(model=embed_model, api_key=api_key, base_url=base_url or None)
    else:
        Settings.embed_model = OpenAIEmbedding(api_key=api_key, base_url=base_url or None)


def build_index(docs: list[dict], api_key: str | None, model_name: str | None, base_url: str = "", embed_model: str | None = None) -> IndexedCorpus:
    li_docs: list[Document] = []
    for d in docs:
        txt = d.get("text", "")
        if not txt:
            continue
        if txt.startswith("[") and "parse error" in txt.lower():
            continue
        li_docs.append(Document(text=txt, metadata=d.get("metadata", {})))

    if LLM_AVAILABLE and api_key and model_name:
        configure_llm(api_key, model_name, base_url, embed_model)
        index = VectorStoreIndex.from_documents(li_docs)
        return IndexedCorpus(index=index, raw_docs=li_docs)
    return IndexedCorpus(index=None, raw_docs=li_docs)


@st.cache_resource(show_spinner=False)
def build_index_cached(docs_key: str, docs: list[dict], api_key: str | None, model_name: str | None, base_url: str = "", embed_model: str | None = None):
    return build_index(docs, api_key, model_name, base_url, embed_model)


def _docs_cache_key(docs: list[dict]) -> str:
    h = hashlib.sha1()
    for d in docs:
        src = d.get("metadata", {}).get("source", "")
        h.update(src.encode("utf-8"))
        h.update((d.get("text", "")[:5000]).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def retrieve_answer(indexed: IndexedCorpus, question: str, k: int = 3) -> tuple[str, list[dict]]:
    if indexed.index is not None:
        try:
            query_engine = indexed.index.as_query_engine(similarity_top_k=k)
            resp = query_engine.query(question)
            sources = []
            seen = set()
            for n in getattr(resp, "source_nodes", [])[:k]:
                meta = n.node.metadata or {}
                src = meta.get("source", meta.get("file_name", "unknown"))
                pg = meta.get("page")
                snip = n.node.get_content()[:500]
                key = (src, pg, snip[:120])
                if key in seen:
                    continue
                seen.add(key)
                sources.append({"source": src, "page": pg, "snippet": snip})
            return getattr(resp, "response", str(resp)), sources
        except Exception as e:
            return f"[LLM query error] {e}", []
    # No LLM index: naive keyword skim over raw docs
    toks = [t for t in re.split(r"\W+", question.lower()) if t]
    scores: list[tuple[int, dict]] = []
    for d in indexed.raw_docs:
        txt = (d.text or "").lower()
        score = sum(txt.count(tok) for tok in toks)
        scores.append((score, {"text": d.text, "metadata": d.metadata}))
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:k]
    snippets = []
    seen = set()
    for _, d in top:
        src = d["metadata"].get("source", "unknown")
        snip = (d.get("text", "") or "")[:500]
        key = (src, None, snip[:120])
        if key in seen:
            continue
        seen.add(key)
        snippets.append({"source": src, "page": None, "snippet": snip})
    return "", snippets


def llm_judge_match(policy_answer: str, third_answer: str, api_key: str | None, model: str | None, base_url: str = "") -> tuple[bool, str]:
    if LLM_AVAILABLE and api_key and model:
        try:
            configure_llm(api_key, model, base_url)
            prompt = (
                "You are a compliance analyst. Compare the company policy answer and the third‑party answer. "
                "Decide if the third‑party meets or exceeds the policy (binary yes/no). "
                "Return a single JSON object with keys: meets(bool), rationale(str).\n\n"
                f"CompanyPolicyAnswer:\n{policy_answer}\n\nThirdPartyAnswer:\n{third_answer}\n"
            )
            resp = Settings.llm.complete(prompt)
            text = getattr(resp, "text", str(resp)).strip()
            obj = try_parse_json(text)
            if obj is not None and isinstance(obj, dict):
                return bool(obj.get("meets", False)), obj.get("rationale", text)
            meets = bool(re.search(r"\b(yes|meets|compliant)\b", text.lower()))
            return meets, text
        except Exception as e:
            return False, f"[LLM judge error] {e}"
    return False, "LLM not available; defaulting to conservative non‑match."


# ------------------------- UI -----------------------------------------------
st.set_page_config(page_title="Policy vs Third‑Party Checker", layout="wide")
st.title("Policy vs Third‑Party Compliance Checker — Per‑Question Vendor Docs")

with st.sidebar:
    st.header("LLM Settings")
    api_key = st.text_input("OpenAI‑compatible API Key", type="password")
    model_name = st.text_input("Model (e.g., gpt-4o-mini)", value="gpt-4o-mini")
    base_url = st.text_input("OpenAI Base URL (optional)", value="")
    embed_model = st.text_input("Embedding model (optional)", value="")
    st.caption("Leave key blank to run without LLM indexing; results may be limited.")

# 1) Global Company Policy upload
st.subheader("1) Upload Company Policy Documents (global)")
policy_files = st.file_uploader(
    "Upload .pdf / .docx / .txt / .md", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
    key="policy_up"
)
policy_parsed: list[dict] = files_to_docs(policy_files, "policy") if policy_files else []
if policy_parsed:
    st.success(f"Loaded {len(policy_parsed)} policy document(s).")

st.divider()

# 2) Dynamic Questions with per-question Third‑Party uploads
st.subheader("2) Questions & Per‑Question Third‑Party Documents")

if "q_items" not in st.session_state:
    st.session_state.q_items = [{"id": str(uuid.uuid4())[:8], "text": ""}]

cols = st.columns([1,1,1,4])
if cols[0].button("➕ Add question", use_container_width=True):
    st.session_state.q_items.append({"id": str(uuid.uuid4())[:8], "text": ""})
    st.rerun()

if cols[1].button("➖ Remove last", use_container_width=True) and len(st.session_state.q_items) > 1:
    st.session_state.q_items.pop()
    st.rerun()

# Render each question block
for idx, item in enumerate(st.session_state.q_items, start=1):
    qid = item["id"]
    with st.container(border=True):
        left, right = st.columns([2,2])
        with left:
            qtext = st.text_input(f"Question {idx}", value=item.get("text",""), key=f"qtext_{qid}")
            st.session_state.q_items[idx-1]["text"] = qtext
        with right:
            st.file_uploader(
                f"Third‑Party docs for Q{idx}", type=["pdf","docx","txt","md"], accept_multiple_files=True, key=f"third_up_{qid}"
            )

st.divider()

# Run
run_btn = st.button(
    "Run Compliance Check", type="primary", use_container_width=True,
    disabled=not (policy_parsed and any(it.get("text","" ).strip() for it in st.session_state.q_items))
)

results_rows: list[dict] = []

if run_btn:
    with st.spinner("Building indices and analyzing…"):
        # Build global policy index (cached)
        policy_key = "pol:" + _docs_cache_key(policy_parsed)
        policy_index = build_index_cached(policy_key, policy_parsed, api_key=api_key, model_name=model_name, base_url=base_url, embed_model=embed_model or None)

        for idx, item in enumerate(st.session_state.q_items, start=1):
            q = (item.get("text") or "").strip()
            if not q:
                continue

            # Build per‑question third‑party index
            third_files = st.session_state.get(f"third_up_{item['id']}") or []
            third_docs = files_to_docs(third_files, "third")
            third_key = f"thr:{item['id']}:{_docs_cache_key(third_docs)}"
            third_index = build_index_cached(third_key, third_docs, api_key=api_key, model_name=model_name, base_url=base_url, embed_model=embed_model or None)

            # Retrieve
            pol_answer, pol_sources = retrieve_answer(policy_index, q, k=3)
            thr_answer, thr_sources = retrieve_answer(third_index, q, k=3)

            mapped_policy_doc = pol_sources[0]["source"] if pol_sources else "(none)"
            mapped_third_doc = thr_sources[0]["source"] if thr_sources else "(none)"

            # Judge (LLM JSON; conservative fallback otherwise)
            meets, rationale = llm_judge_match(pol_answer, thr_answer, api_key=api_key, model=model_name, base_url=base_url)

            def fmt_source(s):
                pg = f" p.{s['page']}" if s.get('page') else ""
                return f"• {s['source']}{pg}: {s['snippet'][:260]}…"

            results_rows.append({
                "#": idx,
                "Question": q,
                "Pass": "✅ PASS" if meets else "❌ FAIL",
                "Policy Doc": mapped_policy_doc,
                "Policy Answer": pol_answer[:900],
                "Policy Evidence": "\n\n".join(fmt_source(s) for s in pol_sources) or "(no evidence)",
                "3P Doc": mapped_third_doc,
                "Third‑Party Answer": thr_answer[:900],
                "Third‑Party Evidence": "\n\n".join(fmt_source(s) for s in thr_sources) or "(no evidence)",
                "3P Docs Count": len(third_docs),
                "Rationale": rationale[:1200],
            })

    st.success("Analysis complete.")

if not results_rows:
    st.stop()

# Results table
df = pd.DataFrame(results_rows)

m1, m2, m3 = st.columns(3)
m1.metric("Questions", len(df))
m2.metric("PASS", int((df["Pass"] == "✅ PASS").sum()))
m3.metric("FAIL", int((df["Pass"] == "❌ FAIL").sum()))

st.dataframe(df, use_container_width=True, height=520)

# Downloads
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Results (CSV)",
    data=csv,
    file_name="compliance_results.csv",
    mime="text/csv",
    use_container_width=True,
)
json_bytes = json.dumps(results_rows, indent=2).encode("utf-8")
st.download_button(
    label="Download Results (JSON)",
    data=json_bytes,
    file_name="compliance_results.json",
    mime="application/json",
    use_container_width=True,
)

st.caption("Global policy corpus is reused for all questions. Third‑party docs are uploaded per question and judged only against that question. Domain‑agnostic, evidence‑driven flow (LLM + citations).")
