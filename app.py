# app.py — Policy vs Third‑Party Compliance Checker (Streamlit)
# -------------------------------------------------------------
# What this app does
# 1) Let users upload Company Policy documents
# 2) Let users upload Third‑Party documents
# 3) Let users paste a list of questions
# Then, for each question:
#   a) Map it to the most relevant policy doc
#   b) Extract the answer from the policy
#   c) Extract the answer from the third‑party docs
#   d) Judge whether the third‑party matches or exceeds the policy
#
# Works with an LLM (via LlamaIndex + OpenAI-compatible API). Includes
# a lightweight offline rules engine for common password policy checks
# as a deterministic fallback.
#
# Quickstart
# ----------
# 1) Create a virtualenv and install requirements:
#    pip install -r requirements.txt
#
# 2) Run the app:
#    streamlit run app.py
#
# 3) Provide your OpenAI-compatible API key and model in the sidebar.
#    (E.g., OpenAI gpt-4o-mini, or any hosted compatible endpoint.)
#
# requirements.txt (put this in a file with the same name)
# -------------------------------------------------------
# streamlit>=1.36.0
# llama-index>=0.10.50
# llama-index-llms-openai>=0.2.3
# llama-index-embeddings-openai>=0.2.3
# PyPDF2>=3.0.1
# python-docx>=1.1.2
# docx2txt>=0.8
# unstructured>=0.14.10
# tiktoken>=0.7.0
# pydantic>=2.8.2
# pandas>=2.2.2
# numpy>=1.26.4
#
# -------------------------------------------------------------

import io
import os
import re
import json
import uuid
import time
import base64
import typing as t
from dataclasses import dataclass

import streamlit as st
import pandas as pd

# ---- File parsing helpers ---------------------------------------------------
from PyPDF2 import PdfReader
import docx2txt


SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}


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
        # docx2txt expects a path or file-like; write to a temp
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
        # treat everything else as text
        try:
            data = uploaded_file.read()
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1", errors="ignore")
        except Exception as e:
            return f"[TXT parse error] {e}"


# ---- LlamaIndex plumbing ----------------------------------------------------
LLM_AVAILABLE = True
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
except Exception as e:  # noqa: E722
    LLM_AVAILABLE = False


@dataclass
class IndexedCorpus:
    index: t.Any | None
    raw_docs: list[Document]


def build_index(docs: list[dict], api_key: str | None | None, model_name: str | None) -> IndexedCorpus:
    """docs: list of {text, metadata}.
    metadata expects keys: {source: filename, type: 'policy'|'third'}
    """
    li_docs: list[Document] = []
    for d in docs:
        li_docs.append(Document(text=d["text"], metadata=d.get("metadata", {})))

    if LLM_AVAILABLE and api_key and model_name:
        Settings.llm = OpenAI(model=model_name, api_key=api_key)
        Settings.embed_model = OpenAIEmbedding(api_key=api_key)
        index = VectorStoreIndex.from_documents(li_docs)
        return IndexedCorpus(index=index, raw_docs=li_docs)
    else:
        # No LLM: store raw docs so we can still do regex/rules on them
        return IndexedCorpus(index=None, raw_docs=li_docs)


def retrieve_answer(indexed: IndexedCorpus, question: str, k: int = 3) -> tuple[str, list[dict]]:
    """Run a retrieval QA against the index.
    Returns (answer_text, sources[{source, snippet}]).
    If no LLM available, returns empty answer and top k text snippets via naive search.
    """
    if indexed.index is not None:
        try:
            query_engine = indexed.index.as_query_engine(similarity_top_k=k)
            resp = query_engine.query(question)
            # LlamaIndex response often has source_nodes with metadata
            sources = []
            for n in getattr(resp, "source_nodes", [])[:k]:
                meta = n.node.metadata or {}
                sources.append({
                    "source": meta.get("source", meta.get("file_name", "unknown")),
                    "snippet": n.node.get_content()[:500],
                })
            return str(resp), sources
        except Exception as e:
            return f"[LLM query error] {e}", []
    else:
        # naive BM25-less search: pick top k docs containing most query tokens
        toks = [t for t in re.split(r"\W+", question.lower()) if t]
        scores: list[tuple[int, dict]] = []  # (score, doc)
        for d in indexed.raw_docs:
            txt = d.text.lower()
            score = sum(txt.count(tok) for tok in toks)
            scores.append((score, {"text": d.text, "metadata": d.metadata}))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:k]
        snippets = []
        for _, d in top:
            txt = d["text"]
            snippets.append({
                "source": d["metadata"].get("source", "unknown"),
                "snippet": txt[:500],
            })
        return "", snippets


# ---- Judging logic ----------------------------------------------------------

def llm_judge_match(policy_answer: str, third_answer: str, api_key: str | None, model: str | None) -> tuple[bool, str]:
    """Ask the LLM to decide if third‑party meets or exceeds policy, with rationale.
    If LLM not available, default to conservative False.
    """
    if LLM_AVAILABLE and api_key and model:
        try:
            Settings.llm = OpenAI(model=model, api_key=api_key)
            prompt = (
                "You are a compliance analyst. Compare the company policy answer and the third‑party answer. "
                "Decide if the third‑party meets or exceeds the policy (binary yes/no). "
                "Return a single JSON object with keys: meets(bool), rationale(str).\n\n"
                f"CompanyPolicyAnswer:\n{policy_answer}\n\nThirdPartyAnswer:\n{third_answer}\n"
            )
            from llama_index.core.prompts import Prompt
            qe = Settings.llm
            resp = qe.complete(prompt)
            text = str(resp).strip()
            # Try to extract JSON
            try:
                obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                return bool(obj.get("meets", False)), obj.get("rationale", text)
            except Exception:
                # Heuristic if no JSON
                meets = bool(re.search(r"\b(yes|meets|compliant)\b", text.lower()))
                return meets, text
        except Exception as e:
            return False, f"[LLM judge error] {e}"
    return False, "LLM not available; defaulting to conservative non‑match."


# ---- Streamlit UI -----------------------------------------------------------
st.set_page_config(page_title="Policy vs Third‑Party Checker", layout="wide")
st.title("Policy vs Third‑Party Compliance Checker")

with st.sidebar:
    st.header("LLM Settings")
    base_url = st.text_input("OpenAI‑compatible base url", type="default", value="")
    api_key = st.text_input("OpenAI‑compatible API Key", type="password", value="")
    model_name = st.text_input("Model (e.g., gpt-4o-mini)", value="gpt-4o-mini")
    st.caption("Leave blank to run in rules/regex mode only (limited features).")

st.markdown("""
Upload your **Company Policies**, **Third‑Party docs**, and paste a **question list**.\
For each question, the app will map to the most relevant policy, extract answers,\
check third‑party answers, and decide whether the third‑party meets or exceeds the policy.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1) Upload Company Policy Documents")
    policy_files = st.file_uploader(
        "Upload .pdf / .docx / .txt / .md", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
        key="policy_up"
    )
    policy_parsed: list[dict] = []
    if policy_files:
        for f in policy_files:
            txt = read_file(f)
            policy_parsed.append({"text": txt, "metadata": {"source": f.name, "type": "policy"}})
        st.success(f"Loaded {len(policy_parsed)} policy document(s).")
        with st.expander("Preview first 1000 chars / doc"):
            for p in policy_parsed:
                st.markdown(f"**{p['metadata']['source']}**\n\n" + "````\n" + p["text"][:1000] + "\n````")

with col2:
    st.subheader("2) Upload Third‑Party Documents")
    third_files = st.file_uploader(
        "Upload .pdf / .docx / .txt / .md", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
        key="third_up"
    )
    third_parsed: list[dict] = []
    if third_files:
        for f in third_files:
            txt = read_file(f)
            third_parsed.append({"text": txt, "metadata": {"source": f.name, "type": "third"}})
        st.success(f"Loaded {len(third_parsed)} third‑party document(s).")
        with st.expander("Preview first 1000 chars / doc"):
            for p in third_parsed:
                st.markdown(f"**{p['metadata']['source']}**\n\n" + "````\n" + p["text"][:1000] + "\n````")

st.subheader("3) Questions")
questions_text = st.text_area(
    "Paste one question per line",
    value=(
        "Does the password policy meet the criteria of our company?\n"
        "Is the password complexity equal or greater than 12 characters?"
    ),
    height=120,
)
questions: list[str] = [q.strip() for q in questions_text.splitlines() if q.strip()]

st.divider()

run_btn = st.button("Run Compliance Check", type="primary", use_container_width=True,
                    disabled=not (policy_parsed and third_parsed and questions))

if run_btn:
    with st.spinner("Building indices and analyzing…"):
        # Build indices
        policy_index = build_index(policy_parsed, api_key=api_key, model_name=model_name)
        third_index = build_index(third_parsed, api_key=api_key, model_name=model_name)

        results_rows = []
        for q in questions:
            # a) Map to most relevant policy doc + policy answer
            pol_answer, pol_sources = retrieve_answer(policy_index, q, k=3)
            mapped_policy_doc = pol_sources[0]["source"] if pol_sources else "(unknown)"

            # b) Third‑party answer
            third_answer, third_sources = retrieve_answer(third_index, q, k=3)
            mapped_third_doc = third_sources[0]["source"] if third_sources else "(unknown)"
            # c) Judge — domain-agnostic LLM verdict only
            meets, rationale = llm_judge_match(pol_answer, third_answer, api_key=api_key, model=model_name)

            results_rows.append(
                {
                    "Question": q,
                    "Pass": "✅ PASS" if meets else "❌ FAIL",
                    "Mapped Policy Doc": mapped_policy_doc,
                    "Policy Answer": pol_answer[:1200],
                    "Policy Evidence": "\n\n".join(f"• {s['source']}: {s['snippet'][:260]}…" for s in pol_sources) or "(no evidence)",
                    "Mapped Third‑Party Doc": mapped_third_doc,
                    "Third‑Party Answer": third_answer[:1200],
                    "Third‑Party Evidence": "\n\n".join(f"• {s['source']}: {s['snippet'][:260]}…" for s in third_sources) or "(no evidence)",
                    "Rationale": rationale[:1500],
                }
            )

    st.success("Analysis complete.")

    df = pd.DataFrame(results_rows)

    # Summary counts
    total = len(df)
    passed = int((df["Pass"] == "✅ PASS").sum())
    failed = total - passed

    m1, m2, m3 = st.columns(3)
    m1.metric("Questions", total)
    m2.metric("Pass", passed)
    m3.metric("Fail", failed)

    st.dataframe(df, use_container_width=True, height=500)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="compliance_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption("This app is domain-agnostic. It uses an evidence-driven rubric flow (LLM + citations) for any question.")
# -------------------- End of app.py --------------------
