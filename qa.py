import os
import io
import re
import json
from typing import List, Tuple, Dict

import streamlit as st
import docx2txt
from PyPDF2 import PdfReader

# ---- LlamaIndex (<=0.12.53) ----
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


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
    top_k = st.slider("Top-K per corpus", 2, 15, 5, 1)
    temperature = st.slider("Draft temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("---")
    st.caption("Evidence-driven flow only. No deterministic validations.")

# ---------------------------
# Configure LlamaIndex defaults
# ---------------------------
def configure_settings():
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url

    Settings.llm = OpenAI(model=chat_model, temperature=temperature, api_key=api_key or None, **kwargs)
    Settings.embed_model = OpenAIEmbedding(model=embed_model, api_key=api_key or None, base_url=base_url or None)
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

configure_settings()

# ---------------------------
# File readers
# ---------------------------
def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    out = []
    for p in reader.pages:
        try:
            out.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(out)

def read_docx(file_bytes: bytes) -> str:
    tmp_path = "tmp_doc.docx"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    text = docx2txt.process(tmp_path)
    return text

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def load_any(file) -> Tuple[str, str]:
    name = file.name
    data = file.read()
    file.seek(0)
    lower = name.lower()
    if lower.endswith(".pdf"):
        return name, read_pdf(data)
    elif lower.endswith(".docx"):
        return name, read_docx(data)
    elif lower.endswith(".txt"):
        return name, read_txt(data)
    else:
        return name, ""

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

# ---------------------------
# Uploaders
# ---------------------------
policy_files = st.file_uploader(
    "Upload Company Policy documents (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"], accept_multiple_files=True
)
email_files = st.file_uploader(
    "Upload Prior Regulator Response emails (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"], accept_multiple_files=True
)

def add_to_docs(files, target_list):
    for f in files:
        name, text = load_any(f)
        if not text.strip():
            st.warning(f"Unsupported or empty file: {name}")
            continue
        target_list.append(Document(text=text, metadata={"source": name}))

if policy_files:
    add_to_docs(policy_files, st.session_state.policy_docs)
if email_files:
    add_to_docs(email_files, st.session_state.email_docs)

# ---------------------------
# Build indexes
# ---------------------------
if st.button("🔧 Build / Rebuild Indexes"):
    if not st.session_state.policy_docs and not st.session_state.email_docs:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Building indexes…"):
            configure_settings()
            st.session_state.policy_index = (
                VectorStoreIndex.from_documents(st.session_state.policy_docs)
                if st.session_state.policy_docs else None
            )
            st.session_state.email_index = (
                VectorStoreIndex.from_documents(st.session_state.email_docs)
                if st.session_state.email_docs else None
            )
            st.session_state.vector_ready = True
        st.success("Indexes built.")

# ---------------------------
# Prompts
# ---------------------------
DRAFT_PROMPT = """You are a compliance assistant drafting replies to financial regulators.
ONLY use the EVIDENCE provided. If evidence is insufficient, state this clearly.

Question:
{question}

Evidence:
{evidence}

Constraints:
- Draft 1–3 short paragraphs
- Formal, neutral tone
- Cite with [Doc: <file>, Node <n>] for support
"""

JUDGE_PROMPT = """Judge the draft response:

1) Is it supported by evidence?
2) Best primary source? ("Policy", "PriorEmail", "Blended", "Insufficient")
3) Confidence 0.0–1.0
4) Keep strongest citations

Return JSON:
{{
 "source_type": "...",
 "confidence": 0.0,
 "keep_citations": ["[Doc: ...]"]
}}
"""

# ---------------------------
# Retrieval helpers
# ---------------------------
def make_retriever(index, k: int):
    if not index:
        return None
    return VectorIndexRetriever(index=index, similarity_top_k=k)

def nodes_to_evidence(nodes) -> Tuple[str, Dict[str, str]]:
    evidence_lines = []
    tag_to_file = {}
    for i, n in enumerate(nodes):
        src = (n.metadata or {}).get("source", "unknown")
        tag = f"[Doc: {src}, Node {i}]"
        text = re.sub(r"\s+", " ", n.get_text()).strip()
        evidence_lines.append(f"{tag} {text}")
        tag_to_file[tag] = src
    return "\n".join(evidence_lines), tag_to_file

def retrieve_across(q: str, k: int):
    nodes = []
    if st.session_state.policy_index:
        r = make_retriever(st.session_state.policy_index, k)
        nodes += r.retrieve(q)
    if st.session_state.email_index:
        r = make_retriever(st.session_state.email_index, k)
        nodes += r.retrieve(q)
    return [n.node for n in nodes]

# ---------------------------
# Dynamic Questions
# ---------------------------
st.markdown("### Ask Regulator Questions")
qs_text = st.text_area("Paste questions (one per line):", height=120)

if st.button("🧠 Generate Responses"):
    if not st.session_state.vector_ready:
        st.error("Please build indexes first.")
    else:
        configure_settings()
        llm = Settings.llm

        questions = [q.strip() for q in qs_text.split("\n") if q.strip()]
        if not questions:
            st.warning("Please enter at least one question.")
        else:
            results = []
            for q in questions:
                nodes = retrieve_across(q, top_k)
                ev_str, tag2file = nodes_to_evidence(nodes) if nodes else ("(no evidence found)", {})

                draft = llm.complete(DRAFT_PROMPT.format(question=q, evidence=ev_str)).text
                judge_raw = llm.complete(JUDGE_PROMPT.format(question=q, draft=draft, evidence=ev_str)).text

                source_type, confidence, keep = "Insufficient", 0.0, []
                try:
                    j = json.loads(judge_raw)
                    source_type = j.get("source_type", "Insufficient")
                    confidence = float(j.get("confidence", 0.0))
                    keep = [str(x) for x in j.get("keep_citations", [])]
                except Exception:
                    pass

                kept_docs, ev_snips = [], []
                for line in ev_str.splitlines():
                    if any(line.startswith(tag) for tag in keep) or not keep:
                        ev_snips.append(line)
                        m = re.search(r"\[Doc: (.*?), Node \d+\]", line)
                        if m:
                            kept_docs.append(m.group(1))

                results.append({
                    "question": q,
                    "suggested_response": draft,
                    "source_type": source_type,
                    "confidence": round(confidence, 3),
                    "source_docs": list(dict.fromkeys(kept_docs))[:5],
                    "evidence_snippets": ev_snips[:5]
                })

            # Render
            for r in results:
                with st.expander(f"❓ {r['question']}"):
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.subheader("Suggested Response")
                        st.write(r["suggested_response"])
                    with col2:
                        st.subheader("Assessment")
                        st.write(f"**Source Type:** {r['source_type']}")
                        st.write(f"**Confidence:** {r['confidence']}")
                        st.write("**Source Docs:**")
                        for d in r["source_docs"]:
                            st.write(f"- {d}")
                    st.subheader("Evidence Snippets")
                    for sn in r["evidence_snippets"]:
                        st.code(sn, language="text")
