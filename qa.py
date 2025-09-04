# app.py — Regulatory Response Assistant (Streamlit + LlamaIndex <= 0.12.53)
# -------------------------------------------------------------------------
# What this app does
# - Upload Company Policy docs and Prior Regulator Response emails (PDF/DOCX/TXT)
# - Dynamically add questions (via text area or table editor)
# - Retrieve evidence from both corpora
# - Draft a regulator-ready response that cites evidence
# - Judge & classify Source Type (Policy / PriorEmail / Blended / Insufficient)
# - Show confidence and evidence snippets
#
# Key updates in this version:
# - Strict JSON-only judge prompt
# - Robust JSON extraction (handles stray text / code fences)
# - Fallback inference for source_type + confidence if judge JSON fails
# - Works with your pinned requirements.txt stack

import os
import io
import re
import json
from typing import List, Tuple, Dict, Optional

import streamlit as st
import docx2txt
from PyPDF2 import PdfReader

# ---- LlamaIndex (<=0.12.53) ----
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore


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

    show_debug = st.toggle("Show Judge Raw Output (debug)", value=False)
    st.markdown("---")
    st.caption("Evidence-driven flow only. No deterministic or hard-coded validations.")


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
    text = docx2txt.process(tmp_path) or ""
    try:
        os.remove(tmp_path)
    except Exception:
        pass
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

# sets to identify which filenames belong to which corpus (used for fallback)
if "policy_files_set" not in st.session_state:
    st.session_state.policy_files_set = set()
if "email_files_set" not in st.session_state:
    st.session_state.email_files_set = set()


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
        name, text = load_any(f)
        if not text.strip():
            st.warning(f"Unsupported or empty file: {name}")
            continue
        target_list.append(Document(text=text, metadata={"source": name}))
        if corpus_name == "policy":
            st.session_state.policy_files_set.add(name)
        elif corpus_name == "email":
            st.session_state.email_files_set.add(name)

if policy_files:
    add_to_docs(policy_files, st.session_state.policy_docs, "policy")
if email_files:
    add_to_docs(email_files, st.session_state.email_docs, "email")

colA, colB = st.columns(2)
with colA:
    if st.session_state.policy_docs:
        st.success(f"Policy docs loaded: {len(st.session_state.policy_docs)}")
with colB:
    if st.session_state.email_docs:
        st.success(f"Prior response docs loaded: {len(st.session_state.email_docs)}")


# ---------------------------
# Build / Rebuild Indexes
# ---------------------------
if st.button("🔧 Build / Rebuild Indexes"):
    if not st.session_state.policy_docs and not st.session_state.email_docs:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Building LlamaIndex vector indexes..."):
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
# Prompts (strict JSON judge)
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
    # Use kept tags if available; else use all
    use_tags = kept_tags or list(tag_to_score.keys())
    scores = [tag_to_score.get(t, 0.0) for t in use_tags]
    if not scores:
        return 0.0
    clipped = [min(1.0, max(0.0, float(s))) for s in scores]
    avg = sum(clipped) / len(clipped)
    # Slight clamp to avoid absolute 0 or 1 in UI
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

def nodes_to_evidence(nodes_ws: List[NodeWithScore]) -> Tuple[str, Dict[str, str], Dict[str, float]]:
    """
    Returns:
      evidence_text: str
      tag_to_file: {tag -> filename}
      tag_to_score: {tag -> similarity_score}
    """
    lines, tag_to_file, tag_to_score = [], {}, {}
    for i, nws in enumerate(nodes_ws):
        node = nws.node
        src = (node.metadata or {}).get("source", "unknown")
        tag = f"[Doc: {src}, Node {i}]"
        text = re.sub(r"\s+", " ", node.get_text()).strip()
        lines.append(f"{tag} {text}")
        tag_to_file[tag] = src
        try:
            tag_to_score[tag] = float(nws.score or 0.0)
        except Exception:
            tag_to_score[tag] = 0.0
    return "\n".join(lines), tag_to_file, tag_to_score


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
            for q in questions:
                with st.expander(f"❓ {q}", expanded=False):
                    # Retrieve (with scores)
                    nodes_ws = retrieve_across(q, top_k)
                    if nodes_ws:
                        ev_str, tag2file, tag2score = nodes_to_evidence(nodes_ws)
                    else:
                        ev_str, tag2file, tag2score = "(no evidence found)", {}, {}

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
                                m = re.search(r"\[Doc: (.*?), Node \d+\]", line)
                                if m:
                                    kept_docs.append(m.group(1))
                    kept_docs = list(dict.fromkeys(kept_docs))[:5]

                    # Render UI
                    left, right = st.columns([2, 1])
                    with left:
                        st.subheader("Suggested Response")
                        st.write(draft)
                    with right:
                        st.subheader("Assessment")
                        st.write(f"**Source Type:** {source_type}")
                        st.write(f"**Confidence:** {round(confidence, 3)}")
                        st.write("**Source Docs:**")
                        if kept_docs:
                            for d in kept_docs:
                                st.write(f"- {d}")
                        else:
                            st.write("- (none)")

                    st.subheader("Evidence Snippets")
                    if ev_snips:
                        for sn in ev_snips[:6]:
                            st.code(sn, language="text")
                    else:
                        st.write("_No supporting evidence found._")

                    if show_debug:
                        st.subheader("Debug: Judge Raw Output")
                        st.code(judge_raw[:4000], language="json")
