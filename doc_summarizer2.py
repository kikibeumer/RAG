import os
import re
import streamlit as st
import PyPDF2
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss

# =================== CONFIG ======================
os.environ["MISTRAL_API_KEY"] = "XzAMIBqxnKMp1IJBZpTapghMF8UEt8H9"

@st.cache_resource
def init_llm():
    return ChatMistralAI(
        model="mistral-small",
        temperature=0,
        max_retries=2,
    )

@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# =================== UTILITIES ===================
def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]{6,}", " ", text)
    return text.strip()

def proofread_and_clean_answer(raw_answer: str, llm) -> str:
    alpha_ratio = len(re.findall(r"[A-Za-z0-9]", raw_answer)) / max(1, len(raw_answer))
    if not raw_answer.strip() or alpha_ratio < 0.05:
        return "The provided context does not contain an intelligible answer to this question."

    prompt = f"""
You are a careful editor. Clean, proofread, and clarify the assistant answer. 
Keep only what comes from the original answer, do not invent new facts.

ANSWER TO PROOFREAD:
\"\"\"{raw_answer}\"\"\""""
    cleaned = llm.invoke(prompt).content.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

# =================== PDF PROCESSING ===================
def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        try:
            pg = page.extract_text()
            if pg:
                text += pg + "\n"
        except Exception:
            continue
    return clean_extracted_text(text)

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    doc = Document(page_content=text, metadata={"source": "uploaded_pdf"})
    return splitter.split_documents([doc])

def create_vector_store(documents, embeddings):
    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    store.add_documents(documents)
    return store

def retrieve(question: str, vector_store, k: int = 4) -> str:
    docs = vector_store.similarity_search(question, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)

# =================== QUESTION GENERATION ===================
def extract_concepts(text: str, llm, max_chars: int = 3000) -> str:
    snippet = text[:max_chars]
    prompt = f"""
Extract up to 12 short document-specific topics or terms as bullets.

EXCERPT:
\"\"\"{snippet}\"\"\""""
    return llm.invoke(prompt).content.strip()

def generate_adaptive_questions(text: str, llm) -> list:
    concepts = extract_concepts(text, llm)
    prompt = f"""
Create 4 document-specific questions using concepts below:
{concepts}"""
    resp = llm.invoke(prompt)
    lines = [l.strip() for l in resp.content.split("\n") if l.strip()]
    qs = []
    for line in lines:
        q = re.sub(r"^\d+\.\s*", "", line).strip()
        if q:
            qs.append(q)
    return qs[:4]

# =================== QA & SUMMARY ===================
def contextual_answer(question: str, vector_store, llm) -> str:
    context = retrieve(question, vector_store)
    prompt = f"""
Answer using ONLY this context:
{context}

Q: {question}
If not present: "The information is not found in the provided context."
"""
    raw = llm.invoke(prompt).content.strip()
    return proofread_and_clean_answer(raw, llm)

def full_summary(qa_pairs, llm) -> str:
    prompt = f"""
Create structured summary:
- 1 sentence main idea
- 3-6 bullet key points
- note limitations
- 1 actionable implication

Q&A:
{qa_pairs}
"""
    return llm.invoke(prompt).content.strip()

# =================== STREAMLIT UI ===================
def main():
    st.set_page_config(page_title="Active PDF Summarizer", page_icon="🧠", layout="wide")
    st.title("🧠 Active PDF Summarizer — Reveal Answers on Demand")

    uploaded = st.file_uploader("Upload PDF", type="pdf")
    show_excerpts = st.checkbox("Include retrieved excerpts in answers", value=False)

    if uploaded and st.button("📘 Analyze Document"):
        with st.spinner("Processing..."):
            text = extract_text_from_pdf(uploaded)
            chunks = split_text(text)
            embeddings = init_embeddings()
            store = create_vector_store(chunks, embeddings)
            llm = init_llm()

            questions = generate_adaptive_questions(text, llm)
            st.session_state["qs"] = questions
            st.session_state["store"] = store
            st.session_state["llm"] = llm
            st.session_state["show_excerpts"] = show_excerpts

        st.success("✔ Analysis complete — You may now reveal answers below!")

    if "qs" in st.session_state:
        st.subheader("🔍 Generated Questions")
        for i, q in enumerate(st.session_state["qs"], 1):
            st.markdown(f"**{i}. {q}**")

        # --- Button to display answers ---
        if st.button("📥 Show Answers & Excerpts"):
            qa_blocks = []
            for q in st.session_state["qs"]:
                answer = contextual_answer(q, st.session_state["store"], st.session_state["llm"])
                st.markdown(f"**Q:** {q}")

                if st.session_state["show_excerpts"]:
                    excerpt_full = retrieve(q, st.session_state["store"], k=1).strip()[:300]
                    st.markdown(f"> **Excerpt:** {excerpt_full}...")

                st.markdown(f"**A:** {answer}")
                qa_blocks.append(f"Q: {q}\nA: {answer}")

            st.session_state["qa_pairs"] = "\n\n".join(qa_blocks)

        # --- Button for summary ---
        if "qa_pairs" in st.session_state and st.button("🧾 Show Final Summary"):
            final = full_summary(st.session_state["qa_pairs"], st.session_state["llm"])
            st.subheader("🧾 Final Structured Summary")
            st.write(final)

    st.markdown("---")
    st.caption("Click buttons to reveal results — cleaner interface & more control!")

if __name__ == "__main__":
    main()
