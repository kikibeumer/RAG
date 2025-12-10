import os
import streamlit as st
import PyPDF2
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss

# =============== CONFIGURATION ==================
os.environ["MISTRAL_API_KEY"] = "XzAMIBqxnKMp1IJBZpTapghMF8UEt8H9"

@st.cache_resource
def init_llm():
    return ChatMistralAI(
        model="mistral-tiny",
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

# =============== PDF PROCESSING ==================
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    doc = Document(page_content=text, metadata={"source": "uploaded_pdf"})
    return text_splitter.split_documents([doc])

def create_vector_store(documents, embeddings):
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents)
    return vector_store

def retrieve(question, vector_store, k=3):
    docs = vector_store.similarity_search(question, k=k)
    return "\n\n".join(doc.page_content for doc in docs)

# =============== ACTIVE SUMMARIZATION LOGIC ==================
def generate_adaptive_questions(text, llm):
    """Ask the model to generate 4 meaningful questions to understand the text."""
    prompt = f"""
    You are creating questions that force deep understanding of a document.

    Based on the content below, generate 4 questions that:
    - Help identify the main ideas
    - Reveal key arguments or problems discussed
    - Ask about contradictions, trade-offs, pros/cons if relevant
    - Push the reader to search inside the text

    TEXT SAMPLE:
    {text[:2000]}  # Only first part to reduce token size

    Output ONLY the 4 questions, numbered 1-4.
    """
    return llm.invoke(prompt).content.split("\n")

def contextual_answer(question, vector_store, llm):
    context = retrieve(question, vector_store, k=4)
    prompt = f"""
    Using ONLY the following context, answer the question as clearly as possible.
    If the answer is not found, say so honestly.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:
    """
    return llm.invoke(prompt).content

def full_summary(qa_pairs, llm):
    prompt = f"""
    You are creating a summary that encourages active reading.

    Here are the questions and answers discovered:

    {qa_pairs}

    Produce a final structured summary that includes:
    - Main idea of the document
    - Most important facts
    - Arguments, solutions, or contradictions if present
    - Pros and cons or limitations when relevant

    Keep it concise but complete.
    """
    return llm.invoke(prompt).content

# =============== STREAMLIT APP ==================
def main():
    st.set_page_config(page_title="Active PDF Summary", page_icon="🧠", layout="wide")
    st.title("🧠 Active PDF Summarizer")
    st.write("Upload a document → The AI asks **4 smart questions** → Full summary is generated.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and st.button("📘 Analyze Document"):
        with st.spinner("Processing document..."):
            text = extract_text_from_pdf(uploaded_file)
            splits = split_text(text)
            embeddings = init_embeddings()
            vector_store = create_vector_store(splits, embeddings)
            llm = init_llm()

        st.success("✔ Document processed")

        with st.spinner("Generating cognitive questions..."):
            questions = generate_adaptive_questions(text, llm)
            questions = [q.strip("0123456789. ") for q in questions if q.strip()]

        st.subheader("🔍 4 Key Understanding Questions")
        for i, q in enumerate(questions, 1):
            st.markdown(f"**{i}. {q}**")

        qa_results = []
        with st.spinner("Reading and analyzing the document..."):
            for q in questions:
                ans = contextual_answer(q, vector_store, llm)
                qa_results.append(f"Q: {q}\nA: {ans}\n")

        st.subheader("📌 Answers Based on the Document")
        for qa in qa_results:
            st.write(qa)

        with st.spinner("Creating final structured summary..."):
            final = full_summary("\n".join(qa_results), llm)

        st.subheader("🧾 Final Summary")
        st.write(final)

    st.markdown("---")
    st.caption("Designed to reduce cognitive offloading and increase document comprehension.")

if __name__ == "__main__":
    main()
