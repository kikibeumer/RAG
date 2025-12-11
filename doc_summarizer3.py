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

# =================== PDF =========================
def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        try:
            pg = page.extract_text()
            if pg:
                text += pg + "\n"
        except:
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
Create 4 document-specific comprehension questions based on the text.
Also generate 1 creative question that forces the student to reflect, apply, or imagine a scenario.

Concepts:
{concepts}
"""
    resp = llm.invoke(prompt)
    lines = [l.strip() for l in resp.content.split("\n") if l.strip()]
    qs = []
    for line in lines:
        q = re.sub(r"^\d+\.\s*", "", line).strip()
        if q:
            qs.append(q)
    return qs[:5]  # now includes creative question

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

def planted_error_summary(qa_pairs, llm):
    prompt = f"""
Create a structured summary of this Q&A.
Insert **exactly one incorrect statement** into the bullet points.
Mark it with: [INCORRECT]

Q&A:
{qa_pairs}
"""
    return llm.invoke(prompt).content.strip()

# =================== STREAMLIT UI ===================
def main():
    st.set_page_config(page_title="Active PDF Summarizer", page_icon="🧠", layout="wide")
    st.title("🧠 Active PDF Summarizer — Learn Better, Not Lazier")

    # ================= PRIVACY BANNER ====================
    ### NEW FEATURE
    st.markdown("""
    <div style="padding:12px;background:#ffdddd;border-left:6px solid red;font-size:18px;">
    ⚠️ <b>Privacy reminder:</b> Would you show this PDF to the whole world?
    <br>Do NOT upload anything sensitive.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PDF", type="pdf")

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

        st.success("✔ Analysis complete — First answer the questions yourself!")

    if "qs" in st.session_state:
        st.subheader("✍️ Your Turn First — Prevent AI Overreliance")

        # USERS MUST ANSWER FIRST
        ### NEW FEATURE
        user_answers = []
        for q in st.session_state["qs"]:
            ans = st.text_area(f"**Q:** {q}\nYour answer:", key=f"user_{q}")
            user_answers.append(ans)

        # Gatekeeping button
        if st.button("Reveal AI Answers"):
            if any(len(a.strip()) < 3 for a in user_answers):
                st.warning("Please answer all questions first (at least 3 characters each).")
                return

            qa_blocks = []
            st.subheader("🤖 AI Answers (Compare with Yours)")
            for q in st.session_state["qs"]:
                answer = contextual_answer(q, st.session_state["store"], st.session_state["llm"])
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A (AI):** {answer}")
                qa_blocks.append(f"Q: {q}\nA: {answer}")

            st.session_state["qa_pairs"] = "\n\n".join(qa_blocks)

        # SUMMARY WITH PLANTED ERROR
        if "qa_pairs" in st.session_state and st.button("🧾 Show Summary (1 Error Hidden!)"):
            summary = planted_error_summary(st.session_state["qa_pairs"], st.session_state["llm"])
            st.subheader("🧾 Summary (Find the Incorrect Statement!)")
            st.write(summary)
            st.session_state["summary"] = summary

        # REWRITE IN YOUR OWN WORDS (skill protection)
        if "summary" in st.session_state:
            st.subheader("✍️ Rewrite the Summary in Your Own Words")

            ### ANTI-COPY-PASTE FEATURE
            user_rewrite = st.text_area("Your rewrite (don’t copy; just explain in your tone):", key="rewrite")
            if st.button("Check Rewrite"):
                orig = st.session_state["summary"]
                if user_rewrite.strip() in orig or len(user_rewrite.strip()) < 30:
                    st.error("It looks like the text might be copied or too short. Try rewriting genuinely.")
                else:
                    st.success("Good! This looks like your own writing.")

        # TEACHER SUBMISSION (simple export)
        if st.button("📤 Export Work for Teacher"):
            st.download_button(
                "Download Work",
                data=f"Questions:\n{st.session_state['qs']}\n\nYour answers:\n{user_answers}\n\nAI Summary:\n{st.session_state.get('summary','')}",
                file_name="student_work.txt"
            )

if __name__ == "__main__":
    main()
