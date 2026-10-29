import re
from pathlib import Path
from typing import List, Dict

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="College AI Assistant (RAG)", page_icon="🎓", layout="wide")

SUPPORTED_TYPES = ["pdf", "txt"]
MIN_DOCUMENTS = 5
RETRIEVAL_THRESHOLD = 0.08


def read_uploaded_file(uploaded_file) -> str:
    """Extract text from an uploaded PDF or TXT file."""
    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(uploaded_file)
            pages = [(page.extract_text() or "") for page in reader.pages]
            return "\n".join(pages).strip()
        return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        st.warning(f"Could not read {uploaded_file.name}: {exc}")
        return ""


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split long text into overlapping character chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_index(documents: List[Dict[str, str]]):
    """Build a TF-IDF search index from document chunks."""
    chunks = []
    for document in documents:
        for number, chunk in enumerate(split_into_chunks(document["text"]), start=1):
            chunks.append({
                "text": chunk,
                "source": document["name"],
                "chunk": number,
            })
    if not chunks:
        return None, [], None
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([item["text"] for item in chunks])
    return vectorizer, chunks, matrix


def retrieve(question: str, vectorizer, chunks, matrix, top_k: int = 4):
    """Return the most relevant chunks and their similarity scores."""
    if not question.strip() or vectorizer is None or matrix is None:
        return []
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()
    order = scores.argsort()[::-1][:top_k]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in order
        if scores[i] > 0
    ]


def make_grounded_answer(results: List[Dict], threshold: float = RETRIEVAL_THRESHOLD) -> str:
    """
    Produce an extractive, source-grounded response.
    This deliberately avoids inventing facts: weak retrieval leads to abstention.
    """
    if not results or results[0]["score"] < threshold:
        return (
            "I couldn't find enough relevant information in the uploaded college "
            "documents to answer that question. Please check the official college "
            "office or upload a relevant document."
        )

    selected = []
    seen = set()
    for result in results:
        if result["score"] < threshold:
            continue
        text = result["text"].strip()
        key = text[:180].lower()
        if key not in seen:
            selected.append(result)
            seen.add(key)
        if len(selected) == 3:
            break

    if not selected:
        return "I couldn't find this information in the uploaded documents."
    return "\n\n".join(item["text"] for item in selected)


st.title("🎓 College AI Assistant")
st.caption("A simple Retrieval-Augmented Generation (RAG) prototype that answers from uploaded college documents and shows sources.")

with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Upload at least five PDF or TXT documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="PDFs should contain selectable text. Scanned PDFs may need OCR first.",
    )
    st.info("The app indexes documents locally for this session. Uploaded files are not saved to disk by this app.")

documents = []
if uploaded_files:
    for file in uploaded_files:
        text = read_uploaded_file(file)
        if text:
            documents.append({"name": file.name, "text": text})

if len(documents) < MIN_DOCUMENTS:
    st.warning(f"Upload at least {MIN_DOCUMENTS} readable PDF/TXT documents to meet the assignment requirement. Currently readable: {len(documents)}.")
elif documents:
    st.success(f"{len(documents)} readable documents loaded.")

if documents:
    with st.spinner("Building searchable knowledge base..."):
        vectorizer, chunks, matrix = build_index(documents)
    st.write(f"**Knowledge base:** {len(documents)} documents · {len(chunks)} searchable chunks")

    st.header("2. Ask a question")
    question = st.text_input("Enter a question about the uploaded documents", placeholder="e.g., What is the attendance requirement?")
    top_k = st.slider("Number of passages to retrieve", min_value=1, max_value=6, value=4)

    if st.button("Find answer", type="primary", disabled=not question.strip()):
        results = retrieve(question, vectorizer, chunks, matrix, top_k=top_k)
        answer = make_grounded_answer(results)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Supporting sources")
        if results and results[0]["score"] >= RETRIEVAL_THRESHOLD:
            shown = 0
            for result in results:
                if result["score"] < RETRIEVAL_THRESHOLD:
                    continue
                with st.expander(f"{result['source']} · chunk {result['chunk']} · relevance {result['score']:.3f}"):
                    st.write(result["text"])
                shown += 1
            if shown == 0:
                st.write("No sufficiently relevant source passages found.")
        else:
            st.write("No supporting source found in the uploaded documents.")

    with st.expander("How this prototype works"):
        st.markdown(
            """
            1. **Load:** extracts text from PDF/TXT files.
            2. **Chunk:** splits long documents into overlapping passages.
            3. **Index:** converts passages into TF-IDF vectors.
            4. **Retrieve:** ranks passages by cosine similarity to the question.
            5. **Answer:** returns retrieved text instead of generating unsupported claims.
            6. **Cite:** displays source filename, chunk number, and relevance score.

            This is an extractive RAG prototype. A production system can replace the answer step
            with an LLM that is explicitly instructed to use only retrieved passages.
            """
        )
else:
    st.markdown("Upload five or more readable PDF/TXT files in the sidebar to begin.")
