# College AI Assistant Using RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) prototype for answering student questions from official college documents.

## Problem statement

A college needs an AI assistant that answers student questions using official documents. Answers should be supported by retrieved source text, and the assistant should abstain when the uploaded documents do not contain enough relevant information.

## Features

- Accepts PDF and TXT documents (upload at least five to satisfy the assignment).
- Extracts text from files.
- Splits long documents into overlapping chunks.
- Builds a searchable TF-IDF knowledge base.
- Retrieves relevant chunks using cosine similarity.
- Shows supporting passages, source filename, chunk number, and relevance score.
- Returns a clear fallback when relevant evidence is not found.
- Includes test cases for single-document, multi-document, and unanswerable questions.

## Project structure

```text
college_rag_assistant/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── academic_calendar.txt
│   ├── attendance_policy.txt
│   ├── library_rules.txt
│   ├── exam_policy.txt
│   └── scholarship_info.txt
└── tests/
    └── test_rag.py
```

## Setup on Windows

Open Command Prompt or the VS Code terminal inside this project folder.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

```bash
.venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will print a local address, commonly `http://localhost:8501`, and may open it in your browser.

## Try the sample documents

Upload all five `.txt` files from the `data/` folder in the app.

Example questions:
- **Single document:** What is the minimum attendance requirement?
- **Two documents:** What are the exam eligibility and library clearance requirements?
- **Not in the documents:** What is the principal's personal phone number?

The last question should trigger the fallback response instead of inventing an answer.

## Run tests

```bash
pytest -q
```

## Design overview

1. **Document loading:** PDF text is extracted with `pypdf`; TXT files are decoded as UTF-8.
2. **Chunking:** each document is split into overlapping text chunks to make retrieval more focused.
3. **Indexing:** `TfidfVectorizer` represents chunks as weighted word and phrase features.
4. **Retrieval:** cosine similarity ranks chunks against the student's question.
5. **Grounded response:** the prototype displays the most relevant retrieved passages verbatim. If the best score is below a conservative threshold, it abstains.
6. **Source display:** each retrieved passage includes its filename, chunk number, and similarity score.

## Important limitations

- This is a lightweight, extractive RAG prototype: it retrieves and presents relevant text rather than paraphrasing with a generative LLM.
- TF-IDF is lexical retrieval. Semantic embeddings can improve retrieval for questions phrased differently from the source.
- The threshold is a simple heuristic, not a calibrated confidence probability. Test it against your documents and adjust `RETRIEVAL_THRESHOLD` in `app.py`.
- Scanned PDFs without selectable text require OCR before upload.
- Sample files are fictional examples for demonstration. Replace them with authorized official college documents before using the assistant for real student queries.
- The app does not persist uploads; files are processed in the current Streamlit session.

## Possible future improvements

- Use a vector database such as FAISS or Chroma.
- Add embedding-based semantic search.
- Add an LLM answer generator that receives only retrieved passages and is instructed to cite them and abstain if evidence is missing.
- Add authentication, access controls, document versioning, and evaluation logging for production use.
