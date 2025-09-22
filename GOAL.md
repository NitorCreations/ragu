# Overall goal
I have MacBook Pro Max M4 with 128 Gb of memory. I would like to create a locally running RAG implementation,
with all models and suitable database running locally. I would have many types of materials such as
- text documents
- PDF (possibly with pictures that may have captions)
- Microsoft Word documents (possibly with pictures that may have captions)
- scanned pictures (TIFF, PNG, JPG)

I want to be able to add these source document types gradually, starting with PDF files that have image captions.
At some point in time I would like to have an UI for digesting document types, but in the beginning we can have just code.
Likewise, I would like to have "Chat GPT" style UI for end users at some point in time, but we can start with just code.

Can you describe a tech stack for this, rough implementation plan and even some code? I am flexible with implementation, 
can ny Python, Typescript, Java or Kotlin for example.

# Tech stack
Tech stack of the project will be as follows:
- llama.cpp / llama-cpp-python (GGUF quantized models, Metal/MPS accel on macOS) for the LLM
    - Perhaps llama-2-13b-chat.Q4_K_M.gguf to start with
- sentence-transformers for embeddings
- Chroma (local persistent) or Qdrant (Docker) for vector store
- PyMuPDF (fitz) + pdfplumber for text extraction
- OCRmyPDF / Tesseract and layoutparser for detecting figures & captions
- Python + FastAPI for a simple backend
- Streamlit or React for a later UI

# Road map
Phase A (days): basic local RAG with PDFs)
- Install Conda / python venv, Homebrew, Tesseract.
- Download a small GGUF LLM (e.g., Llama-2 7B GGUF or other GGUF 7B) and test inference via llama-cpp-python.
- Create ingestion script (PyMuPDF) that extracts text + images and stores chunks in Chroma using sentence-transformers.
- Add OCR fallback with ocrmypdf/Tesseract for scanned pages.
- Implement RAG retriever + LLM prompt assembly (prototype with Python).

Phase B (weeks): improve quality & multi-format ingestion
- Use layoutparser (PubLayNet models) to detect figure bounding boxes and extract captions robustly.
- Add Word (python-docx) and other file formats ingestion; store metadata (file-type, author, timestamps).
- Add deduplication / fingerprinting for images, maintain external_id style stable id across revisions.

Phase C (future): UI + features
- FastAPI backend with endpoints for ingestion / query; Streamlit / React frontend for chat UI and document browser.
- Add user management, logging, and constraints (rate-limiting).
- Optionally integrate a local search frontend (e.g., Haystack or LlamaIndex) or use LangChain/LlamaIndex wrappers for nicer pipelines.