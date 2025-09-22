import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

CHROMA_DIR = "chroma_db"
DIRS_TO_PROCESS = [
    "/Users/rikusarlin/Documents/architecture"
]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800

embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("docs")


# ---------- Helpers ----------
def extract_pdf(page_no, doc_path):
    doc = fitz.open(doc_path)
    page = doc[page_no]
    text = page.get_text("text")
    doc.close()
    return text.strip()

def find_captions_heuristic(page_text):
    if not page_text:
        return []
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln.lower().startswith("figure") or ln.lower().startswith("fig.")]

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]


# ---------- Ingest PDFs ----------
def ingest_pdfs(pdf_dirs):
    # Load already ingested sources
    existing_sources = set()
    try:
        all_docs = collection.get(include=["metadatas"])
        for meta_list in all_docs["metadatas"]:
            for meta in meta_list:
                if "source" in meta:
                    existing_sources.add(meta["source"])
    except Exception:
        existing_sources = set()

    for dir_path in pdf_dirs:
        pdf_path = Path(dir_path)
        if not pdf_path.exists() or not pdf_path.is_dir():
            print(f"Skipping invalid directory: {dir_path}")
            continue

        for pdf_file in pdf_path.glob("*.pdf"):
            if pdf_file.name in existing_sources:
                print(f"Skipping already-ingested file: {pdf_file.name}")
                continue

            print(f"Processing {pdf_file.name}...")
            doc = fitz.open(pdf_file)
            for p in range(doc.page_count):
                text = extract_pdf(p, pdf_file)
                captions = find_captions_heuristic(text)
                chunks = chunk_text(text or " ")
                for idx, chunk in enumerate(chunks):
                    metadata = {
                        "source": str(pdf_file.name),
                        "page": int(p),
                        "chunk_index": int(idx),
                        "captions": "; ".join(captions) if captions else "",
                    }
                    emb = embedder.encode(chunk).tolist()
                    uid = f"{pdf_file.stem}_p{p}_c{idx}"
                    collection.add(
                        documents=[chunk],
                        metadatas=[metadata],
                        ids=[uid],
                        embeddings=[emb]
                    )
            doc.close()

    print("Ingest complete.")


if __name__ == "__main__":
    ingest_pdfs(DIRS_TO_PROCESS)
