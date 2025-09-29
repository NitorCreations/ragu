import pymupdf
import pytesseract
from PIL import Image
import io
from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import unicodedata
import pycountry

CHUNK_SIZE = 800
EMBED_MODEL_NAME = "/Users/rikusarlin/models/e5-base"                       # Change this to your local dir

# ---------- Helpers ----------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())

def to_iso639_3(code: str) -> str:
    """Convert 2-letter ISO code to 3-letter ISO 639-2 for Tesseract."""
    try:
        lang = pycountry.languages.get(alpha_2=code)
        if lang and hasattr(lang, "bibliographic"):
            return lang.bibliographic.lower()
        if lang and hasattr(lang, "alpha_3"):
            return lang.alpha_3.lower()
    except Exception:
        pass
    return code.lower()

def int_to_roman(num):
    val = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    syms = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num

def logical_page_number(pdf_path, page_no):
    """Compute logical page number using PDF PageLabels."""
    doc = pymupdf.open(pdf_path)
    try:
        page_labels = doc.get_page_labels()
        if not page_labels:
            return page_no + 1
        label_info = None
        for label_dict in reversed(page_labels):
            if page_no >= label_dict['startpage']:
                label_info = label_dict
                break
        if label_info:
            first = label_info.get("firstpagenum", 1)
            start = label_info.get("startpage", 0)
            style = label_info.get("style")
            prefix = label_info.get("prefix", "")
            num = first + (page_no - start)
            if style == "r":
                page_label = int_to_roman(num).lower()
            elif style == "R":
                page_label = int_to_roman(num).upper()
            else:
                page_label = str(num)
            return prefix + page_label
        return page_no + 1
    finally:
        doc.close()

def extract_pdf(page_no, doc_path, lang='en'):
    lang_tess = to_iso639_3(lang)
    doc = pymupdf.open(doc_path)
    page = doc[page_no]
    text = page.get_text("text").strip()
    if text:
        doc.close()
        return text, False
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img, lang=lang_tess).strip()
    doc.close()
    return text, True

def find_captions_heuristic(page_text):
    if not page_text:
        return []
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln.lower().startswith("figure") or ln.lower().startswith("fig.")]

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

# ---------- PDF Ingestion ----------
def ingest_pdfs(pdf_dir, collection_name, lang='en'):
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    collection_name_lang = f"{collection_name}_{lang}"
    chroma_dir = f"chroma_{collection_name_lang}"
    client = PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(collection_name_lang)

    existing_sources = set()
    try:
        all_docs = collection.get(include=["metadatas"])
        for meta in all_docs["metadatas"]:
            if "source" in meta:
                existing_sources.add(meta["source"])
    except Exception:
        existing_sources = set()

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists() or not pdf_path.is_dir():
        print(f"Skipping invalid directory: {pdf_dir}")
        return

    for pdf_file in pdf_path.glob("*.pdf"):
        if pdf_file.name in existing_sources:
            print(f"Skipping already-ingested file: {pdf_file.name}")
            continue

        print(f"Processing {pdf_file.name}...")
        doc = pymupdf.open(pdf_file)
        for p in range(doc.page_count):
            text, ocr_used = extract_pdf(p, pdf_file, lang=lang)
            if not text:
                continue
            captions = find_captions_heuristic(text)
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                metadata = {
                    "source": str(pdf_file.resolve()),
                    "page": int(p),
                    "logical_page": logical_page_number(pdf_file, p),
                    "chunk_index": idx,
                    "captions": "; ".join(captions) if captions else "",
                    "ocr_used": "yes" if ocr_used else "no",
                    "lang": lang
                }
                uid = f"{pdf_file.stem}_p{p}_c{idx}"
                emb = embedder.encode(chunk).tolist()
                collection.add(documents=[chunk], metadatas=[metadata], ids=[uid], embeddings=[emb])
        doc.close()
    print("PDF ingestion complete.")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDF documents into ChromaDB collection.")
    parser.add_argument("pdf_dirs", type=str, nargs='+',
                        help="Directories containing PDF files. Format: <collection_name>_<lang> (2-letter ISO lang code)")
    args = parser.parse_args()

    for pdf_dir in args.pdf_dirs:
        dir_name = Path(pdf_dir).name
        parts = dir_name.split('_')
        lang = parts.pop()
        collection_name = '_'.join(parts)
        ingest_pdfs(pdf_dir, collection_name, lang)
