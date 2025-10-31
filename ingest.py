#!/usr/bin/env python3
import os
import io
import re
import unicodedata
import requests
import pycountry
import pytesseract
import pymupdf
from PIL import Image
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langdetect import detect
from pathlib import Path

# ------------- CONFIG -------------
EMBED_MODEL_NAME = "/Users/rikusarlin/models/e5-base"   # adjust for your environment
DOCUMENTS_ROOT = "/Users/rikusarlin/Documents/ragu"     # where to save snapshots
CHUNK_SIZE = 800

# ------------- HELPERS -------------
def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(fragment="", query="").geturl()
    return clean.rstrip("/")

def get_links_for_crawl(url, root_url, session, visited):
    urls = set()
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()

        if not response.url.startswith(root_url):
            return urls

        soup = BeautifulSoup(response.content, "html.parser")
        for a_tag in soup.find_all("a"):
            href = a_tag.attrs.get("href")
            if not href:
                continue
            href = urljoin(response.url, href)
            href = normalize_url(href)
            if not is_valid_url(href):
                continue
            if not href.startswith(root_url):
                continue
            if href in visited:
                continue
            basename = os.path.basename(urlparse(href).path)
            if '.' not in basename or href.lower().endswith(('.html', '.htm', '.pdf')):
                urls.add(href)
    except requests.exceptions.RequestException:
        pass
    return urls

def crawl(seed_url, max_urls=200, max_depth=2, session=None):
    """
    Crawl a website breadth-first, respecting max_urls and max_depth.
    """
    session = session or requests.Session()
    root_url = normalize_url(seed_url)
    visited = set()
    queue = deque([(root_url, 0)])
    visited.add(root_url)

    while queue and len(visited) < max_urls:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        links = get_links_for_crawl(current, root_url, session, visited)
        for link in links:
            if len(visited) >= max_urls:
                break
            visited.add(link)
            queue.append((link, depth + 1))
            print(f"[{len(visited)}/{max_urls}] + {link}")

    print(f"✅ Crawl finished — {len(visited)} unique URLs visited.")
    return visited

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

def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse whitespace."""
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

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

def find_captions_heuristic(page_text):
    if not page_text:
        return []
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln.lower().startswith("figure") or ln.lower().startswith("fig.")]

def get_webpage_title(url):
    """Fetch <title> from web page."""
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return soup.title.string.strip() if soup.title else url
    except Exception:
        return url  # fallback

# ------------- COOKIES -------------

def create_session_with_cookies(cookies_input: str | None) -> requests.Session:
    """Create a requests.Session and populate it with cookies from either a string or a file."""
    session = requests.Session()
    if not cookies_input:
        return session

    if os.path.exists(cookies_input):
        with open(cookies_input, "r") as f:
            cookies_input = f.read().strip()

    for pair in cookies_input.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, value = pair.split("=", 1)
        session.cookies.set(name.strip(), value.strip())

    return session

def playwright_with_cookies(page, session):
    # If no cookies, just skip
    if not session.cookies or len(session.cookies) == 0:
        return
    cookies = []
    for c in session.cookies:
        # Filter only valid cookies
        if not c.name or not c.value:
            continue
        domain = c.domain or urlparse(page.url).hostname
        if not domain:
            continue
        cookies.append({
            "name": c.name,
            "value": c.value,
            "domain": domain,
            "path": c.path or "/"
        })
    if cookies:
        try:
            page.context.add_cookies(cookies)
        except Exception as e:
            print(f"⚠️ Skipping cookies injection — {e}")

# ------------- WEB CRAWLING -------------

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

from collections import deque

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(fragment="", query="").geturl()
    return clean.rstrip("/")

def get_links_for_crawl(url, root_url, session, visited):
    urls = set()
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()

        if not response.url.startswith(root_url):
            return urls

        soup = BeautifulSoup(response.content, "html.parser")
        for a_tag in soup.find_all("a"):
            href = a_tag.attrs.get("href")
            if not href:
                continue
            href = urljoin(response.url, href)
            href = normalize_url(href)
            if not is_valid_url(href):
                continue
            if not href.startswith(root_url):
                continue
            if href in visited:
                continue
            basename = os.path.basename(urlparse(href).path)
            if '.' not in basename or href.lower().endswith(('.html', '.htm', '.pdf')):
                urls.add(href)
    except requests.exceptions.RequestException:
        pass
    return urls

def crawl(seed_url, max_urls=200, max_depth=2, session=None):
    """
    Crawl a website breadth-first, respecting max_urls and max_depth.
    """
    session = session or requests.Session()
    root_url = normalize_url(seed_url)
    visited = set()
    queue = deque([(root_url, 0)])
    visited.add(root_url)

    while queue and len(visited) < max_urls:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        links = get_links_for_crawl(current, root_url, session, visited)
        for link in links:
            if len(visited) >= max_urls:
                break
            visited.add(link)
            queue.append((link, depth + 1))
            print(f"[{len(visited)}/{max_urls}] + {link}")

    print(f"✅ Crawl finished — {len(visited)} unique URLs visited.")
    return visited
# ------------- WEB TO PDF -------------

def save_webpage_as_pdf_headless(url, output_dir, session):
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', url)
    output_path = os.path.join(output_dir, f"{safe_name}.pdf")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        playwright_with_cookies(page, session)
        page.goto(url, timeout=60000)
        page.pdf(path=output_path, format="A4")
        browser.close()

    return output_path

# ------------- PDF EXTRACTION -------------

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

# ------------- INGEST -------------

def ingest_url_as_pdf(url, collection, collection_name_lang, embedder, lang, audience_str, session):
    """Render HTML page to PDF, extract text, embed & store."""
    doc_root = f"{DOCUMENTS_ROOT}/{collection_name_lang}"
    pdf_snapshot = save_webpage_as_pdf_headless(url, doc_root, session)
    doc = pymupdf.open(pdf_snapshot)
    for p in range(doc.page_count):
        text, ocr_used = extract_pdf(p, pdf_snapshot, lang)
        if not text:
            continue
        detected_lang = detect(text)
        if detected_lang != lang:
            print(f"Skipping {url} page {p} (lang mismatch: {detected_lang})")
            continue
        captions = find_captions_heuristic(text)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            uid = f"{url}_p{p}_c{idx}"
            emb = embedder.encode(chunk).tolist()
            metadata = {
                "source": url,
                "snapshot_pdf": pdf_snapshot,
                "page": p,
                "chunk_index": idx,
                "captions": "; ".join(captions) if captions else "",
                "ocr_used": "yes" if ocr_used else "no",
                "lang": lang,
                "audiences": audience_str,
                "title": get_webpage_title(url)
            }
            collection.add(documents=[chunk], metadatas=[metadata], ids=[uid], embeddings=[emb])
    doc.close()

def ingest_pdf_url(path_or_url, collection, embedder, lang, audience_str, session=None):
    """
    Ingest a PDF, either from a URL or a local file path.
    """
    if os.path.exists(path_or_url):
        # Local file
        doc = pymupdf.open(path_or_url)
        pdf_identifier = path_or_url
    else:
        # URL
        session = session or requests.Session()
        response = session.get(path_or_url)
        response.raise_for_status()
        doc = pymupdf.open(stream=io.BytesIO(response.content), filetype="pdf")
        pdf_identifier = path_or_url

    for p in range(doc.page_count):
        text, ocr_used = extract_pdf(p, pdf_identifier, lang)
        if not text:
            continue
        detected_lang = detect(text)
        if detected_lang != lang:
            print(f"Skipping {pdf_identifier} page {p} (lang mismatch: {detected_lang})")
            continue
        captions = find_captions_heuristic(text)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            uid = f"{pdf_identifier}_p{p}_c{idx}"
            emb = embedder.encode(chunk).tolist()
            metadata = {
                "source": pdf_identifier,
                "snapshot_pdf": pdf_identifier,
                "page": p,
                "logical_page": logical_page_number(path_or_url, p),
                "chunk_index": idx,
                "captions": "; ".join(captions) if captions else "",
                "ocr_used": "yes" if ocr_used else "no",
                "lang": lang,
                "audiences": audience_str,
                "title": Path(pdf_identifier).name
            }
            collection.add(documents=[chunk], metadatas=[metadata], ids=[uid], embeddings=[emb])
    doc.close()

# ------------- MAIN INGESTION -------------

def ingest_web_and_pdf(seed_path_or_url, collection_name, lang='en', id_range=None, width=5, audiences=None, cookies=None, max_urls=50):
    audiences = audiences or ["public"]
    audience_str = ",".join(audiences)
    session = create_session_with_cookies(cookies)

    # Init Chroma + embedder
    collection_name_lang = f"{collection_name}_{lang}"
    chroma_dir = f"chroma_{collection_name_lang}"
    client = PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=collection_name_lang)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    urls_to_ingest = []

    # --- Directory ingestion ---
    if os.path.isdir(seed_path_or_url):
        for root, _, files in os.walk(seed_path_or_url):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, f)
                    urls_to_ingest.append(pdf_path)

    # --- Sequential range mode ---
    elif id_range:
        start_id, end_id = id_range
        urls_to_ingest = [f"{seed_path_or_url}{i:0{width}d}" for i in range(start_id, end_id + 1)]

    # --- Web crawling mode ---
    elif seed_path_or_url.startswith("http://") or seed_path_or_url.startswith("https://"):
        urls_to_ingest = crawl(seed_path_or_url, max_urls=max_urls, session=session)

    else:
        raise ValueError(f"Invalid input: {seed_path_or_url} is neither a directory, range base URL, nor HTTP URL.")

    # --- Process ingestion ---
    for url in urls_to_ingest:
        print(f"Processing: {url}")
        try:
            if url.lower().endswith('.pdf'):
                ingest_pdf_url(url, collection, embedder, lang, audience_str, session)
            else:
                ingest_url_as_pdf(url, collection, collection_name_lang, embedder, lang, audience_str, session)
        except Exception as e:
            print(f"❌ Error ingesting {url}: {e}")

    print(f"✅ Ingestion complete for {len(urls_to_ingest)} items.")

# ------------- CLI -------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest website + PDF content into ChromaDB.")
    parser.add_argument("url", type=str, help="Seed URL or base URL for range mode.")
    parser.add_argument("--collection", required=True, help="Chroma collection name (base).")
    parser.add_argument("--lang", type=str, default="en", help="Expected language of content.")
    parser.add_argument("--range", nargs=2, type=int, help="Start and end IDs for sequential URL generation.")
    parser.add_argument("--width", type=int, default=5, help="Zero-padding width for sequential IDs.")
    parser.add_argument("--audiences", nargs="*", default=["public"], help="Audiences for access control.")
    parser.add_argument("--cookies", type=str, help="Cookies string or file path.")
    parser.add_argument("--max-urls", type=int, default=10000, help="Max URLs to crawl.")
    args = parser.parse_args()

    ingest_web_and_pdf(
        args.url,
        args.collection,
        args.lang,
        tuple(args.range) if args.range else None,
        args.width,
        audiences=args.audiences,
        cookies=args.cookies,
        max_urls=args.max_urls
    )
