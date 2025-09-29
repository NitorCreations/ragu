import requests
from bs4 import BeautifulSoup
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from urllib.parse import urljoin, urlparse
import pymupdf  # For PDF processing
import io
import unicodedata
from langdetect import detect
import pycountry

EMBED_MODEL_NAME = "/Users/rikusarlin/models/e5-base"                       # Change this to your local dir

def to_iso639_1(code: str) -> str:
    """Normalize a language code to 2-letter ISO 639-1 if possible."""
    try:
        lang = pycountry.languages.get(alpha_2=code)
        if lang:
            return lang.alpha_2.lower()
        lang = pycountry.languages.get(alpha_3=code)
        if lang and hasattr(lang, "alpha_2"):
            return lang.alpha_2.lower()
    except Exception:
        pass
    return code.lower()

def to_iso639_3(code: str) -> str:
    """Normalize a language code to 3-letter ISO 639-2 if possible."""
    try:
        lang = pycountry.languages.get(alpha_2=code)
        if lang and hasattr(lang, "bibliographic"):
            return lang.bibliographic.lower()
        if lang and hasattr(lang, "alpha_3"):
            return lang.alpha_3.lower()
        lang = pycountry.languages.get(alpha_3=code)
        if lang:
            return lang.alpha_3.lower()
    except Exception:
        pass
    return code.lower()

def generate_sequential_urls(base_url, start_id, end_id, width=5):
    """Generate URLs by appending zero-padded numbers to a base URL."""
    return [f"{base_url}{i:0{width}d}" for i in range(start_id, end_id + 1)]

def chunk_text(text, chunk_size=800):
    return [text[i:i+chunk_size].strip()
            for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_website_links(url, internal_urls, root_url):
    urls = set()
    try:
        response = requests.get(url)
        response.raise_for_status()

        if not response.url.startswith(root_url):
            print(f"Redirected from {url} to {response.url}, which is outside of the root url. Not following links.")
            return urls

        soup = BeautifulSoup(response.content, "html.parser")
        for a_tag in soup.find_all("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(response.url, href)
            parsed_href = urlparse(href)
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not is_valid_url(href):
                continue
            if not href.startswith(root_url):
                continue
            if href in internal_urls:
                continue

            path = urlparse(href).path
            basename = path.split('/')[-1]

            if '.' not in basename or href.lower().endswith(('.htm', '.html', '.pdf')):
                urls.add(href)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url} for link extraction: {e}")

    return urls

def crawl(url, max_urls=50):
    total_urls_visited = 0
    internal_urls = set()
    root_url = url

    def crawl_recursive(url_to_crawl):
        nonlocal total_urls_visited
        if total_urls_visited >= max_urls:
            return

        if url_to_crawl.lower().endswith('.pdf'):
            return

        total_urls_visited += 1
        links = get_all_website_links(url_to_crawl, internal_urls, root_url)
        for link in links:
            if link not in internal_urls:
                internal_urls.add(link)
                print(f"[*] Internal link: {link}")
                crawl_recursive(link)

    internal_urls.add(url)
    crawl_recursive(url)
    return internal_urls

def ingest_web_content(url, collection_name, lang="en", id_range=None):
    """
    Ingests web content from a given URL into the vector database.
    If id_range is given, generate sequential URLs instead of crawling.
    """
    print(f"Starting web ingestion for: {url} [lang={lang}]")

    # Ensure collection name is language-specific
    collection_name_lang = f"{collection_name}_{lang}"
    chroma_dir = f"chroma_{collection_name_lang}"

    client = PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=collection_name_lang)

    # Multilingual embedding model
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    if id_range:
        start_id, end_id = id_range
        urls_to_ingest = generate_sequential_urls(url, start_id, end_id)
    else:
        urls_to_ingest = crawl(url)

    for web_url in urls_to_ingest:
        try:
            print(f"Fetching content from: {web_url}")
            response = requests.get(web_url)
            response.raise_for_status()

            final_url = response.url

            if final_url.lower().endswith('.pdf'):
                # Process PDF
                doc = pymupdf.open(stream=io.BytesIO(response.content), filetype="pdf")
                title = doc.metadata.get('title', '')
                if not title:
                    title = final_url.split('/')[-1]

                for page_num, page in enumerate(doc):
                    text_content = normalize_text(page.get_text())
                    if text_content:
                        detected_lang = to_iso639_1(detect(text_content))
                        expected_lang = to_iso639_1(lang)

                        if detected_lang != expected_lang:
                            print(f"Skipping page (lang={detected_lang}, expected={expected_lang}): {final_url}")
                            continue

                        chunks = chunk_text(text_content)
                        for idx, chunk in enumerate(chunks):
                            embedding = model.encode(chunk).tolist()
                            uid = f"{final_url}-p{page_num}_c{idx}"
                            collection.add(
                                embeddings=[embedding],
                                documents=[chunk],
                                metadatas=[{
                                    "source": final_url,
                                    "page": page_num,
                                    "chunk_index": idx,
                                    "lang": detected_lang,
                                    "type": "web-pdf",
                                    "title": title
                                }],
                                ids=[uid]
                            )
                        print(f"Successfully ingested PDF page: {final_url} (page {page_num})")
                doc.close()

            else:
                # Process HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.title.string if soup.title else ''
                text_content = normalize_text(soup.get_text(separator=' ', strip=True))

                if text_content:
                    detected_lang = detect(text_content)
                    if detected_lang != lang:
                        print(f"Skipping page (lang={detected_lang}, expected={lang}): {final_url}")
                        continue

                    chunks = chunk_text(text_content)
                    for idx, chunk in enumerate(chunks):
                        embedding = model.encode(chunk).tolist()
                        uid = f"{final_url}_c{idx}"
                        collection.add(
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{
                                "source": final_url,
                                "chunk_index": idx,
                                "lang": detected_lang,
                                "type": "web",
                                "title": title
                            }],
                            ids=[uid]
                            )
                    print(f"Successfully ingested: {final_url}")
                else:
                    print(f"No text content found at: {final_url}")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {web_url}: {e}")
        except Exception as e:
            print(f"An error occurred while processing {web_url}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest web content into a ChromaDB collection.")
    parser.add_argument("url", type=str, help="The starting URL to crawl and ingest, or a base URL for sequential mode.")
    parser.add_argument("--collection", type=str, required=True, help="The base name of the ChromaDB collection to use.")
    parser.add_argument("--lang", type=str, default="en", help="The language of the content to be ingested.")
    parser.add_argument("--range", nargs=2, type=int, help="Start and end ID for sequential ingestion (e.g. 1 1425)")
    parser.add_argument("--width", type=int, default=5, help="Zero-padding width for numeric IDs (default=5)")
    args = parser.parse_args()

    ingest_web_content(
        args.url,
        args.collection,
        args.lang,
        tuple(args.range) if args.range else None
    )
