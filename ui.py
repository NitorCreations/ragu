#!/usr/bin/env python3
import os
import io
import re
import unicodedata
import pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from llama_cpp import Llama
import gradio as gr
from langdetect import detect
from pathlib import Path

# ---------- CONFIG ----------
DOC_DIR = "/Users/rikusarlin/Documents/ragu"                                # Adjust as needed
CHROMA_DIR = "."                                                            # Where chroma_... directories are stored
EMBED_MODEL_NAME = "/Users/rikusarlin/models/e5-base"
LLAMA_MODEL_PATH = "/Users/rikusarlin/models/gemma-3-12b-it-Q4_K_M.gguf"
TOP_K = 5
DEFAULT_LANG = "en"

embedder = SentenceTransformer(EMBED_MODEL_NAME)
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=4096, n_gpu_layers=-1, n_threads=12, temperature=0.1, verbose=False)

# ---------- HELPERS ----------
def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())

def discover_collections(base_dir=CHROMA_DIR):
    collections = {}
    for d in os.listdir(base_dir):
        if d.startswith("chroma_") and os.path.isdir(os.path.join(base_dir, d)):
            name = d[len("chroma_"):]
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                theme, lang = parts
                theme_display = theme.replace("_", " ")
                collections.setdefault(theme_display, []).append(lang)
    return collections

def find_file_path(file_name_or_path):
    p = Path(file_name_or_path)
    return p if p.exists() else None

def render_pdf_page(file_path, page_no):
    try:
        if not file_path or not Path(file_path).exists():
            print(f"⚠️ PDF not found: {file_path}")
            return None

        doc = pymupdf.open(file_path)

        if page_no >= doc.page_count:
            print(f"⚠️ Invalid page {page_no}, adjusting to last page {doc.page_count - 1}")
            page_no = doc.page_count - 1
        elif page_no < 0:
            print(f"⚠️ Negative page {page_no}, adjusting to 0")
            page_no = 0

        page = doc[page_no]
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return img

    except Exception as e:
        print(f"❌ Error rendering page {page_no} of {file_path}: {e}")
        return None

# ---------- QUERY & REFERENCES ----------
def answer_query_with_refs(query, theme, selected_audience):
    query = normalize_text(query)
    detected_lang = detect(query)
    collections = discover_collections(CHROMA_DIR)

    if theme not in collections:
        return f"❌ No info for theme '{theme}'", [], {}, {}, {}, gr.update(visible=False), gr.update(visible=False)

    used_lang = detected_lang if detected_lang in collections[theme] else DEFAULT_LANG
    warn = "" if detected_lang in collections[theme] else f"⚠️ Falling back to {DEFAULT_LANG}"

    theme_clean = theme.replace(" ", "_")
    collection_name = f"{theme_clean}_{used_lang}"
    chroma_dir = os.path.join(CHROMA_DIR, f"chroma_{collection_name}")
    client = PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(collection_name)

    q_emb = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K, include=["metadatas", "documents", "distances"])

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]

    # Audience filtering + pick best chunk per (file,page)
    best_chunks = {}  # (file, page) -> (doc, meta, distance)

    for d, m, dist in zip(docs, metas, distances):
        audiences = m.get("audiences") or "public"
        if isinstance(audiences, str):
            audience_list = [a.strip() for a in audiences.split(",")]
        else:
            audience_list = audiences
        if "public" not in audience_list and selected_audience not in audience_list:
            continue

        file_name = m.get("snapshot_pdf") or m.get("source")
        page = m.get("page") or 0
        key = (file_name, page)

        # Keep best (lowest distance = highest similarity)
        if key not in best_chunks or dist < best_chunks[key][2]:
            best_chunks[key] = (d, m, dist)

    if not best_chunks:
        return f"❌ No results for audience '{selected_audience}'", [], {}, {}, {}, gr.update(visible=False), gr.update(visible=False)

    # Sort references by distance (best first)
    sorted_refs = sorted(best_chunks.items(), key=lambda x: x[1][2])

    # References
    references = {}
    label_to_ref = {}
    ref_labels = []

    for idx, ((file_name, page), (d, m, dist)) in enumerate(sorted_refs):
        logical_page = m.get("logical_page")
        if logical_page is None:
            logical_page = page
        else:
            try:
                logical_page = int(logical_page)
            except ValueError:
                # fallback in case the stored value isn't numeric
                logical_page = page + 1
        source_url = m.get("source_url") or m.get("source")
        display_name = m.get("title") or Path(file_name).name
        # Fix some ingestion artifacts, at least try to
        if(display_name.startswith("Log in")):
            display_name = source_url
        label = f"[{idx+1}]: {display_name} p. {logical_page}"
        ref_id = f"ref_{idx}"

        references[ref_id] = (file_name, page, source_url)
        label_to_ref[label] = ref_id
        ref_labels.append(label)

    references_text = "\n".join(ref_labels)

    # LLM prompt
    context = "\n\n".join([d for (_, (d, _, _)) in sorted_refs])
    prompt = f"""
You are a helpful assistant. Use the following retrieved passages to answer the question.
Always reference sources as [1], [2], etc.

Question: {query}

Context:
{context}

Answer (include references):
"""
    out = llm(prompt, max_tokens=512)
    answer_text = out["choices"][0]["text"].strip()
    answer_text = re.split(r'\n\s*(?:references|sources):', answer_text, flags=re.IGNORECASE)[0].strip()

    final_answer = (warn+"\n\n" if warn else "") + answer_text + "\n\nReferences:\n" + references_text

    return (
        final_answer,
        gr.update(choices=ref_labels, value=ref_labels[0] if ref_labels else None, visible=True),
        references,
        label_to_ref,
        {}
    )

# ---------- PDF NAVIGATION ----------
def show_result(selected_label, refs_state, label_to_ref_state, pdf_state):
    if not selected_label or not refs_state or not label_to_ref_state:
        return (
            gr.update(value=None, visible=False),
            pdf_state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    ref_id = label_to_ref_state.get(selected_label)
    if not ref_id or ref_id not in refs_state:
        return (
            gr.update(value=None, visible=False),
            pdf_state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    file_name, page_no, source_url = refs_state[ref_id]
    page_path = find_file_path(file_name)
    if not page_path:
        return (
            gr.update(value=None, visible=False),
            pdf_state,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    img = render_pdf_page(page_path, page_no)
    pdf_state = {
        "ref_id": ref_id,
        "page_no": page_no,
        "total_pages": pymupdf.open(page_path).page_count,
        "file_path": file_name,
        "source_url": source_url,
    }

    link_md = f"[Open Original]({source_url})" if source_url else ""
    return (
        gr.update(value=img, visible=True),
        pdf_state,
        gr.update(value=link_md, visible=bool(source_url)),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )

def navigate_pdf(action, pdf_state):
    if not pdf_state:
        return gr.update(value=None, visible=False), pdf_state, gr.update(visible=False)

    page_no = pdf_state["page_no"]
    total_pages = pdf_state.get("total_pages", page_no + 1)

    if action == "first": page_no = 0
    elif action == "prev": page_no = max(0, page_no - 1)
    elif action == "next": page_no = min(total_pages - 1, page_no + 1)
    elif action == "last": page_no = total_pages - 1

    pdf_state["page_no"] = page_no
    file_name = find_file_path(pdf_state["file_path"])
    img = render_pdf_page(file_name, page_no)
    link_md = f"[Open Original]({pdf_state['source_url']})" if pdf_state.get("source_url") else ""

    return gr.update(value=img, visible=True), pdf_state, gr.update(value=link_md, visible=bool(pdf_state.get("source_url")))


# ---------- UI ----------
custom_css = """
#nav-buttons {
  display: flex;
  flex-direction: row;
  gap: 0.5rem;
  right: 0;
  width: 50vw;
  position: fixed;
  right: 1rem;
  bottom: 2rem;
  background: white;
  padding: 0.5rem;
  border-radius: 0.5rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
  z-index: 100;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Local RAG")

    with gr.Row():
        with gr.Column():
            theme_dropdown = gr.Dropdown(label="Theme", choices=list(discover_collections().keys()), value=None)
            audience_dropdown = gr.Dropdown(label="Audience", choices=["public", "russia", "nitorean"], value="public")
            query_input = gr.Textbox(label="Question", lines=1, placeholder="Ask something...")
            submit_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=20)

        with gr.Column():
            ref_dropdown = gr.Dropdown(label="Select reference", choices=[], interactive=True)
            pdf_image_output = gr.Image(label="PDF Page", visible=False)
            link_output = gr.Markdown(label="Original URL", visible=False)

    with gr.Row(elem_id="nav-buttons"):
        btn_first = gr.Button("First", visible=False)
        btn_prev = gr.Button("Prev", visible=False)
        btn_next = gr.Button("Next", visible=False)
        btn_last = gr.Button("Last", visible=False)

    refs_state = gr.State()
    label_to_ref_state = gr.State()
    pdf_state = gr.State()

    submit_btn.click(
        answer_query_with_refs,
        inputs=[query_input, theme_dropdown, audience_dropdown],
        outputs=[answer_output, ref_dropdown, refs_state, label_to_ref_state, pdf_state]
    )

    ref_dropdown.change(
        show_result,
        inputs=[ref_dropdown, refs_state, label_to_ref_state, pdf_state],
        outputs=[pdf_image_output, pdf_state, link_output, btn_first, btn_prev, btn_next, btn_last]
    )

    btn_first.click(lambda s: navigate_pdf("first", s),
                    inputs=[pdf_state], outputs=[pdf_image_output, pdf_state, link_output])
    btn_prev.click(lambda s: navigate_pdf("prev", s),
                   inputs=[pdf_state], outputs=[pdf_image_output, pdf_state, link_output])
    btn_next.click(lambda s: navigate_pdf("next", s),
                   inputs=[pdf_state], outputs=[pdf_image_output, pdf_state, link_output])
    btn_last.click(lambda s: navigate_pdf("last", s),
                   inputs=[pdf_state], outputs=[pdf_image_output, pdf_state, link_output])

demo.launch()
