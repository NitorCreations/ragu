import os
import io
import re
import unicodedata
from pathlib import Path
import pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from llama_cpp import Llama
import gradio as gr
from langdetect import detect

DOC_DIR = "."
EMBED_MODEL_NAME = "/Users/rikusarlin/models/e5-base"                       # Change this to your local dir
LLAMA_MODEL_PATH = "/Users/rikusarlin/models/gemma-3-12b-it-Q4_K_M.gguf"    # Change this to your local dir
TOP_K = 5
DEFAULT_LANG = "en"

embedder = SentenceTransformer(EMBED_MODEL_NAME)
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=4096, n_gpu_layers=-1, n_threads=12, temperature=0.1)

# ---------- Helpers ----------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())

def discover_collections(base_dir=DOC_DIR):
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
    path = Path(file_name_or_path)
    return path if path.exists() else None

def render_pdf_page(file_path, page_no):
    try:
        doc = pymupdf.open(file_path)
        page_no = max(0, min(page_no, doc.page_count - 1))
        page = doc[page_no]
        pix = page.get_pixmap()
        doc.close()
        return Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception:
        return None

# ---------- Query & References ----------
def answer_query_with_refs(query, theme, selected_audience):
    query = normalize_text(query)
    detected_lang = detect(query)
    collections = discover_collections(DOC_DIR)

    if theme not in collections:
        return f"❌ No info for theme '{theme}'", [], {}, {}, {}, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    used_lang = detected_lang if detected_lang in collections[theme] else DEFAULT_LANG
    warn = "" if detected_lang in collections[theme] else f"⚠️ Falling back to {DEFAULT_LANG}"

    theme_clean = theme.replace(" ", "_")
    collection_name = f"{theme_clean}_{used_lang}"
    chroma_dir = os.path.join(DOC_DIR, f"chroma_{collection_name}")
    client = PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(collection_name)

    q_emb = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K, include=["metadatas","documents"])

    docs = res["documents"][0]
    metas = res["metadatas"][0]

    # Filter results with audience
    filtered_docs = []
    filtered_metas = []
    for d, m in zip(docs, metas):
        audiences = m.get("audiences") or "public"
        audience_list = [a.strip() for a in audiences.split(",")]

        if "public" in audience_list or selected_audience in audience_list:
            filtered_docs.append(d)
            filtered_metas.append(m)

    docs = filtered_docs
    metas = filtered_metas

    if not docs:
        return f"❌ No results available for audience '{selected_audience}'", [], {}, {}, {}, \
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    references = {}
    context = ""
    for idx, (d, m) in enumerate(zip(docs, metas)):
        ref_id = f"ref_{idx}"
        source = m.get("source")
        page = m.get("page")
        logical_page = m.get("logical_page")
        title = m.get("title")

        if page is not None:
            references[ref_id] = (source, page, logical_page or page+1)
        else:
            references[ref_id] = (source, title)
        context += f"{d}\n\n"

    # LLM prompt
    prompt = f"""
    You are a helpful assistant. Use the following retrieved passages to answer the question.
    Always reference sources as [1], [2], etc., corresponding to the passages.

    The question is written in **{detected_lang}**.
    Answer in the same language as the question (do not translate).

    Context:
    {context}

    Question: {query}

    Answer (in {detected_lang}, include references to original passages):
    """
    out = llm(prompt, max_tokens=512)
    answer_text = out["choices"][0]["text"].strip()
    answer_text = re.split(r'\n\s*(?:references|sources):', answer_text, flags=re.IGNORECASE)[0]
    answer_text = re.sub(r'\[\d+\]', '', answer_text).strip()

    # Build references
    seen = {}
    ref_links = []
    ref_map = {}
    references_list = []
    any_pdf = False
    for ref_id, ref_data in references.items():
        if len(ref_data) == 3:  # PDF
            file_name, _, logical_page = ref_data
            key = (file_name, logical_page)
            if key in seen: continue
            seen[key] = len(seen) + 1
            idx = seen[key]
            label = f"[{idx}]: {Path(file_name).name}, p. {logical_page}"
            any_pdf = True
        elif len(ref_data) == 2:  # Web
            url, title = ref_data
            key = url
            if key in seen:
                continue
            seen[key] = len(seen) + 1
            idx = seen[key]
            label = f"[{idx}]: {title or url} ({url})"
            # only track for references_text and web_links_md
            references_list.append(label)
            references[ref_id] = (url, title)
            continue
        else:
            continue

        ref_links.append(label)
        ref_map[label] = ref_id
        references_list.append(label)

    references_text = "\n".join(references_list)
    final_answer = (warn+"\n\n" if warn else "") + answer_text + "\n\nReferences:\n" + references_text

    # Web-only case → show all unique links in right-hand panel
    web_links_md = ""
    if not any_pdf:
        unique_web_refs = []
        for ref_id, ref_data in references.items():
            if isinstance(ref_data, tuple) and len(ref_data) == 2:
                url, title = ref_data
                if url not in [u for u, _ in unique_web_refs]:
                    unique_web_refs.append((url, title))
        web_links_md = "\n".join([f"- [{title or url}]({url})" for url, title in unique_web_refs])

    return (final_answer,
            gr.update(choices=ref_links, value=None if not any_pdf else (ref_links[0] if ref_links else None), visible=any_pdf),
            references,
            {},
            ref_map,
            gr.update(visible=any_pdf),   # pdf_image_output
            gr.update(visible=any_pdf),   # nav buttons row
            gr.update(visible=not any_pdf, value=web_links_md),  # full web links
            gr.update(visible=False))     # single web link (mixed case only, set later)

# ---------- Result Navigation ----------
def show_result(selected_label, refs_state, pdf_state, ref_map):
    if not selected_label or not refs_state:
        return None, pdf_state, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    ref_id = ref_map.get(selected_label)
    if not ref_id or ref_id not in refs_state:
        return None, pdf_state, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    ref_data = refs_state[ref_id]

    # Web reference (only if PDFs are also present)
    if isinstance(ref_data, tuple) and len(ref_data) == 2:
        url, title = ref_data
        link_md = f"[Open source page: {title or url}]({url})"
        return None, pdf_state, link_md, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    # PDF reference
    if isinstance(ref_data, tuple) and len(ref_data) == 3:
        file_name, physical_page, _ = ref_data
        page_path = find_file_path(file_name)
        if not page_path:
            return None, pdf_state, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        total_pages = pymupdf.open(page_path).page_count
        img = render_pdf_page(page_path, physical_page)
        pdf_state = {"ref_id": ref_id, "page_no": physical_page, "total_pages": total_pages, "file_path": file_name}

        return (img, pdf_state, "",
                gr.update(interactive=(physical_page>0), visible=True),
                gr.update(interactive=(physical_page>0), visible=True),
                gr.update(interactive=(physical_page<total_pages-1), visible=True),
                gr.update(interactive=(physical_page<total_pages-1), visible=True))

def navigate_pdf(action, pdf_state):
    if not pdf_state:
        return None, pdf_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    page_no = pdf_state["page_no"]
    total_pages = pdf_state.get("total_pages", page_no+1)

    if action=="first": page_no=0
    elif action=="prev": page_no=max(0,page_no-1)
    elif action=="next": page_no=min(total_pages-1,page_no+1)
    elif action=="last": page_no=total_pages-1

    pdf_state["page_no"]=page_no
    img = render_pdf_page(find_file_path(pdf_state["file_path"]), page_no)
    return img, pdf_state, gr.update(interactive=(page_no>0)), gr.update(interactive=(page_no>0)), gr.update(interactive=(page_no<total_pages-1)), gr.update(interactive=(page_no<total_pages-1))

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("Local RAG")

    with gr.Row():
        with gr.Column():
            theme_dropdown = gr.Dropdown(label="Theme", choices=list(discover_collections().keys()), value=None)
            audience_dropdown = gr.Dropdown(
                label="Audience",
                choices=["public", "confidential", "nitorean"],
                value="public",
                interactive=True
            )
            query_input = gr.Textbox(label="Question", lines=1, max_lines=3, placeholder="Ask something...")
            submit_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=20, max_lines=25)

        with gr.Column():
            with gr.Accordion("References", open=True):
                ref_dropdown = gr.Dropdown(label="Select reference", choices=[], interactive=True)
            pdf_image_output = gr.Image(label="PDF Page", visible=False)
            web_links_output = gr.Markdown(label="All Web Links", visible=False)   # NEW
            link_output = gr.Markdown(label="Single Web Link", visible=False)      # unchanged

            with gr.Row():
                btn_first = gr.Button("First", visible=False)
                btn_prev = gr.Button("Prev", visible=False)
                btn_next = gr.Button("Next", visible=False)
                btn_last = gr.Button("Last", visible=False)

    refs_state = gr.State()
    pdf_state = gr.State()
    ref_map_state = gr.State()

    submit_btn.click(
        answer_query_with_refs,
        inputs=[query_input, theme_dropdown, audience_dropdown],
        outputs=[answer_output, ref_dropdown, refs_state, pdf_state, ref_map_state,
                 pdf_image_output, btn_first, web_links_output, link_output]
    )
    ref_dropdown.change(show_result,
                        inputs=[ref_dropdown, refs_state, pdf_state, ref_map_state],
                        outputs=[pdf_image_output, pdf_state, link_output, btn_first, btn_prev, btn_next, btn_last])
    btn_first.click(lambda s: navigate_pdf("first", s),
                    inputs=[pdf_state],
                    outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])
    btn_prev.click(lambda s: navigate_pdf("prev", s),
                   inputs=[pdf_state],
                   outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])
    btn_next.click(lambda s: navigate_pdf("next", s),
                   inputs=[pdf_state],
                   outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])
    btn_last.click(lambda s: navigate_pdf("last", s),
                   inputs=[pdf_state],
                   outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])

demo.launch()
