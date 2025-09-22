import pymupdf
from pathlib import Path
import gradio as gr
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from llama_cpp import Llama
from PIL import Image
import io
import re

# ---------- CONFIG ----------
CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "/Users/rikusarlin/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
TOP_K = 5
DOC_DIRS = ["/Users/rikusarlin/Documents/architecture"]

embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("docs")

llm = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    n_threads=12,
    temperature=0.1
)

# ---------- Helpers ----------
def render_pdf_page(file_path, page_no):
    try:
        doc = pymupdf.open(file_path)
        page_no = max(0, min(page_no, doc.page_count-1))
        page = doc[page_no]
        pix = page.get_pixmap()
        doc.close()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img
    except Exception:
        return None

def find_file_path(file_name):
    for dir_path in DOC_DIRS:
        path_obj = Path(dir_path)
        for candidate in path_obj.rglob(file_name):
            if candidate.is_file():
                return candidate
    return None

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

def logical_page_number(file_name, page_no):
    """Compute logical page number using PDF PageLabels"""
    path = find_file_path(file_name)
    if not path:
        return page_no + 1
    doc = pymupdf.open(path)
    try:
        page_labels = doc.get_page_labels()
        if not page_labels:
            return page_no + 1

        # Find the label dictionary for the given page number
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

            page_label = ""
            if style == "r":
                page_label = int_to_roman(num).lower()
            elif style == "R":
                page_label = int_to_roman(num).upper()
            else:  # 'D' and other formats
                page_label = str(num)
            
            return prefix + page_label
        else:
            return page_no + 1
    finally:
        doc.close()

# ---------- Query + References ----------
def answer_query_with_refs(query):
    q_emb = embedder.encode(query).tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["metadatas","documents"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    references = {}  # ref_id -> (file_name, physical_page, logical_page)
    context = ""

    for idx, (d, m) in enumerate(zip(docs, metas)):
        ref_id = f"ref_{idx}"
        file_name = m.get("source")
        physical_page = m.get("page")
        logical_page = logical_page_number(file_name, physical_page)
        references[ref_id] = (file_name, physical_page, logical_page)
        context += f"{d}\n\n"

    # LLM prompt
    prompt = f"""
You are a helpful assistant. Use the following retrieved passages to answer the question.
Always reference sources as [1], [2], etc., corresponding to the passages.
Do not include a separate sources list or references outside the references passages.

{context}

Question: {query}

Answer (include references to original passges):
"""
    out = llm(prompt, max_tokens=512)
    answer_text = out["choices"][0]["text"].strip()

    # Remove any generated References/Sources section and inline [n]
    answer_text = re.split(r'\n\s*(?:references|sources):', answer_text, flags=re.IGNORECASE)[0]
    answer_text = re.sub(r'\[\d+\]', '', answer_text).strip()

    # Deduplicate references by (file, logical_page)
    seen = {}
    ref_links = []
    ref_map_for_dropdown = {}
    references_list = []

    for ref_id, (file_name, physical_page, logical_page) in references.items():
        key = (file_name, logical_page)
        if key in seen:
            continue
        seen[key] = len(seen) + 1  # new numbering
        idx = seen[key]
        label = f"[{idx}]: {file_name}, p. {logical_page}"
        ref_links.append(label)
        ref_map_for_dropdown[label] = ref_id
        references_list.append(label)

    references_text = "\n".join(references_list)
    final_answer = answer_text + "\n\nReferences:\n" + references_text

    # Initial PDF state uses physical page for rendering
    pdf_state = {}
    if ref_links:
        first_label = ref_links[0]
        first_ref_id = ref_map_for_dropdown[first_label]
        pdf_state = {
            "ref_id": first_ref_id,
            "page_no": references[first_ref_id][1],  # physical page
        }

    return (
        final_answer,
        gr.update(choices=ref_links, value=ref_links[0] if ref_links else None),
        references,
        pdf_state,
        ref_map_for_dropdown
    )

# ---------- PDF Displaying and navigation ----------
def show_pdf(selected_label, refs_state, pdf_state, ref_map_for_dropdown):
    if not selected_label or not refs_state:
        return None, pdf_state, True, True, True, True

    ref_id = ref_map_for_dropdown.get(selected_label)
    if not ref_id or ref_id not in refs_state:
        return None, pdf_state, True, True, True, True

    file_name, physical_page, _ = refs_state[ref_id]  # unpack correctly
    page_path = find_file_path(file_name)
    if not page_path:
        return None, pdf_state, True, True, True, True

    total_pages = pymupdf.open(page_path).page_count
    img = render_pdf_page(page_path, physical_page)  # use physical page
    pdf_state = {"ref_id": ref_id, "page_no": physical_page, "total_pages": total_pages, "file_path": file_name}

    return (
        img,
        pdf_state,
        gr.update(interactive=(physical_page > 0)),
        gr.update(interactive=(physical_page > 0)),
        gr.update(interactive=(physical_page < total_pages - 1)),
        gr.update(interactive=(physical_page < total_pages - 1)),
    )

def navigate_pdf(action, pdf_state):
    if not pdf_state:
        return None, pdf_state, True, True, True, True

    page_no = pdf_state["page_no"]
    total_pages = pdf_state.get("total_pages", page_no + 1)

    if action == "first":
        page_no = 0
    elif action == "prev":
        page_no = max(0, page_no - 1)
    elif action == "next":
        page_no = min(total_pages - 1, page_no + 1)
    elif action == "last":
        page_no = total_pages - 1

    pdf_state["page_no"] = page_no
    img = render_pdf_page(find_file_path(pdf_state["file_path"]), page_no)

    return (
        img,
        pdf_state,
        gr.update(interactive=(page_no > 0)),
        gr.update(interactive=(page_no > 0)),
        gr.update(interactive=(page_no < total_pages - 1)),
        gr.update(interactive=(page_no < total_pages - 1)),
    )

# ---------- Gradio UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## Local RAG Chat with PDF Navigation")

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Question", lines=1, max_lines=3,
                                     placeholder="Type your question here...",
                                     value="What should I take into account when deciding between modular monolith and microservices?")
            submit_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=20, max_lines=25)

        with gr.Column():
            with gr.Accordion("References", open=True):
                ref_dropdown = gr.Dropdown(label="Select reference", choices=[], interactive=True)
            pdf_image_output = gr.Image(label="PDF Page")

            with gr.Row():
                btn_first = gr.Button("First")
                btn_prev = gr.Button("Prev")
                btn_next = gr.Button("Next")
                btn_last = gr.Button("Last")

    refs_state = gr.State()
    pdf_state = gr.State()
    ref_map_state = gr.State()

    submit_btn.click(
        fn=answer_query_with_refs,
        inputs=query_input,
        outputs=[answer_output, ref_dropdown, refs_state, pdf_state, ref_map_state]
    )

    ref_dropdown.change(
        fn=show_pdf,
        inputs=[ref_dropdown, refs_state, pdf_state, ref_map_state],
        outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last]
    )

    btn_first.click(lambda s: navigate_pdf("first", s), inputs=[pdf_state],
                    outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])
    btn_prev.click(lambda s: navigate_pdf("prev", s), inputs=[pdf_state],
                   outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])
    btn_next.click(lambda s: navigate_pdf("next", s), inputs=[pdf_state],
                   outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])
    btn_last.click(lambda s: navigate_pdf("last", s), inputs=[pdf_state],
                   outputs=[pdf_image_output, pdf_state, btn_first, btn_prev, btn_next, btn_last])

demo.launch()
