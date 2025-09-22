from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from llama_cpp import Llama

CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_MODEL_PATH = "/Users/rikusarlin/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
TOP_K = 4

embedder = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("docs")

llm = Llama(
    model_path=LLAMA_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    n_threads=12,
    temperature=0.2
)


def answer_query(query, top_k=TOP_K):
    q_emb = embedder.encode(query).tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas","documents"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    context = "\n\n=== Retrieved passages ===\n"
    for d, m in zip(docs, metas):
        context += f"Source: {m.get('source')} page {m.get('page')} captions:{m.get('captions')}\n{d}\n\n"

    prompt = f"""
        You are a helpful assistant. Use the following retrieved passages to answer the question.
        Always include references to the source documents (file name and page number) in your answer.
        If the answer is not contained in the passages, say you don't know.

        {context}

        Question: {query}

        Answer (include references):
    """

    out = llm(prompt, max_tokens=512)
    return out["choices"][0]["text"]

if __name__ == "__main__":
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break
        ans = answer_query(q)
        print("\n--- Answer ---\n", ans)
