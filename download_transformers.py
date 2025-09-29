from sentence_transformers import SentenceTransformer

# This will download and cache the model in ~/.cache/huggingface/ by default
SentenceTransformer("intfloat/multilingual-e5-base")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")