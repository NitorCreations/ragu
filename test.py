from chromadb import PersistentClient
from pathlib import Path

client = PersistentClient(path="chroma_NitorIntra_en")  # adjust collection path
coll = client.get_or_create_collection("NitorIntra_en")
res = coll.peek()  # get a few docs
for m in res["metadatas"]:
    print(m.get("source"), m.get("source_url"), m.get("snapshot_pdf"), m.get("page"))
    file_path = m.get("source")
    if not file_path or not Path(file_path).exists():
        print(f"⚠️ PDF not found: {file_path}")
    else:
        print(f"✅  PDF found: {file_path}")
