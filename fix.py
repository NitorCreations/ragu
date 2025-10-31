from chromadb import PersistentClient

client = PersistentClient(path="chroma_Terveys_fi")
coll = client.get_or_create_collection("Terveys_fi")

# Fetch all IDs to process (we'll use get() in chunks)
BATCH_SIZE = 100
offset = 0

while True:
    batch = coll.get(include=["metadatas", "documents", "embeddings"], offset=offset, limit=BATCH_SIZE)
    ids = batch["ids"]
    if not ids:
        break

    new_ids = []
    new_docs = []
    new_embeds = []
    new_metas = []

    for i, m, d, e in zip(batch["ids"], batch["metadatas"], batch["documents"], batch["embeddings"]):
        source = m.get("source_url")
        if source and source.endswith(" - Terveyskirjasto"):
            new_source = source.removesuffix(" - Terveyskirjasto")
            m["source_url"] = source
            print(f"✂️ Updated source_url: {source} -> {new_source}")

        new_ids.append(i)
        new_docs.append(d)
        new_embeds.append(e)
        new_metas.append(m)

    # Re-add the batch with updated metadata (same IDs = overwrite)
    if new_ids:
        coll.add(ids=new_ids, documents=new_docs, embeddings=new_embeds, metadatas=new_metas)

    offset += BATCH_SIZE

print("✅ Metadata update complete.")
