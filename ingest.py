import json
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection("policies")

with open("docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

for doc in docs:

    metadata_text = ""

    metadata = {}

    for k, v in doc["metadata"].items():

        if isinstance(v, list):
            metadata[k] = ", ".join(map(str, v)) if len(v) > 0 else "none"
        else:
            metadata[k] = str(v)

        metadata_text += f"{k}: {metadata[k]} "

    full_text = f"""
    Title: {doc['title']}
    Content: {doc['content']}
    Metadata: {metadata_text}
    """

    embedding = model.encode(full_text).tolist()

    collection.add(
        ids=[str(doc["id"])],
        documents=[full_text],
        embeddings=[embedding],
        metadatas=[metadata]
    )

print("Loaded Successfully")