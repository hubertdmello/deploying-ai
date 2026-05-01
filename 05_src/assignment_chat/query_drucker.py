from ask_drucker_service import _get_drucker_collection

query = input("Enter your query: ")

collection = _get_drucker_collection()
print(f"Embeddings in chroma_db: {collection.count()}")

results = collection.query(
    query_texts=[query],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

docs      = results["documents"][0]
distances = results["distances"][0]
metas     = results["metadatas"][0]

for i, (doc, dist, meta) in enumerate(zip(docs, distances, metas), start=1):
    cosine_sim = 1 - (dist ** 2) / 2
    if dist < 0.5:
        meaning = "Very strong match"
    elif dist < 0.8:
        meaning = "Good/moderate match"
    elif dist < 1.2:
        meaning = "Weak match"
    else:
        meaning = "Likely off-topic"
    print(f"\n{'='*60}")
    print(f"Chunk {i}  |  distance={dist:.4f}  |  cosine_sim={cosine_sim:.4f}  |  {meaning}  |  page={meta.get('page', 'n/a')}")
    print(f"{'='*60}")
    print(doc)
