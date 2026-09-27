from ingest import load_all_docs
from chunk import chunk_text
from embed import model
import faiss
import numpy as np

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve(query, index, chunks, k=3):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)
    results = [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

if __name__ == "__main__":
    docs = load_all_docs()
    all_chunks = []
    for fname, text in docs.items():
        for chunk in chunk_text(text):
            all_chunks.append((chunk, fname))

    chunk_texts = [c[0] for c in all_chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    index = build_index(embeddings)

    test_query = "What is an operating system?"
    results = retrieve(test_query, index, all_chunks, k=3)

    print(f"Query: {test_query}\n")
    for i, (chunk, dist) in enumerate(results):
        print(f"Result {i+1} (distance: {dist:.3f}, source: {chunk[1]})")
        print(chunk[0][:200])
        print()