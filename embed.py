from ingest import load_all_docs
from chunk import chunk_text
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=True)

if __name__ == "__main__":
    docs = load_all_docs()
    all_chunks = []
    for fname, text in docs.items():
        for chunk in chunk_text(text):
            all_chunks.append((chunk, fname))

    print(f"Total chunks across all docs: {len(all_chunks)}")
    embeddings = embed_chunks([c[0] for c in all_chunks])
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Example vector (first 10 dims): {embeddings[0][:10]}")