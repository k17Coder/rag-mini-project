from ingest import load_all_docs

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    docs = load_all_docs()
    for fname, text in docs.items():
        chunks = chunk_text(text)
        print(f"--- {fname} ---")
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk preview:\n{chunks[0][:300]}")
        print()
