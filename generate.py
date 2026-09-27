import os
from dotenv import load_dotenv
from groq import Groq
from retrieve import build_index, retrieve
from embed import model
from ingest import load_all_docs
from chunk import chunk_text

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([c[0][0] for c in retrieved_chunks])
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    docs = load_all_docs()
    all_chunks = []
    for fname, text in docs.items():
        for chunk in chunk_text(text):
            all_chunks.append((chunk, fname))

    embeddings = model.encode([c[0] for c in all_chunks], show_progress_bar=True)
    index = build_index(embeddings)

    query = "What is primary key in dbms?"
    results = retrieve(query, index, all_chunks, k=3)
    answer = generate_answer(query, results)

    print(f"Query: {query}\n")
    print(f"Answer:\n{answer}")