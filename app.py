import streamlit as st
from ingest import load_all_docs
from chunk import chunk_text
from embed import model
from retrieve import build_index, retrieve
from generate import generate_answer

st.title("Chat with your documents")
st.caption("Ask a question and get an answer grounded in your uploaded notes, with sources.")

@st.cache_resource
def setup():
    docs = load_all_docs()
    all_chunks = []
    for fname, text in docs.items():
        for chunk in chunk_text(text):
            all_chunks.append((chunk, fname))
    embeddings = model.encode([c[0] for c in all_chunks], show_progress_bar=True)
    index = build_index(embeddings)
    return index, all_chunks

with st.spinner("Loading and indexing your documents..."):
    index, all_chunks = setup()

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Retrieving and generating answer..."):
        results = retrieve(query, index, all_chunks, k=3)
        answer = generate_answer(query, results)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for chunk, score in results:
        st.markdown(f"**Source:** {chunk[1]} | **Distance:** {score:.3f}")
        st.write(chunk[0][:300] + "...")
        st.divider()