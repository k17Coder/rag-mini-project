from pypdf import PdfReader
import os

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_all_docs(folder="docs"):
    all_text = {}
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            all_text[fname] = load_pdf(os.path.join(folder, fname))
        elif fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                all_text[fname] = f.read()
    return all_text

if __name__ == "__main__":
    docs = load_all_docs()
    for fname, text in docs.items():
        print(f"--- {fname} ---")
        print(f"Length: {len(text)} characters")
        print(text[:300])
        print()