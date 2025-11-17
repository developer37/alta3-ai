#!/usr/bin/env python3
"""
Build a tiny 'index' for CoVe:
- Read local text files
- Split into small chunks
- Call OpenAI Embeddings API to get a vector per chunk
- Save metadata + vectors for fast retrieval later
"""

import os, re, json, pathlib, requests, numpy as np

DOCS = [pathlib.Path("docs_policy.md"), pathlib.Path("docs_faq.txt")]
INDEX_DIR = pathlib.Path("index")
INDEX_DIR.mkdir(exist_ok=True)
CHUNKS_JSON = INDEX_DIR / "chunks.json"     # human-readable chunk info
VECS_NPY    = INDEX_DIR / "embeddings.npy"  # machine-usable vectors

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
EMB_URL = "https://api.openai.com/v1/embeddings"
HDRS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

def chunk(text, max_words=120, overlap=30):
    """Split long text into overlapping word windows."""
    words = re.findall(r"\w+(?:'\w+)?", text)
    i = 0
    while i < len(words):
        piece = " ".join(words[i:i+max_words])
        if piece.strip():
            yield piece
        i += max_words - overlap

def embed(batch_texts):
    """Call the Embeddings API once for a batch of texts."""
    r = requests.post(EMB_URL, headers=HDRS, json={
        "model": EMB_MODEL,
        "input": batch_texts
    }, timeout=60)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]

def main():
    # Build a list of chunks with their origin
    chunks = []
    for p in DOCS:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for i, c in enumerate(chunk(txt)):
            chunks.append({"doc": str(p), "i": i, "text": c})

    # Convert each chunk into an embedding vector
    vecs = []
    for k in range(0, len(chunks), 64):
        vecs.extend(embed([ch["text"] for ch in chunks[k:k+64]]))

    # Normalize vectors so dot product == cosine similarity
    V = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms

    # Save chunk info and vectors
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    np.save(VECS_NPY, V)

    print(f"Wrote {CHUNKS_JSON} with {len(chunks)} chunks")
    print(f"Saved normalized vectors to {VECS_NPY}  <-- this is where the vectors are!")

if __name__ == "__main__":
    main()
