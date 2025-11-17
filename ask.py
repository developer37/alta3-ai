#!/usr/bin/env python3
"""
ask.py
- Turn a user's question into an embedding (query vector)
- Retrieve the most similar chunks via cosine similarity
- Ask an LLM to answer ONLY using those chunks
- Include inline citations and a Sources list
"""

import os, json, pathlib, requests, numpy as np

CHUNKS_JSON = pathlib.Path("index/chunks.json")
VECS_NPY    = pathlib.Path("index/embeddings.npy")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMB_MODEL  = os.environ.get("EMB_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

EMB_URL  = "https://api.openai.com/v1/embeddings"
CHAT_URL = "https://api.openai.com/v1/chat/completions"
HDRS = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

THRESH = 0.15   # if top similarity is below this, refuse ("I do not know")
TOPK   = 3      # how many chunks to include as context
MAX_TOKENS = 250

def embed(text):
    """Return a single L2-normalized embedding for the input text."""
    r = requests.post(EMB_URL, headers=HDRS, json={
        "model": EMB_MODEL,
        "input": text
    }, timeout=60)
    r.raise_for_status()
    vec = np.array(r.json()["data"][0]["embedding"], dtype="float32")
    n = np.linalg.norm(vec) or 1.0
    return vec / n

def call_llm(system_msg, user_msg):
    """Call the Chat Completions API with a system + user message."""
    r = requests.post(
        CHAT_URL,
        headers={**HDRS, "Content-Type": "application/json"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2
        },
        timeout=60
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def main():
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    V = np.load(VECS_NPY)   # shape: (num_chunks, dim)

    q = input("Ask a question: ").strip()
    qv = embed(q)

    sims = V @ qv
    idx = sims.argsort()[::-1][:TOPK]
    best = float(sims[idx[0]]) if len(idx) else 0.0

    if best < THRESH:
        print("I do not know. I could not find relevant context.")
        return

    ctx_lines, src_lines = [], []
    for j, i in enumerate(idx, start=1):
        ctx_lines.append(f"[{j}] {chunks[i]['text']}")
        src_lines.append(f"[{j}] {chunks[i]['doc']}#chunk{chunks[i]['i']}")
    context = "\n\n".join(ctx_lines)
    sources = "\n".join(src_lines)

    system = "Answer ONLY using the provided context. If the answer is not present, say you do not know."
    user = f"""Question: {q}

Context:
{context}

Format:
- A concise answer with inline citations like [1], [2].
- Then a 'Sources' list mapping numbers to file names.

Sources:
{sources}
"""

    print("\nAnswer:\n", call_llm(system, user))

if __name__ == "__main__":
    main()
