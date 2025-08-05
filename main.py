import os
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load and split your document
# ----------------------------
with open("docs/eaa_application.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Very basic chunking (split by sentence)
chunks = [chunk.strip() for chunk in text.split('.') if chunk.strip()]

# ----------------------------
# Create embeddings
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# ----------------------------
# Store embeddings in FAISS
# ----------------------------
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ----------------------------
# User query
# ----------------------------
query = "What is the return policy?"
query_vector = model.encode([query])

# Retrieve top 2 matching chunks
D, I = index.search(np.array(query_vector), k=2)
retrieved = "\n".join([chunks[i] for i in I[0]])

# ----------------------------
# Construct prompt for Ollama
# ----------------------------
prompt = f"""Answer the question using only the context below:

Context:
{retrieved}

Question:
{query}

Answer:"""

# ----------------------------
# Call Ollama API
# ----------------------------
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
)

# Check for HTTP success
response.raise_for_status()

# Parse full response
data = response.json()
print("Full Ollama Response:\n", data)

output = data["response"]
print("\n--- Final Answer ---\n", output.strip())
