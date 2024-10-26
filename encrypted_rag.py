# embeddings
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries.",
    "Encrypted search enables privacy-preserving data retrieval."
]

# Generate embeddings
embeddings = model.encode(documents)

print(f"### Embeddings len: ({len(embeddings[0])})") #: {embeddings[0]}")

# encryption
import tenseal as ts

# Initialize TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2 ** 40

# Encrypt embeddings
encrypted_embeddings = []
for emb in embeddings:
    enc_emb = ts.ckks_vector(context, emb)
    encrypted_embeddings.append(enc_emb)

# query enc
# User query
query = "How does encrypted search work?"

# Generate query embedding
query_embedding = model.encode([query])[0]

# Encrypt query embedding
enc_query_embedding = ts.ckks_vector(context, query_embedding)

print(f"### Encrypted query Embedding: {enc_query_embedding}")

# encrypted dot product
# Compute encrypted similarity scores (dot products)
encrypted_scores = []
for enc_emb in encrypted_embeddings:
    # Element-wise multiplication and summation (dot product)
    enc_score = enc_emb.dot(enc_query_embedding)
    encrypted_scores.append(enc_score)

print(f"### Encrypted Scores: {encrypted_scores}")

# get top k
# Decrypt similarity scores
decrypted_scores = [enc_score.decrypt()[0] for enc_score in encrypted_scores]

print(f"### Decrypted Scores: {decrypted_scores}")


import numpy as np

# Number of top results to retrieve
K = 2

# Get indices of top K similarity scores
top_k_indices = np.argsort(decrypted_scores)[-K:][::-1]

# Retrieve top documents
top_documents = [documents[i] for i in top_k_indices]

# LLM usage
import ollama
# Prepare the prompt for the LLM
prompt = f"Answer the following question based on the provided documents.\n\nQuestion: {query}\n\nDocuments:\n"
for idx, doc in enumerate(top_documents):
    prompt += f"{idx+1}. {doc}\n"

print(f"### Prompt: {prompt}")

try:
    response = ollama.generate(model="llama3.1", prompt=prompt)
except Exception as e:
    print(f"Error generating ollama response: {e}")

print(f"### Response: {response['response']}")
