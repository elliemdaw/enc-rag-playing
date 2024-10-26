# Encrypted RAG PoC
Not sure yet if/how this would be useful, but wanted to play around with it nonetheless.
This runs locally using Ollama.

### Setup and Installation
- Requires ollama installed and running on your machine
`pip install -r requirements.txt`
### Usage
Go into `main.py` and set up the document store and your query.
Then, `python main.py`

### Details
- The encryption of vectors is done using TenSEAL, using the CKKS scheme
- Dot product is used as a proxy for similarity search (higher numbers indicate more similar vectors)

### Next steps
- [x] Refactor into classes
- [ ] Allow user to provide query
- [x] Allow user to provide documents for RAG
- [ ] Move sentence transformers locally as well
- [ ] Actual storage for embeddings

### References:
TenSEAL: https://github.com/OpenMined/TenSEAL
Microsoft SEAL: https://github.com/microsoft/SEAL
Sentence-Transformers: https://www.sbert.net/
Ollama: https://github.com/ollama/ollama
CKKS: Cheon et al., "Homomorphic Encryption for Arithmetic of Approximate Numbers" (https://eprint.iacr.org/2016/421)