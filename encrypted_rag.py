"""
Implements an Encrypted Retrieval-Augmented Generation (RAG) model using TenSEAL and SentenceTransformers. LLM runs locally via Ollama.
"""

import tenseal as ts
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama
import logging
import os
import nltk
from nltk import sent_tokenize

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]:%(funcName)s %(message)s",
)
nltk.download('punkt_tab', quiet=True)

class EncryptedRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=512):
        logging.debug("Initializing EncryptedRAG instance")
        self.model = SentenceTransformer(model_name)
        # Initialize TenSEAL context
        self.context = self._create_context()
        # Initialize storage for documents and encrypted embeddings
        self.documents = []
        self.encrypted_embeddings = []
        self.chunk_size = chunk_size

    def _create_context(self):
        # Private method to create TenSEAL context
        logging.debug("Creating TenSEAL context")
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.generate_galois_keys()
        context.global_scale = 2 ** 40
        return context

    def add_documents(self, inputs):
        # Add documents and generate encrypted embeddings
        documents = self._process_inputs(inputs)
        chunks = self._chunk_documents(documents)
        
        embeddings = self.model.encode(chunks)
        for i, emb in enumerate(embeddings):
            enc_emb = ts.ckks_vector(self.context, emb)
            self.encrypted_embeddings.append(enc_emb)
            self.documents.append(chunks[i])

    def _process_inputs(self, inputs):
        # Converts inputs into list of strings (docs)
        # supports text files and strings (and lists of each)
        documents = []
        if isinstance(inputs, str):
            if os.path.isfile(inputs):
                # input is a filename
                with open(inputs, 'r') as f:
                    content = f.read()
                    documents.append(content)
            else:
                # input is a string
                documents.append(inputs)
        elif isinstance(inputs, list):
            for item in inputs:
                documents.extend(self._process_inputs(item))
        else:
            raise ValueError("Invalid input type. Must be a string, file path, or list of either types.")
        return documents
    
    def _chunk_documents(self, documents):
        # Breaks documents into chunks of max length chunk_size
        chunks = []
        for doc in documents:
            sentences = sent_tokenize(doc)
            current_chunk = ""
            current_length = 0
            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length <= self.chunk_size:
                    current_chunk +=  " "+ sentence
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_length = sentence_length
            if current_chunk:
                chunks.append(current_chunk.strip())
        return chunks

    def encrypt_query(self, query):
        # Generate and encrypt query embedding
        query_embedding = self.model.encode([query])[0]
        enc_query_embedding = ts.ckks_vector(self.context, query_embedding)
        return enc_query_embedding

    def compute_similarity(self, enc_query_embedding):
        # Compute encrypted similarity scores (dot products)
        encrypted_scores = []
        for enc_emb in self.encrypted_embeddings:
            enc_score = enc_emb.dot(enc_query_embedding)
            encrypted_scores.append(enc_score)
        return encrypted_scores

    def decrypt_scores(self, encrypted_scores):
        # Decrypt similarity scores
        decrypted_scores = [enc_score.decrypt()[0] for enc_score in encrypted_scores]
        return decrypted_scores

    def get_top_k_documents(self, decrypted_scores, K=2):
        # Get indices of top K similarity scores
        logging.debug(f"Getting top K documents, K={K}")
        top_k_indices = np.argsort(decrypted_scores)[-K:][::-1]
        # Retrieve top documents
        top_documents = [self.documents[i] for i in top_k_indices]
        return top_documents

    def generate_response(self, query, top_documents, model_name='llama3.1'):
        # Prepare the prompt for the LLM
        prompt = f"Answer the following question based on the provided documents.\n\nQuestion: {query}\n\nDocuments:\n"
        for idx, doc in enumerate(top_documents):
            prompt += f"{idx+1}. {doc}\n"
        # Generate response using Ollama
        try:
            response = ollama.generate(model=model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            logging.error(f"Error generating Ollama response: {e}")
            return None
