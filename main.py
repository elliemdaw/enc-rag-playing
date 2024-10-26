# main.py

from encrypted_rag import EncryptedRAG

def main():
    rag = EncryptedRAG()

    # 'Document store' TODO - replace with actual document store
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries.",
        "Encrypted search enables privacy-preserving data retrieval."
    ]
    rag.add_documents(documents)

    # User query
    query = "How does encrypted search work?"

    # Encrypt query embedding
    enc_query_embedding = rag.encrypt_query(query)

    # Compute encrypted similarity scores
    encrypted_scores = rag.compute_similarity(enc_query_embedding)

    # Decrypt similarity scores
    decrypted_scores = rag.decrypt_scores(encrypted_scores)

    # Get top K documents
    top_documents = rag.get_top_k_documents(decrypted_scores, K=2)
    print("Top documents:")
    print(top_documents)

    # Generate response using LLM
    response = rag.generate_response(query, top_documents, model_name='llama3.1')

    # Display the response
    if response:
        print("LLM Response:")
        print(response)

if __name__ == "__main__":
    main()
