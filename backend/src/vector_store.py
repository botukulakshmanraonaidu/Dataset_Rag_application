from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_store(chunks, model_name="all-MiniLM-L6-v2", store_path="./faiss_index"):
    """
    Generates embeddings and stores them in a FAISS vector store.
    """
    print(f"Generating embeddings using {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print("Indexing chunks into FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving vector store to {store_path}...")
    vector_store.save_local(store_path)
    return vector_store

def load_vector_store(store_path="./faiss_index", model_name="all-MiniLM-L6-v2"):
    """
    Loads a persisted FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    # Example usage (standalone test would require chunks)
    pass
