import os
import json
import openai
import faiss
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")
client = openai.OpenAI(api_key=api_key)

def get_embedding(text: str) -> List[float]:
    """
    Uses OpenAI's text-embedding-ada-002 model to convert text into an embedding.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def build_faiss_index(documents: List[Dict]) -> (faiss.IndexFlatL2, List[Dict]):
    """
    Computes embeddings for each document and builds a FAISS index.
    Returns the FAISS index and a mapping (list) that associates each index position with its document.
    """
    # Determine the embedding dimension using a sample
    sample_emb = get_embedding("sample")
    dim = len(sample_emb)
    
    # Create a FAISS index for Euclidean (L2) distance
    index = faiss.IndexFlatL2(dim)
    
    embeddings = []
    mapping = []  # To keep track of document details
    for doc in documents:
        emb = get_embedding(doc["content"])
        embeddings.append(emb)
        mapping.append(doc)
    
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index, mapping

def save_index(index: faiss.IndexFlatL2, mapping: List[Dict], index_file: str, mapping_file: str):
    """
    Saves the FAISS index to disk and writes the document mapping as JSON.
    """
    faiss.write_index(index, index_file)
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f)

def load_documents() -> List[Dict]:
    """
    Loads all documents from the 'rag' folder.
    """
    documents = []
    rag_folder = 'rag'
    
    # Ensure the rag folder exists
    if not os.path.exists(rag_folder):
        raise ValueError(f"The '{rag_folder}' folder does not exist.")
    
    # Read all text files from the rag folder
    for filename in os.listdir(rag_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(rag_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Use the filename without extension as the title
            title = os.path.splitext(filename)[0]
            documents.append({
                "title": title,
                "content": content
            })
    
    if not documents:
        raise ValueError(f"No text files found in the '{rag_folder}' folder.")
    
    return documents

if __name__ == "__main__":
    docs = load_documents()
    index, mapping = build_faiss_index(docs)
    # Save the index and mapping so that future queries do not need to rebuild the index
    save_index(index, mapping, "faiss_index.bin", "mapping.json")
    print("Index built and saved successfully.")
