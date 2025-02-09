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

def load_index(index_file: str, mapping_file: str) -> (faiss.IndexFlatL2, List[Dict]):
    """
    Loads the FAISS index and the associated document mapping from disk.
    """
    index = faiss.read_index(index_file)
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return index, mapping

def retrieve_relevant_context(query: str, index: faiss.IndexFlatL2, mapping: List[Dict], top_k: int = 5, threshold: float = 0.8) -> str:
    """
    Embeds the query, searches the FAISS index, and returns the concatenated content
    of the most similar documents that meet the similarity threshold.
    """
    query_emb = get_embedding(query)
    query_np = np.array([query_emb]).astype('float32')
    distances, indices = index.search(query_np, top_k)
    
    # Convert L2 distances to similarity scores (0 to 1)
    # L2 distance of 0 means perfect similarity (1.0)
    # Larger L2 distances mean less similarity
    max_l2_distance = 2.0  # Approximate max L2 distance for normalized embeddings
    similarities = 1 - (distances[0] / max_l2_distance)
    
    retrieved_texts = []
    for sim, idx in zip(similarities, indices[0]):
        if idx < len(mapping) and sim >= threshold:
            doc = mapping[idx]
            retrieved_texts.append(f"Document '{doc['title']}' (similarity: {sim:.2f}): {doc['content']}")
    return "\n\n".join(retrieved_texts)

if __name__ == "__main__":
    # Load the prebuilt index and document mapping
    index, mapping = load_index("faiss_index.bin", "mapping.json")
    
    # Get the question from user input
    print("\nEnter your question about the documents:")
    query_text = input("> ")
    
    print("\nSearching through documents for relevant information...")
    
    # Retrieve relevant context from the index
    retrieved_context = retrieve_relevant_context(query_text, index, mapping, top_k=5, threshold=0.7)
    print("=== Retrieved Context ===")
    print(retrieved_context)
    
    # Build the prompt for the chat model
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
        {"role": "user", "content": f"Context:\n{retrieved_context}\n\nQuestion:\n{query_text}"}
    ]
    
    # Call the Chat Completion API with the augmented prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the correct model name if necessary
        messages=messages,
        max_tokens=512,
        temperature=0.7
    )
    
    print("\n=== GPT-4o-mini's Response ===")
    print(response.choices[0].message.content)
