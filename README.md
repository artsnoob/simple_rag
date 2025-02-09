# RAG Implementation with FAISS and OpenAI

This project implements a Retrieval-Augmented Generation (RAG) system using FAISS for efficient similarity search and OpenAI's embeddings and chat models for semantic understanding and response generation.

## Technology Stack

- **Python 3**: Core programming language
- **OpenAI API**: Used for two purposes:
  - `text-embedding-ada-002` model for generating document embeddings
  - `gpt-4o-mini` for generating responses based on retrieved context
- **FAISS (Facebook AI Similarity Search)**: High-performance library for similarity search and clustering of dense vectors
- **NumPy**: For numerical operations and array manipulations
- **python-dotenv**: For managing environment variables

## How It Works

The RAG system operates in two phases:

### 1. Indexing Phase (build_index.py)

1. **Document Loading**:
   - Reads all `.txt` files from the `rag` folder
   - Each document is stored with its filename (without extension) as the title

2. **Embedding Generation**:
   - Each document's content is converted into a high-dimensional vector using OpenAI's `text-embedding-ada-002` model
   - These embeddings capture the semantic meaning of the text

3. **Index Building**:
   - FAISS creates an L2 (Euclidean distance) index from the document embeddings
   - A mapping file is maintained to link index positions to original documents
   - Both the FAISS index and mapping are saved to disk (`faiss_index.bin` and `mapping.json`)

### 2. Query Phase (query_index.py)

1. **Query Processing**:
   - User's question is converted into an embedding using the same OpenAI model

2. **Retrieval**:
   - FAISS searches for the most similar document embeddings
   - Configurable parameters:
     - `top_k`: Number of candidates to retrieve (default: 5)
     - `threshold`: Minimum similarity score (default: 0.7)
   - Results are filtered based on similarity scores to ensure relevance

3. **Response Generation**:
   - Retrieved context is passed to the chat model
   - GPT model generates a response based on the relevant context

## Setup

1. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. Install required packages:
   ```bash
   pip3 install openai faiss-cpu numpy python-dotenv
   ```

3. Place your documents in the `rag` folder as `.txt` files

## Usage

1. First, build the index:
   ```bash
   python3 build_index.py
   ```

2. Then, query the system:
   ```bash
   python3 query_index.py
   ```

3. Enter your question when prompted

The system will:
- Show retrieved context with similarity scores
- Provide a response based on the relevant context

## Implementation Details

- Uses L2 (Euclidean) distance for similarity search
- Converts L2 distances to similarity scores (0-1 scale)
- Filters out less relevant documents using a similarity threshold
- Maintains document mappings for result tracking
- Supports multiple document retrieval with relevance scoring

## Project Structure

```
.
├── .env                  # Environment variables
├── build_index.py       # Index building script
├── query_index.py       # Query processing script
├── faiss_index.bin      # Generated FAISS index
├── mapping.json         # Document mapping file
└── rag/                 # Document folder
    └── *.txt           # Source documents
