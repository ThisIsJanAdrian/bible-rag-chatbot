"""
retrieve.py

Core retrieval functions for Bible RAG chatbot.
Provides reusable functions to query the ChromaDB
collection and return structured chunks of Bible text.
"""

from pathlib import Path
import chromadb

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
TOP_K = 5

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-k relevant Bible chunks for a query.

    Parameters:
        query (str): Natural language query string.
        top_k (int): Number of top results to return.

    Returns:
        list[dict]: Each dict contains:
            {
                "text": str,          # chunk text
                "metadata": dict      # chunk metadata (book, chapter_start, verse_start, chapter_end, verse_end, testament, section)
            }
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    retrieved = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "text": doc,
            "metadata": meta
        })
    return retrieved
