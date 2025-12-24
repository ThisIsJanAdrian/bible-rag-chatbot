"""
test_retrieval.py
Script to test semantic retrieval from ChromaDB collection.
Loads the existing collection, embeds a test query, and 
retrieves similar chunks to verify correct storage and 
retrieval. This script assumes that the collection has 
already been populated with data.

This script does NOT perform embedding.
"""
import json, sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retrieve import get_collection, retrieve_chunks
from retrieval.format_context import format_context
from utils.retrieval_preprocessing import preprocess_query

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
VERSE_INDICES_FILE = BASE_DIR / "data" / "kjv_verse_indices.json"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
TOP_K = 10

# Load verse indices
with open(VERSE_INDICES_FILE, "r", encoding="utf-8") as f:
    verse_indices = json.load(f)
print(f"Loaded verse indices for {len(verse_indices)} chunks.")

# Initialize ChromaDB client
collection = get_collection(str(DB_DIR), CHROMA_COLLECTION_NAME)
print(f"Loaded collection: {CHROMA_COLLECTION_NAME}")
print(f"Total documents: {collection.count()}")

# Acquire and preprocess user query
user_query = input("\nAsk a Bible question:\n> ")
print(f"User query: {user_query}")
clean_query = preprocess_query(user_query)
print(f"Preprocessed query: {clean_query}")

# Perform retrieval
results = retrieve_chunks(collection, clean_query, TOP_K)

# Display results
print("\nTop results:")
formatted_context = format_context(results, verse_indices)
print(formatted_context)