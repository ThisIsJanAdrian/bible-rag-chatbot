"""
test_retrieval.py
Script to test semantic retrieval from ChromaDB collection.
Loads the existing collection, embeds a test query, and 
retrieves similar chunks to verify correct storage and 
retrieval. This script assumes that the collection has 
already been populated with data.

This script does NOT perform embedding.
"""
from pathlib import Path
import chromadb

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
TEST_QUERY = "Paul Timothy servant gospel Jesus Christ"
TOP_K = 5

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
print(f"Loaded collection: {CHROMA_COLLECTION_NAME}")
print(f"Total documents: {collection.count()}")

# Perform retrieval
results = collection.query(
    query_texts=[TEST_QUERY],
    n_results=TOP_K
)
print("\nQuery:")
print(TEST_QUERY)
print("\nTop results:\n")

for i, (doc, meta) in enumerate(
    zip(results["documents"][0], results["metadatas"][0]),
    start=1
):
    ref = f"{meta['book']} {meta['chapter_start']}:{meta['verse_start']}-{meta['verse_end']}"
    print(f"{i}. {ref}")
    print(doc, "\n")