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
import chromadb, json

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
VERSE_INDECES_FILE = BASE_DIR / "data" / "kjv_verse_indeces.json"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
TEST_QUERY = "Gideon deliverance Midianites"
TOP_K = 5

# Load verse indeces
with open(VERSE_INDECES_FILE, "r", encoding="utf-8") as f:
    verse_indeces = json.load(f)
print(f"Loaded verse indices for {len(verse_indeces)} chunks.")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
print(f"Loaded collection: {CHROMA_COLLECTION_NAME}")
print(f"Total documents: {collection.count()}")

# Perform retrieval
results = collection.query(
    query_texts=[TEST_QUERY],
    n_results=TOP_K,
    include=["documents", "metadatas"]
)

# Display results
print("\nQuery:")
print(TEST_QUERY)
print("\nTop results:\n")

for rank, (doc, meta, chunk_id) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["ids"][0]
    ),
    start=1
):
    print(f"{rank}. {meta['book']} "
          f"{meta['chapter_start']}:{meta['verse_start']}-"
          f"{meta['chapter_end']}:{meta['verse_end']}")

    chunk_verse_indeces = verse_indeces.get(chunk_id)

    if not chunk_verse_indeces:
        print("  [No verse indices found for this chunk]\n")
        continue

    for v in chunk_verse_indeces:
        verse_text = doc[v["start"]:v["end"]].strip()
        verse_ref = f"{meta['book']} {v['chapter']}:{v['verse']}"
        print(f"  {verse_ref} — {verse_text}")

    print()