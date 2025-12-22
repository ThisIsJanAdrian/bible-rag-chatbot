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

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
VERSE_INDICES_FILE = BASE_DIR / "data" / "kjv_verse_indices.json"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
QUERY = "For even the Son of man came not to be ministered unto"
TOP_K = 5

# Load verse indices
with open(VERSE_INDICES_FILE, "r", encoding="utf-8") as f:
    verse_indices = json.load(f)
print(f"Loaded verse indices for {len(verse_indices)} chunks.")

# Initialize ChromaDB client
collection = get_collection(str(DB_DIR), CHROMA_COLLECTION_NAME)
print(f"Loaded collection: {CHROMA_COLLECTION_NAME}")
print(f"Total documents: {collection.count()}")

# Perform retrieval
results = retrieve_chunks(collection, QUERY, top_k=3)

# Display results
print("\nQuery:")
print(QUERY)
print("\nTop results:\n")

for rank, item in enumerate(results, start=1):
    doc = item["text"]
    meta = item["metadata"]
    chunk_id = item["id"]

    # Reference formatting
    if meta["chapter_start"] == meta["chapter_end"]:
        ref = f"{meta['book']} {meta['chapter_start']}:{meta['verse_start']}-{meta['verse_end']}"
    else:
        ref = (
            f"{meta['book']} "
            f"{meta['chapter_start']}:{meta['verse_start']}-"
            f"{meta['chapter_end']}:{meta['verse_end']}"
        )

    print(f"{rank}. {ref}")

    chunk_verse_indices = verse_indices.get(chunk_id)

    if not chunk_verse_indices:
        print("  [No verse indices found for this chunk]\n")
        continue

    for v in chunk_verse_indices:
        verse_text = doc[v["start"]:v["end"]].strip()
        verse_ref = f"{meta['book']} {v['chapter']}:{v['verse']}"
        print(f"  {verse_ref} — {verse_text}")

    print()