"""
eval_retrieval.py

This script contains a predefined evaluation set for testing retrieval systems.
Each entry in the evaluation set consists of a query and its expected references.
"""
import sys, json
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retrieve import get_collection, retrieve_chunks
from retrieval.retrieval_preprocessing import preprocess_query

# Define evaluation set
EVAL_SET = [
    {
        "query": "Where does 'Thou shalt not steal' appear in the Bible?",
        "expected_refs": ["Exodus", "Deuteronomy"]
    },
    {
        "query": "In the beginning God created the heaven and the earth",
        "expected_refs": ["Genesis"]
    },
    {
        "query": "What did Jesus teach about loving your enemies?",
        "expected_refs": ["Matthew"]
    },
    {
        "query": "What letters did Paul and Timothy write?",
        "expected_refs": ["Romans", "1 Corinthians", "2 Corinthians", "Colossians", "Philippians", "1 Timothy"]
    },
    {
        "query": "Where does Jesus tell about building on the rock versus the sand?",
        "expected_refs": ["Matthew"]
    },
]

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
VERSE_INDICES_FILE = BASE_DIR / "data" / "kjv_verse_indices.json"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
TOP_K = 5

# Load verse indices
with open(VERSE_INDICES_FILE, "r", encoding="utf-8") as f:
    verse_indices = json.load(f)
print(f"Loaded verse indices for {len(verse_indices)} chunks.")

# Initialize ChromaDB client
collection = get_collection(str(DB_DIR), CHROMA_COLLECTION_NAME)
print(f"Loaded collection: {CHROMA_COLLECTION_NAME}")
print(f"Total documents: {collection.count()}")

def run_eval():
    for item in EVAL_SET:
        query = preprocess_query(item["query"])
        expected_refs = item["expected_refs"]

        print(f"\nQuery: {query}")

        # Retrieve top-k chunks
        chunks = retrieve_chunks(collection, query, top_k=TOP_K)

        # Check expected refs
        found_refs = []
        for chunk in chunks:
            book = chunk.get("metadata", {}).get("book", "")
            for ref in expected_refs:
                if ref in book:
                    found_refs.append(ref)

        found_refs = list(set(found_refs))
        print(f"Expected refs: {expected_refs}")
        print(f"Found refs in top-{TOP_K}: {found_refs}")

        success = "PASS" if set.intersection(set(found_refs), set(expected_refs)) else "FAIL"
        print(f"Result: {success}\n{'-'*60}")

if __name__ == "__main__":
    run_eval()