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
from retrieval.reranking import rerank_chunks

EVAL_SET = [
    # Law / Commandments
    {
        "query": "Where does 'Thou shalt not steal' appear in the Bible?",
        "expected_refs": ["Exodus", "Deuteronomy"]
    },
    {
        "query": "Where does the Bible say 'Thou shalt not commit adultery'?",
        "expected_refs": ["Exodus", "Deuteronomy"]
    },

    # Creation / Narrative
    {
        "query": "In the beginning God created the heaven and the earth",
        "expected_refs": ["Genesis"]
    },
    {
        "query": "What happened when God flooded the earth?",
        "expected_refs": ["Genesis"]
    },

    # Jesus' teachings / Discourse
    {
        "query": "What did Jesus teach about loving your enemies?",
        "expected_refs": ["Matthew"]
    },
    {
        "query": "Where does Jesus tell about building on the rock versus the sand?",
        "expected_refs": ["Matthew"]
    },
    {
        "query": "Blessed are the pure in heart",
        "expected_refs": ["Matthew"]
    },
    {
        "query": "The stone the builders rejected",
        "expected_refs": ["Psalm", "Matthew"]
    },

    # Pauline letters / Epistles
    {
        "query": "What letters did Paul and Timothy write?",
        "expected_refs": ["Romans", "1 Corinthians", "2 Corinthians", "Colossians", "Philippians", "1 Timothy"]
    },
    {
        "query": "What does the Bible say about repentance and godly sorrow?",
        "expected_refs": ["2 Corinthians"]
    },

    # Prophetic / OT
    {
        "query": "Thus saith the LORD, 'Woe unto you who build your houses on the sand.'",
        "expected_refs": ["Isaiah", "Jeremiah", "Ezekiel"]  # stress test to see if it picks OT/NT
    },
    {
        "query": "The days will come when I will bring judgment upon the nations",
        "expected_refs": ["Joel", "Amos", "Ezekiel"]
    },

    # Wisdom / Proverbs
    {
        "query": "What is the meaning of wisdom?",
        "expected_refs": ["Proverbs", "Ecclesiastes", "James"]
    },
    {
        "query": "How should one live a righteous life according to the Bible?",
        "expected_refs": ["Proverbs", "Psalms", "Matthew"]
    },

    # Lookup / Reference
    {
        "query": "Where does the Bible say 'Love your neighbor as yourself'?",
        "expected_refs": ["Leviticus", "Romans", "James"]
    },
    {
        "query": "Which books contain the Ten Commandments?",
        "expected_refs": ["Exodus", "Deuteronomy"]
    },
]

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
VERSE_INDICES_FILE = BASE_DIR / "data" / "kjv_verse_indices.json"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
TOP_K = 20

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
        query = item["query"]
        expected_refs = item["expected_refs"]

        print(f"\nQuery: {query}")

        chunks = retrieve_chunks(collection, query, top_k=TOP_K)
        chunks = rerank_chunks(chunks, query)

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