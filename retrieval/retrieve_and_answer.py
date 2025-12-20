"""
retrieve_and_answer.py

Retrieves relevant Bible text chunks from ChromaDB
using `retrieve.py` and prints them with references 
for easy inspection. Intended for testing retrieval 
before LLM-based answering is added.
"""
from pathlib import Path
from retrieve import retrieve_chunks

# Configuration
QUERY = "Paul Timothy servant gospel Jesus Christ"
TOP_K = 5

def main():
    retrieved = retrieve_chunks(QUERY, top_k=TOP_K)
    print(f"\nQuery:\n{QUERY}\n")
    print(f"Top {TOP_K} results:\n")

    for i, chunk in enumerate(retrieved, start=1):
        meta = chunk["metadata"]
        ref = f"{meta['book']} {meta['chapter_start']}:{meta['verse_start']}-{meta['verse_end']}"
        print(f"{i}. {ref}")
        print(chunk["text"][:200] + "...\n")  # Show first 200 chars of chunk for preview

if __name__ == "__main__":
    main()