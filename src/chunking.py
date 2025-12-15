"""
chunking.py

This module handles splitting the Bible verses loaded 
from ingestion.py into overlapping chunks suitable for 
embeddings in a RAG (Retrieval-Augmented Generation) system.

Each chunk contains:
- 'text': concatenated verse texts
- 'metadata': context information (book, chapter/verse range, testament, section)
"""

from ingestion import load_kjv

def chunk_verses(verses: list[dict], chunk_size: int = 10, chunk_overlap: int = 2) -> list[dict]:
    """
    Create overlapping chunks of verses for embeddings.
    
    Parameters:
        verses (list of dict): Loaded verses from ingestion.py
        chunk_size (int): Number of verses per chunk
        chunk_overlap (int): Number of verses to overlap between chunks

    Returns:
        list[dict]: Each dictionary represents a chunk:
            {
                'text': 'concatenated verse text...',
                'metadata': {
                    'book': str,
                    'chapter_start': int,
                    'verse_start': int,
                    'chapter_end': int,
                    'verse_end': int,
                    'testament': str,
                    'section': str or None
                }
            }
    """
    chunks = []
    for i in range(0, len(verses), chunk_size - chunk_overlap):
        chunk = verses[i:i + chunk_size]
        if not chunk:
            continue
        chunk_text = " ".join([v["text"] for v in chunk])
        chunk_metadata = {
            "book": chunk[0]["book"],
            "chapter_start": chunk[0]["chapter"],
            "verse_start": chunk[0]["verse"],
            "chapter_end": chunk[-1]["chapter"],
            "verse_end": chunk[-1]["verse"],
            "testament": chunk[0]["testament"],
            "section": chunk[0]["section"]
        }
        chunks.append({"text": chunk_text, "metadata": chunk_metadata})
    return chunks

if __name__ == "__main__":
    verses = load_kjv()
    chunks = chunk_verses(verses, chunk_size=10, chunk_overlap=2)
    print(f"Total chunks created: {len(chunks)}")

    # --- Sanity check: print first 2 chunks ---
    for i, chunk in enumerate(chunks[:2]):
        print(f"\nChunk {i+1}:")
        print("Text:", chunk["text"][:100] + "...")
        print("Metadata:", chunk["metadata"])