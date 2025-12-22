"""
chunking.py

Contains functions to create chunks of Bible verses 
for embeddings. Two approaches are implemented:

1. verse-based chunking: fixed number of verses per chunk
2. min-word-based chunking: accumulate verses until a 
minimum word threshold is reached

Each function returns a list of dictionaries with chunk 
text and metadata.
"""

from pathlib import Path

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def chunk_verses(verses: list[dict], chunk_size: int = 7, chunk_overlap: int = 2) -> list[dict]:
    """
    Create overlapping chunks of verses for embeddings.
    Note that this function chunks based on verse count, not token count.
    
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

def chunk_verses_min_first(verses: list[dict], min_words: int = 120, chunk_overlap: int = 2) -> list[dict]:
    """
    Create overlapping chunks of verses for embeddings 
    using a minimum word threshold. Note that this 
    function chunks based on word count, not verse count.

    Parameters:
        verses (list of dict): Loaded verses from ingestion.py
        min_words (int, optional): Minimum number of words per chunk. Defaults to 120.
        chunk_overlap (int, optional): Number of verses to overlap between chunks. Defaults to 2.

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

    Notes:
        - Chunks accumulate verses until the total word count reaches at least min_words.
        - Verses are never split across chunks.
        - The final chunk may contain fewer words than min_words if there are not enough remaining verses.
        - The chunk_overlap parameter ensures semantic continuity between adjacent chunks.
    """
    chunks = []
    current_chunk = []
    current_word_count = 0

    i = 0

    while i < (len(verses)):
        verse = verses[i]
        current_chunk.append(verse)
        current_word_count += len(verse["text"].split())
        if current_word_count >= min_words:
            chunk_text = " ".join([v["text"] for v in current_chunk])
            chunk_metadata = {
                "book": current_chunk[0]["book"],
                "chapter_start": current_chunk[0]["chapter"],
                "verse_start": current_chunk[0]["verse"],
                "chapter_end": current_chunk[-1]["chapter"],
                "verse_end": current_chunk[-1]["verse"],
                "testament": current_chunk[0]["testament"],
                "section": current_chunk[0]["section"]
            }
            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

            if chunk_overlap > 0:
                current_chunk = current_chunk[-chunk_overlap:]
                current_word_count = sum(
                    len(v["text"].split()) for v in current_chunk
                )
            else:
                current_chunk = []
                current_word_count = 0
        i += 1
    
    # Add any remaining verses as a final chunk
    if current_chunk:
        chunk_text = " ".join([v["text"] for v in current_chunk])
        chunk_metadata = {
            "book": current_chunk[0]["book"],
            "chapter_start": current_chunk[0]["chapter"],
            "verse_start": current_chunk[0]["verse"],
            "chapter_end": current_chunk[-1]["chapter"],
            "verse_end": current_chunk[-1]["verse"],
            "testament": current_chunk[0]["testament"],
            "section": current_chunk[0]["section"]
        }
        chunks.append({"text": chunk_text, "metadata": chunk_metadata})

    return chunks

def chunk_verses_min_first_with_indexing(verses: list[dict], min_words: int = 120, chunk_overlap: int = 2) -> list[dict]:
    """
    Create overlapping chunks of verses for embeddings 
    using a minimum word threshold, while preserving 
    verse-level character indices within each chunk.

    Parameters:
        verses (list of dict): Loaded verses from ingestion.py
        min_words (int, optional): Minimum number of words per chunk. Defaults to 120.
        chunk_overlap (int, optional): Number of verses to overlap between chunks. Defaults to 2.

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
                },
                'verse_indices': [
                    {
                        'chapter': int,
                        'verse': int,
                        'start': int,
                        'end': int
                    },
                    ...
                ]
            }

    Notes:
        - Chunks accumulate verses until the total word count reaches at least min_words.
        - Verses are never split across chunks.
        - The final chunk may contain fewer words than min_words if there are not enough remaining verses.
        - The chunk_overlap parameter ensures semantic continuity between adjacent chunks.
        - Verse indices are character indices relative to the chunk's text field and include all characters (spaces, punctuation, and paragraph markers such as '¶').
        - Verse indices enable exact verse retrieval and paragraph-based extraction without re-chunking or re-embedding.
    """
    chunks = []
    current_chunk = []
    current_word_count = 0
    i = 0

    while i < (len(verses)):
        verse = verses[i]
        current_chunk.append(verse)
        current_word_count += len(verse["text"].split())

        if current_word_count >= min_words:
            chunk_text_parts = []
            verse_indices = []
            cursor = 0

            for v in current_chunk:
                text = v["text"]

                start = cursor
                end = start + len(text)

                verse_indices.append({
                    "chapter": v["chapter"],
                    "verse": v["verse"],
                    "start": start,
                    "end": end
                })

                chunk_text_parts.append(text)
                cursor = end + 1  # space separator

            chunk_text = " ".join(chunk_text_parts)
            chunk_metadata = {
                "book": current_chunk[0]["book"],
                "chapter_start": current_chunk[0]["chapter"],
                "verse_start": current_chunk[0]["verse"],
                "chapter_end": current_chunk[-1]["chapter"],
                "verse_end": current_chunk[-1]["verse"],
                "testament": current_chunk[0]["testament"],
                "section": current_chunk[0]["section"]
            }
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata,
                "verse_indices": verse_indices
            })

            if chunk_overlap > 0:
                current_chunk = current_chunk[-chunk_overlap:]
                current_word_count = sum(
                    len(v["text"].split()) for v in current_chunk
                )
            else:
                current_chunk = []
                current_word_count = 0
        i += 1
    
    # Add any remaining verses as a final chunk
    if current_chunk:
        chunk_text_parts = []
        verse_indices = []
        cursor = 0

        for v in current_chunk:
            text = v["text"]

            start = cursor
            end = start + len(text)

            verse_indices.append({
                "chapter": v["chapter"],
                "verse": v["verse"],
                "start": start,
                "end": end
            })

            chunk_text_parts.append(text)
            cursor = end + 1  # space separator

        chunk_text = " ".join(chunk_text_parts)
        chunk_metadata = {
            "book": current_chunk[0]["book"],
            "chapter_start": current_chunk[0]["chapter"],
            "verse_start": current_chunk[0]["verse"],
            "chapter_end": current_chunk[-1]["chapter"],
            "verse_end": current_chunk[-1]["verse"],
            "testament": current_chunk[0]["testament"],
            "section": current_chunk[0]["section"]
        }
        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata,
            "verse_indices": verse_indices
        })
    return chunks

if __name__ == "__main__":
    from ingestion import load_kjv
    verses = load_kjv(DATA_DIR / "kjv")
    # chunks = chunk_verses(verses, chunk_size=10, chunk_overlap=2)
    chunks = chunk_verses_min_first_with_indexing(verses, min_words=120, chunk_overlap=2)
    print(f"Total chunks created: {len(chunks)}")

    # --- Sanity check: print first 2 chunks ---
    # for i, chunk in enumerate(chunks[:2]):
    #     print(f"\nChunk {i+1}:")
    #     print("Text:", chunk["text"][:100] + "...")
    #     print("Metadata:", chunk["metadata"])
    
    # --- Sanity check: print specified range of verses ---
    for i in range (10000, 10001):
        chunk = chunks[i]
        for j in range(len(chunk["verse_indices"])):
            v = chunk["verse_indices"][j]
            print(chunk["text"][v["start"]:v["end"]])