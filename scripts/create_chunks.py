"""
create_chunks.py

Runner script to create verse chunks from the KJV Bible.
Loads raw Bible JSON files using ingestion.py and creates 
chunks using chunking.py. Saves the resulting chunks to 
data/kjv_chunks.json for downstream embedding generation.
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.ingestion import load_kjv
from preprocessing.chunking import chunk_verses, chunk_verses_min_first

from collections import defaultdict

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "kjv_chunks.json"

# Configuration
MIN_WORDS = 120
CHUNK_OVERLAP = 2

# Ingestion
bible_verses = load_kjv(DATA_DIR / "kjv")
print(f"Loaded {len(bible_verses)} verses from the KJV Bible dataset.")

# Organize verses by book
bible_verses_by_book = defaultdict(list)
for verse in bible_verses:
    bible_verses_by_book[verse["book"]].append(verse)

# Chunking
chunks = []
for book, verses in bible_verses_by_book.items():
    book_chunks = chunk_verses_min_first(
        verses,
        min_words=MIN_WORDS,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks.extend(book_chunks)
print(f"Total chunks created: {len(chunks)}.")

# Save chunks to JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"Saved chunks to {OUTPUT_FILE}.")