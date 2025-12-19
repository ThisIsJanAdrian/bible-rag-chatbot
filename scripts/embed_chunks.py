"""
embed_chunks.py

Contains functions to embed Bible text chunks and store 
them in a persistent vector database. Each chunk is 
stored with its text, metadata (book, chapter, verse, 
testament, section), and a unique ID. This script is 
intended to be run once (or deliberately rerun) and 
does not handle querying or LLM interaction.
"""
import json
import uuid
import math
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "kjv_chunks.json"
DB_DIR = BASE_DIR / "data" / "chroma_db"

# Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
DEVICE = "cpu"
EMBED_BATCH_SIZE = 128
INSERT_BATCH_SIZE = 5000

# Load chunks
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}.")

# Load embedding model
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    trust_remote_code=True,
    device=DEVICE)
print("Embedding model loaded.")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=str(DB_DIR))
collection_name = CHROMA_COLLECTION_NAME
try:
    collection = client.get_collection(name=collection_name)
    print(f"Using existing collection: {collection_name}")
except chromadb.errors.NotFoundError:
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME,
            device=DEVICE
        )
    )
    print(f"Created new collection: {collection_name}.")

# Prepare data for insertion
ids = []
texts = []
metadatas = []
for chunk in chunks:
    chunk_id = str(uuid.uuid4())
    ids.append(chunk_id)
    texts.append(chunk["text"])

    # Ensure metadata types are correct for ChromaDB storage
    clean_metadata = {
        "book": str(chunk["metadata"]["book"]),
        "chapter_start": int(chunk["metadata"]["chapter_start"]),
        "verse_start": int(chunk["metadata"]["verse_start"]),
        "chapter_end": int(chunk["metadata"]["chapter_end"]),
        "verse_end": int(chunk["metadata"]["verse_end"]),
        "testament": str(chunk["metadata"]["testament"]),
        "section": str(chunk["metadata"]["section"] or "")
    }
    metadatas.append(clean_metadata)

# Embed and insert chunks into ChromaDB with progress bar
embeddings = []
for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc=f"Embedding chunks in {EMBED_BATCH_SIZE} batches", unit="batch"):
    batch_texts = texts[i:i + EMBED_BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
    embeddings.extend(batch_embeddings)
print("All chunks embedded.")

# Insert chunks into ChromaDB in batches
num_insert_batches = math.ceil(len(ids) / INSERT_BATCH_SIZE)
print(f"Inserting {len(ids)} chunks into ChromaDB in {num_insert_batches} batches...")
for i in tqdm(range(num_insert_batches), desc="Inserting chunks", unit="batch"):
    start_idx = i * INSERT_BATCH_SIZE
    end_idx = min((i + 1) * INSERT_BATCH_SIZE, len(ids))
    collection.add(
        ids=ids[start_idx:end_idx],
        documents=texts[start_idx:end_idx],
        metadatas=metadatas[start_idx:end_idx],
        embeddings=embeddings[start_idx:end_idx]
    )
print(f"Inserted {len(chunks)} chunks into the ChromaDB collection '{collection_name}' at {DB_DIR}.")

# Sanity check: embed 1 chunk
# test_chunk = chunks[0]["text"]
# print(f"Embedding test chunk: {test_chunk[:60]}...")
# embedding = model.encode(test_chunk)
# print(f"Test chunk embedding (first 5 values): {embedding[:5]}")
# print(f"Embedding dimension: {len(embedding)}")