"""
retrieve.py

Core retrieval functions for Bible RAG chatbot. Provides 
reusable functions to query the ChromaDB collection and 
returns semantically relevant Bible text chunks.
"""

import chromadb, time
from typing import List, Dict
from retrieval.preprocessing_query import extract_book_chapter, rewrite_query, normalize_query

def get_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    """
    Initialize and return a ChromaDB collection.

    Parameters:
        db_path (str): Path to the ChromaDB persistent directory.
        collection_name (str): Name of the collection to retrieve.

    Returns:
        chromadb.Collection: The requested ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(name=collection_name)

def retrieve_chunks(collection: chromadb.Collection, query: str, top_k: int, verbose: bool = False) -> List[Dict]:
    """
    Retrieve top-k relevant Bible chunks for a query, optionally filtering by book and chapter.

    Parameters:
        collection (chromadb.Collection): The ChromaDB collection to query.
        query (str): User query.
        top_k (int): Number of top results to return.
        verbose (bool): If True, print debug info.

    Returns:
        List[Dict]: Each dict contains:
            {
                "id": str,            # chunk UUID
                "text": str,          # chunk text
                "metadata": dict      # chunk metadata (book, chapter_start, verse_start, chapter_end, verse_end, testament, section)
                "score": float        # embedding similarity score
            }
    """

    start = time.perf_counter()

    if verbose:
        print("\nPreprocessing query...")

    # Extract book and chapter from query if present
    book, chapter, verse = extract_book_chapter(query)

    if verbose and book:
        print(f"Extracted book: {book}, Chapter: {chapter}, Verse range: {verse}\n")
    
    # Use SLM-rewritten query for retrieval
    query = " ".join([query, rewrite_query(query)])

    if verbose:
        print(f"SLM-rewritten query: {query}")
    
    # Apply query normalization
    query = normalize_query(query)

    if verbose:
        print(f"spaCy-normalized query: {query}")

    elapsed = time.perf_counter() - start

    if verbose:
        print(f"Preprocessing time: {elapsed:.3f}s\n")
        print("\nRetrieving chunks from ChromaDB...")
        
    start = time.perf_counter()

    # Apply metadata filter when specific book detected
    where = {'book': book} if book else None

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for chunk_id, doc, meta, score in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # Apply post-retrieval filter when specific chapter detected
        if chapter is not None:
            if not (meta["chapter_start"] <= int(chapter) and int(chapter) <= meta["chapter_end"]):
                continue  # skip chunks outside the requested chapter
        retrieved.append({
            "id": chunk_id,
            "text": doc,
            "metadata": meta,
            "score": score
        })

    elapsed = time.perf_counter() - start

    if verbose:
        print(f"Retrieval time: {elapsed:.3f}s\n")
        print(f"Retrieved {len(retrieved)} chunks from database.")

    return retrieved