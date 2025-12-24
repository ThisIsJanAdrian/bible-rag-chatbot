"""
retrieval_preprocessing.py

Utilities for preprocessing user queries and re-ranking
retrieved Bible chunks based on query relevance.
"""

import spacy, re
from typing import List, Dict

# Lazy-load spaCy model
_nlp = None

def get_spacy_nlp():
    """
    Lazy-load and return the spaCy NLP model.
    
    Returns:
        spacy.language.Language: The loaded spaCy NLP model.
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp

# Query preprocessing utilities

def preprocess_query(query: str) -> str:
    """
    Normalize a user query for semantic retrieval.

    - Lowercases
    - Lemmatizes
    - Removes stopwords and punctuation
    - Keeps only alphabetic tokens

    Example:
        "Who is Mary, the mother of Jesus?"
        -> "mary mother jesus"
    """
    nlp = get_spacy_nlp()
    doc = nlp(query.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

# Re-ranking utilities

def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.

    Parameters:
        text (str): Input text.

    Returns:
        List[str]: List of tokens.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def extract_phrases(text: str, min_words: int = 3, max_words: int = 5) -> List[str]:
    """
    Extract n-grams (phrases) from text.

    Parameters:
        text (str): Input text.
        min_words (int): Minimum words in phrase.
        max_words (int): Maximum words in phrase.

    Returns:
        List[str]: List of extracted phrases.
    """
    tokens = simple_tokenize(text)
    phrases = []
    for n in range(min_words, max_words + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            phrases.append(phrase)
    return phrases

def compute_phrase_overlap(query: str, chunk_text: str) -> float:
    """
    Compute phrase overlap ratio between query and chunk text.

    Parameters:
        query (str): User query.
        chunk_text (str): Retrieved chunk text.

    Returns:
        float: Overlap ratio (0.0 to 1.0).
    """
    query_phrases = set(extract_phrases(query.lower()))
    chunk_phrases = set(extract_phrases(chunk_text.lower()))
    if not query_phrases:
        return 0.0
    overlap = query_phrases.intersection(chunk_phrases)
    return len(overlap) / len(query_phrases)

def rerank_chunks(chunks: List[Dict], query: str, alpha: float = 0.5, min_score: float = 0.1, verbose: bool = False) -> List[Dict]:
    """
    Re-rank ChromaDB retrieved chunks using embedding similarity
    and multi-word phrase overlap with the user query.

    Parameters:
        chunks (list[dict]): Each dict contains 'text' and 'score' (embedding similarity)
        query (str): Raw user question
        alpha (float): Weight for embedding similarity vs phrase overlap (0-1)
        min_score (float): Minimum combined score to include chunk
        verbose (bool): If True, print debug info

    Returns:
        list[dict]: Re-ranked chunks (descending order)
    """
    filtered_chunks = []

    for chunk in chunks:
        phrase_score = compute_phrase_overlap(query, chunk["text"])
        embedding_score = chunk.get("score", 0.0)
        chunk["re_rank_score"] = alpha * embedding_score + (1 - alpha) * phrase_score

        if chunk["re_rank_score"] >= min_score:
            filtered_chunks.append(chunk)
            if verbose:
                print(f"Chunk text: {chunk['text'][:50]}...")
                print(f"Re-rank score: {chunk['re_rank_score']:.4f} (Embed: {embedding_score:.4f}, Phrase: {phrase_score:.4f})")

    return sorted(filtered_chunks, key=lambda x: x["re_rank_score"], reverse=True)