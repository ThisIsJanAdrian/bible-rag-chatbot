"""
retrieval_preprocessing.py

Utilities for preprocessing user queries and re-ranking
retrieved Bible chunks based on query relevance.
"""

import spacy, re, math
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

def extract_phrases(text: str, min_words: int = 2, max_words: int = 5) -> List[str]:
    """
    Extract n-grams (phrases) from text.

    Parameters:
        text (str): Input text.
        min_words (int): Minimum words in phrase.
        max_words (int): Maximum words in phrase.

    Returns:
        List[str]: List of extracted phrases.
    """
    nlp = get_spacy_nlp()
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha]

    phrases = []
    for n in range(min_words, max_words + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            phrases.append(phrase)
    return phrases

def compute_phrase_overlap(query: str, chunk_text: str, max_words: int = 5, k: float = 3.0) -> float:
    """
    Compute phrase overlap ratio between query and chunk text,
    and apply an exponential bump to emphasize exact matches.

    Parameters:
        query (str): User query.
        chunk_text (str): Retrieved chunk text.
        max_words (int): Maximum words in phrase, refer to extract_phrases().
        k (float): Bump factor for exponential scaling.

    Returns:
        float: Bumped overlap score (0.0 to 1.0).
    """
    query_phrases = set(extract_phrases(query.lower()))
    chunk_phrases = set(extract_phrases(chunk_text.lower()))
    if not query_phrases or not chunk_phrases:
        return 0.0
    overlap = query_phrases.intersection(chunk_phrases)
    raw_score = len(overlap) / min(len(query_phrases), len(chunk_phrases))

    max_phrase_len = max(
        (len(p.split()) for p in overlap),
        default=0
    )
    length_bonus = max_phrase_len / max_words

    combined = 0.7 * raw_score + 0.3 * length_bonus
    bumped = 1 - math.exp(-k * combined)
    return min(1.0, bumped)

def rerank_chunks(chunks: List[Dict], query: str, alpha: float = 0.8, min_score: float = 0.3, verbose: bool = False) -> List[Dict]:
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

        if verbose:
            print(f"Chunk text: {chunk['text'][:50]}...")
            print(f"Re-rank score: {chunk['re_rank_score']:.4f} (Embed: {embedding_score:.4f}, Phrase: {phrase_score:.4f})\n")
        
        if chunk["re_rank_score"] >= min_score:
            filtered_chunks.append(chunk)

    return sorted(filtered_chunks, key=lambda x: x["re_rank_score"], reverse=True)