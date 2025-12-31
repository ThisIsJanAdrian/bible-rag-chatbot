"""
reranking.py

Logic for re-ranking retrieved Bible chunks based on query relevance.

Combines embedding similarity, phrase overlap, and other heuristic
signals to prioritize passages most relevant to the user's question.
"""

import math, re
from typing import List, Dict

from retrieval.retrieval_preprocessing import get_spacy_nlp
from retrieval.query_modes import detect_query_modes

# Configuration
ALPHA_BASELINE = 0.6  # baseline trust embeddings

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

def compute_alpha_from_query_modes(query_modes: Dict[str, float], verbose: bool = False) -> float:
    """
    Compute a dynamic alpha weight for re-ranking based on detected
    query modes.

    Alpha balances embedding similarity and phrase overlap:
        final_score = alpha * embedding_score + (1 - alpha) * phrase_score

    Parameters:
        query_modes (Dict[str, float]): Dictionary of soft query mode confidences (values 0.0 to 1.0), e.g., output of detect_query_modes()
        verbose (bool): If True, print debug info

    Returns:
        float: Alpha weight between 0.1 and 0.9
    """
    alpha = ALPHA_BASELINE

    # Decrease weight on embeddings if query is law/lookup focused
    alpha -= 0.3 * query_modes["law"]
    alpha -= 0.3 * query_modes["lookup"]

    # Reduce the impact of prophetic patterns for NT-like queries
    alpha -= 0.2 * query_modes["prophetic"]

    # Increase weight on embeddings for discourse, wisdom, and open queries
    alpha += 0.2 * query_modes["discourse"]
    alpha += 0.1 * query_modes["wisdom"]
    alpha += 0.1 * query_modes["open"]

    if verbose:
        print(f"Alpha: {alpha:.4f}")

    return min(0.90, max(0.1, alpha))

def rerank_chunks(chunks: List[Dict], query: str, min_score: float = 0.3, verbose: bool = False) -> List[Dict]:
    """
    Re-rank ChromaDB retrieved chunks using embedding similarity
    and multi-word phrase overlap with the user query.

    Parameters:
        chunks (List[Dict]): Each dict contains:
            - 'text' (str): Chunk text
            - 'score' (float): Embedding similarity score
        query (str): Raw user question
        min_score (float): Minimum combined score to include chunk
        verbose (bool): If True, print debug info

    Returns:
        List[Dict]: Re-ranked chunks in descending order by combined score. Each dict has an added 're_rank_score' key.
    """
    query_modes = detect_query_modes(query, verbose=verbose)
    alpha = compute_alpha_from_query_modes(query_modes, verbose=verbose)

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