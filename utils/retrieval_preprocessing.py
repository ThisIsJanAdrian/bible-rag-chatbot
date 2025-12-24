"""
retrieval_preprocessing.py

Utilities for preprocessing user queries and re-ranking
retrieved Bible chunks based on query relevance.
"""

import spacy

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