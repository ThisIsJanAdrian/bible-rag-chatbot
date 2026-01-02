"""
preprocessing_retrieval.py

Utilities for normalizing and preprocessing user queries and Bible text
prior to semantic retrieval.

Includes:

- Tokenization and lemmatization of user queries
- Stopword and punctuation removal
- Phrase extraction for overlap-based re-ranking
- Book and chapter parsing from user queries (extract_book_chapter)

These preprocessing steps are applied before embedding-based retrieval
or phrase comparison, helping improve relevance and grounding
in Scripture passages.
"""

import re, spacy
from typing import Optional

# Full list of KJV books (including numbered ones)
BIBLE_BOOKS = [
    "Genesis","Exodus","Leviticus","Numbers","Deuteronomy",
    "Joshua","Judges","Ruth","1 Samuel","2 Samuel","1 Kings","2 Kings",
    "1 Chronicles","2 Chronicles","Ezra","Nehemiah","Esther","Job",
    "Psalms","Proverbs","Ecclesiastes","Song of Solomon","Isaiah",
    "Jeremiah","Lamentations","Ezekiel","Daniel","Hosea","Joel","Amos",
    "Obadiah","Jonah","Micah","Nahum","Habakkuk","Zephaniah","Haggai",
    "Zechariah","Malachi","Matthew","Mark","Luke","John","Acts","Romans",
    "1 Corinthians","2 Corinthians","Galatians","Ephesians","Philippians",
    "Colossians","1 Thessalonians","2 Thessalonians","1 Timothy","2 Timothy",
    "Titus","Philemon","Hebrews","James","1 Peter","2 Peter","1 John",
    "2 John","3 John","Jude","Revelation",
    "Psalm", "Proverb", "Lamentation", "Revelations" # some variations
]

# Build a regex pattern to detect book names (case-insensitive)
BOOK_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in BIBLE_BOOKS) + r")\b", re.IGNORECASE
)

# Optional chapter/verse pattern, e.g., "13:1-8" or just "13"
CHAPTER_VERSE_PATTERN = re.compile(r"(\d{1,3})(?::(\d{1,3}(?:-\d{1,3})?))?")

def extract_book_chapter(query: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Extract Bible book, optional chapter, and optional verse range from query.

    Returns:
        book (str | None): Name of the book if found
        chapter (int | None): Chapter number if present
        verse_range (str | None): Verse range if present, e.g., "1-8"
    """
    book_match = BOOK_PATTERN.search(query)
    if not book_match:
        return None, None, None

    book = book_match.group(0)

    # Look for numbers immediately after book name
    after_book = query[book_match.end():].strip()
    chapter = None
    verse_range = None

    chap_match = CHAPTER_VERSE_PATTERN.match(after_book)
    if chap_match:
        chapter = int(chap_match.group(1))
        if chap_match.group(2):
            verse_range = chap_match.group(2)

    return book, chapter, verse_range

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