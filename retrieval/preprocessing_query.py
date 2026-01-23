"""
preprocessing_query.py

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

import sys, re, spacy
from typing import Optional
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.hf_utils import check_model_inference_status, query_hf

# Model for query rewriting
REWRITE_SLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

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

    Parameters:
        query (str): User query potentially containing book/chapter info.

    Returns:
        book (str | None): Name of the book if found
        chapter (int | None): Chapter number if present
        verse_range (str | None): Verse range if present, e.g., "1-8"
    """
    book_match = BOOK_PATTERN.search(query)
    if not book_match:
        return None, None, None

    book = book_match.group(0)

    after_book = query[book_match.end():].strip()
    chapter = None
    verse_range = None

    chap_match = CHAPTER_VERSE_PATTERN.match(after_book)
    if chap_match:
        chapter = int(chap_match.group(1))
        if chap_match.group(2):
            verse_range = chap_match.group(2)

    # Normalization of book names
    match book:
        case "Psalm":
            book = "Psalms"
        case "Proverb":
            book = "Proverbs"
        case "Lamentation":
            book = "Lamentations"
        case "Revelations":
            book = "Revelation"
        
    return book, chapter, verse_range

def rewrite_query(query: str) -> str:
    """
    Rewrite user query into retrieval-friendly language without changing meaning.

    Parameters:
        query (str): Original user query.
    
    Returns:
        str: Rewritten query.
    """
    prompt = f"""
    You rewrite Bible search queries for retrieval from a KJV Scripture corpus.

    TASK:
    Rewrite the user query into retrieval-friendly keywords or short phrases.

    RULES:
    - Preserve the original meaning.
    - Do NOT explain, interpret, or summarize theology.
    - Do NOT recall or quote Scripture.
    - Do NOT add verse references.
    - Expand with semantically related terms when helpful.
    - Prefer KJV-style and archaic terms where appropriate
    (e.g., love → charity, sin → iniquity, forgive → remission).
    - Output should resemble a search query or keyword list, not a sentence.

    EXAMPLES:
    Query: give me Scriptures that talk about love in action  
    Rewrite: love charity kindness mercy good works faithful deeds

    Query: forgiveness of sins  
    Rewrite: forgiveness remission sins iniquity transgression mercy

    Query: mark of the beast  
    Rewrite: mark beast number hand forehead worship

    Query: What does the Bible say about love in 1 Corinthians 13?  
    Rewrite: love charity longsuffering kindness envy vaunteth not

    USER QUERY:
    {query}

    Rewrite:
    """.strip()

    # Check Hugging Face model availability
    check_model_inference_status(REWRITE_SLM_MODEL_NAME)

    rewritten = query_hf(model_name=REWRITE_SLM_MODEL_NAME, user_prompt=prompt, system_prompt="", temperature=0.0, max_tokens=256)

    return rewritten

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

def normalize_query(query: str) -> str:
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