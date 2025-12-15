"""
ingestion.py

This module loads the King James Version (KJV) Bible 
dataset from JSON files and converts it into a list of 
structured verse objects. This module is strictly 
responsible only for data ingestion and structuring.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "kjv"

BIBLE_ORDER = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms",
    "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
    "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation"
    ]

def load_kjv() -> list[dict]:
    """
    Load the KJV Bible from JSON files into structured verse objects.
    
    Returns:
        list[dict]: A list of verses, where each verse contains:
            - book (str)
            - chapter (int)
            - verse (int)
            - text (str)
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"KJV data directory not found: {DATA_DIR}")

    verses = []
    for book_name in BIBLE_ORDER:
        book_dir = DATA_DIR / book_name
        if not book_dir.exists() or not book_dir.is_dir():
            continue
        for json_file in book_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for verse in data.get("verses", []):
                    verses.append(
                        {
                        "book": data.get("book_name"),
                        "chapter": verse.get("chapter"),
                        "verse": verse.get("verse"),
                        "text": verse.get("text"),
                        }
                    )
    return verses

if __name__ == "__main__":
    verses = load_kjv()
    print(f"Loaded {len(verses)} verses from the KJV Bible dataset.")
    print(f"Sample: {verses[:5]}")
