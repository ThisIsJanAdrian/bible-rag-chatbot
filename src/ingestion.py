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

# Define the canonical order of books in the KJV Bible
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

# Map books to OT or NT
TESTAMENT = {book: "OT" for book in BIBLE_ORDER[:39]}
TESTAMENT.update({book: "NT" for book in BIBLE_ORDER[39:]})

# Map books to highlight sections of books
SECTION = {book: "Gospels" for book in ["Matthew", "Mark", "Luke", "John"]}
SECTION.update({book: "Pauline Epistles" for book in [
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon"]})
SECTION.update({book: "General Epistles" for book in [
    "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude"]})
SECTION.update({book: "Torah" for book in [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]})
SECTION.update({book: "Historical Books" for book in [
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther"]})
SECTION.update({book: "Wisdom Literature" for book in [
    "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon"]}) 
SECTION.update({book: "Prophets" for book in [
    "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel",
    "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
    "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
    "Malachi"]})

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

        # Sort JSON files numerically by chapter number
        chapter_files = sorted(
            book_dir.iterdir(),
            key=lambda f: int(f.stem)  # f.stem is the filename without extension
        )

        for chapter_file in chapter_files:
            with open(chapter_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for verse in data.get("verses", []):
                    verses.append(
                        {
                        "book": data.get("book_name"),
                        "chapter": verse.get("chapter"),
                        "verse": verse.get("verse"),
                        "text": verse.get("text"),
                        "testament": TESTAMENT.get(data.get("book_name")),
                        "section": SECTION.get(data.get("book_name"), None),
                        }
                    )
    return verses

if __name__ == "__main__":
    verses = load_kjv()
    print(f"Loaded {len(verses)} verses from the KJV Bible dataset.")

    # --- Sanity check: print verses from a specific book ---
    book_to_check = "Revelation"
    book_verses = [v for v in verses if v["book"] == book_to_check]
    print(f"\nVerses from {book_to_check}:")
    for v in book_verses:  # print first 5 for sanity
        print(f'{v["book"]} {v["chapter"]}:{v["verse"]} - {v["text"]} ({v["testament"]}, {v["section"]})')