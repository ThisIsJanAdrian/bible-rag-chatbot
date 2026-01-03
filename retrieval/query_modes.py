"""
query_modes.py

Heuristics for detecting the rhetorical and theological mode of a user
query (e.g., law, discourse, prophetic, wisdom, lookup).

Used to provide soft signals for retrieval and re-ranking without
hard classification or filtering.
"""

import re
from typing import Dict, TypedDict

# Heurestics for identifying query genres
LAW_PATTERNS = [
    # Core commands / prohibitions
    r"\b(?:thou|ye|you|we|they|one)\s+(?:shalt|shall|must|ought)\b",
    r"\b(?:thou|ye|you|we|they|one)\s+(?:shalt|shall)\s+not\b",
    r"\b(?:must|must\s+not)\b",
    r"\bdo\s+not\b",
    r"\bshall\s+not\b",

    # Command language
    r"\bcommand(?:ed|s|ment)?\b",
    r"\bforbid(?:s|den)?\b",
    r"\bprohibit(?:ed|s|ion)?\b",
    r"\bunlawful\b|\bforbidden\b",

    # Obedience / law-keeping
    r"\bkeep(?:ing)?\s+(?:the|my|his)\s+(?:law|command(?:ments)?)\b",
    r"\bobey(?:ed|s|ing)?\b",
    r"\bobserv(?:e|ed|es|ing)\b",

    # Written law
    r"\bit\s+is\s+(?:written|commanded|required)\b",
]

DISCOURSE_PATTERNS = [
    # Explanatory questions
    r"\bwhat\s+(?:does|did|do)\b",
    r"\bwhat\s+is\b",
    r"\bwhat\s+does\s+.*\s+mean\b",
    r"\bwhat\s+did\s+.*\s+mean\b",
    r"\bwhy\s+(?:did|does|do)\b",
    r"\bhow\s+(?:does|did|do)\b",

    # Teaching language
    r"\bexplain(?:s|ed|ing)?\b",
    r"\bteach(?:es|ing|ings)?\b",
    r"\bmean(?:s|ing)?\b",

    # Jesus / apostolic discourse
    r"\b(?:jesus|christ|paul|peter|john)\s+(?:said|says|teaches|taught|writes?)\b",
    r"\bhe\s+said\b|\bhe\s+taught\b",

    # Sermon language
    r"\bblessed\s+are\b",
    r"\bverily\b|\bverily,\s+verily\b",
    r"\bwoe unto\b",
    r"\bbuild .* on the sand\b",
    r"\bjesus said\b",
]

PROPHETIC_PATTERNS = [
    # Divine speech formula
    r"\bthus\s+saith\s+the\s+lord\b",
    r"\bsaith\s+the\s+lord\b",

    # Judgment / future declaration
    r"\bthe\s+days\s+(?:are|will\s+be|are\s+coming)\b",
    r"\bshall\s+come\s+to\s+pass\b",
    r"\bi\s+will\b",
    r"\bi\s+will\s+bring\b",
    r"\bi\s+will\s+send\b",

    # Warning language
    r"\bwoe\s+unto\b",
    r"\brepent\b|\bturn\s+ye\b",

    # Prophetic framing
    r"\bvision\b|\bprophecy\b|\bprophet\b",
]

LOOKUP_PATTERNS = [
    # Location queries
    r"\bwhere\s+(?:do|does|did|is|are)\b",
    r"\bwhere\s+in\s+the\s+bible\b",
    r"\bwhich\s+(?:book|chapter|verse)\b",
    r"\bwhat\s+(?:book|chapter|verse)\b",

    # Appearance / occurrence
    r"\bappear(?:s|ed)?\s+in\b",
    r"\bfind\b|\blocated\b",

    # Explicit references
    r"\bchapter\s+\d+\b",
    r"\bverse\s+\d+\b",
]

WISDOM_PATTERNS = [
    # Wisdom framing
    r"\bwhat\s+does\s+the\s+bible\s+say\s+about\b",
    r"\bwhat\s+is\s+wisdom\b",
    r"\bhow\s+should\s+(?:one|we|a\s+person)\b",

    # Life / moral reflection
    r"\bmeaning\s+of\s+life\b",
    r"\bhow\s+to\s+live\b",
    r"\bwhat\s+is\s+the\s+(?:right|good|proper|correct)\b",

    # Poetic language
    r"\bblessed\s+is\b",
    r"\bhappy\s+is\b",
    r"\bfear\s+of\s+the\s+lord\b",
]

# TypedDict for query modes
class QueryModes(TypedDict):
    law: float
    discourse: float
    prophetic: float
    narrative: float
    wisdom: float
    lookup: float
    open: float

def _score_patterns(query: str, patterns: list[str]) -> float:
    """
    Helper function to score query against a list of regex patterns.

    Parameters:
        query (str): User query.
        patterns (list[str]): List of regex patterns.
    
    Returns:
        float: Ratio of matched patterns (0.0 to 1.0).
    """
    matches = sum(1 for p in patterns if re.search(p, query))
    return min(1.0, matches / len(patterns))

def detect_query_modes(query: str, verbose: bool = False) -> Dict[str, float]:
    """
    Detect rhetorical/theological modes present in the user query.

    Parameters:
        query (str): User query.
        verbose (bool): If True, print debug info.
    
    Returns:
        Dict[str, float]: Mapping of mode -> confidence score (0.0 to 1.0).
    """
    q = query.lower()

    modes = {
        "law": _score_patterns(q, LAW_PATTERNS),
        "discourse": _score_patterns(q, DISCOURSE_PATTERNS),
        "prophetic": _score_patterns(q, PROPHETIC_PATTERNS),
        "lookup": _score_patterns(q, LOOKUP_PATTERNS),
        "wisdom": _score_patterns(q, WISDOM_PATTERNS),
    }
    query_mode = max(modes, key=modes.get)

    # Narrative is weakly inferred
    modes["narrative"] = 0.3 if "what happened" in q else 0.0

    # Open (no specific mood) score is taken from inverse confidence
    max_mode = max(modes.values())
    modes["open"] = max(0.0, 1.0 - max_mode)

    if verbose:
        print(f"Query mode: {query_mode} ({modes[query_mode]:.4f}), open: {modes['open']:.4f}")

    return modes