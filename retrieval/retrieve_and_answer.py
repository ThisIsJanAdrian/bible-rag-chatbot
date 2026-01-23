"""
retrieve_and_answer.py

End-to-end retrieval + answering pipeline for the Bible RAG chatbot.
Retrieves relevant Scripture, formats context, and queries an LLM
under strict grounding rules.

This script assumes:
- ChromaDB is already populated
- Retrieval and formatting modules are available
"""

import sys, json, time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retrieve import retrieve_chunks, get_collection
from retrieval.reranking import rerank_chunks
from retrieval.format_context import format_context
from utils.hf_utils import check_model_inference_status, query_hf

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
VERSE_INDICES_FILE = BASE_DIR / "data" / "kjv_verse_indices.json"

# Database configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"

# Default LLM model for Bible Q&A
MODEL_NAME = "allenai/Olmo-3.1-32B-Instruct"

# Retrieval and LLM parameters
TOP_K = 25
MIN_SCORE = 0.4
MAX_TOKENS = 1024
TEMPERATURE = 0.0

# Initialize external resources
collection = get_collection(str(DB_DIR), CHROMA_COLLECTION_NAME)
with open(VERSE_INDICES_FILE, "r", encoding="utf-8") as f:
    verse_indices = json.load(f)

def retrieve_context(query: str, top_k: int = TOP_K, verbose: bool = False) -> str:
    """
    Retrieve and format Scripture passages relevant to a query.

    Parameters:
        query (str): User query.
        top_k (int): Number of chunks to retrieve.
        verbose (bool): If True, print detailed information.

    Returns:
        str: Formatted Scripture context
    """
    
    retrieved = retrieve_chunks(collection, query, top_k=top_k, verbose=verbose)

    if not retrieved:
        return ""

    start = time.perf_counter()
    reranked = rerank_chunks(retrieved, query, min_score=MIN_SCORE, verbose=verbose)
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"Reranking time: {elapsed:.3f}s\n")

    start = time.perf_counter()
    formatted = format_context(reranked, verse_indices, verbose=verbose)
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"Formatting time: {elapsed:.3f}s\n")

    return formatted

def retrieve_and_answer(query: str, top_k: int = TOP_K, use_llm: bool = False, verbose: bool = False, model: str = MODEL_NAME) -> str:
    """
    Retrieve Scripture passages and optionally generate a grounded answer.

    Parameters:
        query (str): User question
        top_k (int): Number of chunks to retrieve
        use_llm (bool): If True, generate LLM answer; else return Scripture context
        verbose (bool): If True, print detailed information

    Returns:
        str: Scripture context or LLM-generated answer
    """

    context = retrieve_context(query, top_k=top_k, verbose=verbose)

    if not context:
        return "No relevant Scripture passages found for this question."

    if not use_llm:
        return context

    user_prompt = f"""
        RULES:
        - Quote Scripture FIRST, exactly as provided.
        - Quote ONLY passages that directly answer the question.
        - Use the FEWEST passages possible.
        - Prefer ONE passage if it fully answers the question.
        - Use NO MORE THAN THREE passages total.
        - Do NOT include background, surrounding, or loosely related verses.
        - Do NOT explain verse-by-verse.
        - Do NOT add commentary beyond the Summary section.
        - Do NOT restate ideas not present in the quoted text.

        If none of the provided passages directly answer the question,
        do NOT quote all passages.
        Instead, quote ONLY the few most relevant passage(s),
        OR state that the passages do not directly answer the question.

        -----

        OUTPUT FORMAT (MANDATORY):

        Scripture:
        "<verbatim quotation(s) from the passages above>"

        Summary:
        <TWO or MORE sentences summarizing what the quoted Scripture shows>

        -----

        EXAMPLES OF IDEAL BEHAVIOR

        Example 1 — Identity / Relationship Question

        Query:
        Who is the real father of Jesus?

        Scripture:
        "Matthew 1:16 — And Jacob begat Joseph the husband of Mary, of whom was born Jesus, who is called Christ."

        Summary:
        The passage identifies Joseph as the husband of Mary, through whom Jesus was born. It does not explicitly state who Jesus' father is, so the passage does not directly answer the question beyond what is written.

        -----

        Example 2 — Thematic / Virtue Question

        Query:
        What does the Bible say about love in action?

        Scripture:
        "2 Timothy 4:8 — Henceforth there is laid up for me a crown of righteousness, which the Lord, the righteous judge, shall give me at that day: and not to me only, but unto all them also that love his appearing."

        Summary:
        The passage shows that love is expressed through faithful devotion and perseverance. Those who demonstrate love through their actions are promised reward by God.

        -----

        Example 3 — Historical / Opposition Question

        Query:
        What opposition did the Jews receive when rebuilding the temple?

        Scripture:
        "Ezra 4:4-5 — Then the people of the land weakened the hands of the people of Judah, and troubled them in building,
        And hired counsellors against them, to frustrate their purpose, all the days of Cyrus king of Persia, even until the reign of Darius king of Persia."

        "Ezra 4:23 — Then ceased the work of the house of God which is at Jerusalem. So it ceased unto the second year of the reign of Darius king of Persia."

        Summary:
        The passages state that the Jews faced active opposition that weakened and troubled their efforts. Counselors were hired to frustrate the rebuilding, and as a result, the work on the temple ceased for a time.

        -----

        User Query:
        {query}

        Below are the ONLY Scripture passages you may use.
        You MUST NOT quote, paraphrase, or refer to any verse not listed.

        Scripture passages:
        {context}
    """.strip()

    # Check Hugging Face model availability
    check_model_inference_status(model)
    
    start = time.perf_counter()
    answer = query_hf(model_name=model, user_prompt=user_prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, verbose=verbose)
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"Inference time: {elapsed:.3f}s\n")

    return answer