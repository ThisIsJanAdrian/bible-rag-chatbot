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

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"

# Default LLM model for Bible Q&A
MODEL_NAME = "allenai/Olmo-3.1-32B-Instruct"

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
    - Do NOT explain verse-by-verse.
    - Do NOT add commentary beyond the Summary section.
    - Do NOT restate ideas not present in the quoted text.

    -----

    OUTPUT FORMAT (MANDATORY):
    Scripture:
    "<verbatim quotation(s) from the passages above>"

    Summary:
    <TWO or MORE sentences summarizing what the quoted Scripture shows>

    EXAMPLE:
    Scripture:
    "John bare record, saying, I saw the Spirit descending from heaven like a dove, and it abode upon him." (John 1:32)
    "This is he that came by water and blood, even Jesus Christ; not by water only, but by water and blood. And it is the Spirit that beareth witness, because the Spirit is truth." (1 John 5:6)
    "Who is he that overcometh the world, but he that believeth that Jesus is the Son of God?" (1 John 5:5)
    "I am the living bread which came down from heaven..." (John 6:51)
    "I have many things to say and to judge of you: but he that sent me is true; and I speak to the world those things which I have heard of him." (John 8:26)

    Summary:
    The passages affirm that Jesus is the Son of God and emphasize his divine origin and mission. However, they do not explicitly identify or describe Jesus' earthly or human father.

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