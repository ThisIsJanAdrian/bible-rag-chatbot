"""
retrieve_and_answer.py

End-to-end retrieval + answering pipeline for the Bible RAG chatbot.
Retrieves relevant Scripture, formats context, and queries an LLM
under strict grounding rules.

This script assumes:
- ChromaDB is already populated
- Retrieval and formatting modules are available
"""

import sys, json
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

# Robust & free LLM model for Bible Q&A
MODEL_NAME = "allenai/Olmo-3.1-32B-Instruct"

# Fallback LLM for faster response (less robust)
# MODEL_NAME = "swiss-ai/Apertus-8B-Instruct-2509"

TOP_K = 25
MIN_SCORE = 0.4
MAX_TOKENS = 1024
TEMPERATURE = 0.0

# Check Hugging Face model availability
check_model_inference_status(MODEL_NAME)

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
    retrieved = retrieve_chunks(collection, query, top_k=top_k)

    if not retrieved:
        return ""

    reranked = rerank_chunks(retrieved, query, min_score=MIN_SCORE, verbose=verbose)

    return format_context(reranked[:10], verse_indices)

def retrieve_and_answer(query: str, top_k: int = TOP_K, use_llm: bool = False, verbose: bool = False) -> str:
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
        Question:
        {query}

        Below are the Scripture passages retrieved as potentially relevant.
        Each passage is numbered and fully self-contained.
        You MUST only quote or reference the passages below. Do NOT include any verse not listed.

        Scripture passages:
        {context}

        Instructions:
        Answer using the following structure ONLY:

        1. Quoted Scripture (with reference)
        2. Brief explanation of that quoted Scripture

        Repeat as needed.
        Answer the question using only the passages above.
        Quote passages by their chapters and verses. You can choose to cite some or all of the verses depending on their relevance.
        Clarify archaic words or expressions in parentheses if needed, but do not change the meaning.
        Provide concise, text-faithful explanations strictly grounded in the quoted text.
        If none of the passages clearly answer the question, state this explicitly.
        Do not invent information or recall verses outside the listed passages.
        """.strip()

    if verbose:
        print("=== PROMPT SENT TO LLM ===")
        print(user_prompt)

    return query_hf(MODEL_NAME, user_prompt, MAX_TOKENS, TEMPERATURE)