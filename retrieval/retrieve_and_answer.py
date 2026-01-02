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

from retrieve import retrieve_chunks, get_collection
from retrieval.retrieval_preprocessing import preprocess_query
from retrieval.reranking import rerank_chunks
from retrieval.format_context import format_context
from utils.hf_utils import check_model_inference_status, query_hf

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"
VERSE_INDICES_FILE = BASE_DIR / "data" / "kjv_verse_indices.json"

# Configuration
CHROMA_COLLECTION_NAME = "bible_kjv_chunks"
MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
TOP_K = 10
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
        query (str): User question
        top_k (int): Number of chunks to retrieve
        verbose (bool): If True, print detailed information

    Returns:
        str: Formatted Scripture context
    """
    retrieved = retrieve_chunks(collection, preprocess_query(query), top_k=top_k)

    if not retrieved:
        return ""

    reranked = rerank_chunks(retrieved, query, min_score=0.4, verbose=verbose)

    return format_context(reranked, verse_indices)

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

    user_prompt = (
        f"Question: {query}\n\n"
        f"Using ONLY the following Scripture passages, answer the question:\n\n"
        f"{context}\n\n"
    )

    return query_hf(MODEL_NAME, user_prompt, MAX_TOKENS, TEMPERATURE)

if __name__ == "__main__":
    question = input("\nAsk a Bible question:\n> ")
    answer = retrieve_and_answer(question)
    print("\n=== Answer ===\n")
    print(answer)