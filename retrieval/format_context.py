"""
format_context.py

Formats retrieved Bible chunks into a human-readable
context string suitable for LLM grounding.

This module performs formatting ONLY.
No retrieval, no LLM calls, no interpretation.
"""

from typing import List, Dict

def format_context(retrieved_chunks: List[Dict], verse_indices: Dict[str, List[int]]):
    """
    Format retrieved Bible chunks into a readable context
    with clear passage boundaries and verse awareness.

    Parameters:
        retrieved_chunks (list[dict]): Output from retrieve_chunks().
        verse_indices (dict): Mapping of chunk_id -> verse index list.

    Returns:
        str: Formatted context string for LLM input.
    """
    passages = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        chunk_id = chunk["id"]
        text = chunk["text"]
        meta = chunk["metadata"]

        if meta["chapter_start"] == meta["chapter_end"]:
            reference = (
                f"{meta['book']} "
                f"{meta['chapter_start']}:{meta['verse_start']}-"
                f"{meta['verse_end']}"
            )
        else:
            reference = (
                f"{meta['book']} "
                f"{meta['chapter_start']}:{meta['verse_start']}-"
                f"{meta['chapter_end']}:{meta['verse_end']}"
            )

        verse_list = verse_indices.get(chunk_id)
        if not verse_list:
            # Fallback: include full chunk text if verse indices are missing
            formatted_text = text.strip()
        else:
            verses = []
            for v in verse_list:
                verse_text = text[v["start"]:v["end"]].strip()
                verses.append(f"{meta['book']} {v['chapter']}:{v['verse']} — \"{verse_text}\"")
            formatted_text = "\n".join(verses)

        passage_block = f"[Passage {idx}]\nChunk reference: {reference}\n{formatted_text}"
        passages.append(passage_block)

    return "\n\n".join(passages)