"""
app.py

Streamlit entrypoint for Bible Q&A chatbot.

Uses retrieve_and_answer() from retrieval pipeline to fetch Scripture passages
and optionally generate LLM-grounded answers.a
"""

import streamlit as st
import sys, time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retrieve_and_answer import retrieve_and_answer

def render_answer(answer_text: str):
    if "Scripture:" in answer_text:
        parts = answer_text.split("Summary:")
        scripture = parts[0].replace("Scripture:", "").strip()

        st.subheader("📜 Scripture")
        st.code(scripture, language=None)

        st.subheader("📝 Summary")
        st.write(parts[1].strip())
    else:
        st.write(answer_text)

st.title("📖 BibleBro")

query = st.text_input("Ask a Bible question")

if st.button("Ask"):
    with st.spinner("Searching the Scriptures..."):
        answer = retrieve_and_answer(query, verbose=True, use_llm=True)

    render_answer(answer)