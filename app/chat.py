"""
chat.py

Command-line interface for Bible Q&A chatbot.

Uses retrieve_and_answer() from retrieval pipeline to fetch Scripture passages
and optionally generate LLM-grounded answers.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.retrieve_and_answer import retrieve_and_answer

def main():
    print("\nWelcome to the Bible Q&A Chatbot!")
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            question = input("> ")
            if question.lower().strip() in {"quit", "exit", "x"}:
                print("\nGoodbye! Stay blessed.\n")
                break
            if not question.strip():
                continue

            answer = retrieve_and_answer(question, verbose=True, use_llm=True, model="meta-llama/Meta-Llama-3-8B-Instruct")

            print("\n=== Answer ===\n")
            print(answer)
            print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\nPlease try again.\n")

if __name__ == "__main__":
    main()