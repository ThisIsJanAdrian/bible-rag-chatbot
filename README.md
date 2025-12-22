# 📖 BibleBro: a Bible RAG Chatbot (WIP)

BibleBro is a **Retrieval-Augmented Generation (RAG) chatbot** that answers questions strictly using **Bible verses only**, specifically using the **King James Version (KJV)**. This project focuses on **grounded retrieval, verse-level accuracy, and interpretability** to ensure faithful responses from retrieved passages and minimize hallucinations.

Think of BibleBro as your **Bible study aide**, intentionally unbiased and speaks nothing but the truth.

> 🚧 This repository is under active development.

---

## 🎯 Objectives

- Ground all answers strictly in Scripture  
- Preserve verse- and paragraph-level structure  
- Clearly separate **Scripture** from **explanation**  
- Enable exact verse referencing despite chunked embeddings  
- Build an explainable, debuggable RAG pipeline  

---

## 🧠 System Overview

**Pipeline:**

1. **Ingestion** – Load and normalize KJV Bible text  
2. **Chunking** – Verse-aware, overlapping chunks (min-word based)  
3. **Embeddings** – Local embeddings using `BAAI/bge-base-en-v1.5`  
4. **Vector Store** – Persistent ChromaDB storage  
5. **Retrieval** – Semantic search with verse-level reconstruction  
6. **Context Formatting** – Human-readable Scripture blocks  
7. **(Planned)** LLM Integration – Scripture-grounded answers only  

---

## 📂 Project Structure
```
data/
  ├── kjv_chunks.json
  ├── kjv_verse_indeces.json
  └── chroma_db/          # (ignored in git)

preprocessing/
  ├── ingestion.py
  └── chunking.py

retrieval/
  ├── retrieve.py
  └── format_context.py

scripts/
  ├── create_chunks.py
  ├── embed_chunks.py
  └── test_retrieval.py
```

---

## ✅ Progress

- [x] KJV ingestion and normalization  
- [x] Verse-aware chunking with overlap  
- [x] Local embeddings (CPU-based)  
- [x] Persistent vector storage (ChromaDB)  
- [x] Semantic retrieval  
- [x] Verse-level reconstruction within chunks  
- [x] Human-readable formatted context  
- [ ] LLM integration with strict grounding rules  
- [ ] `retrieve_and_answer.py` pipeline  
- [ ] System prompt for Scripture-only answers  
- [ ] Local UI (Streamlit)  
- [ ] Error handling and safeguards  
- [ ] Expanded documentation and examples  

---

## ⚠️ Notes

- Vector database files are intentionally excluded from version control  
- Embeddings are generated locally and cached  
- Project prioritizes **correctness and faithfulness over speed**

---

## 📌 Status

🛠️ **Active development** — expect iteration and refinement.
