# FurnishRAG

RAG system for IKEA buying guides.

**MINTRI Phase 2 -- MEDados, ISEP**

**Authors:** Tatiana Teixeira (1221260), Rodrigo Pacheco (1221900)

## About

FurnishRAG is a Retrieval-Augmented Generation system for answering questions about IKEA products using 20 buying guides as corpus. The system compares 3 chunking strategies and uses hybrid retrieval with dense embeddings and BM25.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
