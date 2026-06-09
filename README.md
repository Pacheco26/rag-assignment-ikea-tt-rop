# FurnishRAG -- RAG System for IKEA Buying Guides

**MINTRI Phase 2 -- MEDados, ISEP**
**Authors:** Tatiana Teixeira (1221260), Rodrigo Pacheco (1221900)

## Overview

FurnishRAG is a Retrieval-Augmented Generation system that lets users ask questions about IKEA products and buying guides. It compares 3 chunking strategies (fixed-size, semantic, hierarchical) across 30 gold-standard queries, evaluating both retrieval and answer quality.

### Use Case

Customers and interior design enthusiasts often need to cross-reference multiple IKEA buying guides to find product dimensions, materials, compatibility, or planning advice. A standalone LLM hallucinates specific product details. FurnishRAG anchors every answer in the actual text of the guides, with verifiable [DXX] citations.

### Corpus

20 IKEA buying guides covering 5 categories:
- **Storage & Organization** (BILLY, KALLAX, BESTA, EKET)
- **Bedroom & Wardrobe** (PAX, Mattresses, Beds, HEMNES Bedroom)
- **Kitchen & Dining** (SEKTION, Kitchen Planning, KUNGSFORS)
- **Living & Outdoor** (HEMNES Living Room, KIVIK, EKTORP, BONDHOLMEN)
- **Home & Sustainability** (Style Guide, Laundry, Circular Design, Bathroom)

## Quick Start

### 1. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set your Groq API key (free at console.groq.com)
```

### 2. Prepare Corpus

Place the 20 IKEA PDFs in `corpus/raw/` (see `corpus/raw/README.md` for the list).

```bash
python scripts/ingest_corpus.py
```

### 3. Build Indices

```bash
python scripts/build_index.py
```

### 4. Run Demo

```bash
python scripts/run_demo.py
# or: streamlit run app/streamlit_app.py
```

### 5. Run Evaluation

```bash
python scripts/run_evaluation.py
```

## Architecture

```
[20 PDFs] -> [Ingestion] -> [3 Chunking Strategies] -> [Hybrid Retrieval] -> [Generation] -> [Answer]
                                  |                           |
                          Fixed-Size (512 tokens)      Dense + BM25 (RRF)
                          Semantic (breakpoint)         alpha=0.7
                          Hierarchical (128/1024)
```

**Stack:**
- LLM: Groq API with `llama-3.1-8b-instant` (free tier)
- Embeddings: `all-MiniLM-L6-v2` (local, 384 dimensions)
- Vector DB: ChromaDB (persistent)
- Sparse index: BM25 via `rank_bm25`

## Chunking Strategies

| Strategy | Parameters | Rationale |
|----------|-----------|-----------|
| Fixed-Size | 512 tokens, overlap 50 | Standard baseline; sentence-aware splits |
| Semantic | 95th percentile breakpoint | Splits at topic shifts detected by embedding distance |
| Hierarchical | child=128, parent=1024 | Small children for precise matching, large parents for generation context |

## Evaluation Results

30 queries (10 factual, 10 comparative, 10 thematic) evaluated across all 3 strategies:

| Strategy | MRR | nDCG@5 | Recall@5 | Prec@5 | Correctness | Faithfulness | Completeness | Citation F1 |
|---|---|---|---|---|---|---|---|---|
| **Fixed-Size** | **0.811** | **0.663** | **0.670** | **0.463** | **0.642** | 0.773 | **0.533** | 0.506 |
| Semantic | 0.781 | 0.649 | 0.652 | 0.436 | 0.620 | **0.802** | 0.513 | **0.520** |
| Hierarchical | 0.656 | 0.561 | 0.578 | 0.437 | 0.530 | 0.773 | 0.390 | 0.371 |

The fixed-size strategy performs best overall on retrieval and correctness. Semantic chunking achieves the highest faithfulness and citation accuracy. The hierarchical approach underperforms on this corpus, likely because IKEA guides have relatively flat document structure where the parent-child separation adds noise without benefit.

## Text Mining Features

1. **Keyword Extraction** -- KeyBERT with MMR diversity
2. **Topic Modelling** -- BERTopic on chunks, with taxonomy alignment
3. **Clustering** -- KMeans/HDBSCAN with UMAP visualization
4. **Taxonomy Classification** -- Embedding-based classification into 5 IKEA product categories
5. **Summarization** -- Abstractive map-reduce via LLM

## Limitations

- **Small LLM:** `llama-3.1-8b-instant` limits answer quality; a larger model would improve correctness
- **Citation F1 peaks at 0.52:** the model sometimes cites wrong guides or misses relevant ones
- **Hierarchical underperformance:** IKEA guides lack the deep section hierarchy that benefits parent-child chunking
- **Corpus scope:** 20 guides covers the main product lines but not the full IKEA catalog

## Project Structure

```
03_Trabalho_Fase2/
  config/          -- settings, prompts
  corpus/          -- raw PDFs, processed JSON, metadata
  src/
    ingestion/     -- PDF parsing, text cleaning, section detection
    chunking/      -- 3 strategies + intrinsic metrics
    indexing/      -- embeddings, ChromaDB, BM25, hybrid retrieval
    generation/    -- RAG pipeline, prompt building, anti-hallucination
    text_mining/   -- keywords, topics, clusters, taxonomy, summarization
    evaluation/    -- test queries, retrieval metrics, LLM-as-judge
  app/             -- Streamlit demo (Query, Explorer, Evaluation pages)
  scripts/         -- CLI entrypoints (ingest, build, evaluate, demo)
  results/         -- evaluation outputs, figures
```
