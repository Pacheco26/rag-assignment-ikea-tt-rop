"""Prompt templates for FurnishRAG."""

# -- RAG System Prompt ---------------------------------------------------------
SYSTEM_PROMPT = """You are FurnishRAG, a knowledgeable furniture assistant specializing in \
IKEA products, buying guides, and home furnishing solutions. You answer questions based \
ONLY on the provided context from IKEA buying guides and product documentation.

Rules:
1. Base every claim on the retrieved context. Cite sources as [DXX] (e.g., [D01], [D04]).
2. If the context does not contain enough information, say so explicitly.
3. Never fabricate product specifications, dimensions, or features not present in the context.
4. When comparing products, present specifications from each guide separately before recommending.
5. Include specific measurements, materials, and capacities when available in the context.
6. Do not state prices as they vary by region and change over time.
"""

# -- RAG Query Prompt ----------------------------------------------------------
QUERY_PROMPT = """Context from IKEA buying guides:
---
{context}
---

Question: {query}

Provide a helpful answer based on the context above. Cite specific buying guides using \
[DXX] notation. Include specific dimensions, materials, and product details when available."""

# -- Anti-Hallucination Verification Prompt ------------------------------------
VERIFICATION_PROMPT = """You are a verification system for a furniture assistant. Given the \
original context from IKEA buying guides and a generated answer, evaluate:

1. **Faithfulness** (0-1): Does every claim in the answer have support in the context?
2. **Groundedness** (0-1): Are all cited dimensions and specifications actually present in the context?
3. **Completeness** (0-1): Does the answer address the question using available context?

Context:
---
{context}
---

Answer to verify:
---
{answer}
---

Original question: {query}

Respond in JSON format:
{{
    "faithfulness": <float>,
    "groundedness": <float>,
    "completeness": <float>,
    "unsupported_claims": [<list of claims not found in context>],
    "verdict": "PASS" or "FAIL"
}}"""

# -- Summarization Prompts -----------------------------------------------------
MAP_SUMMARY_PROMPT = """Summarize the following section from an IKEA buying guide. \
Focus on product features, specifications, available options, materials, and any \
planning or safety information.

Text:
{text}

Summary:"""

REDUCE_SUMMARY_PROMPT = """Combine the following section summaries into a coherent \
overall summary of this IKEA buying guide. Highlight the product range, key features, \
available configurations, materials, and important planning or safety notes.

Section summaries:
{summaries}

Combined summary:"""

# -- LLM-as-Judge Evaluation Prompt --------------------------------------------
JUDGE_CORRECTNESS_PROMPT = """You are an evaluation judge for a furniture assistant. \
Compare the generated answer against the gold-standard reference answer.

Question: {query}

Gold-standard answer: {gold_answer}

Generated answer: {generated_answer}

Rate the generated answer on:
1. **Correctness** (0-1): Does it contain the same key product facts as the gold standard?
2. **Faithfulness** (0-1): Does it avoid stating things not supported by the buying guides?
3. **Completeness** (0-1): Does it cover all important points from the gold standard?

Respond in JSON format:
{{
    "correctness": <float>,
    "faithfulness": <float>,
    "completeness": <float>,
    "explanation": "<brief explanation>"
}}"""
