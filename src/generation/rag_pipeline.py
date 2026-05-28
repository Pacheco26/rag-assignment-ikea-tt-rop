"""Core RAG pipeline: retrieve -> augment -> generate.

Handles the full flow from query to answer with citations.
Uses Groq API (free) with Llama models.
"""

import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError

from config.settings import (
    GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL, MAX_TOKENS, TEMPERATURE, TOP_K,
)
from src.indexing.hybrid_retriever import HybridRetriever
from src.generation.prompt_builder import build_rag_prompt, build_context_string
from src.generation.anti_hallucination import AntiHallucinationGuard

logger = logging.getLogger(__name__)


def get_llm_client():
    """Get OpenAI-compatible client for Groq."""
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
    )


class RAGPipeline:
    """End-to-end RAG pipeline for FurnishRAG."""

    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = LLM_MODEL,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        enable_anti_hallucination: bool = True,
    ):
        self.retriever = retriever
        self.client = get_llm_client()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.guard = AntiHallucinationGuard() if enable_anti_hallucination else None

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        run_llm_verify: bool = False,
        return_context: bool = False,
    ) -> dict:
        """Run the pipeline: retrieve chunks, build prompt, generate answer."""
        start_time = time.time()

        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - start_time

        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the corpus for this query.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "strategy": self.retriever.strategy,
                    "retrieval_time": retrieval_time,
                    "num_chunks": 0,
                },
            }

        context_string = build_context_string(retrieved_chunks)
        messages = build_rag_prompt(question, retrieved_chunks)

        # generate with retry for rate limits
        gen_start = time.time()
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                break
            except RateLimitError as e:
                wait = min(2 ** attempt * 5, 60)
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                if attempt == 4:
                    raise
        generation_time = time.time() - gen_start
        answer = response.choices[0].message.content

        guard_result = None
        final_answer = answer
        confidence = 0.5

        if self.guard:
            guard_result = self.guard.guard(
                answer=answer,
                context=context_string,
                query=question,
                retrieved_chunks=retrieved_chunks,
                run_llm_verify=run_llm_verify,
            )
            final_answer = guard_result["final_answer"]
            confidence = guard_result["confidence"]

        sources = []
        seen_papers = set()
        for chunk in retrieved_chunks:
            paper_id = chunk.get("metadata", {}).get("paper_id", "Unknown")
            if paper_id not in seen_papers:
                seen_papers.add(paper_id)
                sources.append({
                    "paper_id": paper_id,
                    "section": chunk.get("metadata", {}).get("section", ""),
                    "rrf_score": chunk.get("rrf_score", 0),
                })

        total_time = time.time() - start_time

        result = {
            "answer": final_answer,
            "sources": sources,
            "confidence": confidence,
            "metadata": {
                "strategy": self.retriever.strategy,
                "model": self.model,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "num_chunks_retrieved": len(retrieved_chunks),
                "num_papers": len(sources),
                "top_k": top_k,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            },
        }

        if guard_result:
            result["verification"] = {
                "confidence": guard_result["confidence"],
                "citation_check": guard_result["citation_check"],
                "should_refuse": guard_result["should_refuse"],
            }
            if run_llm_verify and "llm_verification" in guard_result:
                result["verification"]["llm_check"] = guard_result["llm_verification"]

        if return_context:
            result["context"] = context_string
            result["retrieved_chunks"] = retrieved_chunks

        logger.info(
            f"RAG query completed in {total_time:.2f}s "
            f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s) "
            f"[{self.retriever.strategy}]"
        )
        return result

    def query_all_strategies(
        self,
        question: str,
        retrievers: dict[str, HybridRetriever],
        top_k: int = TOP_K,
    ) -> dict[str, dict]:
        """Run the same query on all strategies for comparison."""
        results = {}
        for strategy_name, retriever in retrievers.items():
            self.retriever = retriever
            results[strategy_name] = self.query(
                question, top_k=top_k, return_context=True,
            )
        return results
