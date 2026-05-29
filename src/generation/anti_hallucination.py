"""Checks that generated answers are grounded in context.

Verifies citations, computes confidence, and triggers refusal
when context is insufficient.
"""

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from config.settings import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL, CONFIDENCE_THRESHOLD
from config.prompts import VERIFICATION_PROMPT

logger = logging.getLogger(__name__)


class AntiHallucinationGuard:
    """Verifies generated answers against retrieved context."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        self.model = model
        self.confidence_threshold = confidence_threshold

    def verify_citations(self, answer: str, context: str) -> dict:
        """Check that cited document IDs actually exist in the context."""
        # Extract citations from answer (e.g., [D01], [D04])
        cited = set(re.findall(r'\[D\d{2}\]', answer))
        # Extract document IDs present in context
        available = set(re.findall(r'\[?D\d{2}\]?', context))
        # Normalize
        available_normalized = {f"[{p.strip('[]')}]" for p in available}

        valid_citations = cited & available_normalized
        invalid_citations = cited - available_normalized

        return {
            "cited_papers": list(cited),
            "valid_citations": list(valid_citations),
            "invalid_citations": list(invalid_citations),
            "citation_accuracy": (
                len(valid_citations) / len(cited) if cited else 1.0
            ),
            "has_citations": len(cited) > 0,
        }

    def compute_confidence(self, answer: str, retrieved_chunks: list[dict]) -> float:
        """Heuristic confidence score (0-1) based on retrieval quality and citations."""
        if not retrieved_chunks:
            return 0.0

        # Factor 1: Average retrieval quality (from RRF scores)
        scores = [r.get("rrf_score", 0.5) for r in retrieved_chunks]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Factor 2: Citation presence
        has_citations = bool(re.findall(r'\[D\d{2}\]', answer))
        citation_bonus = 0.2 if has_citations else 0.0

        # Factor 3: Number of unique documents in results
        papers = set(r.get("metadata", {}).get("paper_id", "") for r in retrieved_chunks)
        diversity_bonus = min(len(papers) / 3.0, 0.2)

        # Factor 4: Answer substance
        word_count = len(answer.split())
        substance_score = min(word_count / 100.0, 0.3)

        confidence = min(
            avg_score * 0.3 + citation_bonus + diversity_bonus + substance_score,
            1.0,
        )
        return confidence

    def llm_verify(self, answer: str, context: str, query: str) -> dict:
        """Use LLM-as-judge to verify faithfulness and groundedness."""
        prompt = VERIFICATION_PROMPT.format(
            context=context,
            answer=answer,
            query=query,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            result_text = response.choices[0].message.content

            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "faithfulness": 0.5,
                    "groundedness": 0.5,
                    "completeness": 0.5,
                    "unsupported_claims": [],
                    "verdict": "UNKNOWN",
                }
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return {
                "faithfulness": 0.5,
                "groundedness": 0.5,
                "completeness": 0.5,
                "verdict": "ERROR",
                "error": str(e),
            }

    def strip_invalid_citations(self, answer: str, context: str) -> tuple[str, list[str]]:
        """Remove citations that don't appear in the context."""
        citation_result = self.verify_citations(answer, context)
        invalid = citation_result["invalid_citations"]
        if not invalid:
            return answer, []

        cleaned = answer
        for cit in invalid:
            # Remove patterns like [D03] or ", [D03]" or "and [D03]"
            cleaned = re.sub(rf'\s*,?\s*{re.escape(cit)}', '', cleaned)

        # Clean up artifacts: double spaces, empty brackets, orphaned commas
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        cleaned = re.sub(r'\[\s*,\s*', '[', cleaned)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = re.sub(r'\[\s*\]', '', cleaned)
        # Fix "and," or "and ," artifacts
        cleaned = re.sub(r'\band\s*,', 'and', cleaned)
        # Fix trailing "and " before period
        cleaned = re.sub(r'\band\s*\.', '.', cleaned)

        return cleaned.strip(), invalid

    def should_refuse(
        self,
        answer: str,
        context: str,
        retrieved_chunks: list[dict],
    ) -> tuple[bool, str]:
        """Determine if the system should refuse to answer."""
        confidence = self.compute_confidence(answer, retrieved_chunks)
        if confidence < self.confidence_threshold:
            return True, (
                "Low confidence: insufficient relevant information in the "
                "retrieved context to answer this question reliably."
            )

        # Check citation validity — only refuse if ALL citations are invalid
        # (individual invalid citations are stripped in guard())
        citation_result = self.verify_citations(answer, context)
        if (citation_result["has_citations"]
                and citation_result["citation_accuracy"] == 0.0):
            return True, (
                f"All citations invalid: {citation_result['invalid_citations']}. "
                "Answer cites only documents not present in retrieved context."
            )

        return False, "Answer passes basic verification."

    def guard(
        self,
        answer: str,
        context: str,
        query: str,
        retrieved_chunks: list[dict],
        run_llm_verify: bool = False,
    ) -> dict:
        """Run full anti-hallucination pipeline."""
        # Strip invalid citations first (keep the answer, just remove bad refs)
        cleaned_answer, stripped_citations = self.strip_invalid_citations(
            answer, context,
        )
        if stripped_citations:
            logger.info(
                f"Stripped invalid citations: {stripped_citations}"
            )

        citation_check = self.verify_citations(cleaned_answer, context)
        confidence = self.compute_confidence(cleaned_answer, retrieved_chunks)
        should_refuse, refuse_reason = self.should_refuse(
            cleaned_answer, context, retrieved_chunks,
        )

        result = {
            "original_answer": answer,
            "confidence": confidence,
            "citation_check": citation_check,
            "should_refuse": should_refuse,
            "refuse_reason": refuse_reason,
        }
        if stripped_citations:
            result["stripped_citations"] = stripped_citations

        if run_llm_verify:
            result["llm_verification"] = self.llm_verify(
                cleaned_answer, context, query,
            )

        if should_refuse:
            result["final_answer"] = (
                "I cannot provide a reliable answer to this question. "
                f"Reason: {refuse_reason}\n\n"
                "Please try rephrasing the question or asking about specific "
                "products or buying guides in the corpus."
            )
        else:
            result["final_answer"] = cleaned_answer

        return result
