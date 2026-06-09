"""Answer quality metrics using LLM-as-judge.

Evaluates correctness, faithfulness, and groundedness.
"""

import json
import logging
import re
import time
from typing import Optional

from openai import OpenAI, RateLimitError

from config.settings import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL
from config.prompts import JUDGE_CORRECTNESS_PROMPT

logger = logging.getLogger(__name__)


class AnswerMetrics:
    """LLM-as-judge evaluation for generated answers."""

    def __init__(
        self,
        model: str = LLM_MODEL,
    ):
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        self.model = model

    def judge_answer(
        self,
        query: str,
        generated_answer: str,
        gold_answer: str,
    ) -> dict:
        """Use LLM to judge answer quality against gold standard.

        Args:
            query: Original question.
            generated_answer: RAG-generated answer.
            gold_answer: Gold-standard reference answer.

        Returns:
            Dict with correctness, faithfulness, completeness scores.
        """
        prompt = JUDGE_CORRECTNESS_PROMPT.format(
            query=query,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )

        try:
            for attempt in range(5):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=300,
                    )
                    break
                except RateLimitError:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning(f"Judge rate limited, waiting {wait}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
                    if attempt == 4:
                        raise
            result_text = response.choices[0].message.content

            # Parse JSON from response — try multiple strategies
            scores = self._parse_judge_json(result_text)
            if scores is not None:
                return {
                    "correctness": float(scores.get("correctness", 0)),
                    "faithfulness": float(scores.get("faithfulness", 0)),
                    "completeness": float(scores.get("completeness", 0)),
                    "explanation": scores.get("explanation", ""),
                }
            else:
                logger.warning(f"Could not parse judge response: {result_text[:200]}")
                return self._default_scores("Parse error")

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return self._default_scores(str(e))

    def _parse_judge_json(self, text: str) -> Optional[dict]:
        # Strategy 1: standard JSON extraction
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: greedy brace matching for nested content
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 3: regex extract individual numeric scores
        c = re.search(r'"correctness"\s*:\s*([\d.]+)', text)
        f = re.search(r'"faithfulness"\s*:\s*([\d.]+)', text)
        comp = re.search(r'"completeness"\s*:\s*([\d.]+)', text)
        if c and f and comp:
            return {
                "correctness": float(c.group(1)),
                "faithfulness": float(f.group(1)),
                "completeness": float(comp.group(1)),
                "explanation": "Parsed via regex fallback",
            }

        return None

    def _default_scores(self, reason: str) -> dict:
        return {
            "correctness": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "explanation": f"Evaluation failed: {reason}",
        }

    def compute_citation_accuracy(
        self,
        answer: str,
        relevant_paper_ids: list[str],
    ) -> dict:
        """Check if cited documents match expected relevant ones."""
        # Extract citations from answer
        cited = set(re.findall(r'\[?(D(?:0[1-9]|1[0-9]|20))\]?', answer))
        relevant = set(relevant_paper_ids)

        correct_citations = cited & relevant
        incorrect_citations = cited - relevant
        missed_citations = relevant - cited

        precision = len(correct_citations) / len(cited) if cited else 0
        recall = len(correct_citations) / len(relevant) if relevant else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "citation_precision": precision,
            "citation_recall": recall,
            "citation_f1": f1,
            "correct_citations": list(correct_citations),
            "incorrect_citations": list(incorrect_citations),
            "missed_citations": list(missed_citations),
            "total_cited": len(cited),
        }

    def evaluate_answer(
        self,
        query: str,
        generated_answer: str,
        gold_answer: str,
        relevant_paper_ids: list[str],
    ) -> dict:
        """Full answer evaluation: LLM judge scores + citation accuracy."""
        judge_scores = self.judge_answer(query, generated_answer, gold_answer)
        citation_scores = self.compute_citation_accuracy(
            generated_answer, relevant_paper_ids,
        )

        return {
            **judge_scores,
            **citation_scores,
        }
