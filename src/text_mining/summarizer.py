"""Summarization: extractive + abstractive using LLM map-reduce."""

import logging
from typing import Optional

from openai import OpenAI

from config.settings import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL
from config.prompts import MAP_SUMMARY_PROMPT, REDUCE_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class Summarizer:
    """Extractive + abstractive summarization for documents and retrieved chunks."""

    def __init__(
        self,
        model: str = LLM_MODEL,
    ):
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        self.model = model

    def summarize_text(
        self,
        text: str,
        max_tokens: int = 300,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """Summarize a single text passage."""
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = MAP_SUMMARY_PROMPT.format(text=text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def summarize_paper(self, paper_data: dict) -> dict:
        """Summarize a full document using map-reduce over sections."""
        sections = paper_data.get("sections", [])

        if not sections:
            # No sections detected: summarize full text directly
            summary = self.summarize_text(paper_data["full_text"][:8000])
            return {
                "paper_id": paper_data["paper_id"],
                "section_summaries": [],
                "combined_summary": summary,
            }

        # Map: summarize each section
        section_summaries = []
        for section in sections:
            if len(section["text"]) < 50:
                continue
            # Truncate very long sections
            text = section["text"][:6000]
            summary = self.summarize_text(text)
            section_summaries.append({
                "section": section["section"],
                "summary": summary,
            })

        # Reduce: combine section summaries
        all_summaries = "\n\n".join(
            f"**{s['section']}:** {s['summary']}"
            for s in section_summaries
        )

        prompt = REDUCE_SUMMARY_PROMPT.format(summaries=all_summaries)
        combined = self.summarize_text(
            all_summaries, max_tokens=500, custom_prompt=prompt,
        )

        return {
            "paper_id": paper_data["paper_id"],
            "section_summaries": section_summaries,
            "combined_summary": combined,
        }

    def summarize_retrieved_chunks(
        self,
        chunks: list[dict],
        query: str,
    ) -> str:
        """Synthesize a summary from multiple retrieved chunks, focused on the query."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            paper_id = chunk.get("metadata", {}).get("paper_id", "?")
            context_parts.append(f"[{paper_id}]: {chunk['text']}")

        context = "\n\n".join(context_parts)

        prompt = (
            f"Based on the following excerpts from IKEA buying guides, "
            f"provide a concise synthesis addressing "
            f"this question: {query}\n\n"
            f"Excerpts:\n{context}\n\n"
            f"Synthesized summary:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
