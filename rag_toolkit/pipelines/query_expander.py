"""
rag-toolkit / pipelines / query_expander.py

Multi-query generation for improved retrieval coverage.

Classes:
- MultiQueryGenerator: LLM-based query expansion.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class MultiQueryGenerator:
    """Generate multiple re-phrasings of a user query to improve recall.

    Two backends available:
    1. **LLM-based** — uses an OpenAI-compatible chat API to generate alternatives.
    2. **Template-based** — uses simple sentence templates (fallback).
    """

    def __init__(
        self,
        api_key: str = "sk-xxx",
        base_url: str = "http://localhost:8000/v1",
        model: str = "qwen2.5-14b-instruct",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        prompt_template: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._prompt = prompt_template or self._default_prompt()

    @staticmethod
    def _default_prompt() -> str:
        return """\
You are an AI language model assistant. Your task is to generate five different
versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is
to help the user overcome some of the limitations of distance-based similarity
search.

Provide these alternative questions separated by newlines.

Original question: {question}"""

    def generate(self, question: str, count: int = 5) -> List[str]:
        """Generate *count* alternative queries.

        Returns:
            A list of query strings (may be empty on failure).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self._prompt.format(question=question)}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            lines = [
                line.strip().lstrip("0123456789.-* ") for line in content.split("\n") if line.strip()
            ]
            queries = [q.strip('"\'') for q in lines if q.strip()][:count]
            logger.info(f"Generated {len(queries)} alternative queries.")
            return queries
        except Exception as e:
            logger.warning(f"MultiQueryGenerator failed: {e}")
            return []

    def generate_structured(self, question: str) -> List[str]:
        """Generate queries with a structured JSON return format."""
        prompt = f"""\
任务：根据用户输入的query，生成 3-5 个语义相似的问法。
输出严格 JSON: {{"生成问题": ["问题1", "问题2", "问题3"]}}
用户问题: {question}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            # Try to parse JSON
            if "{" in content:
                json_str = content[content.index("{"): content.rindex("}") + 1]
                data = json.loads(json_str)
                queries = data.get("生成问题", [])
                logger.info(f"Structured generation: {len(queries)} queries.")
                return queries
            return []
        except Exception as e:
            logger.warning(f"Structured query generation failed: {e}")
            return []


# ── Template-Based Fallback ───────────────────────────────────────────

def template_expand(question: str) -> List[str]:
    """Simple template-based query expansion (no LLM needed)."""
    expansions = [
        question,
        f"关于{question}的规定",
        f"{question}相关法律要求",
        f"请说明{question}",
    ]
    return list(set(expansions))
