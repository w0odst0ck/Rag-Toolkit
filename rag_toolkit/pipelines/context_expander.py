"""
rag-toolkit / pipelines / context_expander.py

Context expansion and compression utilities for RAG pipelines.

Classes:
- BM25Compressor: reduce context length while preserving query-relevant sentences.
- ContextExpander: expand a paragraph to its surrounding context group.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Compressor:
    """Compress a document by keeping only the sentences most relevant to a query.

    Uses BM25 scoring with a sliding-window de-duplication strategy to avoid
    selecting near-duplicate sentences.
    """

    def __init__(self, rate: float = 0.5, window_size: int = 3):
        """
        Args:
            rate: Target compression ratio (e.g., 0.5 = keep 50% of characters).
            window_size: Disallow selecting sentences within this many positions
                of an already-selected sentence.
        """
        self.rate = rate
        self.window_size = window_size

    def compress(self, context: str, query: Optional[str] = None) -> str:
        """Return a compressed version of *context* keeping only query-relevant sentences."""
        sentences = self._cut_sentences(context)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return ""

        # 短文档不压缩：句子数太少时窗口抑制会误删关键句（如 FAQ 的 Q/A 两短句），
        # 压缩反而丢信息，直接返回原文。
        if len(sentences) <= self.window_size * 2 + 1:
            return context

        tokenized = [jieba.lcut(s) for s in sentences]
        bm25 = BM25Okapi(tokenized)

        tokenized_query = jieba.lcut(query or context)
        scores = bm25.get_scores(tokenized_query)

        sorted_idx = scores.argsort()[::-1]
        target_len = int(len(context) * self.rate)

        selected = []
        now_len = 0
        for idx in sorted_idx:
            if now_len >= target_len:
                break
            # window check
            if any(abs(idx - s) <= self.window_size for s in selected):
                continue
            selected.append(idx)
            now_len += len(sentences[idx])

        selected.sort()
        return "".join(sentences[i] for i in selected)

    @staticmethod
    def _cut_sentences(para: str) -> List[str]:
        para = re.sub(r"([。！？\?])([^”’])", r"\1\n\2", para)
        para = re.sub(r"(\…{2})([^”’])", r"\1\n\2", para)
        para = re.sub(r"([。！？\?][”’])([^。！？\?])", r"\1\n\2", para)
        # 英文标点切分：句点后跟空格+大写/数字（避免小数 3.7、缩写 U.S.）
        para = re.sub(r"\.(\s+[A-Z0-9])", r".\n\1", para)
        # 冒号后跟空格+大写/数字（避免 https://、9:00 时间）
        para = re.sub(r":(\s+[A-Z0-9])", r":\n\1", para)
        return para.split("\n")


class ContextExpander:
    """Expand a hit paragraph to its surrounding context group.

    In structured documents (e.g., regulations), each paragraph belongs to a
    *group_id*. This expander retrieves adjacent paragraphs within the same
    group, plus optionally the groups immediately before and after.
    """

    def __init__(self, expand_before: bool = True, expand_after: bool = True):
        self.expand_before = expand_before
        self.expand_after = expand_after

    def expand(
        self,
        hit_paragraph_id: str,
        hit_group_id: str,
        all_paragraphs: List[dict],
        group_ids: List[str],
    ) -> str:
        """Expand *hit_paragraph_id* with context from surrounding paragraphs.

        Args:
            hit_paragraph_id: The matched paragraph ID.
            hit_group_id: The group ID of the matched paragraph.
            all_paragraphs: All paragraphs from the same document, each as
                ``{"paragraph_id": str, "group_id": str, "content": str}``.
            group_ids: Ordered list of group IDs in the document.

        Returns:
            Concatenated context: ``before_content + hit_content + after_content``.
        """
        content = ""
        try:
            idx = group_ids.index(hit_group_id)
        except ValueError:
            return ""

        before_gid = group_ids[idx - 1] if self.expand_before and idx > 0 else None
        after_gid = group_ids[idx + 1] if self.expand_after and idx < len(group_ids) - 1 else None

        for para in all_paragraphs:
            if para.get("paragraph_id") == hit_paragraph_id:
                content = para.get("content", "")
                break

        before_content = self._group_content(all_paragraphs, before_gid)
        after_content = self._group_content(all_paragraphs, after_gid)

        return before_content + content + after_content

    @staticmethod
    def _group_content(all_paragraphs: List[dict], group_id: Optional[str]) -> str:
        if group_id is None:
            return ""
        parts = [p["content"] for p in all_paragraphs if p.get("group_id") == group_id]
        return "".join(parts)

    @staticmethod
    def group_paragraphs(paragraphs: List[dict]) -> dict:
        """Group a flat paragraph list by ``group_id``, preserving order.

        Returns:
            ``{group_id: sorted_list_of_paragraphs}``
        """
        groups = {}
        for p in paragraphs:
            gid = p.get("group_id")
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(p)
        for gid in groups:
            groups[gid].sort(key=lambda x: int(x.get("paragraph_id", 0)))
        return groups

    @staticmethod
    def get_group_ids(paragraphs: List[dict]) -> List[str]:
        """Return ordered unique group IDs from a paragraph list."""
        seen = set()
        ordered = []
        for p in paragraphs:
            gid = p.get("group_id")
            if gid and gid not in seen:
                seen.add(gid)
                ordered.append(gid)
        return ordered
