"""
rag-toolkit / pipelines / reranker.py

Re-ranking modules for RAG pipelines.

Classes:
- XReranker: thin wrapper around a Xinference (or OpenAI-compatible) rerank model.
- TwoStageReranker: article-level → paragraph-level cascading rerank.
Functions:
- knowsrerank: single-pass rerank with dynamic threshold filtering.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from xinference.client import Client

logger = logging.getLogger(__name__)


# ── XReranker ─────────────────────────────────────────────────────────

class XReranker:
    """Rerank documents using a model hosted on Xinference (or any server that exposes
    the ``<model>/rerank`` endpoint).

    Args:
        base_url: Xinference server URL.
        model_name: Model name registered in Xinference.
        top_k: Number of top documents to keep (overridden by caller if needed).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9997",
        model_name: str = "bge-reranker-v2-m3",
        top_k: int = 10,
    ):
        self._model = Client(base_url).get_model(model_name)
        self.top_k = top_k

    def rerank(self, documents: Sequence[str], query: str) -> List[Tuple[int, float]]:
        """Return a list of ``(index, relevance_score)`` sorted by score descending."""
        if not documents:
            return []
        response = self._model.rerank(documents, query)
        results: list[dict] = response["results"]
        return [(item["index"], item["relevance_score"]) for item in results]


# ── Single-pass Rerank ───────────────────────────────────────────────

def knowsrerank(
    query: str,
    knowledges: List[Dict],
    corpus: List[str],
    threshold: float = 0.7,
    min_count: int = 5,
    threshold_strategy: Literal["fixed", "dynamic"] = "dynamic",
    reranker: Optional[XReranker] = None,
) -> Tuple[List[Dict], List[str]]:
    """Rerank knowledge items against *query* and filter by threshold.

    Args:
        query: User query.
        knowledges: List of dicts (must have ``"title"`` and ``"paragraph_id"``).
        corpus: Parallel list of text strings to rerank.
        threshold: Score cutoff (used directly when strategy is "fixed").
        min_count: Minimum results to keep; triggers dynamic threshold below it.
        threshold_strategy:
            - ``"fixed"``: use *threshold* as-is.
            - ``"dynamic"``: use **mean + 0.5 * std** when results drop below *min_count*.
        reranker: ``XReranker`` instance; created with defaults if omitted.

    Returns:
        ``(reranked_knowledges, paragraph_ids)``
    """
    if reranker is None:
        reranker = XReranker()

    if not knowledges or not corpus:
        return [], []

    try:
        results = reranker.rerank(corpus, query)
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        return [], []

    # Attach title for logging
    for i, item in enumerate(results):
        idx = item[0]
        relevance_score = item[1]
        knowledges[idx]["_rerank_score"] = relevance_score
        logger.info(
            f"[Rerank] Rank {i+1}: title={knowledges[idx].get('title','')} "
            f"score={relevance_score:.4f}"
        )

    scores = [item[1] for item in results]

    # Dynamic threshold
    if threshold_strategy == "fixed":
        final_threshold = threshold
    else:
        above = sum(1 for s in scores if s >= threshold)
        if above >= min_count:
            final_threshold = threshold
        else:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            final_threshold = min(mean_score + 0.5 * std_score, threshold)

    logger.info(f"Rerank threshold ({threshold_strategy}): {final_threshold:.4f}")

    reranked, pids = [], []
    for item in results:
        idx, score = item
        if score < final_threshold:
            continue
        k = dict(knowledges[idx])
        k["score"] = float(score)
        reranked.append(k)
        pids.append(k.get("paragraph_id", ""))

    return reranked, pids


# ── Two-Stage Reranker ───────────────────────────────────────────────

class TwoStageReranker:
    """Cascade rerank: article-level → paragraph-level.

    1. Group paragraphs by article (``law_id``), concat at *paragraph_length* chars,
       rerank articles, keep top *top_k_articles*.
    2. Rerank all paragraphs from surviving articles, keep top *top_k_paragraphs*.
    """

    def __init__(
        self,
        reranker: XReranker,
        top_k_articles: int = 5,
        top_k_paragraphs: int = 10,
        paragraph_length: int = 1024,
    ):
        self.reranker = reranker
        self.top_k_articles = top_k_articles
        self.top_k_paragraphs = top_k_paragraphs
        self.paragraph_length = paragraph_length

    @staticmethod
    def _extract_core(query: str, paragraphs: List[str], max_length: int) -> str:
        """Truncate each paragraph to ``max_length // num_paragraphs`` chars."""
        if not paragraphs:
            return ""
        avg = max_length // len(paragraphs)
        return " ".join(p[:avg] for p in paragraphs)

    def rerank(self, knowledges: List[Dict], query: str) -> List[Dict]:
        if not knowledges:
            return []

        # Stage 1 ── group → article-level rerank
        article_paras: Dict[str, List[str]] = defaultdict(list)
        article_titles: Dict[str, str] = {}

        for k in knowledges:
            aid = str(k.get("law_id", ""))
            if aid not in article_titles:
                article_titles[aid] = k.get("title", "")
            article_paras[aid].append(k.get("content", ""))

        articles = {
            aid: self._extract_core(query, paras, self.paragraph_length)
            for aid, paras in article_paras.items()
        }
        article_ids = list(articles.keys())
        article_texts = list(articles.values())
        if not article_texts:
            return []

        article_results = self.reranker.rerank(article_texts, query)
        article_results.sort(key=lambda x: x[1], reverse=True)
        top_article_ids = set(
            article_ids[idx] for idx, _ in article_results[: self.top_k_articles]
        )

        for i, (idx, score) in enumerate(article_results):
            logger.info(
                f"[Stage1] Rank {i+1}: {article_titles.get(article_ids[idx],'')} "
                f"score={score:.4f}"
            )

        # Stage 2 ── paragraph rerank within top articles
        top_paragraphs: List[str] = []
        top_knowledges: List[Dict] = []
        for k in knowledges:
            if str(k.get("law_id", "")) in top_article_ids:
                top_paragraphs.append(k.get("content", ""))
                top_knowledges.append(k)

        if not top_paragraphs:
            return []

        para_results = self.reranker.rerank(top_paragraphs, query)
        para_results.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, _ in para_results[: self.top_k_paragraphs]]

        reranked = []
        for i in top_indices:
            k = dict(top_knowledges[i])
            k["article_score"] = next(
                (s for aid, s in zip(article_ids, [r[1] for r in article_results])
                 if aid == str(k.get("law_id", ""))),
                0.0,
            )
            k["paragraph_score"] = float(para_results[
                [r[0] for r in para_results].index(i)  # find the right score
            ][1])
            reranked.append(k)

        logger.info(f"[Stage2] Reranked {len(reranked)} paragraphs.")
        return reranked

    def save_stage1_csv(self, data: List[Dict], filename: Optional[str] = None) -> str:
        filename = filename or f"stage1_{datetime.now():%Y%m%d_%H%M%S}.csv"
        if not data:
            return filename
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)
        logger.info(f"Stage1 results saved to {filename}")
        return filename
