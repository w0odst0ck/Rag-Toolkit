"""
rag-toolkit / pipelines / base_rag.py

Generic RAG pipeline base class.

Provides a configurable multi-stage retrieve-rerank-generate flow:
  1. (Optional) Title-level recall to narrow search scope
  2. Query expansion (multi-query generation)
  3. Forward retrieval (within top-k titles) + reverse retrieval (outside top-k)
  4. Rerank with dynamic threshold
  5. Knowledge grouping / deduplication / scoring
  6. LLM-based answer generation

Subclass and override hook methods for domain-specific logic.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import openai
import requests
from openai import OpenAI

from rag_toolkit.core.config import Config
from rag_toolkit.pipelines.context_expander import BM25Compressor, ContextExpander
from rag_toolkit.pipelines.query_expander import MultiQueryGenerator
from rag_toolkit.pipelines.reranker import XReranker, knowsrerank
from rag_toolkit.storage.milvus_manager import MilvusManager

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def get_top_k_ids(scores_dict: Dict[str, Dict], k: int) -> List[str]:
    """Return the top-k keys from a dict of ``{id: {score: ..., ...}}`` sorted by score desc."""
    if not scores_dict:
        return []
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1].get("score", 0), reverse=True)
    return [item[0] for item in sorted_items[:k]]


# ── Knowledge Item Schema ─────────────────────────────────────────────

KNOWLEDGE_FIELDS = [
    "doc_id",           # document / article identifier
    "paragraph_id",     # paragraph / chunk identifier
    "group_id",         # logical group within a document
    "title",            # document title
    "content",          # text content
    "score",            # relevance score (populated after rerank)
]


# ── Base Pipeline ─────────────────────────────────────────────────────

class BaseRAGPipeline(ABC):
    """Abstract base for a complete RAG pipeline.

    Subclass and implement the data-source hooks:

    - ``fetch_document_metadata(doc_ids)``
    - ``fetch_paragraph_content(doc_ids)``
    - ``format_knowledge_prompt(knowledge_list)``

    Or override the relevant methods for deeper customisation.
    """

    def __init__(self, config: Optional[Config] = None, milvus_manager: Optional[MilvusManager] = None):
        self.cfg = config or Config()
        self.milvus = milvus_manager or MilvusManager(config=self.cfg)

        # ── Embedding client ──
        self._embedding_url = f"{self.cfg.EMBEDDING_URL.rstrip('/')}/embeddings"

        # ── Reranker ──
        self.xreranker = XReranker(
            base_url=self.cfg.RERANKER_URL,
            model_name=self.cfg.RERANKER_MODEL,
        )

        # ── LLM ──
        self.llm = OpenAI(
            api_key=self.cfg.LLM_API_KEY,
            base_url=self.cfg.LLM_URL,
        )
        self.llm_model = self.cfg.LLM_MODEL

        # ── Query expander ──
        self.query_expander = MultiQueryGenerator(
            api_key=self.cfg.LLM_API_KEY,
            base_url=self.cfg.LLM_URL,
            model=self.cfg.LLM_MODEL,
        )

        # ── BM25 compressor ──
        self.compressor = BM25Compressor(rate=0.7, window_size=10)

        # ── Context expander ──
        self.context_expander = ContextExpander()

    # ── Abstract Hooks (implement in subclass) ────────────────────────

    @abstractmethod
    def fetch_document_metadata(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """Fetch metadata (title, publish date, status, etc.) for document IDs.

        Returns:
            ``{doc_id: {"title": str, ...}}``
        """
        ...

    @abstractmethod
    def fetch_paragraph_content(self, doc_ids: List[str]) -> Dict[str, List[Dict]]:
        """Fetch full paragraph content for document IDs.

        Returns:
            ``{doc_id: [{"paragraph_id": str, "group_id": str, "content": str}, ...]}``
        """
        ...

    def format_knowledge_prompt(self, knowledge_list: List[Dict]) -> str:
        """Format a list of knowledge items into a single string prompt for the LLM.

        Override for custom formatting (e.g., adding metadata headers).
        """
        parts = []
        for k in knowledge_list:
            title = k.get("title", "")
            content = k.get("content", "")
            parts.append(f"---\n标题：{title}\n内容：{content}\n")
        return "\n".join(parts)

    # ── Embedding ────────────────────────────────────────────────────

    def text_to_embedding(self, texts: List[str]) -> List[float]:
        """Convert text to dense vector via the configured embedding endpoint.

        Expects the endpoint to accept ``{"texts": [...], "return_dense": true}``
        and return ``[{dense_vector: {vector: [...]}}]``.
        """
        payload = {"texts": texts, "return_dense": True, "return_sparse": False}
        try:
            resp = requests.post(self._embedding_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [item["dense_vector"]["vector"] for item in data]
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    # ── Search ───────────────────────────────────────────────────────

    def search_data(
        self,
        query: str,
        collection_name: str,
        vec_limit: int = 300,
        full_limit: int = 300,
        res_limit: int = 100,
        threshold: float = 0.5,
        full_expr: str = "",
        vec_expr: str = "",
    ) -> List[Dict]:
        """Hybrid search (dense + BM25) on *collection_name*.

        Returns a list of hit dicts with keys: ``doc_id``, ``paragraph_id``,
        ``title``, ``content``, ``similarity``.
        """
        embedding = self.text_to_embedding([query])[0]

        try:
            results = self.milvus.hybrid_search(
                collection_name=collection_name,
                text_data=[query],
                text_anns_field="sparse_text",
                vec_data=[embedding],
                vec_anns_field="vector",
                output_fields=["doc_id", "paragraph_id", "title", "content"],
                full_expr=full_expr,
                vec_expr=vec_expr,
                vec_limit=vec_limit,
                full_limit=full_limit,
                res_limit=res_limit,
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

        hits = []
        seen_ids = set()
        for result in results:
            for item in result:
                pid = item["entity"].get("paragraph_id", "")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                score = item.get("distance", 0)
                if score < threshold:
                    continue
                hits.append({
                    "doc_id": item["entity"].get("doc_id", ""),
                    "paragraph_id": pid,
                    "title": item["entity"].get("title", ""),
                    "content": item["entity"].get("content", ""),
                    "similarity": float(score),
                })
        return hits

    # ── Title-Level Recall ───────────────────────────────────────────

    def recall_titles(
        self, query: str, collection_name: str, top_k: int = 50
    ) -> Tuple[List[str], List[str], Dict[str, Dict]]:
        """Recall document IDs by title-level search + optional LLM query rewriting.

        Returns:
            ``(title_ids, queries, statute_info_dict)``
            where ``statute_info_dict = {doc_id: {score, title, ...}}``
        """
        queries = []
        statute_info_dict: Dict[str, Dict] = {}

        try:
            # Get embedding for title-level search
            embedding = self.text_to_embedding([query])[0]
            results = self.milvus.search(
                collection_name=collection_name,
                vectors=[embedding],
                anns_field="vector",
                limit=top_k,
                output_fields=["doc_id", "title"],
            )
            for item in results[0]:
                did = item["entity"].get("doc_id", "")
                score = item.get("distance", 0)
                title = item["entity"].get("title", "")
                if did:
                    statute_info_dict[did] = {"score": score, "title": title}

            # Query expansion (generated as supplementary text search but may be unused)
            try:
                queries = self.query_expander.generate(query)
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Title recall failed: {e}")

        title_ids = list(statute_info_dict.keys())
        return title_ids, queries, statute_info_dict

    # ── Recall & Rerank ──────────────────────────────────────────────

    def recall_and_rerank(
        self,
        query: str,
        collection_name: str,
        condition: Optional[str] = None,
        threshold: float = 0.5,
        threshold_strategy: Literal["fixed", "dynamic"] = "dynamic",
        gold_doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Single recall + rerank cycle.

        Args:
            query: Search query.
            collection_name: Milvus collection.
            condition: Milvus expression filter (e.g., ``doc_id in [...]``).
            threshold: Score threshold after rerank.
            threshold_strategy: "fixed" or "dynamic".
            gold_doc_ids: Boost recall for these doc IDs with context compression.

        Returns:
            List of reranked knowledge dicts with ``score`` field.
        """
        # 1. Initial recall
        hits = self.search_data(query, collection_name, threshold=0.3, full_expr=condition or "", vec_expr=condition or "")
        if not hits:
            return []

        # 2. Deduplicate and build knowledge items
        seen_paras = set()
        knowledges = []
        for h in hits:
            pid = h["paragraph_id"]
            if pid in seen_paras:
                continue
            seen_paras.add(pid)
            knowledges.append({
                "doc_id": h["doc_id"],
                "paragraph_id": pid,
                "title": h["title"],
                "content": h["content"],
                "group_id": "",
            })

        # 3. Attach document metadata
        doc_ids = list(set(k["doc_id"] for k in knowledges))
        metadata = self.fetch_document_metadata(doc_ids)
        for k in knowledges:
            if k["doc_id"] in metadata:
                k["title"] = metadata[k["doc_id"]].get("title", k["title"])
                k.setdefault("status", metadata[k["doc_id"]].get("status", ""))
                k["reference_no"] = metadata[k["doc_id"]].get("reference_no", "")

        # 4. Cross-reference full paragraph content for group expansion
        paragraphs_by_doc = self.fetch_paragraph_content(doc_ids)
        for k in knowledges:
            doc_paras = paragraphs_by_doc.get(k["doc_id"], [])
            for p in doc_paras:
                if p["paragraph_id"] == k["paragraph_id"]:
                    k["group_id"] = p.get("group_id", "")
                    break

            # Expand context: if we have group info, expand to surrounding paragraphs
            if k["group_id"] and k["doc_id"] in paragraphs_by_doc:
                doc_paras = paragraphs_by_doc[k["doc_id"]]
                group_ids = self.context_expander.get_group_ids(doc_paras)
                expanded = self.context_expander.expand(
                    k["paragraph_id"], k["group_id"], doc_paras, group_ids
                )
                if expanded:
                    k["content"] = expanded

        # 5. BM25 compression for gold documents
        gold_ids = set(gold_doc_ids or [])
        for k in knowledges:
            if k["doc_id"] in gold_ids:
                k["content"] = self.compressor.compress(k["content"], query=query)

        # 6. Rerank
        corpus = [
            f"{k.get('title', '')}。{k.get('content', '')}" for k in knowledges
        ]
        reranked, _ = knowsrerank(
            query, knowledges, corpus,
            threshold=threshold,
            threshold_strategy=threshold_strategy,
            reranker=self.xreranker,
        )
        return reranked

    # ── Group & Score ────────────────────────────────────────────────

    def group_and_score(
        self,
        reranked_knowledges: List[Dict],
        top_n_docs: int = 5,
        top_k_paras: int = 5,
        doc_scores: Optional[Dict[str, float]] = None,
        alpha: float = 0.5,
    ) -> List[Dict]:
        """Group reranked paragraphs by doc ID, score and sort documents, then
        return top paragraphs from top documents.

        Scoring:
            doc_score = (1 - alpha) * max_para_score + alpha * title_relevance_score
        """
        # Deduplicate paragraphs (keep max score)
        dedup: Dict[str, Dict] = {}
        for k in reranked_knowledges:
            pid = k["paragraph_id"]
            if pid not in dedup or k.get("score", 0) > dedup[pid].get("score", 0):
                dedup[pid] = k
        deduplicated = list(dedup.values())

        # Group by doc_id
        by_doc: Dict[str, List[Dict]] = defaultdict(list)
        for k in deduplicated:
            by_doc[k["doc_id"]].append(k)

        # Score documents
        doc_scores_out: Dict[str, float] = {}
        for did, paras in by_doc.items():
            para_scores = sorted(paras, key=lambda x: x.get("score", 0), reverse=True)
            top_scores = [p["score"] for p in para_scores[:top_k_paras] if p.get("score")]
            max_score = max(top_scores) if top_scores else 0
            title_score = doc_scores.get(did, 0) if doc_scores else 0
            doc_scores_out[did] = (1 - alpha) * max_score + alpha * title_score

        # Sort documents by score
        sorted_docs = sorted(by_doc.items(), key=lambda x: doc_scores_out[x[0]], reverse=True)

        # Output top paragraphs
        results = []
        for did, paras in sorted_docs[:top_n_docs]:
            paras.sort(key=lambda x: x.get("score", 0), reverse=True)
            results.extend(paras[:top_k_paras])

        logger.info(f"Group & score: {len(results)} paragraphs from top {top_n_docs} docs.")
        return results

    # ── LLM Generation ───────────────────────────────────────────────

    def llm_chat(self, messages: List[Dict], stream: bool = False, temperature: Optional[float] = None) -> str:
        """Call the configured LLM."""
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.cfg.LLM_MAX_TOKENS,
            temperature=temperature or self.cfg.LLM_TEMPERATURE,
            stream=stream,
        )
        if stream:
            full = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full += chunk.choices[0].delta.content
            return full
        return response.choices[0].message.content or ""

    # ── Main RAG Flow ────────────────────────────────────────────────

    def rag(
        self,
        query: str,
        collection_name: str,
        history: Optional[List[Dict]] = None,
        stream: bool = False,
        rewrite: bool = True,
        system_prompt: Optional[str] = None,
        qa_prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete RAG pipeline: recall → rerank → group → generate.

        Args:
            query: User question.
            collection_name: Milvus collection to search.
            history: Previous conversation turns (OpenAI message format).
            stream: Whether to stream LLM output.
            rewrite: Whether to perform query expansion.
            system_prompt: Override default system prompt.
            qa_prompt_template: Override default QA template (use ``{query}`` and ``{context}``).

        Returns:
            ``{"answer": str, "sources": [knowledge_dict, ...], "queries": [...]}``
        """
        history = history or []
        logger.info(f"=== RAG Start === query={query}, collection={collection_name}")

        result = {
            "answer": "",
            "sources": [],
            "queries": [],
        }

        # ── Step 1: Title recall + query expansion ──
        title_ids, queries, title_scores = self.recall_titles(query, collection_name)
        result["queries"] = queries

        filter_cond = None
        reverse_filter_cond = None
        if title_ids:
            ids_expr = ", ".join(f"'{id}'" for id in title_ids)
            filter_cond = f"doc_id in [{ids_expr}]"
            reverse_filter_cond = f"doc_id not in [{ids_expr}]"

        # Build query list
        query_list = [query]
        if queries and rewrite:
            query_list.extend(queries[:5])

        # ── Step 2: Multi-query forward + reverse recall ──
        gold_top_k = 30
        gold_ids = get_top_k_ids(title_scores, k=gold_top_k)

        all_results = []
        for q in query_list:
            logger.info(f"--- Sub-query: {q} ---")
            forward = self.recall_and_rerank(
                q, collection_name,
                condition=filter_cond,
                threshold=0.4,
                gold_doc_ids=gold_ids,
            )
            reverse = self.recall_and_rerank(
                q, collection_name,
                condition=reverse_filter_cond,
                threshold=0.6,
                threshold_strategy="fixed",
            )
            all_results.extend(forward)
            all_results.extend(reverse)

        if not all_results:
            logger.info("No results retrieved.")
            return result

        # ── Step 3: Group, score, top-k selection ──
        title_score_map = {did: info.get("score", 0) for did, info in title_scores.items()}
        top_docs = 30
        top_paras = 3
        final_knowledge = self.group_and_score(
            all_results,
            top_n_docs=top_docs,
            top_k_paras=top_paras,
            doc_scores=title_score_map,
            alpha=0.5,
        )

        result["sources"] = final_knowledge

        # ── Step 4: LLM generation ──
        context_str = self.format_knowledge_prompt(final_knowledge)

        sys_prompt = system_prompt or """\
你是一个基于知识库的问答助手。请严格依据以下提供的信息回答用户问题。
如果信息不足以回答问题，请给出专业建议，不要编造。
引用信息时注明来源（标题/出处）。"""

        qa_prompt = (qa_prompt_template or """\
用户问题: {query}

参考信息:
============
{context}
============

请回答用户问题: {query}""").format(query=query, context=context_str)

        messages = [
            {"role": "system", "content": sys_prompt},
            *history,
            {"role": "user", "content": qa_prompt},
        ]

        answer = self.llm_chat(messages, stream=stream)
        result["answer"] = answer

        logger.info(f"=== RAG End === answer_len={len(answer)}, sources={len(final_knowledge)}")
        return result
