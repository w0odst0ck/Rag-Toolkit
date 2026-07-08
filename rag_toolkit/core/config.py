"""
rag-toolkit / core / config.py

Centralized configuration pattern for RAG systems.
Usage:
    from rag_toolkit.core.config import Config

    cfg = Config()
    cfg.MILVUS_URI = "http://localhost:19530"
    cfg.MILVUS_DB = "default"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Single source of truth for all RAG component connections.

    Override values after init or subclass for domain-specific defaults.
    """

    # ── Vector Store ──────────────────────────────────────────────
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_DB: str = "default"
    MILVUS_TOKEN: Optional[str] = None

    # ── Embedding Model ───────────────────────────────────────────
    EMBEDDING_URL: str = "http://localhost:9997"
    EMBEDDING_MODEL: str = "bge-m3"

    # ── Reranker Model ────────────────────────────────────────────
    RERANKER_URL: str = "http://localhost:9997"
    RERANKER_MODEL: str = "bge-reranker-v2-m3"

    # ── LLM (OpenAI-compatible) ───────────────────────────────────
    LLM_URL: str = "http://localhost:8000/v1"
    LLM_API_KEY: str = "sk-xxx"
    LLM_MODEL: str = "qwen2.5-14b-instruct"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7

    # ── MySQL / Relational DB ─────────────────────────────────────
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "rag"
    MYSQL_CHARSET: str = "utf8mb4"

    # ── RAG Pipeline Defaults ─────────────────────────────────────
    TOP_K_ARTICLES: int = 5
    TOP_K_PARAGRAPHS: int = 10
    PARAGRAPH_LENGTH: int = 1500
    RERANK_THRESHOLD: float = 0.5
    HYBRID_VEC_WEIGHT: float = 0.6
    HYBRID_TEXT_WEIGHT: float = 0.4

    # ── Logging ───────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None  # None = stdout only

    def __post_init__(self) -> None:
        import logging
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            filename=self.LOG_FILE,
        )
