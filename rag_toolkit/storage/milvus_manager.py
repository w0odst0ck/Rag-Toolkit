"""
rag-toolkit / storage / milvus_manager.py

Milvus CRUD and hybrid search wrapper.
Works with PyMilvus >= 2.5 (BM25 full-text search + dense vector hybrid).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from pymilvus import (
    AnnSearchRequest,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)
from pymilvus.milvus_client.index import IndexParams

from rag_toolkit.core.config import Config

logger = logging.getLogger(__name__)


class MilvusManager:
    """High-level wrapper around ``pymilvus.MilvusClient``.

    Provides:
    - Database / collection lifecycle (create, describe, drop)
    - Hybrid search (dense vector + BM25 full-text)
    - Convenient inserts, upserts, deletes, queries.
    """

    def __init__(
        self,
        client: Optional[MilvusClient] = None,
        config: Optional[Config] = None,
    ):
        cfg = config or Config()
        if client:
            self.milvus = client
        else:
            self.milvus = MilvusClient(
                uri=cfg.MILVUS_URI,
                db_name=cfg.MILVUS_DB,
                token=cfg.MILVUS_TOKEN,
            )

    # ── Database ────────────────────────────────────────────────────

    def create_database(self, db_name: str) -> bool:
        try:
            self.milvus.create_database(db_name)
            logger.info(f"Database '{db_name}' created.")
            return True
        except Exception as e:
            logger.error(f"Failed to create database '{db_name}': {e}")
            return False

    def list_databases(self) -> list[str]:
        return self.milvus.list_databases()

    def using_database(self, db_name: str) -> MilvusManager:
        self.milvus.using_database(db_name)
        return self

    # ── Collection ──────────────────────────────────────────────────

    def create_collection(
        self,
        collection_name: str,
        schema: Optional[CollectionSchema] = None,
        dimension: int = 1024,
        auto_id: bool = True,
        **kwargs: Any,
    ) -> bool:
        """Create a collection with a standard hybrid-search schema (dense + sparse).

        If *schema* is provided use it as-is; otherwise build a default one.
        """
        if schema:
            try:
                self.milvus.create_collection(
                    collection_name=collection_name, schema=schema, **kwargs
                )
                return True
            except Exception as e:
                logger.error(f"Create collection failed: {e}")
                return False

        # ── Default schema ──
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=auto_id),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="sparse_text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True
            ),
        ]
        # BM25 function: tokenizer on `sparse_text` → `sparse_vector`
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["sparse_text"],
            output_field_names=["sparse_vector"],
        )
        schema = CollectionSchema(
            fields=fields,
            functions=[bm25_function],
            description="Default hybrid search collection (dense + BM25)",
        )
        try:
            self.milvus.create_collection(
                collection_name=collection_name, schema=schema, **kwargs
            )
            # Dense index (required for vector ANN search).
            ip_dense = IndexParams(
                field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self.milvus.create_index(
                collection_name=collection_name, index_params=ip_dense
            )
            # Sparse index (required for hybrid_search BM25 on the sparse_vector
            # output field of the built-in BM25 function).
            ip_sparse = IndexParams(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
            self.milvus.create_index(
                collection_name=collection_name, index_params=ip_sparse
            )
            logger.info(f"Collection '{collection_name}' created and indexed.")
            return True
        except Exception as e:
            logger.error(f"Create collection failed: {e}")
            return False

    def has_collection(self, collection_name: str) -> bool:
        return self.milvus.has_collection(collection_name)

    def drop_collection(self, collection_name: str) -> bool:
        try:
            self.milvus.drop_collection(collection_name)
            logger.info(f"Collection '{collection_name}' dropped.")
            return True
        except Exception as e:
            logger.error(f"Drop collection failed: {e}")
            return False

    def list_collections(self) -> list[str]:
        return self.milvus.list_collections()

    # ── CRUD ────────────────────────────────────────────────────────

    def insert(self, collection_name: str, data: list[dict]) -> list[int]:
        return self.milvus.insert(collection_name, data)

    def upsert(self, collection_name: str, data: list[dict]) -> list[int]:
        return self.milvus.upsert(collection_name, data)

    def delete(self, collection_name: str, ids: list[int]) -> None:
        self.milvus.delete(collection_name, ids)

    def query(self, collection_name: str, expr: str, output_fields: Optional[list[str]] = None):
        return self.milvus.query(collection_name, filter=expr, output_fields=output_fields)

    # ── Search ──────────────────────────────────────────────────────

    def search(
        self,
        collection_name: str,
        vectors: list[list[float]],
        anns_field: str = "vector",
        limit: int = 100,
        output_fields: Optional[list[str]] = None,
        expr: str = "",
        **kwargs: Any,
    ):
        # pymilvus >= 2.5 renames the search filter kwarg to `filter`;
        # `expr` is kept here for call-site compatibility and mapped through.
        return self.milvus.search(
            collection_name=collection_name,
            data=vectors,
            anns_field=anns_field,
            limit=limit,
            output_fields=output_fields,
            filter=expr,
            **kwargs,
        )

    def hybrid_search(
        self,
        collection_name: str,
        vec_data: list[list[float]],
        text_data: list[str],
        vec_anns_field: str = "vector",
        text_anns_field: str = "sparse_vector",
        vec_limit: int = 300,
        full_limit: int = 300,
        res_limit: int = 100,
        output_fields: Optional[list[str]] = None,
        rerank: Any = None,
        vec_expr: str = "",
        full_expr: str = "",
        vec_search_params: Optional[dict] = None,
        full_text_search_params: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> list[list[dict]]:
        """Hybrid search combining dense vector and BM25 full-text retrieval.

        Uses ``WeightedRanker`` by default to fuse results from both pipelines.

        ``limit`` is an alias for ``res_limit`` kept for call-site compatibility.
        """
        if limit is not None:
            res_limit = limit
        output_fields = output_fields or ["*"]
        ef = (vec_limit + full_limit) * 2

        vec_search_params = vec_search_params or {"params": {"ef": ef}}
        full_text_search_params = full_text_search_params or {
            "metric_type": "BM25",
            "params": {"drop_ratio_building": 0.2, "drop_ratio_search": 0.2},
        }

        vec_req = AnnSearchRequest(
            data=vec_data,
            anns_field=vec_anns_field,
            param=vec_search_params,
            limit=vec_limit,
            expr=vec_expr,
        )
        text_req = AnnSearchRequest(
            data=text_data,
            anns_field=text_anns_field,
            param=full_text_search_params,
            limit=full_limit,
            expr=full_expr,
        )

        rerank = rerank or WeightedRanker(0.4, 0.6)  # text_weight, vec_weight

        # pymilvus >= 2.5 names the fused-rank kwarg `ranker`;
        # `rerank` is kept here for call-site compatibility and mapped through.
        return self.milvus.hybrid_search(
            collection_name=collection_name,
            reqs=[vec_req, text_req],
            ranker=rerank,
            limit=res_limit,
            output_fields=output_fields,
        )
