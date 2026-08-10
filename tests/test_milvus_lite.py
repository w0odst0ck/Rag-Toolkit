"""Milvus Lite smoke tests — pin the pymilvus 2.5.x compatibility fixes.

Uses ``MilvusManager`` with an injected milvus-lite client backed by a temp
file db, so no external Milvus / Docker / network is required.

Covers the fixed behaviour:
- ``create_collection`` builds the default schema + BOTH indexes (dense + sparse)
- ``insert`` works
- dense ``search`` hits the right row
- ``hybrid_search`` (dense + BM25) hits the right row
- ``drop_collection`` cleans up
"""


import numpy as np
import pytest
from pymilvus import MilvusClient

from rag_toolkit.storage.milvus_manager import MilvusManager

DIM = 8
COLLECTION = "test_milvus_lite"

DOCS = [
    {"id": 1, "text": "Milvus is an open source vector database built for similarity search."},
    {"id": 2, "text": "Kubernetes is a container orchestration platform for cloud workloads."},
    {"id": 3, "text": "Python is a general purpose programming language used by data scientists."},
]


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    """MilvusManager backed by a fresh milvus-lite db in a temp dir."""
    db_path = str(tmp_path_factory.mktemp("milvus_lite") / "test_milvus_lite.db")
    client = MilvusClient(uri=db_path)
    return MilvusManager(client=client)


@pytest.fixture(scope="module")
def populated_collection(mgr):
    """Create the default-schema collection, insert docs, return its name."""
    rng = np.random.default_rng(42)
    vecs = [rng.normal(size=DIM).tolist() for _ in DOCS]
    rows = [
        {"id": d["id"], "vector": vecs[i], "text": d["text"], "sparse_text": d["text"]}
        for i, d in enumerate(DOCS)
    ]
    # Stash the vectors so dense-search tests can query near doc 1.
    mgr._test_vecs = vecs

    ok = mgr.create_collection(
        collection_name=COLLECTION, dimension=DIM, auto_id=False
    )
    assert ok, "create_collection (default schema + dual indexes) failed"

    ids = mgr.insert(COLLECTION, rows)
    assert len(ids) == len(DOCS)

    yield COLLECTION

    mgr.drop_collection(COLLECTION)


def test_create_collection_with_dual_indexes(mgr, populated_collection):
    """Default schema created and both dense + sparse indexes exist."""
    assert mgr.has_collection(populated_collection)
    # milvus-lite's list_indexes returns the indexed field names as strings.
    index_fields = set(mgr.milvus.list_indexes(populated_collection))
    assert "vector" in index_fields
    assert "sparse_vector" in index_fields


def test_dense_search_hits_right_row(mgr, populated_collection):
    """Dense ANN search returns the closest row first."""
    rng = np.random.default_rng(7)
    query_vec = (np.array(mgr._test_vecs[0]) + rng.normal(scale=0.01, size=DIM)).tolist()
    hits = mgr.search(
        populated_collection,
        vectors=[query_vec],
        limit=3,
        output_fields=["text"],
    )
    assert hits and hits[0]
    assert hits[0][0]["id"] == 1
    assert hits[0][0]["entity"]["text"] == DOCS[0]["text"]


def _milvus_lite_major():
    """Installed milvus-lite major version (3.x is not compatible with pymilvus 2.5.x gRPC)."""
    try:
        import importlib.metadata

        return int(importlib.metadata.version("milvus-lite").split(".")[0])
    except Exception:
        return 0


def test_hybrid_search_dense_plus_bm25(mgr, populated_collection):
    """Hybrid (dense + BM25) returns doc 1 for a matching text query."""
    if _milvus_lite_major() >= 3:
        pytest.xfail(
            "milvus-lite>=3 expects pymilvus 3.x gRPC proto "
            "(HybridSearchRequest.function_score), incompatible with pymilvus 2.5.x; "
            "downgrade to milvus-lite 2.5.x to exercise hybrid_search"
        )
    rng = np.random.default_rng(7)
    query_vec = (np.array(mgr._test_vecs[0]) + rng.normal(scale=0.01, size=DIM)).tolist()
    results = mgr.hybrid_search(
        populated_collection,
        vec_data=[query_vec],
        text_data=["vector database"],
        vec_limit=5,
        full_limit=5,
        res_limit=3,
        output_fields=["text"],
    )
    assert results and results[0]
    ids = [hit["id"] for hit in results[0]]
    assert 1 in ids, f"hybrid_search missed doc 1, got ids={ids}"


def test_drop_collection(mgr, populated_collection):
    """drop_collection removes the collection."""
    assert mgr.drop_collection(populated_collection)
    assert not mgr.has_collection(populated_collection)
