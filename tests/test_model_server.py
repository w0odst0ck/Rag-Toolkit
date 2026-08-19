"""model_server 端点协议测试（全部假模型注入，禁止真实加载 / HF 下载）。

- 假模型注入 `_models`，ensure_model 命中即返回，绝不触发 `_load_model`。
- 回归用例 11–13 机器化断言现有 /embeddings、/v1/rerank、/healthz 语义不变。
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scripts import model_server

DIM = 4  # 假模型维度；协议层测试不需要 1024，1024 由实机验证保证


class FakeEmbed:
    def encode(self, texts, **kwargs):
        n = len(texts)
        return {"dense_vecs": np.arange(n * DIM, dtype=float).reshape(n, DIM)}


class FakeRerank:
    def score_pairs(self, pairs):
        return [float(i + 1) / float(len(pairs)) for i, _ in enumerate(pairs)]


@pytest.fixture
def client(monkeypatch):
    # 防预热兜底：即使开发机 shell 误设 MODEL_PRELOAD=1 也不会触发真实加载。
    monkeypatch.delenv("MODEL_PRELOAD", raising=False)
    monkeypatch.setattr(
        model_server,
        "_models",
        {
            "bge-m3": FakeEmbed(),
            "bge-reranker-v2-m3": FakeRerank(),
        },
    )

    # 红线兜底：任何测试若意外走到 _load_model（真实加载 / HF 下载），立刻失败。
    # 需要走加载路径的用例（如 load_error_503）在测试内自行覆盖该 spy。
    def _guard_load(name):
        raise AssertionError(
            f"_load_model called with '{name}' — 测试必须用假模型注入，禁止真实加载"
        )

    monkeypatch.setattr(model_server, "_load_model", _guard_load)
    with TestClient(model_server.app) as c:
        yield c


# ── /v1/embeddings 主路径 ─────────────────────────────────────────────


def test_v1_embeddings_ok_multiple_inputs(client):
    resp = client.post(
        "/v1/embeddings", json={"model": "bge-m3", "input": ["a", "b b", "c c c"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 3
    for i, item in enumerate(body["data"]):
        assert item["object"] == "embedding"
        assert item["index"] == i  # 顺序与 input 严格一一对应（钉死）
        assert len(item["embedding"]) == DIM
        assert item["embedding"] == [float(i * DIM + j) for j in range(DIM)]
    assert body["model"] == "bge-m3"
    # "a"(1) + "b b"(2) + "c c c"(3) → 确定性 6（钉死）
    assert body["usage"] == {"prompt_tokens": 6, "total_tokens": 6}


def test_v1_embeddings_single_string_input(client):
    resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1  # str 归一化为单元素数组
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == DIM
    assert body["usage"] == {"prompt_tokens": 1, "total_tokens": 1}


# ── /v1/embeddings 校验失败路径 ────────────────────────────────────────


def test_v1_embeddings_unknown_model_404(client, monkeypatch):
    def spy_load(name):
        raise AssertionError("_load_model must not be called for unknown model")

    monkeypatch.setattr(model_server, "_load_model", spy_load)
    resp = client.post("/v1/embeddings", json={"model": "gpt-4", "input": ["x"]})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "not an embedding model" in detail
    assert "bge-m3" in detail  # available 列表含可用模型


def test_v1_embeddings_empty_input_422(client):
    resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": []})
    assert resp.status_code == 422


def test_v1_embeddings_empty_string_422(client):
    for bad in ["", "   "]:
        resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": bad})
        assert resp.status_code == 422


def test_v1_embeddings_missing_input_422(client):
    resp = client.post("/v1/embeddings", json={"model": "bge-m3"})
    assert resp.status_code == 422


def test_v1_embeddings_non_string_item_422(client):
    for bad in [[1, 2], [{"x": 1}]]:
        resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": bad})
        assert resp.status_code == 422


# ── /v1/embeddings 运行时失败路径 ──────────────────────────────────────


def test_v1_embeddings_load_error_503(client, monkeypatch):
    monkeypatch.setattr(model_server, "_models", {})  # 清空，强制走懒加载路径

    def boom(name):
        raise FileNotFoundError(f"local model directory not found: {name}")

    monkeypatch.setattr(model_server, "_load_model", boom)
    resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": ["x"]})
    assert resp.status_code == 503
    assert model_server._models.get("bge-m3") is None  # 失败状态不缓存


def test_v1_embeddings_encode_error_500(client, monkeypatch):
    embed = model_server._models["bge-m3"]

    def boom_encode(self, texts, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(embed, "encode", boom_encode)
    resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": ["x"]})
    assert resp.status_code == 500
    assert "encode failed" in resp.json()["detail"]


def test_v1_embeddings_bad_dense_output_500(client, monkeypatch):
    embed = model_server._models["bge-m3"]

    monkeypatch.setattr(embed, "encode", lambda texts, **kw: {})
    resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": ["a", "b"]})
    assert resp.status_code == 500
    assert "unexpected output shape" in resp.json()["detail"]

    monkeypatch.setattr(
        embed, "encode", lambda texts, **kw: {"dense_vecs": np.zeros((1, DIM))}
    )
    resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": ["a", "b"]})
    assert resp.status_code == 500


# ── 回归：现有端点语义不变 ─────────────────────────────────────────────


def test_v1_rerank_regression(client):
    resp = client.post(
        "/v1/rerank",
        json={"model": "bge-reranker-v2-m3", "query": "q", "documents": ["a", "b", "c"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    results = body["results"]
    assert len(results) == 3
    assert all({"index", "relevance_score"} == set(r.keys()) for r in results)
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)  # 降序
    assert {r["index"] for r in results} == {0, 1, 2}


def test_embeddings_legacy_regression(client):
    resp = client.post("/embeddings", json={"texts": ["a", "b"], "return_dense": True})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert all(list(item.keys()) == ["dense_vector"] for item in body)
    assert all(len(item["dense_vector"]["vector"]) == DIM for item in body)


def test_healthz_regression(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    models = body["models"]
    assert "bge-m3" in models and "bge-reranker-v2-m3" in models


# ── /v1/models ─────────────────────────────────────────────────────────


def test_v1_models_ok(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [item["id"] for item in body["data"]]
    assert "bge-m3" in ids and "bge-reranker-v2-m3" in ids
    # data 顺序与 /healthz models 一致（两者同为 sorted(MODEL_PATHS)）
    healthz_models = client.get("/healthz").json()["models"]
    assert ids == healthz_models


def test_v1_models_openai_shape(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    for item in resp.json()["data"]:
        assert set(item.keys()) == {"id", "object", "owned_by"}
        assert item["object"] == "model"
        assert item["owned_by"] == "rag-toolkit"


def test_v1_models_empty_registry(client, monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_PATHS", {})
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json() == {"object": "list", "data": []}


def test_v1_models_no_model_load(client, monkeypatch):
    def spy_load(name):
        raise AssertionError(f"_load_model must not be called by /v1/models (got '{name}')")

    monkeypatch.setattr(model_server, "_load_model", spy_load)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["object"] == "list"


def test_v1_models_regression_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    # models 不变：内容与排序均钉死（与 sorted(MODEL_PATHS) 同源）
    assert resp.json()["models"] == ["bge-m3", "bge-reranker-v2-m3"]
