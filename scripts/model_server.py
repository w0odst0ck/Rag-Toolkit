"""
Rag-Toolkit / scripts / model_server.py

自建轻量 model server(替代 xinference),单文件 FastAPI 应用。

端点:
    GET  /healthz
        → {"status": "ok", "models": ["bge-m3", "bge-reranker-v2-m3"]}
    POST /embeddings
        body  {"texts": [...], "return_dense": true, "return_sparse": false}
        → [{"dense_vector": {"vector": [...]}}]  每条 text 一项
          (仅 return_dense=true 时含 dense_vector)
    POST /v1/rerank
        body  {"model": "...", "documents": [...], "query": "...", "top_n": null, ...}
        → {"results": [{"index": i, "relevance_score": s}]}  按分数降序

模型:
    bge-m3             → BGEM3FlagModel("/home/l/.cache/hf-bge-m3", use_fp16=False)
    bge-reranker-v2-m3 → FlagReranker("/home/l/.cache/hf-reranker", use_fp16=False)
    设备:torch.cuda.is_available() 时 device="cuda",否则 CPU。

加载策略:懒加载(首次请求才加载模型),threading.Lock 防并发重复加载,
加载失败时返回 503(状态不缓存,下次请求可重试)。
可选预热:环境变量 MODEL_PRELOAD=1 时启动即加载全部模型
(避免重启后首请求触发 30s+ 加载;预热失败仅告警,首次请求仍会重试)。

运行(在 Rag-Toolkit 目录):
    .venv/bin/python -m uvicorn scripts.model_server:app --port 9997
    # 带预热:
    MODEL_PRELOAD=1 .venv/bin/python -m uvicorn scripts.model_server:app --port 9997
    # 或直接执行本文件(入口 uvicorn.run 同样支持 lifespan):
    .venv/bin/python scripts/model_server.py
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from FlagEmbedding import BGEM3FlagModel
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── 模型注册表 ────────────────────────────────────────────────────────
# 本地模型路径,禁止触发 huggingface 下载。
MODEL_PATHS: dict[str, str] = {
    "bge-m3": "/home/l/.cache/hf-bge-m3",
    "bge-reranker-v2-m3": "/home/l/.cache/hf-reranker",
}
# 只有 reranker 类模型允许用于 /v1/rerank
RERANK_MODELS = {"bge-reranker-v2-m3"}
EMBED_MODELS = {"bge-m3"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 启动预热 ────────────────────────────────────────────────────────
# MODEL_PRELOAD=1 时启动即加载全部模型（避免重启后首请求 30s+ 加载超时）。
# 不用 --preload 参数：那是 uvicorn 自己的 worker 预加载语义，会冲突。
_PRELOAD_ENV = "MODEL_PRELOAD"


def preload_models() -> None:
    """启动预热：加载 MODEL_PATHS 全部模型。

    - 复用 ensure_model 的加载函数 `_load_model`，在 _load_lock 内执行，
      与懒加载路径互斥（预热期间到达的请求会阻塞在锁上）。
    - 单个模型失败只告警不阻断：预热失败不影响服务启动，
      首次请求仍会走 ensure_model 懒加载重试。
    """
    t0 = time.perf_counter()
    loaded = 0
    total = len(MODEL_PATHS)
    for name in MODEL_PATHS:
        with _load_lock:
            if name in _models:  # 已加载（如预热前被请求触发）
                loaded += 1
                continue
            try:
                _models[name] = _load_model(name)
                loaded += 1
                logger.info(f"Model '{name}' loaded (device={DEVICE}).")
            except Exception as e:  # noqa: BLE001 — 预热兜底：单模型失败仅告警，不阻断启动
                logger.warning(f"Model preload failed for '{name}' (will retry on first request): {e}")
    elapsed = time.perf_counter() - t0
    logger.info(f"Model preload finished: loaded {loaded}/{total} in {elapsed:.2f}s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动生命周期：MODEL_PRELOAD=1 时同步预热（启动期阻塞可接受，
    预热的意义就是让启动变慢而不是首请求变慢）。"""
    if os.environ.get(_PRELOAD_ENV) == "1":
        preload_models()
    yield


app = FastAPI(title="Rag-Toolkit Model Server", version="0.1.0", lifespan=lifespan)

class RerankerModel:
    """bge-reranker-v2-m3 via transformers native API.

    不用 FlagReranker.compute_score:新版 transformers 移除了
    ``XLMRobertaTokenizer.prepare_for_model``,FlagEmbedding 内部调用会
    AttributeError。这里直接用 AutoModelForSequenceClassification + tokenizer
    对,输入 [[query, doc], ...],输出 sigmoid 后的相关性分数。
    """

    def __init__(self, path: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        self.model.eval()
        self.device = device
        logger.info(f"RerankerModel loaded from {path} on {device}")

    def score_pairs(self, pairs: list[list[str]]) -> list[float]:
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
            return torch.sigmoid(logits).cpu().tolist()


# ── 懒加载状态 ────────────────────────────────────────────────────────
_load_lock = threading.Lock()
_models: dict[str, Any] = {}  # model_name -> 已加载的模型实例


def _load_model(name: str) -> Any:
    """按名称构造模型实例(仅在锁内调用)。"""
    path = MODEL_PATHS[name]
    # 显式校验本地目录:防止路径缺失/拼错时 FlagEmbedding 底层
    # 把路径当 model id 回退到 HuggingFace Hub 下载(禁止触发下载)。
    if not os.path.isdir(path):
        raise FileNotFoundError(f"local model directory not found: {path}")
    logger.info(f"Loading model '{name}' from {path} on {DEVICE} ...")
    if name in EMBED_MODELS:
        return BGEM3FlagModel(path, use_fp16=False, devices=[DEVICE])
    if name in RERANK_MODELS:
        return RerankerModel(path, DEVICE)
    raise ValueError(f"unknown model: {name}")


def ensure_model(name: str) -> Any:
    """懒加载入口:未加载则触发加载;加载失败抛 503(允许下次重试)。

    并发语义:threading.Lock 保证同一模型只加载一次;加载期间的其他
    请求阻塞在锁上,加载完成后直接复用实例。
    """
    model = _models.get(name)
    if model is not None:
        return model
    with _load_lock:
        model = _models.get(name)
        if model is not None:
            return model
        try:
            model = _load_model(name)
        except Exception as e:
            logger.exception(f"Failed to load model '{name}'")
            raise HTTPException(status_code=503, detail=f"model '{name}' not ready: {e}") from e
        _models[name] = model
        logger.info(f"Model '{name}' loaded (device={DEVICE}).")
        return model


# ── 请求/响应模型 ─────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    return_dense: bool = True
    return_sparse: bool = False
    normalize: bool = False
    # 兼容 xinference 请求里可能携带的其它字段(max_length 等),忽略即可


class RerankRequest(BaseModel):
    model: str = "bge-reranker-v2-m3"
    documents: list[str] = Field(..., min_length=1)
    query: str
    top_n: int | None = None
    # 兼容 xinference/cohere 风格的其它字段,忽略即可


# ── 端点 ──────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "models": sorted(MODEL_PATHS.keys())}


@app.post("/embeddings")
def embeddings(req: EmbeddingRequest) -> list[dict]:
    if not req.return_dense and not req.return_sparse:
        raise HTTPException(
            status_code=400, detail="at least one of return_dense / return_sparse must be true"
        )
    model = ensure_model("bge-m3")

    kwargs: dict[str, Any] = {"return_dense": req.return_dense}
    if req.return_sparse:
        kwargs["return_sparse"] = True
    if req.normalize:
        kwargs["normalize_embeddings"] = True

    try:
        out = model.encode(req.texts, **kwargs)
    except Exception as e:
        logger.exception("Embedding encode failed")
        raise HTTPException(status_code=500, detail=f"encode failed: {e}") from e

    results: list[dict] = []
    dense_vecs = out.get("dense_vecs")
    for i, _ in enumerate(req.texts):
        item: dict = {}
        if req.return_dense and dense_vecs is not None:
            item["dense_vector"] = {"vector": dense_vecs[i].tolist()}
        results.append(item)
    return results


@app.post("/v1/rerank")
def rerank(req: RerankRequest) -> dict:
    if req.model not in RERANK_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"model '{req.model}' is not a rerank model (available: {sorted(RERANK_MODELS)})",
        )
    model = ensure_model(req.model)

    pairs = [[req.query, doc] for doc in req.documents]
    try:
        scores = model.score_pairs(pairs)  # list[float],与 documents 一一对应
    except Exception as e:
        logger.exception("Rerank score_pairs failed")
        raise HTTPException(status_code=500, detail=f"rerank failed: {e}") from e

    ranked = sorted(
        ({"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)),
        key=lambda item: item["relevance_score"],
        reverse=True,
    )
    if req.top_n is not None and req.top_n > 0:
        ranked = ranked[: req.top_n]
    return {"results": ranked}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9997)
