# RAG Toolkit

可复用 RAG 系统构建工具包。从实战项目中沉淀的通用组件。

## 目录结构

```
rag-toolkit/
├── pipelines/              # RAG 流水线组件
│   ├── base_rag.py         # 通用 RAG 基类（召回→重排→分组→LLM 全链路）
│   ├── reranker.py         # 重排器（XReranker / TwoStageReranker / knowsrerank）
│   ├── query_expander.py   # 多 Query 生成（LLM + 模板）
│   └── context_expander.py # 上下文压缩 / 扩展
├── storage/                # 存储层
│   ├── milvus_manager.py   # Milvus CRUD + Hybrid Search
│   └── mysql_manager.py    # MySQL 连接 + 查询
├── core/                   # 基础设施
│   ├── config.py           # 集中配置
│   ├── utils.py            # 工具函数（超时控制、文本处理）
│   └── prompts.py          # Prompt 模板
├── deploy/
│   └── docker-compose.yml  # Milvus Standalone 部署
├── examples/
│   └── data_migration.py   # MySQL→Milvus 数据迁移示例
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt
```

### 2. 部署向量库

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### 3. 使用 RAG Pipeline

```python
from rag_toolkit.core.config import Config
from rag_toolkit.storage.milvus_manager import MilvusManager
from rag_toolkit.pipelines.base_rag import BaseRAGPipeline

# 配置
cfg = Config()
cfg.MILVUS_URI = "http://localhost:19530"
cfg.LLM_URL = "http://localhost:8000/v1"
cfg.LLM_MODEL = "qwen2.5-14b-instruct"
cfg.LLM_API_KEY = "your-api-key"

# 创建 pipeline
pipeline = BaseRAGPipeline(config=cfg)

# 查询
result = pipeline.rag(
    query="什么是商业银行的资本充足率要求？",
    collection_name="my_knowledge_base",
)
print(result["answer"])
```

### 4. 子类化实现业务逻辑

```python
class MyRAGPipeline(BaseRAGPipeline):
    def fetch_document_metadata(self, doc_ids):
        # 从数据库或 API 获取文档信息
        ...

    def fetch_paragraph_content(self, doc_ids):
        # 获取段落级内容
        ...

    def format_knowledge_prompt(self, knowledge_list):
        # 自定义知识块格式化
        ...
```

## 架构概览

```
用户 query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: 标题全文检索 + Query 改写                    │
│    ├─→ 向量搜索标题 → 缩小候选范围 (titleIds)         │
│    ├─→ LLM 生成多轮查询 → 扩展召回                    │
│    └─→ 构建前向/反向过滤条件                          │
├─────────────────────────────────────────────────────┤
│  Step 2: 多路召回                                    │
│    ├─→ 正向召回（topk 标题内，threshold 0.4）         │
│    ├─→ 反向召回（topk 标题外，threshold 0.6）         │
│    └─→ 多 query 并行                                 │
├─────────────────────────────────────────────────────┤
│  Step 3: 重排 + 分组                                 │
│    ├─→ Reranker 打分 + 动态阈值过滤                   │
│    ├─→ 按 doc_id 分组 → 文章级评分                    │
│    └─→ 取 top_n 文章 × top_k 段落                    │
├─────────────────────────────────────────────────────┤
│  Step 4: LLM 生成                                    │
│    └─→ 组装 prompt → 调用 LLM → 返回答案 + 来源      │
└─────────────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 说明 |
|------|------|
| Hybrid Search | 稠密向量 + BM25 全文搜索 + WeightedRanker 融合 |
| Multi-query | LLM 生成多版本查询，提高召回覆盖率 |
| Two-stage Rerank | 先文章级、后段落级级联重排 |
| 动态阈值 | 自适应分数阈值（mean + std），避免全丢或全收 |
| 上下文压缩 | BM25 + 窗口去重，减少无用上下文 |
| 段落组扩展 | 命中段 → 扩展至相邻段落组 |
| 超时控制 | multiprocess 终止超时函数 |
| 集中配置 | 单点修改所有连接参数 |
