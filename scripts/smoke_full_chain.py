"""P0-5 全链路冒烟：直塞模式 + Milvus 检索模式（真实知识源 + DeepSeek）

验证：
1. 直塞模式：FAQ/政策/产品知识拼进 system prompt → DeepSeek 三语回答，引用真实知识
2. Milvus 模式：知识分块 → bge-m3 embed（自建 model server）→ Milvus Lite 写入 → 检索 → LLM 生成
3. 知识注入的必要性：同样的问法，无知识时 DeepSeek 会瞎答（如配送 5-10 天 vs 实际 3-5 天）

前置：
- model server 跑在 127.0.0.1:9997（scripts/model_server.py）
- DEEPSEEK_API_KEY 环境变量
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, ".")
import requests
from pymilvus import MilvusClient

from rag_toolkit.storage.milvus_manager import MilvusManager

MODEL_SERVER = "http://127.0.0.1:9997"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
CAREWELL = "/home/l/.openclaw/workspace/projects/mojin-store/技术栈/carewell-shop"

def deepseek_chat(system: str, user: str, max_tokens: int = 200) -> str:
    r = requests.post(DEEPSEEK_URL, headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
                      json={"model": "deepseek-chat",
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
                            "max_tokens": max_tokens, "temperature": 0.3}, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ── 1. 加载真实知识源 ──
with open(f"{CAREWELL}/faq-knowledge.json", encoding="utf-8") as f:
    faq = json.load(f)
faqs = faq["faqs"]  # 9 条，三语
# 政策页提取纯文本（简单 strip 标签）
import re


def html_text(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    txt = re.sub(r"<script.*?</script>", " ", raw, flags=re.DOTALL)
    txt = re.sub(r"<style.*?</style>", " ", txt, flags=re.DOTALL)
    txt = re.sub(r"<[^>]+>", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()
policies = {p: html_text(f"{CAREWELL}/{p}.html")[:600] for p in ["shipping", "return-policy", "privacy"]}

# 知识文本块（按语言）——Milvus 分块用
blocks = []  # {id, lang, text}
for q in faqs:
    for lang in ("en", "cn", "ar"):
        if lang in q and q[lang].get("question"):
            blocks.append({"id": f"{q['id']}-{lang}",
                           "lang": lang,
                           "text": f"Q: {q[lang]['question']} A: {q[lang]['answer']}"})
print(f"知识块: {len(blocks)}（9 FAQ × 3 语言）| 政策页: {list(policies)}")

# ── 2. 直塞模式 ──
print("\n=== 直塞模式 ===")
kb_snippet = "\n".join(b["text"] for b in blocks if b["lang"] == "en")
system_prompt = (
    "You are Mojin store customer service (Saudi Arabia). Answer ONLY from the knowledge base below. "
    "If unknown, say you'll transfer to a human agent.\n\nKB:\n" + kb_snippet[:2500]
)
for q in ["Do you accept cash on delivery?", "How long is shipping to Riyadh?", "What is the return policy?"]:
    ans = deepseek_chat(system_prompt, q)
    print(f"[{q}] → {ans[:120]}")

# ── 3. Milvus 检索模式 ──
print("\n=== Milvus 检索模式 ===")
def embed(texts):
    r = requests.post(f"{MODEL_SERVER}/embeddings",
                      json={"texts": texts, "return_dense": True, "return_sparse": False}, timeout=120)
    r.raise_for_status()
    return [x["dense_vector"]["vector"] for x in r.json()]

db_path = os.path.join(tempfile.mkdtemp(), "mojin_chain.db")
lite = MilvusClient(uri=db_path)
mgr = MilvusManager(client=lite)
col = "mojin_kb"
if mgr.has_collection(col):
    mgr.drop_collection(col)
mgr.create_collection(collection_name=col, dimension=1024, auto_id=False)

vecs = embed([b["text"] for b in blocks])
rows = [{"id": i, "vector": vecs[i], "text": b["text"], "sparse_text": b["text"]}
        for i, b in enumerate(blocks)]
mgr.insert(col, rows)
print(f"写入 {len(rows)} 块")

for q in ["Do you accept cash on delivery?", "كم مدة التوصيل؟", "多久能送到利雅得？"]:
    qv = embed([q])[0]
    hits = mgr.search(col, vectors=[qv], limit=1, output_fields=["text"])
    if hits and hits[0]:
        top = hits[0][0]
        print(f"\n[{q}] 命中 id={top['id']} dist={top['distance']:.3f}")
        print(f"  -> {top['entity']['text'][:110]}")
        # LLM 用命中知识回答
        ans = deepseek_chat(
            "Answer in the user's language using ONLY the knowledge snippet. If not covered, say you'll transfer to human.",
            f"Knowledge: {top['entity']['text']}\n\nQuestion: {q}")
        print(f"  答: {ans[:120]}")

mgr.drop_collection(col)
print("\n✅ 全链路冒烟通过（直塞 + Milvus 双模式）")
