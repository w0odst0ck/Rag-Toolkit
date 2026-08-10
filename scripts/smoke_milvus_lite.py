"""Step 1.5 冒烟：bge-m3 embed → Milvus Lite 写入 → 检索 闭环

关键验证点：
1. Rag-Toolkit 的 MilvusManager 直接注入 Milvus Lite client（业务代码零改动）
2. bge-m3 多语言 embedding（中/英/阿）
3. 默认 schema（含 BM25 函数）+ dense 检索命中
"""
import sys, os, tempfile
sys.path.insert(0, ".")

from pymilvus import MilvusClient
from rag_toolkit.storage.milvus_manager import MilvusManager

# ── 1. bge-m3 embedding ──
from FlagEmbedding import BGEM3FlagModel
print("加载 bge-m3 ...")
model = BGEM3FlagModel("/home/l/.cache/hf-bge-m3", use_fp16=False)

docs = [
    {"id": 1, "text": "We support Cash on Delivery across Saudi Arabia, no prepayment needed.", "lang": "en"},
    {"id": 2, "text": "订单发货后一般 3-5 天送达沙特主要城市，利雅得 2 天。", "lang": "zh"},
    {"id": 3, "text": "نحن ندعم الدفع عند الاستلام في جميع أنحاء المملكة العربية السعودية.", "lang": "ar"},
    {"id": 4, "text": "Our caps are 100% cotton, machine washable, one size fits most.", "lang": "en"},
]
texts = [d["text"] for d in docs]
emb = model.encode(texts)["dense_vecs"]

# ── 2. Milvus Lite + Rag-Toolkit MilvusManager（注入 client，零改动）──
db_path = os.path.join(tempfile.mkdtemp(), "mojin_smoke.db")
lite = MilvusClient(uri=db_path)
mgr = MilvusManager(client=lite)
print("MilvusManager 初始化 OK | 集合:", mgr.list_collections())

collection = "mojin_faq_smoke"
if mgr.has_collection(collection):
    mgr.drop_collection(collection)
ok = mgr.create_collection(collection_name=collection, dimension=emb.shape[1], auto_id=False)
print(f"集合创建 {'OK' if ok else 'FAIL'} | dim={emb.shape[1]}")

rows = [
    {"id": d["id"], "vector": emb[i].tolist(), "text": d["text"], "sparse_text": d["text"]}
    for i, d in enumerate(docs)
]
ids = mgr.insert(collection, rows)
print("插入 OK | ids:", ids)

# ── 3. 检索验证（dense + hybrid 双通道）──
queries = {
    "COD 支付 (en)": "Do you accept cash on delivery?",
    "发货时效 (zh)": "多久能送到利雅得？",
    "COD 阿语 (ar)": "هل يمكنني الدفع عند الاستلام؟",
}
for label, q in queries.items():
    qv = model.encode([q])["dense_vecs"][0].tolist()
    hits = mgr.search(collection, vectors=[qv], limit=1, output_fields=["text"])
    top = hits[0][0] if hits and hits[0] else None
    if top:
        ent = top.get("entity", {})
        print(f"\n[{label}] 命中 id={top.get('id')} dist={top.get('distance'):.4f}")
        print(f"  -> {str(ent.get('text'))[:70]}")
    else:
        print(f"\n[{label}] ❌ 无命中")

    # hybrid（dense + BM25）
    try:
        hh = mgr.hybrid_search(collection, vec_data=[qv], text_data=[q], limit=1, output_fields=["text"])
        if hh and hh[0]:
            print(f"  hybrid 命中 id={hh[0][0].get('id')} score={hh[0][0].get('distance'):.4f}")
    except Exception as e:
        print(f"  hybrid ❌: {type(e).__name__}: {str(e)[:100]}")

mgr.drop_collection(collection)
print("\n冒烟测试结束 ✅")
