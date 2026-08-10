"""下载 BAAI/bge-m3 模型到本地缓存（Step 1.4）"""
import sys
sys.path.insert(0, ".")

from FlagEmbedding import BGEM3FlagModel

print("Downloading BAAI/bge-m3 ...")
model = BGEM3FlagModel("/home/l/.cache/hf-bge-m3", use_fp16=False)
print("Model loaded OK")
# 触发一次前向，验证推理可用
out = model.encode(["hello mojin", "沙特独立站客服"])
print("Embed dim:", out["dense_vecs"].shape)
print("DONE")
