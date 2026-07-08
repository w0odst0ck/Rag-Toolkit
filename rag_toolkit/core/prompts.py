"""
rag-toolkit / core / prompts.py

Reusable prompt templates for RAG pipelines.

All templates expect ``str.format()`` keyword arguments.
"""

# ── Relevance Scoring (reranker) ──────────────────────────────────────
RELEVANCE_PROMPT = """\
任务：评估用户query与内容的相关性。

评分标准：
5 - 直接回答：内容直接并较完整地回答了用户的一个或多个问题
4 - 部分回答：内容部分回答了用户的问题，但需结合其他信息才能完整回答
3 - 相关信息：内容与问题密切相关，但不直接回答问题
2 - 辅助信息：内容与问题有一定关联，只能提供背景或辅助信息
1 - 微弱相关：内容与问题几乎无关
0 - 无关

内容来源："{node_source}"
内容："""{node_text}"""
用户问题："""{query}"""

请严格按以下 JSON 格式回复：
{{"评分依据": "<理由>", "相关性评分": <0|1|2|3|4|5>}}
"""

# ── Multi‑Query Generation ───────────────────────────────────────────
MULTI_QUERY_GENERATE = """\
任务：根据用户输入的query，生成 3-5 个语义相似的问法，用于向量检索的 query 扩展。

要求：
1. 保持原问题的核心意图
2. 可以换同义词、调整语序、从不同角度表达
3. 输出严格为 JSON 格式

用户问题: {query}

输出格式：
{{"生成问题": ["问题1", "问题2", "问题3"]}}
"""

MULTI_QUERY_PROMPT_EN = """\
You are an AI language model assistant. Your task is to generate five different versions
of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help overcome
some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}
"""

# ── QA / Generation ──────────────────────────────────────────────────
QA_PROMPT = """\
任务：基于知识库中查询到的信息，回答用户问题。

规则：
1. 仅使用下文提供的信息来回答，不要使用你掌握的其他知识
2. 如果答案涉及多个知识点，尽量列表详细回答
3. 答案需前后逻辑一致
4. 如果无法回答，给出专业建议，而非编造信息
5. 引用信息时标注信息来源

用户问题: {query}

查询得到的相关信息:
============
{context}
============

请回答用户问题：{query}
"""

STRUCTURED_QA_PROMPT = """\
任务：基于从知识库中查询到的信息，结构化回答用户问题。

回答格式要求：
1. 直接回答用户问题
2. 引用相关来源（标题 / 出处）
3. 如有必要，提供解读或解释
4. 请仅使用下文提供的信息，不要编造

用户问题: {query}

参考信息:
{context}

请给出结构化回答：
"""

# ── Query Rewriting (for Chinese domain RAG) ─────────────────────────
QUERY_REWRITE_PROMPT = """\
# 角色设定
- 你是一位专业的信息检索专员，擅长将用户的问题转化为检索系统易于理解的格式。

# 技能
- 分析用户问题的核心内容，确定检索关键词
- 将复杂问题拆分成多个子问题
- 对每个子问题进行通用化扩写和引导式扩写

# 输入
用户问题：{question}

# 输出格式（严格JSON）
{{"query": ["改写后的问题1", "改写后的问题2", ...]}}
"""
