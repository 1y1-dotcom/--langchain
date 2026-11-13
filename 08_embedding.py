"""
08 · Embedding 与语义相似度

对应原仓库：09-embedding/Text2Embedding_DashScope.py、Text2Embedding_CosSimilarity.py

Embedding 做的事：把一段文本压成一个固定长度的浮点向量。
    语义越接近的文本，向量在空间里的方向越接近。
    判断「方向接近」用余弦相似度：cos(a,b) = a·b / (|a||b|)，取值 -1~1，越接近 1 越相似。

这是 RAG 的地基：
    文档 -> 向量入库；问题 -> 向量；找最相似的文档块 -> 塞进提示词 -> 模型回答。

运行：
    python 08_embedding.py
"""

import math

from config import get_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """手写余弦相似度，便于理解向量检索到底在算什么。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)  # 加一个极小值避免除以 0


def main() -> None:
    embeddings = get_embeddings()

    sentences = [
        "如何用 Python 读取一个 CSV 文件",
        "Python 中 pandas 读取表格数据的方法",
        "今天中午吃什么比较好",
        "红烧肉的家常做法",
    ]

    # embed_documents：批量把文本转成向量（返回 list[list[float]]）
    vectors = embeddings.embed_documents(sentences)
    print(f"向量维度：{len(vectors[0])}，共 {len(vectors)} 条")
    print("第一条向量前 5 维：", [round(v, 4) for v in vectors[0][:5]])
    print("=" * 60)

    # ---------- 1. 两两相似度矩阵 ----------
    # 预期：句 1 和句 2 相似度高；句 3 和句 4 相似度高；跨组很低
    print("【相似度矩阵】")
    print(" " * 12, " ".join(f"句{i + 1:>6}" for i in range(len(sentences))))
    for i, vec_i in enumerate(vectors):
        row = [f"{cosine_similarity(vec_i, vec_j):.4f}" for vec_j in vectors]
        print(f"句{i + 1:<8}", " ".join(f"{v:>7}" for v in row))
    print("=" * 60)

    # ---------- 2. 语义检索：给一个问题，找最相似的句子 ----------
    query = "怎么用 pandas 导入 csv"
    # embed_query：把查询也变成同一空间的向量
    query_vec = embeddings.embed_query(query)

    scored = sorted(
        ((cosine_similarity(query_vec, vec), text) for vec, text in zip(vectors, sentences)),
        reverse=True,
    )
    print(f"查询：{query}")
    for score, text in scored:
        print(f"  {score:.4f}  {text}")
    print("=" * 60)

    # 结论：得分最高的是「pandas 读取表格」，说明检索靠的是语义而不是字面关键词——
    # query 里没有出现 "pandas 读取表格数据的方法" 中的全部字，但意思接近。


if __name__ == "__main__":
    main()
