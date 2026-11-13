"""
09 · 最小 RAG（检索增强生成）

对应原仓库：10-rag/*、19-RAG检索增强生成.md

RAG 解决的是「模型不知道你公司的私有知识」这个问题。
两个阶段：
    索引阶段：文档 -> 切块 -> 向量化 -> 存进向量库
    查询阶段：问题 -> 向量 -> 检索最相似的 k 个块 -> 拼进提示词 -> 模型回答

本文件为了零外部依赖，向量库用 InMemoryVectorStore（进程内存），
    真实项目换成 Redis / Qdrant / Milvus / pgvector，接口几乎一样。

运行：
    python 09_rag_minimal.py
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_chat_model, get_embeddings

# ---------- 1. 一份模型不可能知道的"私有知识" ----------
RAW_TEXT = """
云栖科技员工手册（2026 版）

一、年假制度：入职满一年的员工每年享有 12 天带薪年假；满三年增加到 15 天；满五年增加到 20 天。
年假需提前 3 个工作日在内部系统提交申请，由直属主管审批。

二、远程办公：每周三、周五可以申请远程办公，每月最多 8 天。远程办公当天需保持在线，
并在日报中注明工作地点。试用期员工不适用远程办公政策。

三、报销流程：差旅报销需在出差结束后 7 个工作日内提交，附发票原件与行程单。
单笔金额超过 5000 元需部门负责人和财务总监双重审批。超过 30 天未提交的报销申请不再受理。

四、设备领用：新员工入职当天可领取笔记本电脑一台、显示器一台。设备损坏需提交维修申请，
人为损坏按设备原价的 30% 赔偿。

五、保密要求：员工不得将公司内部文档上传至任何公网 AI 工具。违反者按信息安全管理制度处理。
"""


def build_retriever():
    """索引阶段：切块 -> 向量化 -> 入库 -> 返回检索器。"""
    # 先包成 Document（RAG 里的标准数据结构：page_content + metadata）
    doc = Document(page_content=RAW_TEXT, metadata={"source": "员工手册"})

    # 切块：chunk_size 是一块的目标字符数；chunk_overlap 让相邻块有重叠，避免语义被切断
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", ""],
    )
    chunks = splitter.split_documents([doc])
    print(f"切块数量：{len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  块{i + 1}: {chunk.page_content[:40]}...")

    # from_documents 内部会调用 embedding 模型，把每块变成向量后写入内存向量库
    vector_store = InMemoryVectorStore.from_documents(chunks, get_embeddings())

    # as_retriever：把向量库包装成 Runnable，输入查询字符串，输出最相似的 k 个 Document
    return vector_store.as_retriever(search_kwargs={"k": 2})


def main() -> None:
    retriever = build_retriever()
    print("=" * 60)

    question = "我入职满三年，每年有多少天年假？远程办公每月最多几天？"

    # ---------- 2. 先单独看一下检索结果（调试 RAG 的关键一步） ----------
    print("【检索到的片段】")
    for i, doc in enumerate(retriever.invoke(question), 1):
        print(f"--- 片段{i} ---")
        print(doc.page_content)
    print("=" * 60)

    # ---------- 3. 组装 RAG 链 ----------
    # {context} 来自检索器；{question} 用 RunnablePassthrough 原样透传
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你只能依据下面的资料回答问题。资料中没有的内容，直接回答“资料中未提及”。\n\n资料：\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | get_chat_model(temperature=0)
        | StrOutputParser()
    )
    answer = rag_chain.invoke(question)
    print("【有 RAG】")
    print("问题：", question)
    print("回答：", answer)
    print("=" * 60)

    # ---------- 4. 对照组：不检索，只靠模型自身知识 ----------
    no_rag_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "你只能依据下面的资料回答问题。资料中没有的内容，直接回答“资料中未提及”。\n\n资料：\n（无）"),
                ("human", "{question}"),
            ]
        )
        | get_chat_model(temperature=0)
        | StrOutputParser()
    )
    print("【无 RAG 对照】")
    print("回答：", no_rag_chain.invoke({"question": question}))
    print("=" * 60)
    print("结论：同一问题，没有检索时模型只能编或拒答；RAG 让回答有据可依。")


if __name__ == "__main__":
    main()
