"""
16 · LangGraph + RAG：带「相关性判断」的检索工作流

对应原仓库：19-RAG检索增强生成.md、25-LangGraph高级特性.md（把 RAG 写成图）

普通 RAG 是直线：检索 -> 生成。问题是：检索不到相关内容时，模型仍可能硬编。
本文件把它升级成带分支的图：

    START -> retrieve -> grade（判断资料是否相关）
                            |
                    相关 ---+--- 不相关
                      |             |
                  generate       fallback（明确告知无资料）

这正是「图式工作流」的价值：把业务规则（这里的相关性判断）显式画在图上，
而不是藏在提示词里祈祷模型遵守。

运行：
    python 16_langgraph_rag.py
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import get_chat_model, get_embeddings

# ---------- 知识库（模型不知道的私有信息） ----------
DOCS = [
    Document(
        page_content=(
            "云栖科技报销制度：差旅报销需在出差结束后 7 个工作日内提交，"
            "附发票原件与行程单。单笔超过 5000 元需部门负责人和财务总监双重审批。"
            "超过 30 天未提交的报销申请不再受理。"
        ),
        metadata={"topic": "报销"},
    ),
    Document(
        page_content=(
            "云栖科技年假制度：入职满一年享 12 天带薪年假，满三年 15 天，满五年 20 天。"
            "年假需提前 3 个工作日在内部系统申请，由直属主管审批。"
        ),
        metadata={"topic": "年假"},
    ),
    Document(
        page_content=(
            "云栖科技远程办公制度：每周三、周五可申请远程办公，每月最多 8 天。"
            "远程当天需保持在线并在日报中注明工作地点，试用期员工不适用。"
        ),
        metadata={"topic": "远程办公"},
    ),
]

# 索引阶段：一次性把文档灌进内存向量库
VECTOR_STORE = InMemoryVectorStore.from_documents(DOCS, get_embeddings())
RETRIEVER = VECTOR_STORE.as_retriever(search_kwargs={"k": 2})

model = get_chat_model(temperature=0)


class RagState(TypedDict):
    question: str
    context: str
    verdict: str  # 相关 / 不相关
    answer: str


# ---------- 节点 1：检索 ----------
def retrieve(state: RagState) -> dict:
    docs = RETRIEVER.invoke(state["question"])
    context = "\n\n".join(d.page_content for d in docs)
    print(f"  [retrieve] 命中 {len(docs)} 个片段")
    return {"context": context}


# ---------- 节点 2：相关性判断 ----------
# 注意：这里的判断由模型完成，但它的输出只用于「路由」，不直接给用户看。
def grade(state: RagState) -> dict:
    prompt = (
        "判断下面的资料能否回答用户问题。只输出一个词：相关 或 不相关。\n\n"
        f"资料：\n{state['context']}\n\n问题：{state['question']}"
    )
    raw = model.invoke(prompt).content.strip()
    verdict = "相关" if "相关" in raw and "不相关" not in raw else "不相关"
    print(f"  [grade] 模型输出={raw!r} -> {verdict}")
    return {"verdict": verdict}


def route_after_grade(state: RagState) -> str:
    return "generate" if state["verdict"] == "相关" else "fallback"


# ---------- 节点 3/4：生成 / 兜底 ----------
def generate(state: RagState) -> dict:
    prompt = (
        "只依据下面的资料回答，不要编造。资料未提及的内容直接说“资料中未提及”。\n\n"
        f"资料：\n{state['context']}\n\n问题：{state['question']}"
    )
    return {"answer": model.invoke(prompt).content}


def fallback(state: RagState) -> dict:
    return {"answer": "抱歉，知识库中没有找到与这个问题相关的资料，建议咨询 HR 或财务。"}


def build_app():
    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("generate", generate)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "fallback": "fallback"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)
    return builder.compile()


def main() -> None:
    app = build_app()
    print("【图结构】")
    app.get_graph().print_ascii()

    questions = [
        "出差报销需要在几天内提交？超过 5000 元要谁审批？",
        "公司食堂中午几点开饭？",  # 知识库里没有 -> 应走 fallback
    ]
    for q in questions:
        print(f"\n用户：{q}")
        result = app.invoke({"question": q, "context": "", "verdict": "", "answer": ""})
        print("判断：", result["verdict"])
        print("回答：", result["answer"])


if __name__ == "__main__":
    main()
