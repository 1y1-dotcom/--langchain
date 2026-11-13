"""
13 · 条件边（Conditional Edge）——图里的 if/else

对应原仓库：05-edge/Edge_Conditional.py、Edge_ConditionalEntryPoint.py

普通边是「固定路线」，条件边是「路由器」：
    add_conditional_edges(源节点, 路由函数, {返回值: 目标节点})

路由函数只负责「看状态、返回一个标签」，不要在节点内部写一堆 if 去决定下一个节点。
    这样图结构一眼能看懂：节点做业务，边做分流。

本文件做一个「客服问题分类 -> 不同专家回答」的小图。

运行：
    python 13_langgraph_conditional.py
"""

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import get_chat_model

model = get_chat_model(temperature=0)


class RouteState(TypedDict):
    question: str
    category: str  # 技术 / 生活 / 其他
    answer: str


# ---------- 节点 1：分类 ----------
def classify(state: RouteState) -> dict:
    prompt = (
        "把下面的用户问题分类，只输出一个词：技术、生活、其他。不要解释。\n"
        f"问题：{state['question']}"
    )
    raw = model.invoke(prompt).content.strip()
    # 模型偶尔会带标点或多余字，这里做一次兜底归一化
    category = "其他"
    for label in ("技术", "生活"):
        if label in raw:
            category = label
            break
    print(f"  [classify] 模型原始输出={raw!r} -> 归类={category}")
    return {"category": category}


# ---------- 路由函数：只返回标签 ----------
def route_by_category(state: RouteState) -> str:
    return state["category"]


# ---------- 节点 2/3/4：不同分支的处理器 ----------
def tech_expert(state: RouteState) -> dict:
    answer = model.invoke(
        f"你是资深工程师，用三步以内给出可执行的技术建议：{state['question']}"
    ).content
    return {"answer": f"[技术专家] {answer}"}


def life_expert(state: RouteState) -> dict:
    answer = model.invoke(
        f"你是生活顾问，用轻松的语气给出实用建议：{state['question']}"
    ).content
    return {"answer": f"[生活顾问] {answer}"}


def general_expert(state: RouteState) -> dict:
    answer = model.invoke(f"请简洁回答：{state['question']}").content
    return {"answer": f"[通用助手] {answer}"}


def build_graph():
    builder = StateGraph(RouteState)
    builder.add_node("classify", classify)
    builder.add_node("tech", tech_expert)
    builder.add_node("life", life_expert)
    builder.add_node("general", general_expert)

    builder.add_edge(START, "classify")

    # 关键：classify 之后不是固定去某个节点，而是由 route_by_category 决定
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "技术": "tech",
            "生活": "life",
            "其他": "general",
        },
    )

    # 三个分支都走向终点
    builder.add_edge("tech", END)
    builder.add_edge("life", END)
    builder.add_edge("general", END)
    return builder.compile()


def main() -> None:
    app = build_graph()
    print("【图结构】")
    app.get_graph().print_ascii()

    for question in [
        "Python 服务内存泄漏怎么排查？",
        "最近总是失眠，有什么办法？",
        "帮我起一个宠物店的名字。",
    ]:
        print(f"\n用户：{question}")
        result = app.invoke({"question": question, "category": "", "answer": ""})
        print("分类：", result["category"])
        print("回答：", result["answer"])


if __name__ == "__main__":
    main()
