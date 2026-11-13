"""
11 · LangGraph 最小对话图

对应原仓库：案例与源码-3-LangGraph框架/01-helloworld/LangGraphLLM.py

LangGraph 的思维方式：
    把应用写成一个「状态机 / 有向图」。
    State（共享状态）在节点之间流动，节点是普通函数，边决定下一步去哪。

三个核心概念：
    State  一个 TypedDict，描述图里共享的数据结构
    Node   函数：接收 state，返回「对 state 的部分更新」
    Edge   连接节点；有普通边和条件边

本文件是最小形态：START -> model -> END。

运行：
    python 11_langgraph_hello.py
"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import get_chat_model


# ---------- 1. 定义状态 ----------
# messages 字段用 add_messages 规约器（Reducer）：
#   节点返回 {"messages": [新消息]} 时，框架会「追加」而不是「覆盖」旧消息。
#   如果去掉 Annotated[..., add_messages]，第二次更新会把历史全部冲掉。
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- 2. 定义节点 ----------
def model_node(state: ChatState) -> dict:
    """把当前消息列表交给模型，返回模型的新消息。"""
    reply = get_chat_model(temperature=0.3).invoke(state["messages"])
    # 只返回增量，add_messages 负责合并到 state["messages"]
    return {"messages": [reply]}


# ---------- 3. 建图 ----------
builder = StateGraph(ChatState)
builder.add_node("model", model_node)
builder.add_edge(START, "model")
builder.add_edge("model", END)

# compile 之后才是可执行对象；真实项目里这里还会传 checkpointer 做持久化
app = builder.compile()


def main() -> None:
    result = app.invoke(
        {"messages": [HumanMessage(content="用一句话解释 LangGraph 是什么。")]}
    )
    print("模型回答：", result["messages"][-1].content)

    print("\n【消息列表（add_messages 合并结果）】")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content}")

    # ---------- 4. 打印图结构 ----------
    # get_graph() 是调试 LangGraph 最实用的工具：先看清拓扑，再查逻辑
    print("\n【图结构】")
    app.get_graph().print_ascii()


if __name__ == "__main__":
    main()
