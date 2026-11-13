"""
14 · 状态持久化（Checkpointer）——多轮对话与断点

对应原仓库：07-senior/state_persistence/MemoryPersistence.py、SqlitePersistence.py

没有 checkpointer 时，每次 invoke 都是从零开始。
加上 checkpointer 后：
    每次执行都会保存状态快照；
    thread_id 相同的调用会接着上一次的状态继续。

三个常用 API：
    compile(checkpointer=...)
    app.invoke(input, {"configurable": {"thread_id": "..."}})
    app.get_state(config) / app.get_state_history(config)

InMemorySaver 只存在进程内存；生产环境换 SqliteSaver / PostgresSaver / Redis。

运行：
    python 14_langgraph_persistence.py
"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import get_chat_model


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    # 注意：这里拿到的是「该 thread 累积的全部消息」，所以模型能接着上文
    reply = get_chat_model(temperature=0).invoke(state["messages"])
    return {"messages": [reply]}


def build_app():
    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    # 关键差异：编译时挂上 checkpointer
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    app = build_app()

    # thread_id 就是会话 ID。同一个 ID = 同一条对话线程
    config = {"configurable": {"thread_id": "user-1001"}}

    print("【第一轮】")
    app.invoke({"messages": [HumanMessage(content="我叫小明，正在学习 LangGraph。")]}, config)
    print("用户：我叫小明，正在学习 LangGraph。")

    print("\n【第二轮：不重复自我介绍，看模型是否记得】")
    result = app.invoke({"messages": [HumanMessage(content="我叫什么？在学什么？")]}, config)
    print("用户：我叫什么？在学什么？")
    print("助手：", result["messages"][-1].content)

    # ---------- 查看当前状态 ----------
    snapshot = app.get_state(config)
    print("\n【当前状态】消息条数 =", len(snapshot.values["messages"]))
    print("下一个待执行节点 =", snapshot.next)  # 已结束，所以是空元组

    # ---------- 查看历史快照 ----------
    # 每个超级步（节点执行）都会留下一条 checkpoint，可用于调试和时间回溯
    history = list(app.get_state_history(config))
    print("\n【历史快照数量】", len(history))
    for i, item in enumerate(reversed(history)):
        print(f"  快照{i}: 消息数={len(item.values.get('messages', []))}, next={item.next}")

    # ---------- 换一个 thread_id：互不干扰 ----------
    other = {"configurable": {"thread_id": "user-2002"}}
    result2 = app.invoke({"messages": [HumanMessage(content="我叫什么？")]}, other)
    print("\n【新会话 user-2002 问同样的问题】")
    print("助手：", result2["messages"][-1].content)


if __name__ == "__main__":
    main()
