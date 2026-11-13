"""
15 · 用 LangGraph 手写一个 ReAct Agent

对应原仓库：08-multi_agent/LangGraphAgent.py、26-LangGraph多智能体与A2A.md

第 10 章用 create_agent 一行创建了 Agent。本章把它拆开，看清内部其实就是一张图：

    START -> assistant（模型决策）
                 |
         有 tool_calls ? --是--> tools（执行工具）--> 回到 assistant
                 |
                否
                 v
                END

这个「带环的图」就是 Agent 的本质：
    模型只负责决策；工具执行在 tools 节点；条件边决定是继续循环还是结束。

运行：
    python 15_langgraph_agent.py
"""

from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import get_chat_model

MAX_STEPS = 6  # 防止模型反复调用工具造成死循环


# ---------- 工具 ----------
@tool
def get_weather(city: str) -> str:
    """查询城市天气。city 用中文城市名，如：杭州。"""
    data = {"杭州": "多云 18~25℃", "北京": "晴 8~20℃", "上海": "小雨 16~22℃"}
    return data.get(city, f"暂无 {city} 的天气数据")


@tool
def get_clothing_advice(temperature_hint: str) -> str:
    """根据温度提示给出穿衣建议。temperature_hint 例如：18~25℃。"""
    return f"针对 {temperature_hint}：建议薄外套 + 长裤，早晚可加一件针织衫。"


TOOLS = [get_weather, get_clothing_advice]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int  # 已执行轮数，用于限制循环


# ---------- 节点 1：模型决策 ----------
def assistant(state: AgentState) -> dict:
    model = get_chat_model(temperature=0).bind_tools(TOOLS)
    reply = model.invoke(state["messages"])
    return {"messages": [reply], "steps": state.get("steps", 0) + 1}


# ---------- 节点 2：执行工具 ----------
def tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_messages = []
    for call in last.tool_calls:
        fn = TOOLS_BY_NAME[call["name"]]
        result = fn.invoke(call["args"])
        print(f"  [tools] {call['name']}({call['args']}) -> {result}")
        # tool_call_id 必须对应，模型才能把结果和请求配对
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": tool_messages}


# ---------- 条件边：继续调工具，还是结束 ----------
def should_continue(state: AgentState) -> str:
    if state.get("steps", 0) >= MAX_STEPS:
        return "end"
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_app():
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "assistant")

    # 条件边：assistant 之后二选一
    builder.add_conditional_edges(
        "assistant",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # 关键的一笔：工具执行完回到 assistant，形成循环
    builder.add_edge("tools", "assistant")
    return builder.compile()


def main() -> None:
    app = build_app()
    print("【图结构：注意 tools -> assistant 的环】")
    app.get_graph().print_ascii()

    question = "杭州今天天气如何？我应该穿什么？"
    print("\n用户：", question)
    result = app.invoke(
        {"messages": [HumanMessage(content=question)], "steps": 0},
        # LangGraph 的循环受 recursion_limit 保护，这里显式放宽一点
        {"recursion_limit": 20},
    )

    print("\n【消息轨迹】")
    for msg in result["messages"]:
        if getattr(msg, "tool_calls", None):
            print("AIMessage: 决定调用", [c["name"] for c in msg.tool_calls])
        elif isinstance(msg, ToolMessage):
            print("ToolMessage:", msg.content)
        else:
            print(f"{type(msg).__name__}: {msg.content}")


if __name__ == "__main__":
    main()
