"""
12 · State 与 Reducer（状态合并规则）

对应原仓库：03-state/reducers/*、03-state/DefState.py

Reducer 决定「节点返回的更新」如何合并进现有状态：
    默认行为            直接覆盖
    operator.add        数值相加 / 列表拼接
    add_messages        消息追加（上一章已用）
    自定义函数          f(left, right) -> merged，可以实现去重、取最大值等

为什么必须懂 Reducer？
    并行分支同时写同一个字段时，没有 Reducer 就会互相覆盖；
    有了 Reducer，合并规则由你定义。

运行：
    python 12_langgraph_state.py
"""

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ---------- 1. 带 Reducer 的状态 ----------
class PipelineState(TypedDict):
    # operator.add 对 list 是拼接：节点返回 ["a"]，会追加到已有列表后面
    logs: Annotated[list[str], operator.add]
    # 对 int 是相加
    total: Annotated[int, operator.add]
    # 不写 Annotated 的字段是默认覆盖语义
    final: str


# ---------- 2. 节点 ----------
def step_a(state: PipelineState) -> dict:
    return {"logs": ["A 执行"], "total": 1}


def step_b(state: PipelineState) -> dict:
    return {"logs": ["B 执行"], "total": 10}


def step_c(state: PipelineState) -> dict:
    # 这里读取的是 A、B 合并之后的状态
    return {"logs": [f"C 汇总：total={state['total']}"], "total": 100}


def finalize(state: PipelineState) -> dict:
    return {"final": "；".join(state["logs"])}


def main() -> None:
    builder = StateGraph(PipelineState)
    builder.add_node("A", step_a)
    builder.add_node("B", step_b)
    builder.add_node("C", step_c)
    builder.add_node("finalize", finalize)

    # START 同时连 A 和 B：两个节点并行执行
    builder.add_edge(START, "A")
    builder.add_edge(START, "B")
    # A、B 都完成后，C 才会执行（C 有两个入边 = 等两者都结束）
    builder.add_edge("A", "C")
    builder.add_edge("B", "C")
    builder.add_edge("C", "finalize")
    builder.add_edge("finalize", END)

    app = builder.compile()
    result = app.invoke({"logs": ["开始"], "total": 0, "final": ""})

    print("【最终状态】")
    print("logs  =", result["logs"])
    print("total =", result["total"])  # 0 + 1 + 10 + 100
    print("final =", result["final"])

    print("\n【图结构：A、B 并行，C 等待汇合】")
    app.get_graph().print_ascii()

    # 观察点：
    # 1) total 是累加的（operator.add），不是最后一次的 100
    # 2) A/B 的 logs 顺序由调度决定，但两条都保留（未互相覆盖）
    # 3) final 字段没有 Reducer，finalize 直接覆盖


if __name__ == "__main__":
    main()
