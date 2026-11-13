"""
10 · ReAct Agent（推理 + 行动）

对应原仓库：12-agent/AgentReact.py、21-Agent智能体.md

ReAct = Reason + Act。它不是一个新模型，而是一种「循环」：
    模型推理 -> 决定调用哪个工具 -> 程序执行 -> 结果回填 -> 模型再推理 -> ... -> 输出答案

第 07 章我们手写了工具调用循环；本章用 LangChain 1.x 的 create_agent 把循环交给框架。
    注意：框架替你省掉的是「循环代码」，不是「模型执行工具」这件事——
    工具始终在你的进程里执行。

运行：
    python 10_agent_react.py
"""

from langchain.agents import create_agent
from langchain_core.tools import tool

from config import get_chat_model

# ---------- 模拟业务数据 ----------
PRODUCTS = {
    "无线耳机": [
        {"id": "WH-1000XM5", "name": "索尼 WH-1000XM5", "popularity": 95, "price": 299},
        {"id": "AIRMAX", "name": "苹果 AirPods Max", "popularity": 92, "price": 549},
        {"id": "QC45", "name": "Bose QC45", "popularity": 88, "price": 329},
    ],
    "游戏鼠标": [
        {"id": "GPW", "name": "罗技 G Pro 无线", "popularity": 90, "price": 129},
        {"id": "VIPER", "name": "雷蛇 Viper V2 Pro", "popularity": 87, "price": 149},
    ],
}

INVENTORY = {
    "WH-1000XM5": {"stock": 10, "location": "仓库-A"},
    "AIRMAX": {"stock": 5, "location": "仓库-C"},
    "QC45": {"stock": 0, "location": "仓库-B"},
    "GPW": {"stock": 8, "location": "仓库-C"},
    "VIPER": {"stock": 12, "location": "仓库-A"},
}


@tool
def search_products(category: str) -> str:
    """按类别搜索产品，返回按受欢迎度排序的产品列表。category 可选：无线耳机、游戏鼠标。"""
    print(f"  [工具执行] search_products({category!r})")
    items = PRODUCTS.get(category)
    if not items:
        return f"没有找到类别：{category}"
    items = sorted(items, key=lambda x: x["popularity"], reverse=True)
    lines = [f"{i}. {p['name']}（ID: {p['id']}，受欢迎度 {p['popularity']}%，￥{p['price']}）"
             for i, p in enumerate(items, 1)]
    return "\n".join(lines)


@tool
def check_inventory(product_id: str) -> str:
    """查询某个产品的库存。product_id 必须是 search_products 返回的产品 ID。"""
    print(f"  [工具执行] check_inventory({product_id!r})")
    info = INVENTORY.get(product_id)
    if not info:
        return f"未找到产品 ID：{product_id}"
    status = "有货" if info["stock"] > 0 else "缺货"
    return f"{product_id}: {status}，库存 {info['stock']} 件，位置 {info['location']}"


def main() -> None:
    # create_agent = 模型 + 工具 + 系统提示 + 内置循环
    agent = create_agent(
        model=get_chat_model(temperature=0),
        tools=[search_products, check_inventory],
        system_prompt=(
            "你是电商助手，按 ReAct 模式工作："
            "先判断需要什么信息，再调用工具获取，"
            "根据工具结果继续推理，直到能给出完整答案。不要编造库存数据。"
            "已经成功获取的信息不要重复查询。"
        ),
    )

    question = "最受欢迎的无线耳机是哪款？它还有库存吗？"
    print("用户：", question)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    print("\n【完整消息轨迹】")
    for msg in result["messages"]:
        role = type(msg).__name__
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                print(f"{role}: 决定调用 {call['name']}({call['args']})")
        else:
            print(f"{role}: {msg.content}")


if __name__ == "__main__":
    main()
