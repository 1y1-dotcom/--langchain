"""
07 · 工具调用（Tool Calling）

对应原仓库：08-tools/*、17-Tools工具调用.md

核心结论（很重要）：
    模型不会真的执行代码。它只输出「调用意图」——一个 JSON，说明想调用哪个工具、传什么参数。
    真正执行工具的是我们自己的程序。这就是 Agent 能被审计、能控制权限的原因。

一次工具调用的完整流程：
    user 消息
      -> 模型返回 AIMessage(tool_calls=[{name, args, id}])
      -> 程序执行对应函数，把结果包成 ToolMessage(tool_call_id=...)
      -> 把 ToolMessage 追加进消息列表，再次调用模型
      -> 模型基于工具结果给出最终回答（这次没有 tool_calls）

运行：
    python 07_tools.py
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from config import get_chat_model


# ---------- 定义工具 ----------
# @tool 会读取：函数名 -> 工具名；docstring -> 给模型看的说明；类型注解 -> 参数 JSON Schema。
# docstring 写得含糊，模型就会乱传参数——它不是"猜"代码，而是在读这段说明。
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。city 使用中文城市名，例如：杭州、北京。"""
    fake_weather = {
        "杭州": "多云，18~25℃，东南风 2 级",
        "北京": "晴，8~20℃，微风",
        "上海": "小雨，16~22℃，东风 3 级",
    }
    return fake_weather.get(city, f"暂时没有 {city} 的天气数据")


@tool
def add(a: float, b: float) -> str:
    """计算两个数字的和。a 和 b 都必须是数字。"""
    return str(a + b)


def main() -> None:
    tools = [get_weather, add]
    tools_by_name = {t.name: t for t in tools}

    # bind_tools：把工具说明书交给模型。此时模型"知道"有哪些工具可用
    model = get_chat_model(temperature=0).bind_tools(tools)

    messages = [HumanMessage(content="杭州今天天气怎么样？另外帮我算一下 12.5 + 7.5。")]

    # ---------- 循环：模型决策 -> 程序执行 -> 回填结果 ----------
    # 注意：一次可能只返回一个工具调用（不同模型的并行能力不同），
    # 所以必须用循环，直到模型不再要求调用工具为止。
    for round_no in range(1, 6):
        print(f"\n【第 {round_no} 轮：模型决策】")
        ai_msg = model.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            print("模型不再调用工具，给出最终回答：")
            print(ai_msg.content)
            break

        print("content:", repr(ai_msg.content))  # 有工具调用时通常为空
        # 程序侧执行工具：执行哪个函数、传什么参数，完全由我们根据 tool_calls 决定，
        # 因此可以在这一步做权限校验、参数校验、超时控制。
        for call in ai_msg.tool_calls:
            print(f"  调用意图：{call['name']}({call['args']})")
            tool_fn = tools_by_name[call["name"]]
            result = tool_fn.invoke(call["args"])  # args 已经是 dict
            print(f"  程序执行结果：{result}")
            # tool_call_id 必须与上面的 id 对应，模型才知道这是哪次调用的结果
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    else:
        print("达到最大轮数，模型仍在请求调用工具，已强制停止。")

    # 完整轨迹：可以看到 Human -> AI(tool_calls) -> Tool -> AI 的配对关系
    print("\n【完整消息轨迹】")
    for msg in messages:
        if getattr(msg, "tool_calls", None):
            print(f"{type(msg).__name__}: 请求调用 {[c['name'] for c in msg.tool_calls]}")
        else:
            print(f"{type(msg).__name__}: {msg.content}")


if __name__ == "__main__":
    main()
