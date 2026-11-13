"""
01 · 最小模型调用（Hello LLM）

对应原仓库：案例与源码-2-LangChain框架/01-helloworld/*、02-models_io/ModelIO_*

本章要掌握的三件事：
    1. 调用三件套：api_key（身份）+ base_url（地址）+ model（模型名）。
    2. 模型对象是 Runnable：支持 invoke（一次）、stream（流式）、batch（批量）。
    3. 输入既可以是字符串，也可以是「消息列表」——后者才能区分 system / user / ai 角色。

运行：
    python 01_hello_model.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from config import describe, get_chat_model


def main() -> None:
    # 打印当前用的是哪个厂商/模型（不含 Key），出问题时先看这里
    print("当前模型配置：")
    print(describe())
    print("=" * 60)

    model = get_chat_model()

    # ---------- 写法一：直接传字符串 ----------
    # 框架会自动包成一条 HumanMessage（user 角色）
    reply = model.invoke("用一句话解释什么是大语言模型（LLM）。")
    print("【字符串输入】")
    print(reply.content)

    # AIMessage 不只有 content，还带元数据：实际模型名、token 用量等
    print("模型名：", reply.response_metadata.get("model_name"))
    print("token：", reply.usage_metadata)
    print("=" * 60)

    # ---------- 写法二：消息列表（可控制角色） ----------
    # system 用来设定人设/规则，user 是本轮问题；多轮对话就是不断往列表里追加消息
    messages = [
        SystemMessage(content="你是一名严谨的技术讲师，回答不超过两句话。"),
        HumanMessage(content="LangChain 里的 Runnable 是什么？"),
    ]
    reply = model.invoke(messages)
    print("【消息列表输入】")
    print(reply.content)
    print("=" * 60)

    # ---------- 写法三：流式输出 ----------
    # stream 返回迭代器，每个 chunk 只带一小段文本；适合做打字机效果
    print("【流式输出】")
    for chunk in model.stream("分三点说明 Agent 和普通 Workflow 的区别。"):
        print(chunk.content, end="", flush=True)
    print("\n" + "=" * 60)

    # 附：批量调用（内部并发），一次问多个问题
    print("【批量调用】")
    replies = model.batch(["1+1 等于几？", "中国的首都是哪里？"])
    for q, r in zip(["1+1 等于几？", "中国的首都是哪里？"], replies):
        print(f"Q: {q}\nA: {r.content}\n")


if __name__ == "__main__":
    main()
