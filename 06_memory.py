"""
06 · 对话记忆（Memory）

对应原仓库：07-memory/Memory_InMemoryChatMessageHistory.py、Memory_RunnableWithMessageHistory.py

先想清楚一件事：模型本身是无状态的。
    「记住上文」不是模型的能力，而是每次请求时把历史消息一起发过去的结果。

本文件演示两种做法：
    1. 手工维护消息列表（最直观，理解原理）
    2. RunnableWithMessageHistory（框架封装，按 session_id 自动存取历史）

注意：这里的 InMemory 历史只存在进程内存里，程序退出就没了。
      生产环境会换成 Redis / 数据库（原仓库的 RedisChatMessageHistory 案例）。

补充：LangChain 1.x 已把 RunnableWithMessageHistory 标记为 deprecated（不建议新项目使用），
      但它的思路仍是标准做法：按 session_id 存取历史 + 每次调用前把历史拼进提示词。
      新版更推荐用 LangGraph 的 checkpointer 管理多轮状态（见 14_langgraph_persistence.py）。

运行：
    python 06_memory.py
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import get_chat_model

model = get_chat_model(temperature=0)


def manual_memory_demo() -> None:
    """做法一：手动把历史消息拼进 messages 列表。"""
    print("【1. 手工维护历史】")
    history: list = []

    for question in ["我叫小明，正在学 LangGraph。", "我叫什么？在学什么？"]:
        history.append(HumanMessage(content=question))
        reply = model.invoke(history)
        history.append(reply)  # 关键：把模型回复也放回历史
        print("用户：", question)
        print("助手：", reply.content)
    print("历史消息条数：", len(history))
    print("=" * 60)


def runnable_with_history_demo() -> None:
    """做法二：RunnableWithMessageHistory 自动管理历史。"""
    print("【2. RunnableWithMessageHistory】")

    # MessagesPlaceholder：在提示词里挖一个「历史消息」的槽位，
    # 框架每次调用前会把该 session 的历史记录填进来。
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个简洁的助手，回答不超过两句话。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | model | StrOutputParser()

    # session_id -> 历史对象。真实项目中这里换成 Redis 等持久化存储
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chat = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 同一个 session_id 的对话会共享历史
    config = {"configurable": {"session_id": "user-A"}}
    for question in ["我最喜欢的语言是 Python。", "我最喜欢的语言是什么？"]:
        answer = chat.invoke({"input": question}, config=config)
        print("用户：", question)
        print("助手：", answer)

    # 换一个 session_id：历史互不影响，模型不知道 user-A 说过什么
    other = {"configurable": {"session_id": "user-B"}}
    print("用户（另一个会话）：我最喜欢的语言是什么？")
    print("助手：", chat.invoke({"input": "我最喜欢的语言是什么？"}, config=other))
    print("=" * 60)

    # 本地存储里实际存了什么
    print("user-A 的历史条数：", len(store["user-A"].messages))
    print("user-B 的历史条数：", len(store["user-B"].messages))


if __name__ == "__main__":
    manual_memory_demo()
    runnable_with_history_demo()
