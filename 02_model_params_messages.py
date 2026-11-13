"""
02 · 模型参数与多角色消息

对应原仓库：02-models_io/ModelIO_Params.py、04-prompt/invoke/*

知识点：
    - temperature：随机性。0 最稳定，适合抽取/分类/工具调用；0.7+ 适合创意写作。
    - max_tokens：限制回复长度，能防止跑偏和浪费额度。
    - system / user / assistant 三种角色各自的职责。
    - 把历史 assistant 消息也放回列表，模型就能「接着上文」回答。

运行：
    python 02_model_params_messages.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_chat_model


def main() -> None:
    # temperature=0：同样的问题尽量给同样的答案
    strict_model = get_chat_model(temperature=0.0, max_tokens=80)

    print("【temperature=0，限制 80 token】")
    reply = strict_model.invoke("用一句话说明什么是向量数据库。")
    print(reply.content)
    # 注意：output_tokens 是「实际输出」的 token 数，部分模型还会把内部推理算进去，
    # 所以可能略大于 80；正文长度仍受 max_tokens 约束。
    print("实际计费 token（含思考）：", reply.usage_metadata.get("output_tokens"))
    print("=" * 60)

    # 手工构造一段「已经发生过」的多轮对话
    # assistant 角色 = 模型之前的回复，放回去是为了让模型知道上下文
    conversation = [
        SystemMessage(content="你是一个只回答技术问题的助手，回答要口语化。"),
        HumanMessage(content="我叫小明，正在学 LangChain。"),
        AIMessage(content="你好小明！LangChain 是一个用来编排大模型应用开发的框架。"),
        HumanMessage(content="我刚才说我叫什么？我在学什么？"),
    ]

    print("【多轮消息：模型能否记住上文】")
    print(get_chat_model(temperature=0).invoke(conversation).content)
    print("=" * 60)

    # 角色约束小实验：同一个问题，system 不同，答案风格不同
    question = "介绍一下 Redis。"
    for system_text in ["你是小学老师，用比喻解释。", "你是运维工程师，只讲部署要点。"]:
        msgs = [SystemMessage(content=system_text), HumanMessage(content=question)]
        print(f"【system = {system_text}】")
        print(get_chat_model(temperature=0.5).invoke(msgs).content)
        print("-" * 60)


if __name__ == "__main__":
    main()
