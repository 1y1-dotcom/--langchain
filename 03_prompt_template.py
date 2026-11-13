"""
03 · 提示词模板（Prompt Template）

对应原仓库：04-prompt/prompt_templates/*、04-prompt/chat_prompt_template/*

为什么需要模板？
    把「固定指令」和「变量」分开：模板写一次，变量每次传。
    更重要的是：模板对象也是 Runnable，可以用 | 直接和模型串成链。

两类模板：
    PromptTemplate      -> 产出一段纯字符串（适合单轮补全式提示）
    ChatPromptTemplate  -> 产出消息列表（能区分 system / human / ai，推荐）

运行：
    python 03_prompt_template.py
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from config import get_chat_model


def main() -> None:
    # ---------- 1. 字符串模板 ----------
    # {product} 是变量占位符；调用时通过 invoke({"product": ...}) 填入
    string_tpl = PromptTemplate.from_template(
        "请为产品「{product}」写一句不超过 20 字的广告语，突出它最大的卖点。"
    )
    # format() 只做填充，不调用模型，便于调试最终提示词
    print("【PromptTemplate 填充结果】")
    print(string_tpl.format(product="降噪耳机"))
    print("=" * 60)

    # ---------- 2. 聊天模板 ----------
    # ("system", ...) 和 ("human", ...) 是消息角色的简写
    chat_tpl = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名{role}，回答要简洁，不超过三句话。"),
            ("human", "请解释一下{concept}。"),
        ]
    )

    # format_messages 会产出 SystemMessage / HumanMessage 对象列表
    print("【ChatPromptTemplate 产出的消息结构】")
    for msg in chat_tpl.format_messages(role="数据库工程师", concept="向量索引"):
        print(type(msg).__name__, "->", msg.content)
    print("=" * 60)

    # ---------- 3. 模板 + 模型 = 最小链 ----------
    # 模板负责「组织问题」，模型负责「回答」，输入仍然是普通 dict
    chain = chat_tpl | get_chat_model(temperature=0.3)
    answer = chain.invoke({"role": "产品经理", "concept": "RAG（检索增强生成）"})
    print("【模板 + 模型】")
    print(answer.content)
    print("=" * 60)

    # ---------- 4. partial：预填一部分变量 ----------
    # 场景：system 里的角色固定，只有 concept 每次变化
    partial_tpl = chat_tpl.partial(role="中学老师")
    # partial 后可以直接和模型组合成链，调用时只传剩下的变量
    partial_chain = partial_tpl | get_chat_model(temperature=0.3)
    print("【partial 预填 role 后，只需再传 concept】")
    print(partial_chain.invoke({"concept": "什么是 API"}).content)
    # 注意：partial 后仍然必须把 concept 填上，否则会报缺少变量


if __name__ == "__main__":
    main()
