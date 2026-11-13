"""
04 · 输出解析（Output Parser）

对应原仓库：05_parser/StrOutputParserDemo.py、JsonOutputParserDemo.py、StructuredOutput_Pydantic.py

模型返回的是 AIMessage，程序真正需要的是「可用的数据」。解析器就是这层转换：
    StrOutputParser        AIMessage -> str            最常用，去掉包装
    JsonOutputParser       AIMessage -> dict           要模型按 JSON 输出
    with_structured_output AIMessage -> Pydantic 对象   最稳的工程做法（推荐）

运行：
    python 04_output_parser.py
"""

import json

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import get_chat_model


# ---------- 用 Pydantic 描述「我想要的结构」 ----------
class ProductReview(BaseModel):
    """模型需要输出的字段定义，Field 的 description 会作为提示告诉模型含义。"""

    sentiment: str = Field(description="情感倾向，只能是 正面/负面/中性 之一")
    score: int = Field(description="评分，1 到 5 的整数")
    keywords: list[str] = Field(description="评论中提到的 2-3 个关键词")
    summary: str = Field(description="一句话总结")


def main() -> None:
    model = get_chat_model(temperature=0)

    # ---------- 1. StrOutputParser ----------
    # 不加解析器时拿到 AIMessage；加了之后直接拿到字符串，链的终点更「干净」
    str_chain = ChatPromptTemplate.from_template("用一句话介绍{topic}") | model | StrOutputParser()
    text = str_chain.invoke({"topic": "MCP 协议"})
    # LangChain 1.x 返回的是 str 的子类 TextAccessor，所以用 isinstance 判断更准确
    print("【StrOutputParser】isinstance(text, str) =", isinstance(text, str))
    print(text)
    print("=" * 60)

    # ---------- 2. JsonOutputParser ----------
    # parser.get_format_instructions() 会生成一段 JSON 格式要求，塞进提示词里
    json_parser = JsonOutputParser()
    json_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个信息抽取助手。只输出 JSON，不要输出解释。"),
            ("human", "从这句话中抽取 姓名、城市、职业：{sentence}\n{format}"),
        ]
    ).partial(format=json_parser.get_format_instructions())

    json_chain = json_prompt | model | json_parser
    data = json_chain.invoke({"sentence": "张伟住在杭州，是一名后端工程师。"})
    print("【JsonOutputParser】type =", type(data).__name__)
    print(json.dumps(data, ensure_ascii=False))
    print("=" * 60)

    # ---------- 3. with_structured_output（工程首选） ----------
    # 不用自己写"请输出 JSON"，框架会把 Pydantic 模型转成 JSON Schema 交给模型，
    # 并自动解析、校验。字段缺失或类型不对会直接报错，比正则可靠得多。
    #
    # method="function_calling"：通过「工具调用」传 Schema。
    # 为什么要显式指定？新版默认走 response_format(json_schema)，
    # 而部分国内模型对 response_format 支持不完整，会忽略它并返回自然语言，
    # 导致解析报 "Invalid JSON"。工具调用模式的兼容性更好。
    structured_model = model.with_structured_output(ProductReview, method="function_calling")
    review_text = "耳机音质很棒，降噪效果一流，就是戴久了有点夹头，整体还是推荐。"
    review: ProductReview = structured_model.invoke(
        f"分析下面这条商品评论：\n{review_text}"
    )
    print("【with_structured_output】type =", type(review).__name__)
    print(review.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
