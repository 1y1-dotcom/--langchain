"""
05 · LCEL 链式调用（Runnable 管道）

对应原仓库：06-lcel/*（lcel.py、LCEL_MultiStepChainDemo.py、RunnableParallel/Branch/Lambda）

LCEL = LangChain Expression Language，核心只有一个符号：|
    A | B 的意思是「把 A 的输出交给 B 的输入」。
    因为 Prompt、Model、Parser、Retriever、自定义函数都被包装成 Runnable，
    所以它们可以像 Unix 管道一样自由拼接。

本文件覆盖四种最常用组合：
    1. 顺序链        prompt | model | parser
    2. 自定义函数    RunnableLambda（在链中间插入普通 Python 逻辑）
    3. 并行          RunnableParallel（同一份输入，同时走多条链）
    4. 条件分支      RunnableBranch（按输入内容选不同的链）

运行：
    python 05_lcel_chain.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel

from config import get_chat_model

model = get_chat_model(temperature=0.3)
parser = StrOutputParser()


def main() -> None:
    # ---------- 1. 顺序链 ----------
    explain_chain = (
        ChatPromptTemplate.from_template("用不超过 50 字解释：{concept}")
        | model
        | parser
    )
    print("【1. 顺序链】")
    print(explain_chain.invoke({"concept": "LangChain 中的 LCEL 表达式"}))
    print("=" * 60)

    # ---------- 2. RunnableLambda：链中插入普通函数 ----------
    # 应用场景：清洗输入、拼装参数、对模型输出做后处理。
    # 函数收到的是「上一步的输出」，返回值会传给下一步。
    def add_context(text: str) -> str:
        print("  [RunnableLambda] 收到模型输出，长度 =", len(text))
        return f"【已由后处理函数加工】{text}"

    chain_with_lambda = explain_chain | RunnableLambda(add_context)
    print("【2. RunnableLambda】")
    print(chain_with_lambda.invoke({"concept": "LangChain 中的 Runnable 接口"}))
    print("=" * 60)

    # ---------- 3. RunnableParallel：一份输入，多条链并行 ----------
    # 返回一个 dict，键名由你指定；两条链共享同一个输入 dict。
    # 实际耗时约等于较慢的那条，而不是两条相加。
    translate_chain = (
        ChatPromptTemplate.from_template("把这句话翻译成英文，只输出译文：{text}")
        | model
        | parser
    )
    summary_chain = (
        ChatPromptTemplate.from_template("用一句话总结：{text}") | model | parser
    )
    parallel_chain = RunnableParallel(english=translate_chain, summary=summary_chain)

    print("【3. RunnableParallel】")
    result = parallel_chain.invoke({"text": "LangGraph 用图结构描述智能体的执行流程。"})
    print("english:", result["english"])
    print("summary:", result["summary"])
    print("=" * 60)

    # ---------- 4. RunnableBranch：条件分支 ----------
    # 第一个参数是 (判断函数, 分支链)，判断函数返回 True 就走这条；
    # 最后一个参数是兜底链（所有条件都不满足时执行）。
    code_chain = (
        ChatPromptTemplate.from_template("请给出 Python 代码示例：{question}") | model | parser
    )
    general_chain = ChatPromptTemplate.from_template("请直接回答：{question}") | model | parser

    branch_chain = RunnableBranch(
        (lambda x: "代码" in x["question"], code_chain),
        general_chain,
    )
    print("【4. RunnableBranch：命中代码分支】")
    print(branch_chain.invoke({"question": "用代码演示列表推导式"}))
    print("-" * 60)
    print("【4. RunnableBranch：走兜底分支】")
    print(branch_chain.invoke({"question": "什么是列表推导式"}))

    # ---------- 附：查看链的结构 ----------
    # LCEL 链本质是一棵 Runnable 树，get_graph() 可以打印出来帮助理解
    print("=" * 60)
    print("【链结构】")
    explain_chain.get_graph().print_ascii()


if __name__ == "__main__":
    main()
