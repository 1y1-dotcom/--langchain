"""
17 · 统一入口：校园制度问答助手（命令行）

前面 16 个脚本分别验证一层能力，这个文件把它们收成一个「能提问的应用」：
    输入问题 -> 检索制度文档 -> 判断资料是否相关 -> 回答 + 显示出处

它不重复实现 RAG，而是直接复用 16_langgraph_rag.py 里已经验证过的图。

运行：
    python cli_app.py                              # 交互模式，输入 q 退出
    python cli_app.py "出差报销几天内提交？"        # 单次提问模式
"""

import importlib
import sys

# 文件名以数字开头，不能用 import 语句，只能用 importlib 动态导入
rag = importlib.import_module("16_langgraph_rag")

# 编译一次，后续每个问题复用同一张图
APP = rag.build_app()


def answer(question: str) -> None:
    """跑一次图，并把回答和出处打印出来。"""
    result = APP.invoke(
        {"question": question, "context": "", "verdict": "", "answer": ""}
    )
    print(f"\n问题：{question}")
    print(f"判断：{result['verdict']}")
    print(f"回答：{result['answer']}")

    if result["verdict"] == "相关":
        # 再检索一次只是为了拿到 metadata 里的主题，用来显示「出处」
        docs = rag.RETRIEVER.invoke(question)
        topics = "、".join(d.metadata.get("topic", "未知") for d in docs)
        print(f"出处：{topics}")
        for i, doc in enumerate(docs, 1):
            print(f"  [{i}] {doc.page_content}")
    else:
        print("出处：（知识库中没有相关资料，已走兜底分支）")


def main() -> None:
    # 命令行带了问题：只回答一次，适合演示时直接跑
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
        return

    print("=" * 50)
    print("校园制度问答助手（输入 q 退出）")
    print("可问：报销、年假、远程办公")
    print("试试：出差报销几天内提交？")
    print("=" * 50)
    while True:
        try:
            question = input("\n你问：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            answer(question)


if __name__ == "__main__":
    main()
