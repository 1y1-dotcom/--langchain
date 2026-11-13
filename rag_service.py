""" 
18 · 核心服务层：知识库 + 会话存储 + 可改写查询的 RAG 图

这个文件把「演示脚本」升级成「可被界面调用的服务」：
    1. SQLite 存三类数据：文档、会话、消息（标准库 sqlite3，零额外依赖）
    2. 向量检索可以随文档增删而重建（InMemoryVectorStore + 千问向量模型）
    3. RAG 图在 16 号脚本的基础上多了一步「查询改写」，支持多轮追问
    4. 对外只暴露普通函数，命令行、网页、单元自检都能直接调用

数据库文件默认放在 data/app.db，首次启动自动建目录、建表并写入 3 篇制度文档。

自检：
    python rag_service.py
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

import config

# ---------------------------------------------------------------------------
# 1. 常量与初始数据
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "app.db"
TOP_K = 2

FALLBACK_ANSWER = "抱歉，知识库中没有找到与这个问题相关的资料，建议咨询相关部门或补充制度文档。"

# 首次启动写入的制度文档，原文来自 16_langgraph_rag.py 的 DOCS
SEED_DOCS: list[tuple[str, str]] = [
    (
        "报销",
        "云栖科技报销制度：差旅报销需在出差结束后 7 个工作日内提交，"
        "附发票原件与行程单。单笔超过 5000 元需部门负责人和财务总监双重审批。"
        "超过 30 天未提交的报销申请不再受理。",
    ),
    (
        "年假",
        "云栖科技年假制度：入职满一年享 12 天带薪年假，满三年 15 天，满五年 20 天。"
        "年假需提前 3 个工作日在内部系统申请，由直属主管审批。",
    ),
    (
        "远程办公",
        "云栖科技远程办公制度：每周三、周五可申请远程办公，每月最多 8 天。"
        "远程当天需保持在线并在日报中注明工作地点，试用期员工不适用。",
    ),
]

# 切块规则与 09/16 号脚本保持一致
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", ""],
)

# 模型只创建一次，config 里已做 lru_cache
MODEL = config.get_chat_model(temperature=0)


# ---------------------------------------------------------------------------
# 2. SQLite 基础
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    verdict TEXT,
    sources_json TEXT,
    elapsed_ms INTEGER,
    feedback INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@contextmanager
def _db():
    """每次操作新建连接，避免多线程共用同一个 sqlite 连接。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

"""_db和init_db函数为数据库获取链接和初始化函数"""
def init_db() -> None:
    """建目录、建表，并在文档表为空时写入初始制度文档。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db() as conn:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count == 0:
            now = _now()
            conn.executemany(
                "INSERT INTO documents(topic, content, created_at) VALUES (?, ?, ?)",
                [(topic, content, now) for topic, content in SEED_DOCS],
            )


# ---------------------------------------------------------------------------
# 3. 文档管理与向量索引
# ---------------------------------------------------------------------------
_INDEX_LOCK = threading.Lock()
_VECTOR_STORE: InMemoryVectorStore | None = None
_CHUNK_COUNT = 0


def list_documents() -> list[dict]:
    """返回全部文档，附带字数，方便前端展示。"""
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, topic, content, created_at FROM documents ORDER BY id"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "topic": row["topic"],
            "content": row["content"],
            "chars": len(row["content"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def add_document(topic: str, content: str) -> dict:
    """新增一篇文档，并重建向量索引。"""
    topic = (topic or "").strip()
    content = (content or "").strip()
    if not topic:
        raise ValueError("文档主题不能为空")
    if len(content) < 10:
        raise ValueError("文档正文太短，至少 10 个字")

    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO documents(topic, content, created_at) VALUES (?, ?, ?)",
            (topic, content, _now()),
        )
        doc_id = cur.lastrowid
    rebuild_index()
    return {"id": doc_id, "topic": topic, "content": content, "chars": len(content)}


def delete_document(doc_id: int) -> bool:
    """删除文档并重建索引，返回是否真的删掉了。"""
    with _db() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (int(doc_id),))
        deleted = cur.rowcount > 0
    if deleted:
        rebuild_index()
    return deleted


def rebuild_index() -> int:
    """把数据库里的文档重新切块、向量化，返回片段数量。"""
    global _VECTOR_STORE, _CHUNK_COUNT

    with _INDEX_LOCK:
        chunks: list[Document] = []
        for doc in list_documents():
            chunk_doc = Document(
                page_content=doc["content"],
                metadata={"topic": doc["topic"], "doc_id": doc["id"]},
            )
            chunks.extend(SPLITTER.split_documents([chunk_doc]))

        _CHUNK_COUNT = len(chunks)
        embeddings = config.get_embeddings()
        if chunks:
            _VECTOR_STORE = InMemoryVectorStore.from_documents(chunks, embeddings)
        else:
            _VECTOR_STORE = InMemoryVectorStore(embedding=embeddings)
    return _CHUNK_COUNT


def _ensure_index() -> InMemoryVectorStore:
    """索引懒加载：第一次检索或统计时才真正调用向量接口。"""
    if _VECTOR_STORE is None:
        rebuild_index()
    assert _VECTOR_STORE is not None
    return _VECTOR_STORE


def search_documents(query: str, k: int = TOP_K) -> list[dict]:
    """
    检索最相关的 k 个片段。

    注意：langchain_core 新版 InMemoryVectorStore 的 similarity_search_with_score
    返回的是余弦相似度（越大越相关），所以对外 score 直接取该值并归一到 0~1；
    旧版本返回距离时才需要用 1 - distance 换算。
    """
    store = _ensure_index()
    if _CHUNK_COUNT == 0:
        return []

    results = []
    for doc, raw_score in store.similarity_search_with_score(query, k=k):
        score = max(0.0, min(1.0, float(raw_score)))
        results.append(
            {
                "topic": doc.metadata.get("topic", "未知"),
                "content": doc.page_content,
                "score": round(score, 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# 4. RAG 图：rewrite -> retrieve -> grade -> generate / fallback
# ---------------------------------------------------------------------------
class RagState(TypedDict):
    question: str
    history: list
    standalone: str
    context: str
    verdict: str
    answer: str
    sources: list
    max_score: float


def _node_rewrite(state: RagState) -> dict:
    """把带指代的追问改写成独立问题；没有历史就直接用原问题。"""
    question = state.get("question", "")
    history = state.get("history") or []
    if not history:
        return {"standalone": question}

    lines = []
    for item in history[-6:]:
        role = "用户" if item.get("role") == "user" else "助手"
        content = (item.get("content") or "").strip()
        if content:
            # 历史回答可能很长，截断一下，避免改写请求过大
            lines.append(f"{role}：{content[:200]}")

    prompt = (
        "下面是一段对话历史。请把用户最新的问题改写成一个不依赖上下文、"
        "单独拿出来也能看懂的完整问题。\n"
        "要求：只输出改写后的问题本身，不要解释，不要加引号。\n\n"
        f"对话历史：\n" + "\n".join(lines) + "\n\n"
        f"最新问题：{question}\n"
        "改写后的问题："
    )
    try:
        rewritten = MODEL.invoke(prompt).content.strip()
    except Exception:
        rewritten = ""
    return {"standalone": rewritten or question}


def _node_retrieve(state: RagState) -> dict:
    """取 top-k 片段，拼上下文，同时把出处带上（含相似度）。"""
    query = state.get("standalone") or state.get("question", "")
    sources = search_documents(query, TOP_K)
    context = "\n\n".join(item["content"] for item in sources)
    max_score = max((item["score"] for item in sources), default=0.0)
    return {"context": context, "sources": sources, "max_score": max_score}


def _node_grade(state: RagState) -> dict:
    """让模型判断资料能否回答问题，输出只用于路由。"""
    context = state.get("context", "")
    if not context:
        return {"verdict": "不相关"}

    prompt = (
        "判断下面的资料能否回答用户问题。只输出一个词：相关 或 不相关。\n\n"
        f"资料：\n{context}\n\n问题：{state.get('standalone') or state.get('question', '')}"
    )
    raw = MODEL.invoke(prompt).content.strip()
    verdict = "相关" if "相关" in raw and "不相关" not in raw else "不相关"
    return {"verdict": verdict}


def _node_generate(state: RagState) -> dict:
    """只依据检索到的资料回答，资料没写到的内容明确说明。"""
    context = state.get("context", "")
    if not context:
        return {"answer": FALLBACK_ANSWER}

    prompt = (
        "只依据下面的资料回答，不要编造。资料未提及的内容直接说“资料中未提及”。"
        "语气简洁，能用一两句话说清就不要展开。\n\n"
        f"资料：\n{context}\n\n问题：{state.get('standalone') or state.get('question', '')}"
    )
    return {"answer": MODEL.invoke(prompt).content.strip()}


def _node_fallback(state: RagState) -> dict:
    return {"answer": FALLBACK_ANSWER}


def _route_after_grade(state: RagState) -> str:
    return "generate" if state.get("verdict") == "相关" else "fallback"


_GRAPH_LOCK = threading.Lock()
_GRAPH = None


def _build_graph():
    builder = StateGraph(RagState)
    builder.add_node("rewrite", _node_rewrite)
    builder.add_node("retrieve", _node_retrieve)
    builder.add_node("grade", _node_grade)
    builder.add_node("generate", _node_generate)
    builder.add_node("fallback", _node_fallback)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"generate": "generate", "fallback": "fallback"},
    )
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)
    return builder.compile()


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# 5. 会话与消息
# ---------------------------------------------------------------------------
def create_session(title: str | None = None) -> dict:
    """新建会话，返回会话信息。"""
    now = _now()
    session = {
        "id": uuid.uuid4().hex[:12],
        "title": (title or "新对话").strip()[:40],
        "created_at": now,
        "updated_at": now,
    }
    with _db() as conn:
        conn.execute(
            "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session["id"], session["title"], now, now),
        )
    return session


def list_sessions() -> list[dict]:
    """按最近更新时间倒序列出会话。"""
    with _db() as conn:
        rows = conn.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.role = 'user') AS msg_count "
            "FROM sessions s ORDER BY s.updated_at DESC"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "msg_count": row["msg_count"],
        }
        for row in rows
    ]


def get_session(session_id: str | None) -> dict | None:
    if not session_id:
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_session(session_id: str) -> bool:
    """删除会话及其全部消息。"""
    with _db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


def get_messages(session_id: str) -> list[dict]:
    """按时间顺序返回会话消息，出处从 JSON 还原。"""
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, session_id, role, content, verdict, sources_json, elapsed_ms, "
            "feedback, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    messages = []
    for row in rows:
        messages.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "verdict": row["verdict"],
                "sources": _loads(row["sources_json"]),
                "elapsed_ms": row["elapsed_ms"],
                "feedback": row["feedback"],
                "created_at": row["created_at"],
            }
        )
    return messages


def _loads(text: str | None) -> list:
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _insert_message(
    session_id: str,
    role: str,
    content: str,
    verdict: str | None = None,
    sources: list | None = None,
    elapsed_ms: int | None = None,
) -> int:
    """写一条消息，返回自增 id（前端用它提交反馈）。"""
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content, verdict, sources_json, "
            "elapsed_ms, feedback, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (
                session_id,
                role,
                content,
                verdict,
                json.dumps(sources or [], ensure_ascii=False),
                elapsed_ms,
                _now(),
            ),
        )
        return int(cur.lastrowid)


def _touch_session(session_id: str, title: str | None = None) -> None:
    """更新会话时间；如果给了新标题就一起改。"""
    with _db() as conn:
        if title:
            conn.execute(
                "UPDATE sessions SET updated_at = ?, title = ? WHERE id = ?",
                (_now(), title, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_now(), session_id),
            )


# ---------------------------------------------------------------------------
# 6. 问答主流程
# ---------------------------------------------------------------------------
def _suggestions(verdict: str, sources: list) -> list[str]:
    """用模板生成 3 条追问，不再额外调用模型。"""
    if verdict == "相关" and sources:
        topic = sources[0].get("topic") or "这项制度"
        return [
            f"{topic}的具体申请流程是什么？",
            f"{topic}制度还有哪些注意事项？",
            f"如果不符合{topic}的规定会怎样？",
        ]

    topics = [doc["topic"] for doc in list_documents()[:3]]
    if topics:
        return [f"请介绍一下{topic}制度" for topic in topics]
    return ["知识库里现在有哪些制度？", "报销需要在几天内提交？", "年假有几天？"]


def ask(question: str, session_id: str | None = None, history: list | None = None) -> dict:
    """
    执行一次制度问答。

    history 由前端传最近 6 条消息用于改写；没传时从数据库里取，
    保证命令行或接口直接调用也能正确解析「那需要提前几天申请」这类追问。
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")

    session = get_session(session_id)
    if session is None:
        session = create_session(title=question[:24])
    session_id = session["id"]

    if history:
        recent = [
            {"role": item.get("role"), "content": item.get("content")}
            for item in history[-6:]
            if isinstance(item, dict) and item.get("content")
        ]
    else:
        recent = [
            {"role": item["role"], "content": item["content"]}
            for item in get_messages(session_id)[-6:]
        ]

    _insert_message(session_id, "user", question)

    started = time.perf_counter()
    result = _get_graph().invoke(
        {
            "question": question,
            "history": recent,
            "standalone": "",
            "context": "",
            "verdict": "",
            "answer": "",
            "sources": [],
            "max_score": 0.0,
        }
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    verdict = result.get("verdict") or "不相关"
    answer = (result.get("answer") or "").strip() or FALLBACK_ANSWER
    sources = (result.get("sources") or []) if verdict == "相关" else []
    standalone = result.get("standalone") or question

    message_id = _insert_message(
        session_id, "assistant", answer, verdict, sources, elapsed_ms
    )
    _touch_session(session_id)

    return {
        "session_id": session_id,
        "message_id": message_id,
        "answer": answer,
        "verdict": verdict,
        "sources": sources,
        "standalone": standalone,
        "elapsed_ms": elapsed_ms,
        "suggestions": _suggestions(verdict, sources),
    }


def save_feedback(message_id: int, value: int) -> bool:
    """记录点赞 / 点踩，value 只能是 1 或 -1。"""
    if value not in (1, -1):
        raise ValueError("value 只能是 1 或 -1")
    with _db() as conn:
        cur = conn.execute(
            "UPDATE messages SET feedback = ? WHERE id = ?", (int(value), int(message_id))
        )
        return cur.rowcount > 0


def stats() -> dict:
    """统计面板数据：文档数、片段数、提问数、兜底率、平均耗时。"""
    _ensure_index()
    with _db() as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_questions = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role = 'user'"
        ).fetchone()[0]
        fallback_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role = 'assistant' AND verdict IS NOT NULL "
            "AND verdict != '相关'"
        ).fetchone()[0]
        avg_elapsed = conn.execute(
            "SELECT AVG(elapsed_ms) FROM messages WHERE role = 'assistant' "
            "AND elapsed_ms IS NOT NULL"
        ).fetchone()[0]

    return {
        "doc_count": doc_count,
        "chunk_count": _CHUNK_COUNT,
        "total_questions": total_questions,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total_questions, 4)
        if total_questions
        else 0.0,
        "avg_elapsed_ms": int(avg_elapsed) if avg_elapsed else 0,
    }


# ---------------------------------------------------------------------------
# 7. 初始化与自检
# ---------------------------------------------------------------------------
init_db()


if __name__ == "__main__":
    print("=" * 50)
    print("rag_service 自检")
    print("=" * 50)
    print(f"数据库：{DB_PATH}")
    print(f"文档数：{len(list_documents())}")
    print(f"片段数：{rebuild_index()}")

    demo = ask("年假有几天？")
    print(f"\n问题：年假有几天？")
    print(f"改写：{demo['standalone']}")
    print(f"判断：{demo['verdict']}")
    print(f"回答：{demo['answer']}")
    print(f"耗时：{demo['elapsed_ms']} ms")
    print("出处：", [f"{s['topic']}({s['score']})" for s in demo["sources"]])
    print("追问：", demo["suggestions"])
    print("统计：", stats())
