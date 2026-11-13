"""
统一模型接入层（所有复现脚本共用这一个文件）

为什么单独抽出来？
    原仓库每个案例都重复写 api_key / base_url / model 三件套，模型一换就要改十几个文件。
    这里把「模型接入」收敛成一个工厂函数：脚本里只需要 get_chat_model()，
    换厂商时只改 .env，不动任何案例代码。

支持的厂商（通过 .env 里的 REPRO_PROVIDER 切换）：
    qwen      阿里千问（当前默认：qwen-plus 对话 + text-embedding-v3 向量）
    deepseek  DeepSeek（deepseek-chat 对话；官方没有向量接口，RAG 自动借道千问）

.env 变量名（沿用原仓库的写法，带连字符也能读）：
    aliQwen-api        千问 Key，也兼容 DASHSCOPE_API_KEY
    deepseek-chat-api  DeepSeek Key，也兼容 deepseek-api / DEEPSEEK_API_KEY

关键点（对应原仓库第 11 章 Model I/O）：
    1. 国内平台几乎都提供「OpenAI 兼容接口」，所以统一用 ChatOpenAI 这个客户端。
    2. 调用三件套：api_key（身份）、base_url（地址）、model（用哪个模型）。
    3. 模型对象实现了 Runnable 接口，因此可以 invoke / stream / batch，
       也可以直接参与 LCEL 管道（prompt | model | parser）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------------------------------------------------------------------------
# 1. 读取 .env
# ---------------------------------------------------------------------------
# 注意：原仓库的变量名带连字符（aliQwen-api、deepseek-api），
# os.getenv("aliQwen-api") 虽然能取到，但 shell 里无法 export。
# 这里用 dotenv_values 解析成一个普通 dict，兼容两种来源：
#   优先级：进程环境变量 > 当前目录 .env > 脚本同级目录 .env
_ENV_PATH = Path(__file__).with_name(".env")
_ENV_FILE = dotenv_values(_ENV_PATH, encoding="utf-8") if _ENV_PATH.exists() else {}


def env(*names: str, default: str | None = None) -> str | None:
    """按顺序取第一个非空环境变量，顺带去掉可能残留的引号和空格。"""
    for name in names:
        value = os.getenv(name) or _ENV_FILE.get(name)
        if value and str(value).strip():
            return str(value).strip().strip("'\"")
    return default


# ---------------------------------------------------------------------------
# 2. 厂商配置表
# ---------------------------------------------------------------------------
# 每个厂商只需要四个信息：Key、Base URL、对话模型名、向量模型名。
_PROVIDERS: dict[str, dict[str, str | None]] = {
    "qwen": {
        # 千问：阿里云百炼（DashScope）的 OpenAI 兼容接口
        "api_key": env("aliQwen-api", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # qwen-plus：通用对话模型，支持工具调用，速度和效果比较均衡
        # 想更便宜/更快可换 qwen-turbo；想更强可换 qwen-max
        "chat_model": env("REPRO_QWEN_MODEL", "REPRO_CHAT_MODEL", default="qwen-plus"),
        # text-embedding-v3：千问的向量模型（默认 1024 维），RAG 案例使用
        "embedding_model": env("REPRO_QWEN_EMBED_MODEL", "REPRO_EMBED_MODEL", default="text-embedding-v3"),
    },
    "deepseek": {
        # 兼容三种写法：.env 里的 deepseek-chat-api / deepseek-api，以及标准的 DEEPSEEK_API_KEY
        "api_key": env("deepseek-chat-api", "deepseek-api", "DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com/v1",
        # deepseek-chat：通用对话模型，支持工具调用
        "chat_model": env("REPRO_DEEPSEEK_MODEL", "REPRO_CHAT_MODEL", default="deepseek-chat"),
        # DeepSeek 官方没有 embedding 接口，向量模型借道千问
        "embedding_model": None,
        "embedding_provider": "qwen",
    },
}

# 默认厂商：千问（有免费额度）。想切 DeepSeek 就在 .env 里写 REPRO_PROVIDER=deepseek
PROVIDER = (env("REPRO_PROVIDER", default="qwen") or "qwen").lower()
if PROVIDER not in _PROVIDERS:
    raise ValueError(f"未知的 REPRO_PROVIDER={PROVIDER!r}，可选：{list(_PROVIDERS)}")


# ---------------------------------------------------------------------------
# 3. 工厂函数
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_chat_model(temperature: float = 0.3, max_tokens: int | None = None) -> ChatOpenAI:
    """
    返回一个聊天模型对象（Runnable）。

    temperature 越低越稳定，做结构化输出/工具调用时建议 0~0.3；
    写文案类任务可以调到 0.7 以上。
    """
    cfg = _PROVIDERS[PROVIDER]
    if not cfg["api_key"]:
        raise RuntimeError(
            f"没有找到 {PROVIDER} 的 API Key。\n"
            f"请在 {_ENV_PATH} 中填写，或改成可用的 REPRO_PROVIDER。"
        )

    kwargs: dict = {
        "model": cfg["chat_model"],
        "api_key": cfg["api_key"],
        "base_url": cfg["base_url"],
        "temperature": temperature,
        "timeout": 90,
        # 免费模型有 RPM 限制，遇到 429 时让客户端自动退避重试几次
        "max_retries": 4,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def _embedding_cfg() -> dict[str, str | None]:
    """
    取向量模型配置。

    DeepSeek 没有 embedding 接口，所以它的配置里 embedding_model 是 None，
    这里会自动切到 embedding_provider 指定的厂商（默认千问）。
    """
    cfg = _PROVIDERS[PROVIDER]
    if not cfg.get("embedding_model"):
        cfg = _PROVIDERS[cfg.get("embedding_provider") or "qwen"]
    return cfg


@lru_cache(maxsize=None)
def get_embeddings() -> OpenAIEmbeddings:
    """返回向量模型（用于 Embedding / RAG 案例）。"""
    cfg = _embedding_cfg()
    if not cfg["api_key"]:
        raise RuntimeError(
            f"没有找到向量模型所需的 API Key。\n"
            f"当前对话厂商是 {PROVIDER}，但 RAG 案例需要千问的向量模型，"
            f"请在 {_ENV_PATH} 中填写 aliQwen-api（千问 Key）。"
        )
    return OpenAIEmbeddings(
        model=cfg["embedding_model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=60,
        # 千问的 embedding 接口单次最多 10 条文本，分批发送更稳
        chunk_size=10,
        # 关键：默认会先用 tiktoken 把文本切成 token id 再发送，
        # 百炼只接受原始字符串，否则报 400 InvalidParameter。
        check_embedding_ctx_length=False,
    )


def describe() -> str:
    """打印当前配置（不泄露 Key），排查问题时很有用。"""
    cfg = _PROVIDERS[PROVIDER]
    embed_cfg = _embedding_cfg()
    embed_provider = PROVIDER if cfg.get("embedding_model") else (cfg.get("embedding_provider") or "qwen")
    key = cfg["api_key"] or ""
    masked = f"已配置（{len(key)} 字符）" if key else "（未配置）"
    embed_desc = str(embed_cfg["embedding_model"])
    if embed_provider != PROVIDER:
        embed_desc += f"（由 {embed_provider} 提供）"
    return (
        f"provider={PROVIDER}\n"
        f"chat_model={cfg['chat_model']}\n"
        f"embedding_model={embed_desc}\n"
        f"base_url={cfg['base_url']}\n"
        f"api_key={masked}"
    )


if __name__ == "__main__":
    # 直接运行本文件即可自检：打印配置并做一次最小调用
    print(describe())
    print("-" * 50)
    reply = get_chat_model().invoke("你好，请用一句话介绍你自己。")
    print("模型回复：", reply.content)
