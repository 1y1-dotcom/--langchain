"""
18 · Web 版产品入口：校园制度问答助手 + 自由对话

用 Python 标准库 http.server 起服务，不依赖 Flask / Streamlit：
    python web_app.py
浏览器打开 http://127.0.0.1:8000

页面能力：
    制度问答：查询改写 + 本地检索 + 相关性判断 + 本地与联网融合/固定兜底
    自由对话：直接调用大模型，不检索、不入库
    左侧栏：会话列表、知识库增删、统计面板
"""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
import rag_service

# 自由对话用的模型：温度稍高一点，语气更自然
CHAT_MODEL = config.get_chat_model(temperature=0.5)
CHAT_SYSTEM = (
    "你是一个友好的中文助手，回答简洁、准确、有条理。"
    "遇到不确定的信息要说明不确定，不要编造。"
)

# 自由对话保留的轮数：一轮 = 用户 + 助手两条消息，所以实际最多 12 条
MEMORY_ROUNDS = 6

# 启动时收集一次展示信息（不发起网络请求）
INFO = {
    "provider": config.PROVIDER,
    "chat_model": rag_service.MODEL.model_name,
    "embedding_model": config.get_embeddings().model,
    "top_k": rag_service.TOP_K,
}

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>校园制度问答助手</title>
<style>
  :root {
    --ink: #0f172a;
    --ink-2: #1e293b;
    --muted: #64748b;
    --line: #e2e8f0;
    --bg: #f6f8fc;
    --brand: #2563eb;
    --ok: #059669;
    --ok-bg: #d1fae5;
    --warn: #b45309;
    --warn-bg: #fef3c7;
    --sidebar: #0b1220;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  }
  .app {
    display: grid;
    grid-template-columns: 330px 1fr;
    height: 100vh;
    overflow: hidden;
  }
  .hidden { display: none !important; }

  /* ---------- 左侧栏 ---------- */
  .sidebar {
    background: radial-gradient(1200px 400px at -20% -10%, #1e3a8a 0%, transparent 55%), var(--sidebar);
    color: #e2e8f0;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow-y: auto;
  }
  .brand { display: flex; gap: 12px; align-items: center; }
  .logo {
    width: 42px; height: 42px; border-radius: 12px; flex: 0 0 42px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    display: grid; place-items: center;
    font-size: 20px; font-weight: 700; color: #fff;
    box-shadow: 0 8px 20px rgba(37, 99, 235, .35);
  }
  .brand h1 { font-size: 16px; margin: 0 0 3px; letter-spacing: .5px; }
  .brand p { margin: 0; font-size: 11px; color: #94a3b8; }

  .status {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: #a7f3d0;
    background: rgba(16, 185, 129, .08);
    border: 1px solid rgba(16, 185, 129, .25);
    padding: 8px 12px; border-radius: 10px;
  }
  .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #10b981; box-shadow: 0 0 0 3px rgba(16, 185, 129, .18);
  }

  .primary {
    width: 100%; border: 0; border-radius: 10px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: #fff; font-size: 13px; font-weight: 600;
    padding: 10px 12px; cursor: pointer;
  }
  .primary:hover { filter: brightness(1.06); }

  .panel {
    background: rgba(255, 255, 255, .035);
    border: 1px solid rgba(148, 163, 184, .14);
    border-radius: 12px;
    padding: 12px;
  }
  .panel h2 {
    display: flex; align-items: center; justify-content: space-between;
    font-size: 12px; color: #94a3b8; margin: 0 0 10px;
    font-weight: 600; letter-spacing: 1px;
  }
  .panel h2 em { font-style: normal; color: #64748b; font-weight: 400; letter-spacing: 0; }
  .muted { margin: 0; font-size: 11.5px; color: #94a3b8; line-height: 1.7; }

  .session-list { display: grid; gap: 6px; max-height: 148px; overflow-y: auto; }
  .session {
    display: flex; align-items: center; gap: 8px;
    background: rgba(15, 23, 42, .5);
    border: 1px solid rgba(148, 163, 184, .12);
    border-radius: 8px; padding: 7px 9px; cursor: pointer;
  }
  .session:hover { border-color: rgba(96, 165, 250, .5); }
  .session.active { border-color: #3b82f6; background: rgba(37, 99, 235, .18); }
  .s-main { flex: 1; min-width: 0; }
  .s-main b {
    display: block; font-size: 12px; color: #e2e8f0; font-weight: 600;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .s-main span { font-size: 10.5px; color: #94a3b8; }
  .s-del {
    background: none; border: 0; color: #64748b;
    font-size: 15px; line-height: 1; padding: 0 2px; cursor: pointer;
  }
  .s-del:hover { color: #f87171; }

  .doc-list { display: grid; gap: 7px; max-height: 168px; overflow-y: auto; margin-bottom: 8px; }
  .doc {
    background: rgba(15, 23, 42, .5);
    border: 1px solid rgba(148, 163, 184, .12);
    border-radius: 8px; padding: 8px 10px;
  }
  .doc-head { display: flex; align-items: center; gap: 6px; }
  .doc-head b { font-size: 12px; color: #dbeafe; }
  .doc-head span { font-size: 10.5px; color: #64748b; }
  .doc-del {
    margin-left: auto; background: none; border: 0;
    color: #64748b; font-size: 14px; line-height: 1; cursor: pointer; padding: 0;
  }
  .doc-del:hover { color: #f87171; }
  .doc p {
    margin: 4px 0 0; font-size: 11px; color: #94a3b8; line-height: 1.6;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  .ghost-btn {
    width: 100%; background: rgba(255, 255, 255, .06);
    border: 1px dashed rgba(148, 163, 184, .35);
    color: #cbd5e1; font-size: 12px; padding: 7px;
    border-radius: 8px; cursor: pointer;
  }
  .ghost-btn:hover { border-color: #60a5fa; color: #dbeafe; }
  .ghost-btn:disabled { opacity: .55; cursor: not-allowed; }
  .rag-settings { margin: 10px 0; padding-top: 10px; border-top: 1px solid rgba(148, 163, 184, .14); }
  .setting-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .setting-field label { display: block; margin-bottom: 4px; font-size: 10.5px; color: #94a3b8; }
  .setting-field input {
    width: 100%; padding: 6px 8px; border-radius: 7px;
    border: 1px solid rgba(148, 163, 184, .3); background: rgba(15, 23, 42, .6);
    color: #e2e8f0; font-size: 12px; outline: none;
  }
  .setting-field input:focus { border-color: #60a5fa; }
  .settings-note { margin: 6px 0 8px; font-size: 10.5px; color: #64748b; line-height: 1.6; }

  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat {
    background: rgba(15, 23, 42, .5);
    border: 1px solid rgba(148, 163, 184, .12);
    border-radius: 8px; padding: 8px 10px;
  }
  .stat span { display: block; font-size: 10.5px; color: #94a3b8; }
  .stat b { font-size: 14px; color: #e2e8f0; }

  .side-foot { margin-top: auto; font-size: 10.5px; color: #475569; line-height: 1.8; }

  /* ---------- 主区 ---------- */
  .main { display: flex; flex-direction: column; min-width: 0; min-height: 0; background: var(--bg); }
  .topbar {
    height: 60px; flex: 0 0 60px;
    display: flex; align-items: center; gap: 16px;
    padding: 0 24px;
    background: rgba(255, 255, 255, .8);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
  }
  .tabs { display: flex; gap: 4px; padding: 3px; background: #eef2f7; border-radius: 10px; }
  .tab {
    border: 0; background: transparent; color: var(--muted);
    font-size: 12.5px; padding: 7px 14px; border-radius: 8px; cursor: pointer;
  }
  .tab.active {
    background: #fff; color: var(--brand); font-weight: 600;
    box-shadow: 0 1px 4px rgba(15, 23, 42, .08);
  }
  .current {
    margin-left: auto; font-size: 12px; color: var(--muted);
    max-width: 46%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .ghost {
    background: #fff; border: 1px solid var(--line); color: var(--muted);
    font-size: 12px; padding: 6px 12px; border-radius: 8px; cursor: pointer;
  }
  .ghost:hover { border-color: #bfdbfe; color: var(--brand); }

  .chat {
    flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain;
    -webkit-overflow-scrolling: touch;
    padding: 26px 24px 10px; scroll-behavior: smooth;
  }
  .chat-inner { max-width: 800px; margin: 0 auto; display: grid; gap: 18px; }

  .welcome { text-align: center; padding: 32px 10px 10px; }
  .welcome h2 { font-size: 22px; margin: 0 0 8px; }
  .welcome p { color: var(--muted); font-size: 13.5px; margin: 0 0 24px; line-height: 1.8; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; text-align: left; }
  .card-q {
    background: #fff; border: 1px solid var(--line); border-radius: 12px;
    padding: 14px 16px; cursor: pointer; transition: .18s;
  }
  .card-q:hover {
    border-color: #93c5fd; transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(37, 99, 235, .08);
  }
  .card-q b { display: block; font-size: 13px; margin-bottom: 4px; color: var(--ink-2); }
  .card-q span { font-size: 12px; color: var(--muted); }

  .msg { display: flex; gap: 10px; animation: rise .25s ease; }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  .msg .avatar {
    width: 30px; height: 30px; border-radius: 9px; flex: 0 0 30px;
    display: grid; place-items: center; font-size: 13px; font-weight: 700; color: #fff;
  }
  .msg.user { flex-direction: row-reverse; }
  .msg.user .avatar { background: #0f172a; }
  .msg.bot .avatar { background: linear-gradient(135deg, #2563eb, #06b6d4); }
  .bubble {
    max-width: 88%; background: #fff; border: 1px solid var(--line);
    border-radius: 14px; padding: 14px 16px; font-size: 14px; line-height: 1.85;
    box-shadow: 0 2px 8px rgba(15, 23, 42, .03);
  }
  .msg.user .bubble { background: #eff6ff; border-color: #dbeafe; }
  .bubble .text { white-space: pre-wrap; word-break: break-word; }

  .meta { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; flex-wrap: wrap; }
  .badge { font-size: 11px; padding: 3px 9px; border-radius: 999px; }
  .badge.ok { background: var(--ok-bg); color: var(--ok); }
  .badge.warn { background: var(--warn-bg); color: var(--warn); }
  .badge.free { background: #e0f2fe; color: #0369a1; }
  .badge.web { background: #e0e7ff; color: #4f46e5; }
  .time { font-size: 11px; color: #94a3b8; margin-left: auto; }

  .steps { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }
  .step {
    font-size: 11px; color: #475569; background: #f1f5f9;
    border-radius: 6px; padding: 4px 8px; max-width: 100%;
  }
  .step b { color: var(--brand); font-weight: 600; }

  .sources { margin-top: 12px; }
  .sources > h3 {
    font-size: 11.5px; color: var(--muted); margin: 0 0 7px;
    font-weight: 600; letter-spacing: .5px;
  }
  .src {
    background: #f8fafc; border: 1px solid #eef2f7;
    border-left: 3px solid #93c5fd; border-radius: 0 8px 8px 0;
    padding: 9px 11px; margin-bottom: 7px;
  }
  .src-head { display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
  .src-head b { color: var(--brand); }
  .src-head .score {
    margin-left: auto; font-size: 11px; color: #0369a1;
    background: #e0f2fe; border-radius: 999px; padding: 1px 7px;
  }
  .bar { height: 4px; border-radius: 999px; background: #e2e8f0; margin: 6px 0; overflow: hidden; }
  .bar i { display: block; height: 100%; background: linear-gradient(90deg, #2563eb, #06b6d4); }
  .src p { margin: 4px 0 0; font-size: 12.5px; color: #475569; line-height: 1.75; }
  .src-link { color: #2563eb; text-decoration: none; word-break: break-all; }
  .src-link:hover { text-decoration: underline; }

  .feedback {
    display: flex; align-items: center; gap: 8px;
    margin-top: 11px; font-size: 11.5px; color: #94a3b8;
  }
  .feedback button {
    background: #f8fafc; border: 1px solid var(--line); border-radius: 8px;
    padding: 3px 9px; cursor: pointer; font-size: 12px;
  }
  .feedback button:hover { border-color: #93c5fd; }
  .feedback button.on { border-color: #2563eb; background: #eff6ff; }
  .fb-note { font-size: 11px; color: var(--ok); }

  .suggests { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
  .chip {
    background: #eff6ff; border: 1px solid #dbeafe; color: #1d4ed8;
    font-size: 11.5px; padding: 5px 11px; border-radius: 999px; cursor: pointer;
  }
  .chip:hover { background: #dbeafe; }

  .tools { display: flex; gap: 8px; margin-top: 10px; }
  .tools button {
    background: none; border: 0; padding: 0; cursor: pointer;
    font-size: 11.5px; color: #94a3b8;
  }
  .tools button:hover { color: var(--brand); }

  .typing { display: inline-flex; gap: 4px; padding: 4px 2px; }
  .typing i {
    width: 6px; height: 6px; border-radius: 50%; background: #94a3b8;
    animation: blink 1.2s infinite ease-in-out;
  }
  .typing i:nth-child(2) { animation-delay: .2s; }
  .typing i:nth-child(3) { animation-delay: .4s; }
  @keyframes blink { 0%, 60%, 100% { opacity: .25; } 30% { opacity: 1; } }

  .composer-wrap { padding: 10px 24px 18px; background: linear-gradient(transparent, var(--bg) 30%); }
  .quick { max-width: 800px; margin: 0 auto 9px; display: flex; gap: 8px; flex-wrap: wrap; }
  .quick button {
    background: #fff; border: 1px solid var(--line); color: var(--muted);
    font-size: 12px; padding: 6px 13px; border-radius: 999px; cursor: pointer;
  }
  .quick button:hover { border-color: #93c5fd; color: var(--brand); }
  .quick button.hot { border-color: #fcd34d; color: var(--warn); }
  .composer {
    max-width: 800px; margin: 0 auto;
    background: #fff; border: 1px solid var(--line); border-radius: 14px;
    padding: 10px 10px 10px 16px;
    display: flex; align-items: flex-end; gap: 10px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
  }
  .composer:focus-within { border-color: #93c5fd; box-shadow: 0 8px 24px rgba(37, 99, 235, .1); }
  textarea {
    flex: 1; border: 0; outline: 0; resize: none;
    font-family: inherit; font-size: 14px; line-height: 1.6;
    max-height: 120px; padding: 6px 0; background: transparent; color: var(--ink);
  }
  #ask {
    background: linear-gradient(135deg, var(--brand), #06b6d4);
    color: #fff; border: 0; border-radius: 10px;
    padding: 9px 18px; font-size: 13px; cursor: pointer; white-space: nowrap;
  }
  #ask:disabled { opacity: .55; cursor: not-allowed; }
  .composer-hint {
    max-width: 800px; margin: 8px auto 0;
    font-size: 11px; color: #94a3b8; text-align: center;
  }
  .error { color: #b91c1c; }

  /* ---------- 弹窗 / 提示 ---------- */
  .modal {
    position: fixed; inset: 0; z-index: 50;
    background: rgba(15, 23, 42, .45);
    display: grid; place-items: center; padding: 20px;
  }
  .modal-card {
    width: min(520px, 100%); background: #fff; border-radius: 14px;
    padding: 20px; box-shadow: 0 24px 60px rgba(15, 23, 42, .25);
  }
  .modal-card h3 { margin: 0 0 14px; font-size: 16px; }
  .field { margin-bottom: 12px; }
  .field label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
  .field input, .field textarea {
    width: 100%; border: 1px solid var(--line); border-radius: 9px;
    padding: 9px 11px; font-family: inherit; font-size: 13px;
    outline: none; color: var(--ink);
  }
  .field textarea { resize: vertical; min-height: 130px; line-height: 1.7; }
  .field input:focus, .field textarea:focus {
    border-color: #93c5fd; box-shadow: 0 0 0 3px rgba(37, 99, 235, .1);
  }
  .modal-error { font-size: 12px; color: #b91c1c; min-height: 16px; margin-bottom: 6px; }
  .modal-actions { display: flex; gap: 10px; justify-content: flex-end; }
  .btn {
    border: 1px solid var(--line); background: #fff; color: var(--muted);
    font-size: 13px; padding: 8px 16px; border-radius: 9px; cursor: pointer;
  }
  .btn.primary {
    width: auto; color: #fff; border: 0;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
  }

  .toast {
    position: fixed; top: 16px; left: 50%; z-index: 80;
    transform: translateX(-50%) translateY(-20px);
    opacity: 0; pointer-events: none;
    background: #0f172a; color: #f8fafc; font-size: 12.5px;
    padding: 10px 18px; border-radius: 10px; max-width: 80vw;
    box-shadow: 0 12px 30px rgba(15, 23, 42, .3);
    transition: .25s;
  }
  .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  .toast.ok { background: #065f46; }

  .menu-btn { display: none; }
  .scrim { display: none; }

  @media (max-width: 720px) {
    .app { grid-template-columns: 1fr; height: 100dvh; overflow: hidden; }
    .sidebar {
      position: fixed; top: 0; bottom: 0; left: 0; width: 84%; max-width: 330px;
      z-index: 30; overflow-y: auto; transform: translateX(-100%);
      transition: transform .25s ease; box-shadow: 12px 0 30px rgba(15, 23, 42, .18);
    }
    .sidebar.open { transform: translateX(0); }
    .scrim {
      display: block; position: fixed; inset: 0; z-index: 25;
      background: rgba(15, 23, 42, .45); opacity: 0; pointer-events: none;
      transition: opacity .2s ease;
    }
    .scrim.show { opacity: 1; pointer-events: auto; }
    .menu-btn {
      display: inline-flex; align-items: center; justify-content: center;
      width: 34px; height: 34px; flex: 0 0 34px; border: 1px solid var(--line);
      background: #fff; color: var(--ink); border-radius: 8px; font-size: 16px; cursor: pointer;
    }
    .main { min-height: 0; }
    .chat { padding: 18px 14px 8px; }
    .cards { grid-template-columns: 1fr; }
    .bubble { max-width: 92%; }
    .composer-wrap { padding: 8px 14px 14px; }
    .topbar { padding: 0 12px; gap: 8px; }
    .current { display: none; }
  }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="logo">制</div>
      <div>
        <h1>云栖制度助手</h1>
        <p>RAG 问答 · 自由对话</p>
      </div>
    </div>

    <div class="status"><span class="dot"></span><span id="status">连接中…</span></div>

    <button class="primary" id="new-session">+ 新建对话</button>

    <section class="panel">
      <h2>会话</h2>
      <div id="session-list" class="session-list"></div>
    </section>

    <section class="panel">
      <h2>知识库 <em id="doc-count">—</em></h2>
      <div id="docs" class="doc-list"></div>
      <div class="rag-settings">
        <div class="setting-row">
          <div class="setting-field">
            <label for="chunk-size">切块大小（50~500）</label>
            <input id="chunk-size" type="number" min="50" max="500" step="10" value="120">
          </div>
          <div class="setting-field">
            <label for="chunk-overlap">重叠字符（0~100）</label>
            <input id="chunk-overlap" type="number" min="0" max="100" step="5" value="20">
          </div>
        </div>
        <p class="settings-note" id="settings-note">重叠字符必须小于切块大小，修改后需重建索引。</p>
        <button class="ghost-btn" id="apply-rag-settings">应用并重建索引</button>
      </div>
      <button class="ghost-btn" id="add-doc">+ 添加文档</button>
    </section>

    <section class="panel">
      <h2>统计</h2>
      <div class="stats">
        <div class="stat"><span>提问数</span><b id="stat-ask">—</b></div>
        <div class="stat"><span>兜底率</span><b id="stat-fb">—</b></div>
        <div class="stat"><span>平均耗时</span><b id="stat-ms">—</b></div>
        <div class="stat"><span>文档 / 片段</span><b id="stat-docs">—</b></div>
      </div>
    </section>

    <div class="side-foot">
      模型：<span id="model-name">—</span><br>
      向量：<span id="embed-name">—</span><br>
      检索：<span id="topk">k=2</span> · 本地制度优先 · 每次联网融合补充
    </div>
  </aside>
  <div class="scrim" id="scrim"></div>

  <main class="main">
    <div class="topbar">
      <button class="menu-btn" id="menu-btn" aria-label="打开侧栏">☰</button>
      <div class="tabs">
        <button class="tab active" data-mode="rag">制度问答</button>
        <button class="tab" data-mode="free">自由对话</button>
      </div>
      <span class="current" id="current-session">新对话</span>
      <button class="ghost" id="clear">清空对话</button>
    </div>

    <!-- 制度问答 -->
    <div class="chat" id="chat-rag">
      <div class="chat-inner" id="rag-inner">
        <div class="welcome" id="rag-welcome">
          <h2>有什么制度问题，直接问我</h2>
          <p>我会先改写问题、检索制度文档，再结合公开网络补充背景。<br>本地制度始终优先，联网失败时仍保留本地答案。</p>
          <div class="cards" id="rag-cards"></div>
        </div>
      </div>
    </div>

    <!-- 自由对话 -->
    <div class="chat hidden" id="chat-free">
      <div class="chat-inner" id="free-inner">
        <div class="welcome" id="free-welcome">
          <h2>随便聊点什么</h2>
          <p>这里不查知识库，直接由大模型回答。<br>可以写作、翻译、解释概念，也能接着上一句继续问。</p>
          <div class="cards" id="free-cards"></div>
        </div>
      </div>
    </div>

    <div class="composer-wrap">
      <div class="quick" id="quick"></div>
      <form class="composer" id="form">
        <textarea id="question" rows="1" placeholder="输入你的问题…"></textarea>
        <button type="submit" id="ask">发送</button>
      </form>
      <div class="composer-hint" id="hint">Enter 发送 · Shift + Enter 换行 · 本地制度优先 · 无资料时联网补充</div>
    </div>
  </main>
</div>

<!-- 添加文档弹窗 -->
<div class="modal hidden" id="modal">
  <div class="modal-card">
    <h3>添加知识库文档</h3>
    <div class="field">
      <label for="modal-topic">主题（如：门禁管理）</label>
      <input id="modal-topic" maxlength="40" placeholder="一句话概括这篇文档讲什么">
    </div>
    <div class="field">
      <label for="modal-content">正文</label>
      <textarea id="modal-content" placeholder="粘贴制度原文，至少 10 个字"></textarea>
    </div>
    <div class="modal-error" id="modal-error"></div>
    <div class="modal-actions">
      <button class="btn" id="modal-cancel">取消</button>
      <button class="btn primary" id="modal-submit">保存并重建索引</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
  const RAG_EXAMPLES = [
    { title: '报销制度', q: '出差报销需要在几天内提交？超过 5000 元要谁审批？', hint: '差旅 · 审批流程' },
    { title: '年假制度', q: '入职满三年有几天年假？需要提前几天申请？', hint: '假期 · 申请规则' },
    { title: '远程办公', q: '每周可以远程办公几天？试用期员工适用吗？', hint: '办公 · 适用人群' },
    { title: '联网补充', q: 'LangChain 官方文档的入口是什么？', hint: '本地无资料时联网检索' }
  ];
  const FREE_EXAMPLES = [
    { title: '解释概念', q: '用三句话解释什么是 RAG，并举一个例子。', hint: '学习 · 概念解释' },
    { title: '框架对比', q: 'LangChain 和 LangGraph 的区别是什么？', hint: '技术 · 对比分析' },
    { title: '写邮件', q: '帮我写一封请假邮件，理由是发烧，语气正式一点。', hint: '写作 · 邮件' },
    { title: '翻译', q: '把这句话翻译成英文：今天天气很好，我们下午去打球。', hint: '翻译 · 中译英' }
  ];

  const $ = id => document.getElementById(id);
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  const ragChat = $('chat-rag'), freeChat = $('chat-free');
  const ragInner = $('rag-inner'), freeInner = $('free-inner');
  const input = $('question'), askBtn = $('ask'), quick = $('quick'), hint = $('hint');

  let mode = 'rag';
  let busy = false;
  let sessionId = null;
  let sessions = [];
  let ragHistory = [];    // 当前会话的制度问答历史，用于查询改写
  let freeHistory = [];   // 自由对话历史，只存在浏览器内存里
  let savedRagSettings = { chunk_size: 120, chunk_overlap: 20 };

  // ---------- 网络请求统一封装 ----------
  async function api(path, options) {
    let resp;
    try {
      resp = await fetch(path, options || {});
    } catch (e) {
      throw new Error('网络连接失败，请确认服务已启动');
    }
    let data = null;
    try { data = await resp.json(); } catch (e) { data = null; }
    if (!resp.ok) {
      throw new Error((data && data.error) || ('请求失败（HTTP ' + resp.status + '）'));
    }
    return data;
  }

  // ---------- 顶部提示 ----------
  let toastTimer = null;
  function toast(message, kind) {
    const el = $('toast');
    el.textContent = message;
    el.className = 'toast show' + (kind === 'ok' ? ' ok' : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = 'toast'; }, 4200);
  }

  function fmtTime(text) {
    return String(text || '').slice(5, 16);
  }

  // ---------- 初始化：服务信息 / 文档 / 统计 / 会话 ----------
  async function loadInfo() {
    const info = await api('/api/info');
    $('status').textContent = '服务正常 · ' + info.chat_model;
    $('model-name').textContent = info.chat_model;
    $('embed-name').textContent = info.embedding_model;
    $('topk').textContent = 'k=' + info.top_k;
  }

  function fillRagSettings(data) {
    savedRagSettings = {
      chunk_size: Number(data.chunk_size),
      chunk_overlap: Number(data.chunk_overlap)
    };
    $('chunk-size').value = savedRagSettings.chunk_size;
    $('chunk-overlap').value = savedRagSettings.chunk_overlap;
    if (data.limits) {
      $('chunk-size').min = data.limits.chunk_size.min;
      $('chunk-size').max = data.limits.chunk_size.max;
      $('chunk-overlap').min = data.limits.chunk_overlap.min;
      $('chunk-overlap').max = data.limits.chunk_overlap.max;
    }
    if (Number.isInteger(data.chunk_count)) {
      $('settings-note').textContent = '当前 ' + data.chunk_size + ' / ' + data.chunk_overlap +
        ' · 共 ' + data.chunk_count + ' 段 · 重叠字符必须小于切块大小';
    }
  }

  async function loadRagSettings() {
    fillRagSettings(await api('/api/rag-settings'));
  }

  async function loadDocs() {
    const docs = await api('/api/docs');
    $('doc-count').textContent = docs.length + ' 篇';
    const box = $('docs');
    if (!docs.length) {
      box.innerHTML = '<p class="muted">知识库为空，请添加文档</p>';
      return;
    }
    box.innerHTML = docs.map(d => (
      '<div class="doc">' +
        '<div class="doc-head"><b>' + esc(d.topic) + '</b><span>' + d.chars + ' 字</span>' +
          '<button class="doc-del" data-doc="' + d.id + '" title="删除这篇文档">×</button></div>' +
        '<p>' + esc(d.content) + '</p>' +
      '</div>'
    )).join('');
    box.querySelectorAll('[data-doc]').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!window.confirm('确定删除这篇文档吗？删除后会重建索引。')) return;
        try {
          await api('/api/docs/' + btn.dataset.doc, { method: 'DELETE' });
          await loadDocs();
          await loadStats();
          toast('文档已删除，索引已重建', 'ok');
        } catch (e) {
          toast('删除失败：' + e.message);
        }
      });
    });
  }

  async function loadStats() {
    const s = await api('/api/stats');
    $('stat-ask').textContent = s.total_questions;
    $('stat-fb').textContent = Math.round(s.fallback_rate * 100) + '%';
    $('stat-ms').textContent = s.avg_elapsed_ms + ' ms';
    $('stat-docs').textContent = s.doc_count + ' / ' + s.chunk_count;
  }

  function renderSessions() {
    const box = $('session-list');
    if (!sessions.length) {
      box.innerHTML = '<p class="muted">还没有会话，提问会自动创建</p>';
      return;
    }
    box.innerHTML = sessions.map(s => (
      '<div class="session ' + (s.id === sessionId ? 'active' : '') + '" data-sid="' + s.id + '">' +
        '<div class="s-main"><b>' + esc(s.title) + '</b>' +
          '<span>' + esc(fmtTime(s.updated_at)) + ' · ' + s.msg_count + ' 问</span></div>' +
        '<button class="s-del" data-del="' + s.id + '" title="删除会话">×</button>' +
      '</div>'
    )).join('');

    box.querySelectorAll('.session').forEach(el => {
      el.addEventListener('click', async () => {
        if (el.dataset.sid === sessionId) return;
        try {
          await selectSession(el.dataset.sid);
        } catch (e) {
          toast('加载会话失败：' + e.message);
        }
      });
    });
    box.querySelectorAll('[data-del]').forEach(btn => {
      btn.addEventListener('click', async ev => {
        ev.stopPropagation();
        if (!window.confirm('删除这个会话及全部消息？')) return;
        try {
          await api('/api/sessions/' + btn.dataset.del, { method: 'DELETE' });
          if (btn.dataset.del === sessionId) {
            sessionId = null;
            ragHistory = [];
            clearRagChat();
            $('current-session').textContent = '新对话';
          }
          await loadSessions(false);
          toast('会话已删除', 'ok');
        } catch (e) {
          toast('删除失败：' + e.message);
        }
      });
    });
  }

  async function loadSessions(selectFirst) {
    sessions = await api('/api/sessions');
    renderSessions();
    if (!sessionId && selectFirst && sessions.length) {
      await selectSession(sessions[0].id);
    }
  }

  async function selectSession(id) {
    const msgs = await api('/api/sessions/' + id + '/messages');
    sessionId = id;
    const found = sessions.find(s => s.id === id);
    $('current-session').textContent = found ? found.title : '对话';
    ragHistory = [];
    clearRagChat();
    msgs.forEach(m => {
      if (m.role === 'user') {
        addUser(m.content, ragInner, ragChat);
      } else {
        addRagBot(
          { answer: m.content, verdict: m.verdict || '不相关', answer_source: m.answer_source, sources: m.sources || [], elapsed_ms: m.elapsed_ms || 0, suggestions: [] },
          { message_id: m.id, feedback: m.feedback }
        );
      }
      ragHistory.push({ role: m.role, content: m.content });
    });
    if (ragHistory.length > 12) ragHistory = ragHistory.slice(-12);
    if (msgs.length) $('rag-welcome').style.display = 'none';
    renderSessions();
  }

  // ---------- 示例卡片 ----------
  function renderCards(boxId, items) {
    const box = $(boxId);
    box.innerHTML = items.map((e, i) => (
      '<div class="card-q" data-i="' + i + '"><b>' + esc(e.title) + '</b><span>' + esc(e.hint) + '</span></div>'
    )).join('');
    box.querySelectorAll('.card-q').forEach(el => {
      el.addEventListener('click', () => send(items[+el.dataset.i].q));
    });
  }
  renderCards('rag-cards', RAG_EXAMPLES);
  renderCards('free-cards', FREE_EXAMPLES);

  function renderQuick() {
    const items = mode === 'rag' ? RAG_EXAMPLES : FREE_EXAMPLES;
    quick.innerHTML = items.map((e, i) => (
      '<button data-i="' + i + '" class="' + (mode === 'rag' && i === 3 ? 'hot' : '') + '">' + esc(e.title) + '</button>'
    )).join('');
    quick.querySelectorAll('button').forEach(el => {
      el.addEventListener('click', () => send(items[+el.dataset.i].q));
    });
  }
  renderQuick();

  // ---------- 模式切换 ----------
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      mode = tab.dataset.mode;
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
      ragChat.classList.toggle('hidden', mode !== 'rag');
      freeChat.classList.toggle('hidden', mode !== 'free');
      hint.textContent = mode === 'rag'
        ? 'Enter 发送 · Shift + Enter 换行 · 本地制度优先 · 每次联网融合补充'
        : 'Enter 发送 · Shift + Enter 换行 · 直接调用大模型，不检索知识库';
      input.placeholder = mode === 'rag' ? '输入你的制度问题…' : '想问点什么？写作、翻译、解释都可以…';
      renderQuick();
      input.focus();
    });
  });

  // ---------- 输入框 ----------
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      $('form').requestSubmit();
    }
  });
  $('form').addEventListener('submit', e => {
    e.preventDefault();
    send(input.value.trim());
  });

  // ---------- 新建 / 清空 ----------
  $('new-session').addEventListener('click', async () => {
    try {
      const s = await api('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
      });
      sessionId = s.id;
      ragHistory = [];
      clearRagChat();
      $('current-session').textContent = s.title;
      await loadSessions(false);
      toast('已新建对话', 'ok');
      input.focus();
    } catch (e) {
      toast('新建会话失败：' + e.message);
    }
  });

  $('clear').addEventListener('click', () => {
    if (mode === 'rag') {
      sessionId = null;
      ragHistory = [];
      clearRagChat();
      $('current-session').textContent = '新对话';
      renderSessions();
    } else {
      freeHistory = [];
      freeInner.querySelectorAll('.msg').forEach(el => el.remove());
      $('free-welcome').style.display = '';
    }
    input.value = '';
    input.focus();
  });

  // ---------- 移动端侧栏 ----------
  const sidebarEl = document.querySelector('.sidebar');
  const scrimEl = $('scrim');
  const closeSidebar = () => {
    sidebarEl.classList.remove('open');
    scrimEl.classList.remove('show');
  };
  $('menu-btn').addEventListener('click', () => {
    sidebarEl.classList.toggle('open');
    scrimEl.classList.toggle('show');
  });
  scrimEl.addEventListener('click', closeSidebar);

  // ---------- 渲染工具 ----------
  function scrollBottom(box) {
    requestAnimationFrame(() => { box.scrollTop = box.scrollHeight; });
  }

  function clearRagChat() {
    ragInner.querySelectorAll('.msg').forEach(el => el.remove());
    $('rag-welcome').style.display = '';
  }

  function addUser(text, inner, chatBox) {
    const el = document.createElement('div');
    el.className = 'msg user';
    el.innerHTML = '<div class="avatar">你</div><div class="bubble"><div class="text">' + esc(text) + '</div></div>';
    inner.appendChild(el);
    scrollBottom(chatBox);
    return el;
  }

  function addTyping(inner, chatBox) {
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML = '<div class="avatar">AI</div><div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>';
    inner.appendChild(el);
    scrollBottom(chatBox);
    return el;
  }

  function addError(message, inner, chatBox) {
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML = '<div class="avatar">AI</div><div class="bubble"><div class="text error">' + esc(message) + '</div></div>';
    inner.appendChild(el);
    scrollBottom(chatBox);
  }

  function copyBlock(answer) {
    return '<div class="tools"><button data-copy>复制回答</button></div>';
  }

  function bindCopy(el, answer) {
    const btn = el.querySelector('[data-copy]');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(answer);
        btn.textContent = '已复制';
        setTimeout(() => { btn.textContent = '复制回答'; }, 1500);
      } catch (e) {
        btn.textContent = '复制失败';
      }
    });
  }

  function addRagBot(data, opts) {
    opts = opts || {};
    const ok = data.verdict === '相关';
    const sources = data.sources || [];
    const hasWeb = sources.some(s => s.type === 'web');
    const inferredSource = ok ? (hasWeb ? 'hybrid' : 'local') : (hasWeb ? 'web' : 'fallback');
    const answerSource = data.answer_source || inferredSource;
    const isHybrid = answerSource === 'hybrid';
    const isWebOnly = answerSource === 'web';
    const isFallback = answerSource === 'fallback';
    const badgeClass = isFallback ? 'warn' : (isHybrid || isWebOnly ? 'web' : 'ok');
    const badgeText = {
      hybrid: '本地 + 联网', local: '制度文档', web: '联网补充', fallback: '联网失败 · 已兜底'
    }[answerSource] || '制度文档';
    const localSources = sources.filter(s => s.type !== 'web');
    const webSources = sources.filter(s => s.type === 'web');

    const steps = [
      { name: '本地检索', detail: ok ? '命中 ' + localSources.length + ' 段' : '资料不可用' },
      { name: '相关性判断', detail: data.verdict || '—' }
    ];
    if (ok) {
      steps.push({ name: '本地生成', detail: '已生成' });
      if (isHybrid) {
        steps.push(
          { name: '联网搜索', detail: '已补充公开背景' },
          { name: '融合回答', detail: '本地制度优先' }
        );
      } else {
        steps.push({ name: '联网搜索', detail: '未获得有效结果 · 使用本地答案' });
      }
    } else if (isWebOnly) {
      steps.push({ name: '联网搜索', detail: '已生成' });
    } else {
      steps.push(
        { name: '联网搜索', detail: '未获得有效结果' },
        { name: '固定兜底', detail: '已拦截' }
      );
    }
    if (data.standalone && opts.question && data.standalone !== opts.question) {
      steps.unshift({ name: '查询改写', detail: data.standalone });
    }
    const stepsHtml = steps.map(s => (
      '<span class="step">' + esc(s.name) + ' <b>' + esc(s.detail) + '</b></span>'
    )).join('');

    function renderSource(s, i) {
      if (s.type === 'web') {
        const url = String(s.url || '');
        if (!/^https?:\/\//i.test(url)) return '';
        const title = '[' + (s.index || i + 1) + '] ' + (s.title || url);
        return '<div class="src">' +
          '<div class="src-head"><b>' + esc(title) + '</b></div>' +
          '<p><a class="src-link" href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">' + esc(url) + '</a></p>' +
        '</div>';
      }
      const pct = Math.max(0, Math.min(100, Math.round((s.score || 0) * 100)));
      return '<div class="src">' +
        '<div class="src-head"><b>' + esc(s.topic) + '</b><span class="score">相似度 ' + pct + '%</span></div>' +
        '<div class="bar"><i style="width:' + pct + '%"></i></div>' +
        '<p>' + esc(s.content) + '</p>' +
      '</div>';
    }
    function renderSourceGroup(title, items) {
      return items.length ? '<div class="sources"><h3>' + esc(title) + '</h3>' +
        items.map(renderSource).join('') + '</div>' : '';
    }
    const sourcesHtml = renderSourceGroup('本地制度出处', localSources) +
      renderSourceGroup('公开网页背景', webSources);

    const mid = opts.message_id;
    const voted = opts.feedback === 1 || opts.feedback === -1 ? opts.feedback : 0;
    const fbHtml = mid ? (
      '<div class="feedback">' +
        '<span>这条回答有帮助吗？</span>' +
        '<button data-v="1" class="' + (voted === 1 ? 'on' : '') + '">👍</button>' +
        '<button data-v="-1" class="' + (voted === -1 ? 'on' : '') + '">👎</button>' +
        '<span class="fb-note">' + (voted ? '已记录反馈' : '') + '</span>' +
      '</div>'
    ) : '';

    const chips = (opts.suggestions || []).length ? (
      '<div class="suggests">' + opts.suggestions.map(q => (
        '<button class="chip" data-q="' + esc(q) + '">' + esc(q) + '</button>'
      )).join('') + '</div>'
    ) : '';

    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML =
      '<div class="avatar">AI</div>' +
      '<div class="bubble">' +
        '<div class="meta">' +
          '<span class="badge ' + badgeClass + '">' + badgeText + '</span>' +
          '<span class="time">' + (data.elapsed_ms || 0) + ' ms</span>' +
        '</div>' +
        '<div class="text">' + esc(data.answer) + '</div>' +
        '<div class="steps">' + stepsHtml + '</div>' +
        sourcesHtml +
        fbHtml +
        chips +
        copyBlock(data.answer) +
      '</div>';

    bindCopy(el, data.answer);

    const fbBox = el.querySelector('.feedback');
    if (fbBox && mid) {
      fbBox.querySelectorAll('button[data-v]').forEach(btn => {
        btn.addEventListener('click', async () => {
          try {
            await api('/api/feedback', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message_id: mid, value: Number(btn.dataset.v) })
            });
            fbBox.querySelectorAll('button[data-v]').forEach(b => b.classList.toggle('on', b === btn));
            fbBox.querySelector('.fb-note').textContent = '已记录反馈，谢谢';
          } catch (e) {
            toast('反馈提交失败：' + e.message);
          }
        });
      });
    }

    el.querySelectorAll('.chip').forEach(btn => {
      btn.addEventListener('click', () => send(btn.dataset.q));
    });

    ragInner.appendChild(el);
    scrollBottom(ragChat);
  }

  function addFreeBot(data) {
    const el = document.createElement('div');
    el.className = 'msg bot';
    el.innerHTML =
      '<div class="avatar">AI</div>' +
      '<div class="bubble">' +
        '<div class="meta">' +
          '<span class="badge free">自由对话 · 未检索</span>' +
          '<span class="time">' + (data.elapsed_ms || 0) + ' ms</span>' +
        '</div>' +
        '<div class="text">' + esc(data.answer) + '</div>' +
        copyBlock(data.answer) +
      '</div>';
    bindCopy(el, data.answer);
    freeInner.appendChild(el);
    scrollBottom(freeChat);
  }

  // ---------- 发送 ----------
  async function send(question) {
    question = (question || '').trim();
    if (!question || busy) return;

    busy = true;
    askBtn.disabled = true;
    askBtn.textContent = '思考中…';
    input.value = '';
    input.style.height = 'auto';

    const isRag = mode === 'rag';
    const inner = isRag ? ragInner : freeInner;
    const chatBox = isRag ? ragChat : freeChat;
    (isRag ? $('rag-welcome') : $('free-welcome')).style.display = 'none';

    addUser(question, inner, chatBox);
    const typing = addTyping(inner, chatBox);

    try {
      if (isRag) {
        const data = await api('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: question, session_id: sessionId, history: ragHistory.slice(-6) })
        });
        typing.remove();
        sessionId = data.session_id;
        addRagBot(data, {
          message_id: data.message_id,
          question: question,
          feedback: 0,
          suggestions: data.suggestions || []
        });
        ragHistory.push({ role: 'user', content: question });
        ragHistory.push({ role: 'assistant', content: data.answer });
        if (ragHistory.length > 12) ragHistory = ragHistory.slice(-12);

        await loadSessions(false);
        const current = sessions.find(s => s.id === sessionId);
        if (current) $('current-session').textContent = current.title;
        loadStats().catch(e => toast('统计刷新失败：' + e.message));
      } else {
        const payload = freeHistory.concat([{ role: 'user', content: question }]);
        const data = await api('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: payload })
        });
        typing.remove();
        addFreeBot(data);
        freeHistory = payload.concat([{ role: 'assistant', content: data.answer }]).slice(-12);
      }
    } catch (err) {
      typing.remove();
      addError('请求失败：' + err.message, inner, chatBox);
      toast('请求失败：' + err.message);
    } finally {
      busy = false;
      askBtn.disabled = false;
      askBtn.textContent = '发送';
      input.focus();
    }
  }

  // ---------- 切块配置：校验、保存并重建 ----------
  $('apply-rag-settings').addEventListener('click', async () => {
    const sizeText = $('chunk-size').value.trim();
    const overlapText = $('chunk-overlap').value.trim();
    const chunkSize = Number(sizeText);
    const chunkOverlap = Number(overlapText);
    if (!/^\d+$/.test(sizeText) || !/^\d+$/.test(overlapText) ||
        !Number.isInteger(chunkSize) || !Number.isInteger(chunkOverlap)) {
      toast('切块大小和重叠字符都必须是整数');
      return;
    }
    if (chunkSize < 50 || chunkSize > 500) {
      toast('切块大小必须在 50~500 之间');
      return;
    }
    if (chunkOverlap < 0 || chunkOverlap > 100) {
      toast('重叠字符必须在 0~100 之间');
      return;
    }
    if (chunkOverlap >= chunkSize) {
      toast('重叠字符必须小于切块大小');
      return;
    }

    const btn = $('apply-rag-settings');
    btn.disabled = true;
    btn.textContent = '正在重建索引…';
    try {
      const data = await api('/api/rag-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap })
      });
      fillRagSettings(data);
      await loadStats();
      toast('配置已保存，索引重建完成', 'ok');
    } catch (e) {
      $('chunk-size').value = savedRagSettings.chunk_size;
      $('chunk-overlap').value = savedRagSettings.chunk_overlap;
      toast('应用失败，已保留原配置：' + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '应用并重建索引';
    }
  });

  // ---------- 添加文档弹窗 ----------
  function openModal() {
    $('modal').classList.remove('hidden');
    $('modal-error').textContent = '';
    $('modal-topic').value = '';
    $('modal-content').value = '';
    $('modal-topic').focus();
  }
  function closeModal() {
    $('modal').classList.add('hidden');
  }

  $('add-doc').addEventListener('click', openModal);
  $('modal-cancel').addEventListener('click', closeModal);
  $('modal').addEventListener('click', e => {
    if (e.target.id === 'modal') closeModal();
  });
  $('modal-submit').addEventListener('click', async () => {
    const topic = $('modal-topic').value.trim();
    const content = $('modal-content').value.trim();
    if (!topic) {
      $('modal-error').textContent = '请填写文档主题';
      return;
    }
    if (content.length < 10) {
      $('modal-error').textContent = '正文太短，至少 10 个字';
      return;
    }
    $('modal-submit').disabled = true;
    $('modal-submit').textContent = '重建索引中…';
    try {
      await api('/api/docs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topic, content: content })
      });
      closeModal();
      await loadDocs();
      await loadStats();
      toast('文档已添加，索引已重建', 'ok');
    } catch (e) {
      $('modal-error').textContent = '添加失败：' + e.message;
    } finally {
      $('modal-submit').disabled = false;
      $('modal-submit').textContent = '保存并重建索引';
    }
  });

  // ---------- 启动 ----------
  (async function init() {
    try { await loadInfo(); } catch (e) { toast('服务连接失败：' + e.message); }
    try { await loadDocs(); } catch (e) { toast('知识库加载失败：' + e.message); }
    try { await loadRagSettings(); } catch (e) { toast('切块配置加载失败：' + e.message); }
    try { await loadStats(); } catch (e) { toast('统计加载失败：' + e.message); }
    try { await loadSessions(true); } catch (e) { toast('会话加载失败：' + e.message); }
  })();
</script>
</body>
</html>
"""


def _to_lc_messages(items: list) -> list:
    """把前端发来的 [{role, content}] 转成 LangChain 消息，最多保留 6 轮（12 条）。"""
    mapping = {"user": HumanMessage, "assistant": AIMessage}
    messages = [SystemMessage(content=CHAT_SYSTEM)]
    for item in items[-MEMORY_ROUNDS * 2:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in mapping and content:
            messages.append(mapping[role](content=content))
    return messages


class Handler(BaseHTTPRequestHandler):
    # ---------- 路由 ----------
    def do_GET(self):
        try:
            self._handle_get()
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        try:
            self._handle_post()
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_DELETE(self):
        try:
            self._handle_delete()
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _handle_get(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/info":
            self._send_json(200, INFO)
        elif path == "/api/docs":
            self._send_json(200, rag_service.list_documents())
        elif path == "/api/rag-settings":
            self._send_json(200, rag_service.get_rag_settings_info())
        elif path == "/api/stats":
            self._send_json(200, rag_service.stats())
        elif path == "/api/sessions":
            self._send_json(200, rag_service.list_sessions())
        elif path.startswith("/api/sessions/") and path.endswith("/messages"):
            parts = path.strip("/").split("/")
            self._send_json(200, rag_service.get_messages(parts[2]))
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def _handle_post(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/ask":
            self._handle_ask()
        elif path == "/api/chat":
            self._handle_chat()
        elif path == "/api/docs":
            self._handle_add_doc()
        elif path == "/api/rag-settings":
            self._handle_rag_settings()
        elif path == "/api/sessions":
            payload = self._read_json()
            self._send_json(200, rag_service.create_session(payload.get("title")))
        elif path == "/api/feedback":
            self._handle_feedback()
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def _handle_delete(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        if path.startswith("/api/docs/") and len(parts) == 3 and parts[2].isdigit():
            ok = rag_service.delete_document(int(parts[2]))
            self._send_json(200 if ok else 404, {"ok": ok})
        elif path.startswith("/api/sessions/") and len(parts) == 3:
            ok = rag_service.delete_session(parts[2])
            self._send_json(200 if ok else 404, {"ok": ok})
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    # ---------- 业务处理 ----------
    def _handle_ask(self) -> None:
        payload = self._read_json()
        question = (payload.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "问题不能为空"})
            return

        history = payload.get("history")
        if not isinstance(history, list):
            history = []

        result = rag_service.ask(question, payload.get("session_id"), history)
        self._send_json(200, result)

    def _handle_add_doc(self) -> None:
        payload = self._read_json()
        try:
            doc = rag_service.add_document(payload.get("topic"), payload.get("content"))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, "doc": doc})

    def _handle_rag_settings(self) -> None:
        payload = self._read_json()
        try:
            result = rag_service.apply_rag_settings(
                payload.get("chunk_size"), payload.get("chunk_overlap")
            )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, {"ok": True, **result})

    def _handle_feedback(self) -> None:
        payload = self._read_json()
        try:
            ok = rag_service.save_feedback(
                int(payload.get("message_id")), int(payload.get("value"))
            )
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc) or "参数不合法"})
            return
        self._send_json(200 if ok else 404, {"ok": ok})

    def _handle_chat(self) -> None:
        started = time.perf_counter()
        payload = self._read_json()
        items = payload.get("messages") or []
        if not isinstance(items, list) or not any(
            (m.get("content") or "").strip() for m in items if isinstance(m, dict)
        ):
            self._send_json(400, {"error": "消息不能为空"})
            return

        reply = CHAT_MODEL.invoke(
            _to_lc_messages([m for m in items if isinstance(m, dict)])
        )
        self._send_json(
            200,
            {
                "answer": reply.content,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    # ---------- 基础工具 ----------
    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("  [web]", fmt % args)


def main() -> None:
    port = 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"服务已启动：http://127.0.0.1:{port}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
