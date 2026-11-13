# 智能体开发实训 · LangChain / LangGraph 核心能力复现与制度问答应用

把课程案例中的 LangChain + LangGraph 核心主线重写成 16 个可独立运行、带中文注释的脚本，
并在其上封装了具备产品形态的**校园制度问答应用**（网页 + 命令行）。

- 输入问题 → 查询改写 → 检索制度文档 → 判断资料是否相关 → 回答并显示出处
- 多轮会话存入 SQLite，支持切换 / 删除会话、点赞点踩、追问建议
- 知识库可在页面里增删文档，增删后自动重建向量索引
- 知识库外的问题不硬编，自动走兜底分支
- 模型底座可切换：阿里千问 / DeepSeek，改 `.env` 即可，业务代码不动

---

## 一、项目结构

```
ai-agents-repro/
├── .env                  # 真实 Key（已在 .gitignore 中排除，不要提交）
├── .env-example          # Key 模板
├── .gitignore
├── requirements.txt
├── config.py             # ★ 统一模型接入层（所有脚本共用）
├── rag_service.py        # ★ 核心服务层：SQLite 存储 + 知识库 + RAG 图
├── cli_app.py            # ★ 统一入口：命令行制度问答助手
├── web_app.py            # ★ 网页版入口：浏览器问答页面（仅用标准库）
├── data/app.db           # 运行时自动生成的 SQLite（文档 / 会话 / 消息）
│
├── 01_hello_model.py             # 模型调用三件套 / invoke / stream / batch
├── 02_model_params_messages.py   # temperature / max_tokens / 多角色消息
├── 03_prompt_template.py         # PromptTemplate / ChatPromptTemplate / partial
├── 04_output_parser.py           # Str / JSON / with_structured_output
├── 05_lcel_chain.py              # 顺序链 / Lambda / 并行 / 分支
├── 06_memory.py                  # 手工历史 + RunnableWithMessageHistory
├── 07_tools.py                   # @tool + bind_tools + ToolMessage（手写调用循环）
├── 08_embedding.py               # Embedding + 余弦相似度 + 语义检索
├── 09_rag_minimal.py             # 切块 / 内存向量库 / RAG 链 / 无 RAG 对照
├── 10_agent_react.py             # create_agent（框架版 ReAct）
│
├── 11_langgraph_hello.py         # 最小图：START -> model -> END
├── 12_langgraph_state.py         # State / Reducer / 并行与汇合
├── 13_langgraph_conditional.py   # 条件边路由（图里的 if/else）
├── 14_langgraph_persistence.py   # Checkpointer / thread_id / 多轮
├── 15_langgraph_agent.py         # 手写 ReAct 循环图（tool -> assistant 回环）
└── 16_langgraph_rag.py           # RAG + 相关性判断分支的图工作流

代码导读.md                       # 逐章注释：每个脚本在讲什么
答辩材料.md                       # 项目介绍、演示脚本、知识点问答
实验报告.md                       # Vibe Coding 过程与心得
```

---

## 二、环境要求

- Python 3.10
- 可访问阿里云百炼（DashScope）或 DeepSeek 的 API

从零安装：

```powershell
cd ai-agents-repro
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本机已自带 `.venv`，可直接跳到下一步。

---

## 三、配置 Key

复制模板并填写：

```powershell
Copy-Item .env-example .env
```

`.env` 内容：

```ini
# 默认厂商：qwen 或 deepseek
REPRO_PROVIDER=qwen

# 千问 Key（必填：对话 + 向量都靠它）
aliQwen-api = '你的千问 Key'

# DeepSeek Key（可选）
deepseek-chat-api = '你的 DeepSeek Key'
```

说明：

- 千问申请地址：<https://bailian.console.aliyun.com> → API-KEY
- DeepSeek 申请地址：<https://platform.deepseek.com> → API keys
- **DeepSeek 官方没有向量接口**。切到 `REPRO_PROVIDER=deepseek` 后，对话走 `deepseek-chat`，
  RAG 的向量自动借道千问，所以 `aliQwen-api` 仍然要填。

自检：

```powershell
.\.venv\Scripts\python.exe config.py
```

看到 `provider=qwen`、`chat_model=qwen-plus`、`api_key=已配置` 即正常。

---

## 四、运行

统一入口（推荐先跑这个）：

```powershell
# 网页版：启动后浏览器打开 http://127.0.0.1:8000
.\.venv\Scripts\python.exe web_app.py

# 命令行交互模式，输入 q 退出
.\.venv\Scripts\python.exe cli_app.py

# 命令行单次提问
.\.venv\Scripts\python.exe cli_app.py "出差报销需要在几天内提交？"
```

网页版包含：

- **两种模式**：制度问答（RAG + 出处）与自由对话（直接调模型、支持多轮），顶部一键切换，历史互不干扰
- 左侧栏：服务状态、会话列表（切换 / 删除）、知识库文档（主题 + 字数，可删除，弹窗新增）、
  统计卡片（提问数 / 兜底率 / 平均耗时 / 文档与片段数）
- 右侧：消息气泡，回答带相关性徽章、检索步骤、耗时；出处卡片用进度条展示相似度
- 每条回答可点赞 / 点踩（写入数据库），并给出 3 条模板追问，点击直接继续问
- 会话切换时从 `/api/sessions/<id>/messages` 拉取历史；新增 / 删除文档后自动重建索引
- 常驻示例按钮：报销 / 年假 / 远程办公 / 知识库外兜底；自由对话下为概念解释 / 框架对比 / 写邮件 / 翻译
- 支持新建 / 清空对话、复制回答，手机端自适应

网页版依赖的 JSON 接口（出错时返回带 `error` 字段的 JSON）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/info` | 厂商、对话模型、向量模型、top_k |
| GET / POST | `/api/docs` | 文档列表 / 新增文档 `{topic, content}` |
| DELETE | `/api/docs/<id>` | 删除文档并重建索引 |
| GET / POST | `/api/sessions` | 会话列表 / 新建会话 |
| DELETE | `/api/sessions/<id>` | 删除会话及其消息 |
| GET | `/api/sessions/<id>/messages` | 会话历史消息 |
| POST | `/api/ask` | `{question, session_id, history}` 制度问答 |
| POST | `/api/feedback` | `{message_id, value}`，value 为 1 或 -1 |
| GET | `/api/stats` | 统计面板数据 |
| POST | `/api/chat` | `{messages}` 自由对话 |

会话、知识库和历史消息都存在 `data/app.db`（SQLite，第一次启动自动建库并写入 3 篇制度文档），
删除该文件即可恢复初始状态。

单个脚本：
```powershell
.\.venv\Scripts\python.exe 01_hello_model.py
.\.venv\Scripts\python.exe 07_tools.py
.\.venv\Scripts\python.exe 16_langgraph_rag.py
```

建议按编号顺序运行：01 → 10 是 LangChain 主线，11 → 16 是 LangGraph 主线。
脚本会真实调用模型，**请逐个运行，不要并发**，避免触发限流。

---

## 五、技术要点

| 能力 | 实现位置 | 关键点 |
|---|---|---|
| 模型接入 | `config.py` | 厂商表 + 工厂函数，`lru_cache` 复用客户端 |
| 提示词 | `03` | `ChatPromptTemplate`、`partial` 预填变量 |
| 结构化输出 | `04` | `with_structured_output(..., method="function_calling")` |
| 工具调用 | `07` | 模型给意图，程序执行，`ToolMessage` 回填，循环到无调用 |
| RAG | `09` | 切块 + 重叠、内存向量库、`k=2` 检索、无 RAG 对照 |
| 框架版 Agent | `10` | `create_agent(model, tools, system_prompt)` |
| 图编排 | `11`~`14` | State、Reducer、条件边、Checkpointer |
| 手写 Agent 图 | `15` | `tools → assistant` 回边 + `recursion_limit` |
| 带判断的 RAG | `16` | `grade` 节点 + 条件边 + `fallback` |
| 服务层 | `rag_service.py` | SQLite 存文档/会话/消息，知识库增删后重建索引 |
| 查询改写 | `rag_service.py` | 有历史时先把追问改写成独立问题，再检索 |
| 统一入口 | `cli_app.py` / `web_app.py` | 命令行复用 16 的图；网页版走服务层，带会话与知识库管理 |

---

## 六、常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `401 Unauthorized` | Key 为空或填错 | 检查 `.env` 的 `aliQwen-api` / `deepseek-chat-api` |
| `429` 限流 | 免费额度并发限制 | 等 1 分钟、逐个运行；客户端已自动重试 4 次 |
| `400 InvalidParameter: contents is neither str...` | 向量接口不接受 token id | 已设置 `check_embedding_ctx_length=False` |
| 工具调用不执行 | 模型不支持 function calling | 换 `qwen-plus` / `deepseek-chat` |

更多细节见 [代码导读.md](代码导读.md)。

---

## 七、说明

- 代码按课程案例的知识点重写，每个脚本开头标注了对应的原仓库目录，便于对照阅读。
- 与原仓库相比的主要改动：统一模型接入层、修复工具调用只跑一轮的问题、
  结构化输出改用工具调用模式、RAG 增加相关性判断与兜底、补充中文注释。
- **提交时请勿包含 `.env`**（内含真实 Key），只提交 `.env-example`。
