# Agent 知识库问答服务
[github.com/wmf-hub/rag-agent-project.git](https://github.com/wmf-hub/rag-agent-project.git)

基于 **LangGraph** 的 ReAct 知识库问答后端：混合检索 + Cross-Encoder 精排 + 多轮/长期记忆 + SSE 流式输出。

`Python 3.12` · `FastAPI` · `LangGraph` · `Milvus` · `PostgreSQL` · `DeepSeek` · `Docker`

---

## 1. 这是什么

一个可本地一键启动的企业知识库问答 Agent。用户提问后，Agent 自行判断是否需要检索：需要则调用知识库工具，经「向量 + BM25 双路召回 → RRF 融合 → Cross-Encoder 精排」拿到最相关的片段，带着 `[doc1]` `[doc2]` 引用标记生成答案，并把结构化引用（来源 + 相关度分数）一并返回给前端渲染脚注。

除了问答本身，项目重点做了三件事：

- **记忆分层**：短期记忆（会话维度，checkpointer）与长期记忆（用户维度，独立表 + 异步抽取）互不干扰
- **可观测**：request_id 全链路串联，trace 覆盖节点 / LLM / 工具耗时
- **可运维**：统一的异常分层与错误码、健康探针、Docker 交付

---

## 2. 核心特性

| 模块 | 能力 |
|---|---|
| 检索 | dense 向量 + jieba 分词 BM25 双路召回，RRF 融合，bge-reranker 精排，分数阈值过滤 |
| 生成 | DeepSeek（OpenAI 兼容协议），强制保留 `[docN]` 引用标记，无相关内容时明确告知而非编造 |
| 对话 | LangGraph `agent ⇄ tools` ReAct 循环，节点级重试策略，工具异常兜底为模型可读的一句话 |
| 记忆 | 短期：PostgreSQL checkpointer 按 `thread_id` 多轮上下文；长期：按 `user_id` 存取用户事实与偏好 |
| 流式 | `POST /chat/stream` 逐 token 推送，引用单独事件，客户端断开即停止推理 |
| 写入 | 内容哈希主键 + `upsert`，重复导入幂等；`--reset` 支持全量重建 |
| 可观测 | 单行日志 + request_id、访问日志、LLM/工具耗时 trace |
| 健壮性 | 异常按 LLM / 工具 / 编排 / 存储分层，映射 HTTP 状态码与 `retryable` 语义 |

---

## 3. 架构

```
浏览器 / curl
      │  POST /chat         (JSON)
      │  POST /chat/stream  (SSE)
      ▼
┌────────────────────────────────────────────────────────┐
│ FastAPI (app/main.py)                                  │
│  RequestIDMiddleware  →  request_id / 访问日志          │
│  /chat  →  读长期记忆 → 调图 → 返回 reply + citations   │
└───────────────────────┬────────────────────────────────┘
                        ▼
┌────────────────────────────────────────────────────────┐
│ LangGraph (app/graph.py)                               │
│  START → agent ⇄ tools → END                           │
│   agent: DeepSeek，system prompt 每轮重建并注入记忆      │
│   tools: search_knowledge_base / get_current_time      │
└───────┬───────────────────────────────┬────────────────┘
        ▼                               ▼
┌──────────────────┐          ┌─────────────────────────┐
│ PostgreSQL       │          │ Milvus                  │
│  checkpoints     │          │  knowledge_base         │
│  user_memories   │          │  （dense + sparse）      │
└──────────────────┘          └───────────┬─────────────┘
                                          ▼
                          Ollama (qwen3-embedding) + bge-reranker
```

一次请求的完整路径：

```
用户提问
  → 取该 user_id 的长期记忆，渲染成文本注入 system prompt
  → agent 判断能否直接回答；需要知识时调用 search_knowledge_base
  → 向量 + BM25 召回 Top-10 → RRF 融合 → rerank 取 Top-3（低于阈值丢弃）
  → 返回带 [doc1..3] 的上下文 → 模型生成答案（引用原样保留）
  → 结构化引用挂在 ToolMessage.artifact 上，随响应返回
  → 回答发出后，后台异步抽取本轮值得长期记住的信息
```

---

## 4. 技术选型与理由

| 选择 | 为什么不是别的 |
|---|---|
| **LangGraph** | 相比手写 ReAct 循环，它把状态、分支、checkpoint、重试都做了；换编排框架只需改 `graph.py`，工具和检索层不动 |
| **Milvus** | 需要同时支持稠密向量和 BM25 稀疏检索，且能 Filter；faiss 本地库在多进程下没有服务化能力 |
| **PostgreSQL 存 checkpoint** | 会话状态与业务数据同库同事务语义，运维成本低；`AsyncPostgresSaver` 直接复用连接池 |
| **bge-reranker 而非直接截断** | 向量召回只看语义相似，Top-3 里常混入无关片段。实测同一查询：相关 0.887 / 无关 0.0013，区分度足够支撑阈值过滤 |
| **RRF 而非加权分数相加** | dense 与 sparse 分数量纲不同，RRF 只依赖排名，免调参、更稳定 |
| **引用挂 `ToolMessage.artifact`** | 不塞进 state（会污染上下文并占用 token）、不塞进全局变量（并发会串号），且天然随 checkpoint 按会话隔离 |
| **纯 ASGI 中间件** | `BaseHTTPMiddleware` 会给每个请求额外包一层 task，破坏 SSE 场景下的 `request.is_disconnected()` |

---

## 5. 快速开始

### 5.1 前置依赖

- Python 3.12
- Docker Desktop（跑 Milvus / PostgreSQL）
- Ollama，并已拉取 embedding 模型：`ollama pull qwen3-embedding:0.6b`
- 一个 OpenAI 兼容的大模型 API Key（默认 DeepSeek）

### 5.2 启动基础设施

```bash
docker compose up -d
```

| 服务 | 端口 | 用途 |
|---|---|---|
| postgres | 5432 | checkpointer / 长期记忆 |
| milvus-standalone | 19530, 9091 | 向量库 |
| minio | 9000, 9001 | Milvus 对象存储 |
| etcd | — | Milvus 元数据 |

### 5.3 配置

```bash
cp .env.example .env
```

必填项：

```ini
LLM_API_KEY=sk-xxx
POSTGRES_USER=...      # 与 docker-compose 里的环境变量保持一致
POSTGRES_PASSWORD=...
POSTGRES_DB=short_memory
```

Milvus 二选一：

```ini
# 远程服务（推荐，配合 docker compose）
MILVUS_URI=http://localhost:19530
# 或本地文件库（路径必须是纯 ASCII，faiss 写不了中文路径）
# MILVUS_DB_PATH=E:/milvus_data/milvus.db
```

### 5.4 安装依赖

```bash
python -m venv .venv && .\.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

> torch 体积较大，纯 CPU 推理可先装 CPU 版：
> `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`

### 5.5 导入知识库

```bash
python "script/导入知识文档.py"            # 增量（幂等，可重复执行）
python "script/导入知识文档.py" --reset    # 删集合重建（换 embedding 模型时用）
```

### 5.6 启动服务

```bash
python run.py
```

> 建议用 `run.py` 而不是直接 `uvicorn app.main:app`：Windows 上 psycopg 的异步连接不支持默认的 `ProactorEventLoop`，`run.py` 会显式切换到 `SelectorEventLoop`。

开发时热重载：`APP_RELOAD=true python run.py`

### 5.7 冒烟验证

```bash
curl http://127.0.0.1:8000/livez      # {"status":"ok"}
curl http://127.0.0.1:8000/           # 依赖探活：milvus / ollama
```

浏览器打开 `http://127.0.0.1:8000/docs` 可直接调接口，或用 `script/流式前端.html` 看打字机效果。

---

## 6. 接口

### 6.1 `POST /chat`

```json
// 请求
{ "message": "年假有几天", "thread_id": "jack_001", "user_id": "jack" }

// 响应
{
  "reply": "员工入职满一年享有 5 天带薪年假 [doc1]。",
  "citations": [
    { "content": "年假制度：员工入职满一年享有5天带薪年假…", "source": "员工手册.pdf", "score": 0.8873 }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `thread_id` | 会话 ID，决定短期记忆边界；同一个 `thread_id` 才能延续上下文 |
| `user_id` | 用户 ID，决定长期记忆边界；`anonymous` 不写记忆（避免共用） |
| `citations` | 只包含**本轮**检索命中的片段，非检索轮次为空数组 |

### 6.2 `POST /chat/stream`（SSE）

请求体同 `/chat`，响应为 `text/event-stream`：

| 事件 | 数据 | 时机 |
|---|---|---|
| `token` | `{"text": "员工"}` | 逐 token |
| `citations` | `{"citations": [...]}` | 检索工具执行完 |
| `done` | `{}` | 流结束 |
| `error` | `{"code","message","retryable","request_id"}` | 流中出错（此时 HTTP 状态码已是 200） |

```js
// 注意：EventSource 只支持 GET，POST 需用 fetch + ReadableStream 手工解析
const resp = await fetch('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message, thread_id, user_id }),
});
```

### 6.3 健康检查

| 路径 | 语义 |
|---|---|
| `GET /livez` | 存活探针，只要进程能响应就 200（Docker HEALTHCHECK 用这个） |
| `GET /` | 就绪探针，探测 milvus / ollama，任一不可用返回 503 |

### 6.4 错误响应

所有错误统一结构，`code` 供程序判断、`message` 供人读，堆栈只进日志：

```json
{
  "error": { "code": "llm_unavailable", "message": "模型服务暂时不可用，请稍后重试。", "retryable": true },
  "request_id": "62b5d12301a1"
}
```

| code | HTTP | 含义 |
|---|---|---|
| `invalid_request` | 422 | 参数不合法（带字段级 details） |
| `agent_step_limit` | 422 | 工具调用轮数超限 |
| `llm_unavailable` | 503 | 模型上游不可用/限流 |
| `storage_unavailable` | 503 | 数据库不可用 |
| `agent_timeout` | 504 | 图执行超时 |
| `internal_error` | 500 | 未捕获异常 |

---

## 7. 目录结构

```
agent_project/
├── app/
│   ├── main.py            # FastAPI 入口、路由、异常处理器、健康检查
│   ├── graph.py           # LangGraph 图与节点定义
│   ├── config.py          # 全部配置项（.env → 常量）
│   ├── errors.py          # 业务异常与第三方异常翻译
│   ├── memory.py          # 长期记忆：表结构、读写、抽取
│   ├── context.py         # request_id 上下文
│   ├── logging_config.py  # 日志格式与过滤器
│   ├── middleware.py      # 纯 ASGI 的 request_id 中间件
│   ├── tracing.py         # 轻量 span 与 LLM/工具回调
│   ├── tools.py           # 普通工具（当前时间）
│   └── rag/
│       ├── vector_store.py # 建集合、分块、向量化、写入、dense/sparse 检索
│       └── retriever.py    # 混合召回 + RRF + rerank，封装为 Tool
├── script/                 # 导入/验证脚本与流式演示前端
├── run.py                  # 启动入口（统一事件循环与日志）
├── Dockerfile / .dockerignore
├── docker-compose.yml      # Milvus / etcd / MinIO / PostgreSQL
└── requirements.txt        # 直接依赖（requirements.lock.txt 为完整快照）
```

---

## 8. 关键设计决策

**引用为什么走 `ToolMessage.artifact`**
最初把引用暂存在模块级全局变量里，由 API 层读取——并发请求会互相覆盖，且非检索轮次会把上一轮的引用带出来。改为挂在工具返回的消息 `artifact` 上：不发给模型（省 token）、随 checkpoint 按会话隔离、业务层只取「最后一条用户消息之后」的消息，天然不会串号。

**system prompt 为什么每轮重建**
如果把注入记忆的 system message 写进历史，checkpoint 会把第一轮的版本永久固定，之后记忆更新了也不生效。现在每轮从 `state["memory_block"]` 重新拼装，并过滤掉历史里的旧 system message。

**写入为什么是幂等的**
早期用自增 `id` 做主键，重跑导入脚本会产生主键冲突和重复实体。改为 `sha256(chunk)[:15]` 生成稳定 int64 主键 + `upsert`，同一份文档反复导入结果一致。

**异常为什么分层**
LLM 上游错误（429/超时）适合重试并返回 503；工具错误不该让整个请求失败，而是翻译成一句话交给模型告知用户；编排层超限要返回 422 而不是 500。分层后 API 层只需要一个映射函数，不用写一堆 if-else。

**为什么没用 ORM**
本项目只有两张自建表，且大量逻辑是 SQL 级别的 `ON CONFLICT`、`RETURNING`，直接写 SQL + `dict_row` 比引入 ORM 更直观、更好调试。

---

## 9. 实测参考

本地环境（Windows + Docker Desktop + 单进程），仅供参考：

| 指标 | 数值 |
|---|---|
| 冷启动 | 13~17s，其中 rerank 模型预热 13~17s（1GB 权重，可 `RERANK_PRELOAD=false` 关闭） |
| 单轮对话（未触发检索） | LLM 耗时 1.4~1.5s，端到端 1.5~1.7s |
| rerank 区分度 | 查询「年假有几天」：相关文档 0.8873 / 无关 0.0013 / 0.00004 |

---

## 10. 已知限制

- **知识库只有 3 条示例文档**，尚未接入 PDF/DOCX 等真实文档加载器
- `thread_id` 由客户端传入且默认 `"default"`，多个客户端不传就会共用同一段上下文
- `user_id` 由客户端自报，仅用于记忆隔离，**不具备鉴权能力**（任何人都可以声称是别人）
- 并发保护是**单进程内**的 `asyncio.Lock`，`--workers > 1` 时失效（需换 PostgreSQL advisory lock）
- 暂无自动化测试与 CI
- 未做限流与配额

---

## 11. Roadmap

- [ ] 文档加载器（PDF / DOCX / Markdown）与目录批量导入
- [ ] 离线评测集与指标（Recall@3 / MRR），对比纯向量与混合检索
- [ ] 会话管理接口：列表 / 详情 / 重命名 / 删除
- [ ] `thread_id` 服务端签发 + 归属校验（跨用户返回 404）
- [ ] pytest 用例（假模型 + 内存 checkpointer）与 GitHub Actions
- [ ] 多进程并发保护（advisory lock）

---

## 12. 开发约定

- **日志**：统一格式 `时间 级别 [request_id] 模块: 内容`，第三方库降噪到 WARNING
- **trace**：节点/LLM/工具耗时都带同一个 `request_id`，可直接按 id grep 出一次请求的完整链路
- **配置**：所有可调参数集中在 `app/config.py`，缺关键项（API Key、Milvus 地址）启动即失败，不留隐式默认值
- **错误**：业务异常继承 `AppError`，自带 `code / http_status / retryable`；第三方异常经 `translate()` 翻译，不要把 `str(exc)` 直接返回给前端
