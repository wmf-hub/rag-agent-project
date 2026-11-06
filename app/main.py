import asyncio
import json
import logging
import httpx
from pydantic import Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, ToolMessage

from app.config import (
    AGENT_RECURSION_LIMIT,
    AGENT_TIMEOUT,
    OLLAMA_BASE_URL,
    PG_POOL_MAX_SIZE,
    PG_POOL_MIN_SIZE,
    PG_POOL_TIMEOUT,
    POSTGRES_DB_URI,
    RERANK_PRELOAD,
)
from app.graph import build_graph
from app.errors import AppError, AgentTimeoutError, translate
from app.logging_config import setup_logging
from app.memory import ensure_memory_table, extract_and_save, load_memories, render_memories
from app.middleware import RequestIDMiddleware
from app.rag.retriever import preload_reranker
from app.rag.vector_store import get_milvus_client
from app.tracing import TraceCallbackHandler, trace_span

setup_logging()   # 必须早于任何模块级日志：让日志格式和级别立刻生效

logger = logging.getLogger(__name__)

# ========== 连接池 + 异步 checkpointer + lifespan 生命周期管理 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = AsyncConnectionPool(
        POSTGRES_DB_URI,
        # psycopg_pool 默认 max_size 跟随 min_size（=4），SSE + 后台任务容易超时，显式放大
        min_size=PG_POOL_MIN_SIZE,
        max_size=PG_POOL_MAX_SIZE,
        timeout=PG_POOL_TIMEOUT,
        open=False,    # psycopg 3.2+ 在异步环境下不允许在构造函数里隐式 open
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    try:
        await pool.open(wait=True, timeout=10)
    except Exception:
        logger.critical("Postgres 连接失败，服务无法启动")
        raise

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    await ensure_memory_table(pool)          # 长期记忆表

    # 编译图 并 存到 app.state 中，导入 app.main 不再触发任何数据库连接（连接推迟到启动时）。
    app.state.pool = pool                    # 长期记忆读写要用
    app.state.checkpointer = checkpointer
    app.state.graph = build_graph(checkpointer)

    # 预热 rerank 模型：把 1GB 模型的加载成本从第一个用户请求挪到这里
    if RERANK_PRELOAD:
        try:
            await asyncio.to_thread(preload_reranker)
        except Exception:
            logger.warning("rerank 模型预热失败，首个检索请求会现场加载", exc_info=True)

    logger.info("服务就绪：连接池 %s~%s", PG_POOL_MIN_SIZE, PG_POOL_MAX_SIZE)
    try:
        yield
    finally:
        logger.info("服务关闭中…")
        await pool.close()

app = FastAPI(title="Agent Project", lifespan=lifespan)

# ========== CORS：浏览器从别的 origin（如 file:// 或其它端口）访问时必须放行，否则 fetch 会被拦 ==========
app.add_middleware(
    CORSMiddleware,
    # 演示环境放开全部；生产请改成具体域名，如 ["https://your-app.com"]
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 最后注册 = 最外层：保证 request_id 在任何处理器/日志之前就位
app.add_middleware(RequestIDMiddleware)


def _request_id(request: Request) -> str:
    """从 scope 里取 request_id（异常冒泡后 ContextVar 可能已清空，所以读 scope）。"""
    return getattr(request.state, "request_id", "-")

# ========== 全局异常处理器 ==========
@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    """
    处理 AppError 异常并记录日志，根据 HTTP 状态码返回 JSON 格式的错误信息。
    Args:
        request: FastAPI 请求对象。
        exc: AppError 异常对象。

    Returns:
        JSONResponse: 包含错误信息的 JSON 响应。
    """
    log = logger.error if exc.http_status >= 500 else logger.warning
    log("AppError code=%s detail=%s", exc.code, exc.detail, exc_info=exc.cause)
    payload = exc.to_dict()
    payload["request_id"] = _request_id(request)   # 报错也带上 id，方便用户报障时对日志
    return JSONResponse(status_code=exc.http_status, content=payload)

@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """
    处理 RequestValidationError 异常，根据 HTTP 状态码返回 JSON 格式的错误信息。
    Args:
        request: FastAPI 请求对象。
        exc: RequestValidationError 异常对象。

    Returns:
        JSONResponse: 包含错误信息的 JSON 响应。
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "请求参数不合法",
                "retryable": False,
                # jsonable_encoder 是必须的：ctx 里可能塞着异常对象，直接 dumps 会炸
                "details": jsonable_encoder(exc.errors()),
            },
            "request_id": _request_id(request),
        },
    )

@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """
    处理未捕获的异常并记录日志，根据 HTTP 状态码返回 JSON 格式的错误信息。
    Args:
        request: FastAPI 请求对象。
        exc: 未捕获的异常对象。

    Returns:
        JSONResponse: 包含错误信息的 JSON 响应。
    """
    logger.exception("未捕获异常: %s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "error":{
                "code": "internal_error",
                "message": "服务内部错误，请稍后重试。",
                "retryable": False
            },
            "request_id": _request_id(request),
        },
    )

# ========== 健康检查 ==========
async def _probe_milvus() -> bool:
    try:
        # pymilvus 是同步阻塞的，必须丢线程池，别卡住事件循环
        await asyncio.wait_for(asyncio.to_thread(get_milvus_client().list_collections), timeout=3)
        return True
    except Exception:
        return False

async def _probe_ollama() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        return resp.status_code == 200
    except Exception:
        return False

@app.get("/livez", include_in_schema=False)
async def livez():
    """存活探针：进程还能响应就 200，不探任何外部依赖（给 Docker HEALTHCHECK 用）。"""
    return {"status": "ok"}


@app.get("/")
async def health(response: Response):
    checks = {
        "milvus": await _probe_milvus(),
        "ollama": await _probe_ollama(),
    }
    ok = all(checks.values())
    response.status_code = 200 if ok else 503
    return {"status": "ok" if ok else "degraded", "checks": checks}

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="用户问题")
    thread_id: str = Field(default="default", description="对话 ID")
    user_id: str = Field(default="anonymous", description="用户 ID")

class ChatResponse(BaseModel):
    reply: str
    citations: list[dict] = []

def citations_of_last_turn(messages: list) -> list[dict]:
    """
    只取「本轮」检索工具产出的引用。

    带上 checkpointer 之后，result["messages"] 是整个 thread 的历史消息，
    如果直接取最后一条带引用的 ToolMessage，问一句「现在几点」也会把
    上几轮检索的引用带出来。所以先定位最后一条 HumanMessage，
    只看它之后的消息。

    Args:
        messages: 图执行完返回的完整消息列表

    Returns:
        list[dict]: 本轮最后一次检索命中的引用，没有检索则为空列表
    """
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break

    citations: list[dict] = []
    for message in messages[start:]:
        if isinstance(message, ToolMessage) and isinstance(message.artifact, list):
            citations = message.artifact
    return citations

# ========== 聊天接口 ==========
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    config = {
        "configurable": {
            "thread_id": req.thread_id
        },
        "recursion_limit": AGENT_RECURSION_LIMIT,
        # 每次请求一个新 handler：按 run_id 记起止时间，输出 llm / tool 的耗时
        "callbacks": [TraceCallbackHandler()],
    }

    # 1. 读取长期记忆（user 维度），渲染后注入本轮 system prompt
    memory_block = render_memories(await load_memories(app.state.pool, req.user_id))

    try:
        with trace_span("chat", thread_id=req.thread_id, user_id=req.user_id):
            result = await asyncio.wait_for(
                app.state.graph.ainvoke(
                    {
                        "messages": [{"role": "user", "content": req.message}],
                        "memory_block": memory_block,
                    },
                    config=config,
                ),
                timeout=AGENT_TIMEOUT,
            )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise AgentTimeoutError("图执行超时", cause=exc) from exc
    except Exception as exc:
        raise translate(exc) from exc

    new_msg = result["messages"][-1]
    reply = new_msg.content or ""

    # 2. 回答已经准备好了，记忆抽取放到响应发出之后跑，不占用用户等待时间
    background_tasks.add_task(
        extract_and_save, app.state.pool, req.user_id, req.message, reply
    )

    return ChatResponse(
        reply=reply,
        citations=citations_of_last_turn(result["messages"]),
    )

# ========== 流式聊天接口 ==========
def sse_pack(event: str, data: dict) -> str:
    """
    按 SSE 协议打包一条事件。

    格式固定为「event: 类型 + data: JSON + 空行」，空行是消息结束的分隔符，
    少了它前端会一直等不到这条消息。

    Args:
        event: 事件类型，前端按这个字段分流处理
        data: 事件负载，会被 json 序列化

    Returns:
        str: 可直接写入响应流的字符串
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

async def stream_chat(req: ChatRequest, request: Request):
    """
    把图的执行过程转成 SSE 事件流。

    同时订阅两种 stream_mode：
    - messages: LLM 逐 token 输出，用于打字机效果
    - updates: 节点执行完的状态更新，用来拿 ToolMessage.artifact 里的引用

    Args:
        req: 聊天请求
        request: FastAPI 请求对象，用于感知客户端断开

    Yields:
        str: SSE 格式的事件字符串
    """
    graph = app.state.graph
    config = {
        "configurable": {
            "thread_id": req.thread_id
        },
        "recursion_limit": AGENT_RECURSION_LIMIT,
        # 流式请求的 trace 靠它还原子调用和工具调用的耗时；
        # 整条流的总耗时由 RequestIDMiddleware 的访问日志覆盖
        "callbacks": [TraceCallbackHandler()],
    }

    # 长期记忆：流式接口同样注入（和 /chat 走同一套逻辑）
    memory_block = render_memories(await load_memories(app.state.pool, req.user_id))
    answer_parts: list[str] = []

    try:
        async for mode, data in graph.astream(
            {
                "messages": [{"role": "user", "content": req.message}],
                "memory_block": memory_block,
            },
            config=config,
            stream_mode=["messages", "updates"],
            timeout=AGENT_TIMEOUT,
        ):
            # 客户端提前断开时及时退出，避免继续白跑推理、继续消耗 token
            if await request.is_disconnected():
                return

            if mode == "messages":
                chunk, metadata = data
                # messages 模式会把 tools 节点的 ToolMessage 正文也吐出来，
                # 那段是知识库原文，不是给用户看的回答，必须过滤掉
                if metadata.get("langgraph_node") != "agent":
                    continue
                text = chunk.content
                if isinstance(text, str) and text:
                    answer_parts.append(text)
                    yield sse_pack("token", {"text": text})

            elif mode == "updates":
                for node, update in data.items():
                    for message in update.get("messages", []):
                        # 检索结果挂在 artifact 上，发完引用前端就能渲染脚注
                        if isinstance(message, ToolMessage) and isinstance(
                            message.artifact, list
                        ):
                            yield sse_pack("citations", {"citations": message.artifact})
        yield sse_pack("done", {})

        # 回答已经推给前端了（前端收到 done 就停止渲染），这里再抽记忆，
        # 用户感知不到额外等待
        await extract_and_save(
            app.state.pool, req.user_id, req.message, "".join(answer_parts)
        )
    except asyncio.CancelledError:
        # 客户端断开时 Starlette 会取消这个生成器，属于正常收尾，不要当错误上报
        raise
    except AppError as exc:
        yield sse_pack("error", {
            "code": exc.code,
            "message": exc.user_message,
            "retryable": exc.retryable,
            "request_id": _request_id(request),
        })
    except Exception as exc:
        # 流已经开始，HTTP 状态码已经发出去了，只能在流里报错
        err = translate(exc)  # 关键：不再把 str(exc) 发给前端
        logger.exception("流式对话失败 thread_id=%s", req.thread_id)
        yield sse_pack("error", {
            "code": err.code,
            "message": err.user_message,
            "retryable": err.retryable,
            "request_id": _request_id(request),
        })

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    return StreamingResponse(
        stream_chat(req, request),
        media_type="text/event-stream",    # text/event-stream 是 SSE 的标准 MIME 类型
        headers={
            # 关闭缓存，否则中间层可能攒够一批才发，打字机效果就没了
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 nginx 的响应缓冲，保证每个 token 立即到达浏览器
            "X-Accel-Buffering": "no",
        },
    )






















