"""
轻量 trace：不接外部服务，只把「谁、花了多久、成功还是失败」写进同一套日志。

一次请求的所有 span 都带同一个 request_id，所以能直接串起来：

    17:20:01 INFO [a1b2c3d4e5f6] app.trace: agent_node start messages=3
    17:20:02 INFO [a1b2c3d4e5f6] app.trace: llm ok 1490.2ms
    17:20:02 INFO [a1b2c3d4e5f6] app.trace: agent_node ok 1523.4ms

以后想升级成 LangSmith 不用改这里的代码：设了 LANGSMITH_API_KEY 之后
LangChain 会自动埋点，本地这套 span 日志可以继续留着做离线排查。
"""
import asyncio
import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler

import app.config as config
from app.context import get_request_id

logger = logging.getLogger("app.trace")


def _rid() -> str:
    return get_request_id() or "-"


@contextmanager
def trace_span(name: str, **fields: Any):
    """
    记录一段代码的耗时。

    异常会被记录后原样抛出，不吞掉——错误处理交给上层的 AppError 体系。
    """
    if not config.TRACE_ENABLED:
        yield
        return

    extra = {"request_id": _rid()}
    detail = " ".join(f"{k}={v}" for k, v in fields.items())
    started = time.perf_counter()
    logger.info("%s start %s", name, detail, extra=extra)

    try:
        yield
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        logger.warning(
            "%s failed %.1fms %s: %s",
            name,
            elapsed,
            type(exc).__name__,
            exc,
            extra=extra,
        )
        raise
    else:
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("%s ok %.1fms", name, elapsed, extra=extra)


def traceable(name: str | None = None) -> Callable:
    """
    给同步或异步函数套一个 span。

    用法：
        @traceable("extract_memory")
        async def extract_and_save(...): ...
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with trace_span(span_name):
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


class TraceCallbackHandler(BaseCallbackHandler):
    """
    把 LLM 和工具的调用也纳入 trace。

    挂在 config["callbacks"] 上即可，LangChain 会在每次调用前后回调这里。
    只需要开始/结束时间，所以按 run_id 记一个起始时间戳。
    """

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self._labels: dict[str, str] = {}

    def _begin(self, key: str, label: str, detail: str = "") -> None:
        if not config.TRACE_ENABLED:
            return
        self._starts[key] = time.perf_counter()
        self._labels[key] = label
        logger.info("%s start %s", label, detail, extra={"request_id": _rid()})

    def _finish(self, key: str) -> None:
        started = self._starts.pop(key, None)
        label = self._labels.pop(key, "unknown")
        if started is None:
            return
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("%s ok %.1fms", label, elapsed, extra={"request_id": _rid()})

    def _fail(self, key: str, error: BaseException) -> None:
        started = self._starts.pop(key, None)
        label = self._labels.pop(key, "unknown")
        elapsed = (time.perf_counter() - started) * 1000 if started else 0.0
        logger.warning(
            "%s failed %.1fms %s: %s",
            label,
            elapsed,
            type(error).__name__,
            error,
            extra={"request_id": _rid()},
        )

    @staticmethod
    def _model_name(serialized, invocation_params) -> str:
        """
        尽力拿到真实模型名。

        ChatOpenAI 把模型名放在 serialized["kwargs"]["model_name"] 里，
        直接打印整个 kwargs 会把 openai_api_key 也带进日志，所以只挑白名单字段。
        """
        for source in (invocation_params or {}, (serialized or {}).get("kwargs") or {}):
            for key in ("model_name", "model", "azure_deployment"):
                value = source.get(key)
                if value:
                    return str(value)
        ident = (serialized or {}).get("id") or []
        return str(ident[-1]) if ident else "llm"

    # ---- LLM ----
    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        model = self._model_name(serialized, kwargs.get("invocation_params"))
        self._begin(str(run_id), "llm", f"model={model}")

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        self._finish(str(run_id))

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        self._fail(str(run_id), error)

    # ---- 工具 ----
    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs) -> None:
        name = (serialized or {}).get("name", "tool")
        self._begin(str(run_id), f"tool.{name}")

    def on_tool_end(self, output, *, run_id=None, **kwargs) -> None:
        self._finish(str(run_id))

    def on_tool_error(self, error, *, run_id=None, **kwargs) -> None:
        self._fail(str(run_id), error)
