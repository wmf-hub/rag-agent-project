"""
纯 ASGI 中间件：给每个请求分配 request_id，写回响应头，并记一条访问日志。

为什么不用 @app.middleware("http")（BaseHTTPMiddleware）：
    它会给每个请求额外包一层 anyio task，在 SSE + request.is_disconnected()
    这种场景下有已知的兼容问题（客户端断开感知不到）。纯 ASGI 中间件不改变
    调用栈，行为和自己写路由一致。
"""
import logging
import time

from starlette.datastructures import MutableHeaders

from app.context import new_request_id, set_request_id

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 上游（nginx / 网关）已经带了就沿用，方便跨服务串联
        request_id = ""
        for key, value in scope.get("headers") or []:
            if key == b"x-request-id":
                request_id = value.decode("latin-1")[:64]
                break
        request_id = request_id or new_request_id()

        # 两处都放：
        # - scope["state"] 给异常处理器读（异常冒泡后 ContextVar 可能已被清掉）
        # - ContextVar 给日志 / trace 读
        scope.setdefault("state", {})["request_id"] = request_id
        set_request_id(request_id)

        status = {"code": 0}
        started = time.perf_counter()

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            # 容器 HEALTHCHECK 每 30s 打一次 /livez，别刷爆 INFO 日志；排障时把 LOG_LEVEL 调成 DEBUG
            log = logger.debug if scope.get("path") == "/livez" else logger.info
            log(
                "%s %s -> %s %.1fms",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status["code"] or "-",
                elapsed,
                extra={"request_id": request_id},
            )
