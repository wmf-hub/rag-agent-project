"""
启动入口：python run.py
前端启动：python -m http.server 8080

为什么需要它（而不是直接 `uvicorn app.main:app`）：

1. **Windows 事件循环**：psycopg 的异步连接不支持 ProactorEventLoop。
   uvicorn 只有在 `--reload` 或多 worker 时才会自己切到 SelectorEventLoop，
   直接 `uvicorn app.main:app` 会在 lifespan 里连不上数据库。
   这里用 `asyncio.run(..., loop_factory=...)` 显式指定，跨平台都稳。
2. **日志统一**：关掉 uvicorn 自带的 log_config，全部交给 app.logging_config，
   这样每条日志都带 request_id。
3. **访问日志**：交给 RequestIDMiddleware（带 request_id），关掉 uvicorn 的 access_log。
"""
import asyncio
import sys

import uvicorn

from app.config import APP_HOST, APP_PORT, APP_RELOAD
from app.logging_config import setup_logging


def main() -> None:
    setup_logging()

    # 热重载必须交给 uvicorn 自己的子进程机制；此时它内部会自动选用 SelectorEventLoop
    if APP_RELOAD:
        uvicorn.run(
            "app.main:app",
            host=APP_HOST,
            port=APP_PORT,
            reload=True,
            log_config=None,
        )
        return

    # 延迟导入：确保 setup_logging() 先于 app 模块里的模块级日志执行
    from app.main import app

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=APP_HOST,
            port=APP_PORT,
            log_config=None,   # 用我们自己的 logging 配置
            access_log=False,  # 访问日志由 RequestIDMiddleware 输出
        )
    )

    # Python 3.12 起 asyncio.run 支持 loop_factory
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    asyncio.run(server.serve(), loop_factory=loop_factory)


if __name__ == "__main__":
    main()
