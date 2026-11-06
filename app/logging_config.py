"""
统一的日志配置。

单行格式、每条都带 request_id，这样一次请求跨 checkpointer / 记忆读写 / 后台抽取
的所有日志能直接 grep 出来：

    2026-11-05 17:20:01,234 INFO    [a1b2c3d4e5f6] app.main: 收到请求
"""
import logging
import sys

import app.config as config
from app.context import get_request_id


class RequestIdFilter(logging.Filter):
    """把当前上下文里的 request_id 注入每条日志记录。

    如果调用方已经用 extra={"request_id": ...} 显式指定，就尊重它——
    线程池里跑的代码（比如同步工具）拿不到 ContextVar，只能靠显式传。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", ""):
            record.request_id = get_request_id() or "-"
        return True


# 这些第三方库在 INFO 级别非常吵，统一压到 WARNING
_NOISY_LOGGERS = {
    "httpx": logging.WARNING,
    "httpx2": logging.WARNING,   # langchain 内部 vendored 的 httpx，日志名不同
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "pymilvus": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "transformers": logging.WARNING,
    "jieba": logging.WARNING,
    "asyncio": logging.WARNING,
}


def setup_logging() -> None:
    """幂等：重复调用不会叠加 handler（reload、多次 import 都安全）。"""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(config.LOG_LEVEL)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(config.LOG_LEVEL)

    for name, level in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(level)
