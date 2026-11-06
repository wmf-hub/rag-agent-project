"""请求级上下文：目前只放 request_id，供日志和 trace 把同一次请求的日志串起来。"""
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    """生成一个短 id（12 位十六进制），足够定位、又不至于把日志撑长。"""
    return uuid.uuid4().hex[:12]


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()
