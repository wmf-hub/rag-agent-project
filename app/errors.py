"""
统一的业务异常与第三方异常翻译。

原则：
1. 只把「可以告诉用户」的话放进 user_message；堆栈、连接串、内部路径一律只进日志。
2. 每个异常自带 code / http_status / retryable，API 层直接映射。
"""
import logging
import openai
import psycopg
import psycopg_pool
from langgraph.errors import GraphRecursionError

logger = logging.getLogger(__name__)

class AppError(Exception):
    """可以直接映射成 HTTP 响应的业务异常基类。"""

    code = "internal_error"                 # 机器可读的错误码，前端/日志据此分支
    http_status = 500                       # 该异常映射成的 HTTP 状态码
    retryable = False                       # 客户端重试是否有意义
    user_message = "服务内部错误，请稍后重试。"  # 唯一允许返回给用户的文案

    def __init__(self, detail: str = "", *, cause: Exception | None = None):
        super().__init__(detail or self.user_message)
        self.detail = detail          # 给日志看
        self.cause = cause            # 保留原始异常，便于 exc_info

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.user_message,
                "retryable": self.retryable,
            }
        }


class UpstreamLLMError(AppError):
    code = "llm_unavailable"
    http_status = 503
    retryable = True
    user_message = "模型服务暂时不可用，请稍后重试。"


class AgentTimeoutError(AppError):
    code = "agent_timeout"
    http_status = 504
    retryable = True
    user_message = "处理超时，请简化问题后重试。"


class AgentStepLimitError(AppError):
    code = "agent_step_limit"
    http_status = 422
    retryable = False
    user_message = "这个问题需要太多轮工具调用，请换个问法。"


class StorageError(AppError):
    code = "storage_unavailable"
    http_status = 503
    retryable = True
    user_message = "会话存储暂时不可用，请稍后重试。"


class ConfigError(AppError):
    code = "config_error"
    http_status = 500
    retryable = False
    user_message = "服务配置有误，请联系管理员。"


def translate(exc: Exception) -> AppError:
    """把第三方异常翻译成 AppError。认不出来的一律按 500 处理（只记日志）。"""
    if isinstance(exc, AppError):
        return exc

    # ---- 编排层 ----
    if isinstance(exc, GraphRecursionError):
        return AgentStepLimitError(str(exc), cause=exc)

    # ---- LLM 上游：注意顺序，子类要排在父类前面 ----
    if isinstance(exc, openai.AuthenticationError):
        return ConfigError("LLM 鉴权失败，请检查 LLM_API_KEY", cause=exc)
    if isinstance(exc, openai.RateLimitError):
        return UpstreamLLMError("LLM 限流(429)", cause=exc)
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return UpstreamLLMError("LLM 连接失败或超时", cause=exc)
    if isinstance(exc, openai.APIStatusError):
        return UpstreamLLMError(f"LLM 上游返回 {exc.status_code}", cause=exc)

    # ---- 存储 ----
    if isinstance(exc, (psycopg.OperationalError, psycopg_pool.PoolTimeout)):
        return StorageError("Postgres 不可用", cause=exc)

    return AppError(f"{type(exc).__name__}: {exc}", cause=exc)