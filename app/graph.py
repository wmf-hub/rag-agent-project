import logging
from typing import Annotated, TypedDict

import openai
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from app.config import get_llm
from app.rag.retriever import search_knowledge_base
from app.tools import get_current_time
from app.tracing import trace_span

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # 长期记忆的渲染文本，每轮由 API 层注入。不写进消息历史，只用于拼 system prompt
    memory_block: str


TOOLS = [get_current_time, search_knowledge_base]

SYSTEM_PROMPT = """你是一个基于知识库的问答助手。

规则：
1. 当用户询问需要参考资料、文档内容或事实性知识时，必须先调用 search_knowledge_base 工具检索。
2. 回答中凡引用知识库内容，必须保留原始标记 [doc1]、[doc2] 等，格式不可更改。
3. 如果知识库没有返回相关内容，如实告知用户，不要编造。
4. 不涉及知识库的问题（如时间、闲聊），正常回答。
5. 如果工具返回「不可用 / 失败」，直接把情况告诉用户，不要再重复调用同一个工具。"""


def should_continue(state: AgentState) -> str:
    """
    判断是否需要调用工具

    Args:
        state (AgentState): 包含消息的当前状态

    Returns:
        str: 下一个节点的名称，"tools" 或 END
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def handle_tool_error(exc: Exception) -> str:
    """
    工具异常的兜底：翻译成「模型能读懂、但不会泄密」的一句话。

    注意：参数必须标注成 Exception。ToolNode 会用 _infer_handled_types()
    从注解推断要捕获哪些异常，标成 MilvusException 就只会捕那一种。
    """
    logger.warning("工具执行失败: %s: %s", type(exc).__name__, exc)

    if isinstance(exc, OSError):          # 连接被拒 / 超时 / 断网
        return "知识库检索服务暂时不可用（连接失败）。请直接告知用户稍后再试，不要重试本工具。"
    return "知识库检索失败。请直接告知用户稍后再试，不要重试本工具。"


# 只重试「瞬时」错误。鉴权错、参数错重试没有意义，只会把失败拖慢。
LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=0.5,  # 初始重试间隔，单位秒
    backoff_factor=2.0,    # 每次重试间隔增加因子，例如 2.0 表示间隔0.5 -> 1.0 -> 2.0 -> ...
    max_interval=8.0,      # 最大重试间隔，单位秒
    max_attempts=3,        # 最大重试次数
    jitter=True,           # 开启随机抖动，避免同时重试
    retry_on=(             # 重试的异常类型，其他异常不重试
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    ),
)


def build_graph(checkpointer=None, llm=None):
    """
    编译图。

    所有节点/连接都在函数里创建，而不是模块级——这样：
    - import 本模块不会触发任何网络/模型初始化（没配 key 也能 import、能跑测试）
    - 可以注入假模型做单测：build_graph(memory_checkpointer, llm=FakeLLM())

    Args:
        checkpointer: 异步 checkpointer
        llm: 可注入的模型；不传则按 .env 配置创建

    Returns:
        CompiledStateGraph: 编译好的图
    """
    model = (llm or get_llm()).bind_tools(TOOLS)

    async def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
        """
        调用 DeepSeek 进行推理

        Args:
            state (AgentState): 包含消息的当前状态
            config (RunnableConfig): 运行时配置，透传给模型以便 callbacks/trace 生效

        Returns:
            AgentState: 包含新消息的状态
        """
        # 每轮都重建 system prompt：
        # 记忆是会变的，如果像以前那样「历史里没有 SystemMessage 才加一条」，
        # 第一轮注入的记忆会被 checkpoint 永久固定下来，后面更新了也不生效。
        history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

        memory_block = state.get("memory_block") or ""
        system_prompt = f"{SYSTEM_PROMPT}\n\n{memory_block}" if memory_block else SYSTEM_PROMPT

        # ainvoke 走原生异步，不占线程池；以前的 .invoke 是同步阻塞调用
        with trace_span("agent_node", history=len(history)):
            response = await model.ainvoke(
                [SystemMessage(content=system_prompt), *history], config=config
            )
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node, retry_policy=LLM_RETRY_POLICY)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=handle_tool_error))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)
