import logging
import threading
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import app.config as config
from app.rag.vector_store import (
    get_embeddings,
    search_dense,
    search_hybrid,
    hybrid_available,
)

logger = logging.getLogger(__name__)

MODEL_NAME = str(config.RERANK_MODEL)

# 模型单例 + 初始化锁
_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    """
    线程安全地加载 rerank 模型。

    双重检查：并发请求同时进来时只会真正加载一次，
    避免 1GB 的模型被两个线程各加载一份（内存直接翻倍）。
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                logger.info("加载 rerank 模型: %s", MODEL_NAME)
                _reranker = HuggingFaceCrossEncoder(model_name=MODEL_NAME)
                logger.info("rerank 模型加载完成")
    return _reranker


def preload_reranker() -> None:
    """启动预热：把模型加载成本从第一个用户请求里挪到启动阶段。"""
    get_reranker()

def _search(query: str, top_k: int = 10) -> list[dict]:
    """
    召回候选：混合检索（dense + BM25）优先，不可用时退化为纯向量检索

    Args:
        query: 查询文本
        top_k: 返回 Top-K 的候选结果

    Returns:
        list[dict]: 包含 Top-K 结果的列表，每个元素是一个包含内容和元数据的字典
    """
    query_vec = get_embeddings().embed_query(query)

    # 自动降级
    if config.HYBRID_ENABLED and hybrid_available():
        return search_hybrid(query, query_vec, top_k)

    return search_dense(query_vec, top_k)

def _rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 3,
    score_threshold: float | None = None,
) -> list[tuple[dict, float]]:
    """
    对检索到的候选进行 rerank 精排

    Args:
        query: 查询文本
        candidates: 检索到的候选列表
        top_n: 返回 Top-N 的精排结果
        score_threshold: 相关度阈值，低于该分数的候选直接丢弃；None 表示使用配置值

    Returns:
        list[tuple[dict, float]]: 包含 Top-N 结果的列表，每个元素是一个包含候选和分数的元组
    """
    if score_threshold is None:
        score_threshold = config.RERANK_SCORE_THRESHOLD

    # 通过单例取模型：首次调用会加载，之后直接复用
    reranker = get_reranker()

    # Cross-Encoder 计算每个候选的分数
    pairs = [(query, candidate["entity"]["content"]) for candidate in candidates]
    scores = list(reranker.score(pairs))

    # 先按阈值过滤，再按分数降序取 Top-N：
    # 先截断会让「3 条全不相关」照样返回，过滤后为空才能让上层走「未找到」分支
    scored = sorted(
        (
            (candidate, float(score))
            for candidate, score in zip(candidates, scores)
            if float(score) >= score_threshold
        ),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return scored

def _search_and_rerank(
    query: str,
    top_k: int = 10,
    top_n: int = 3,
    score_threshold: float | None = None,
) -> tuple[list[dict], str]:
    """
    内部函数：Milvus 初检 + Rerank 精排

    Args:
        query: 查询文本
        top_k: 向量检索返回的 Top-K 数量
        top_n: rerank返回的 Top-N 数量
        score_threshold: 相关度阈值，低于该分数的候选直接丢弃；None 表示使用配置值

    Returns:
        tuple[list[dict], str]: Top-N 的结构化结果列表，和一个包含 Top-N 结果的带引用标记的字符串结果
    """
    candidates = _search(query, top_k)

    # 空集合 / 无候选时直接返回：CrossEncoder 对空列表打分没有意义，
    # 某些版本还会直接抛异常
    if not candidates:
        return [], ""

    scored = _rerank(query, candidates, top_n, score_threshold)

    # 全部候选都低于阈值：视为知识库没有相关内容
    if not scored:
        return [], ""

    docs = []
    contexts = []
    for i, (candidate, score) in enumerate(scored, 1):
        content = candidate["entity"]["content"]
        source = candidate["entity"]["metadata"]["source"]
        docs.append(
            {
                "content": content,
                "source": source,
                "score": score,
            }
        )
        contexts.append(f"[doc{i}] 来源：{source}\n内容：{content}")

    return docs, "\n\n".join(contexts)


@tool
def search_knowledge_base(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """
    搜索内部知识库。当用户询问需要参考资料、文档或事实性知识的问题时调用此工具。
    返回带 [docN] 引用标记的文档片段。回答时请保留这些引用标记。

    Args:
        query: 查询文本
        tool_call_id: 工具调用 ID，用于在 checkpoint 中存储结果

    Returns:
        ToolMessage: 返回包含检索结果的 ToolMessage，检索结果包含引用标记
    """
    docs, context = _search_and_rerank(query, top_k=10, top_n=3)

    if not docs:
        context = "知识库中未找到相关内容。"

    # 关键点：citations 不放进模块级全局变量，也不塞进 state，
    # 而是挂在 ToolMessage.artifact 上。
    # - artifact 不会发给大模型（不污染上下文、不浪费 token）
    # - 它跟着这一条 ToolMessage 存在 checkpoint 里，天然按 thread 隔离
    # - 业务层从本轮最后一条用户消息之后的消息里取，就不会串号
    return ToolMessage(
        content=context,
        tool_call_id=tool_call_id,
        artifact=docs,
    )