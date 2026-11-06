import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# ===== 模型 / 组件 =====
# 大语言模型配置
MODEL = os.getenv("LLM_MODEL")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# 文本分割配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# 向量模型配置
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))

# Milvus 向量数据库配置
# 优先用远程服务地址（http://host:19530）；为空时才退回本地 milvus-lite 文件库
MILVUS_URI = os.getenv("MILVUS_URI", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
if not MILVUS_URI:
    db_path = os.getenv("MILVUS_DB_PATH")
    if not db_path:
        raise RuntimeError(
            "未配置 Milvus 连接方式：请在 .env 中填写 MILVUS_URI=http://localhost:19530（远程服务），"
            "或填写 MILVUS_DB_PATH=E:/milvus_data/milvus.db（本地 milvus-lite 文件库）"
        )
    MILVUS_URI = str((PROJECT_ROOT / db_path).resolve())
if not MILVUS_URI.startswith("http") and any(ord(ch) > 127 for ch in MILVUS_URI):
    raise RuntimeError(
        f"本地 Milvus 库路径不能包含非 ASCII 字符（faiss 无法写入）：{MILVUS_URI}\n"
        "请把 MILVUS_DB_PATH 改为纯英文路径，例如 E:/milvus_data/milvus.db"
    )

# Rerank 模型配置
RERANK_MODEL = PROJECT_ROOT / os.getenv("RERANK_MODEL")
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.3"))

# 混合检索配置（向量检索 + BM25 关键词检索，再由 ranker 融合）
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() in ("1", "true", "yes")
HYBRID_RANKER = os.getenv("HYBRID_RANKER", "rrf").lower()
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
HYBRID_SPARSE_WEIGHT = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))
BM25_TOKENIZER = os.getenv("BM25_TOKENIZER", "jieba")
CONTENT_MAX_LENGTH = int(os.getenv("CONTENT_MAX_LENGTH", "8192"))

# PostgresSQL 数据库配置（短期记忆）
POSTGRES_DB_URI = "postgresql://{}:{}@localhost:5432/{}".format(os.getenv("POSTGRES_USER"), os.getenv("POSTGRES_PASSWORD"), os.getenv("POSTGRES_DB"))

# ===== 稳定性 / 错误处理 =====
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "90"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "12"))

# ===== 长期记忆（跨会话的用户事实与偏好） =====
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() in ("1", "true", "yes")
MEMORY_MAX_ITEMS = int(os.getenv("MEMORY_MAX_ITEMS", "20"))

# ===== 服务 / 运行方式 =====
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_RELOAD = os.getenv("APP_RELOAD", "false").lower() in ("1", "true", "yes")

# ===== 日志 =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s",
)

# ===== Postgres 连接池 =====
# psycopg_pool 默认 min_size=4 且 max_size 跟随 min_size，也就是最多 4 条连接。
# SSE 长连接 + 后台记忆抽取并发时很容易 PoolTimeout，这里显式放大。
PG_POOL_MIN_SIZE = int(os.getenv("PG_POOL_MIN_SIZE", "2"))
PG_POOL_MAX_SIZE = int(os.getenv("PG_POOL_MAX_SIZE", "16"))
PG_POOL_TIMEOUT = float(os.getenv("PG_POOL_TIMEOUT", "10"))

# ===== 可观测性 =====
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "true").lower() in ("1", "true", "yes")
# 启动时预热 rerank 模型：把 1GB 模型的加载成本从第一个用户请求里挪到启动阶段
RERANK_PRELOAD = os.getenv("RERANK_PRELOAD", "true").lower() in ("1", "true", "yes")



def get_llm(temperature: float = 0):
    """获取大语言模型实例"""
    return ChatOpenAI(
        model=MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,  # 429/5xx/超时由 openai SDK 自动退避重试
    )




















