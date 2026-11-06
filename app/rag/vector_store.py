import hashlib
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)
from pymilvus.client.types import LoadState
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import app.config as config

CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
MILVUS_URI = config.MILVUS_URI
EMBEDDING_DIM = config.EMBEDDING_DIM
COLLECTION_NAME = config.COLLECTION_NAME

# 全局单例
_client: MilvusClient | None = None
_embeddings: OllamaEmbeddings | None = None
_hybrid_ready: bool | None = None

def get_embeddings():
    """获取嵌入模型"""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    return _embeddings

def get_milvus_client():
    """获取 Milvus 客户端"""
    global _client
    if _client is None:
        _client = MilvusClient(MILVUS_URI)
    return _client

def ensure_collection_loaded():
    """确保集合已加载到内存（检索前必须处于 Loaded 状态）"""
    client = get_milvus_client()
    if not client.has_collection(COLLECTION_NAME):
        return
    state = client.get_load_state(COLLECTION_NAME)["state"]
    if state != LoadState.Loaded:
        client.load_collection(COLLECTION_NAME)

def _build_schema():
    """
    构建混合检索用的显式 schema。

    与 fast-create 的差别：
    - content 显式声明为 VARCHAR 并开启 enable_analyzer，BM25 只能基于带分词器的字段生成
    - sparse 是 BM25 函数的输出字段，由 Milvus 在写入时自动生成，调用方不要自己填
    - 关闭 enable_dynamic_field，避免 content 既做显式字段又做动态字段造成歧义
    """
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=config.CONTENT_MAX_LENGTH,
        enable_analyzer=True,
        analyzer_params={"tokenizer": config.BM25_TOKENIZER},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("metadata", DataType.JSON)
    schema.add_function(
        Function(
            name="content_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse"],
        )
    )
    return schema

def _build_index_params():
    """为 dense 与 sparse 两个向量字段分别建索引"""
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        "vector", index_type="AUTOINDEX", metric_type="COSINE"
    )
    # BM25 字段必须用 SPARSE_INVERTED_INDEX，且 metric 固定为 BM25
    index_params.add_index(
        "sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
    )
    return index_params

def init_collection():
    """初始化 Milvus 集合"""
    client = get_milvus_client()
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=_build_schema(),
            index_params=_build_index_params(),
        )
    ensure_collection_loaded()

def reset_collection():
    """删除集合再重建，用于全量重建索引（换 embedding 模型、换切分参数时必须用）"""
    global _hybrid_ready
    client = get_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    _hybrid_ready = None
    init_collection()

def hybrid_available() -> bool:
    """
    判断当前集合是否具备混合检索能力（存在 sparse 字段）。

    老集合是用 fast-create 建的，只有 vector 字段，硬走 hybrid_search 会报字段不存在，
    这里做一次探测并缓存，让新旧集合都能跑。
    """
    global _hybrid_ready
    if _hybrid_ready is None:
        client = get_milvus_client()
        if not client.has_collection(COLLECTION_NAME):
            _hybrid_ready = False
        else:
            # describe_collection 返回的 fields 是 dict 列表，不是 FieldSchema 对象
            fields = {
                f["name"] if isinstance(f, dict) else f.name
                for f in client.describe_collection(COLLECTION_NAME)["fields"]
            }
            _hybrid_ready = "sparse" in fields and "content" in fields
    return _hybrid_ready

def search_dense(query_vec: list[float], top_k: int) -> list[dict]:
    """纯向量 dense（语义） 检索"""
    client = get_milvus_client()
    ensure_collection_loaded()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        # 集合同时有 dense(vector) 与 sparse(sparse) 两个向量字段，
        # 不显式指定 anns_field 时 Milvus 无法判断该搜哪个，会报 code=1100
        anns_field="vector",
        limit=top_k,
        output_fields=["content", "metadata"],
    )
    return results[0] if results else []

def search_hybrid(query: str, query_vec: list[float], top_k: int) -> list[dict]:
    """
    混合检索：dense（语义）+ sparse（BM25 关键词），再由 ranker 融合。

    Args:
        query: 原始查询文本，BM25 分支直接用它做分词检索
        query_vec: 查询向量，dense 分支用
        top_k: 每条路各自召回数量，也是融合后的返回数量

    Returns:
        list[dict]: 与 client.search 同构的结果（含 id / distance / entity）
    """
    client = get_milvus_client()
    ensure_collection_loaded()

    dense_req = AnnSearchRequest(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=top_k,
    )
    # sparse 分支的 data 必须是原始文本：milvus 内部会用同一套分词器算 TF
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=top_k,
    )

    if config.HYBRID_RANKER == "weighted":
        ranker = WeightedRanker(
            config.HYBRID_DENSE_WEIGHT, config.HYBRID_SPARSE_WEIGHT
        )
    else:
        ranker = RRFRanker(config.HYBRID_RRF_K)

    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=ranker,
        limit=top_k,
        output_fields=["content", "metadata"],
    )
    return results[0] if results else []

def _chunk_id(content: str) -> int:
    """
    用 chunk 内容的哈希生成稳定主键。Milvus 的 id 是 int64 主键（auto_id=False）。同一条 chunk 永远得到同一个 id，配合 upsert 就变成幂等导入。

    Args:
        content (str): chunk 内容

    Returns:
        int: 稳定主键
    """
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)   # 60 bit，落在 int64 安全范围内

def ingest_documents(docs: list[Document], reset: bool = False) -> int:
    """
    文档入库函数：接收原始文档列表，分块、向量化、写入 Milvus。

    Args:
        docs (list[Document]): 语料文档列表，每个文档为 Document 对象
        reset (bool): 是否先删除集合并重建（全量重建索引时置 True）

    Returns:
        int: 入库的实体数量
    """
    if reset:
        reset_collection()

    client = get_milvus_client()
    init_collection()
    embeddings = get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", "。", "？", "！", " ", ""],
        length_function=len
    )

    # 用 dict 按 id 去重：同一批文档里出现完全相同的 chunk 时只保留一条
    entities: dict[int, dict] = {}
    for doc in docs:
        # 分块
        chunks = splitter.split_text(doc.page_content)
        if not chunks:
            continue
        # 向量化
        vectors = embeddings.embed_documents(chunks)
        # 格式化数据
        for chunk, vec in zip(chunks, vectors):
            uid = _chunk_id(chunk)
            if uid in entities:
                continue
            entities[uid] = {
                "id": uid,
                "vector": vec,
                "content": chunk,
                "metadata": {
                    "source": doc.metadata.get("source", "")
                },
            }

    data = list(entities.values())
    if not data:
        return 0

    # upsert 而不是 insert：主键已存在则覆盖，不存在则插入
    # 注意：sparse 字段是 BM25 函数生成的，这里不要传，否则会被判为非法输入
    client.upsert(collection_name=COLLECTION_NAME, data=data)
    # 落盘，确保插入数据立即可被检索到
    client.flush(COLLECTION_NAME)

    return len(data)