# ---------------------------------------------------------------------------
# 构建：
#   docker build -t agent-project:latest .
# 运行（模型权重不打进镜像，用 -v 挂载）：
#   docker run --rm -p 8000:8000 --env-file .env \
#     -v "${PWD}/app/rag/models:/app/app/rag/models:ro" \
#     --add-host=host.docker.internal:host-gateway \
#     agent-project:latest
#
# 容器里要用「容器视角」的地址，别直接照抄 .env：
#   MILVUS_URI=http://milvus-standalone:19530        （与 docker-compose 同一网络）
#   OLLAMA_BASE_URL=http://host.docker.internal:11434
#   POSTGRES_* / MILVUS_DB_PATH 用的是宿主机 localhost，容器内连不到
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# torch 运行时需要 libgomp1；curl 供 HEALTHCHECK 使用
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# 先装 CPU 版 torch，再把 requirements 里同名的 torch 行剔除：
# 否则 pip 认为 2.14.1+cpu 不满足 "torch==2.14.1"，会重新拉 PyPI 上的 CUDA 版（2GB+）。
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && grep -v -E '^torch==' requirements.txt > /tmp/requirements-notorch.txt \
 && pip install -r /tmp/requirements-notorch.txt

COPY app ./app
COPY run.py ./

# 不用 root 跑
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 探活只看进程是否还能响应，不探 Milvus/Ollama——依赖挂了不该把容器反复重启
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/livez || exit 1

CMD ["python", "run.py"]
