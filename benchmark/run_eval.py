"""
检索质量评测：纯向量 vs 混合检索 vs 混合 + 精排。

用法（在项目根目录）：
    python benchmark/run_eval.py

设计说明：
- 语料写入独立的临时集合 eval_kb，不碰业务集合 knowledge_base，跑完自动删除
- 四条链路共用同一个召回池（top_k=10），排序后统一截断比较，保证可比
- 指标在「文档粒度」上算：Top-N 里只要命中 gold 文档的任意 chunk 就算召回
- 语料 = 基础文档 + 近义干扰文档（分公司/海外/旧版规则，数字互相冲突），
  这样才能测出「混合检索抗术语漂移」和「精排抗近义干扰」的价值
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

import app.config as config  # noqa: E402
from app.rag import vector_store as vs  # noqa: E402
from app.rag.retriever import get_reranker  # noqa: E402

EVAL_COLLECTION = "eval_kb"
BENCH_DIR = Path(__file__).resolve().parent
CUTOFFS = (1, 3)

VARIANTS = [
    ("纯向量（dense）", "dense"),
    ("混合检索（dense + BM25, RRF）", "hybrid"),
    ("混合 + Cross-Encoder 精排", "rerank"),
    (f"混合 + 精排 + 阈值 {config.RERANK_SCORE_THRESHOLD}", "threshold"),
]


def _dense_search_with_anns_field(query_vec: list[float], top_k: int) -> list[dict]:
    """
    评测进程内替换 vs.search_dense。

    原因：生产版 search_dense 调用 client.search 时没有传 anns_field，
    而新 schema 同时有 dense(vector) 与 sparse(sparse) 两个向量字段，
    Milvus 会报 code=1100 "multiple anns_fields exist"。
    （线上默认走 hybrid 分支所以没暴露；把 HYBRID_ENABLED 关掉就会踩到。）

    这里只补一个 anns_field="vector"，检索行为与生产完全一致。
    """
    client = vs.get_milvus_client()
    vs.ensure_collection_loaded()
    results = client.search(
        collection_name=vs.COLLECTION_NAME,
        data=[query_vec],
        anns_field="vector",
        limit=top_k,
        output_fields=["content", "metadata"],
    )
    return results[0] if results else []


def load_json(name: str):
    with open(BENCH_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def source_of(hit: dict) -> str:
    entity = hit.get("entity") or {}
    metadata = entity.get("metadata") or {}
    return metadata.get("source") or ""


def rank_of(sources: list[str], gold: str, cutoff: int) -> int:
    """gold 在截断位置内的名次（从 1 开始），未命中返回 0。"""
    for i, source in enumerate(sources[:cutoff], 1):
        if source == gold:
            return i
    return 0


def evaluate(top_k: int) -> dict:
    corpus = load_json("corpus.json") + load_json("distractors.json")
    queries = load_json("queries.json") + load_json("queries_hard.json")

    # ---- 写入独立评测集合 ----
    vs.COLLECTION_NAME = EVAL_COLLECTION
    vs._hybrid_ready = None

    docs = [
        Document(page_content=item["content"], metadata={"source": item["source"]})
        for item in corpus
    ]
    chunk_count = vs.ingest_documents(docs, reset=True)
    print(f"语料：{len(corpus)} 篇 -> {chunk_count} 个 chunk（集合 {EVAL_COLLECTION}）")

    if not vs.hybrid_available():
        raise RuntimeError("评测集合没有 sparse 字段，混合检索不可用，检查建集合逻辑")

    embeddings = vs.get_embeddings()
    reranker = get_reranker()

    # 只在本进程内替换：生产版 search_dense 在新 schema 下缺 anns_field（见函数说明）
    vs.search_dense = _dense_search_with_anns_field

    records = []
    for i, item in enumerate(queries, 1):
        query, gold = item["query"], item["gold"]
        vec = embeddings.embed_query(query)

        dense_sources = [source_of(h) for h in vs.search_dense(vec, top_k)]
        hybrid_hits = vs.search_hybrid(query, vec, top_k)
        hybrid_sources = [source_of(h) for h in hybrid_hits]

        pairs = [(query, (h.get("entity") or {}).get("content", "")) for h in hybrid_hits]
        scores = reranker.score(pairs) if pairs else []
        ranked = sorted(zip(hybrid_hits, scores), key=lambda x: float(x[1]), reverse=True)

        rerank_sources = [source_of(h) for h, _ in ranked]
        threshold_sources = [
            source_of(h) for h, s in ranked if float(s) >= config.RERANK_SCORE_THRESHOLD
        ]

        record = {
            "query": query,
            "gold": gold,
            "type": item.get("type", "keyword"),
            "dense": dense_sources,
            "hybrid": hybrid_sources,
            "rerank": rerank_sources,
            "threshold": threshold_sources,
            "top1_score": float(ranked[0][1]) if ranked else 0.0,
        }
        records.append(record)

        d1 = rank_of(dense_sources, gold, 1)
        r1 = rank_of(rerank_sources, gold, 1)
        mark = "✓" if r1 else ("✗" if not rank_of(rerank_sources, gold, 3) else "△")
        fixed = " [精排纠正]" if (not d1 and r1) else ""
        print(f"  [{i:2d}/{len(queries)}] {mark} {query}{fixed}")

    summary = {
        "records": records,
        "corpus": len(corpus),
        "chunks": chunk_count,
        "queries": len(queries),
    }
    for _, key in VARIANTS:
        summary[key] = {
            "recall": {
                n: sum(1 for r in records if rank_of(r[key], r["gold"], n)) / len(records)
                for n in CUTOFFS
            },
            "mrr": sum(
                (1 / rank_of(r[key], r["gold"], 3)) if rank_of(r[key], r["gold"], 3) else 0
                for r in records
            )
            / len(records),
            "empty": sum(1 for r in records if not r[key]) / len(records),
        }

    for qtype in sorted({r["type"] for r in records}):
        subset = [r for r in records if r["type"] == qtype]
        for _, key in VARIANTS:
            summary[f"{key}::{qtype}"] = {
                n: sum(1 for r in subset if rank_of(r[key], r["gold"], n)) / len(subset)
                for n in CUTOFFS
            }
    return summary


def render_markdown(summary: dict) -> str:
    n1, n3 = CUTOFFS
    lines = [
        f"# 检索评测结果（语料 {summary['corpus']} 篇 / {summary['chunks']} chunk，query {summary['queries']} 条）",
        "",
        f"| 配置 | Recall@{n1} | Recall@{n3} | MRR@{n3} | 无结果率 |",
        "|---|---|---|---|---|",
    ]
    for label, key in VARIANTS:
        s = summary[key]
        lines.append(
            f"| {label} | {s['recall'][n1]*100:.1f}% | {s['recall'][n3]*100:.1f}% | "
            f"{s['mrr']:.3f} | {s['empty']*100:.1f}% |"
        )

    types = sorted({r["type"] for r in summary["records"]})
    lines += [
        "",
        f"## 按 query 类型拆分（Recall@{n1}）",
        "",
        f"| 类型 | 条数 | 纯向量 | 混合 + 精排 | 提升 |",
        "|---|---|---|---|---|",
    ]
    for qtype in types:
        count = sum(1 for r in summary["records"] if r["type"] == qtype)
        dense = summary[f"dense::{qtype}"][n1]
        rerank = summary[f"rerank::{qtype}"][n1]
        lines.append(
            f"| {qtype} | {count} | {dense*100:.1f}% | {rerank*100:.1f}% | {(rerank-dense)*100:+.1f} pt |"
        )

    fixed = [
        r
        for r in summary["records"]
        if not rank_of(r["dense"], r["gold"], 1) and rank_of(r["rerank"], r["gold"], 1)
    ]
    if fixed:
        lines += ["", f"## 精排纠正的 case（{len(fixed)} 条）", ""]
        for r in fixed:
            lines.append(
                f"- `{r['query']}` → dense 第 {rank_of(r['dense'], r['gold'], 10) or '未命中'} 名，"
                f"精排提到第 1 名（{r['gold']}）"
            )

    misses = [
        r for r in summary["records"] if not rank_of(r["rerank"], r["gold"], 3)
    ]
    if misses:
        lines += ["", f"## 未命中 Top-3（{len(misses)} 条）", ""]
        for r in misses:
            lines.append(
                f"- `{r['query']}` → 期望 `{r['gold']}`，实际：{r['rerank'][:3]}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--top-k", type=int, default=10, help="召回池大小")
    parser.add_argument("--keep", action="store_true", help="保留 eval_kb 集合，便于排查")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        summary = evaluate(args.top_k)
    finally:
        if not args.keep:
            try:
                vs.get_milvus_client().drop_collection(EVAL_COLLECTION)
            except Exception:
                pass
    cost = time.perf_counter() - started

    report = render_markdown(summary)
    print("\n" + report)
    print(f"耗时 {cost:.1f}s")

    out = BENCH_DIR / "latest_result.md"
    out.write_text(report, encoding="utf-8")
    print(f"结果已写入 {out.relative_to(ROOT)}")
    if not args.keep:
        print(f"已清理临时集合 {EVAL_COLLECTION}")


if __name__ == "__main__":
    main()
