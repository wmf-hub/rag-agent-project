import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from app.rag.vector_store import ingest_documents  # noqa: E402

DOCS = [
    Document(page_content="年假制度：员工入职满一年享有5天带薪年假，满三年享有10天。", metadata={"source": "员工手册.pdf"}),
    Document(page_content="报销流程：所有报销需在费用发生后30天内提交，逾期不予受理。", metadata={"source": "财务制度.pdf"}),
    Document(page_content="产品A支持API接入，单次请求最大token数为4096。", metadata={"source": "产品文档.md"}),
]


def main():
    parser = argparse.ArgumentParser(description="把知识文档写入 Milvus")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先删除集合并重建（换 embedding 模型或切分参数时用）",
    )
    args = parser.parse_args()

    count = ingest_documents(DOCS, reset=args.reset)
    print(f"入库完成：{count} 条 chunk")


if __name__ == "__main__":
    main()
