import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    path
    for path in sys.path
    if not path or Path(path).resolve() != SCRIPT_DIR
]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphrag.pipeline import GraphRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Test ecommerce Graph RAG pipeline.")
    parser.add_argument("query", nargs="?", default="推荐几款我可能喜欢的手机")
    parser.add_argument("--conversation_id", default="cli-demo")
    parser.add_argument("--user_id", type=int, default=51)
    args = parser.parse_args()

    pipeline = GraphRAGPipeline()
    try:
        result = pipeline.run(args.query, args.conversation_id, args.user_id)
    finally:
        pipeline.close()

    print("GraphRAG 调试执行完成")
    print(f"records: {len(result.get('records', []))}")
    print(f"trace_path: {result['trace_path']}")


if __name__ == "__main__":
    main()

