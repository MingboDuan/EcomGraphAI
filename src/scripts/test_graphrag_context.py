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


SCENARIOS = {
    "cosmetic": [
        "有没有带保湿功能的口红，都是什么品牌的，帮我详细介绍下",
        "那兰蔻的有哪些？",
        "它们都是什么颜色的？",
    ],
    "behavior": [
        "用户51收藏过哪些商品？",
        "里面有没有电视？",
        "这个电视是什么品牌和尺寸？",
    ],
    "tv": [
        "你家是否有索尼的平板电视的？都是多少尺寸的？",
        "有没有55英寸的？",
        "这个品牌还有其它尺寸吗？",
    ],
}


def short_text(text: str, max_len: int = 220) -> str:
    """压缩多行文本，便于在命令行查看上下文测试结果。"""
    text = " ".join(str(text or "").split())
    return text if len(text) <= max_len else text[:max_len] + "..."


def print_round_result(index: int, query: str, result: dict) -> None:
    """打印单轮对话的核心调试信息。"""
    step_status = result.get("step_status", {})
    memory_detail = step_status.get("上下文记忆", {}).get("detail", "")
    intent = result.get("intent", {})
    entities = result.get("entities", [])
    records = result.get("records", [])

    print(f"\n========== 第 {index} 轮 ==========")
    print(f"用户问题: {query}")
    print(f"上下文记忆: {memory_detail}")
    print(f"实际检索问题: {result.get('retrieval_query', query)}")
    print(f"意图识别: {intent.get('intent')} | {intent.get('reason', '')}")
    print(f"实体数量: {len(entities)} | {entities}")
    print(f"查询结果数: {len(records)}")
    print(f"trace_path: {result.get('trace_path')}")
    print(f"回答摘要: {short_text(result.get('answer', ''))}")


def run_context_test(scenario: str, conversation_id: str, user_id: int) -> None:
    """在同一个 GraphRAGPipeline 实例中连续提问，验证上下文记忆是否生效。"""
    queries = SCENARIOS[scenario]
    pipeline = GraphRAGPipeline()
    try:
        print("GraphRAG 上下文记忆测试")
        print(f"scenario: {scenario}")
        print(f"conversation_id: {conversation_id}")
        print(f"user_id: {user_id}")
        print("说明: 第 1 轮应显示“无历史上下文”，第 2 轮开始应显示“已读取历史上下文”。")
        for index, query in enumerate(queries, 1):
            result = pipeline.run(query, conversation_id=conversation_id, user_id=user_id)
            print_round_result(index, query, result)
    finally:
        pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test GraphRAG context memory in one Python process.")
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="cosmetic",
        help="内置测试场景: cosmetic / behavior / tv",
    )
    parser.add_argument("--conversation_id", default="ctx-demo")
    parser.add_argument("--user_id", type=int, default=51)
    args = parser.parse_args()
    run_context_test(args.scenario, args.conversation_id, args.user_id)


if __name__ == "__main__":
    main()

