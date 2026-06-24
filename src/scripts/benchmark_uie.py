"""
UIE 推理速度基准测试

测试微调后 UIE 模型在电商短文本上的推理吞吐量，
统计平均每秒可处理的文本条数及平均耗时。
"""

import sys
import time
from pathlib import Path

# 将 src 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config

# 导入 uie_pytorch
sys.path.insert(0, str(config.EXTERNAL_LIB_DIR / 'uie_pytorch'))

from uie_predictor import UIEPredictor

# ======================== 配置 ========================
FINETUNED_MODEL_PATH = str(config.CHECKPOINT_DIR / 'uie_0608' / 'model_best')

# 电商短文本测试集（平均长度约30字符）
TEST_TEXTS = [
    "小米12S Ultra 12GB+256GB 冷杉绿 5G手机",
    "Apple iPhone 12 (A2404) 64GB 黑色 双卡双待手机",
    "Redmi 10X 4G 4GB+128GB 冰雾白 游戏智能手机",
    "华为Mate 50 Pro 8GB+256GB 昆仑破晓 鸿蒙手机",
    "OPPO Reno9 Pro 16GB+256GB 皓月黑 5G手机",
    "vivo X90 Pro+ 12GB+256GB 原黑 蔡司影像旗舰",
    "联想拯救者Y9000P 2022 i9-12900H RTX3060 钛晶灰",
    "Apple MacBook Pro 14英寸 M2 Pro 16GB+512GB 深空灰",
    "TCL 85Q6 85英寸 4K超高清 液晶平板电视机",
    "小米电视4A 70英寸 4K超高清 智能网络液晶平板教育电视",
    "Nike耐克男士Air Max运动鞋 黑白配色 气垫缓震",
    "Adidas阿迪达斯三叶草经典贝壳头板鞋 白色 男女同款",
    "美的空调挂机 1.5匹变频冷暖 新一级能效 白色壁挂式",
    "格力空调 3匹变频柜机 新一级能效 客厅立柜式 白色",
    "海尔冰箱 510升对开门 风冷无霜 变频节能 银色",
    "CAREMiLLE珂曼奶油小方口红 雾面滋润保湿持久丝缎唇膏",
    "Sony索尼WH-1000XM5 头戴式无线降噪蓝牙耳机 黑色",
    "小米手环7 NFC版 血氧检测 智能运动手环 夜跃黑",
    "戴尔灵越15 i5-1240P 16GB+512GB 15.6英寸轻薄笔记本",
    "荣耀Magic5 Pro 12GB+256GB 亮黑色 5G手机",
]

# 实体抽取 schema
ENTITY_SCHEMA = ['商品', '品牌', '颜色', '品类', '运行内存', '机身内存', '版本']

# 关系抽取 schema
RELATION_SCHEMA = {'商品': ['品牌', '颜色', '品类', '运行内存', '机身内存', '版本']}


def benchmark(ie, texts, schema, task_name, num_rounds=5):
    """
    对给定的 schema 和文本列表进行多轮推理测速。

    Args:
        ie: UIEPredictor 实例
        texts: 测试文本列表
        schema: 抽取 schema
        task_name: 任务名称（用于打印）
        num_rounds: 测试轮数（取平均值）
    """
    ie.set_schema(schema)

    avg_len = sum(len(t) for t in texts) / len(texts)
    total_texts = len(texts) * num_rounds

    print(f"\n{'─' * 50}")
    print(f"任务: {task_name}")
    print(f"文本数量: {len(texts)} 条 × {num_rounds} 轮 = {total_texts} 条")
    print(f"平均文本长度: {avg_len:.1f} 字符")
    print(f"{'─' * 50}")

    # 预热（第一次推理通常较慢）
    _ = ie(texts[0])

    # 正式测试
    latencies = []
    for round_idx in range(num_rounds):
        start = time.perf_counter()
        for text in texts:
            _ = ie(text)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        throughput = len(texts) / elapsed
        print(f"  第 {round_idx + 1} 轮: {elapsed:.3f}s, {throughput:.2f} 条/秒")

    total_time = sum(latencies)
    avg_throughput = total_texts / total_time
    avg_latency = total_time / total_texts * 1000  # ms

    print(f"\n  ■ 平均吞吐量: {avg_throughput:.2f} 条/秒")
    print(f"  ■ 平均单条耗时: {avg_latency:.1f} ms")
    print(f"  ■ 总耗时: {total_time:.3f}s ({total_texts} 条)")

    return avg_throughput, avg_latency


def main():
    print("=" * 60)
    print("UIE 推理速度基准测试")
    print("=" * 60)

    # 加载微调后模型
    print(f"\n加载模型: {FINETUNED_MODEL_PATH}")
    ie = UIEPredictor(model='uie-base', task_path=FINETUNED_MODEL_PATH, schema=[])
    print("模型加载完成")

    # 实体抽取测速
    ent_throughput, ent_latency = benchmark(
        ie, TEST_TEXTS, ENTITY_SCHEMA, "实体抽取", num_rounds=5)

    # 关系抽取测速
    rel_throughput, rel_latency = benchmark(
        ie, TEST_TEXTS, RELATION_SCHEMA, "关系抽取", num_rounds=5)

    # 汇总
    avg_len = sum(len(t) for t in TEST_TEXTS) / len(TEST_TEXTS)
    print(f"\n{'=' * 60}")
    print("测试汇总")
    print(f"{'=' * 60}")
    print(f"  模型路径: {FINETUNED_MODEL_PATH}")
    print(f"  测试文本数: {len(TEST_TEXTS)} 条")
    print(f"  平均文本长度: {avg_len:.1f} 字符")
    print(f"  实体抽取吞吐量: {ent_throughput:.2f} 条/秒 (单条 {ent_latency:.1f}ms)")
    print(f"  关系抽取吞吐量: {rel_throughput:.2f} 条/秒 (单条 {rel_latency:.1f}ms)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

