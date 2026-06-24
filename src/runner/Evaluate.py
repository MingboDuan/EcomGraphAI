"""
拼写纠错模型评估脚本

支持三种模型，在测试集上进行严格句子粒度评估。

用法:
python src/runner/Evaluate.py --model bert
python src/runner/Evaluate.py --model t5
python src/runner/Evaluate.py --model t5_base
"""
import os
import sys
import time
import argparse
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from configs import config
from models.spell_check_bert import SpellCheckBert
from models.spell_check_t5 import SpellCheckT5
from runner.Predictor import SpellCheckBertPredictor, SpellCheckT5Predictor


# ======================== 评估指标 ========================
def compute_sentence_level_metrics(input_texts, label_texts, pred_texts):
    """
    严格句子粒度评估（参考 pycorrector / SIGHAN 评估标准）。

    正负样本定义：
    - 正样本：src != tgt（原句有错，需要纠错）
    - 负样本：src == tgt（原句无错，不需要纠错）

    四种判定：
    - TP: 正样本 且 模型纠正后与 tgt 完全相同
    - FN: 正样本 但 模型纠正后与 tgt 不同（纠错失败）
    - TN: 负样本 且 模型输出与 tgt 完全相同（正确保持）
    - FP: 负样本 但 模型输出与 tgt 不同（误改）

    评估指标:
    1. accuracy:   (TP + TN) / total
    2. precision:  TP / (TP + FP)
    3. recall:     TP / (TP + FN)
    4. f1:         2 * P * R / (P + R)
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for src, tgt, pred in zip(input_texts, label_texts, pred_texts):
        if src != tgt:
            # 正样本：原句有错
            if pred == tgt:
                TP += 1
            else:
                FN += 1
        else:
            # 负样本：原句无错
            if pred == tgt:
                TN += 1
            else:
                FP += 1

    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'total': total,
    }


# ======================== 评估器基类 ========================
class BaseEvaluator:
    def __init__(self, predictor, tokenizer, test_dataset, batch_size=32):
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def _decode_texts(self, token_ids_list):
        """将 token ID 列表解码为纯文本列表"""
        texts = self.tokenizer.batch_decode(token_ids_list, skip_special_tokens=True)
        texts = [t.replace(' ', '') for t in texts]
        return texts

    def evaluate(self):
        """在测试集上评估模型，返回指标字典"""
        all_input_texts = []
        all_label_texts = []
        all_pred_texts = []

        self.test_dataset.set_format(type='torch')
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        start_time = time.time()
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 解码输入和标签为文本
            input_texts = self._decode_texts(batch['input_ids'].tolist())
            label_texts = self._decode_texts(batch['labels'].tolist())

            # 用 Predictor 获取模型预测（走完整的 tokenize → model → decode 流程）
            pred_texts = self.predictor.predict(input_texts)

            all_input_texts.extend(input_texts)
            all_label_texts.extend(label_texts)
            all_pred_texts.extend(pred_texts)

        cost_time = time.time() - start_time

        # 计算指标
        metrics = compute_sentence_level_metrics(all_input_texts, all_label_texts, all_pred_texts)
        metrics['cost_time'] = cost_time

        return metrics, all_input_texts, all_label_texts, all_pred_texts


class SpellCheckBertEvaluator(BaseEvaluator):
    """BERT 拼写纠错模型评估器"""
    pass


class SpellCheckT5Evaluator(BaseEvaluator):
    """T5 拼写纠错模型评估器"""
    pass


# ======================== 工具函数 ========================
def print_metrics(metrics):
    """打印评估结果"""
    print(f"\n{'='*50}")
    print(f"Sentence Level Evaluation Results:")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}, TN: {metrics['TN']}")
    print(f"  Total: {metrics['total']}, Cost: {metrics['cost_time']:.2f}s")
    print(f"{'='*50}")


def print_samples(input_texts, label_texts, pred_texts, n=10):
    """打印前 n 个样本的预测结果"""
    print(f"\n--- Sample Predictions (first {n}) ---")
    for i in range(min(n, len(input_texts))):
        src, tgt, pred = input_texts[i], label_texts[i], pred_texts[i]
        status = '✓' if pred == tgt else '✗'
        print(f"[{i+1}] {status}")
        print(f"  input : {src}")
        print(f"  label : {tgt}")
        print(f"  pred  : {pred}")
        print()


# ======================== 主程序 ========================
def evaluate_bert():
    from datasets import load_from_disk

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese', use_fast=False)
    model = SpellCheckBert()
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / 'spell_check_bert' / 'best.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试集
    dataset_dict = load_from_disk(str(config.DATA_DIR / 'spell_check' / 'processed' / 'bert'))
    test_dataset = dataset_dict['test']
    print(f"Test dataset size: {len(test_dataset)}")

    # 创建预测器和评估器
    predictor = SpellCheckBertPredictor(model, tokenizer, device)
    evaluator = SpellCheckBertEvaluator(predictor, tokenizer, test_dataset, batch_size=64)

    # 评估
    metrics, input_texts, label_texts, pred_texts = evaluator.evaluate()
    print_metrics(metrics)
    print_samples(input_texts, label_texts, pred_texts)

    return metrics


def evaluate_t5():
    """评估基于 mengzi-t5-base-chinese-correction 微调的 T5 模型"""
    from datasets import load_from_disk

    # 加载 tokenizer 和模型（使用 correction 预训练权重初始化，再加载微调后的 best.pt）
    tokenizer = AutoTokenizer.from_pretrained(
        config.PRE_TRAINED_DIR / 'mengzi-t5-base-chinese-correction', use_fast=False)
    model = SpellCheckT5(pretrained_path=config.PRE_TRAINED_DIR / 'mengzi-t5-base-chinese-correction')
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / 'spell_check_t5' / 'best.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试集
    dataset_dict = load_from_disk(str(config.DATA_DIR / 'spell_check' / 'processed' / 't5'))
    test_dataset = dataset_dict['test'].select(range(1000))
    print(f"[t5] Test dataset size: {len(test_dataset)}")

    # 创建预测器和评估器
    predictor = SpellCheckT5Predictor(model, tokenizer, device)
    evaluator = SpellCheckT5Evaluator(predictor, tokenizer, test_dataset, batch_size=2)

    # 评估
    metrics, input_texts, label_texts, pred_texts = evaluator.evaluate()
    print_metrics(metrics)
    print_samples(input_texts, label_texts, pred_texts)

    return metrics


def evaluate_t5_base():
    """评估基于原始 mengzi-t5 基座训练的 T5 模型"""
    from datasets import load_from_disk

    # 加载 tokenizer 和模型（原始基座，旧 checkpoint 的 linear 带 bias）
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'mengzi-t5', use_fast=False)
    model = SpellCheckT5()  # 默认加载 mengzi-t5 基座
    # 旧 checkpoint 中 linear 带 bias，新模型 linear 无 bias，用 strict=False 兼容
    model.load_state_dict(
        torch.load(config.CHECKPOINT_DIR / 'spell_check_t5_base' / 'best.pt'), strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试集
    dataset_dict = load_from_disk(str(config.DATA_DIR / 'spell_check' / 'processed' / 't5'))
    test_dataset = dataset_dict['test'].select(range(1000))
    print(f"[t5_base] Test dataset size: {len(test_dataset)}")

    # 创建预测器和评估器
    predictor = SpellCheckT5Predictor(model, tokenizer, device)
    evaluator = SpellCheckT5Evaluator(predictor, tokenizer, test_dataset, batch_size=2)

    # 评估
    metrics, input_texts, label_texts, pred_texts = evaluator.evaluate()
    print_metrics(metrics)
    print_samples(input_texts, label_texts, pred_texts)

    return metrics


if __name__ == '__main__':
    """
    python src/runner/Evaluate.py --model bert       # 评估 BERT 模型
    python src/runner/Evaluate.py --model t5         # 评估微调后的 T5（基于 correction 模型）
    python src/runner/Evaluate.py --model t5_base    # 评估原始基座训练的 T5
    """
    parser = argparse.ArgumentParser(description='拼写纠错模型评估')
    parser.add_argument('--model', type=str, default='t5', choices=['bert', 't5', 't5_base'],
                        help='选择要评估的模型: bert / t5 / t5_base')
    args = parser.parse_args()

    if args.model == 'bert':
        evaluate_bert()
    elif args.model == 't5':
        evaluate_t5()
    elif args.model == 't5_base':
        evaluate_t5_base()

