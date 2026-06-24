"""
T5 拼写纠错模型训练脚本

使用 TensorBoard 查看训练日志:
tensorboard --logdir d:/Data/Graduate_Study/AI_Study/Course_Learn/Ch15_Project_Recommendation_Graph/graph/logs/spell_check_t5

启动训练并保存终端输出:
python src/runner/train_spell_check_t5.py 2>&1 | Tee-Object -FilePath logs/spell_check_t5/log_0524.txt
"""
import os
import sys
from pathlib import Path

# 解决 Windows + Conda 下 OpenMP 库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 将 src 目录加入 sys.path，以便导入兄弟包 configs、models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk
from transformers import AutoTokenizer
from configs import config
from models.spell_check_t5 import SpellCheckT5
from runner.Trainer import TrainingConfig, Seq2SeqTrainer

# 加载 tokenizer，用于将 token ID 解码为文本
tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'mengzi-t5-base-chinese-correction', use_fast=False)


# ======================== 评估函数 ========================
def compute_metrics(predictions, labels, inputs):
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
    # 解码为文本
    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    input_texts = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    # T5 tokenizer decode 后可能带空格，去掉所有空格保留纯中文
    pred_texts = [t.replace(' ', '') for t in pred_texts]
    label_texts = [t.replace(' ', '') for t in label_texts]
    input_texts = [t.replace(' ', '') for t in input_texts]

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
    }


# ======================== 加载数据集（全量） ========================
dataset_dict = load_from_disk(str(config.DATA_DIR / 'spell_check' / 'processed' / 't5'))

train_dataset = dataset_dict['train'].select(range(500))
valid_dataset = dataset_dict['valid'].select(range(100))
test_dataset = dataset_dict['test'].select(range(100))

print(f"train: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}")

# ======================== 创建模型 ========================
# 基于已训练好的中文纠错模型进行微调（encoder+decoder+lm_head 权重均已预训练）
model = SpellCheckT5(pretrained_path=config.PRE_TRAINED_DIR / 'mengzi-t5-base-chinese-correction')

# ======================== 训练配置 ========================
# 全量数据 500 / 4 ≈ 125 步/epoch
training_config = TrainingConfig(
    epochs=3,                
    train_batch_size=4,       
    valid_batch_size=4,
    test_batch_size=4,
    lr=5e-5,                  
    output_dir=config.CHECKPOINT_DIR / 'spell_check_t5',
    logs_dir=config.ROOT_DIR / 'logs' / 'spell_check_t5',
    log_steps=10,
    save_steps=20,
    eval_steps=40,
)

# ======================== 训练 ========================
trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    test_dataset=test_dataset,
    training_config=training_config,
    compute_metrics=compute_metrics,
)
trainer.train()

# ======================== 测试 ========================
test_metrics = trainer.evaluate(dtype='test')
print(f"Test metrics: {test_metrics}")

