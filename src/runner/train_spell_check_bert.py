"""
BERT 拼写纠错模型训练脚本

使用 TensorBoard 查看训练日志:
tensorboard --logdir d:/Data/Graduate_Study/AI_Study/Course_Learn/Ch15_Project_Recommendation_Graph/graph/logs/spell_check_bert

启动训练并保存终端输出:
python src/runner/train_spell_check_bert.py 2>&1 | Tee-Object -FilePath logs/spell_check_bert/log_0411.txt
"""
import os
import sys
from pathlib import Path

# 解决 Windows + Conda 下 OpenMP 库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 将 src 目录加入 sys.path，以便导入兄弟包 configs、models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk
from configs import config
from models.spell_check_bert import SpellCheckBert
from runner.Trainer import TrainingConfig, Trainer


# ======================== 评估函数 ========================
def compute_metrics(predictions, labels, inputs):
    """
    三者维度均为【batch_size*num_batch，seq_len】
    
    评估指标（忽略 padding 位置，pad_token_id=0）：
    1. char_accuracy:  逐字符准确率（预测字符 == 正确字符 的比例）
    2. sentence_accuracy: 句子级完全匹配率（整句全部正确的比例）
    3. correction_precision: 纠错精确率（模型做出的修改中，改对了多少）
    4. correction_recall:    纠错召回率（需要纠正的错别字中，模型纠对了多少）
    5. correction_f1:        纠错 F1
    """
    char_correct = 0
    char_total = 0
    sentence_correct = 0
    sentence_total = len(predictions)
    # 纠错级别：以 input != label 的位置为"需要纠错的位置"
    tp = 0  # 需要纠错且模型预测正确
    fp = 0  # 不需要纠错但模型做了修改（改错了）
    fn = 0  # 需要纠错但模型没改对

    # zip 将多个可迭代对象逐位置配对，每次取出对应位置的元素组成一个元组
    for pred, label, inp in zip(predictions, labels, inputs):  # 遍历每个句子
        sent_ok = True # 标记该句子是否全部位置都预测正确
        for p, l, i in zip(pred, label, inp):  # 遍历每个字符位置
            if l == 0:  # 跳过 padding
                continue
            char_total += 1
            if p == l:
                char_correct += 1
            else:
                sent_ok = False
            # 纠错级别统计
            need_correction = (i != l)  # 该位置是错别字
            model_changed = (p != i)    # 模型做了修改
            if need_correction and p == l:
                tp += 1
            elif not need_correction and model_changed:
                fp += 1
            elif need_correction and p != l:
                fn += 1
        if sent_ok:
            sentence_correct += 1

    char_accuracy = char_correct / char_total if char_total > 0 else 0
    sentence_accuracy = sentence_correct / sentence_total if sentence_total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'char_accuracy': char_accuracy,
        'sentence_accuracy': sentence_accuracy,
        'correction_precision': precision,
        'correction_recall': recall,
        'correction_f1': f1,
    }


# ======================== 加载数据集（全量） ========================
dataset_dict = load_from_disk(str(config.DATA_DIR / 'spell_check' / 'processed' / 'bert'))

train_dataset = dataset_dict['train']
valid_dataset = dataset_dict['valid']
test_dataset = dataset_dict['test']

print(f"train: {len(train_dataset)}, valid: {len(valid_dataset)}, test: {len(test_dataset)}")

# ======================== 创建模型 ========================
model = SpellCheckBert()

# ======================== 训练配置 ========================
# 全量数据 218567 / 64 ≈ 3415 步/epoch
training_config = TrainingConfig(
    epochs=3,
    train_batch_size=64,
    valid_batch_size=64,
    test_batch_size=64,
    lr=5e-5,
    output_dir=config.CHECKPOINT_DIR / 'spell_check_bert',
    logs_dir=config.ROOT_DIR / 'logs' / 'spell_check_bert',
)

# ======================== 训练 ========================
trainer = Trainer(
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

