import os
from dataclasses import dataclass
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# 集中管理所有超参数
@dataclass
class TrainingConfig:
    # 训练参数
    epochs: int = 3
    train_batch_size: int = 32
    valid_batch_size: int = 32
    test_batch_size: int = 32
    lr: float = 5e-5
    enable_amp: bool = True  # 是否开启混合精度（AMP）

    # 路径相关
    output_dir: Path = Path('./checkpoint')
    logs_dir: Path = Path('./logs')

    # 早停相关：耐心值 3 次，默认监控 loss
    early_stop_patience: int = 3
    early_stop_metric: str = 'loss'

    # step相关：每 50 步记日志、100 步存 checkpoint、200 步做验证
    log_steps: int = 50
    save_steps: int = 100
    eval_steps: int = 200


class Trainer:
    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 training_config,
                 compute_metrics=None,  # 可选的自定义评估函数，由调用方传入（如准确率、F1 等）
                 optimizer=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = training_config
        self.compute_metrics = compute_metrics
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        # 创建目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 全局step
        self.global_step = 1

        # tensorboard
        self.writer = SummaryWriter(log_dir=self.config.logs_dir)

        # 早停相关
        self.early_stop_best_score = -float('inf')  # 初始化为负无穷
        self.early_stop_counter = 0  # 连续未改善次数

        # amp相关
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.config.enable_amp)

    def train(self):
        # 加载checkpoint
        self._load_checkpoint()

        # 获取数据集
        dataloader = self._get_dataloader(dtype='train')
        # 训练
        for epoch in range(1, 1 + self.config.epochs):
            for batch_id, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
                # 处理断点续训
                current_step = (epoch - 1) * len(dataloader) + batch_id # 当前训练所在step
                if current_step < self.global_step: # 会从checkpoint加载全局step
                    continue

                # 训练一个batch
                loss = self._train_step(batch)

                # 判断是否要保存日志
                """
                TensorBoard 读取后渲染为折线图：
                TensorBoard 面板分组：
                    ├─ train/
                    │   └─ loss          ← 每 50 步一个点的折线图
                    └─ valid/
                        ├─ loss          ← 每 200 步一个点
                        ├─ char_accuracy
                        ├─ sentence_accuracy
                        ├─ correction_precision
                        ├─ correction_recall
                        └─ correction_f1
                """
                if self.global_step % self.config.log_steps == 0:
                    self.writer.add_scalar('train/loss', loss, self.global_step)
                    # [Epoch:1|step:100] Train Loss:4.3
                    tqdm.write(f'[Epoch:{epoch}|step:{self.global_step}] Train Loss:{loss:.4f}')

                # 判断是否要保存checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

                # 判断是否要进行评估（早停）
                if self.global_step % self.config.eval_steps == 0:
                    # 验证模型
                    metrics = self.evaluate(dtype='valid')
                    # loss:0.01 | accuracy:0.98 | f1:0.98
                    metrics_str = "|".join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'[Epoch:{epoch}|step:{self.global_step}] Valid {metrics_str}')

                    # 将验证指标写入 TensorBoard,验证指标为字典，k为指标名，v为指标值
                    # add_scalar 的格式:
                    # writer.add_scalar(tag, scalar_value, global_step)
                    for k, v in metrics.items():
                        self.writer.add_scalar(f'valid/{k}', v, self.global_step)

                    # 早停判断
                    if self._should_early_stop(metrics):
                        tqdm.write('early stop')
                        return

                self.global_step += 1

    def _train_step(self, batch):
        self.model.train()  # 切换到训练模式
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        # 前向传播
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.config.enable_amp):
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def evaluate(self, dtype='test'):
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_inputs = []
        dataloader = self._get_dataloader(dtype)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=dtype):
                outputs = self._evaluate_step(batch)
                total_loss += outputs['loss'].item()

                # 收集预测结果、标签和输入
                if self.compute_metrics is not None:
                    predictions = outputs['predictions']
                    # tolist 将各种数据结构转换为 Python 原生的列表（list）
                    # 遍历所有 batch 后，维度会变为验证集数量个长度为 64 的列表
                    all_predictions.extend(predictions.tolist())
                    all_labels.extend(batch['labels'].tolist())
                    all_inputs.extend(batch['input_ids'].tolist())

        # 统计评估结果
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(all_predictions, all_labels, all_inputs)
        else:
            metrics = {}
        metrics['loss'] = total_loss / len(dataloader)
        return metrics

    def _evaluate_step(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        outputs = self.model(input_ids, attention_mask, labels)
        return outputs

    def _get_dataloader(self, dtype='train'):
        if dtype == 'train':
            dataset = self.train_dataset
            batch_size = self.config.train_batch_size
        elif dtype == 'valid':
            dataset = self.valid_dataset
            batch_size = self.config.valid_batch_size
        elif dtype == 'test':
            dataset = self.test_dataset
            batch_size = self.config.test_batch_size
        else:
            raise ValueError('Invalid dtype')

        dataset.set_format(type='torch')
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'early_stop_best_score': self.early_stop_best_score,
            'early_stop_counter': self.early_stop_counter
        }
        torch.save(checkpoint, self.config.output_dir / 'checkpoint.pt')

    def _load_checkpoint(self):
        checkpoint_path = self.config.output_dir / 'checkpoint.pt'
        if checkpoint_path.exists():
            print("检查点存在，开始加载")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict']) # 模型状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 优化器状态
            self.scaler.load_state_dict(checkpoint['scaler_state_dict']) # 混合精度状态
            self.global_step = checkpoint['global_step'] # 全局步数
            self.early_stop_best_score = checkpoint['early_stop_best_score'] # 早停最佳分数
            self.early_stop_counter = checkpoint['early_stop_counter'] # 早停计数
        else:
            print("检查不存在，从头训练")

    def _should_early_stop(self, metrics):
        score = metrics[self.config.early_stop_metric]
        if self.config.early_stop_metric == 'loss':
            score = -score

        if score > self.early_stop_best_score:
            self.early_stop_best_score = score
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), self.config.output_dir / 'best.pt')
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.config.early_stop_patience:
                return True
            else:
                return False


class Seq2SeqTrainer(Trainer):
    """T5 等 Seq2Seq 模型专用，重写 _evaluate_step 以使用 generate() 获取预测"""
    def _evaluate_step(self, batch):
        # 从模型的forward方法中获取loss
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        outputs = self.model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        result = {'loss': loss}
        # 从模型的generate方法中获取predictions
        if self.compute_metrics is not None:
            predictions = self.model.generate(input_ids, attention_mask)
            # predictions.shape = (batch_size, seq_len)

            result['predictions'] = predictions
        return result

