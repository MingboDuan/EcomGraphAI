import torch
from torch import nn, tensor
from transformers import AutoModel

from configs import config

"""
基于 BERT 的逐字符纠错模型，将拼写纠错建模为 token 级分类任务：
对输入序列的每个位置，预测该位置应该是哪个字（从整个词表中选）
"""
class SpellCheckBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.bert.config.pad_token_id) # padding 位置不参与损失计算

    def forward(self, input_ids, attention_mask, labels=None):
        # input_ids 与 attention_mask 形状均为：(batch_size, seq_len)
        outputs = self.bert(input_ids, attention_mask)

        # last_hidden_state：(batch_size, seq_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state 

        logits = self.linear(last_hidden_state)

        # logits.shape: [batch_size,seq_len,vocab_size]
        predictions = torch.argmax(logits, dim=-1) # 取每个位置得分最高的字作为预测结果

        # predictions.shape: [batch_size,seq_len]

        # torch.cat([torch.full((batch_size, 1), 0),attention_mask[:,2:],torch.full((batch_size, 1), 0)])
        # predictions.shape: [batch_size,seq_len]

        # 清理 padding 位置
        predictions = predictions.masked_fill(attention_mask == 0, self.bert.config.pad_token_id) # padding 位置为 True，填充 pad_token_id

        result = {'predictions': predictions}
        if labels is not None:
            loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))  # 展平成(N, C) 与 (N,) 的格式，N = 样本数，C = 类别数
            result['loss'] = loss
        return result

