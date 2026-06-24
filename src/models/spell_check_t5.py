import torch
from torch import nn
from transformers import AutoModel, AutoModelForSeq2SeqLM

from configs import config


class SpellCheckT5(nn.Module):
    def __init__(self, pretrained_path=None):
        """
        :param pretrained_path: 预训练模型路径。
            - None: 使用原始 mengzi-t5 基座（self.linear 随机初始化）
            - 指定路径（如 mengzi-t5-base-chinese-correction）: 加载 T5ForConditionalGeneration，
              用其 encoder+decoder 初始化 self.t5，用其 lm_head 初始化 self.linear
        """
        super().__init__()
        if pretrained_path is None:
            pretrained_path = config.PRE_TRAINED_DIR / 'mengzi-t5'

        # 只加载一次 T5ForConditionalGeneration，从中提取 encoder+decoder 和 lm_head
        full_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
        # 提取内部的 T5Model（encoder + decoder），不会复制权重，只是引用
        self.t5 = full_model.get_encoder().config  # 仅为了拿 config
        self.t5 = AutoModel.from_config(full_model.config)
        # 将 full_model 中 encoder+decoder 的权重复制到 self.t5
        self.t5.load_state_dict(
            {k: v for k, v in full_model.state_dict().items()
             if not k.startswith('lm_head')}, strict=False)
        # 用 lm_head 权重初始化 self.linear
        self.linear = nn.Linear(full_model.config.d_model, full_model.config.vocab_size, bias=False)
        if hasattr(full_model, 'lm_head'):
            self.linear.load_state_dict(full_model.lm_head.state_dict())
        del full_model  # 释放内存

        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.t5.config.pad_token_id)

    def forward(self, input_ids, attention_mask, labels):
        """
        前向传播
        :param input_ids: 原始序列
        :param attention_mask:  原始序列的mask
        :param labels: 目标序列
        :return:
        """
        # 处理解码器的输入
        decoder_input_ids = self.t5._shift_right(labels)  # 右移一位后的 labels

        outputs = self.t5(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids)

        last_hidden_state = outputs.last_hidden_state
        # last_hidden_state.shape = [batch_size, seq_len, hidden_size]
        logits = self.linear(last_hidden_state)
        # logits.shape = [batch_size, seq_len, vocab_size]
        predictions = logits.argmax(dim=-1)

        loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

        return {'loss': loss, 'predictions': predictions}

    # 在推理时逐步生成纠错后的句子(predictions)
    # 手写 beam search 解码器
    def generate(self, input_ids, attention_mask, num_beams=3, max_length=64):
        # input_ids.shape = [batch_size, seq_len]
        # attention_mask.shape = [batch_size, seq_len]

        # 所需参数
        batch_size = input_ids.shape[0]
        device = input_ids.device
        # 当 batch_size > 1 时，用于定位每个样本在"展开后的 beam 数组"中的起始位置，用于初始化beam分数。
        # 因为后面所有 beam 都被压平到一个大 batch 里了，需要这个偏移量把“样本内索引”映射回“全局索引”。
        example_offset = torch.arange(batch_size, device=device) * num_beams
        vocab_size = self.t5.config.vocab_size

        # 编码
        encoder_outputs = self.t5.encoder(input_ids, attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state
        # last_hidden_state.shape = [batch_size, seq_len, hidden_size]

        # 处理编码器的输出( [batch_size*num_beams, seq_len, hidden_size] )，逐元素重复
        encoder_hidden_states = last_hidden_state.repeat_interleave(num_beams, dim=0)
        encoder_attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        # 解码
        # 各beam是否已经完成
        is_finish = torch.zeros([batch_size * num_beams], dtype=torch.bool, device=device)

        # 各个beam的分数
        # 一开始只让每一个batch中第一个 beam 分数为 0（有效），其他设为 -inf（不会被选中）
        # 这样第一步 top-k 自然会从第一个 beam 扩展出 num_beams 条不同路径
        beam_scores = torch.full([batch_size * num_beams, 1], -float('inf'), device=device)
        beam_scores[example_offset, 0] = 0


        # 准备第一步的输入
        # 第一轮只有 beam0 的候选是有效的
        decoder_input_ids = torch.full([batch_size * num_beams, 1],
                                       self.t5.config.decoder_start_token_id,
                                       device=device)
        for t in range(max_length):
            decoder_outputs = self.t5.decoder(input_ids=decoder_input_ids,
                                              encoder_hidden_states=encoder_hidden_states,
                                              encoder_attention_mask=encoder_attention_mask)
            last_hidden_state = decoder_outputs.last_hidden_state
            logits = self.linear(last_hidden_state)
            # logits.shape = [batch_size*num_beams, seq_len, vocab_size]

            # 每次都取最后一个位置的logits
            next_token_logits = logits[:, -1, :]
            # next_token_logits.shape = [batch_size*num_beams, vocab_size]

            # 转为概率分布
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            # next_token_scores.shape = [batch_size*num_beams, vocab_size]

            # 处理已经完成的Beam的下一个token的得分，使得已结束 beam 不会再生成新内容
            if is_finish.any():
                next_token_scores[is_finish, :] = -float('inf')
                next_token_scores[is_finish, self.t5.config.eos_token_id] = 0

            total_scores = beam_scores + next_token_scores

            # 对total_scores进行reshape，方便获取全局topk
            total_scores = total_scores.reshape(batch_size, -1)
            # total_scores.shape = [batch_size, num_beams*vocab_size]

            # 获取topk([batch_size, num_beams]])
            topk_values, topk_indices = torch.topk(total_scores, k=num_beams, dim=-1)

            # 处理beam_scores
            beam_scores = topk_values.reshape(-1, 1)
            # beam_scores.shape = [batch_size*num_beams, 1]

            # 处理下一步的decoder_input_ids
            # 获取下一个token的id
            topk_indices = topk_indices.reshape(-1, 1)
            next_token_ids = topk_indices % vocab_size
            # next_token_ids.shape = [batch_size*num_beams, 1]

            # 获取历史beam
            beam_indices = (topk_indices // vocab_size).reshape(-1) + example_offset.repeat_interleave(num_beams, dim=0)

            # 判断是否已经生成完毕
            is_finish = is_finish[beam_indices] | (next_token_ids.reshape(-1) == self.t5.config.eos_token_id)
            if is_finish.all():
                break

            # 拼接得到下一步的decoder_input_ids
            decoder_input_ids = torch.cat([decoder_input_ids[beam_indices], next_token_ids], dim=-1)
            # decoder_input_ids.shape = [batch_size*num_beams, seq_len]

        # 选择每个样本的分值最高的beam
        beam_scores = beam_scores.reshape(batch_size, num_beams)
        best_beam_indices = beam_scores.argmax(dim=-1) + example_offset
        return decoder_input_ids[best_beam_indices]

