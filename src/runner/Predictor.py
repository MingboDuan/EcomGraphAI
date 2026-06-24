import os
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

from configs import config
from models.spell_check_bert import SpellCheckBert
from models.spell_check_t5 import SpellCheckT5


class BasePredictor:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def predict(self, inputs: list[str] | str):
        pass


class SpellCheckBertPredictor(BasePredictor):

    # BERT 版拼写纠错预测器
    def predict(self, inputs: list[str] | str):

        # 兼容两种输入：
        # 单条文本："我喜你。"
        # 多条文本：["我喜你。", "她很开新。"]
        is_str = isinstance(inputs, str) # 判断inputs是否是的单条文本类型
        if is_str:
            inputs = [inputs]

        # 处理输入数据
        inputs = self.tokenizer(inputs,
                                truncation=True,
                                padding='max_length',
                                max_length=64,
                                return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        predictions = outputs['predictions']

        batch_result: list[str] = self.tokenizer.batch_decode(predictions, skip_special_tokens=True) # 解码预测结果，自动去掉特殊token
        batch_result = [result.replace(' ', '') for result in batch_result] # 去掉空格
        if is_str:
            return batch_result[0] # 如果用户最初输入的是单条字符串，就返回单条字符串
        return batch_result


class SpellCheckT5Predictor(BasePredictor):

    def predict(self, inputs: list[str] | str):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]

        # 处理输入数据
        inputs = self.tokenizer(inputs,
                                truncation=True,
                                padding='max_length',
                                max_length=64,
                                return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model.generate(input_ids, attention_mask)

        batch_result: list[str] = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        batch_result = [result.replace(' ', '') for result in batch_result]
        if is_str:
            return batch_result[0]
        return batch_result


if __name__ == '__main__':
    # 测试SpellCheckBert
    # model = SpellCheckBert()
    # model.load_state_dict(torch.load(config.CHECKPOINT_DIR / 'spell_check_bert' / 'best.pt'))
    
    # tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # predictor = SpellCheckBertPredictor(model, tokenizer, device)
    
    # print(predictor.predict(
    #     ['再加上在工作的地方有机会见面别的人，也可以学习新的至识，最后经验越来越多。职业女生会增加她们的新智。',
    #      '安照孙阳自已的画说']))

    # 测试SpellCheckT5（基于 mengzi-t5-base-chinese-correction 微调）
    model = SpellCheckT5(pretrained_path=config.PRE_TRAINED_DIR / 'mengzi-t5-base-chinese-correction')
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / 'spell_check_t5' / 'best.pt'))

    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'mengzi-t5-base-chinese-correction')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictor = SpellCheckT5Predictor(model, tokenizer, device)

    print(predictor.predict(
        ['再加上在工作的地方有机会见面别的人，也可以学习新的至识，最后经验越来越多。职业女生会增加她们的新智。',
         '我喜你。']))

