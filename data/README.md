# Data

本仓库只保留少量 UIE 标注样例与数据目录结构，不提交大体积训练数据、商品图片和中间处理文件。

请按需自行准备：

```text
data/spell_check/raw/data.txt              # 拼写纠错训练数据
data/uie_0608/raw/doccano.jsonl           # UIE 标注数据，可替换为自己的 Doccano 导出文件
data/uie_0608/processed/                  # 由 external_lib/uie_pytorch/doccano.py 生成
data/images/                              # 商品详情图片，需与 gmall.sku_image 中的 sku_id 对应
```
