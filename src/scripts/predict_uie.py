import sys
from pathlib import Path

# 将 src 目录加入 sys.path，以便导入 configs 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import config

# 保证在命令行中执行代码时，可以导入外部依赖uie_pytorch
sys.path.insert(0, str(config.EXTERNAL_LIB_DIR / 'uie_pytorch'))

from uie_predictor import UIEPredictor
from pprint import pprint

# ======================== 初始化 UIE 模型（加载微调后的模型权重） ========================
FINETUNED_MODEL_PATH = str(config.CHECKPOINT_DIR / 'uie_0608' / 'model_best')
ie = UIEPredictor(model='uie-base', task_path=FINETUNED_MODEL_PATH, schema=[])
print(f"已加载微调模型: {FINETUNED_MODEL_PATH}")

# ======================== 实体抽取测试 ========================
print("\n" + "=" * 60)
print("【实体抽取 - 电商商品】")
print("=" * 60)

# schema 使用 doccano 标注数据中定义的实体类型
schema = ['商品', '品牌', '颜色', '品类', '运行内存', '机身内存', '版本']
ie.set_schema(schema)

# 测试用例1: 手机商品
text1 = "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 12GB+256GB 冷杉绿 5G手机"
print(f"\n输入: {text1}")
pprint(ie(text1))

# 测试用例2: 手机商品（Redmi）
text2 = "Redmi 10X 4G Helio G85游戏芯 4800万超清四摄 5020mAh大电量 小孔全面屏 128GB大存储 4GB+128GB 冰雾白 游戏智能手机 小米 红米"
print(f"\n输入: {text2}")
pprint(ie(text2))

# 测试用例3: iPhone
text3 = "Apple iPhone 12 (A2404) 128GB 黑色 支持移动联通电信5G 双卡双待手机"
print(f"\n输入: {text3}")
pprint(ie(text3))

# 测试用例4: 笔记本电脑
schema_pc = ['商品', '品牌', '品类', '处理器', '显卡', '颜色']
ie.set_schema(schema_pc)

text4 = "联想（Lenovo） 拯救者Y9000P 2022 16英寸游戏笔记本电脑 i9-12900H RTX3060 钛晶灰"
print(f"\n输入: {text4}")
pprint(ie(text4))

# 测试用例5: 电视
schema_tv = ['品牌', '品类', '屏幕尺寸', '尺码', '分辨率', '电视类型']
ie.set_schema(schema_tv)

text5 = "TCL 85Q6 85英寸 巨幕私人影院电视 4K超高清 AI智慧屏 全景全面屏 MEMC运动防抖 2+16GB 液晶平板电视机"
print(f"\n输入: {text5}")
pprint(ie(text5))

# ======================== 关系抽取测试 ========================
print("\n" + "=" * 60)
print("【关系抽取 - 电商商品属性】")
print("=" * 60)

# 关系抽取：以"商品"为主体，抽取其关联属性
schema_rel = {'商品': ['品牌', '颜色', '品类', '运行内存', '机身内存', '版本']}
ie.set_schema(schema_rel)

text6 = "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 12GB+256GB 冷杉绿 5G手机"
print(f"\n输入: {text6}")
pprint(ie(text6))

text7 = "Redmi 10X 4G Helio G85游戏芯 4800万超清四摄 5020mAh大电量 小孔全面屏 128GB大存储 4GB+128GB 冰雾白 游戏智能手机 小米 红米"
print(f"\n输入: {text7}")
pprint(ie(text7))

text8 = "Apple iPhone 12 (A2404) 64GB 红色 支持移动联通电信5G 双卡双待手机"
print(f"\n输入: {text8}")
pprint(ie(text8))

