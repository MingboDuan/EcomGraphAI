import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pymysql
from pymysql.cursors import DictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasync.sku_image_detail_graph_sync import DEFAULT_SCHEMA


MYSQL_CONFIG = {
    "host": os.getenv("GMALL_MYSQL_HOST", "localhost"),
    "port": int(os.getenv("GMALL_MYSQL_PORT", "3306")),
    "user": os.getenv("GMALL_MYSQL_USER", "root"),
    "password": os.getenv("GMALL_MYSQL_PASSWORD", "your_mysql_password"),
    "database": os.getenv("GMALL_MYSQL_DATABASE", "gmall"),
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
    "autocommit": False,
}

BACKUP_TABLES = [
    "base_category1",
    "base_category2",
    "base_category3",
    "base_trademark",
    "base_attr_info",
    "base_attr_value",
    "spu_info",
    "sku_info",
    "sku_attr_value",
    "sku_sale_attr_value",
]

DRY_RUN_MODE = False
BUSINESS_ATTR_EXCLUDE = {"分类", "品类"}

CATEGORY_TREE = [
    ("手机数码", "手机通讯", "智能手机"),
    ("手机数码", "电脑办公", "游戏笔记本"),
    ("手机数码", "电脑办公", "轻薄笔记本"),
    ("手机数码", "电脑办公", "平板电脑"),
    ("手机数码", "数码配件", "蓝牙耳机"),
    ("家用电器", "大家电", "平板电视"),
    ("家用电器", "大家电", "冰箱"),
    ("家用电器", "大家电", "洗衣机"),
    ("家用电器", "生活电器", "扫地机器人"),
    ("家用电器", "厨房电器", "空气炸锅"),
    ("家用电器", "厨房电器", "咖啡机"),
    ("食品生鲜", "粮油调味", "大米"),
    ("食品生鲜", "粮油调味", "食用油"),
    ("美妆个护", "面部护肤", "精华乳液"),
    ("美妆个护", "香水彩妆", "口红"),
    ("服饰鞋靴", "女装", "连衣裙"),
    ("服饰鞋靴", "运动鞋", "跑步鞋"),
]

TRADEMARKS = [
    "小米",
    "Redmi",
    "华为",
    "荣耀",
    "Apple",
    "三星",
    "vivo",
    "OPPO",
    "联想",
    "ThinkPad",
    "戴尔",
    "华硕",
    "H&U&W",
    "海信",
    "TCL",
    "索尼",
    "美的",
    "苏泊尔",
    "九阳",
    "格力",
    "容声",
    "西门子",
    "JBL",
    "漫步者",
    "金龙鱼",
    "十月稻田",
    "欧莱雅",
    "兰蔻",
    "完美日记",
    "安踏",
    "李宁",
    "Nike",
]

ATTR_VALUES = {
    "尺码": ["S", "M", "L", "XL", "XXL", "36码", "37码", "38码", "39码", "40码", "41码", "42码", "43码", "44码", "45码"],
    "分辨率": ["1920*1080", "2560*1440", "2K", "4K", "3840*2160"],
    "屏幕尺寸": ["6.3英寸", "6.59英寸", "6.7英寸", "6.83英寸", "6.9英寸", "10.9英寸", "11英寸", "12.4英寸", "12.9英寸", "14英寸", "15英寸", "15.6英寸", "55英寸", "65英寸", "75英寸"],
    "电视类型": ["全面屏电视", "Mini LED电视", "OLED电视", "智慧屏电视"],
    "版本": ["8GB+128GB", "8GB+256GB", "12GB+256GB", "16GB+512GB"],
    "颜色": ["黑色", "白色", "银色", "蓝色", "绿色", "红色", "米色"],
    "机身内存": ["128GB", "256GB", "512GB", "1TB"],
    "运行内存": ["6GB", "8GB", "12GB", "16GB"],
    "内存": ["8GB", "16GB", "32GB"],
    "硬盘": ["512GB SSD", "1TB SSD", "2TB SSD"],
    "显卡": ["集成显卡", "MX450", "RTX 4050", "RTX 4060", "RTX 4070"],
    "处理器": ["骁龙8 Gen2", "天玑9200", "天玑9400", "麒麟9000", "Intel i5", "Intel i7", "AMD R7"],
    "类别": ["智能手机", "平板电脑", "蓝牙耳机", "游戏本", "轻薄本", "电视", "冰箱", "洗衣机", "扫地机器人", "空气炸锅", "咖啡机", "大米", "食用油", "护肤品", "彩妆", "连衣裙", "跑步鞋"],
    "分类": ["手机", "笔记本", "电视", "粮油", "美妆", "服饰"],
    "品类": ["手机", "笔记本", "电视", "大米", "食用油", "口红", "跑步鞋"],
    "粮食调味": ["东北大米", "五常大米", "花生油", "玉米油", "酱油"],
    "香水彩妆": ["口红", "粉底液", "香水", "眼影盘"],
    "功效": ["补水", "保湿", "控油", "防晒", "抗皱"],
    "电池容量": ["4500mAh", "5000mAh", "5200mAh", "5630mAh", "6000mAh", "7500mAh", "90Wh"],
    "摄像头像素": ["4800万", "5000万", "1亿像素", "1200万"],
    "散热方式": ["双风扇", "双铜管", "液冷散热", "VC均热板"],
    "解锁方式": ["指纹解锁", "人脸解锁", "屏下指纹"],
}

SPU_TEMPLATES = [
    ("智能手机", ["小米", "Redmi", "华为", "荣耀", "Apple"], ["颜色", "版本", "运行内存", "机身内存", "处理器", "电池容量", "摄像头像素", "解锁方式", "屏幕尺寸"]),
    ("游戏笔记本", ["联想", "ThinkPad", "戴尔", "华硕"], ["颜色", "内存", "硬盘", "显卡", "处理器", "散热方式", "屏幕尺寸", "分辨率"]),
    ("轻薄笔记本", ["联想", "ThinkPad", "戴尔", "华硕"], ["颜色", "内存", "硬盘", "处理器", "屏幕尺寸", "分辨率"]),
    ("平板电脑", ["Apple", "华为", "小米", "三星"], ["颜色", "版本", "运行内存", "机身内存", "处理器", "电池容量", "屏幕尺寸", "分辨率", "解锁方式", "类别"]),
    ("蓝牙耳机", ["Apple", "华为", "小米", "索尼", "JBL", "漫步者"], ["颜色", "电池容量", "类别"]),
    ("平板电视", ["海信", "TCL", "小米", "索尼"], ["屏幕尺寸", "分辨率", "电视类型", "颜色"]),
    ("冰箱", ["美的", "海信", "容声", "西门子"], ["颜色", "类别"]),
    ("洗衣机", ["美的", "海信", "西门子"], ["颜色", "类别"]),
    ("扫地机器人", ["小米", "美的"], ["颜色", "电池容量", "类别"]),
    ("空气炸锅", ["美的", "苏泊尔", "九阳"], ["颜色", "类别"]),
    ("咖啡机", ["美的", "苏泊尔", "九阳"], ["颜色", "类别"]),
    ("大米", ["十月稻田", "金龙鱼"], ["粮食调味", "分类", "类别"]),
    ("食用油", ["金龙鱼", "十月稻田"], ["粮食调味", "分类", "类别"]),
    ("精华乳液", ["欧莱雅", "兰蔻"], ["功效", "类别", "分类"]),
    ("口红", ["完美日记", "欧莱雅", "兰蔻"], ["颜色", "香水彩妆", "功效", "分类"]),
    ("连衣裙", ["李宁", "安踏"], ["尺码", "颜色", "类别", "分类"]),
    ("跑步鞋", ["安踏", "李宁", "Nike"], ["尺码", "颜色", "类别", "分类"]),
]

PRODUCT_CONFIGS = {
    "智能手机": {
        "models": ["影像旗舰", "性能旗舰", "轻薄长续航", "游戏增强版", "Pro Max", "Ultra"],
        "attrs": {
            "颜色": ["黑色", "白色", "银色", "蓝色", "绿色"],
            "版本": ["8GB+128GB", "8GB+256GB", "12GB+256GB", "16GB+512GB"],
            "运行内存": ["8GB", "12GB", "16GB"],
            "机身内存": ["128GB", "256GB", "512GB"],
            "处理器": ["骁龙8 Gen2", "天玑9200", "天玑9400", "麒麟9000"],
            "电池容量": ["4500mAh", "5000mAh", "5200mAh", "5630mAh", "6000mAh", "7500mAh"],
            "摄像头像素": ["4800万", "5000万", "1亿像素"],
            "解锁方式": ["指纹解锁", "人脸解锁", "屏下指纹"],
            "屏幕尺寸": ["6.3英寸", "6.59英寸", "6.7英寸", "6.83英寸", "6.9英寸"],
            "品类": ["手机"],
            "分类": ["手机"],
            "类别": ["智能手机"],
        },
        "name_attrs": ["颜色", "版本", "处理器", "电池容量", "摄像头像素", "屏幕尺寸"],
    },
    "游戏笔记本": {
        "models": ["拯救者", "暗影精灵", "游匣", "天选", "战神", "电竞版"],
        "attrs": {
            "颜色": ["黑色", "银色", "蓝色"],
            "内存": ["16GB", "32GB"],
            "硬盘": ["512GB SSD", "1TB SSD", "2TB SSD"],
            "显卡": ["RTX 4050", "RTX 4060", "RTX 4070", "MX450"],
            "处理器": ["Intel i5", "Intel i7", "AMD R7"],
            "散热方式": ["双风扇", "双铜管", "液冷散热"],
            "屏幕尺寸": ["15.6英寸", "16英寸", "17.3英寸"],
            "分辨率": ["1920*1080", "2560*1440", "2K"],
            "品类": ["笔记本"],
            "分类": ["笔记本"],
            "类别": ["游戏本"],
        },
        "name_attrs": ["颜色", "处理器", "内存", "硬盘", "显卡", "屏幕尺寸", "分辨率"],
    },
    "轻薄笔记本": {
        "models": ["MateBook", "小新", "ThinkBook", "灵越", "Air", "商务本"],
        "attrs": {
            "颜色": ["银色", "白色", "黑色", "蓝色"],
            "内存": ["8GB", "16GB", "32GB"],
            "硬盘": ["512GB SSD", "1TB SSD"],
            "处理器": ["Intel i5", "Intel i7", "AMD R7"],
            "屏幕尺寸": ["14英寸", "15英寸", "15.6英寸"],
            "分辨率": ["1920*1080", "2K", "4K"],
            "品类": ["笔记本"],
            "分类": ["笔记本"],
            "类别": ["轻薄本"],
        },
        "name_attrs": ["颜色", "处理器", "内存", "硬盘", "屏幕尺寸", "分辨率"],
    },
    "平板电脑": {
        "models": ["学习平板", "影音平板", "Pro手写笔套装", "轻薄办公", "旗舰平板", "护眼平板"],
        "attrs": {
            "颜色": ["黑色", "白色", "银色", "蓝色", "绿色"],
            "版本": ["8GB+128GB", "8GB+256GB", "12GB+256GB", "16GB+512GB"],
            "运行内存": ["8GB", "12GB", "16GB"],
            "机身内存": ["128GB", "256GB", "512GB"],
            "处理器": ["骁龙8 Gen2", "天玑9200", "麒麟9000"],
            "电池容量": ["5000mAh", "6000mAh", "7500mAh"],
            "屏幕尺寸": ["10.9英寸", "11英寸", "12.4英寸", "12.9英寸"],
            "分辨率": ["2K", "2560*1440", "4K"],
            "解锁方式": ["指纹解锁", "人脸解锁"],
            "类别": ["平板电脑"],
        },
        "name_attrs": ["颜色", "版本", "处理器", "电池容量", "屏幕尺寸", "分辨率", "解锁方式", "类别"],
    },
    "蓝牙耳机": {
        "models": ["主动降噪", "运动挂耳", "半入耳", "HiFi音质", "长续航", "游戏低延迟"],
        "attrs": {
            "颜色": ["黑色", "白色", "银色", "蓝色", "绿色"],
            "电池容量": ["4500mAh", "5000mAh", "5200mAh"],
            "类别": ["蓝牙耳机"],
        },
        "name_attrs": ["颜色", "电池容量", "类别"],
    },
    "平板电视": {
        "models": ["星河", "视界", "Mini LED", "影院版", "智慧屏", "旗舰版"],
        "attrs": {
            "颜色": ["黑色", "银色"],
            "屏幕尺寸": ["55英寸", "65英寸", "75英寸"],
            "分辨率": ["4K", "3840*2160"],
            "电视类型": ["全面屏电视", "Mini LED电视", "OLED电视", "智慧屏电视"],
            "品类": ["电视"],
            "分类": ["电视"],
            "类别": ["电视"],
        },
        "name_attrs": ["屏幕尺寸", "分辨率", "电视类型", "颜色"],
    },
    "冰箱": {
        "models": ["风冷无霜", "十字对开门", "一级能效", "母婴保鲜", "大容量", "嵌入式"],
        "attrs": {
            "颜色": ["黑色", "白色", "银色", "蓝色", "米色"],
            "类别": ["冰箱"],
        },
        "name_attrs": ["颜色", "类别"],
    },
    "洗衣机": {
        "models": ["滚筒洗烘", "波轮大容量", "超薄嵌入式", "除菌洗", "静音变频", "母婴洗"],
        "attrs": {
            "颜色": ["白色", "银色", "黑色", "蓝色"],
            "类别": ["洗衣机"],
        },
        "name_attrs": ["颜色", "类别"],
    },
    "扫地机器人": {
        "models": ["全能基站", "扫拖一体", "自动集尘", "激光导航", "智能避障"],
        "attrs": {
            "颜色": ["白色", "黑色", "银色"],
            "电池容量": ["4500mAh", "5000mAh", "5200mAh"],
            "类别": ["扫地机器人"],
            "分类": ["家电"],
            "品类": ["扫地机器人"],
        },
        "name_attrs": ["颜色", "电池容量", "类别"],
    },
    "空气炸锅": {
        "models": ["可视窗口", "大容量", "低脂烘烤", "智能菜单", "家用多功能"],
        "attrs": {
            "颜色": ["白色", "黑色", "米色", "绿色"],
            "类别": ["空气炸锅"],
            "分类": ["家电"],
            "品类": ["空气炸锅"],
        },
        "name_attrs": ["颜色", "类别"],
    },
    "咖啡机": {
        "models": ["意式半自动", "全自动研磨", "胶囊便携", "奶泡一体", "家用小型", "办公室商用"],
        "attrs": {
            "颜色": ["白色", "黑色", "银色", "米色"],
            "类别": ["咖啡机"],
        },
        "name_attrs": ["颜色", "类别"],
    },
    "大米": {
        "models": ["东北长粒香", "五常稻花香", "有机珍珠米", "家庭装", "真空锁鲜"],
        "attrs": {
            "粮食调味": ["东北大米", "五常大米"],
            "分类": ["粮油"],
            "类别": ["大米"],
            "品类": ["大米"],
        },
        "name_attrs": ["粮食调味", "类别"],
    },
    "食用油": {
        "models": ["压榨花生油", "非转基因玉米油", "葵花籽油", "家庭桶装", "低芥酸菜籽油"],
        "attrs": {
            "粮食调味": ["花生油", "玉米油"],
            "分类": ["粮油"],
            "类别": ["食用油"],
            "品类": ["食用油"],
        },
        "name_attrs": ["粮食调味", "类别"],
    },
    "精华乳液": {
        "models": ["玻尿酸补水", "烟酰胺提亮", "修护保湿", "抗皱紧致", "敏感肌舒缓"],
        "attrs": {
            "功效": ["补水", "保湿", "控油", "防晒", "抗皱"],
            "分类": ["美妆"],
            "类别": ["护肤品"],
            "品类": ["精华乳液"],
        },
        "name_attrs": ["功效", "类别"],
    },
    "口红": {
        "models": ["丝绒哑光", "水光镜面", "小细跟", "持久显色", "礼盒装"],
        "attrs": {
            "颜色": ["红色", "米色", "橘色", "豆沙色", "玫瑰色"],
            "香水彩妆": ["口红"],
            "功效": ["保湿", "持久", "显色"],
            "分类": ["美妆"],
            "类别": ["彩妆"],
            "品类": ["口红"],
        },
        "name_attrs": ["颜色", "香水彩妆", "功效"],
    },
    "连衣裙": {
        "models": ["法式收腰", "通勤气质", "碎花雪纺", "针织长袖", "夏季薄款", "小黑裙"],
        "attrs": {
            "尺码": ["S", "M", "L", "XL", "XXL"],
            "颜色": ["黑色", "白色", "蓝色", "绿色", "红色", "米色"],
            "分类": ["服饰"],
            "类别": ["连衣裙"],
            "品类": ["连衣裙"],
        },
        "name_attrs": ["颜色", "尺码", "类别"],
    },
    "跑步鞋": {
        "models": ["缓震跑鞋", "竞速训练", "透气网面", "轻量支撑", "马拉松系列"],
        "attrs": {
            "尺码": ["36码", "37码", "38码", "39码", "40码", "41码", "42码", "43码", "44码", "45码"],
            "颜色": ["黑色", "白色", "蓝色", "绿色", "红色", "米色"],
            "分类": ["服饰"],
            "类别": ["跑步鞋"],
            "品类": ["跑步鞋"],
        },
        "name_attrs": ["颜色", "尺码", "类别"],
    },
}

SKU_NAME_EXTRAS = {
    "智能手机": [
        ["官方标配", "全网通5G"],
        ["碎屏险套装", "快充套装"],
        ["拍照旗舰套装", "高刷护眼屏"],
        ["游戏性能套装", "散热保护壳"],
        ["长续航套装", "原装充电器"],
    ],
    "游戏笔记本": [
        ["官方标配", "RGB背光键盘"],
        ["电竞套装", "高刷屏"],
        ["设计师套装", "高色域屏"],
        ["游戏进阶套装", "增强散热"],
        ["办公游戏双用", "扩展坞套装"],
    ],
    "轻薄笔记本": [
        ["官方标配", "便携办公"],
        ["学生套装", "护眼屏"],
        ["商务套装", "指纹解锁"],
        ["设计办公套装", "高色域屏"],
        ["轻薄长续航", "金属机身"],
    ],
    "平板电脑": [
        ["手写笔套装", "学生网课"],
        ["键盘保护套", "轻办公"],
        ["影音娱乐款", "高刷护眼屏"],
        ["旗舰套装", "多窗口协同"],
        ["儿童学习款", "护眼模式"],
    ],
    "蓝牙耳机": [
        ["主动降噪版", "通勤适用"],
        ["运动防汗版", "稳固佩戴"],
        ["游戏低延迟版", "清晰通话"],
        ["长续航套装", "充电仓"],
        ["HiFi音质版", "入耳舒适"],
    ],
    "平板电视": [
        ["客厅影院款", "MEMC防抖"],
        ["游戏电视款", "低延迟"],
        ["智慧语音款", "远场语音"],
        ["护眼大屏款", "高色域"],
        ["壁挂安装套装", "送装服务"],
    ],
    "冰箱": [
        ["一级能效", "风冷无霜"],
        ["母婴保鲜", "独立控温"],
        ["大容量家庭款", "低噪运行"],
        ["嵌入式设计", "纤薄机身"],
        ["十字对开门", "净味保鲜"],
    ],
    "洗衣机": [
        ["洗烘一体", "除菌洗"],
        ["大容量家庭款", "变频静音"],
        ["母婴洗护", "高温筒自洁"],
        ["超薄嵌入", "小户型适用"],
        ["智能投放", "节水节能"],
    ],
    "扫地机器人": [
        ["自动集尘版", "适合大户型"],
        ["扫拖一体版", "强力吸尘"],
        ["宠物家庭版", "毛发防缠绕"],
        ["智能避障版", "激光导航"],
        ["全能基站版", "自动回洗"],
    ],
    "空气炸锅": [
        ["4L家用款", "低脂烘烤"],
        ["5L大容量", "可视窗口"],
        ["智能菜单款", "不粘内胆"],
        ["双旋钮款", "易清洗"],
        ["家庭套装", "空气循环加热"],
    ],
    "咖啡机": [
        ["意式浓缩", "奶泡一体"],
        ["全自动研磨", "办公室适用"],
        ["胶囊便携", "一键萃取"],
        ["家用小型", "易清洗"],
        ["商用款", "连续出杯"],
    ],
    "大米": [
        ["5kg袋装", "真空锁鲜"],
        ["10kg家庭装", "当季新米"],
        ["2.5kg尝鲜装", "软糯香甜"],
        ["礼盒装", "产地直供"],
        ["家庭囤货装", "低温存储"],
    ],
    "食用油": [
        ["5L桶装", "物理压榨"],
        ["3瓶组合装", "家庭烹饪"],
        ["小瓶尝鲜装", "凉拌热炒"],
        ["家庭囤货装", "清香不腻"],
        ["礼盒装", "厨房常备"],
    ],
    "精华乳液": [
        ["50ml正装", "清爽不黏腻"],
        ["30ml尝鲜装", "敏感肌可用"],
        ["补水套装", "早晚护理"],
        ["修护套装", "换季护理"],
        ["礼盒装", "日常护肤"],
    ],
    "口红": [
        ["单支装", "通勤显白"],
        ["礼盒装", "持久不易掉色"],
        ["热门色号", "哑光质地"],
        ["约会妆容", "顺滑易涂"],
        ["日常百搭", "滋润不拔干"],
    ],
    "连衣裙": [
        ["通勤款", "收腰显瘦"],
        ["夏季薄款", "垂感面料"],
        ["约会款", "法式气质"],
        ["日常休闲款", "舒适透气"],
        ["礼服感款", "显高版型"],
    ],
    "跑步鞋": [
        ["缓震款", "日常慢跑"],
        ["竞速训练款", "轻量回弹"],
        ["透气网面款", "夏季运动"],
        ["支撑稳定款", "长距离训练"],
        ["耐磨外底款", "通勤运动"],
    ],
}


def setup_logger() -> logging.Logger:
    log_dir = PROJECT_ROOT / "logs" / "gmall_data_enhance"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("gmall_data_enhance")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("log_path: %s", log_path)
    return logger


def connect():
    return pymysql.connect(**MYSQL_CONFIG)


def fetch_columns(cursor, table: str) -> List[str]:
    cursor.execute(
        """
        SELECT column_name AS column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    return [row["column_name"] for row in cursor.fetchall()]


def inspect_database(cursor, logger: logging.Logger) -> None:
    for table in BACKUP_TABLES:
        cursor.execute(f"SELECT COUNT(*) AS count FROM {table}")
        count = cursor.fetchone()["count"]
        columns = fetch_columns(cursor, table)
        logger.info("%s: count=%s columns=%s", table, count, columns)

    cursor.execute("SELECT id, name FROM base_category1 ORDER BY id LIMIT 20")
    logger.info("category1_sample: %s", cursor.fetchall())
    cursor.execute("SELECT id, name, category1_id FROM base_category2 ORDER BY id LIMIT 20")
    logger.info("category2_sample: %s", cursor.fetchall())
    cursor.execute("SELECT id, name, category2_id FROM base_category3 ORDER BY id LIMIT 20")
    logger.info("category3_sample: %s", cursor.fetchall())
    cursor.execute("SELECT id, attr_name FROM base_attr_info ORDER BY id")
    attr_rows = cursor.fetchall()
    logger.info("base_attr_info: %s", attr_rows)
    allowed_attrs = set(DEFAULT_SCHEMA)
    invalid_attrs = [row["attr_name"] for row in attr_rows if row["attr_name"] not in allowed_attrs]
    logger.info("invalid_attr_names: %s", invalid_attrs)
    cursor.execute("SELECT id, spu_id, sku_name FROM sku_info WHERE id BETWEEN 36 AND 43 ORDER BY id")
    logger.info("sku_36_43_sample: %s", cursor.fetchall())
    cursor.execute(
        """
        SELECT bc3.name AS category3_name, MIN(ski.sku_name) AS sample_sku_name, COUNT(*) AS sku_count
        FROM sku_info ski
        JOIN base_category3 bc3 ON ski.category3_id = bc3.id
        GROUP BY bc3.name
        ORDER BY bc3.name
        """
    )
    logger.info("category_sku_samples: %s", cursor.fetchall())
    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM sku_info ski
        JOIN base_category3 bc3 ON ski.category3_id = bc3.id
        WHERE
            (bc3.name IN ('连衣裙', '跑步鞋') AND (ski.sku_name LIKE '%英寸%' OR ski.sku_name LIKE '%寸%' OR ski.sku_name LIKE '%GB%' OR ski.sku_name LIKE '%SSD%'))
            OR (bc3.name IN ('大米', '食用油') AND (ski.sku_name LIKE '%英寸%' OR ski.sku_name LIKE '%GB%' OR ski.sku_name LIKE '%SSD%' OR ski.sku_name LIKE '%像素%'))
            OR (bc3.name IN ('冰箱', '洗衣机', '咖啡机', '蓝牙耳机') AND (ski.sku_name LIKE '%SSD%' OR ski.sku_name LIKE '%RTX%' OR ski.sku_name LIKE '%像素%' OR ski.sku_name LIKE '%显卡%'))
            OR (bc3.name IN ('智能手机', '游戏笔记本', '轻薄笔记本', '平板电视') AND ski.sku_name LIKE '%连衣裙%')
        """
    )
    logger.info("suspicious_sku_name_count: %s", cursor.fetchone()["count"])
    cursor.execute("SELECT COUNT(*) AS count FROM (SELECT sku_name FROM sku_info GROUP BY sku_name HAVING COUNT(*) > 1) t")
    logger.info("duplicate_sku_name_group_count: %s", cursor.fetchone()["count"])
    cursor.execute("SELECT COUNT(*) AS count FROM sku_attr_value WHERE attr_name IN ('分类', '品类')")
    logger.info("excluded_attr_row_count: %s", cursor.fetchone()["count"])
    cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM sku_info ski
        JOIN base_category3 bc3 ON ski.category3_id = bc3.id
        JOIN sku_attr_value sav ON ski.id = sav.sku_id
        WHERE
            (bc3.name IN ('连衣裙', '跑步鞋') AND sav.attr_name IN ('屏幕尺寸', '分辨率', '内存', '硬盘', '显卡', '处理器', '电池容量', '摄像头像素'))
            OR (bc3.name IN ('大米', '食用油') AND sav.attr_name IN ('屏幕尺寸', '分辨率', '内存', '硬盘', '显卡', '处理器', '电池容量', '摄像头像素', '颜色', '尺码'))
            OR (bc3.name IN ('智能手机') AND sav.attr_name IN ('尺码', '硬盘', '显卡', '电视类型', '粮食调味', '香水彩妆', '功效'))
            OR (bc3.name IN ('平板电视') AND sav.attr_name IN ('尺码', '运行内存', '机身内存', '内存', '硬盘', '显卡', '处理器', '电池容量', '摄像头像素'))
            OR (bc3.name IN ('冰箱', '洗衣机', '咖啡机', '蓝牙耳机') AND sav.attr_name IN ('尺码', '屏幕尺寸', '分辨率', '版本', '机身内存', '运行内存', '内存', '硬盘', '显卡', '处理器', '摄像头像素', '电视类型'))
        """
    )
    logger.info("suspicious_attr_row_count: %s", cursor.fetchone()["count"])
    audit_path = export_sku_audit(cursor)
    logger.info("sku_audit_path: %s", audit_path)


def export_sku_audit(cursor) -> Path:
    """按 sku_id 导出逐条审计结果，包含分类、品牌、SPU 和属性。"""
    audit_dir = PROJECT_ROOT / "output" / "gmall_data_enhance_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "sku_audit.jsonl"
    cursor.execute(
        """
        SELECT
            ski.id AS sku_id,
            ski.sku_name,
            spi.id AS spu_id,
            spi.spu_name,
            bc1.name AS category1_name,
            bc2.name AS category2_name,
            bc3.name AS category3_name,
            bt.tm_name AS trademark_name
        FROM sku_info ski
        LEFT JOIN spu_info spi ON ski.spu_id = spi.id
        LEFT JOIN base_category3 bc3 ON ski.category3_id = bc3.id
        LEFT JOIN base_category2 bc2 ON bc3.category2_id = bc2.id
        LEFT JOIN base_category1 bc1 ON bc2.category1_id = bc1.id
        LEFT JOIN base_trademark bt ON ski.tm_id = bt.id
        ORDER BY ski.id
        """
    )
    sku_rows = cursor.fetchall()
    cursor.execute(
        """
        SELECT sku_id, attr_name, value_name
        FROM sku_attr_value
        ORDER BY sku_id, attr_name, value_name
        """
    )
    attr_map = defaultdict(list)
    for row in cursor.fetchall():
        attr_map[row["sku_id"]].append({"attr_name": row["attr_name"], "attr_value": row["value_name"]})

    with audit_path.open("w", encoding="utf-8") as f:
        for row in sku_rows:
            row["attrs"] = attr_map.get(row["sku_id"], [])
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return audit_path


def backup_tables(cursor, logger: logging.Logger) -> Path:
    backup_dir = PROJECT_ROOT / "output" / "gmall_data_enhance_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for table in BACKUP_TABLES:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        backup_path = backup_dir / f"{table}.jsonl"
        with backup_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        logger.info("backup %s rows=%s path=%s", table, len(rows), backup_path)
    return backup_dir


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return ""


def insert_row(cursor, table: str, data: Dict[str, Any]) -> int:
    columns = fetch_columns(cursor, table)
    filtered = {key: value for key, value in data.items() if key in columns}
    names = ", ".join(filtered.keys())
    placeholders = ", ".join(["%s"] * len(filtered))
    cursor.execute(f"INSERT INTO {table} ({names}) VALUES ({placeholders})", list(filtered.values()))
    return cursor.lastrowid


def replace_categories(cursor, logger: logging.Logger) -> Dict[str, int]:
    cursor.execute("DELETE FROM base_category3")
    cursor.execute("DELETE FROM base_category2")
    cursor.execute("DELETE FROM base_category1")
    reset_auto_increment(cursor, ["base_category1", "base_category2", "base_category3"])

    category1_ids: Dict[str, int] = {}
    category2_ids: Dict[Tuple[str, str], int] = {}
    category3_ids: Dict[str, int] = {}

    for category1, category2, category3 in CATEGORY_TREE:
        if category1 not in category1_ids:
            category1_ids[category1] = insert_row(cursor, "base_category1", {"name": category1})
        key2 = (category1, category2)
        if key2 not in category2_ids:
            category2_ids[key2] = insert_row(
                cursor,
                "base_category2",
                {"name": category2, "category1_id": category1_ids[category1]},
            )
        category3_ids[category3] = insert_row(
            cursor,
            "base_category3",
            {"name": category3, "category2_id": category2_ids[key2]},
        )

    logger.info("replace_categories: category1=%s category2=%s category3=%s", len(category1_ids), len(category2_ids), len(category3_ids))
    return category3_ids


def replace_trademarks(cursor, logger: logging.Logger) -> Dict[str, int]:
    cursor.execute("DELETE FROM base_trademark")
    reset_auto_increment(cursor, ["base_trademark"])
    trademark_ids = {}
    for name in TRADEMARKS:
        trademark_ids[name] = insert_row(cursor, "base_trademark", {"tm_name": name, "logo_url": "", "create_time": None, "operate_time": None})
    logger.info("replace_trademarks: %s", len(trademark_ids))
    return trademark_ids


def replace_attrs(cursor, logger: logging.Logger) -> Dict[str, int]:
    cursor.execute("DELETE FROM sku_attr_value")
    cursor.execute("DELETE FROM sku_sale_attr_value")
    cursor.execute("DELETE FROM base_attr_value")
    cursor.execute("DELETE FROM base_attr_info")
    reset_auto_increment(cursor, ["sku_attr_value", "sku_sale_attr_value", "base_attr_value", "base_attr_info"])

    attr_ids = {}
    attr_values = collect_attr_values()
    allowed_attrs = [name for name in DEFAULT_SCHEMA if name in attr_values]
    for attr_name in allowed_attrs:
        attr_ids[attr_name] = insert_row(cursor, "base_attr_info", {"attr_name": attr_name})
        for value_name in attr_values[attr_name]:
            insert_row(cursor, "base_attr_value", {"value_name": value_name, "attr_id": attr_ids[attr_name]})

    logger.info("replace_attrs: attr_count=%s value_count=%s", len(allowed_attrs), sum(len(attr_values[name]) for name in allowed_attrs))
    return attr_ids


def collect_attr_values() -> Dict[str, List[str]]:
    """汇总全局属性值和各品类专属属性值，保证基础属性值字典覆盖实际 SKU。"""
    values = {name: set(items) for name, items in ATTR_VALUES.items() if name not in BUSINESS_ATTR_EXCLUDE}
    for config in PRODUCT_CONFIGS.values():
        for attr_name, attr_values in config["attrs"].items():
            if attr_name in BUSINESS_ATTR_EXCLUDE:
                continue
            values.setdefault(attr_name, set()).update(attr_values)
    return {name: sorted(items) for name, items in values.items()}


def clear_spu_sku(cursor) -> None:
    cursor.execute("DELETE FROM sku_attr_value")
    cursor.execute("DELETE FROM sku_sale_attr_value")
    cursor.execute("DELETE FROM sku_info")
    cursor.execute("DELETE FROM spu_info")
    reset_auto_increment(cursor, ["sku_attr_value", "sku_sale_attr_value", "sku_info", "spu_info"])


def reset_auto_increment(cursor, tables: List[str]) -> None:
    """重置自增 ID，保证重新生成的 SKU/SPU 编号连续，便于和本地图片数据联动。"""
    if DRY_RUN_MODE:
        return
    for table in tables:
        cursor.execute(f"ALTER TABLE {table} AUTO_INCREMENT = 1")


def choose_category3(category_name: str, category3_ids: Dict[str, int]) -> int:
    if category_name in category3_ids:
        return category3_ids[category_name]
    return random.choice(list(category3_ids.values()))


def build_spu_name(trademark: str, category: str, index: int) -> str:
    config = PRODUCT_CONFIGS[category]
    model = random.choice(config["models"])
    return f"{trademark}{model}{category} {index:03d}"


def build_sku_name(spu_name: str, category: str, attrs: Dict[str, str], extras: List[str]) -> str:
    parts = [spu_name]
    attr_order = PRODUCT_CONFIGS[category]["name_attrs"] + [
        key for key in attrs if key not in PRODUCT_CONFIGS[category]["name_attrs"]
    ]
    for key in attr_order:
        if key in attrs:
            parts.append(attrs[key])
    parts.extend(extras)
    return " ".join(parts)


def build_category_attrs(category: str) -> Dict[str, str]:
    """按品类生成属性，避免把屏幕尺寸、服装尺码、粮油属性串到错误品类上。"""
    config = PRODUCT_CONFIGS[category]
    selected_attrs = {}
    for attr_name, values in config["attrs"].items():
        if attr_name in BUSINESS_ATTR_EXCLUDE:
            continue
        selected_attrs[attr_name] = random.choice(values)
    return selected_attrs


def build_sku_extras(category: str) -> List[str]:
    """生成只用于名称描述的规格卖点，提高同一 SPU 下不同 SKU 的区分度。"""
    return list(random.choice(SKU_NAME_EXTRAS[category]))


def insert_spu_sku_data(
    cursor,
    category3_ids: Dict[str, int],
    trademark_ids: Dict[str, int],
    attr_ids: Dict[str, int],
    target_spu_count: int,
    logger: logging.Logger,
) -> None:
    clear_spu_sku(cursor)
    spu_count = 0
    sku_count = 0
    attr_count = 0
    generated_sku_names = set()

    while spu_count < target_spu_count:
        category_name, trademarks, attr_names = random.choice(SPU_TEMPLATES)
        trademark = random.choice(trademarks)
        spu_name = build_spu_name(trademark, category_name, spu_count + 1)
        category3_id = choose_category3(category_name, category3_ids)
        tm_id = trademark_ids[trademark]

        spu_id = insert_row(
            cursor,
            "spu_info",
            {
                "spu_name": spu_name,
                "description": f"{spu_name}，适用于电商推荐知识图谱构建。",
                "category3_id": category3_id,
                "tm_id": tm_id,
            },
        )
        spu_count += 1

        sku_per_spu = random.randint(2, 3)
        for sku_index in range(1, sku_per_spu + 1):
            selected_attrs, sku_name = build_unique_sku_payload(
                category_name,
                spu_name,
                generated_sku_names,
                sku_index,
            )
            generated_sku_names.add(sku_name)
            sku_id = insert_row(
                cursor,
                "sku_info",
                {
                    "spu_id": spu_id,
                    "price": round(random.uniform(59, 8999), 2),
                    "sku_name": sku_name,
                    "sku_desc": sku_name,
                    "weight": round(random.uniform(0.1, 5.0), 2),
                    "tm_id": tm_id,
                    "category3_id": category3_id,
                    "sku_default_img": "",
                },
            )
            sku_count += 1

            for attr_name, attr_value in selected_attrs.items():
                insert_row(
                    cursor,
                    "sku_attr_value",
                    {
                        "attr_id": attr_ids[attr_name],
                        "value_id": 0,
                        "sku_id": sku_id,
                        "attr_name": attr_name,
                        "value_name": attr_value,
                    },
                )
                attr_count += 1

            for sale_attr_name in ("颜色", "版本", "尺码"):
                if sale_attr_name in selected_attrs:
                    insert_row(
                        cursor,
                        "sku_sale_attr_value",
                        {
                            "sku_id": sku_id,
                            "spu_id": spu_id,
                            "sale_attr_value_id": 0,
                            "sale_attr_id": attr_ids.get(sale_attr_name, 0),
                            "sale_attr_name": sale_attr_name,
                            "sale_attr_value_name": selected_attrs[sale_attr_name],
                        },
                    )

    logger.info("insert_spu_sku_data: spu=%s sku=%s sku_attr=%s", spu_count, sku_count, attr_count)
    fix_image_linked_skus(cursor, category3_ids, trademark_ids, attr_ids, logger)


def build_unique_sku_payload(
    category_name: str,
    spu_name: str,
    generated_sku_names: set,
    sku_index: int,
) -> Tuple[Dict[str, str], str]:
    """生成不重复的 SKU 名称和属性，名称中尽量覆盖该 SKU 的主要属性。"""
    for _ in range(50):
        selected_attrs = build_category_attrs(category_name)
        extras = build_sku_extras(category_name)
        sku_name = build_sku_name(spu_name, category_name, selected_attrs, extras)
        if sku_name not in generated_sku_names:
            return selected_attrs, sku_name

    selected_attrs = build_category_attrs(category_name)
    extras = build_sku_extras(category_name) + [f"{sku_index}件套"]
    sku_name = build_sku_name(spu_name, category_name, selected_attrs, extras)
    return selected_attrs, sku_name


def insert_sku_attrs(cursor, sku_id: int, spu_id: int, attrs: Dict[str, str], attr_ids: Dict[str, int]) -> None:
    """重写指定 SKU 的属性，保证业务侧属性和本地商品详情图片语义一致。"""
    cursor.execute("DELETE FROM sku_attr_value WHERE sku_id = %s", (sku_id,))
    cursor.execute("DELETE FROM sku_sale_attr_value WHERE sku_id = %s", (sku_id,))
    for attr_name, attr_value in attrs.items():
        insert_row(
            cursor,
            "sku_attr_value",
            {
                "attr_id": attr_ids[attr_name],
                "value_id": 0,
                "sku_id": sku_id,
                "attr_name": attr_name,
                "value_name": attr_value,
            },
        )
        if attr_name in {"颜色", "版本", "尺码"}:
            insert_row(
                cursor,
                "sku_sale_attr_value",
                {
                    "sku_id": sku_id,
                    "spu_id": spu_id,
                    "sale_attr_value_id": 0,
                    "sale_attr_id": attr_ids[attr_name],
                    "sale_attr_name": attr_name,
                    "sale_attr_value_name": attr_value,
                },
            )


def fix_image_linked_skus(
    cursor,
    category3_ids: Dict[str, int],
    trademark_ids: Dict[str, int],
    attr_ids: Dict[str, int],
    logger: logging.Logger,
) -> None:
    """固定 sku_id 36-43 的业务数据，使其与本地商品详情图片大致对应。"""
    fixed_spus = {
        14: ("H&U&W 游戏笔记本 014", "游戏笔记本", "H&U&W"),
        15: ("华为 MateBook D15 笔记本 015", "轻薄笔记本", "华为"),
        16: ("H&U&W 4K液晶屏笔记本 016", "轻薄笔记本", "H&U&W"),
        17: ("vivo S20 智能手机 017", "智能手机", "vivo"),
        18: ("Redmi Turbo 4 Pro 智能手机 018", "智能手机", "Redmi"),
        19: ("HUAWEI Mate 40 Pro 智能手机 019", "智能手机", "华为"),
        20: ("Apple iPhone 16 Pro Max 智能手机 020", "智能手机", "Apple"),
        21: ("OPPO Find X8 智能手机 021", "智能手机", "OPPO"),
    }
    for spu_id, (spu_name, category3_name, trademark_name) in fixed_spus.items():
        cursor.execute(
            """
            UPDATE spu_info
            SET spu_name = %s,
                description = %s,
                category3_id = %s,
                tm_id = %s
            WHERE id = %s
            """,
            (
                spu_name,
                f"{spu_name}，用于本地商品详情图片知识图谱样例。",
                category3_ids[category3_name],
                trademark_ids[trademark_name],
                spu_id,
            ),
        )

    fixed_skus = {
        36: (14, "H&U&W 游戏笔记本 014 Intel 11代酷睿 i7 MX450 15.6英寸", "H&U&W", "游戏笔记本", {"颜色": "黑色", "内存": "32GB", "硬盘": "2TB SSD", "显卡": "MX450", "处理器": "Intel i7", "屏幕尺寸": "15.6英寸", "分辨率": "1920*1080", "电池容量": "90Wh", "散热方式": "双风扇", "解锁方式": "指纹解锁"}),
        37: (15, "华为 MateBook D15 笔记本 15英寸 1920*1080 Intel i5", "华为", "轻薄笔记本", {"颜色": "银色", "内存": "8GB", "硬盘": "512GB SSD", "处理器": "Intel i5", "屏幕尺寸": "15英寸", "分辨率": "1920*1080"}),
        38: (16, "H&U&W 4K液晶屏笔记本 016 背光键盘 双向散热", "H&U&W", "轻薄笔记本", {"颜色": "银色", "内存": "16GB", "硬盘": "512GB SSD", "屏幕尺寸": "15.6英寸", "分辨率": "4K", "散热方式": "双风扇"}),
        39: (17, "vivo S20 智能手机 玉露莹白 5000mAh", "vivo", "智能手机", {"颜色": "白色", "版本": "8GB+256GB", "运行内存": "8GB", "机身内存": "256GB", "电池容量": "5000mAh", "摄像头像素": "5000万"}),
        40: (18, "Redmi Turbo 4 Pro 智能手机 6.83英寸 7500mAh", "Redmi", "智能手机", {"颜色": "黑色", "版本": "12GB+256GB", "运行内存": "12GB", "机身内存": "256GB", "电池容量": "7500mAh", "屏幕尺寸": "6.83英寸"}),
        41: (19, "HUAWEI Mate 40 Pro 智能手机 麒麟9000 徕卡影像", "华为", "智能手机", {"颜色": "银色", "版本": "8GB+256GB", "运行内存": "8GB", "机身内存": "256GB", "摄像头像素": "1200万", "处理器": "麒麟9000"}),
        42: (20, "Apple iPhone 16 Pro Max 智能手机 6.9英寸 4800万像素", "Apple", "智能手机", {"颜色": "黑色", "版本": "12GB+512GB", "运行内存": "12GB", "机身内存": "512GB", "摄像头像素": "4800万", "分辨率": "4K", "屏幕尺寸": "6.9英寸"}),
        43: (21, "OPPO Find X8 智能手机 天玑9400 5630mAh", "OPPO", "智能手机", {"颜色": "白色", "版本": "12GB+256GB", "运行内存": "12GB", "机身内存": "256GB", "处理器": "天玑9400", "电池容量": "5630mAh", "屏幕尺寸": "6.59英寸"}),
    }
    for sku_id, (spu_id, sku_name, trademark_name, category3_name, attrs) in fixed_skus.items():
        cursor.execute(
            """
            UPDATE sku_info
            SET spu_id = %s,
                sku_name = %s,
                sku_desc = %s,
                tm_id = %s,
                category3_id = %s
            WHERE id = %s
            """,
            (
                spu_id,
                sku_name,
                sku_name,
                trademark_ids[trademark_name],
                category3_ids[category3_name],
                sku_id,
            ),
        )
        insert_sku_attrs(cursor, sku_id, spu_id, attrs, attr_ids)

    logger.info("fix_image_linked_skus: sku_ids=%s", sorted(fixed_skus))


def count_tables(cursor) -> Dict[str, int]:
    result = {}
    for table in BACKUP_TABLES:
        cursor.execute(f"SELECT COUNT(*) AS count FROM {table}")
        result[table] = cursor.fetchone()["count"]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and enrich gmall data for ecommerce KG construction.")
    parser.add_argument("--target_spu_count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    global DRY_RUN_MODE
    DRY_RUN_MODE = args.dry_run
    random.seed(args.seed)
    logger = setup_logger()

    conn = connect()
    try:
        with conn.cursor() as cursor:
            if args.inspect:
                inspect_database(cursor, logger)
                return

            backup_dir = backup_tables(cursor, logger)
            category3_ids = replace_categories(cursor, logger)
            trademark_ids = replace_trademarks(cursor, logger)
            attr_ids = replace_attrs(cursor, logger)
            insert_spu_sku_data(cursor, category3_ids, trademark_ids, attr_ids, args.target_spu_count, logger)
            table_counts = count_tables(cursor)
            logger.info("table_counts_after: %s", table_counts)
            logger.info("backup_dir: %s", backup_dir)

            if args.dry_run:
                conn.rollback()
                logger.info("dry_run: rollback")
            else:
                conn.commit()
                logger.info("commit: success")
    except Exception:
        conn.rollback()
        logger.exception("rollback: failed to enhance gmall data")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()

