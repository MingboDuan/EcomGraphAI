import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
EXTERNAL_LIB_DIR = PROJECT_ROOT / "external_lib"

BGE_MODEL_DIR = PRETRAINED_DIR / "bge-base-zh-v1.5"
UIE_MODEL_DIR = CHECKPOINT_DIR / "uie_0608" / "model_best"
SPELL_MODEL_DIR = PRETRAINED_DIR / "mengzi-t5-base-chinese-correction"
SPELL_CHECKPOINT = CHECKPOINT_DIR / "spell_check_t5" / "best.pt"

LOG_DIR = PROJECT_ROOT / "logs" / "graphrag"
TRACE_DIR = PROJECT_ROOT / "output" / "graphrag_traces"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

LLM_PROVIDER = os.getenv("GRAPHRAG_LLM_PROVIDER", "tongyi")
LLM_MODEL = os.getenv("GRAPHRAG_LLM_MODEL", "qwen-turbo")
LLM_BASE_URL = os.getenv(
    "GRAPHRAG_LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY", "")

DEFAULT_TOP_K = int(os.getenv("GRAPHRAG_TOP_K", "8"))
DEFAULT_RESULT_LIMIT = int(os.getenv("GRAPHRAG_RESULT_LIMIT", "30"))

GRAPH_SCHEMA_TEXT = """
节点:
- SKU(sku_id, sku_name)
- SPU(spu_name)
- Category1(category1_name)
- Category2(category2_name)
- Category3(category3_name)
- Trademark(trademark_name)
- Attr(attr_name, attr_value)
- User(user_id)

关系:
- (SKU)-[:Belong]->(SPU)
- (SPU)-[:Belong]->(Category3)
- (Category3)-[:Belong]->(Category2)
- (Category2)-[:Belong]->(Category1)
- (SPU)-[:Belong]->(Trademark)
- (SKU)-[:Have]->(Attr)
- (User)-[:View]->(SKU)
- (User)-[:Click]->(SKU)
- (User)-[:Favorite]->(SKU)
""".strip()

NODE_INDEX_CONFIG = {
    "SKU": {"text_property": "sku_name", "vector_index": "sku_vector", "fulltext_index": "sku_fulltext"},
    "SPU": {"text_property": "spu_name", "vector_index": "spu_vector", "fulltext_index": "spu_fulltext"},
    "Category1": {"text_property": "category1_name", "vector_index": "category1_vector", "fulltext_index": "category1_fulltext"},
    "Category2": {"text_property": "category2_name", "vector_index": "category2_vector", "fulltext_index": "category2_fulltext"},
    "Category3": {"text_property": "category3_name", "vector_index": "category3_vector", "fulltext_index": "category3_fulltext"},
    "Trademark": {"text_property": "trademark_name", "vector_index": "trademark_vector", "fulltext_index": "trademark_fulltext"},
    "Attr": {"text_property": "search_text", "vector_index": "attr_vector", "fulltext_index": "attr_fulltext"},
}

UIE_SCHEMA = [
    "尺码",
    "分辨率",
    "屏幕尺寸",
    "电视类型",
    "颜色",
    "版本",
    "机身内存",
    "运行内存",
    "内存",
    "硬盘",
    "显卡",
    "处理器",
    "类别",
    "粮食调味",
    "香水彩妆",
    "功效",
    "电池容量",
    "摄像头像素",
    "散热方式",
    "解锁方式",
]

