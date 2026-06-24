import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs import config


MYSQL_CONFIG = {
    "host": os.getenv("GMALL_MYSQL_HOST", "localhost"),
    "port": int(os.getenv("GMALL_MYSQL_PORT", "3306")),
    "user": os.getenv("GMALL_MYSQL_USER", "root"),
    "password": os.getenv("GMALL_MYSQL_PASSWORD", "your_mysql_password"),
    "database": os.getenv("GMALL_MYSQL_DATABASE", "gmall"),
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
}

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

DEFAULT_EASYOCR_MODEL_DIR = PROJECT_ROOT / "pretrained" / "easyoc"
DEFAULT_UIE_MODEL_DIR = PROJECT_ROOT / "checkpoint" / "uie_0608" / "model_best"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "sku_image_detail_graph_sync" / "sku_image_detail_entities.jsonl"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "sku_image_detail_graph_sync"

# 只保留 doccano 训练集中出现过、且适合商品详情图抽取的属性，降低未训练标签导致的误抽风险。
DEFAULT_SCHEMA = [
    "尺码",
    "分辨率",
    "屏幕尺寸",
    "电视类型",
    "版本",
    "颜色",
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


def setup_logger(log_dir: Path) -> logging.Logger:
    """创建日志器，同时输出到终端和日志文件。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("sku_image_detail_graph_sync")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("log_path: %s", log_path)
    return logger


def read_sku_image_url(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """从 gmall.sku_image 读取本地商品详情图片地址。"""
    sql = """
        SELECT
            sku_id,
            img_url
        FROM sku_image
        WHERE img_url LIKE %s
        ORDER BY sku_id, img_url
    """
    if limit is not None:
        sql += " LIMIT %s"

    with pymysql.connect(**MYSQL_CONFIG) as connection:
        with connection.cursor() as cursor:
            if limit is None:
                cursor.execute(sql, ("/data%",))
            else:
                cursor.execute(sql, ("/data%", limit))
            return list(cursor.fetchall())


def resolve_local_image_path(img_url: str) -> Path:
    """将数据库中的 /data/images/... 转为本机项目内图片路径。"""
    return PROJECT_ROOT / img_url.lstrip("/").replace("/", os.sep)


def build_ocr_reader(model_dir: Path, use_gpu: Optional[bool], logger: logging.Logger):
    """构建 EasyOCR 识别器，默认在 CUDA 可用时启用 GPU。"""
    import easyocr
    import torch

    model_dir.mkdir(parents=True, exist_ok=True)
    user_network_dir = model_dir / "user_network"
    user_network_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    enable_gpu = cuda_available if use_gpu is None else use_gpu
    if enable_gpu and not cuda_available:
        raise RuntimeError("已指定使用 GPU，但 torch.cuda.is_available() 为 False。")

    logger.info("torch_cuda_available: %s", cuda_available)
    logger.info("easyocr_gpu: %s", enable_gpu)
    return easyocr.Reader(
        ["ch_sim", "en"],
        gpu=enable_gpu,
        model_storage_directory=str(model_dir),
        user_network_directory=str(user_network_dir),
    )


def read_image_text(reader, image_path: Path) -> str:
    """识别单张图片文字，只返回纯文本。"""
    result = reader.readtext(str(image_path), detail=0)
    return " ".join(text.strip() for text in result if text and text.strip())


def get_sku_image_text(
    sku_image_urls: List[Dict[str, Any]],
    reader,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """按 sku_image 表记录逐张 OCR，生成 sku_id、img_url、text。"""
    records = []
    total = len(sku_image_urls)

    for index, item in enumerate(sku_image_urls, start=1):
        sku_id = item["sku_id"]
        img_url = item["img_url"]
        image_path = resolve_local_image_path(img_url)

        if not image_path.exists():
            logger.warning("[%s/%s] missing_image: %s", index, total, image_path)
            text = ""
        else:
            start_time = time.perf_counter()
            text = read_image_text(reader, image_path)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "[%s/%s] OCR sku_id=%s img_url=%s text_len=%s seconds=%.4f",
                index,
                total,
                sku_id,
                img_url,
                len(text),
                elapsed,
            )

        records.append({"sku_id": sku_id, "img_url": img_url, "text": text})

    return records


def build_spell_checker(logger: logging.Logger):
    """加载 T5 纠错模型，复用项目里的 SpellCheckT5Predictor 批量预测器。"""
    import torch
    from transformers import AutoTokenizer

    from models.spell_check_t5 import SpellCheckT5
    from runner.Predictor import SpellCheckT5Predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("spell_check_device: %s", device)

    model = SpellCheckT5(pretrained_path=config.PRE_TRAINED_DIR / "mengzi-t5-base-chinese-correction")
    checkpoint_path = config.CHECKPOINT_DIR / "spell_check_t5" / "best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / "mengzi-t5-base-chinese-correction")
    return SpellCheckT5Predictor(model, tokenizer, device)


def batch_spell_check(
    records: List[Dict[str, Any]],
    predictor,
    batch_size: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """对 OCR 文本进行批量纠错，并将结果写入 checked_text。"""
    texts = [record["text"] for record in records]
    checked_texts: List[str] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch_texts = [text if text else "" for text in texts[start : start + batch_size]]
        start_time = time.perf_counter()
        batch_result = predictor.predict(batch_texts)
        elapsed = time.perf_counter() - start_time

        checked_texts.extend(batch_result)
        logger.info(
            "spell_check_batch: %s-%s/%s seconds=%.4f",
            start + 1,
            min(start + batch_size, total),
            total,
            elapsed,
        )

    for record, checked_text in zip(records, checked_texts):
        record["checked_text"] = checked_text
    return records


def release_cuda_cache() -> None:
    """释放前一阶段模型占用的 CUDA 缓存，降低连续加载 OCR/T5/UIE 时显存压力。"""
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_uie_predictor(schema: List[str], batch_size: int, device: str, logger: logging.Logger):
    """加载微调后的 UIE 模型，并设置批量实体抽取 schema。"""
    sys.path.insert(0, str(config.EXTERNAL_LIB_DIR / "uie_pytorch"))
    from uie_predictor import UIEPredictor

    logger.info("uie_model_dir: %s", DEFAULT_UIE_MODEL_DIR)
    logger.info("uie_schema: %s", schema)
    logger.info("uie_device: %s", device)
    return UIEPredictor(
        model="uie-base",
        task_path=str(DEFAULT_UIE_MODEL_DIR),
        schema=schema,
        device=device,
        batch_size=batch_size,
        max_seq_len=512,
    )


def extract_sku_entities(
    records: List[Dict[str, Any]],
    schema: List[str],
    batch_size: int,
    device: str,
    min_probability: float,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """对纠错后的商品详情文本进行 UIE 批量实体抽取。"""
    predictor = build_uie_predictor(schema, batch_size, device, logger)
    texts = [record.get("checked_text") or record.get("text") or "" for record in records]

    start_time = time.perf_counter()
    results = safe_batch_uie_predict(predictor, texts, batch_size, logger)
    logger.info("uie_extract_count: %s seconds=%.4f", len(results), time.perf_counter() - start_time)

    for record, result in zip(records, results):
        entities = []
        for attr_name, values in result.items():
            if attr_name not in schema:
                continue
            for value in values:
                attr_value = str(value.get("text", "")).strip()
                probability = float(value.get("probability", 0.0))
                if not attr_value or probability < min_probability:
                    continue
                entities.append(
                    {
                        "sku_id": record["sku_id"],
                        "img_url": record["img_url"],
                        "attr_name": attr_name,
                        "attr_value": attr_value,
                        "probability": probability,
                    }
                )
        record["entities"] = entities
    return records


def safe_batch_uie_predict(predictor, texts: List[str], batch_size: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    """批量调用 UIE；若某个批次触发库内部异常，则降级为逐条抽取。"""
    all_results: List[Dict[str, Any]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch_texts = texts[start : start + batch_size]
        try:
            batch_results = predictor(batch_texts)
        except Exception as exc:
            logger.warning(
                "uie_batch_failed: %s-%s/%s, fallback_to_single, error=%s",
                start + 1,
                min(start + batch_size, total),
                total,
                exc,
            )
            batch_results = []
            for offset, text in enumerate(batch_texts, start=start + 1):
                try:
                    single_result = predictor(text)
                    batch_results.append(single_result[0] if single_result else {})
                except Exception as single_exc:
                    logger.warning("uie_single_failed: index=%s, error=%s", offset, single_exc)
                    batch_results.append({})

        all_results.extend(batch_results)
        logger.info(
            "uie_batch: %s-%s/%s",
            start + 1,
            min(start + batch_size, total),
            total,
        )

    return all_results


def flatten_unique_entities(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将每张图片上的实体拉平，并按 sku_id、attr_name、attr_value 去重。"""
    seen = set()
    unique_entities = []
    for record in records:
        for entity in record.get("entities", []):
            key = (entity["sku_id"], entity["attr_name"], entity["attr_value"])
            if key in seen:
                continue
            seen.add(key)
            unique_entities.append(entity)
    return unique_entities


def keep_best_entity_per_attr(records: List[Dict[str, Any]], logger: logging.Logger) -> List[Dict[str, Any]]:
    """同一 SKU 的同一属性名只保留置信度最高的实体，避免一个属性写入多个候选值。"""
    best_entities: Dict[tuple, Dict[str, Any]] = {}
    total_entities = 0

    for record in records:
        for entity in record.get("entities", []):
            total_entities += 1
            key = (entity["sku_id"], entity["attr_name"])
            old_entity = best_entities.get(key)
            if old_entity is None or entity["probability"] > old_entity["probability"]:
                best_entities[key] = entity

    best_keys = {
        (
            entity["sku_id"],
            entity["img_url"],
            entity["attr_name"],
            entity["attr_value"],
        )
        for entity in best_entities.values()
    }

    for record in records:
        record["entities"] = [
            entity
            for entity in record.get("entities", [])
            if (
                entity["sku_id"],
                entity["img_url"],
                entity["attr_name"],
                entity["attr_value"],
            )
            in best_keys
        ]

    logger.info(
        "best_entity_filter: before=%s after=%s removed=%s",
        total_entities,
        len(best_entities),
        total_entities - len(best_entities),
    )
    return records


def write_sku_entities(driver, sku_entities: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, int]:
    """将详情图抽取出的属性写入 Neo4j，不删除已有业务库图谱。"""
    skipped_existing_attr = 0
    skipped_missing_sku = 0
    created_or_merged = 0

    for entity in sku_entities:
        records, _, _ = driver.execute_query(
            """
            MATCH (sku:SKU {sku_id: $sku_id})
            OPTIONAL MATCH (sku)-[:Have]->(attr_exist:Attr {attr_name: $attr_name})
            WITH sku, attr_exist
            WHERE attr_exist IS NULL
            MERGE (attr:Attr {attr_name: $attr_name, attr_value: $attr_value})
            MERGE (sku)-[:Have]->(attr)
            RETURN count(attr) AS written_count
            """,
            parameters_=entity,
            database_=NEO4J_DATABASE,
        )

        written_count = records[0]["written_count"] if records else 0
        if written_count:
            created_or_merged += 1
            continue

        check_records, _, _ = driver.execute_query(
            """
            OPTIONAL MATCH (sku:SKU {sku_id: $sku_id})
            OPTIONAL MATCH (sku)-[:Have]->(attr_exist:Attr {attr_name: $attr_name})
            RETURN sku IS NULL AS missing_sku, attr_exist IS NOT NULL AS existing_attr
            """,
            parameters_=entity,
            database_=NEO4J_DATABASE,
        )
        if check_records and check_records[0]["missing_sku"]:
            skipped_missing_sku += 1
        else:
            skipped_existing_attr += 1

    stats = {
        "created_or_merged": created_or_merged,
        "skipped_existing_attr": skipped_existing_attr,
        "skipped_missing_sku": skipped_missing_sku,
    }
    logger.info("neo4j_write_stats: %s", stats)
    return stats


def load_entities_from_output(output_path: Path) -> List[Dict[str, Any]]:
    """从上一次输出结果中读取详情侧实体，用于刷新旧图谱关系。"""
    if not output_path.exists():
        return []

    entities = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            entities.extend(record.get("entities", []))
    return entities


def delete_detail_entities(driver, sku_entities: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, int]:
    """删除上一轮详情侧写入的属性关系，并清理不再被引用的 Attr 节点。"""
    deleted_relationships = 0
    deleted_orphan_attrs = 0

    for entity in sku_entities:
        records, _, _ = driver.execute_query(
            """
            MATCH (sku:SKU {sku_id: $sku_id})-[rel:Have]->(attr:Attr {
                attr_name: $attr_name,
                attr_value: $attr_value
            })
            DELETE rel
            RETURN count(rel) AS deleted_count
            """,
            parameters_=entity,
            database_=NEO4J_DATABASE,
        )
        deleted_relationships += records[0]["deleted_count"] if records else 0

        records, _, _ = driver.execute_query(
            """
            MATCH (attr:Attr {
                attr_name: $attr_name,
                attr_value: $attr_value
            })
            WHERE NOT (attr)<--()
            DELETE attr
            RETURN count(attr) AS deleted_count
            """,
            parameters_=entity,
            database_=NEO4J_DATABASE,
        )
        deleted_orphan_attrs += records[0]["deleted_count"] if records else 0

    stats = {
        "deleted_relationships": deleted_relationships,
        "deleted_orphan_attrs": deleted_orphan_attrs,
    }
    logger.info("neo4j_delete_old_detail_stats: %s", stats)
    return stats


def log_entities_by_sku(sku_entities: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """按 sku_id 打印本轮详情侧抽取到的实体，方便核对每个商品新增了哪些属性。"""
    grouped_entities = defaultdict(list)
    for entity in sku_entities:
        grouped_entities[entity["sku_id"]].append(entity)

    logger.info("detail_entities_by_sku:")
    for sku_id in sorted(grouped_entities):
        entity_text = "；".join(
            f"{entity['attr_name']}={entity['attr_value']}({entity['probability']:.4f})"
            for entity in sorted(grouped_entities[sku_id], key=lambda item: item["attr_name"])
        )
        logger.info("  sku_id=%s: %s", sku_id, entity_text)


def get_graph_count(driver) -> Dict[str, int]:
    """查询当前 Neo4j 图谱中的节点数和关系数。"""
    records, _, _ = driver.execute_query(
        """
        MATCH (n)
        WITH count(n) AS node_count
        MATCH ()-[r]->()
        RETURN node_count, count(r) AS relationship_count
        """,
        database_=NEO4J_DATABASE,
    )
    if not records:
        return {"node_count": 0, "relationship_count": 0}
    return {
        "node_count": records[0]["node_count"],
        "relationship_count": records[0]["relationship_count"],
    }


def log_detail_graph_summary(
    before_count: Dict[str, int],
    after_count: Dict[str, int],
    new_detail_entities: int,
    logger: logging.Logger,
) -> None:
    """打印商品详情侧本轮新增量，以及写入后的图谱总规模。"""
    added_attr_nodes = after_count["node_count"] - before_count["node_count"]
    added_have_relationships = after_count["relationship_count"] - before_count["relationship_count"]

    logger.info("本轮商品详情侧新增：")
    logger.info("new_detail_entities: %s", new_detail_entities)
    logger.info("新增 Attr 节点: %s", added_attr_nodes)
    logger.info("新增 Have 关系: %s", added_have_relationships)
    logger.info("当前图谱：")
    logger.info(
        "node_count: %s + %s = %s",
        before_count["node_count"],
        added_attr_nodes,
        after_count["node_count"],
    )
    logger.info(
        "relationship_count: %s + %s = %s",
        before_count["relationship_count"],
        added_have_relationships,
        after_count["relationship_count"],
    )


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """保存 OCR、纠错、实体抽取和写图前的中间结果，便于排查。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="从商品详情图片抽取属性并增量写入 Neo4j 图谱。")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--easyocr_model_dir", type=Path, default=DEFAULT_EASYOCR_MODEL_DIR)
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条 sku_image 记录，便于调试。")
    parser.add_argument("--spell_batch_size", type=int, default=8)
    parser.add_argument("--uie_batch_size", type=int, default=16)
    parser.add_argument("--min_probability", type=float, default=0.6, help="实体抽取最小置信度阈值。")
    parser.add_argument("--skip_spell_check", action="store_true", help="兼容旧参数；当前默认已禁用 T5 纠错。")
    parser.add_argument("--dry_run", action="store_true", help="只生成抽取结果，不写入 Neo4j。")
    parser.add_argument("--refresh_detail_attrs", action="store_true", help="写入前先删除上一轮详情侧抽取属性关系。")

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--gpu", action="store_true", help="OCR 和 UIE 强制使用 GPU。")
    device_group.add_argument("--cpu", action="store_true", help="OCR 和 UIE 强制使用 CPU。")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_dir.resolve())
    use_gpu = True if args.gpu else False if args.cpu else None
    output_path = args.output_path.resolve()
    old_detail_entities = load_entities_from_output(output_path) if args.refresh_detail_attrs else []
    if args.refresh_detail_attrs:
        logger.info("old_detail_entity_count: %s", len(old_detail_entities))

    sku_image_urls = read_sku_image_url(limit=args.limit)
    logger.info("sku_image_url_count: %s", len(sku_image_urls))

    reader = build_ocr_reader(args.easyocr_model_dir.resolve(), use_gpu, logger)
    records = get_sku_image_text(sku_image_urls, reader, logger)
    del reader
    release_cuda_cache()

    # 商品详情图中包含大量型号、尺寸、容量、分辨率等关键数字信息。
    # T5 纠错模型会倾向于把 OCR 文本改写成更自然的中文句子，可能误改 7.83、7.9 英寸、
    # 5000mAh、1920x1080 等商品参数，导致后续 UIE 实体抽取结果偏移。
    # 因此这里不再启用 T5 纠错，直接使用 OCR 原始 text 作为 checked_text 输入 UIE。
    # 如确需重新启用，可恢复下面三行并移除当前 checked_text 赋值逻辑：
    # spell_checker = build_spell_checker(logger)
    # records = batch_spell_check(records, spell_checker, args.spell_batch_size, logger)
    # release_cuda_cache()
    for record in records:
        record["checked_text"] = record["text"]
    logger.info("spell_check_disabled: use raw OCR text for UIE extraction")

    uie_device = "gpu" if (use_gpu is True or (use_gpu is None and _torch_cuda_available())) else "cpu"
    records = extract_sku_entities(
        records,
        DEFAULT_SCHEMA,
        args.uie_batch_size,
        uie_device,
        args.min_probability,
        logger,
    )
    records = keep_best_entity_per_attr(records, logger)
    sku_entities = flatten_unique_entities(records)

    write_jsonl(records, output_path)
    logger.info("output_path: %s", output_path)
    logger.info("unique_entity_count: %s", len(sku_entities))
    log_entities_by_sku(sku_entities, logger)

    if args.dry_run:
        logger.info("dry_run: true, skip neo4j writing")
        return

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        if args.refresh_detail_attrs and old_detail_entities:
            delete_detail_entities(driver, old_detail_entities, logger)
        graph_count_before_write = get_graph_count(driver)
        write_sku_entities(driver, sku_entities, logger)
        graph_count_after_write = get_graph_count(driver)
        log_detail_graph_summary(
            graph_count_before_write,
            graph_count_after_write,
            len(sku_entities),
            logger,
        )


def _torch_cuda_available() -> bool:
    """独立检测 CUDA，避免主流程直接依赖 torch 顶层导入。"""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()

