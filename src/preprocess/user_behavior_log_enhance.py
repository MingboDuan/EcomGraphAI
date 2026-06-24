import argparse
import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pymysql
from pymysql.cursors import DictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "user_behavior_log_enhance"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "output" / "user_behavior_log_backup"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "output" / "user_behavior_log_audit"

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

BEHAVIOR_TYPES = ("view", "click", "favorite")


def setup_logger(log_dir: Path) -> logging.Logger:
    """创建日志对象，同时输出到终端和文件。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("user_behavior_log_enhance")
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


def ensure_behavior_column(cursor, logger: logging.Logger) -> None:
    """确保 user_view_log 支持浏览、点击、收藏等行为类型。"""
    cursor.execute("SHOW COLUMNS FROM user_view_log LIKE 'behavior_type'")
    if cursor.fetchone():
        logger.info("behavior_type_column: exists")
        return

    cursor.execute(
        """
        ALTER TABLE user_view_log
        ADD COLUMN behavior_type VARCHAR(20) NOT NULL DEFAULT 'view' COMMENT '用户行为类型：view/click/favorite'
        AFTER sku_id
        """
    )
    logger.info("behavior_type_column: added")


def backup_user_view_log(cursor, backup_root: Path, logger: logging.Logger) -> Path:
    """备份当前 user_view_log，便于回滚和追踪数据变化。"""
    backup_dir = backup_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "user_view_log.jsonl"

    cursor.execute("SELECT * FROM user_view_log ORDER BY id")
    rows = cursor.fetchall()
    with backup_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    logger.info("backup_user_view_log: rows=%s path=%s", len(rows), backup_path)
    return backup_path


def read_sku_pool(cursor) -> List[Dict[str, Any]]:
    """读取 SKU、品牌、类目信息，用于按用户兴趣偏好生成行为日志。"""
    cursor.execute(
        """
        SELECT
            ski.id AS sku_id,
            ski.sku_name,
            bt.tm_name AS trademark_name,
            bc1.name AS category1_name,
            bc2.name AS category2_name,
            bc3.name AS category3_name
        FROM sku_info ski
        LEFT JOIN base_trademark bt ON ski.tm_id = bt.id
        LEFT JOIN base_category3 bc3 ON ski.category3_id = bc3.id
        LEFT JOIN base_category2 bc2 ON bc3.category2_id = bc2.id
        LEFT JOIN base_category1 bc1 ON bc2.category1_id = bc1.id
        ORDER BY ski.id
        """
    )
    return list(cursor.fetchall())


def build_indexes(sku_pool: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """按三级品类和品牌建立索引，便于模拟用户兴趣聚焦。"""
    by_category = defaultdict(list)
    by_trademark = defaultdict(list)
    for sku in sku_pool:
        by_category[sku["category3_name"]].append(sku)
        by_trademark[sku["trademark_name"]].append(sku)
    return dict(by_category), dict(by_trademark)


def random_behavior_time() -> datetime:
    """生成 2020-2026 年之间的行为时间，不再使用 2010 年代数据。"""
    start = datetime(2020, 1, 1, 0, 0, 0)
    end = datetime(2026, 6, 15, 23, 59, 59)
    seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, seconds))


def choose_sku_for_user(
    sku_pool: List[Dict[str, Any]],
    by_category: Dict[str, List[Dict[str, Any]]],
    by_trademark: Dict[str, List[Dict[str, Any]]],
    preferred_categories: List[str],
    preferred_trademarks: List[str],
) -> Dict[str, Any]:
    """按用户偏好选择 SKU，少量穿插其他商品用于模拟探索行为。"""
    roll = random.random()
    candidates: List[Dict[str, Any]]
    if roll < 0.65:
        candidates = by_category[random.choice(preferred_categories)]
    elif roll < 0.85:
        candidates = by_trademark[random.choice(preferred_trademarks)]
    else:
        candidates = sku_pool
    return random.choice(candidates)


def build_user_profiles(
    user_count: int,
    categories: List[str],
    trademarks: List[str],
) -> Dict[int, Dict[str, List[str]]]:
    """生成用户兴趣画像，每个用户偏好 2-3 个品类和 1-3 个品牌。"""
    profiles = {}
    for user_id in range(1, user_count + 1):
        profiles[user_id] = {
            "categories": random.sample(categories, k=min(random.randint(2, 3), len(categories))),
            "trademarks": random.sample(trademarks, k=min(random.randint(1, 3), len(trademarks))),
        }
    return profiles


def generate_behavior_logs(
    sku_pool: List[Dict[str, Any]],
    target_count: int,
    user_count: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """模拟用户浏览、点击、收藏序列，用于后续构建用户兴趣图谱。"""
    by_category, by_trademark = build_indexes(sku_pool)
    profiles = build_user_profiles(user_count, list(by_category), list(by_trademark))
    logs: List[Dict[str, Any]] = []

    user_ids = list(profiles)
    while len(logs) < target_count:
        user_id = random.choice(user_ids)
        profile = profiles[user_id]
        session_time = random_behavior_time()
        session_len = random.randint(3, 8)

        for step in range(session_len):
            if len(logs) >= target_count:
                break
            sku = choose_sku_for_user(
                sku_pool,
                by_category,
                by_trademark,
                profile["categories"],
                profile["trademarks"],
            )
            base_time = session_time + timedelta(minutes=random.randint(0, 45), seconds=random.randint(0, 59))

            logs.append(
                {
                    "user_id": user_id,
                    "sku_id": sku["sku_id"],
                    "behavior_type": "view",
                    "view_time": base_time,
                }
            )

            if len(logs) < target_count and random.random() < 0.38:
                logs.append(
                    {
                        "user_id": user_id,
                        "sku_id": sku["sku_id"],
                        "behavior_type": "click",
                        "view_time": base_time + timedelta(seconds=random.randint(5, 90)),
                    }
                )

            if len(logs) < target_count and random.random() < 0.12:
                logs.append(
                    {
                        "user_id": user_id,
                        "sku_id": sku["sku_id"],
                        "behavior_type": "favorite",
                        "view_time": base_time + timedelta(seconds=random.randint(30, 180)),
                    }
                )

    logs.sort(key=lambda item: item["view_time"])
    logger.info("generated_behavior_logs: %s", len(logs))
    return logs[:target_count]


def replace_user_view_log(cursor, logs: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """清空旧日志并写入新的用户行为日志。"""
    cursor.execute("DELETE FROM user_view_log")
    cursor.execute("ALTER TABLE user_view_log AUTO_INCREMENT = 1")

    cursor.executemany(
        """
        INSERT INTO user_view_log (user_id, sku_id, behavior_type, view_time)
        VALUES (%s, %s, %s, %s)
        """,
        [(item["user_id"], item["sku_id"], item["behavior_type"], item["view_time"]) for item in logs],
    )
    logger.info("insert_user_view_log: %s", len(logs))


def export_audit(logs: List[Dict[str, Any]], audit_dir: Path, logger: logging.Logger) -> Path:
    """导出生成结果，便于后续检查用户行为分布。"""
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "user_behavior_log_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as f:
        for item in logs:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    logger.info("audit_path: %s", audit_path)
    return audit_path


def log_summary(cursor, logger: logging.Logger) -> None:
    """打印行为类型和年份分布。"""
    cursor.execute("SELECT behavior_type, COUNT(1) AS count FROM user_view_log GROUP BY behavior_type ORDER BY behavior_type")
    logger.info("behavior_type_count: %s", cursor.fetchall())
    cursor.execute("SELECT YEAR(view_time) AS year, COUNT(1) AS count FROM user_view_log GROUP BY YEAR(view_time) ORDER BY year")
    logger.info("year_count: %s", cursor.fetchall())
    cursor.execute("SELECT MIN(view_time) AS min_time, MAX(view_time) AS max_time FROM user_view_log")
    logger.info("time_range: %s", cursor.fetchone())
    cursor.execute("SELECT COUNT(DISTINCT user_id) AS user_count, COUNT(DISTINCT sku_id) AS sku_count FROM user_view_log")
    logger.info("distinct_count: %s", cursor.fetchone())


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance gmall.user_view_log with view/click/favorite behaviors.")
    parser.add_argument("--target_count", type=int, default=6000, help="生成用户行为日志总条数。")
    parser.add_argument("--user_count", type=int, default=180, help="模拟用户数量。")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="只读取并打印 user_view_log 当前数据分布。")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--backup_dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    logger = setup_logger(args.log_dir.resolve())

    conn = connect()
    try:
        with conn.cursor() as cursor:
            if args.inspect:
                log_summary(cursor, logger)
                return

            ensure_behavior_column(cursor, logger)
            backup_user_view_log(cursor, args.backup_dir.resolve(), logger)
            sku_pool = read_sku_pool(cursor)
            logger.info("sku_pool_count: %s", len(sku_pool))
            logs = generate_behavior_logs(sku_pool, args.target_count, args.user_count, logger)
            export_audit(logs, args.audit_dir.resolve(), logger)
            replace_user_view_log(cursor, logs, logger)
            log_summary(cursor, logger)

        if args.dry_run:
            conn.rollback()
            logger.info("dry_run: rollback")
        else:
            conn.commit()
            logger.info("commit: success")
    except Exception:
        conn.rollback()
        logger.exception("rollback: failed to enhance user behavior log")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()

