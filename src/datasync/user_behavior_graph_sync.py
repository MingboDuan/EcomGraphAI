import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "user_behavior_graph_sync"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "user_behavior_graph_sync"

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

BEHAVIOR_REL_TYPES = {
    "view": "View",
    "click": "Click",
    "favorite": "Favorite",
}


def setup_logger(log_dir: Path) -> logging.Logger:
    """创建日志对象，同时输出到终端和日志文件。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("user_behavior_graph_sync")
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


def safe_print(value: Any = "") -> None:
    """避免 Windows 控制台编码异常影响脚本输出。"""
    print(str(value).encode("ascii", errors="backslashreplace").decode("ascii"))


def mysql_connect():
    """连接本机 gmall 业务数据库。"""
    return pymysql.connect(**MYSQL_CONFIG)


def read_user_behavior_log(limit: int | None = None) -> List[Dict[str, Any]]:
    """从 user_view_log 读取用户浏览、点击、收藏行为日志。"""
    sql = """
        SELECT
            id AS log_id,
            user_id,
            sku_id,
            COALESCE(behavior_type, 'view') AS behavior_type,
            view_time
        FROM user_view_log
        WHERE user_id IS NOT NULL
          AND sku_id IS NOT NULL
          AND view_time IS NOT NULL
        ORDER BY id
    """
    params: tuple[Any, ...] = ()
    if limit:
        sql += " LIMIT %s"
        params = (limit,)

    with mysql_connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = list(cursor.fetchall())

    for row in rows:
        row["behavior_type"] = str(row.get("behavior_type") or "view").lower()
        if row.get("view_time") is not None:
            row["view_time"] = row["view_time"].strftime("%Y-%m-%d %H:%M:%S")
    return rows


def export_user_behavior_log(rows: List[Dict[str, Any]], output_dir: Path, logger: logging.Logger) -> Path:
    """保存本轮读取到的行为日志，方便后续排查和复现。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "user_behavior_log.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    logger.info("output_path: %s", output_path)
    return output_path


def create_constraints(driver) -> None:
    """创建用户唯一约束，保证重复运行时不会重复创建 User 节点。"""
    driver.execute_query(
        "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        database_=NEO4J_DATABASE,
    )


def remove_old_user_behavior(driver, logger: logging.Logger) -> Dict[str, int]:
    """只清理用户行为侧节点和关系，不影响已有商品业务图谱。"""
    records, _, _ = driver.execute_query(
        """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[r:View|Click|Favorite]->()
        WITH collect(r) AS rels, collect(DISTINCT u) AS users
        FOREACH (rel IN rels | DELETE rel)
        FOREACH (user IN users | DETACH DELETE user)
        RETURN size(rels) AS deleted_relationships, size(users) AS deleted_users
        """,
        database_=NEO4J_DATABASE,
    )
    result = records[0].data() if records else {"deleted_relationships": 0, "deleted_users": 0}
    logger.info("refresh_user_behavior: %s", result)
    return result


def _write_behavior_batch(driver, rel_type: str, rows: List[Dict[str, Any]]) -> int:
    """按关系类型批量写入行为关系，每条日志对应一条关系。"""
    if not rows:
        return 0

    query = f"""
    UNWIND $rows AS row
    MATCH (sku:SKU {{sku_id: row.sku_id}})
    MERGE (user:User {{user_id: row.user_id}})
    MERGE (user)-[rel:{rel_type} {{log_id: row.log_id}}]->(sku)
    ON CREATE SET
        rel.view_time = row.view_time,
        rel.behavior_type = row.behavior_type
    RETURN count(rel) AS written_count
    """
    records, _, _ = driver.execute_query(query, rows=rows, database_=NEO4J_DATABASE)
    return int(records[0]["written_count"]) if records else 0


def write_user_behavior_log(driver, user_behavior_log: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, int]:
    """将用户行为日志写入 Neo4j，只新建 User 节点和行为关系，不创建 SKU 节点。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {key: [] for key in BEHAVIOR_REL_TYPES}
    skipped = 0
    for item in user_behavior_log:
        behavior_type = item.get("behavior_type", "view")
        if behavior_type not in grouped:
            skipped += 1
            continue
        grouped[behavior_type].append(item)

    written_by_type: Dict[str, int] = {}
    for behavior_type, rel_type in BEHAVIOR_REL_TYPES.items():
        written_by_type[behavior_type] = _write_behavior_batch(driver, rel_type, grouped[behavior_type])

    written_by_type["skipped_unknown_behavior"] = skipped
    logger.info("write_user_behavior_log: %s", written_by_type)
    return written_by_type


def query_graph_summary(driver) -> Dict[str, Any]:
    """查询当前图谱整体规模和用户行为侧规模。"""
    records, _, _ = driver.execute_query(
        """
        MATCH (n)
        WITH count(n) AS node_count
        MATCH ()-[r]->()
        WITH node_count, count(r) AS relationship_count
        OPTIONAL MATCH (u:User)
        WITH node_count, relationship_count, count(u) AS user_count
        OPTIONAL MATCH ()-[v:View]->()
        WITH node_count, relationship_count, user_count, count(v) AS view_count
        OPTIONAL MATCH ()-[c:Click]->()
        WITH node_count, relationship_count, user_count, view_count, count(c) AS click_count
        OPTIONAL MATCH ()-[f:Favorite]->()
        RETURN
            node_count,
            relationship_count,
            user_count,
            view_count,
            click_count,
            count(f) AS favorite_count
        """,
        database_=NEO4J_DATABASE,
    )
    return records[0].data() if records else {}


def parse_args():
    parser = argparse.ArgumentParser(description="Sync gmall.user_view_log to Neo4j user behavior graph.")
    parser.add_argument("--limit", type=int, default=None, help="只同步前 N 条用户行为日志，调试时使用。")
    parser.add_argument("--refresh_user_behavior", action="store_true", help="写入前清理上一次用户行为侧 User 节点和行为关系。")
    parser.add_argument("--dry_run", action="store_true", help="只读取和导出日志，不写入 Neo4j。")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_dir.resolve())

    user_behavior_log = read_user_behavior_log(args.limit)
    export_user_behavior_log(user_behavior_log, args.output_dir.resolve(), logger)
    logger.info("read_user_behavior_log: %s", len(user_behavior_log))

    if args.dry_run:
        logger.info("dry_run: skip neo4j write")
        safe_print(f"user_behavior_log: {len(user_behavior_log)}")
        return

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        create_constraints(driver)
        before = query_graph_summary(driver)
        logger.info("before_graph: %s", before)

        if args.refresh_user_behavior:
            remove_old_user_behavior(driver, logger)
            before = query_graph_summary(driver)
            logger.info("after_refresh_graph: %s", before)

        written = write_user_behavior_log(driver, user_behavior_log, logger)
        after = query_graph_summary(driver)
        logger.info("after_graph: %s", after)

    safe_print(f"user_behavior_log: {len(user_behavior_log)}")
    safe_print(f"written: {written}")
    safe_print(f"graph_summary: {after}")


if __name__ == "__main__":
    main()

