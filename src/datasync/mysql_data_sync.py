import os
from contextlib import contextmanager
from typing import Any, Dict, List

import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor


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


@contextmanager
def mysql_cursor():
    conn = pymysql.connect(**MYSQL_CONFIG)
    try:
        with conn.cursor() as cursor:
            yield cursor
    finally:
        conn.close()


def read_sku_base_info(cursor) -> List[Dict[str, Any]]:
    cursor.execute(
        """
        SELECT
            ski.id AS sku_id,
            ski.sku_name,
            spi.spu_name,
            bc3.name AS category3_name,
            bc2.name AS category2_name,
            bc1.name AS category1_name,
            bt.tm_name AS trademark_name
        FROM sku_info ski
        LEFT JOIN spu_info spi ON ski.spu_id = spi.id
        LEFT JOIN base_category3 bc3 ON spi.category3_id = bc3.id
        LEFT JOIN base_category2 bc2 ON bc3.category2_id = bc2.id
        LEFT JOIN base_category1 bc1 ON bc2.category1_id = bc1.id
        LEFT JOIN base_trademark bt ON spi.tm_id = bt.id
        """
    )
    return list(cursor.fetchall())


def read_sku_attr_info(cursor) -> List[Dict[str, Any]]:
    cursor.execute(
        """
        SELECT
            sku_id,
            attr_name,
            value_name AS attr_value
        FROM sku_attr_value
        UNION ALL
        SELECT
            sku_id,
            sale_attr_name AS attr_name,
            sale_attr_value_name AS attr_value
        FROM sku_sale_attr_value
        """
    )
    return list(cursor.fetchall())


def clear_graph(driver):
    driver.execute_query(
        "MATCH (n) DETACH DELETE n",
        database_=NEO4J_DATABASE,
    )


def write_sku_base_info(driver, sku_base_info: List[Dict[str, Any]]):
    for sku in sku_base_info:
        if not all(
            sku.get(key)
            for key in (
                "sku_id",
                "sku_name",
                "spu_name",
                "category3_name",
                "category2_name",
                "category1_name",
                "trademark_name",
            )
        ):
            continue

        driver.execute_query(
            """
            MERGE (sku:SKU {sku_id: $sku_id, sku_name: $sku_name})
            MERGE (spu:SPU {spu_name: $spu_name})
            MERGE (cate3:Category3 {category3_name: $category3_name})
            MERGE (cate2:Category2 {category2_name: $category2_name})
            MERGE (cate1:Category1 {category1_name: $category1_name})
            MERGE (tm:Trademark {trademark_name: $trademark_name})
            MERGE (sku)-[:Belong]->(spu)
            MERGE (spu)-[:Belong]->(cate3)
            MERGE (cate3)-[:Belong]->(cate2)
            MERGE (cate2)-[:Belong]->(cate1)
            MERGE (spu)-[:Belong]->(tm)
            """,
            parameters_=sku,
            database_=NEO4J_DATABASE,
        )


def write_sku_attr_info(driver, sku_attr_info: List[Dict[str, Any]]):
    for attr in sku_attr_info:
        if not all(attr.get(key) for key in ("sku_id", "attr_name", "attr_value")):
            continue

        driver.execute_query(
            """
            MATCH (sku:SKU {sku_id: $sku_id})
            MERGE (attr:Attr {attr_name: $attr_name, attr_value: $attr_value})
            MERGE (sku)-[:Have]->(attr)
            """,
            parameters_=attr,
            database_=NEO4J_DATABASE,
        )


def safe_print(value: Any = ""):
    print(str(value).encode("ascii", errors="backslashreplace").decode("ascii"))


def main():
    with mysql_cursor() as cursor:
        sku_base_info = read_sku_base_info(cursor)
        sku_attr_info = read_sku_attr_info(cursor)

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        clear_graph(driver)
        write_sku_base_info(driver, sku_base_info)
        write_sku_attr_info(driver, sku_attr_info)

        records, _, _ = driver.execute_query(
            """
            MATCH (n)
            WITH count(n) AS node_count
            MATCH ()-[r]->()
            RETURN node_count, count(r) AS relationship_count
            """,
            database_=NEO4J_DATABASE,
        )

    safe_print(f"sku_base_info: {len(sku_base_info)}")
    safe_print(f"sku_attr_info: {len(sku_attr_info)}")
    safe_print(records[0].data() if records else {})


if __name__ == "__main__":
    main()

