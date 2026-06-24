from typing import Any, Dict, List

from neo4j import GraphDatabase

from . import config
from .components import BgeEmbedding, build_fulltext


def connect_driver():
    return GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))


def create_constraints(driver) -> None:
    constraints = [
        "CREATE CONSTRAINT sku_id_unique IF NOT EXISTS FOR (n:SKU) REQUIRE n.sku_id IS UNIQUE",
        "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (n:User) REQUIRE n.user_id IS UNIQUE",
    ]
    for query in constraints:
        driver.execute_query(query, database_=config.NEO4J_DATABASE)


def create_indexes(driver, dimensions: int = 768) -> None:
    for label, cfg in config.NODE_INDEX_CONFIG.items():
        driver.execute_query(
            f"""
            CREATE VECTOR INDEX {cfg['vector_index']} IF NOT EXISTS
            FOR (n:{label}) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            database_=config.NEO4J_DATABASE,
        )
        driver.execute_query(
            f"""
            CREATE FULLTEXT INDEX {cfg['fulltext_index']} IF NOT EXISTS
            FOR (n:{label}) ON EACH [n.fulltext]
            """,
            database_=config.NEO4J_DATABASE,
        )


def read_nodes_without_embedding(driver, label: str, text_property: str, limit: int | None = None) -> List[Dict[str, Any]]:
    prepare_clause = ""
    if label == "Attr":
        prepare_clause = "SET n.search_text = coalesce(n.attr_name, '') + ' ' + coalesce(n.attr_value, '') WITH n"
    limit_clause = " LIMIT $limit" if limit else ""
    records, _, _ = driver.execute_query(
        f"""
        MATCH (n:{label})
        {prepare_clause}
        WHERE n.embedding IS NULL OR n.fulltext IS NULL
        RETURN elementId(n) AS node_id, coalesce(n.{text_property}, '') AS text
        {limit_clause}
        """,
        limit=limit,
        database_=config.NEO4J_DATABASE,
    )
    return [record.data() for record in records if record["text"]]


def upsert_node_index_fields(driver, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    driver.execute_query(
        """
        UNWIND $rows AS row
        MATCH (n)
        WHERE elementId(n) = row.node_id
        SET n.embedding = row.embedding,
            n.fulltext = row.fulltext
        """,
        rows=rows,
        database_=config.NEO4J_DATABASE,
    )


def build_graph_indexes(limit_per_label: int | None = None, logger=None) -> Dict[str, int]:
    embedding = BgeEmbedding()
    stats: Dict[str, int] = {}
    with connect_driver() as driver:
        driver.verify_connectivity()
        create_constraints(driver)
        create_indexes(driver)

        for label, cfg in config.NODE_INDEX_CONFIG.items():
            nodes = read_nodes_without_embedding(driver, label, cfg["text_property"], limit_per_label)
            stats[label] = len(nodes)
            if logger:
                logger.info("index_nodes label=%s count=%s", label, len(nodes))
            for start in range(0, len(nodes), 64):
                batch = nodes[start : start + 64]
                texts = [item["text"] for item in batch]
                vectors = embedding.encode(texts)
                rows = [
                    {
                        "node_id": item["node_id"],
                        "embedding": vector,
                        "fulltext": build_fulltext(item["text"]),
                    }
                    for item, vector in zip(batch, vectors)
                ]
                upsert_node_index_fields(driver, rows)
    return stats


if __name__ == "__main__":
    from .logger import setup_logger

    log = setup_logger("graphrag_index")
    log.info("build_graph_indexes: %s", build_graph_indexes(logger=log))

