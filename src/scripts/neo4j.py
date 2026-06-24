import argparse
import os
import sys
from pathlib import Path


# The file is named neo4j.py, which would shadow the official neo4j package
# when this script is executed directly. Remove this directory before import.
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

from neo4j import GraphDatabase


DEFAULT_URI = "neo4j://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "your_neo4j_password"
DEFAULT_DATABASE = "neo4j"

DEFAULT_QUERY = """
MATCH (n)
WITH count(n) AS node_count
MATCH ()-[r]->()
RETURN node_count, count(r) AS relationship_count
"""

SAMPLE_QUERY = """
MATCH (a)-[r]->(b)
RETURN
    labels(a) AS source_labels,
    coalesce(a.name, a.title, elementId(a)) AS source,
    type(r) AS relationship,
    labels(b) AS target_labels,
    coalesce(b.name, b.title, elementId(b)) AS target
LIMIT $limit
"""


def safe_print(value=""):
    text = str(value)
    print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def get_config():
    return {
        "uri": os.getenv("NEO4J_URI", DEFAULT_URI),
        "user": os.getenv("NEO4J_USER", DEFAULT_USER),
        "password": os.getenv("NEO4J_PASSWORD", DEFAULT_PASSWORD),
        "database": os.getenv("NEO4J_DATABASE", DEFAULT_DATABASE),
    }


def run_query(driver, query, database, **parameters):
    records, summary, _ = driver.execute_query(
        query,
        database_=database,
        **parameters,
    )
    return records, summary


def print_records(records):
    if not records:
        safe_print("No records returned.")
        return

    for index, record in enumerate(records, start=1):
        safe_print(f"[{index}] {record.data()}")


def main():
    parser = argparse.ArgumentParser(description="Test Neo4j connection and run Cypher queries.")
    parser.add_argument("--query", default=None, help="Cypher query to execute.")
    parser.add_argument("--limit", default=10, type=int, help="Sample relationship query limit.")
    args = parser.parse_args()

    config = get_config()
    auth = (config["user"], config["password"])

    with GraphDatabase.driver(config["uri"], auth=auth) as driver:
        driver.verify_connectivity()
        safe_print(f"Connected to {config['uri']} / database={config['database']}")

        query = args.query or DEFAULT_QUERY
        records, summary = run_query(driver, query, config["database"])
        print_records(records)
        safe_print(
            "Query returned {count} records in {time} ms.".format(
                count=len(records),
                time=summary.result_available_after,
            )
        )

        if args.query is None:
            safe_print("\nSample relationships:")
            sample_records, sample_summary = run_query(
                driver,
                SAMPLE_QUERY,
                config["database"],
                limit=args.limit,
            )
            print_records(sample_records)
            safe_print(
                "Sample query returned {count} records in {time} ms.".format(
                    count=len(sample_records),
                    time=sample_summary.result_available_after,
                )
            )


if __name__ == "__main__":
    main()

