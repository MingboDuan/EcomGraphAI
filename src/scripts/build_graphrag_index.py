import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graphrag.graph_index import build_graph_indexes
from graphrag.logger import setup_logger


def main():
    logger = setup_logger("graphrag_index")
    stats = build_graph_indexes(logger=logger)
    logger.info("build_graphrag_index_done: %s", stats)
    print(stats)


if __name__ == "__main__":
    main()

