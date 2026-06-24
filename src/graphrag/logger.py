import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from . import config


def setup_logger(name: str = "graphrag", log_dir: Path | None = None) -> logging.Logger:
    log_dir = log_dir or config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("log_path: %s", log_path)
    return logger


def save_trace(trace_id: str, trace: Dict[str, Any]) -> Path:
    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = config.TRACE_DIR / f"{trace_id}.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)
    return trace_path


def to_json(value: Any) -> str:
    """把中间结果转成中文可读 JSON，方便直接查看日志。"""
    return json.dumps(value, ensure_ascii=False, default=str)

