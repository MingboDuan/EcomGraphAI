import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained" / "easyoc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ocr_demo"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "ocr_demo"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def setup_logger(log_dir: Path) -> logging.Logger:
    """创建同时输出到终端和日志文件的 OCR 演示日志器。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("ocr_demo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("log_path: %s", log_path)
    return logger


def natural_key(base_dir: Path, path: Path) -> List[Any]:
    """按自然顺序排序图片路径，避免 10.jpg 排在 2.jpg 前面。"""
    parts = []
    for part in path.relative_to(base_dir).parts:
        stem = Path(part).stem
        parts.append(int(stem) if stem.isdigit() else stem)
    return parts


def iter_image_paths(image_dir: Path) -> Iterable[Path]:
    """递归查找目录下支持 OCR 的图片文件。"""
    image_paths = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    for path in sorted(image_paths, key=lambda item: natural_key(image_dir, item)):
        yield path


def collect_demo_images(image_path: Optional[Path], image_dir: Path, limit: int) -> List[Path]:
    """收集演示图片：优先使用指定单张图片，否则取目录下前 N 张图片。"""
    if image_path is not None:
        return [image_path.resolve()]
    return [path.resolve() for path in list(iter_image_paths(image_dir.resolve()))[:limit]]


def build_reader(model_dir: Path, use_gpu: Optional[bool], logger: logging.Logger):
    """构建 EasyOCR 识别器；默认检测 CUDA，可用时自动启用 GPU。"""
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


def normalize_bbox(bbox) -> List[List[float]]:
    """将 EasyOCR 返回的文本框坐标转为可 JSON 序列化的浮点数。"""
    return [[float(point[0]), float(point[1])] for point in bbox]


def recognize_image(reader, image_path: Path) -> Dict[str, Any]:
    """识别单张图片，保留文本、置信度、文本框坐标和耗时。"""
    start_time = time.perf_counter()
    results = reader.readtext(str(image_path))
    elapsed = time.perf_counter() - start_time

    items = []
    for bbox, text, confidence in results:
        text = text.strip()
        if not text:
            continue
        items.append(
            {
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": normalize_bbox(bbox),
            }
        )

    return {
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "ocr_text": " ".join(item["text"] for item in items),
        "text_blocks": items,
        "elapsed_seconds": round(elapsed, 4),
    }


def write_json(records: List[Dict[str, Any]], output_dir: Path) -> Path:
    """将 OCR 演示结果保存到 output/ocr_demo 目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ocr_demo_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="本地电商图片 EasyOCR 文字识别简单演示。")
    parser.add_argument("--image_path", type=Path, default=None, help="指定单张图片进行 OCR。")
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR, help="演示图片目录。")
    parser.add_argument("--limit", type=int, default=100, help="未指定 image_path 时，默认识别前 N 张图片。")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="EasyOCR 模型目录。")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="JSON 结果保存目录。")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR, help="日志保存目录。")

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--gpu", action="store_true", help="强制使用 GPU。")
    device_group.add_argument("--cpu", action="store_true", help="强制使用 CPU。")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_dir.resolve())
    use_gpu = True if args.gpu else False if args.cpu else None

    image_paths = collect_demo_images(args.image_path, args.image_dir, args.limit)
    logger.info("image_count: %s", len(image_paths))
    if not image_paths:
        logger.warning("未找到可识别的图片。")
        return

    reader = build_reader(args.model_dir.resolve(), use_gpu, logger)
    records = []

    for index, image_path in enumerate(image_paths, start=1):
        record = recognize_image(reader, image_path)
        records.append(record)
        logger.info(
            "[%s/%s] %s text_blocks=%s seconds=%.4f",
            index,
            len(image_paths),
            record["image_path"],
            len(record["text_blocks"]),
            record["elapsed_seconds"],
        )
        logger.info("ocr_text: %s", record["ocr_text"])

    output_path = write_json(records, args.output_dir.resolve())
    logger.info("output_path: %s", output_path)


if __name__ == "__main__":
    main()

