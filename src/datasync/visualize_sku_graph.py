import argparse
import html
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "sku_graph_visualization"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "sku_graph_visualization"
DEFAULT_DETAIL_ENTITY_PATH = PROJECT_ROOT / "output" / "sku_image_detail_graph_sync" / "sku_image_detail_entities.jsonl"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def setup_logger(log_dir: Path) -> logging.Logger:
    """创建可同时输出到终端和文件的日志器。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("sku_graph_visualization")
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


def load_detail_entity_keys(entity_path: Path, sku_id: int) -> set[Tuple[str, str]]:
    """读取商品详情抽取结果，用于判断哪些 Attr 来自商品详情图谱补充。"""
    detail_keys = set()
    if not entity_path.exists():
        return detail_keys

    with entity_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("sku_id", -1)) != sku_id:
                continue
            for entity in record.get("entities", []):
                detail_keys.add((entity.get("attr_name", ""), entity.get("attr_value", "")))
    return detail_keys


def query_sku_graph(driver, sku_id: int) -> Dict[str, Any]:
    """查询指定 SKU 的业务链路和属性节点。"""
    base_records, _, _ = driver.execute_query(
        """
        MATCH (sku:SKU {sku_id: $sku_id})
        OPTIONAL MATCH (sku)-[:Belong]->(spu:SPU)
        OPTIONAL MATCH (spu)-[:Belong]->(cate3:Category3)
        OPTIONAL MATCH (cate3)-[:Belong]->(cate2:Category2)
        OPTIONAL MATCH (cate2)-[:Belong]->(cate1:Category1)
        OPTIONAL MATCH (spu)-[:Belong]->(tm:Trademark)
        RETURN
            sku AS sku,
            spu AS spu,
            cate3 AS cate3,
            cate2 AS cate2,
            cate1 AS cate1,
            tm AS tm
        """,
        sku_id=sku_id,
        database_=NEO4J_DATABASE,
    )
    attr_records, _, _ = driver.execute_query(
        """
        MATCH (sku:SKU {sku_id: $sku_id})-[:Have]->(attr:Attr)
        RETURN attr
        ORDER BY attr.attr_name, attr.attr_value
        """,
        sku_id=sku_id,
        database_=NEO4J_DATABASE,
    )

    base = base_records[0].data() if base_records else {}
    attrs = [record["attr"] for record in attr_records]
    return {"base": base, "attrs": attrs}


def node_properties(node) -> Dict[str, Any]:
    """将 Neo4j Node 转为普通 dict，便于渲染。"""
    return dict(node) if node is not None else {}


def add_node(nodes: List[Dict[str, Any]], node_id: str, label: str, text: str, x: int, y: int, group: str) -> None:
    """添加一个可视化节点。"""
    if not text:
        return
    nodes.append({"id": node_id, "label": label, "text": text, "x": x, "y": y, "group": group})


def add_edge(edges: List[Dict[str, str]], source: str, target: str, label: str, group: str) -> None:
    """添加一条可视化关系。"""
    edges.append({"source": source, "target": target, "label": label, "group": group})


def build_visual_graph(sku_id: int, graph_data: Dict[str, Any], detail_keys: set[Tuple[str, str]]) -> Dict[str, Any]:
    """组织前端 SVG 所需的节点、边和统计信息。"""
    base = graph_data["base"]
    attrs = [node_properties(attr) for attr in graph_data["attrs"]]
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []

    sku = node_properties(base.get("sku"))
    spu = node_properties(base.get("spu"))
    cate3 = node_properties(base.get("cate3"))
    cate2 = node_properties(base.get("cate2"))
    cate1 = node_properties(base.get("cate1"))
    tm = node_properties(base.get("tm"))

    add_node(nodes, "sku", "SKU", f"{sku.get('sku_name', '')}\nsku_id={sku_id}", 670, 380, "sku")
    add_node(nodes, "spu", "SPU", spu.get("spu_name", ""), 420, 190, "business")
    add_node(nodes, "cate3", "Category3", cate3.get("category3_name", ""), 170, 110, "business")
    add_node(nodes, "cate2", "Category2", cate2.get("category2_name", ""), 170, 270, "business")
    add_node(nodes, "cate1", "Category1", cate1.get("category1_name", ""), 170, 430, "business")
    add_node(nodes, "tm", "Trademark", tm.get("trademark_name", ""), 420, 560, "business")

    if spu:
        add_edge(edges, "sku", "spu", "Belong", "business")
    if cate3:
        add_edge(edges, "spu", "cate3", "Belong", "business")
    if cate2:
        add_edge(edges, "cate3", "cate2", "Belong", "business")
    if cate1:
        add_edge(edges, "cate2", "cate1", "Belong", "business")
    if tm:
        add_edge(edges, "spu", "tm", "Belong", "business")

    detail_count = 0
    original_attr_count = 0
    for index, attr in enumerate(attrs):
        key = (attr.get("attr_name", ""), attr.get("attr_value", ""))
        is_detail = key in detail_keys
        group = "detail" if is_detail else "attr"
        if is_detail:
            detail_count += 1
        else:
            original_attr_count += 1

        x = 1060 if is_detail else 690
        y = 130 + (detail_count - 1) * 145 if is_detail else 650 + original_attr_count * 115
        node_id = f"attr_{index}"
        add_node(nodes, node_id, "Attr", f"{attr.get('attr_name')}: {attr.get('attr_value')}", x, y, group)
        add_edge(edges, "sku", node_id, "Have", group)

    return {
        "sku_id": sku_id,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "business_nodes": len([node for node in nodes if node["group"] == "business"]),
            "detail_attrs": detail_count,
            "other_attrs": original_attr_count,
        },
    }


def escape_text(value: str) -> str:
    """转义 HTML 文本。"""
    return html.escape(str(value or ""))


def wrap_text(value: str, max_chars: int) -> List[str]:
    """按字符数拆行，完整保留文本内容，不做截断。"""
    wrapped_lines: List[str] = []
    for raw_line in str(value or "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        while len(line) > max_chars:
            wrapped_lines.append(line[:max_chars])
            line = line[max_chars:]
        wrapped_lines.append(line)
    return wrapped_lines or [""]


def render_svg_node(node: Dict[str, Any]) -> str:
    """渲染单个 SVG 节点。"""
    styles = {
        "sku": {
            "fill": "#1F4E79",
            "stroke": "#15395B",
            "text": "#FFFFFF",
            "width": 360,
            "max_chars": 24,
            "shadow": "#9BB7D4",
        },
        "business": {
            "fill": "#E8F2FF",
            "stroke": "#5B8CCB",
            "text": "#14345A",
            "width": 300,
            "max_chars": 20,
            "shadow": "#C7D9EE",
        },
        "detail": {
            "fill": "#E7F8ED",
            "stroke": "#22A65A",
            "text": "#14532D",
            "width": 320,
            "max_chars": 22,
            "shadow": "#C8EBD4",
        },
        "attr": {
            "fill": "#F1F5F9",
            "stroke": "#9CA3AF",
            "text": "#374151",
            "width": 300,
            "max_chars": 20,
            "shadow": "#D7DEE8",
        },
    }
    style = styles[node["group"]]
    lines = wrap_text(node["text"], style["max_chars"])
    width = style["width"]
    height = max(86, 48 + len(lines) * 18)
    x = node["x"] - width // 2
    y = node["y"] - height // 2

    line_spans = []
    for line_index, line in enumerate(lines):
        line_spans.append(
            f'<tspan x="{node["x"]}" dy="{0 if line_index == 0 else 18}">{escape_text(line)}</tspan>'
        )

    return f"""
    <g class="node node-{node['group']}">
      <rect x="{x + 5}" y="{y + 6}" width="{width}" height="{height}" rx="10" fill="{style['shadow']}" opacity="0.35"></rect>
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="{style['fill']}" stroke="{style['stroke']}" stroke-width="1.6"></rect>
      <text x="{node['x']}" y="{y + 24}" text-anchor="middle" font-size="12" font-weight="700" fill="{style['text']}">{escape_text(node['label'])}</text>
      <text x="{node['x']}" y="{y + 48}" text-anchor="middle" font-size="13" fill="{style['text']}">{''.join(line_spans)}</text>
    </g>
    """


def render_svg_edge(edge: Dict[str, str], node_map: Dict[str, Dict[str, Any]]) -> str:
    """渲染单条 SVG 边。"""
    source = node_map[edge["source"]]
    target = node_map[edge["target"]]
    colors = {"business": "#4F83CC", "detail": "#16A34A", "attr": "#6B7280"}
    color = colors.get(edge["group"], "#6B7280")
    mid_x = (source["x"] + target["x"]) / 2
    mid_y = (source["y"] + target["y"]) / 2
    return f"""
    <g class="edge edge-{edge['group']}">
      <line x1="{source['x']}" y1="{source['y']}" x2="{target['x']}" y2="{target['y']}" stroke="{color}" stroke-width="2.2" marker-end="url(#{edge['group']}-arrow)"></line>
      <text x="{mid_x}" y="{mid_y - 8}" text-anchor="middle" font-size="12" fill="{color}">{escape_text(edge['label'])}</text>
    </g>
    """


def render_html(graph: Dict[str, Any], output_path: Path) -> None:
    """生成包含 SVG 图谱的 HTML 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_map = {node["id"]: node for node in graph["nodes"]}
    edges_svg = "\n".join(render_svg_edge(edge, node_map) for edge in graph["edges"])
    nodes_svg = "\n".join(render_svg_node(node) for node in graph["nodes"])

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>SKU {graph['sku_id']} 图谱可视化</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: #111827;
      background: #EEF3F8;
    }}
    header {{
      padding: 24px 34px 12px;
      background: #FFFFFF;
      border-bottom: 1px solid #DDE6F0;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; }}
    .summary {{ color: #4B5563; font-size: 14px; line-height: 1.7; }}
    .legend {{
      display: flex;
      gap: 18px;
      padding: 14px 34px 20px;
      align-items: center;
      flex-wrap: wrap;
      background: #FFFFFF;
    }}
    .legend-item {{
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 14px;
      color: #374151;
    }}
    .swatch {{ width: 18px; height: 18px; border-radius: 4px; border: 1px solid #6B7280; }}
    .canvas {{
      margin: 24px 28px 32px;
      background: #FFFFFF;
      border: 1px solid #DDE6F0;
      border-radius: 8px;
      overflow: auto;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }}
    svg {{ min-width: 1380px; min-height: 920px; display: block; }}
    .lane-label {{ font-size: 13px; font-weight: 700; fill: #64748B; letter-spacing: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>SKU {graph['sku_id']} 知识图谱可视化</h1>
    <div class="summary">
      业务节点 {graph['stats']['business_nodes']} 个，商品详情新增属性 {graph['stats']['detail_attrs']} 个，其他属性 {graph['stats']['other_attrs']} 个。
    </div>
  </header>
  <section class="legend">
    <div class="legend-item"><span class="swatch" style="background:#2F5D9F"></span>当前 SKU</div>
    <div class="legend-item"><span class="swatch" style="background:#D7E8FF"></span>业务数据库原有节点关系</div>
    <div class="legend-item"><span class="swatch" style="background:#DDF7E5"></span>商品详情新增节点关系</div>
    <div class="legend-item"><span class="swatch" style="background:#ECEFF3"></span>其他已有属性</div>
  </section>
  <main class="canvas">
    <svg viewBox="0 0 1380 920" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="business-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#4F83CC"></path>
        </marker>
        <marker id="detail-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#16A34A"></path>
        </marker>
        <marker id="attr-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#6B7280"></path>
        </marker>
      </defs>
      <rect x="34" y="46" width="500" height="760" rx="8" fill="#F8FBFF" stroke="#E2E8F0"></rect>
      <rect x="560" y="46" width="280" height="760" rx="8" fill="#F8FAFC" stroke="#E2E8F0"></rect>
      <rect x="875" y="46" width="430" height="760" rx="8" fill="#F6FEF8" stroke="#D9F4E3"></rect>
      <text x="60" y="78" class="lane-label">业务数据库原有链路</text>
      <text x="590" y="78" class="lane-label">当前 SKU</text>
      <text x="905" y="78" class="lane-label">商品详情新增属性</text>
      {edges_svg}
      {nodes_svg}
    </svg>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="可视化指定 SKU 的业务图谱与商品详情新增图谱。")
    parser.add_argument("--sku_id", type=int, default=2)
    parser.add_argument("--detail_entity_path", type=Path, default=DEFAULT_DETAIL_ENTITY_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.log_dir.resolve())
    detail_keys = load_detail_entity_keys(args.detail_entity_path.resolve(), args.sku_id)
    logger.info("detail_entity_count_for_sku_%s: %s", args.sku_id, len(detail_keys))

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        graph_data = query_sku_graph(driver, args.sku_id)

    graph = build_visual_graph(args.sku_id, graph_data, detail_keys)
    output_path = args.output_dir.resolve() / f"sku_{args.sku_id}_graph.html"
    render_html(graph, output_path)

    logger.info("nodes: %s", len(graph["nodes"]))
    logger.info("edges: %s", len(graph["edges"]))
    logger.info("output_path: %s", output_path)


if __name__ == "__main__":
    main()

