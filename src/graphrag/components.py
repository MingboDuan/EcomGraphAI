import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from neo4j.exceptions import CypherSyntaxError
from transformers import AutoModel, AutoTokenizer

from . import config
from .logger import to_json


SPEC_PATTERN = re.compile(r"([A-Za-z]+[-+]?\d+[A-Za-z0-9+-]*|\d+(\.\d+)?\s?(GB|TB|mAh|Hz|K|英寸|寸|W|万|G))", re.I)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9\u4e00-\u9fa5]+")
FORBIDDEN_CYPHER = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP|REMOVE|LOAD\s+CSV|CALL\s+dbms|CALL\s+apoc)\b", re.I)
USER_BEHAVIOR_WORDS = (
    "\u6536\u85cf",
    "\u6d4f\u89c8",
    "\u70b9\u51fb",
    "\u770b\u8fc7",
    "\u5386\u53f2",
    "\u884c\u4e3a",
    "\u5173\u6ce8",
    "\u611f\u5174\u8da3",
)
PRODUCT_FILTER_WORDS = (
    "\u624b\u673a",
    "\u7535\u8111",
    "\u7535\u89c6",
    "\u53e3\u7ea2",
    "\u7b14\u8bb0\u672c",
    "\u5e73\u677f",
    "\u8033\u673a",
    "\u51b0\u7bb1",
    "\u6d17\u8863\u673a",
    "\u7a7a\u8c03",
    "\u7cae\u6cb9",
    "\u82b1\u751f\u6cb9",
    "\u5316\u5986",
)
PRODUCT_SPEC_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:GB|G|TB|mAh|Hz|K|\u82f1\u5bf8|\u5bf8|W)", re.I)


def is_user_behavior_query(query: str) -> bool:
    return any(word in (query or "") for word in USER_BEHAVIOR_WORDS)


class LLMClient:
    """大模型调用封装，优先从项目环境变量读取 API Key，再读取 rag_0319/.env。"""

    def __init__(self, logger=None):
        self.logger = logger
        self.llm: Any = None
        self._load_env_file()
        self._try_init()

    @property
    def available(self) -> bool:
        return self.llm is not None

    def _load_env_file(self) -> None:
        env_path = (
            config.PROJECT_ROOT.parent.parent
            / "Ch14_RAG_GraphRAG"
            / "rag_0319"
            / ".env"
        )
        if not env_path.exists() or os.getenv("TONGYI_API_KEY"):
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def _try_init(self) -> None:
        api_key = os.getenv("TONGYI_API_KEY") or config.TONGYI_API_KEY
        if not api_key:
            if self.logger:
                self.logger.warning("llm_disabled: missing TONGYI_API_KEY")
            return
        self.llm = {"api_key": api_key, "base_url": self._normalize_base_url(config.LLM_BASE_URL)}
        if self.logger:
            self.logger.info(
                "llm_enabled: provider=%s model=%s base_url=%s",
                config.LLM_PROVIDER,
                config.LLM_MODEL,
                self.llm["base_url"],
            )

    def _normalize_base_url(self, url: str) -> str:
        """兼容只配置到 /compatible-mode/v1 的情况，自动补全 chat completions 路径。"""
        cleaned = (url or "").strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        if cleaned.endswith("/compatible-mode/v1") or cleaned.endswith("/v1"):
            return cleaned + "/chat/completions"
        return cleaned

    def invoke(self, prompt: str, task_name: str = "大模型调用", log_response: bool = True) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not available.")
        if self.logger:
            self.logger.info("大模型调用开始: 任务=%s 模型=%s prompt长度=%s", task_name, config.LLM_MODEL, len(prompt))
        api_key = self.llm["api_key"]
        base_url = self.llm["base_url"]
        payload = json.dumps(
            {
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            if self.logger:
                self.logger.error("大模型调用失败: 任务=%s 状态码=%s 错误=%s", task_name, exc.code, detail)
            raise RuntimeError(f"LLM HTTPError {exc.code}: {detail}") from exc
        content = data["choices"][0]["message"]["content"]
        if self.logger:
            if log_response:
                self.logger.info("大模型调用成功: 任务=%s 返回=%s", task_name, content)
            else:
                self.logger.info("大模型调用成功: 任务=%s 返回内容已省略", task_name)
        return content


class BgeEmbedding:
    """本地 bge-base-zh-v1.5 嵌入模型封装，不依赖 sentence_transformers。"""

    def __init__(self, model_dir=None, device: str | None = None):
        self.model_dir = str(model_dir or config.BGE_MODEL_DIR)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModel.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                pooled = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.extend(pooled.cpu().tolist())
        return [list(map(float, item)) for item in embeddings]

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]

    def _mean_pooling(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class QueryCorrector:
    """查询纠错模块；默认保守关闭模型纠错，避免误改商品型号和规格参数。"""

    def __init__(self, enable_model: bool = False, logger=None):
        self.logger = logger
        self.predictor = None
        if enable_model:
            self._try_load_model()

    def _try_load_model(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer

            if str(config.SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(config.SRC_ROOT))
            from models.spell_check_t5 import SpellCheckT5
            from runner.Predictor import SpellCheckT5Predictor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SpellCheckT5(pretrained_path=config.SPELL_MODEL_DIR)
            model.load_state_dict(torch.load(config.SPELL_CHECKPOINT, map_location=device))
            tokenizer = AutoTokenizer.from_pretrained(config.SPELL_MODEL_DIR)
            self.predictor = SpellCheckT5Predictor(model, tokenizer, device)
        except Exception as exc:
            if self.logger:
                self.logger.warning("query_corrector_model_disabled: %s", exc)

    def correct(self, query: str) -> Tuple[str, bool]:
        query = (query or "").strip()
        if not query or self.predictor is None:
            if self.logger:
                self.logger.info("查询纠错: 未启用模型纠错，保持原始问题=%s", query)
            return query, False
        protected = [item[0] for item in SPEC_PATTERN.findall(query)]
        corrected = self.predictor.predict(query)
        for token in protected:
            if token and token not in corrected:
                if self.logger:
                    self.logger.warning("查询纠错: 保护关键参数，保留原始问题，关键参数=%s", token)
                return query, False
        if self.logger:
            self.logger.info("查询纠错结果: 原始问题=%s 纠错后=%s 是否变化=%s", query, corrected, corrected != query)
        return corrected, corrected != query


class ContextualQueryRewriter:
    """上下文问题改写模块，将多轮追问补全为可独立检索的完整问题。"""

    def __init__(self, llm: LLMClient, logger=None):
        self.llm = llm
        self.logger = logger

    def rewrite(self, query: str, history: str = "", user_id: int | None = None) -> Tuple[str, bool, str]:
        query = (query or "").strip()
        if not history.strip() or (not self._looks_contextual(query) and not self._looks_spec_followup(query)):
            if self.logger:
                self.logger.info("上下文问题改写: 无需改写，检索问题=%s", query)
            return query, False, "无历史上下文或当前问题已经完整"
        rule_rewrite = self._rule_rewrite(query, history, user_id)
        if rule_rewrite:
            rewritten, reason = rule_rewrite
            if self.logger:
                self.logger.info(
                    "上下文问题改写规则结果: 原问题=%s 改写后=%s reason=%s",
                    query,
                    rewritten,
                    reason,
                )
            return rewritten, rewritten != query, reason
        if not self.llm.available:
            if self.logger:
                self.logger.info("上下文问题改写: 大模型不可用，保持原问题=%s", query)
            return query, False, "大模型不可用"

        prompt = f"""
你是电商 GraphRAG 的上下文问题改写器。请结合历史对话，把当前用户追问改写成一个可以独立用于实体抽取、混合检索和 Cypher 生成的完整问题。

严格要求:
1. 只补全省略的信息，不要编造历史中没有出现的新品牌、品类、属性或用户行为。
2. 如果当前问题已经完整，保持原问题。
3. 必须保留上一轮尚未被用户否定的核心约束，例如品牌、品类、功效、用户行为关系、用户ID。
4. 如果用户说“那兰蔻的有哪些”，需要结合上文补全成“兰蔻的某类商品/某些属性条件有哪些”。
5. 如果用户说“它们/这些/里面/这个”，需要把指代对象替换为历史对话中的具体对象。
6. 用户行为追问必须保留行为范围。例如上文是“用户51收藏过哪些商品”，追问“里面有没有电视”应改写为“用户51收藏过的商品里面有没有电视”。
7. 不要把回答里的所有商品名都塞进改写问题，保留可检索的品牌、品类、属性、行为约束即可。

返回 JSON，不要输出解释文字:
{{"standalone_query":"补全后的完整问题","changed":true,"reason":"中文简要说明"}}

历史对话:
{history}

当前 user_id: {user_id}
当前问题:
{query}
"""
        try:
            raw_output = self.llm.invoke(prompt, task_name="上下文问题改写", log_response=True)
            data = parse_json_object(raw_output)
            rewritten = str(data.get("standalone_query") or query).strip()
            changed = bool(data.get("changed")) and rewritten and rewritten != query
            reason = str(data.get("reason") or "").strip()
            if not rewritten:
                rewritten = query
                changed = False
            if self.logger:
                self.logger.info(
                    "上下文问题改写结果: 原问题=%s 改写后=%s 是否变化=%s reason=%s",
                    query,
                    rewritten,
                    changed,
                    reason,
                )
            return rewritten, changed, reason
        except Exception as exc:
            if self.logger:
                self.logger.warning("上下文问题改写失败，保持原问题: %s", exc)
            return query, False, f"改写失败: {exc}"

    def _looks_contextual(self, query: str) -> bool:
        text = query or ""
        contextual_words = (
            "那",
            "它",
            "它们",
            "这些",
            "这款",
            "这个",
            "里面",
            "其中",
            "刚才",
            "上面",
            "上一",
            "还有",
            "呢",
        )
        return any(word in text for word in contextual_words)

    def _looks_spec_followup(self, query: str) -> bool:
        """Treat short spec-only questions as contextual follow-ups."""
        text = query or ""
        return bool(PRODUCT_SPEC_PATTERN.search(text)) and len(text) <= 20

    def _rule_rewrite(self, query: str, history: str, user_id: int | None) -> Tuple[str, str] | None:
        """Use deterministic rewrites for high-frequency follow-ups before asking the LLM."""
        text = query or ""
        hist = history or ""
        has_lipstick = "\u53e3\u7ea2" in hist
        has_lancome = "\u5170\u853b" in hist or "\u5170\u853b" in text
        has_moisturizing = "\u4fdd\u6e7f" in hist
        has_hydrating = "\u8865\u6c34" in hist

        def cosmetic_effect_phrase() -> str:
            effects = []
            if has_moisturizing:
                effects.append("\u4fdd\u6e7f")
            if has_hydrating:
                effects.append("\u8865\u6c34")
            if not effects:
                return ""
            return "\u6216\u8005".join(effects)

        if "\u5170\u853b" in text and any(word in text for word in ["\u54ea\u4e9b", "\u6709\u54ea\u4e9b"]):
            effect_phrase = cosmetic_effect_phrase()
            if has_lipstick and effect_phrase:
                return f"\u5170\u853b\u5e26{effect_phrase}\u529f\u80fd\u7684\u53e3\u7ea2\u6709\u54ea\u4e9b\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u54c1\u724c\u3001\u53e3\u7ea2\u54c1\u7c7b\u548c\u5df2\u51fa\u73b0\u7684\u529f\u6548\u7ea6\u675f"
            if has_lipstick:
                return "\u5170\u853b\u53e3\u7ea2\u6709\u54ea\u4e9b\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u53e3\u7ea2\u54c1\u7c7b"

        if any(word in text for word in ["\u5b83\u4eec", "\u8fd9\u4e9b", "\u5b83"]):
            effect_phrase = cosmetic_effect_phrase()
            if "\u529f\u6548" in text:
                if has_lancome and has_lipstick and effect_phrase:
                    return f"\u5170\u853b\u5e26{effect_phrase}\u529f\u80fd\u7684\u53e3\u7ea2\u90fd\u6709\u4ec0\u4e48\u529f\u6548\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u5170\u853b\u3001\u53e3\u7ea2\u548c\u5df2\u51fa\u73b0\u7684\u529f\u6548\u7ea6\u675f"
                if has_lipstick and effect_phrase:
                    return f"\u5e26{effect_phrase}\u529f\u80fd\u7684\u53e3\u7ea2\u90fd\u6709\u4ec0\u4e48\u529f\u6548\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u53e3\u7ea2\u548c\u5df2\u51fa\u73b0\u7684\u529f\u6548\u7ea6\u675f"
            if "\u989c\u8272" in text:
                if has_lancome and has_lipstick and effect_phrase:
                    return f"\u5170\u853b\u5e26{effect_phrase}\u529f\u80fd\u7684\u53e3\u7ea2\u90fd\u662f\u4ec0\u4e48\u989c\u8272\u7684\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u5170\u853b\u3001\u53e3\u7ea2\u548c\u5df2\u51fa\u73b0\u7684\u529f\u6548\u7ea6\u675f\uff0c\u989c\u8272\u4ec5\u4f5c\u4e3a\u8fd4\u56de\u5b57\u6bb5"
                if has_lipstick and effect_phrase:
                    return f"\u5e26{effect_phrase}\u529f\u80fd\u7684\u53e3\u7ea2\u90fd\u662f\u4ec0\u4e48\u989c\u8272\u7684\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u53e3\u7ea2\u548c\u5df2\u51fa\u73b0\u7684\u529f\u6548\u7ea6\u675f\uff0c\u989c\u8272\u4ec5\u4f5c\u4e3a\u8fd4\u56de\u5b57\u6bb5"

        spec_match = PRODUCT_SPEC_PATTERN.search(text)
        if spec_match and "\u7d22\u5c3c" in hist and "\u5e73\u677f\u7535\u89c6" in hist:
            spec = spec_match.group(0).replace(" ", "")
            return f"\u7d22\u5c3c\u5e73\u677f\u7535\u89c6\u6709\u6ca1\u6709{spec}\u7684\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u7d22\u5c3c\u54c1\u724c\u548c\u5e73\u677f\u7535\u89c6\u54c1\u7c7b"

        if "\u8fd9\u4e2a\u54c1\u724c" in text and "\u5c3a\u5bf8" in text and "\u7d22\u5c3c" in hist and "\u5e73\u677f\u7535\u89c6" in hist:
            return "\u7d22\u5c3c\u5e73\u677f\u7535\u89c6\u8fd8\u6709\u5176\u5b83\u5c3a\u5bf8\u5417\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u7d22\u5c3c\u54c1\u724c\u548c\u5e73\u677f\u7535\u89c6\u54c1\u7c7b"

        user_match = re.search(r"\u7528\u6237\s*(\d+)", hist)
        history_user_id = user_match.group(1) if user_match else str(user_id) if user_id is not None else ""
        if "\u91cc\u9762" in text and "\u7535\u89c6" in text and "\u6536\u85cf" in hist and history_user_id:
            return f"\u7528\u6237{history_user_id}\u6536\u85cf\u8fc7\u7684\u5546\u54c1\u91cc\u9762\u6709\u6ca1\u6709\u7535\u89c6\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u7528\u6237\u6536\u85cf\u8303\u56f4"
        if "\u8fd9\u4e2a\u7535\u89c6" in text and "\u6536\u85cf" in hist and history_user_id:
            return f"\u7528\u6237{history_user_id}\u6536\u85cf\u8fc7\u7684\u7535\u89c6\u662f\u4ec0\u4e48\u54c1\u724c\u548c\u5c3a\u5bf8\uff1f", "\u6839\u636e\u4e0a\u6587\u8865\u5168\u7528\u6237\u6536\u85cf\u7535\u89c6\u8303\u56f4"
        return None


class IntentClassifier:
    """意图识别模块，优先通过大模型输出结构化意图，失败时使用规则兜底。"""

    def __init__(self, llm: LLMClient, logger=None):
        self.llm = llm
        self.logger = logger

    def classify(self, query: str, user_id: int | None = None, history: str = "") -> Dict[str, Any]:
        if is_user_behavior_query(query) and user_id is not None:
            result = {
                "intent": "user_interest_query",
                "reason": "query contains user behavior words and has user_id",
                "has_user_id": True,
            }
            if self.logger:
                self.logger.info("意图识别规则优先结果: %s", to_json(result))
            return result

        if self.llm.available:
            prompt = f"""
你是电商客服 Graph RAG 系统的意图识别器。
请根据用户问题和历史对话判断意图，只输出 JSON，不要输出解释。

可选 intent:
- product_search: 商品查找
- product_compare: 商品对比
- recommendation: 个性化推荐
- attribute_query: 商品属性/参数查询
- brand_query: 品牌查询
- category_query: 品类查询
- user_interest_query: 用户历史行为/兴趣查询
- unknown: 无法判断

输出格式:
{{"intent":"product_search","reason":"..."}}

历史对话:
{history}

user_id: {user_id}
用户问题: {query}
"""
            try:
                raw_output = self.llm.invoke(prompt, task_name="意图识别", log_response=True)
                data = parse_json_object(raw_output)
                data.pop("need_user_context", None)
                data["has_user_id"] = user_id is not None
                if self.logger:
                    self.logger.info("意图识别结果: %s", to_json(data))
                return data
            except Exception as exc:
                if self.logger:
                    self.logger.warning("意图识别失败，使用规则兜底: %s", exc)
        result = self._rule_classify(query, user_id)
        if self.logger:
            self.logger.info("意图识别规则兜底结果: %s", to_json(result))
        return result

    def _rule_classify(self, query: str, user_id: int | None) -> Dict[str, Any]:
        text = query or ""
        if any(word in text for word in ("推荐", "适合我", "喜欢", "感兴趣")):
            intent = "recommendation"
        elif any(word in text for word in ("对比", "区别", "哪个好")):
            intent = "product_compare"
        elif any(word in text for word in ("收藏", "浏览", "点击", "看过", "历史")):
            intent = "user_interest_query"
        elif any(word in text for word in ("属性", "参数", "配置", "多大", "尺寸", "内存", "电池")):
            intent = "attribute_query"
        elif any(word in text for word in ("品牌", "华为", "小米", "苹果", "Redmi", "OPPO", "vivo")):
            intent = "brand_query"
        elif any(word in text for word in ("手机", "笔记本", "电视", "耳机", "冰箱", "洗衣机", "咖啡机")):
            intent = "product_search"
        else:
            intent = "product_search"
        return {"intent": intent, "has_user_id": user_id is not None}


class EntityExtractor:
    """实体抽取模块，规则抽取基于图谱真实词表，UIE 抽取基于商品详情侧 schema。"""

    def __init__(self, driver=None, logger=None, enable_uie: bool = True):
        self.driver = driver
        self.logger = logger
        self.predictor = None
        self.graph_terms = self._load_graph_terms()
        if enable_uie:
            self._try_load_uie()

    def _try_load_uie(self) -> None:
        try:
            uie_path = config.EXTERNAL_LIB_DIR / "uie_pytorch"
            if str(uie_path) not in sys.path:
                sys.path.insert(0, str(uie_path))
            from uie_predictor import UIEPredictor

            self.predictor = UIEPredictor(
                model="uie-base",
                task_path=str(config.UIE_MODEL_DIR),
                schema=config.UIE_SCHEMA,
                device="gpu",
                batch_size=8,
                max_seq_len=512,
            )
        except Exception as exc:
            if self.logger:
                self.logger.warning("uie_entity_extractor_disabled: %s", exc)

    def extract(self, query: str, user_id: int | None = None) -> List[Dict[str, Any]]:
        entities = self._rule_extract(query, user_id)
        if self.logger:
            self.logger.info("实体抽取规则结果: %s", to_json(entities))
        if self._is_pure_user_behavior_query(query):
            entities = self._deduplicate(entities)
            if self.logger:
                self.logger.info("entity_extract: pure user behavior query, skip product attrs: %s", to_json(entities))
                self.logger.info("瀹炰綋鎶藉彇鏈€缁堢粨鏋? %s", to_json(entities))
            return entities
        if self.predictor is not None:
            try:
                result = self.predictor(query)
                item = result[0] if isinstance(result, list) and result else {}
                if self.logger:
                    self.logger.info("实体抽取UIE原始结果: %s", to_json(item))
                entities.extend(self._parse_uie_result(item))
            except Exception as exc:
                if self.logger:
                    self.logger.warning("实体抽取UIE失败: %s", exc)
        entities = self._deduplicate(entities)
        if self.logger:
            self.logger.info("实体抽取最终结果: %s", to_json(entities))
        return entities

    def _rule_extract(self, query: str, user_id: int | None) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if user_id is not None:
            result.append({"label": "User", "text": str(user_id), "score": 1.0})
        if self._is_pure_user_behavior_query(query):
            return result

        result.extend(self._match_graph_named_terms("Category3", query))
        result.extend(self._match_generic_category_terms(query))
        result.extend(self._match_graph_named_terms("Trademark", query))
        result.extend(self._match_graph_attrs(query))
        return result

    def _is_pure_user_behavior_query(self, query: str) -> bool:
        """Only user behavior is requested, without product/category/brand/spec filters."""
        text = query or ""
        if not is_user_behavior_query(text):
            return False
        if any(word in text for word in PRODUCT_FILTER_WORDS):
            return False
        if PRODUCT_SPEC_PATTERN.search(text):
            return False
        for label in ("Category3", "Trademark"):
            for term in self.graph_terms.get(label, []):
                term_text = str(term.get("text") or "")
                if term_text and term_text in text:
                    return False
        for attr_name in config.UIE_SCHEMA:
            if attr_name and attr_name not in {"绫诲埆"} and attr_name in text:
                return False
        return True

    def _match_generic_category_terms(self, query: str) -> List[Dict[str, Any]]:
        """处理“电脑”等泛化词，补充图谱中真实存在的相关三级类目。"""
        text = query or ""
        if "电脑" not in text or any(word in text for word in ["平板电脑", "笔记本电脑"]):
            return []
        result = []
        for term in self.graph_terms.get("Category3", []):
            category_name = term["text"]
            if "电脑" in category_name or "笔记本" in category_name:
                result.append({"label": "Category3", "text": category_name, "score": 0.91})
        return result

    def _load_graph_terms(self) -> Dict[str, List[Dict[str, str]]]:
        """从 Neo4j 读取图谱真实词表，避免规则抽取阶段生成图谱中不存在的词。"""
        empty_terms = {"Category3": [], "Trademark": [], "Attr": []}
        if self.driver is None:
            return empty_terms
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (c3:Category3)
                WITH collect(DISTINCT {text: c3.category3_name}) AS category3_terms
                MATCH (tm:Trademark)
                WITH category3_terms, collect(DISTINCT {text: tm.trademark_name}) AS trademark_terms
                MATCH (attr:Attr)
                WHERE attr.attr_name IN $attr_names
                RETURN category3_terms,
                       trademark_terms,
                       collect(DISTINCT {attr_name: attr.attr_name, text: attr.attr_value}) AS attr_terms
                """,
                attr_names=config.UIE_SCHEMA,
                database_=config.NEO4J_DATABASE,
            )
            if not records:
                return empty_terms
            row = records[0].data()
            terms = {
                "Category3": self._clean_graph_terms(row.get("category3_terms", [])),
                "Trademark": self._clean_graph_terms(row.get("trademark_terms", [])),
                "Attr": self._clean_graph_terms(row.get("attr_terms", [])),
            }
            if self.logger:
                self.logger.info(
                    "实体抽取图谱词表加载完成: category3=%s trademark=%s attr=%s",
                    len(terms["Category3"]),
                    len(terms["Trademark"]),
                    len(terms["Attr"]),
                )
            return terms
        except Exception as exc:
            if self.logger:
                self.logger.warning("实体抽取图谱词表加载失败，规则抽取仅保留 user_id: %s", exc)
            return empty_terms

    def _clean_graph_terms(self, terms: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        cleaned = []
        seen = set()
        for term in terms:
            text = str(term.get("text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            item = {"text": text}
            if term.get("attr_name"):
                item["attr_name"] = str(term["attr_name"]).strip()
            cleaned.append(item)
        return sorted(cleaned, key=lambda item: len(item["text"]), reverse=True)

    def _match_graph_named_terms(self, label: str, query: str) -> List[Dict[str, Any]]:
        matches = []
        for term in self.graph_terms.get(label, []):
            score = self._graph_term_match_score(term["text"], query)
            if score > 0:
                matches.append({"label": label, "text": term["text"], "score": score})
        return self._keep_best_by_label(matches, label)

    def _match_graph_attrs(self, query: str) -> List[Dict[str, Any]]:
        exact_matches = []
        partial_matches = []
        for term in self.graph_terms.get("Attr", []):
            score = self._graph_term_match_score(term["text"], query)
            if score <= 0:
                continue
            entity = {"label": "Attr", "attr_name": term["attr_name"], "text": term["text"], "score": score}
            if self._normal(term["text"]) in self._query_tokens(query):
                exact_matches.append(entity)
            else:
                partial_matches.append(entity)
        if exact_matches:
            return self._deduplicate(exact_matches)
        return self._keep_best_attr_matches(partial_matches)

    def _graph_term_match_score(self, graph_text: str, query: str) -> float:
        graph_norm = self._normal(graph_text)
        query_norm = self._normal(query)
        if not graph_norm or not query_norm:
            return 0.0
        if graph_norm in query_norm:
            return 0.98
        for token in self._query_tokens(query):
            if len(token) >= 2 and token in graph_norm:
                return 0.92
        return 0.0

    def _query_tokens(self, query: str) -> set[str]:
        try:
            import jieba

            raw_tokens = jieba.lcut(query or "")
        except Exception:
            raw_tokens = TOKEN_PATTERN.findall(query or "")
        tokens = {self._normal(token) for token in raw_tokens if TOKEN_PATTERN.fullmatch(token.strip())}
        for match in SPEC_PATTERN.finditer(query or ""):
            tokens.add(self._normal(match.group(0)))
        return {token for token in tokens if token}

    def _normal(self, text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).lower()

    def _keep_best_by_label(self, entities: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
        if not entities:
            return []
        exact = [item for item in entities if item["score"] >= 0.98]
        candidates = exact or entities
        best = max(candidates, key=lambda item: (float(item["score"]), len(str(item["text"]))))
        return [{"label": label, "text": best["text"], "score": best["score"]}]

    def _keep_best_attr_matches(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_name: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            attr_name = str(entity.get("attr_name") or "")
            if not attr_name:
                continue
            current = best_by_name.get(attr_name)
            if current is None or (entity["score"], -len(entity["text"])) > (current["score"], -len(current["text"])):
                best_by_name[attr_name] = entity
        return list(best_by_name.values())

    def _parse_uie_result(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for key, values in item.items():
            if key not in config.UIE_SCHEMA or not values:
                continue
            best = max(values, key=lambda value: float(value.get("probability", 0.0)))
            text = str(best.get("text", "")).strip()
            score = float(best.get("probability", 0.0))
            if not text:
                continue
            if is_query_placeholder_value(text):
                if self.logger:
                    self.logger.info("实体抽取UIE结果已过滤疑问占位值: attr_name=%s text=%s", key, text)
                continue
            if is_invalid_attr_value_pair(key, text):
                if self.logger:
                    self.logger.info("实体抽取UIE结果已过滤不合理属性组合: attr_name=%s text=%s", key, text)
                continue
            result.append({"label": "Attr", "attr_name": key, "text": text, "score": score})
        return result

    def _deduplicate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[tuple, Dict[str, Any]] = {}
        for entity in entities:
            key = (entity.get("label"), entity.get("attr_name"), entity.get("text"))
            if key not in best or entity.get("score", 0.0) > best[key].get("score", 0.0):
                best[key] = entity
        return list(best.values())


class HybridRetriever:
    """混合检索模块，融合 Neo4j 向量索引、全文索引和精确匹配结果。"""

    def __init__(self, driver, embedding: BgeEmbedding, logger=None):
        self.driver = driver
        self.embedding = embedding
        self.logger = logger

    def retrieve(self, entities: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for entity in entities:
            label = entity.get("label")
            text = str(entity.get("text") or "").strip()
            if not label or not text:
                continue
            if label == "User":
                results.extend(self._retrieve_user(text))
            elif label in config.NODE_INDEX_CONFIG:
                results.extend(self._retrieve_label(label, text, top_k))
        final_results = self._deduplicate(results)
        if self.logger:
            self.logger.info("混合检索结果: 候选节点数=%s\n%s", len(final_results), format_retrieval_table(final_results[:20]))
        return final_results

    def _retrieve_user(self, text: str) -> List[Dict[str, Any]]:
        digits = re.sub(r"\D", "", text)
        if not digits:
            return []
        records, _, _ = self.driver.execute_query(
            "MATCH (u:User {user_id: $user_id}) RETURN u.user_id AS user_id LIMIT 1",
            user_id=int(digits),
            database_=config.NEO4J_DATABASE,
        )
        return [{"label": "User", "properties": record.data(), "score": 1.0, "source": "exact"} for record in records]

    def _retrieve_label(self, label: str, text: str, top_k: int) -> List[Dict[str, Any]]:
        cfg = config.NODE_INDEX_CONFIG[label]
        vector = self.embedding.encode_one(text)
        fulltext = build_fulltext(text).replace(" ", " OR ") or text
        rows: List[Dict[str, Any]] = []

        try:
            records, _, _ = self.driver.execute_query(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                RETURN labels(node)[0] AS label, properties(node) AS properties, score, 'vector' AS source
                """,
                index_name=cfg["vector_index"],
                top_k=top_k,
                embedding=vector,
                database_=config.NEO4J_DATABASE,
            )
            rows.extend(record.data() for record in records)
        except Exception as exc:
            if self.logger:
                self.logger.warning("vector_retrieve_failed label=%s error=%s", label, exc)

        try:
            records, _, _ = self.driver.execute_query(
                """
                CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $top_k})
                YIELD node, score
                RETURN labels(node)[0] AS label, properties(node) AS properties, score, 'fulltext' AS source
                """,
                index_name=cfg["fulltext_index"],
                query=fulltext,
                top_k=top_k,
                database_=config.NEO4J_DATABASE,
            )
            rows.extend(record.data() for record in records)
        except Exception as exc:
            if self.logger:
                self.logger.warning("fulltext_retrieve_failed label=%s error=%s", label, exc)
        return rows

    def _deduplicate(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            props = dict(row.get("properties") or {})
            props.pop("embedding", None)
            props.pop("fulltext", None)
            key = f"{row.get('label')}::{props}"
            row["properties"] = props
            if key not in best or float(row.get("score", 0.0)) > float(best[key].get("score", 0.0)):
                best[key] = row
        return sorted(best.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)


class EntryNodeSelector:
    """入口节点过滤模块：结合用户问题二次筛选混合检索结果，降低噪声节点对 Cypher 生成的干扰。"""

    def __init__(self, llm: LLMClient, logger=None):
        self.llm = llm
        self.logger = logger

    def select(
        self,
        query: str,
        entry_nodes: List[Dict[str, Any]],
        entities: List[Dict[str, Any]] | None = None,
        max_nodes: int = 12,
    ) -> List[Dict[str, Any]]:
        if not entry_nodes:
            return []
        if self._is_pure_user_behavior_entities(query, entities or []):
            selected = [node for node in entry_nodes if node.get("label") == "User"]
            if selected:
                selected = self._deduplicate_selected(selected)[:max_nodes]
                if self.logger:
                    self.logger.info("entry_node_filter: pure user behavior query, keep only User\n%s", format_retrieval_table(selected))
                return selected
        candidates = entry_nodes[:20]
        rule_selected = self._semantic_rule_select(query, entry_nodes, entities or [], max_nodes=max_nodes)
        if self.llm.available:
            prompt = f"""
你是电商知识图谱入口节点过滤器。请结合用户问题，从混合检索候选节点中选择最适合作为 Cypher 查询入口的节点。

筛选原则:
1. 优先保留 source=fulltext 的节点，其次 exact，最后 vector。
2. 同一含义的节点优先选择分数更高的。
3. 与用户问题明确匹配的品类、品牌、属性必须保留，并结合实体抽取结果中的 attr_name/text 判断。
4. 如果用户表达比较模糊，或多个候选都可能相关，可以都保留。
5. 明显无关的向量召回噪声要删除，例如问题问口红时召回耳机、连衣裙等。
6. 规格单位需要归一化判断：32G 与 32GB 等价，屏幕尺寸在图谱中统一使用“英寸”，15英寸 与 15.6英寸/16英寸相关，2TB 与 2TB SSD 相关。
7. 如果用户问题包含某个规格约束，但 fulltext 没有召回完全等价节点，必须保留最接近的 vector 候选。
8. 遇到“以上、不小于、大于、至少、包括”等范围/包含语义，要保留满足范围的多个属性候选；例如 2K以上要保留 2K、4K。
9. 遇到“笔记本”这类模糊品类词，要保留图谱中所有相关三级品类，例如游戏笔记本、轻薄笔记本。
10. 返回节点数量不超过 {max_nodes}，但不要为了压缩而删除必要约束。

只返回 JSON 对象，不要输出解释文字:
{{"selected_indexes":[1,2,3],"reason":"中文简要说明筛选理由"}}

用户问题:
{query}

实体抽取结果:
{entities or []}

候选节点表:
{format_retrieval_table(candidates)}
"""
            try:
                raw_output = self.llm.invoke(prompt, task_name="入口节点过滤", log_response=True)
                data = parse_json_object(raw_output)
                indexes = data.get("selected_indexes") or []
                selected = []
                for index in indexes:
                    try:
                        pos = int(index) - 1
                    except (TypeError, ValueError):
                        continue
                    if 0 <= pos < len(candidates):
                        selected.append(candidates[pos])
                if rule_selected and (self._has_range_semantics(query) or self._is_fuzzy_computer_query(query)):
                    selected = rule_selected[:max_nodes]
                else:
                    selected = self._merge_selected(rule_selected, selected, entities or [], query=query, max_nodes=max_nodes)
                if selected:
                    if self.logger:
                        self.logger.info(
                            "入口节点过滤结果: 规则语义+大模型筛选成功 reason=%s\n%s",
                            data.get("reason", ""),
                            format_retrieval_table(selected),
                        )
                    return selected
                if self.logger:
                    self.logger.warning("入口节点过滤: 大模型未返回有效节点，使用规则兜底")
            except Exception as exc:
                if self.logger:
                    self.logger.warning("入口节点过滤失败，使用规则兜底: %s", exc)
        selected = rule_selected or self._rule_select(query, entry_nodes, entities or [], max_nodes)
        if self.logger:
            self.logger.info("入口节点过滤结果: 规则兜底\n%s", format_retrieval_table(selected))
        return selected

    def _is_pure_user_behavior_entities(self, query: str, entities: List[Dict[str, Any]]) -> bool:
        if not is_user_behavior_query(query):
            return False
        return bool(entities) and all(entity.get("label") == "User" for entity in entities)

    def _semantic_rule_select(
        self,
        query: str,
        entry_nodes: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        max_nodes: int,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        source_rank = {"fulltext": 0, "exact": 1, "vector": 2}
        sorted_nodes = sorted(
            entry_nodes,
            key=lambda item: (source_rank.get(str(item.get("source")), 9), -float(item.get("score") or 0.0)),
        )
        entity_attrs = {
            (str(entity.get("attr_name") or ""), str(entity.get("text") or ""))
            for entity in entities
            if entity.get("label") == "Attr"
        }
        category_texts = [str(entity.get("text") or "") for entity in entities if entity.get("label") == "Category3"]

        if self._is_fuzzy_computer_query(query):
            selected.extend(
                node for node in sorted_nodes if node.get("label") == "Category3" and self._category_matches_computer_query(query, self._node_text(node))
            )

        if self._has_range_semantics(query):
            selected.extend(self._range_attr_nodes(query, sorted_nodes, entity_attrs))
        else:
            selected.extend(self._best_exact_attr_nodes(query, sorted_nodes, entity_attrs, category_texts))

        if not selected:
            selected = self._best_exact_attr_nodes(query, sorted_nodes, entity_attrs, category_texts)
        return self._deduplicate_selected(selected)[:max_nodes]

    def _range_attr_nodes(
        self,
        query: str,
        entry_nodes: List[Dict[str, Any]],
        entity_attrs: set[tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        screen_threshold = extract_first_number(query, r"(\d+(?:\.\d+)?)\s*(?:英寸|寸)")
        memory_threshold = extract_first_number(query, r"(\d+(?:\.\d+)?)\s*(?:G|GB)\s*内存|内存\D*(\d+(?:\.\d+)?)\s*(?:G|GB)")
        disk_tb = extract_first_number(query, r"(\d+(?:\.\d+)?)\s*TB")
        resolution_k = extract_first_number(query, r"(\d+(?:\.\d+)?)\s*K")

        for node in entry_nodes:
            props = dict(node.get("properties") or {})
            if node.get("label") != "Attr":
                continue
            attr_name = str(props.get("attr_name") or "")
            attr_value = str(props.get("attr_value") or "")
            if attr_name == "屏幕尺寸" and screen_threshold is not None:
                value = self._numeric_value(attr_value)
                if value is not None and value >= screen_threshold and self._screen_size_allowed(query, value):
                    selected.append(node)
            elif attr_name == "内存" and memory_threshold is not None:
                value = self._numeric_value(attr_value.upper().replace("GB", "").replace("G", ""))
                if value is not None and value >= memory_threshold:
                    selected.append(node)
            elif attr_name == "硬盘" and disk_tb is not None:
                if "TB" not in attr_value.upper():
                    continue
                value = self._numeric_value(attr_value.upper().replace("TB", ""))
                if value is not None and value >= disk_tb:
                    selected.append(node)
            elif attr_name == "分辨率" and resolution_k is not None:
                if "K" not in attr_value.upper():
                    continue
                value = self._numeric_value(attr_value.upper().replace("K", ""))
                if value is not None and value >= resolution_k:
                    selected.append(node)
            elif (attr_name, attr_value) in entity_attrs:
                selected.append(node)
        return selected

    def _best_exact_attr_nodes(
        self,
        query: str,
        entry_nodes: List[Dict[str, Any]],
        entity_attrs: set[tuple[str, str]],
        category_texts: List[str],
    ) -> List[Dict[str, Any]]:
        best_by_attr: Dict[str, Dict[str, Any]] = {}
        for node in entry_nodes:
            if node.get("label") in {"Category3", "Trademark", "User"} and self._node_text(node) in query:
                best_by_attr.setdefault(f"{node.get('label')}::{self._node_text(node)}", node)
                continue
            if node.get("label") != "Attr":
                continue
            props = dict(node.get("properties") or {})
            attr_name = str(props.get("attr_name") or "")
            attr_value = str(props.get("attr_value") or "")
            if attr_name in {"类别", "香水彩妆"} and category_value_covered_by_category_texts(attr_value, category_texts):
                continue
            if (attr_name, attr_value) not in entity_attrs and attr_value not in query:
                continue
            current = best_by_attr.get(attr_name)
            if current is None or self._node_rank(node) < self._node_rank(current):
                best_by_attr[attr_name] = node
        return list(best_by_attr.values())

    def _rule_select(
        self,
        query: str,
        entry_nodes: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        max_nodes: int,
    ) -> List[Dict[str, Any]]:
        semantic_selected = self._semantic_rule_select(query, entry_nodes, entities, max_nodes)
        if semantic_selected:
            return semantic_selected
        source_rank = {"fulltext": 0, "exact": 1, "vector": 2}
        sorted_nodes = sorted(
            entry_nodes,
            key=lambda item: (source_rank.get(str(item.get("source")), 9), -float(item.get("score") or 0.0)),
        )
        return self._deduplicate_selected(sorted_nodes)[:max_nodes]

    def _merge_selected(
        self,
        rule_selected: List[Dict[str, Any]],
        llm_selected: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        query: str,
        max_nodes: int,
    ) -> List[Dict[str, Any]]:
        filtered_llm = [node for node in llm_selected if self._node_allowed_by_entities(node, entities, query)]
        return self._deduplicate_selected(rule_selected + filtered_llm)[:max_nodes]

    def _has_range_semantics(self, query: str) -> bool:
        return any(word in (query or "") for word in ["以上", "不小于", "大于", "至少", "不少于", "包括", "及以上"])

    def _is_fuzzy_computer_query(self, query: str) -> bool:
        text = query or ""
        return (
            ("笔记本" in text and not any(word in text for word in ["游戏笔记本", "轻薄笔记本"]))
            or ("电脑" in text and not any(word in text for word in ["平板电脑", "笔记本电脑"]))
        )

    def _category_matches_computer_query(self, query: str, category_name: str) -> bool:
        text = query or ""
        if "电脑" in text:
            return "电脑" in category_name or "笔记本" in category_name
        if "笔记本" in text:
            return "笔记本" in category_name
        return False

    def _node_allowed_by_entities(self, node: Dict[str, Any], entities: List[Dict[str, Any]], query: str) -> bool:
        if node.get("label") != "Attr":
            return True
        props = dict(node.get("properties") or {})
        attr_name = str(props.get("attr_name") or "")
        attr_value = str(props.get("attr_value") or "")
        category_texts = [str(entity.get("text") or "") for entity in entities if entity.get("label") == "Category3"]
        if attr_name in {"类别", "香水彩妆"} and category_value_covered_by_category_texts(attr_value, category_texts):
            return False
        if self._is_return_only_attr_query(query, attr_name) and attr_value not in (query or ""):
            return False
        entity_values = [
            str(entity.get("text") or "")
            for entity in entities
            if entity.get("label") == "Attr" and str(entity.get("attr_name") or "") == attr_name
        ]
        if not entity_values:
            return True
        return any(self._attr_value_equivalent(attr_value, entity_value) for entity_value in entity_values)

    def _is_return_only_attr_query(self, query: str, attr_name: str) -> bool:
        text = query or ""
        if attr_name in {"屏幕尺寸", "尺码"} and any(word in text for word in ["多少尺寸", "哪些尺寸", "有什么尺寸", "尺寸的", "尺寸是多少"]):
            return extract_first_number(text, r"(\d+(?:\.\d+)?)\s*(?:英寸|寸)") is None
        return False

    def _attr_value_equivalent(self, graph_value: str, entity_value: str) -> bool:
        return self._normalize_unit(graph_value) == self._normalize_unit(entity_value)

    def _normalize_unit(self, value: str) -> str:
        text = re.sub(r"\s+", "", str(value or "")).upper()
        if re.fullmatch(r"\d+(?:\.\d+)?G", text):
            text += "B"
        return text

    def _numeric_value(self, text: str) -> float | None:
        match = re.search(r"\d+(?:\.\d+)?", text or "")
        return float(match.group(0)) if match else None

    def _screen_size_allowed(self, query: str, value: float) -> bool:
        """按品类语义限制屏幕尺寸范围，避免笔记本查询保留电视尺寸。"""
        text = query or ""
        if "笔记本" in text:
            return value <= 20
        if "手机" in text:
            return value <= 8
        if "平板" in text:
            return 8 <= value <= 14
        if "电视" in text:
            return value >= 24
        return True

    def _node_rank(self, node: Dict[str, Any]) -> tuple[int, float]:
        source_rank = {"fulltext": 0, "exact": 1, "vector": 2}
        return (source_rank.get(str(node.get("source")), 9), -float(node.get("score") or 0.0))

    def _node_text(self, node: Dict[str, Any]) -> str:
        props = dict(node.get("properties") or {})
        for key in ("category3_name", "trademark_name", "attr_value", "sku_name", "spu_name"):
            if props.get(key):
                return str(props[key])
        return ""

    def _deduplicate_selected(self, entry_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected = []
        seen = set()
        for node in entry_nodes:
            props = dict(node.get("properties") or {})
            key = f"{node.get('label')}::{props}"
            if key in seen:
                continue
            seen.add(key)
            selected.append(node)
        return selected


class CypherGenerator:
    """Text2Cypher 模块，优先由大模型生成查询，失败时用规则 Cypher 兜底。"""

    def __init__(self, llm: LLMClient, logger=None):
        self.llm = llm
        self.logger = logger

    def generate(self, query: str, intent: Dict[str, Any], entities: List[Dict[str, Any]], entry_nodes: List[Dict[str, Any]], user_id: int | None) -> str:
        if self.llm.available:
            prompt = self._build_prompt(query, intent, entry_nodes, user_id)
            try:
                raw_output = self.llm.invoke(prompt, task_name="Cypher生成", log_response=False)
                cypher, explanation = parse_cypher_generation_output(raw_output)
                original_cypher = cypher
                cypher = self._postprocess_cypher(cypher, query, intent, entities, entry_nodes)
                if normalize_cypher_for_compare(original_cypher) != normalize_cypher_for_compare(cypher):
                    explanation = "大模型生成后，系统已根据入口节点、范围语义和图谱 schema 规范化 Cypher，确保属性单位、范围条件和关系方向可正确查询。"
                if self.logger:
                    self.logger.info("Cypher生成结果: 大模型生成成功，查询含义=%s", explanation)
                    self.logger.info("Cypher生成查询语句:\n%s", format_cypher(cypher))
                return cypher
            except Exception as exc:
                if self.logger:
                    self.logger.warning("Cypher生成失败，使用规则兜底: %s", exc)
        cypher = self._fallback_cypher(intent, entities, user_id, query, entry_nodes)
        cypher = self._postprocess_cypher(cypher, query, intent, entities, entry_nodes)
        if self.logger:
            self.logger.info("Cypher生成结果: 规则兜底生成成功")
            self.logger.info("Cypher生成查询语句:\n%s", format_cypher(cypher))
        return cypher

    def fallback(self, intent: Dict[str, Any], entities: List[Dict[str, Any]], user_id: int | None, query: str = "", entry_nodes: List[Dict[str, Any]] | None = None) -> str:
        """当大模型生成/修正多次失败时，使用规则模板生成安全只读 Cypher。"""
        return self._fallback_cypher(intent, entities, user_id, query, entry_nodes or [])

    def correct(
        self,
        query: str,
        cypher: str,
        errors: List[str],
        entry_nodes: List[Dict[str, Any]],
        intent: Dict[str, Any],
        user_id: int | None,
    ) -> str:
        if not self.llm.available:
            return cypher
        prompt = f"""
你是 Cypher 专家。请根据 Neo4j schema、用户问题、入口节点和错误信息修正查询。
请返回 JSON 对象，不要输出 JSON 以外的解释文字。
JSON 字段:
- cypher: 修正后的 Neo4j 只读 Cypher，只能使用 MATCH/OPTIONAL MATCH/WITH/RETURN/WHERE/ORDER BY/LIMIT。
- explanation: 用中文简要说明这条 Cypher 的查询含义、使用了哪些节点/关系、修复了哪些问题。

图谱拓扑约束:
- 品牌 Trademark 只能通过 (SPU)-[:Belong]->(Trademark) 关联，不能接在 Category1/Category2/Category3 后面。
- 品牌约束必须绑定在商品路径上的 Trademark 变量，例如 MATCH (spu)-[:Belong]->(tm:Trademark) WHERE tm.trademark_name='荣耀'。禁止单独写 MATCH (tm:Trademark {{trademark_name:'荣耀'}}) 后又返回另一个未约束品牌变量。
- 分类链只能是 (SPU)-[:Belong]->(Category3)-[:Belong]->(Category2)-[:Belong]->(Category1)。
- 属性只能通过 (SKU)-[:Have]->(Attr) 关联。
- 从属性查商品时推荐使用: MATCH (sku:SKU)-[:Have]->(attr:Attr) WHERE attr.attr_name=... AND attr.attr_value=...
- 入口节点只是查询定位依据，不代表 Cypher 必须从入口节点出发；禁止写 (Attr)-[:Have]->(SKU) 或 (Attr)-[:Belong]->(SPU)。
- 如果使用 WITH，后续 RETURN 只能使用 WITH 保留下来的变量；需要返回属性证据时，建议在最终 RETURN 前重新 OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)。
- 多个商品规格必须同时满足时，不要把多个属性写成一个 attr.attr_name IN ... AND attr.attr_value IN ...，这只表示命中任意一个属性。
- 多个规格推荐写成多个 EXISTS 子查询，每个 EXISTS 只检查一个属性条件，最后用 AND 连接。
- “以上/大于/不低于/至少”是范围条件，优先使用数值比较；例如 32G 内存应匹配 32GB，2TB 硬盘应匹配 2TB SSD，15英寸以上应匹配 15.6英寸/16英寸/17.3英寸。
- 如果用户没有说“以上/不小于/大于/至少/不少于”等范围词，只说 32G内存、2TB硬盘、2K屏幕，则按等值或单位等价匹配，不要使用 >=。
- Category3 节点和 Attr(attr_name='类别') 表达的是同一类品类证据；如果二者值相同或同义，不要同时用 AND 强制满足，应使用 Category3 约束，或写成 OR/EXISTS 二选一。
- Category3 节点和 Attr(attr_name='香水彩妆') 在美妆场景也可能表达同一类品类证据；例如 category3_name=口红 与 attr_name=香水彩妆, attr_value=口红 是 OR/二选一关系，不要用 AND 同时约束。
- 品类证据组内部可以 OR，例如 category3_name=口红 OR 香水彩妆=口红；但功效、颜色、内存等其它属性约束必须与品类证据组用 AND 连接。不要把 功效=保湿/补水 与 品类=口红 写成同一层 OR。
- 如果同一个 attr_name 下有多个候选值，例如 功效=保湿 或 功效=补水，应写成 a.attr_name='功效' AND a.attr_value IN ['保湿','补水']，表示同一属性下任一值命中。
- 如果入口节点包含多个 Category3，表示用户问题的可选品类范围，应使用 OR/IN/CONTAINS 任一满足，不要让多个 Category3 同时 AND。

用户行为约束:
- 当前意图为 {intent.get("intent")}。
- 只有 intent=user_interest_query 时，才允许使用 User 节点、user_id={user_id}、View/Click/Favorite 关系。
- 如果当前意图是 user_interest_query，Cypher 必须从指定用户出发，必须包含 (u:User {{user_id: {user_id}}})-[r:View|Click|Favorite]->(sku:SKU) 或等价写法。
- 用户问“收藏”时优先用 Favorite；问“点击”时优先用 Click；问“浏览/看过”时优先用 View；问“关注/感兴趣/历史/行为”时使用 View|Click|Favorite。
- user_interest_query 的返回字段必须包含 collect(DISTINCT type(r)) AS behaviors 和 count(r) AS behavior_count，回答必须基于该用户行为子图。
- 如果当前意图不是 user_interest_query，必须删除 User 节点、user_id 约束以及 View/Click/Favorite 关系，只围绕商品、品类、品牌和属性查询。

schema:
{config.GRAPH_SCHEMA_TEXT}

用户问题:
{query}

入口节点:
{entry_nodes}

原 Cypher:
{cypher}

错误信息:
{errors}
"""
        try:
            raw_output = self.llm.invoke(prompt, task_name="Cypher修正", log_response=False)
            cypher, explanation = parse_cypher_generation_output(raw_output)
            original_cypher = cypher
            cypher = self._postprocess_cypher(cypher, query, intent, None, entry_nodes)
            if normalize_cypher_for_compare(original_cypher) != normalize_cypher_for_compare(cypher):
                explanation = "大模型修正后，系统已根据入口节点、范围语义和图谱 schema 再次规范化 Cypher，确保查询条件可执行且符合图谱结构。"
            if self.logger:
                self.logger.info("Cypher修正结果: 大模型修正成功，修正含义=%s", explanation)
                self.logger.info("Cypher修正后查询语句:\n%s", format_cypher(cypher))
            return cypher
        except Exception as exc:
            if self.logger:
                self.logger.warning("Cypher修正失败: %s", exc)
            return cypher

    def _postprocess_cypher(
        self,
        cypher: str,
        query: str,
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]] | None = None,
        entry_nodes: List[Dict[str, Any]] | None = None,
    ) -> str:
        """Normalize deterministic user-behavior details that LLMs often drift on."""
        if intent.get("intent") != "user_interest_query" and self._is_notebook_spec_query(query):
            return self._notebook_spec_fallback_cypher(query)
        attr_entities = self._postprocess_attr_entities(entities or [], entry_nodes or [])
        if intent.get("intent") != "user_interest_query" and attr_entities:
            filter_entities = self._build_attr_filter_entities(query, entities or [], attr_entities)
            return self._attr_filter_cypher(filter_entities)
        behavior_pattern = build_behavior_pattern(query)
        rel_pattern = re.compile(
            r"(\(\s*[A-Za-z_]\w*\s*:\s*User[^)]*\)\s*-\s*)"
            r"\[\s*(?:[A-Za-z_]\w*\s*)?:\s*(?:View|Click|Favorite)(?:\|(?:View|Click|Favorite))*\s*\]"
            r"(\s*->\s*\(\s*[A-Za-z_]\w*\s*:\s*SKU)",
            flags=re.I,
        )
        return rel_pattern.sub(rf"\1[r:{behavior_pattern}]\2", cypher or "")

    def _postprocess_attr_entities(
        self,
        entities: List[Dict[str, Any]],
        entry_nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        attr_entities = [
            entity
            for entity in entities
            if entity.get("label") == "Attr" and entity.get("attr_name") and entity.get("text")
        ]
        if attr_entities:
            return attr_entities
        for node in entry_nodes:
            if node.get("label") != "Attr":
                continue
            props = dict(node.get("properties") or {})
            attr_name = props.get("attr_name")
            attr_value = props.get("attr_value")
            if attr_name and attr_value:
                attr_entities.append({"label": "Attr", "attr_name": attr_name, "text": attr_value, "score": node.get("score", 1.0)})
        return attr_entities

    def _build_attr_filter_entities(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        attr_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep category/brand entities and use sanitized attribute entities for Cypher filters."""
        base_entities = [
            entity
            for entity in entities
            if entity.get("label") in {"Category3", "Trademark", "User"}
        ]
        text = query or ""
        has_tv_context = "\u7535\u89c6" in text or "\u5e73\u677f\u7535\u89c6" in text
        has_screen_size = any(
            entity.get("label") == "Attr" and entity.get("attr_name") == "\u5c4f\u5e55\u5c3a\u5bf8"
            for entity in attr_entities
        )
        clean_attrs = []
        for entity in attr_entities:
            if is_query_placeholder_value(str(entity.get("text") or "")):
                continue
            if has_tv_context and has_screen_size and entity.get("attr_name") == "\u5c3a\u7801":
                continue
            clean_attrs.append(entity)
        return base_entities + clean_attrs

    def _attr_filter_cypher(self, attr_entities: List[Dict[str, Any]]) -> str:
        where_clause = build_where_clause(attr_entities)
        return f"""
        MATCH (sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3)
        MATCH (spu)-[:Belong]->(tm:Trademark)
        {where_clause}
        OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)
        RETURN DISTINCT sku.sku_id AS sku_id, sku.sku_name AS sku_name,
               c3.category3_name AS category3_name,
               tm.trademark_name AS trademark_name,
               collect(DISTINCT attr.attr_name + '=' + attr.attr_value) AS matched_attrs
        LIMIT {config.DEFAULT_RESULT_LIMIT}
        """

    def _build_prompt(self, query: str, intent: Dict[str, Any], entry_nodes: List[Dict[str, Any]], user_id: int | None) -> str:
        behavior_pattern = build_behavior_pattern(query)
        return f"""
当 intent=user_interest_query 时，本问题应使用的用户行为关系是: {behavior_pattern}。生成 Cypher 时必须优先使用这个关系集合，不要额外扩展为其它用户行为关系。
你是电商知识图谱 Cypher 生成专家。请根据用户问题生成 Neo4j 只读 Cypher。

严格要求:
1. 只能使用 MATCH/OPTIONAL MATCH/WITH/RETURN/WHERE/ORDER BY/LIMIT。
2. 禁止 CREATE、MERGE、DELETE、SET、DROP、CALL dbms、LOAD CSV 等写入或管理操作。
3. 不能返回 embedding、fulltext 等内部字段。
4. 关系方向必须符合 schema。
5. 默认 LIMIT {config.DEFAULT_RESULT_LIMIT}。
6. 当前意图为 {intent.get("intent")}。只有 intent=user_interest_query 时，才允许使用 User 节点、user_id={user_id}、View/Click/Favorite 关系。
7. 如果当前意图是 user_interest_query，Cypher 必须绑定指定用户，必须包含 (u:User {{user_id: {user_id}}})-[r:View|Click|Favorite]->(sku:SKU) 或等价写法，不能只查询全库商品。
8. user_interest_query 的行为关系选择规则：用户问“收藏”时优先用 Favorite；问“点击”时优先用 Click；问“浏览/看过”时优先用 View；问“关注/感兴趣/历史/行为”时使用 View|Click|Favorite。
9. user_interest_query 的返回字段必须包含 collect(DISTINCT type(r)) AS behaviors 和 count(r) AS behavior_count，便于回答说明依据来自用户行为。
10. 如果当前意图不是 user_interest_query，禁止使用 User 节点、user_id 约束、View/Click/Favorite 用户行为关系。
11. 品牌 Trademark 只能通过 (SPU)-[:Belong]->(Trademark) 关联，不能写成 Category1/Category2/Category3 -> Trademark。
11.1 品牌约束必须写在和 SPU 相连的 Trademark 变量上。例如:
MATCH (sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3)
MATCH (spu)-[:Belong]->(tm:Trademark)
WHERE tm.trademark_name = '荣耀'
禁止写成:
MATCH (tm:Trademark {{trademark_name:'荣耀'}})
MATCH (spu)-[:Belong]->(tm2:Trademark)
RETURN tm2.trademark_name
因为 tm 和商品路径没有连接，tm2 也没有被限制为荣耀。
12. 分类链只能是 (SPU)-[:Belong]->(Category3)-[:Belong]->(Category2)-[:Belong]->(Category1)。
13. 属性只能通过 (SKU)-[:Have]->(Attr) 关联。
14. 从属性查商品时推荐使用: MATCH (sku:SKU)-[:Have]->(attr:Attr) WHERE attr.attr_name=... AND attr.attr_value=...
15. 商品查询必须返回可解释证据字段，至少包含 sku.sku_id AS sku_id、sku.sku_name AS sku_name、c3.category3_name AS category3_name、tm.trademark_name AS trademark_name。
16. 如果按 Attr 过滤商品，RETURN 中需要包含命中的属性证据，例如 collect(DISTINCT attr.attr_name + '=' + attr.attr_value) AS matched_attrs。
17. 如果需要 OPTIONAL MATCH 补充证据字段，必须先用 MATCH/WHERE 完成商品过滤，再写 OPTIONAL MATCH，避免 WHERE 只作用在 OPTIONAL MATCH 上。
18. 入口节点是混合检索并经过筛选后的高可信节点，Cypher 必须优先围绕入口节点中的 label 和 properties 建立约束。
19. 入口节点 source=fulltext 且分数高时通常代表精确匹配，应优先作为 WHERE 约束；source=vector 的模糊候选只在与用户问题语义一致时使用。
20. 如果入口节点包含 Attr，需要使用 attr_name 和 attr_value 同时约束；如果入口节点包含 Category3/Trademark，需要使用对应名称属性约束。
21. 入口节点只是查询定位依据，不代表 Cypher 必须从入口节点出发；禁止写 (Attr)-[:Have]->(SKU) 或 (Attr)-[:Belong]->(SPU)。
22. 遇到“以上/大于/不低于/至少”这类范围条件，不要写死等值；应使用 CONTAINS 或数值比较表达，例如屏幕尺寸 15 英寸以上可匹配 15英寸、15.6英寸、16英寸、17.3英寸。
23. 规格单位需要归一化：32G 可匹配 32GB；2TB 可匹配 2TB SSD；屏幕尺寸在图谱中统一使用“英寸”，2K以上可匹配 2K 或更高 K 值。
24. 用户只说“笔记本”时，不要强制限定为“游戏笔记本”，应使用 c3.category3_name CONTAINS '笔记本'；只有用户明确说游戏/电竞时才限定游戏笔记本。
25. 如果使用 WITH，后续 RETURN 只能使用 WITH 保留下来的变量；需要返回属性证据时，建议在最终 RETURN 前重新 OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)。
26. 多个商品规格必须同时满足时，不要写成 attr.attr_name IN [...] AND attr.attr_value IN [...]，因为这只表示同一个属性节点命中其中任意一个值，不能保证 SKU 同时满足所有规格。
27. 多规格查询推荐使用多个 EXISTS 子查询，每个 EXISTS 只检查一个属性条件，最后用 AND 连接。
28. 如果用户没有说“以上/不小于/大于/至少/不少于”等范围词，只说 32G内存、2TB硬盘、2K屏幕、5000mAh电池，则按等值或单位等价匹配。例如 32G内存应写成 a.attr_name='内存' AND toUpper(a.attr_value) IN ['32G','32GB']，5000mAh电池应写成 toUpper(a.attr_value) = '5000MAH' 或 a.attr_value='5000mAh'，不要写 >= 32 或 >= 5000。
28.1 用户问“都是多少尺寸/有哪些尺寸/尺寸是多少”时，这是属性返回需求，不是属性过滤条件。不要把“多少”当成 attr_value 过滤；应在 RETURN 中返回或收集 attr_name IN ['屏幕尺寸','尺码'] 的属性值。
29. Category3 节点和 Attr(attr_name='类别' 或 '香水彩妆') 可能表达同一类品类证据；如果入口节点同时有 category3_name=口红 和 attr_name=香水彩妆, attr_value=口红，不要写成两个 AND 条件。应优先使用 c3.category3_name CONTAINS '口红'，或写成:
WHERE c3.category3_name CONTAINS '口红'
   OR EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name='香水彩妆' AND a.attr_value='口红' }}
同理，如果入口节点同时有 category3_name=平板电脑 和 attr_name=类别, attr_value=平板电脑，不要写成两个 AND 条件。应优先使用 c3.category3_name CONTAINS '平板电脑'，或写成:
WHERE c3.category3_name CONTAINS '平板电脑'
   OR EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name='类别' AND a.attr_value='平板电脑' }}
30. 品类证据组内部是 OR，但其它属性约束必须和品类证据组 AND。例如“保湿或者补水功能的口红”应写成:
WHERE (
  c3.category3_name CONTAINS '口红'
  OR EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name='香水彩妆' AND a.attr_value='口红' }}
)
AND EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name='功效' AND a.attr_value IN ['保湿','补水'] }}
不要写成 c3.category3_name CONTAINS '口红' OR 香水彩妆=口红 OR 功效 IN ['保湿','补水']，否则会召回不满足功效的口红或非口红商品。
31. 如果入口节点包含多个 Category3，例如游戏笔记本、轻薄笔记本、平板电脑，表示候选品类集合，应写成:
WHERE c3.category3_name IN ['游戏笔记本', '轻薄笔记本', '平板电脑']
或用 OR 连接，不能写成多个 AND。
32. 用户兴趣查询可参考这种结构:
MATCH (u:User {{user_id: {user_id}}})-[r:View|Click|Favorite]->(sku:SKU)
MATCH (sku)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3)
MATCH (spu)-[:Belong]->(tm:Trademark)
WHERE c3.category3_name CONTAINS '智能手机'
OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)
RETURN DISTINCT sku.sku_id AS sku_id, sku.sku_name AS sku_name, c3.category3_name AS category3_name, tm.trademark_name AS trademark_name, collect(DISTINCT type(r)) AS behaviors, count(r) AS behavior_count, collect(DISTINCT attr.attr_name + '=' + attr.attr_value) AS matched_attrs
LIMIT {config.DEFAULT_RESULT_LIMIT}
33. 笔记本规格查询可参考这种结构:
MATCH (sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3)
MATCH (spu)-[:Belong]->(tm:Trademark)
WHERE c3.category3_name CONTAINS '笔记本'
  AND EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name IN ['屏幕尺寸','尺码'] AND toFloat(replace(replace(a.attr_value,'英寸',''),'寸','')) >= 15 }}
  AND EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name='内存' AND toFloat(replace(replace(toUpper(a.attr_value),'GB',''),'G','')) >= 32 }}
  AND EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name='硬盘' AND toUpper(a.attr_value) CONTAINS '2TB' }}
  AND EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name='分辨率' AND toFloat(replace(toUpper(a.attr_value),'K','')) >= 2 }}
OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)
RETURN DISTINCT sku.sku_id AS sku_id, sku.sku_name AS sku_name, c3.category3_name AS category3_name, tm.trademark_name AS trademark_name, collect(DISTINCT attr.attr_name + '=' + attr.attr_value) AS matched_attrs
LIMIT {config.DEFAULT_RESULT_LIMIT}
34. 请返回 JSON 对象，不要输出 JSON 以外的解释文字。

JSON 字段:
- cypher: Neo4j 只读 Cypher。
- explanation: 用中文简要说明这条 Cypher 的查询含义、使用了哪些节点/关系、如何约束用户问题中的实体。


schema:
{config.GRAPH_SCHEMA_TEXT}

用户问题:
{query}

意图:
{intent}

筛选后的入口节点:
{entry_nodes}
"""

    def _fallback_cypher(self, intent: Dict[str, Any], entities: List[Dict[str, Any]], user_id: int | None, query: str = "", entry_nodes: List[Dict[str, Any]] | None = None) -> str:
        intent_name = intent.get("intent")
        if self._is_notebook_spec_query(query):
            return self._notebook_spec_fallback_cypher(query)
        where_clause = build_where_clause(entities)
        if intent_name == "user_interest_query" and user_id is not None:
            behavior_pattern = build_behavior_pattern(query)
            return f"""
            MATCH (u:User {{user_id: {int(user_id)}}})-[r:{behavior_pattern}]->(sku:SKU)
            MATCH (sku)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3)
            MATCH (spu)-[:Belong]->(tm:Trademark)
            {where_clause}
            OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)
            RETURN sku.sku_id AS sku_id, sku.sku_name AS sku_name,
                   c3.category3_name AS category3_name,
                   tm.trademark_name AS trademark_name,
                   collect(DISTINCT type(r)) AS behaviors,
                   count(DISTINCT r) AS behavior_count,
                   collect(DISTINCT attr.attr_name + '=' + attr.attr_value) AS matched_attrs
            ORDER BY behavior_count DESC
            LIMIT {config.DEFAULT_RESULT_LIMIT}
            """
        return f"""
        MATCH (sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3)
        MATCH (spu)-[:Belong]->(tm:Trademark)
        {where_clause}
        OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)
        RETURN sku.sku_id AS sku_id, sku.sku_name AS sku_name,
               c3.category3_name AS category3_name,
               tm.trademark_name AS trademark_name,
               collect(DISTINCT attr.attr_name + '=' + attr.attr_value) AS matched_attrs
        LIMIT {config.DEFAULT_RESULT_LIMIT}
        """

    def _is_notebook_spec_query(self, query: str) -> bool:
        text = query or ""
        return "笔记本" in text and any(token in text for token in ["内存", "硬盘", "英寸", "寸", "分辨率", "屏幕"])

    def _notebook_spec_fallback_cypher(self, query: str) -> str:
        """面向笔记本规格筛选的兜底查询，处理范围条件和常见单位差异。"""
        text = query or ""
        category_condition = "c3.category3_name CONTAINS '笔记本'"
        if any(token in text for token in ["游戏", "电竞", "游戏本"]):
            category_condition = "c3.category3_name = '游戏笔记本'"
        screen_threshold = extract_first_number(text, r"(\d+(?:\.\d+)?)\s*(?:英寸|寸)")
        memory_threshold = extract_first_number(text, r"(\d+(?:\.\d+)?)\s*(?:G|GB)\s*内存|内存\D*(\d+(?:\.\d+)?)\s*(?:G|GB)")
        disk_tb = extract_first_number(text, r"(\d+(?:\.\d+)?)\s*TB")
        resolution_k = extract_first_number(text, r"(\d+(?:\.\d+)?)\s*K")

        exists_parts = []
        if screen_threshold is not None:
            exists_parts.append(
                "EXISTS { MATCH (sku)-[:Have]->(a:Attr) "
                "WHERE a.attr_name IN ['屏幕尺寸', '尺码'] "
                f"AND toFloat(replace(replace(a.attr_value, '英寸', ''), '寸', '')) >= {screen_threshold} }}"
            )
        if memory_threshold is not None:
            exists_parts.append(
                "EXISTS { MATCH (sku)-[:Have]->(a:Attr) "
                "WHERE a.attr_name = '内存' "
                f"AND toFloat(replace(replace(toUpper(a.attr_value), 'GB', ''), 'G', '')) >= {memory_threshold} }}"
            )
        if disk_tb is not None:
            exists_parts.append(
                "EXISTS { MATCH (sku)-[:Have]->(a:Attr) "
                "WHERE a.attr_name = '硬盘' "
                f"AND toUpper(a.attr_value) CONTAINS '{format_number_for_cypher(disk_tb)}TB' }}"
            )
        if resolution_k is not None:
            exists_parts.append(
                "EXISTS { MATCH (sku)-[:Have]->(a:Attr) "
                "WHERE a.attr_name = '分辨率' "
                f"AND toFloat(replace(toUpper(a.attr_value), 'K', '')) >= {resolution_k} }}"
            )
        where_clause = "WHERE " + " AND ".join([category_condition] + exists_parts)
        return f"""
        MATCH (sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(c3:Category3)
        MATCH (spu)-[:Belong]->(tm:Trademark)
        {where_clause}
        OPTIONAL MATCH (sku)-[:Have]->(attr:Attr)
        RETURN sku.sku_id AS sku_id, sku.sku_name AS sku_name,
               c3.category3_name AS category3_name,
               tm.trademark_name AS trademark_name,
               collect(DISTINCT attr.attr_name + '=' + attr.attr_value) AS matched_attrs
        LIMIT {config.DEFAULT_RESULT_LIMIT}
        """


class CypherValidator:
    """Cypher 校验模块：本地做安全和语法校验，大模型做语义校验。"""

    def __init__(self, driver, llm: LLMClient, logger=None):
        self.driver = driver
        self.llm = llm
        self.logger = logger

    def validate(
        self,
        cypher: str,
        query: str = "",
        entry_nodes: List[Dict[str, Any]] | None = None,
        entities: List[Dict[str, Any]] | None = None,
        intent: Dict[str, Any] | None = None,
        use_llm_semantic: bool = True,
    ) -> List[str]:
        errors = self._local_validate(cypher)
        if not errors:
            errors.extend(self._intent_scope_validate(cypher, intent or {}))
        if not errors:
            errors.extend(self._schema_path_validate(cypher))
        if not errors:
            errors.extend(self._entity_constraint_validate(cypher, entities or [], query))
        if not errors:
            errors.extend(self._behavior_relation_validate(cypher, query, intent or {}))
        if not errors and use_llm_semantic:
            errors.extend(self._llm_semantic_validate(cypher, query, entry_nodes or [], intent or {}))
        if not errors and not use_llm_semantic and self.logger:
            self.logger.info("Cypher语义校验: 规则兜底查询跳过大模型语义校验，仅保留安全和语法校验")
        if self.logger:
            self.logger.info("cypher_validate_errors: %s", errors)
        return errors

    def _intent_scope_validate(self, cypher: str, intent: Dict[str, Any]) -> List[str]:
        intent_name = intent.get("intent")
        if intent_name == "user_interest_query":
            errors = []
            if not re.search(r":User\b", cypher or "", flags=re.I):
                errors.append("当前意图是 user_interest_query，Cypher 必须包含 User 节点。")
            if not re.search(r"\buser_id\s*:", cypher or "", flags=re.I):
                errors.append("当前意图是 user_interest_query，Cypher 必须包含指定 user_id 约束。")
            if not re.search(r":(View|Click|Favorite)\b", cypher or "", flags=re.I):
                errors.append("当前意图是 user_interest_query，Cypher 必须包含 View/Click/Favorite 用户行为关系。")
            if not re.search(r"\btype\s*\(\s*r\s*\)|\bbehaviors\b|\bbehavior_count\b", cypher or "", flags=re.I):
                errors.append("当前意图是 user_interest_query，Cypher 应返回 behaviors 或 behavior_count，便于确认结果来自用户行为。")
            return errors
        if re.search(r"(:User\b|\buser_id\b|:View\b|:Click\b|:Favorite\b)", cypher or "", flags=re.I):
            return ["当前意图不是 user_interest_query，Cypher 不允许使用 User 节点、user_id 或 View/Click/Favorite 用户行为关系。"]
        return []

    def _behavior_relation_validate(self, cypher: str, query: str, intent: Dict[str, Any]) -> List[str]:
        if intent.get("intent") != "user_interest_query":
            return []
        expected = build_behavior_pattern(query)
        if expected == "View|Click|Favorite":
            return []
        used = set()
        for rel_expr in re.findall(r"\[\s*\w*\s*:([A-Za-z|]+)", cypher or "", flags=re.I):
            for rel_type in rel_expr.split("|"):
                if rel_type.lower() in {"view", "click", "favorite"}:
                    used.add(rel_type.capitalize())
        if used and used != {expected}:
            return [f"用户行为关系不匹配: 当前问题应使用 {expected}，但 Cypher 使用了 {sorted(used)}"]
        return []

    def _schema_path_validate(self, cypher: str) -> List[str]:
        """检查 Cypher 中显式写出的节点关系是否符合当前电商图谱 schema。"""
        label_by_alias = self._extract_node_alias_labels(cypher)
        allowed_edges = {
            ("SKU", "Have", "Attr"),
            ("SKU", "Belong", "SPU"),
            ("SPU", "Belong", "Category3"),
            ("Category3", "Belong", "Category2"),
            ("Category2", "Belong", "Category1"),
            ("SPU", "Belong", "Trademark"),
            ("User", "View", "SKU"),
            ("User", "Click", "SKU"),
            ("User", "Favorite", "SKU"),
        }
        errors: List[str] = []
        for source_alias, rel_type, target_alias in self._extract_directed_edges(cypher):
            source_label = label_by_alias.get(source_alias)
            target_label = label_by_alias.get(target_alias)
            if not source_label or not target_label:
                continue
            edge = (source_label, rel_type, target_label)
            if edge not in allowed_edges:
                errors.append(
                    f"Cypher 关系路径不符合图谱 schema: ({source_label})-[:{rel_type}]->({target_label})。"
                    "正确关系包括 SKU-Have-Attr、SKU-Belong-SPU、SPU-Belong-Category3、"
                    "Category3-Belong-Category2、Category2-Belong-Category1、SPU-Belong-Trademark。"
                )
        return errors

    def _extract_node_alias_labels(self, cypher: str) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        for alias, label in re.findall(r"\(\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)", cypher or ""):
            labels[alias] = self._normalize_label(label)
        return labels

    def _extract_directed_edges(self, cypher: str) -> List[Tuple[str, str, str]]:
        edges: List[Tuple[str, str, str]] = []
        # 只匹配单个节点括号内的内容，避免跨 MATCH/EXISTS 片段误把不相邻节点连成一条边。
        node = r"\(\s*([A-Za-z_]\w*)(?:\s*:\s*[A-Za-z_]\w*)?[^()\n{}]*\)"
        rel = r"\[\s*:\s*([A-Za-z_]\w*)\s*\]"
        for left, rel_type, right in re.findall(node + r"\s*-\s*" + rel + r"\s*->\s*" + node, cypher or ""):
            edges.append((left, rel_type, right))
        for left, rel_type, right in re.findall(node + r"\s*<-\s*" + rel + r"\s*-\s*" + node, cypher or ""):
            edges.append((right, rel_type, left))
        return edges

    def _normalize_label(self, label: str) -> str:
        mapping = {
            "sku": "SKU",
            "spu": "SPU",
            "attr": "Attr",
            "category1": "Category1",
            "category2": "Category2",
            "category3": "Category3",
            "trademark": "Trademark",
            "user": "User",
        }
        return mapping.get(str(label or "").lower(), label)

    def _entity_constraint_validate(self, cypher: str, entities: List[Dict[str, Any]], query: str = "") -> List[str]:
        errors: List[str] = []
        for entity in entities:
            label = entity.get("label")
            text = str(entity.get("text") or "")
            if not text:
                continue
            if label == "User":
                if not self._user_constraint_covered(cypher, text):
                    errors.append(f"Cypher 缺少实体约束: User={text}")
                continue
            if label == "Category3" and not self._category_constraint_covered(cypher, text):
                errors.append(f"Cypher 缺少实体约束: {label}={text}")
            elif label == "Trademark" and not self._trademark_constraint_covered(cypher, text):
                errors.append(f"Cypher 缺少实体约束: {label}={text}")
            elif label == "Attr" and is_query_placeholder_value(text):
                continue
            elif label == "Attr" and not self._attr_constraint_covered(cypher, entity, query):
                errors.append(f"Cypher 缺少实体约束: Attr={text}")
        return errors

    def _user_constraint_covered(self, cypher: str, text: str) -> bool:
        if not text:
            return False
        value = re.escape(str(text))
        patterns = [
            rf"\(\s*\w*\s*:\s*User\s*\{{[^}}]*\buser_id\s*:\s*{value}\b",
            rf"\b\w+\s*\.\s*user_id\s*=\s*{value}\b",
            rf"\buser_id\s*:\s*{value}\b",
        ]
        return any(re.search(pattern, cypher or "", flags=re.I) for pattern in patterns)

    def _trademark_constraint_covered(self, cypher: str, text: str) -> bool:
        """品牌约束必须绑定在与 SPU 相连的 Trademark 节点上，避免孤立品牌 MATCH 造成假约束。"""
        if not text:
            return False
        label_by_alias = self._extract_node_alias_labels(cypher)
        connected_trademark_aliases = {
            target_alias
            for source_alias, rel_type, target_alias in self._extract_directed_edges(cypher)
            if rel_type == "Belong"
            and label_by_alias.get(source_alias) == "SPU"
            and label_by_alias.get(target_alias) == "Trademark"
        }
        if not connected_trademark_aliases:
            return False
        return any(self._alias_has_property_constraint(cypher, alias, "trademark_name", text) for alias in connected_trademark_aliases)

    def _alias_has_property_constraint(self, cypher: str, alias: str, property_name: str, value: str) -> bool:
        escaped_alias = re.escape(alias)
        escaped_property = re.escape(property_name)
        escaped_value = re.escape(value)
        inline_pattern = rf"\(\s*{escaped_alias}\s*:\s*\w+\s*\{{[^}}]*{escaped_property}\s*:\s*['\"]{escaped_value}['\"]"
        where_equal_pattern = rf"\b{escaped_alias}\s*\.\s*{escaped_property}\s*=\s*['\"]{escaped_value}['\"]"
        where_contains_pattern = rf"\b{escaped_alias}\s*\.\s*{escaped_property}\s+CONTAINS\s+['\"]{escaped_value}['\"]"
        return bool(
            re.search(inline_pattern, cypher or "")
            or re.search(where_equal_pattern, cypher or "")
            or re.search(where_contains_pattern, cypher or "")
        )

    def _category_constraint_covered(self, cypher: str, text: str) -> bool:
        """判断三级类目约束是否已覆盖，兼容“笔记本”覆盖具体笔记本子类的情况。"""
        if not text:
            return False
        if text in cypher:
            return True
        broad_terms = ["笔记本", "手机", "电视", "耳机", "冰箱", "洗衣机", "咖啡机", "口红", "香水"]
        for term in broad_terms:
            if term in text and term in cypher:
                return True
        return False

    def _attr_constraint_covered(self, cypher: str, entity: Dict[str, Any], query: str = "") -> bool:
        attr_name = str(entity.get("attr_name") or "")
        text = str(entity.get("text") or "")
        if attr_name in {"类别", "香水彩妆"} and self._category_constraint_covered(cypher, text):
            return True
        if not attr_name or not self._attr_name_covered(cypher, attr_name):
            return False
        if self._range_attr_constraint_covered(cypher, query, attr_name, text):
            return True
        candidates = {text}
        upper_text = text.upper()
        if upper_text.endswith("G") and not upper_text.endswith("GB"):
            candidates.add(text + "B")
            candidates.add(upper_text[:-1] + "GB")
        if text.endswith("英寸"):
            candidates.add(text.replace("英寸", "寸"))
        if text.endswith("寸"):
            candidates.add(text.replace("寸", "英寸"))
        number_match = re.search(r"\d+(?:\.\d+)?", text)
        if number_match:
            candidates.add(number_match.group(0))
        return any(candidate and candidate in cypher for candidate in candidates)

    def _attr_name_covered(self, cypher: str, attr_name: str) -> bool:
        """兼容 UIE schema 与图谱属性名称存在同义表达的情况。"""
        if attr_name in cypher:
            return True
        equivalent_names = {
            "尺码": ["屏幕尺寸"],
            "屏幕尺寸": ["尺码"],
        }
        return any(name in cypher for name in equivalent_names.get(attr_name, []))

    def _range_attr_constraint_covered(self, cypher: str, query: str, attr_name: str, text: str) -> bool:
        """范围查询中，>= 数值表达可覆盖实体字面值，例如 15英寸以上覆盖 15英寸。"""
        if not self._has_range_semantics(query):
            return False
        number_match = re.search(r"\d+(?:\.\d+)?", text or "")
        if not number_match:
            return False
        number = number_match.group(0)
        compact = re.sub(r"\s+", "", cypher or "").upper()
        name_ok = self._attr_name_covered(cypher, attr_name)
        if not name_ok:
            return False
        if attr_name in {"屏幕尺寸", "尺码", "内存", "分辨率"}:
            return bool(re.search(rf">=\s*{re.escape(number)}\b", compact))
        if attr_name == "硬盘":
            return text.upper() in compact or number in compact
        return False

    def _has_range_semantics(self, query: str) -> bool:
        return any(word in (query or "") for word in ["以上", "不小于", "大于", "至少", "不少于", "包括", "及以上"])

    def _local_validate(self, cypher: str) -> List[str]:
        errors: List[str] = []
        cleaned = (cypher or "").strip()
        if not cleaned:
            return ["Cypher 为空。"]
        if FORBIDDEN_CYPHER.search(cleaned):
            errors.append("Cypher 包含写入、删除或管理类危险操作。")
        if "embedding" in cleaned or "fulltext" in cleaned:
            errors.append("Cypher 不应返回 embedding/fulltext 内部索引字段。")
        if not errors:
            try:
                self.driver.execute_query("EXPLAIN " + cleaned, database_=config.NEO4J_DATABASE)
            except CypherSyntaxError as exc:
                errors.append(f"Cypher 语法错误: {exc}")
            except Exception as exc:
                errors.append(f"Cypher EXPLAIN 失败: {exc}")
        return errors

    def _llm_semantic_validate(self, cypher: str, query: str, entry_nodes: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[str]:
        if not self.llm.available:
            return []
        if self._is_notebook_spec_semantic_pass(cypher, query):
            if self.logger:
                self.logger.info("Cypher语义校验结果: 通过")
            return []
        prompt = f"""
你是 Neo4j Cypher 审核专家。请根据 schema、用户问题和入口节点检查 Cypher 的语义是否正确。
重点检查:
1. 节点标签、属性名、关系类型是否符合 schema。
2. 关系方向是否正确。
3. 是否遗漏用户问题中的关键约束。
4. 查询结果是否能够回答用户问题。
5. 是否误用 OPTIONAL MATCH 导致 WHERE 过滤无效。
6. 如果当前意图不是 user_interest_query，是否错误使用了 User 节点、user_id、View/Click/Favorite 用户行为关系。

重要原则:
1. 如果 Cypher 已经包含用户问题的主要约束，不要吹毛求疵。
2. 属性约束可以通过 EXISTS {{ MATCH (sku)-[:Have]->(:Attr {{attr_name, attr_value}}) }} 表达，这是正确写法。
3. (SKU)-[:Have]->(Attr)、(SKU)-[:Belong]->(SPU)、(SPU)-[:Belong]->(Category3)、(SPU)-[:Belong]->(Trademark) 都是正确方向。
4. 只有 intent=user_interest_query 才允许使用 User 节点、user_id、View/Click/Favorite 用户行为关系；其它意图必须围绕商品、品类、品牌和属性查询。
4.1 非 user_interest_query 仍然允许、而且经常需要使用 Attr 节点查询商品属性；例如“花生油”“保湿”“32GB内存”“4K电视”都应通过 (SKU)-[:Have]->(Attr) 约束。不要因为意图不是 user_interest_query 就否定 Attr 节点。
5. Category3 与 Attr(attr_name='类别' 或 '香水彩妆') 可表达同一类品类证据；例如 Cypher 已经约束 c3.category3_name CONTAINS '口红' 时，不要再判定缺少 Attr=口红。
6. 品类证据组内部可以 OR，例如 category3_name=口红 OR 香水彩妆=口红；但功效、颜色、内存等其它属性约束必须与品类证据组 AND。
7. 同一 attr_name 下多个候选值是 OR，例如 功效=保湿 或 功效=补水 可写成 a.attr_name='功效' AND a.attr_value IN ['保湿','补水']。
8. 只在确定无法回答问题、关系方向确实错误、遗漏核心约束、或存在明显 schema 错误时判定失败。

严格返回 JSON 对象，不要输出解释:
{{"valid": true, "errors": []}}
或
{{"valid": false, "errors": ["具体错误"]}}

schema:
{config.GRAPH_SCHEMA_TEXT}

用户问题:
{query}

意图:
{intent}

入口节点:
{entry_nodes}

Cypher:
{cypher}
"""
        try:
            raw_output = self.llm.invoke(prompt, task_name="Cypher语义校验", log_response=False)
            result = parse_json_object_or_list(raw_output)
            if isinstance(result, dict):
                if result.get("valid"):
                    if self.logger:
                        self.logger.info("Cypher语义校验结果: 通过")
                    return []
                filtered_errors = self._filter_semantic_errors(cypher, [str(item) for item in result.get("errors", [])])
                if self.logger:
                    if filtered_errors:
                        self.logger.info("Cypher语义校验结果: 发现有效问题 %s", to_json(filtered_errors))
                    else:
                        self.logger.info("Cypher语义校验结果: 通过")
                return filtered_errors
            if isinstance(result, list):
                filtered_errors = self._filter_semantic_errors(cypher, [str(item) for item in result])
                if self.logger:
                    if filtered_errors:
                        self.logger.info("Cypher语义校验结果: 发现有效问题 %s", to_json(filtered_errors))
                    else:
                        self.logger.info("Cypher语义校验结果: 通过")
                return filtered_errors
            return []
        except Exception as exc:
            if self.logger:
                self.logger.warning("Cypher语义校验失败: %s", exc)
            return []

    def _filter_semantic_errors(self, cypher: str, errors: List[str]) -> List[str]:
        """过滤大模型语义校验中的明显误报，本地安全校验仍然是最终底线。"""
        filtered = []
        uses_user_context = bool(re.search(r"(:User\b|\buser_id\b|:View\b|:Click\b|:Favorite\b)", cypher or "", flags=re.I))
        for error in errors:
            if any(token in error for token in ["关系方向错误", "关系方向是否正确"]) and "-[:Belong]->" in (cypher or ""):
                continue
            if "未使用入口节点" in error and re.search(r":Attr\b|attr_name|attr_value|EXISTS\s*\{", cypher or "", flags=re.I):
                continue
            if "节点标签、属性名、关系类型不符合 schema" in error and any(label in error for label in ["Category3", "Trademark", "Attr"]):
                continue
            if "遗漏用户问题中的关键约束" in error and "品牌" in error and "trademark_name" in (cypher or ""):
                continue
            if "查询结果是否能够回答用户问题" in error and "品牌" in error and "trademark_name" in (cypher or ""):
                continue
            if "遗漏用户问题中的关键约束" in error and "Attr" in error and re.search(r":Attr\b|attr_name|attr_value", cypher or "", flags=re.I):
                continue
            if "误用 OPTIONAL MATCH" in error and "EXISTS" in (cypher or ""):
                continue
            if "关系方向是否正确" in error and "spu" in error and "Category3" in error:
                continue
            if "Attr" in error and any(token in error for token in ["不是 user_interest_query", "非 user_interest_query", "不符合 schema", "不允许"]):
                continue
            if "关系方向" in error and "SPU" in error and "Trademark" in error:
                continue
            if "属性约束写法错误" in error and "toUpper" in error:
                continue
            if any(token in error for token in ["属性约束写法错误", "属性约束表达方式错误"]) and re.search(
                r"EXISTS\s*\{\s*MATCH\s*\(sku\)-\[:Have\]->\(a:Attr\).*toFloat",
                cypher or "",
                flags=re.I | re.S,
            ):
                continue
            if not uses_user_context and any(token in error for token in ["User", "user_id", "View", "Click", "Favorite", "用户行为"]):
                continue
            filtered.append(error)
        return filtered

    def _is_notebook_spec_semantic_pass(self, cypher: str, query: str) -> bool:
        """笔记本规格模板已由本地规则生成并通过 EXPLAIN，避免大模型误判数值范围写法。"""
        text = query or ""
        compact = cypher or ""
        return (
            "笔记本" in text
            and "EXISTS" in compact
            and "toFloat" in compact
            and "c3.category3_name CONTAINS '笔记本'" in compact
            and "a.attr_name = '内存'" in compact
            and "a.attr_name = '硬盘'" in compact
            and "a.attr_name = '分辨率'" in compact
        )


class GraphExecutor:
    """Neo4j 查询执行模块，统一清理 embedding/fulltext 等内部字段。"""

    def __init__(self, driver, logger=None):
        self.driver = driver
        self.logger = logger

    def execute(self, cypher: str) -> List[Dict[str, Any]]:
        records, summary, _ = self.driver.execute_query(cypher, database_=config.NEO4J_DATABASE)
        rows = [clean_record(record.data()) for record in records]
        if self.logger:
            self.logger.info("Neo4j查询结果: 结果数=%s 耗时ms=%s", len(rows), summary.result_available_after)
            self.logger.info("Neo4j查询记录: %s", to_json(rows[:20]))
        return rows


class ConversationMemory:
    """轻量级内存会话，保存最近几轮对话用于多轮追问。"""

    def __init__(self, max_rounds: int = 20):
        self.max_messages = max_rounds * 2
        self.messages: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def format_history(self, conversation_id: str) -> str:
        history = self.messages.get(conversation_id, [])
        return "\n".join(f"{item['role']}: {item['content']}" for item in history[-self.max_messages :])

    def append(self, conversation_id: str, role: str, content: str) -> None:
        self.messages[conversation_id].append({"role": role, "content": content})
        self.messages[conversation_id] = self.messages[conversation_id][-self.max_messages :]


class AnswerGenerator:
    """回复生成模块，把图谱查询结果和历史上下文交给大模型生成客服回复。"""

    def __init__(self, llm: LLMClient, logger=None):
        self.llm = llm
        self.logger = logger

    def generate(self, query: str, records: List[Dict[str, Any]], history: str = "", entities: List[Dict[str, Any]] | None = None) -> str:
        context = format_records(records)
        constraints = format_entity_constraints(entities or [])
        if self.llm.available:
            prompt = f"""
你是一个专业、可靠的中文电商客服助手。
请严格基于【图谱检索结果】回答用户问题，不要编造图谱中不存在的商品、价格、库存或优惠信息。
如果结果不足，请明确说明，并给出可继续追问的方向。
重要规则:
1. 如果【图谱检索结果】非空，表示 Neo4j 已经检索到满足【已应用查询约束】的候选商品，不要回答“没有找到”“没有明确提到”。
2. 单条结果里可能只展示商品、品牌、品类等字段；这时需要结合【已应用查询约束】说明这些商品满足用户问题中的属性条件。
3. 回答中优先汇总品牌，再列举代表性 SKU，避免重复商品名堆叠。

历史对话:
{history}

已应用查询约束:
{constraints}

图谱检索结果:
{context}

用户问题:
{query}

回答:
"""
            try:
                raw_output = self.llm.invoke(prompt, task_name="回答生成", log_response=False).strip()
                if self.logger:
                    self.logger.info("回答生成结果: %s", raw_output)
                if records and is_empty_answer(raw_output):
                    fallback = build_record_answer(query, records, entities or [])
                    if self.logger:
                        self.logger.warning("回答生成疑似误判为空，已切换为规则答案: %s", fallback)
                    return fallback
                return raw_output
            except Exception as exc:
                if self.logger:
                    self.logger.warning("answer_llm_failed: %s", exc)
        if not records:
            return "我没有在当前知识图谱中查询到足够相关的结果，可以换一个品牌、品类或具体参数再试。"
        return build_record_answer(query, records, entities or [])


GLOBAL_MEMORY = ConversationMemory()


def build_fulltext(text: str) -> str:
    """将中文文本转为适合 Neo4j fulltext 索引的分词文本。"""
    try:
        import jieba

        tokens = jieba.lcut(text or "")
    except Exception:
        tokens = re.findall(TOKEN_PATTERN, text or "")
    return " ".join(token.strip() for token in tokens if TOKEN_PATTERN.fullmatch(token.strip()))


def extract_cypher(text: str) -> str:
    """从大模型输出中提取 Cypher 代码块。"""
    block = re.search(r"```(?:cypher)?\s*(.*?)```", text, flags=re.I | re.S)
    return block.group(1).strip() if block else text.strip()


def parse_cypher_generation_output(text: str) -> Tuple[str, str]:
    """解析大模型返回的 Cypher 与中文含义说明，兼容 JSON 和纯 Cypher 两种输出。"""
    try:
        result = parse_json_object(text)
        cypher = extract_cypher(str(result.get("cypher") or ""))
        explanation = str(result.get("explanation") or "大模型未返回查询含义说明。").strip()
        if cypher:
            return cypher, explanation
    except Exception:
        pass
    cypher = extract_cypher(text)
    return cypher, "大模型返回了纯 Cypher，未提供查询含义说明。"


def normalize_cypher_for_compare(cypher: str) -> str:
    """压缩空白后比较 Cypher，避免格式差异影响日志说明判断。"""
    return re.sub(r"\s+", " ", cypher or "").strip()


def build_where_clause(entities: List[Dict[str, Any]]) -> str:
    """根据实体抽取结果构造兜底 Cypher 的 WHERE 条件。"""
    where_parts = []
    attr_groups: Dict[str, List[str]] = {}
    category_texts = [
        str(entity.get("text", "")).replace("'", "\\'")
        for entity in entities
        if entity.get("label") == "Category3" and entity.get("text")
    ]
    if category_texts:
        category_parts = [f"c3.category3_name CONTAINS '{text}'" for text in category_texts]
        where_parts.append("(" + " OR ".join(category_parts) + ")")
    for entity in entities:
        label = entity.get("label")
        text = str(entity.get("text", "")).replace("'", "\\'")
        if label == "Trademark":
            where_parts.append(f"tm.trademark_name CONTAINS '{text}'")
        elif label == "Category3":
            continue
        elif label == "SKU":
            where_parts.append(f"sku.sku_name CONTAINS '{text}'")
        elif label == "Attr":
            attr_name = entity.get("attr_name")
            if is_query_placeholder_value(text):
                continue
            if is_invalid_attr_value_pair(str(attr_name or ""), text):
                continue
            if attr_name in {"类别", "香水彩妆"} and category_value_covered_by_category_texts(text, category_texts):
                continue
            if attr_name:
                attr_groups.setdefault(str(attr_name), []).append(text)
            else:
                where_parts.append(f"EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_value CONTAINS '{text}' }}")
    for attr_name, values in attr_groups.items():
        unique_values = list(dict.fromkeys(values))
        if len(unique_values) == 1:
            where_parts.append(
                f"EXISTS {{ MATCH (sku)-[:Have]->(:Attr {{attr_name: '{attr_name}', attr_value: '{unique_values[0]}'}}) }}"
            )
        else:
            quoted_values = ", ".join(f"'{value}'" for value in unique_values)
            where_parts.append(
                f"EXISTS {{ MATCH (sku)-[:Have]->(a:Attr) WHERE a.attr_name = '{attr_name}' AND a.attr_value IN [{quoted_values}] }}"
            )
    return "WHERE " + " AND ".join(where_parts) if where_parts else ""


def is_query_placeholder_value(text: str) -> bool:
    """Return True when UIE extracts interrogative words as an attribute value."""
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return True
    exact_placeholders = {
        "\u591a\u5c11",
        "\u51e0",
        "\u591a\u5927",
        "\u4ec0\u4e48",
        "\u54ea\u4e9b",
        "\u4ec0\u4e48\u989c\u8272",
        "\u591a\u5c11\u5c3a\u5bf8",
        "\u4ec0\u4e48\u5c3a\u5bf8",
        "\u54ea\u4e9b\u5c3a\u5bf8",
        "\u4ec0\u4e48\u54c1\u724c",
        "\u4ec0\u4e48\u529f\u6548",
    }
    if normalized in exact_placeholders:
        return True
    interrogatives = ("\u4ec0\u4e48", "\u591a\u5c11", "\u54ea\u4e9b", "\u51e0")
    query_fields = ("\u989c\u8272", "\u5c3a\u5bf8", "\u54c1\u724c", "\u529f\u6548", "\u7c7b\u522b", "\u5206\u7c7b")
    return any(word in normalized for word in interrogatives) and any(field in normalized for field in query_fields)


def is_invalid_attr_value_pair(attr_name: str, text: str) -> bool:
    """过滤 UIE 偶发的跨 schema 误抽取，例如把护肤品抽成香水彩妆属性。"""
    name = str(attr_name or "").strip()
    value = re.sub(r"\s+", "", str(text or ""))
    if name == "香水彩妆" and value in {"护肤品", "面部护肤", "精华乳液", "面霜", "乳液"}:
        return True
    if name == "面部护肤" and value in {"口红", "彩妆", "香水"}:
        return True
    return False


def category_value_covered_by_category_texts(attr_value: str, category_texts: List[str]) -> bool:
    """判断“类别/香水彩妆”属性值是否已被三级类目约束覆盖，例如 平板电视 覆盖 电视。"""
    value = str(attr_value or "")
    if not value:
        return False
    for category_text in category_texts:
        category = str(category_text or "")
        if value and category and (value in category or category in value):
            return True
    return False


def build_behavior_pattern(query: str) -> str:
    """根据用户问题选择行为关系；关注/兴趣类问题默认覆盖浏览、点击、收藏。"""
    text = query or ""
    if "收藏" in text:
        return "Favorite"
    if "点击" in text:
        return "Click"
    if any(word in text for word in ["浏览", "看过"]):
        return "View"
    return "View|Click|Favorite"


def extract_first_number(text: str, pattern: str) -> float | None:
    """从文本中按正则提取第一个数字，兼容多个捕获组。"""
    match = re.search(pattern, text or "", flags=re.I)
    if not match:
        return None
    for item in match.groups():
        if item:
            return float(item)
    return None


def format_number_for_cypher(value: float) -> str:
    """把 2.0 这类数字转为 2，避免生成 2.0TB 这种不符合属性值的文本。"""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def clean_record(value: Any) -> Any:
    """递归清理查询结果中的内部字段。"""
    if isinstance(value, dict):
        return {key: clean_record(item) for key, item in value.items() if key not in {"embedding", "fulltext"}}
    if isinstance(value, list):
        return [clean_record(item) for item in value]
    return value


def format_records(records: List[Dict[str, Any]], max_rows: int = 20) -> str:
    """将 Neo4j 查询结果转成适合放入 Prompt 的文本。"""
    if not records:
        return "没有查询到相关图谱结果。"
    lines = []
    for index, row in enumerate(records[:max_rows], start=1):
        parts = [f"{key}={value}" for key, value in row.items() if value not in (None, "", [])]
        lines.append(f"{index}. " + "，".join(parts))
    return "\n".join(lines)


def format_retrieval_table(rows: List[Dict[str, Any]], max_rows: int = 20) -> str:
    """把混合检索候选节点格式化成日志表格，便于人工排查召回质量。"""
    if not rows:
        return "| 序号 | 类型 | 来源 | 分数 | 属性 |\n|---:|---|---|---:|---|\n| - | - | - | - | 无候选节点 |"
    table = ["| 序号 | 类型 | 来源 | 分数 | 属性 |", "|---:|---|---|---:|---|"]
    for index, row in enumerate(rows[:max_rows], start=1):
        score = row.get("score", 0.0)
        try:
            score_text = f"{float(score):.4f}"
        except (TypeError, ValueError):
            score_text = str(score)
        table.append(
            "| {index} | {label} | {source} | {score} | {props} |".format(
                index=index,
                label=escape_table_cell(row.get("label", "")),
                source=escape_table_cell(row.get("source", "")),
                score=score_text,
                props=escape_table_cell(format_properties_inline(row.get("properties") or {})),
            )
        )
    return "\n".join(table)


def format_properties_inline(properties: Dict[str, Any]) -> str:
    """把节点属性压缩成单行文本，过滤 embedding/fulltext 等内部字段。"""
    items = []
    for key, value in properties.items():
        if key in {"embedding", "fulltext"}:
            continue
        items.append(f"{key}={value}")
    return "；".join(items) if items else "-"


def escape_table_cell(value: Any) -> str:
    """避免 Markdown 表格中的换行和竖线破坏日志表格。"""
    return str(value).replace("\n", " ").replace("|", "\\|")


def format_cypher(cypher: str) -> str:
    """统一格式化日志中的 Cypher，保留可读性并去掉多余空行。"""
    lines = [line.rstrip() for line in (cypher or "").strip().splitlines()]
    return "\n".join(line for line in lines if line.strip())


def format_entity_constraints(entities: List[Dict[str, Any]]) -> str:
    """把实体抽取结果转成回答阶段可读的查询约束说明。"""
    if not entities:
        return "无显式实体约束。"
    parts = []
    for entity in entities:
        label = entity.get("label")
        text = entity.get("text")
        if not text:
            continue
        if label == "Attr":
            parts.append(f"{entity.get('attr_name') or '属性'}={text}")
        elif label == "Category3":
            parts.append(f"三级品类={text}")
        elif label == "Trademark":
            parts.append(f"品牌={text}")
        else:
            parts.append(f"{label}={text}")
    return "；".join(parts) if parts else "无显式实体约束。"


def is_empty_answer(answer: str) -> bool:
    """判断大模型是否在有检索结果时误答为无结果。"""
    text = answer or ""
    return any(pattern in text for pattern in ["没有找到", "没有查询到", "没有明确提到", "暂无相关", "未找到"])


def build_record_answer(query: str, records: List[Dict[str, Any]], entities: List[Dict[str, Any]], max_rows: int = 10) -> str:
    """根据图谱记录生成稳定的简短客服回答，用于大模型不可用或误判为空的情况。"""
    constraints = format_entity_constraints(entities)
    brand_names: List[str] = []
    seen_brands = set()
    seen_products = set()
    product_lines = []
    for row in records:
        brand = row.get("trademark_name") or row.get("brand_name")
        if brand and brand not in seen_brands:
            seen_brands.add(brand)
            brand_names.append(str(brand))
        sku_id = row.get("sku_id")
        sku_name = row.get("sku_name") or row.get("product_name")
        category3 = row.get("category3_name")
        if not sku_name:
            continue
        product_key = (sku_id, sku_name, brand)
        if product_key in seen_products:
            continue
        seen_products.add(product_key)
        line = f"- {sku_name}"
        if sku_id is not None:
            line += f"（sku_id={sku_id}）"
        if brand:
            line += f"，品牌：{brand}"
        if category3:
            line += f"，品类：{category3}"
        product_lines.append(line)
        if len(product_lines) >= max_rows:
            break
    brand_text = "、".join(brand_names) if brand_names else "图谱结果中的相关品牌"
    products = "\n".join(product_lines) if product_lines else format_records(records, max_rows=max_rows)
    return (
        f"查到了。当前图谱按“{constraints}”进行检索，匹配到的品牌包括：{brand_text}。\n"
        f"代表性商品如下：\n{products}\n"
        "这些结果来自知识图谱中的 SKU、品类、品牌和属性关系；价格、库存等信息当前图谱没有提供。"
    )


def parse_json_object(text: str) -> Dict[str, Any]:
    """从大模型输出中解析 JSON 对象。"""
    result = parse_json_object_or_list(text)
    if not isinstance(result, dict):
        raise ValueError("LLM output is not a JSON object.")
    return result


def parse_json_object_or_list(text: str) -> Any:
    """兼容大模型输出代码块或前后解释文字时的 JSON 解析。"""
    cleaned = text.strip()
    block = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.I | re.S)
    if block:
        cleaned = block.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(1))

