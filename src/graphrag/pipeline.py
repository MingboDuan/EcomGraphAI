import re
import time
from datetime import datetime
from typing import Any, Dict

from neo4j import GraphDatabase

from . import config
from .components import (
    AnswerGenerator,
    BgeEmbedding,
    ContextualQueryRewriter,
    CypherGenerator,
    CypherValidator,
    EntityExtractor,
    EntryNodeSelector,
    GLOBAL_MEMORY,
    GraphExecutor,
    HybridRetriever,
    IntentClassifier,
    LLMClient,
    QueryCorrector,
)
from .logger import save_trace, setup_logger
from .logger import to_json


def extract_explicit_user_id(query: str) -> int | None:
    """Extract an explicitly mentioned user id from the natural-language query."""
    match = re.search(r"(?:用户|user)\s*[:：]?\s*(\d+)", query or "", flags=re.I)
    return int(match.group(1)) if match else None


class GraphRAGPipeline:
    """电商 Graph RAG 主流程：纠错、意图识别、实体抽取、检索、Text2Cypher、校验、执行、生成回答。"""

    def __init__(self, logger=None):
        self.logger = logger or setup_logger()
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        self.driver.verify_connectivity()

        self.embedding = BgeEmbedding()
        self.llm = LLMClient(self.logger)
        self.corrector = QueryCorrector(enable_model=False, logger=self.logger)
        self.context_rewriter = ContextualQueryRewriter(self.llm, self.logger)
        self.intent_classifier = IntentClassifier(self.llm, self.logger)
        self.entity_extractor = EntityExtractor(driver=self.driver, logger=self.logger, enable_uie=True)
        self.retriever = HybridRetriever(self.driver, self.embedding, self.logger)
        self.entry_node_selector = EntryNodeSelector(self.llm, self.logger)
        self.cypher_generator = CypherGenerator(self.llm, self.logger)
        self.cypher_validator = CypherValidator(self.driver, self.llm, self.logger)
        self.executor = GraphExecutor(self.driver, self.logger)
        self.answer_generator = AnswerGenerator(self.llm, self.logger)

    def close(self) -> None:
        self.driver.close()

    def health(self) -> Dict[str, Any]:
        self.driver.verify_connectivity()
        return {
            "status": "ok",
            "neo4j": "connected",
            "embedding_model": str(config.BGE_MODEL_DIR),
            "llm_available": self.llm.available,
        }

    def run(self, query: str, conversation_id: str = "default", user_id: int | None = None) -> Dict[str, Any]:
        explicit_user_id = extract_explicit_user_id(query)
        if explicit_user_id is not None and explicit_user_id != user_id:
            self.logger.info("GraphRAG用户ID解析: 问题中显式用户ID=%s，覆盖传入user_id=%s", explicit_user_id, user_id)
            user_id = explicit_user_id
        trace_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.perf_counter()
        trace: Dict[str, Any] = {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "query": query,
        }
        step_status: Dict[str, Dict[str, Any]] = {}
        self.logger.info("GraphRAG流程开始: trace_id=%s 问题=%s 会话=%s 用户=%s", trace_id, query, conversation_id, user_id)

        history = GLOBAL_MEMORY.format_history(conversation_id)
        step_status["上下文记忆"] = {"success": True, "detail": "已读取历史上下文" if history else "无历史上下文"}
        self.logger.info("步骤1-上下文记忆: 成功 detail=%s", step_status["上下文记忆"]["detail"])

        corrected_query, corrected = self.corrector.correct(query)
        trace["corrected_query"] = corrected_query
        trace["query_corrected"] = corrected
        step_status["查询纠错"] = {"success": True, "detail": f"纠错后问题={corrected_query}; 是否变化={corrected}"}
        self.logger.info("步骤2-查询纠错: 成功 detail=%s", step_status["查询纠错"]["detail"])

        retrieval_query, context_rewritten, rewrite_reason = self.context_rewriter.rewrite(corrected_query, history, user_id)
        trace["retrieval_query"] = retrieval_query
        trace["context_rewritten"] = context_rewritten
        step_status["上下文问题改写"] = {
            "success": True,
            "detail": f"检索问题={retrieval_query}; 是否变化={context_rewritten}; reason={rewrite_reason}",
        }
        self.logger.info("步骤2.5-上下文问题改写: 成功 detail=%s", step_status["上下文问题改写"]["detail"])

        intent = self.intent_classifier.classify(retrieval_query, user_id, history)
        trace["intent"] = intent
        step_status["意图识别"] = {"success": bool(intent.get("intent")), "detail": to_json(intent)}
        self.logger.info("步骤3-意图识别: %s detail=%s", "成功" if step_status["意图识别"]["success"] else "失败", step_status["意图识别"]["detail"])

        use_user_context = intent.get("intent") == "user_interest_query"
        active_user_id = user_id if use_user_context else None
        entities = self.entity_extractor.extract(retrieval_query, active_user_id)
        trace["entities"] = entities
        trace["use_user_context"] = use_user_context
        step_status["实体抽取"] = {"success": bool(entities), "detail": f"实体数量={len(entities)}; 实体={to_json(entities)}"}
        self.logger.info("步骤4-实体抽取: %s detail=%s", "成功" if step_status["实体抽取"]["success"] else "失败", step_status["实体抽取"]["detail"])

        raw_entry_nodes = self.retriever.retrieve(entities, config.DEFAULT_TOP_K)
        trace["raw_entry_nodes"] = raw_entry_nodes[:20]
        step_status["混合检索"] = {"success": bool(raw_entry_nodes), "detail": f"召回节点数={len(raw_entry_nodes)}"}
        self.logger.info("步骤5-混合检索: %s detail=%s", "成功" if step_status["混合检索"]["success"] else "失败", step_status["混合检索"]["detail"])

        entry_nodes = self.entry_node_selector.select(retrieval_query, raw_entry_nodes, entities=entities)
        trace["entry_nodes"] = entry_nodes[:20]
        step_status["入口节点过滤"] = {"success": bool(entry_nodes), "detail": f"保留节点数={len(entry_nodes)}"}
        self.logger.info("步骤5.5-入口节点过滤: %s detail=%s", "成功" if step_status["入口节点过滤"]["success"] else "失败", step_status["入口节点过滤"]["detail"])

        cypher = self.cypher_generator.generate(retrieval_query, intent, entities, entry_nodes, active_user_id)
        step_status["Cypher生成"] = {"success": bool(cypher), "detail": "已生成，查询语句已写入日志"}
        self.logger.info("步骤6-Cypher生成: %s detail=%s", "成功" if step_status["Cypher生成"]["success"] else "失败", step_status["Cypher生成"]["detail"])
        errors = self.cypher_validator.validate(cypher, retrieval_query, entry_nodes, entities, intent=intent)
        self.logger.info("步骤7-Cypher校验: %s detail=%s", "成功" if not errors else "发现问题", to_json(errors))
        for _ in range(2):
            if not errors:
                break
            cypher = self.cypher_generator.correct(retrieval_query, cypher, errors, entry_nodes, intent, active_user_id)
            self.logger.info("步骤8-Cypher修正: 已执行，修正后查询语句已写入日志")
            errors = self.cypher_validator.validate(cypher, retrieval_query, entry_nodes, entities, intent=intent)
            self.logger.info("步骤8-Cypher修正后校验: %s detail=%s", "成功" if not errors else "仍有问题", to_json(errors))

        fallback_used = False
        if errors:
            self.logger.warning("Cypher校验多次未通过，启用规则兜底: %s", errors)
            cypher = self.cypher_generator.fallback(intent, entities, active_user_id, retrieval_query, entry_nodes)
            errors = self.cypher_validator.validate(
                cypher,
                retrieval_query,
                entry_nodes,
                entities,
                intent=intent,
                use_llm_semantic=False,
            )
            fallback_used = True
            self.logger.info("步骤9-规则兜底Cypher: %s detail=%s", "成功" if not errors else "失败", to_json(errors))

        step_status["Cypher校验"] = {"success": not errors, "detail": "校验通过" if not errors else to_json(errors)}

        records = []
        if errors:
            answer = "当前问题暂时无法生成可靠的图谱查询，请换一种方式描述需求。"
            step_status["Neo4j查询"] = {"success": False, "detail": "Cypher未通过校验，未执行查询"}
        else:
            try:
                records = self.executor.execute(cypher)
                if not records and not fallback_used:
                    self.logger.warning("Neo4j查询结果为空，启用规则兜底重新查询")
                    fallback_cypher = self.cypher_generator.fallback(intent, entities, active_user_id, retrieval_query, entry_nodes)
                    fallback_errors = self.cypher_validator.validate(
                        fallback_cypher,
                        retrieval_query,
                        entry_nodes,
                        entities,
                        intent=intent,
                        use_llm_semantic=False,
                    )
                    if not fallback_errors:
                        cypher = fallback_cypher
                        errors = fallback_errors
                        fallback_used = True
                        records = self.executor.execute(cypher)
                        self.logger.info("步骤10-空结果规则兜底: 成功 detail=重新查询结果数=%s", len(records))
                step_status["Neo4j查询"] = {"success": True, "detail": f"查询完成，结果数={len(records)}"}
                answer = self.answer_generator.generate(retrieval_query, records, history, entities)
                step_status["回答生成"] = {"success": bool(answer), "detail": "回答生成成功" if answer else "回答为空"}
                self.logger.info("步骤11-回答生成: %s", "成功" if answer else "失败")
            except Exception as exc:
                self.logger.exception("graph_query_failed")
                trace["execute_error"] = str(exc)
                answer = "图谱查询执行时出现异常，请稍后重试或换一种方式描述问题。"
                step_status["Neo4j查询"] = {"success": False, "detail": str(exc)}
                step_status["回答生成"] = {"success": False, "detail": "查询异常，生成兜底回答"}

        trace["cypher"] = cypher
        trace["cypher_errors"] = errors
        trace["cypher_rule_fallback_used"] = fallback_used
        trace["step_status"] = step_status
        trace["records"] = records[: config.DEFAULT_RESULT_LIMIT]
        trace["answer"] = answer
        trace["elapsed_seconds"] = round(time.perf_counter() - start_time, 4)
        trace_path = save_trace(trace_id, trace)
        trace["trace_path"] = str(trace_path)
        step_status["Trace保存"] = {"success": True, "detail": str(trace_path)}
        self.logger.info("步骤12-Trace保存: 成功 path=%s", trace_path)

        self.logger.info("========== GraphRAG执行汇总 ==========")
        for step_name, status in step_status.items():
            self.logger.info(
                "汇总-%s: %s detail=%s",
                step_name,
                "成功" if status.get("success") else "失败",
                status.get("detail", ""),
            )
        self.logger.info("========== GraphRAG执行结束 ==========")

        GLOBAL_MEMORY.append(conversation_id, "user", query)
        GLOBAL_MEMORY.append(conversation_id, "assistant", answer)

        self.logger.info(
            "trace_id=%s intent=%s entities=%s records=%s elapsed=%.4f",
            trace_id,
            intent.get("intent"),
            len(entities),
            len(records),
            trace["elapsed_seconds"],
        )
        return trace

