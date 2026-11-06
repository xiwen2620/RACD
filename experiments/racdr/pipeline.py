
import logging
import time
import traceback
from threading import Lock
from typing import Any, Dict, List, Tuple

from config import CONFIG, validate_config
from case_processing import build_structured_case

try:
    from kg_retrieval import retrieve_kg_with_debug, retrieve_kg
except ImportError:
    from kg_retrieval import retrieve_kg

    def retrieve_kg_with_debug(structured_case: Dict[str, Any], top_k: int = 3):
        kg_context = retrieve_kg(structured_case, top_k=top_k)
        return kg_context, {
            "retrieved_kgs": [],
            "warning": "retrieve_kg_with_debug not available; using retrieve_kg fallback.",
        }


try:
    from agents import gen_agent_outputs
except ImportError:
    from results import gen_agent_outputs


logger = logging.getLogger(__name__)


_CONFIG_VALIDATED = False
_CONFIG_VALIDATION_LOCK = Lock()


def ensure_runtime_validated() -> None:
    global _CONFIG_VALIDATED
    if _CONFIG_VALIDATED:
        return
    with _CONFIG_VALIDATION_LOCK:
        if _CONFIG_VALIDATED:
            return
        validate_config()
        _CONFIG_VALIDATED = True


def safe_get_patient_id_from_item(item: Dict[str, Any]) -> str:
    """
    从原始 item 中尽量提取 patient_id。
    这个函数用于 build_structured_case 失败时，仍能返回一个可识别的错误结果。
    """
    if not isinstance(item, dict):
        return "unknown"

    for key in ["patient_id", "id", "Patient ID", "ID", "subject_id", "hadm_id"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return "unknown"


def safe_get_patient_id_from_structured_case(structured_case: Dict[str, Any]) -> str:
    """
    从 structured_case 中提取 patient_id。
    """
    try:
        patient = structured_case.get("patient", {})
        patient_id = patient.get("patient_id", "unknown")
        if patient_id is None or not str(patient_id).strip():
            return "unknown"
        return str(patient_id).strip()
    except Exception:
        return "unknown"


def summarize_retrieved_kgs(kg_debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把 KG debug 结果压缩成汇总信息，避免 summary 文件过大。
    完整 matched_terms 会保存在 _internal_raw["kg_debug"] 中。
    """
    retrieved = kg_debug.get("retrieved_kgs", []) if isinstance(kg_debug, dict) else []

    summary = []

    for item in retrieved:
        matched_terms = item.get("matched_terms", {}) or {}

        matched_signal_count = 0
        positive_or_uncertain_count = 0
        negative_or_historical_count = 0

        graph_edge_count = len(matched_terms.get("graph_edges", []) or [])
        clinical_fact_count = len(matched_terms.get("clinical_facts", []) or [])

        for group_name, matches in matched_terms.items():
            if group_name == "graph_edges":
                continue
            if not isinstance(matches, list):
                continue

            matched_signal_count += len(matches)

            for match in matches:
                polarity = match.get("polarity", "")
                if polarity in ["positive", "uncertain"]:
                    positive_or_uncertain_count += 1
                elif polarity in ["negative", "historical"]:
                    negative_or_historical_count += 1

        summary.append(
            {
                "file_name": item.get("file_name", ""),
                "disease_id": item.get("disease_id", ""),
                "disease_name_en": item.get("disease_name_en", ""),
                "score": item.get("score", 0),
                "matched_signal_count": matched_signal_count,
                "positive_or_uncertain_signal_count": positive_or_uncertain_count,
                "negative_or_historical_signal_count": negative_or_historical_count,
                "missing_critical_tests": item.get("missing_critical_tests", []),
                "clinical_fact_match_count": clinical_fact_count,
                "graph_edge_path_count": graph_edge_count,
                "force_recall": item.get("force_recall", False),
                "min_score_used": item.get("min_score_used", None),
            }
        )

    return summary


def build_shared_context(
    structured_case: Dict[str, Any],
    kg_context: str,
    kg_debug: Dict[str, Any],
) -> Dict[str, Any]:
    """
    构建传给 Agent 的共享上下文。
    注意：
    - Agent 主要读取 case 和 kg_context。
    - kg_debug 也传入，后续如果 prompts 或 agents 想用结构化 KG 信息，也可以直接使用。
    """
    return {
        "patient_id": safe_get_patient_id_from_structured_case(structured_case),
        "case": structured_case,
        "kg_context": kg_context,
        "kg_debug": kg_debug,
    }


def run_agents_with_retry(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    带重试的 Agent 调用。
    避免单次 API 抖动导致整个批处理崩掉。
    """
    max_attempts = max(1, int(CONFIG.get("MAX_RETRIES", 3)))
    retry_delay = max(0, float(CONFIG.get("RETRY_DELAY", 5)))

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return gen_agent_outputs(shared_context)

        except Exception as e:
            last_error = e
            patient_id = shared_context.get("patient_id", "unknown")

            logger.warning(
                "Agent call failed for patient_id=%s, attempt=%d/%d: %s",
                patient_id,
                attempt,
                max_attempts,
                e,
            )

            if attempt < max_attempts and retry_delay > 0:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Agent calls failed after {max_attempts} attempts: {last_error}"
    ) from last_error


def make_error_result(
    patient_id: str,
    stage: str,
    error: Exception,
    item: Dict[str, Any] | None = None,
    structured_case: Dict[str, Any] | None = None,
    kg_context: str | None = None,
    kg_debug: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    构建失败病例结果。
    这样单个病例失败不会直接中断整个批处理。
    """
    return {
        "patient_id": patient_id,
        "status": "error",
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "case_mode": CONFIG.get("USE_CASE_MODE", "default"),
        "kg_retrieved_diseases": summarize_retrieved_kgs(kg_debug or {}),
        "_internal_raw": {
            "input_item": item,
            "structured_case": structured_case,
            "kg_context": kg_context,
            "kg_debug": kg_debug,
            "traceback": traceback.format_exc(),
        },
    }


def process_single_patient(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    单病例处理入口。

    main.py 会调用这个函数：
        process_single_patient(item)

    流程：
    1. 原始数据 item -> structured_case
    2. structured_case -> KG retrieval
    3. case + KG context -> Agent A/B/C/D
    4. 返回病例级结果
    """
    stage = "start"
    patient_id = safe_get_patient_id_from_item(item)

    structured_case = None
    kg_context = None
    kg_debug = None

    try:
        ensure_runtime_validated()

        stage = "build_structured_case"
        structured_case = build_structured_case(item)
        patient_id = safe_get_patient_id_from_structured_case(structured_case)

        stage = "retrieve_kg"
        kg_top_k = int(CONFIG.get("KG_TOP_K", 3))

        kg_context, kg_debug = retrieve_kg_with_debug(
            structured_case,
            top_k=kg_top_k,
        )

        stage = "build_shared_context"
        shared_context = build_shared_context(
            structured_case=structured_case,
            kg_context=kg_context,
            kg_debug=kg_debug,
        )

        stage = "run_agents"
        agent_outputs = run_agents_with_retry(shared_context)

        if not isinstance(agent_outputs, dict):
            raise TypeError(
                f"gen_agent_outputs() must return dict, got {type(agent_outputs)}"
            )

        stage = "build_result"

        result = {
            "patient_id": patient_id,
            "status": "success",
            "case_mode": structured_case.get("case_mode", CONFIG.get("USE_CASE_MODE", "default")),
            "kg_retrieved_diseases": summarize_retrieved_kgs(kg_debug or {}),
            "agent_a": agent_outputs.get("agent_a"),
            "agent_b": agent_outputs.get("agent_b"),
            "agent_c": agent_outputs.get("agent_c"),
            "agent_d_final": agent_outputs.get("agent_d_final"),
            "_internal_raw": {
                "input_item": item,
                "structured_case": structured_case,
                "kg_context": kg_context,
                "kg_debug": kg_debug,
                "shared_context": shared_context,
                "agent_outputs": agent_outputs,
            },
        }

        return result

    except Exception as e:
        logger.exception(
            "Failed to process patient_id=%s at stage=%s",
            patient_id,
            stage,
        )

        return make_error_result(
            patient_id=patient_id,
            stage=stage,
            error=e,
            item=item,
            structured_case=structured_case,
            kg_context=kg_context,
            kg_debug=kg_debug,
        )
