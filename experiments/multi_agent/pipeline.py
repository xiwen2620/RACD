import logging
import time
import traceback
from threading import Lock
from typing import Any, Dict

from config import CONFIG, validate_config
from agents import gen_agent_outputs

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
    if not isinstance(item, dict):
        return "unknown"
    for key in ["patient_id", "id", "Patient ID", "ID", "subject_id", "hadm_id"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def build_shared_context(item):
    if not isinstance(item, dict):
        raise TypeError("Dataset row must be a dictionary")

    combined_text = item.get("combined_text", "")

    if not isinstance(combined_text, str) or not combined_text.strip():
        raise ValueError("Dataset row must contain a nonempty combined_text")

    return {
        "patient_id": safe_get_patient_id_from_item(item),
        "case": {
            "combined_text": combined_text
        },
        "kg_context": "",
        "kg_debug": {},
    }

def run_agents_with_retry(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    max_attempts = max(1, int(CONFIG.get("MAX_RETRIES", 3)))
    retry_delay = max(0, float(CONFIG.get("RETRY_DELAY", 5)))
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return gen_agent_outputs(shared_context)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Agent call failed for patient_id=%s, attempt=%d/%d: %s",
                shared_context.get("patient_id", "unknown"),
                attempt,
                max_attempts,
                exc,
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
) -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "status": "error",
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "case_mode": "raw_combined_text",
        "kg_retrieved_diseases": [],
        "agent_a": None,
        "agent_b": None,
        "agent_c": None,
        "agent_d_final": None,
        "_internal_raw": {
            "input_item": item,
            "kg_context": "",
            "kg_debug": {},
            "traceback": traceback.format_exc(),
        },
    }


def process_single_patient(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    No-KG ablation pipeline:
      raw item -> combined_text -> Agent A -> B -> C -> D -> result

    Directly use combined_text from the input dataset.
    No case_processing or KG module is called.
    """
    stage = "start"
    patient_id = safe_get_patient_id_from_item(item)

    try:
        ensure_runtime_validated()

        # 1. 直接从原始数据建立共享上下文
        stage = "build_shared_context"
        shared_context = build_shared_context(item)

        # 2. 调用四个 Agent
        stage = "run_agents"
        agent_outputs = run_agents_with_retry(shared_context)

        if not isinstance(agent_outputs, dict):
            raise TypeError(
                f"gen_agent_outputs() must return dict, got {type(agent_outputs)}"
            )

        # 3. 构建最终结果
        stage = "build_result"

        return {
            "patient_id": patient_id,
            "status": "success",
            "case_mode": "raw_combined_text",
            "kg_retrieved_diseases": [],
            "agent_a": agent_outputs.get("agent_a"),
            "agent_b": agent_outputs.get("agent_b"),
            "agent_c": agent_outputs.get("agent_c"),
            "agent_d_final": agent_outputs.get("agent_d_final"),
            "_internal_raw": {
                "input_item": item,
                "kg_context": "",
                "kg_debug": {},
                "shared_context": shared_context,
                "agent_outputs": agent_outputs,
            },
        }

    except Exception as exc:
        logger.exception(
            "Failed to process patient_id=%s at stage=%s",
            patient_id,
            stage,
        )

        return make_error_result(
            patient_id=patient_id,
            stage=stage,
            error=exc,
            item=item,
        )