
from typing import Any, Dict, List, Optional

import agents as _agents




def get_client():
    return _agents.get_client()


def call_llm(
    system_prompt_template: str,
    shared_context: Dict[str, Any],
    model_name: str,
    extra_user_content: Optional[str] = None,
) -> Dict[str, Any]:
    return _agents.call_llm(
        system_prompt_template=system_prompt_template,
        shared_context=shared_context,
        model_name=model_name,
        extra_user_content=extra_user_content,
    )


def run_agent_a(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    return _agents.run_agent_a(shared_context)


def run_agent_b(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    return _agents.run_agent_b(shared_context)


def run_agent_c(
    shared_context: Dict[str, Any],
    agent_a_output: Dict[str, Any],
    agent_b_output: Dict[str, Any],
) -> Dict[str, Any]:
    return _agents.run_agent_c(
        shared_context=shared_context,
        agent_a_output=agent_a_output,
        agent_b_output=agent_b_output,
    )


def run_agent_d(
    shared_context: Dict[str, Any],
    agent_a_output: Dict[str, Any],
    agent_b_output: Dict[str, Any],
    agent_c_output: Dict[str, Any],
) -> Dict[str, Any]:
    return _agents.run_agent_d(
        shared_context=shared_context,
        agent_a_output=agent_a_output,
        agent_b_output=agent_b_output,
        agent_c_output=agent_c_output,
    )


def gen_agent_outputs(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    return _agents.gen_agent_outputs(shared_context)


def get_final_diagnoses(agent_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 Agent D 输出中取最终 top5。
    兼容传入完整 result 或 agent_outputs。
    """
    if not isinstance(agent_outputs, dict):
        return []

    final_output = agent_outputs.get("agent_d_final", {})

    if not isinstance(final_output, dict):
        return []

    diagnoses = final_output.get("top5_diagnoses", [])

    if isinstance(diagnoses, list):
        return diagnoses

    return []


def get_diagnostic_synthesis(agent_outputs: Dict[str, Any]) -> str:
    if not isinstance(agent_outputs, dict):
        return ""

    final_output = agent_outputs.get("agent_d_final", {})

    if not isinstance(final_output, dict):
        return ""

    return str(final_output.get("diagnostic_synthesis", "")).strip()


def _diagnosis_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("diagnosis", "")).strip()
    return str(item).strip()


def _extract_top5_names(top5: List[Dict[str, Any]]) -> List[str]:
    names = []
    for item in top5:
        name = _diagnosis_name(item)
        if name:
            names.append(name)
    return names




def _extract_quote_from_evidence_item(evidence_item: Any) -> str:
    if isinstance(evidence_item, str):
        return evidence_item.strip()

    if isinstance(evidence_item, dict):
        for key in [
            "quoted_text",
            "verbatim_case_text",
            "quote",
            "case_quote",
            "case_text",
            "text",
            "evidence",
        ]:
            value = evidence_item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

    return ""


def _extract_source_from_evidence_item(evidence_item: Any) -> str:
    if not isinstance(evidence_item, dict):
        return "agent_d_final.primary_supporting_evidence"

    for key in ["data_source", "source", "field", "case_field"]:
        value = evidence_item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return "agent_d_final.primary_supporting_evidence"


def _build_top5_evidence(
    top5: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    evidence_summary: List[Dict[str, Any]] = []

    for idx, diagnosis_item in enumerate(top5, start=1):
        if not isinstance(diagnosis_item, dict):
            continue

        evidence_items = diagnosis_item.get("primary_supporting_evidence", [])
        if not isinstance(evidence_items, list):
            evidence_items = [evidence_items]

        normalized_evidence = []
        for evidence_item in evidence_items:
            quote = _extract_quote_from_evidence_item(evidence_item)
            if not quote:
                continue

            normalized_evidence.append(
                {
                    "quote": quote,
                    "source": _extract_source_from_evidence_item(evidence_item),
                }
            )

        evidence_summary.append(
            {
                "rank": diagnosis_item.get("rank", idx),
                "diagnosis": _diagnosis_name(diagnosis_item),
                "evidence_quote_count": len(normalized_evidence),
                "evidence": normalized_evidence,
                "adjudication_reasoning": str(diagnosis_item.get("adjudication_reasoning", "")).strip(),
            }
        )

    return evidence_summary


def build_compact_result_view(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    agent_outputs = {
        "agent_a": result.get("agent_a"),
        "agent_b": result.get("agent_b"),
        "agent_c": result.get("agent_c"),
        "agent_d_final": result.get("agent_d_final"),
    }

    top5 = get_final_diagnoses(agent_outputs)
    top5_names = _extract_top5_names(top5)
    kg_retrieved = result.get("kg_retrieved_diseases", [])
    if not isinstance(kg_retrieved, list):
        kg_retrieved = []

    compact = {
        "patient_id": result.get("patient_id", "unknown"),
        "status": result.get("status", "unknown"),
        "case_mode": result.get("case_mode", ""),
        "kg_retrieved_diseases": kg_retrieved,
        "kg_retrieved_disease_ids": [
            str(item.get("disease_id", "")).strip()
            for item in kg_retrieved
            if isinstance(item, dict) and str(item.get("disease_id", "")).strip()
        ],
        "diagnostic_synthesis": get_diagnostic_synthesis(agent_outputs),
        "top5_diagnoses": top5,
        "top5_diagnosis_names": top5_names,
        "top5_evidence": _build_top5_evidence(top5),
        "top_diagnoses": top5,
        "top1_diagnosis_name": top5_names[0] if top5_names else "",
    }

    if result.get("status") == "error":
        compact["failed_stage"] = result.get("failed_stage", "")
        compact["error_type"] = result.get("error_type", "")
        compact["error_message"] = result.get("error_message", "")

    return compact


def get_top_diagnosis(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    compact = build_compact_result_view(result)
    diagnoses = compact.get("top5_diagnoses", [])

    if diagnoses:
        return diagnoses[0]

    return None


__all__ = [
    "get_client",
    "call_llm",
    "run_agent_a",
    "run_agent_b",
    "run_agent_c",
    "run_agent_d",
    "gen_agent_outputs",
    "get_final_diagnoses",
    "get_diagnostic_synthesis",
    "build_compact_result_view",
    "get_top_diagnosis",
]
