from typing import Any, Dict

def build_structured_case(
    item: Dict[str, Any]
) -> Dict[str, Any]:

    patient_id = str(item["patient_id"]).strip()
    combined_text = item["combined_text"]

    if not patient_id:
        raise ValueError("patient_id 不能为空")

    if not isinstance(combined_text, str) or not combined_text.strip():
        raise ValueError("combined_text 不能为空")

    return {
        "case_mode": "precleaned",
        "patient": {
            "patient_id": patient_id
        },
        "source_text": {
            "case_report": combined_text
        },
        "normalized_summary": ""
    }