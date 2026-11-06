import json
from typing import Any, Dict, Optional

from openai import OpenAI

from config import CONFIG
from prompts import (
    AGENT_A_PROMPT,
    AGENT_B_PROMPT,
    AGENT_C_PROMPT,
    AGENT_D_FINAL_PROMPT,
)
from utils import extract_json_object

def get_client() -> OpenAI:
    return OpenAI(
        api_key=CONFIG["HARDCODED_API_KEY"],
        base_url=CONFIG["API_BASE"],#从config中读取应变吗API KEY和对应参数
    )


def safe_json_dumps(obj: Any) -> str:
    # 开始异常保护块，因为 json.dumps 遇到不可序列化对象时可能抛出 TypeError。
    try:
        # 将 obj 转为 JSON，保留原字符，并使用两空格缩进生成更易读的结构。
        return json.dumps(obj, ensure_ascii=False, indent=2)
    # 只捕获类型不可序列化导致的 TypeError，其他异常仍会向上抛出，避免隐藏未知问题。
    except TypeError:
        # 当标准 JSON 序列化失败时，用对象的字符串表示兜底，确保函数仍返回 str。
        return str(obj)


def build_case_content(shared_context: Dict[str, Any]) -> str:
    case = shared_context.get("case", {})

    if not isinstance(case, dict):
        raise TypeError("shared_context['case'] must be a dict")

    combined_text = case.get("combined_text", "")

    if not isinstance(combined_text, str) or not combined_text.strip():
        raise ValueError("combined_text is empty or not a string")

    return "CASE_TEXT:\n" + combined_text


def build_user_content(
    shared_context: Dict[str, Any],
    extra_user_content: Optional[str] = None,
) -> str:
    patient_id = shared_context.get("patient_id", "")
    parts = []

    if patient_id:
        parts.append(f"PATIENT_ID:\n{patient_id}")

    parts.append(build_case_content(shared_context))

    if extra_user_content:
        parts.append(
            "ADDITIONAL_CONTEXT:\n"
            f"{extra_user_content}"
        )

    return "\n\n===\n\n".join(parts)


def build_system_prompt(system_prompt_template: str, shared_context: Dict[str, Any]) -> str:
    """Return the agent prompt unchanged; this ablation has no KG injection."""
    return system_prompt_template


def call_llm(
    system_prompt_template: str,
    shared_context: Dict[str, Any],
    model_name: str,
    extra_user_content: Optional[str] = None,
) -> Dict[str, Any]:
    client = get_client()

    system_prompt = build_system_prompt(
        system_prompt_template=system_prompt_template,
        shared_context=shared_context,
    )

    user_content = build_user_content(
        shared_context=shared_context,
        extra_user_content=extra_user_content,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return extract_json_object(content)



def run_agent_a(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    return call_llm(
        system_prompt_template=AGENT_A_PROMPT,
        shared_context=shared_context,
        model_name=CONFIG["AGENT_MODELS"]["agent_a"],
    )


def run_agent_b(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    return call_llm(
        system_prompt_template=AGENT_B_PROMPT,
        shared_context=shared_context,
        model_name=CONFIG["AGENT_MODELS"]["agent_b"],
    )

def run_agent_c(
    shared_context: Dict[str, Any],
    agent_a_output: Dict[str, Any],
    agent_b_output: Dict[str, Any],
) -> Dict[str, Any]:
    extra_context = (
        "AGENT_A_OUTPUT_JSON:\n"
        f"{safe_json_dumps(agent_a_output)}\n\n"
        "AGENT_B_OUTPUT_JSON:\n"
        f"{safe_json_dumps(agent_b_output)}"
    )

    return call_llm(
        system_prompt_template=AGENT_C_PROMPT,
        shared_context=shared_context,
        model_name=CONFIG["AGENT_MODELS"]["agent_c"],
        extra_user_content=extra_context,
    )


def run_agent_d(
    shared_context: Dict[str, Any],
    agent_a_output: Dict[str, Any],
    agent_b_output: Dict[str, Any],
    agent_c_output: Dict[str, Any],
) -> Dict[str, Any]:
    extra_context = (
        "AGENT_A_OUTPUT_JSON:\n"
        f"{safe_json_dumps(agent_a_output)}\n\n"
        "AGENT_B_OUTPUT_JSON:\n"
        f"{safe_json_dumps(agent_b_output)}\n\n"
        "AGENT_C_OUTPUT_JSON:\n"
        f"{safe_json_dumps(agent_c_output)}"
    )

    return call_llm(
        system_prompt_template=AGENT_D_FINAL_PROMPT,
        shared_context=shared_context,
        model_name=CONFIG["AGENT_MODELS"]["agent_d_final"],
        extra_user_content=extra_context,
    )


def gen_agent_outputs(shared_context: Dict[str, Any]) -> Dict[str, Any]:
    agent_a = run_agent_a(shared_context)
    agent_b = run_agent_b(shared_context)
    agent_c = run_agent_c(shared_context, agent_a, agent_b)
    agent_d_final = run_agent_d(shared_context, agent_a, agent_b, agent_c)

    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "agent_c": agent_c,
        "agent_d_final": agent_d_final,
        "kg_context": "",
    }
