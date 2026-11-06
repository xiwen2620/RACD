import os
import json
import re
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = CURRENT_DIR.parent
RACDR_DIR = EXPERIMENTS_DIR / "racdr"

if not RACDR_DIR.is_dir():
    raise FileNotFoundError(f"RACDR directory not found: {RACDR_DIR}")

sys.path.insert(0, str(RACDR_DIR))


import pandas as pd
from openai import OpenAI
from tqdm import tqdm


try:
    from case_processing import build_structured_case
except ImportError as exc:
    raise ImportError(
        "ImportError"
    ) from exc


try:
    from results import build_compact_result_view
except ImportError as exc:
    raise ImportError(
        "ImportError"
    ) from exc


try:
    from config import CONFIG, validate_kg_files
    from kg_retrieval import retrieve_kg_with_debug
    from pipeline import build_shared_context, summarize_retrieved_kgs
    from agents import (
        build_user_content as build_agent_user_content,
        build_system_prompt as build_agent_system_prompt,
    )
    from prompts import KG_EXTENSION
except ImportError as exc:
    raise ImportError(
        "ImportError"
    ) from exc



API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://YOUR_API_ENDPOINT/v1")

DEFAULT_MODEL_NAME = "gpt-5.5"
DEFAULT_DATA_PATH = "/PATH/TO/YOUR/mimic6000.csv"
DEFAULT_OUTPUT_PATH = "./results/base_gpt-5.5_kg_clinical_results.json"
DEFAULT_FORK_NUMS = 25
DEFAULT_SAMPLE_SIZE = None
DEFAULT_TEMPERATURE = 0.1  # 与主系统 agents.py 保持一致

GOLD_OR_REFERENCE_KEYS = [
    "expected_kg",
    "expected_kgs",
    "gold_kg",
    "gold_kgs",
    "expected_disease_id",
    "expected_disease_ids",
    "gold_disease_id",
    "gold_disease_ids",
    "final_diagnosis",
    "Final Diagnosis",
    "discharge_diagnosis",
    "Discharge Diagnosis",
    "diagnosis",
    "Diagnosis",
    "primary_diagnosis",
    "Primary Diagnosis",
]




SYSTEM_PROMPT_TEMPLATE = r"""
You are a senior attending physician performing a clinical diagnostic task.

IMPORTANT EXPERIMENTAL CONDITION:
- You are the ONLY diagnostic model in this baseline.
- You DO have access to retrieved clinical knowledge graph context.
- You do NOT have access to outputs from other diagnostic agents.
- Use the KG as clinical prior/reference information, but base final diagnoses and quoted evidence on the clinical case.

Your task is to produce a final ranked differential diagnosis.

### EVIDENCE RULES
1. Every diagnosis included in top5_diagnoses must be supported by at least one explicit finding in the provided clinical case.
2. primary_supporting_evidence MUST contain exact, verbatim excerpts copied from the provided case.
3. Do not paraphrase, summarize, rewrite, translate, normalize, or invent evidence.
4. Do not use external facts or KG content as if they were findings present in the case.
5. Missing information must remain missing; do not assume it is present.
6. If a diagnosis cannot be supported by at least one verbatim case quote, do not include it.

### RANKING RULES
1. Rank diagnoses by overall clinical plausibility based on the available case evidence.
2. Consider both clinical fit and the importance of serious/time-sensitive diagnoses when the case evidence supports them.
3. Retrieved KG diseases may be prioritized for consideration, but the KG is not a fixed answer list.
4. Rank 1 is the highest-priority diagnosis.
5. Return up to 5 diagnoses.
6. If fewer than 5 diagnoses remain supportable by verbatim case evidence, return fewer than 5 rather than inventing diagnoses or evidence.

""" + KG_EXTENSION + r"""


### OUTPUT REQUIREMENTS
Return valid JSON only.
Do not include Markdown, comments, or text outside the JSON object.

Use exactly this schema:
{{
  "agent": "agent_d",
  "role": "final_diagnostic_adjudicator",
  "diagnostic_synthesis": "concise overall synthesis based on the clinical case and retrieved KG reference",
  "top5_diagnoses": [
    {{
      "rank": 1,
      "diagnosis": "diagnosis name",
      "primary_supporting_evidence": [
        "Exact verbatim quote 1",
        "Exact verbatim quote 2"
      ],
      "adjudication_reasoning": "brief explanation of why this diagnosis is ranked here after reconciling case evidence with KG guidance"
    }}
  ]
}}
""".strip()



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-GPT + KG baseline using the same prospective preprocessing, "
            "KG retrieval, KG prompt integration, and final output structure as the full system."
        )
    )

    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Final compact summary JSON path. The file is a JSON list, matching clinical_results_v2.json.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory for full per-case JSON files. Default: output_path without .json suffix.",
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api_key", type=str, default=API_KEY)
    parser.add_argument("--base_url", type=str, default=BASE_URL)
    parser.add_argument("--fork_nums", type=int, default=DEFAULT_FORK_NUMS)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Use the first N records. Use 0 or a negative value to run all records.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Default 0.1 to match the full system.",
    )

    return parser.parse_args()



def clean_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
        records = df.to_dict(orient="records")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str).fillna("")
        records = df.to_dict(orient="records")
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records.")
        records = data
    elif suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL line {line_no} must be a JSON object.")
                records.append(record)
    else:
        raise ValueError(f"Unsupported input file format: {suffix}")

    normalized: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records):
        if not isinstance(raw, dict):
            continue
        item = {str(k): clean_scalar(v) for k, v in raw.items()}

        # 只做兼容补充，不覆盖原字段。
        if not str(item.get("sex", "")).strip() and str(item.get("gender", "")).strip():
            item["sex"] = item.get("gender", "")

        if not any(str(item.get(k, "")).strip() for k in ["patient_id", "id", "Patient ID", "ID", "subject_id", "hadm_id"]):
            item["patient_id"] = str(idx)

        normalized.append(item)

    return normalized


def extract_patient_id(item: Dict[str, Any], fallback: Any = "unknown") -> str:
    for key in ["patient_id", "id", "Patient ID", "ID", "subject_id", "hadm_id"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback)


def sanitize_filename(value: Any) -> str:
    text = str(value or "unknown").strip() or "unknown"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120] or "unknown"


def save_json_atomic(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)



def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model output")

    cleaned = str(text).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    if "```" in cleaned:
        for part in cleaned.split("```"):
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate.startswith("{"):
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(cleaned)):
            char = cleaned[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start: idx + 1]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                    break

    raise ValueError("Unable to parse model output as a JSON object")


def normalize_agent_d_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        parsed = {}

    top5 = parsed.get("top5_diagnoses", [])
    if not isinstance(top5, list):
        top5 = []

    normalized_items: List[Dict[str, Any]] = []
    for pos, item in enumerate(top5, start=1):
        if not isinstance(item, dict):
            continue

        diagnosis = str(item.get("diagnosis", "") or "").strip()
        if not diagnosis:
            continue

        evidence = item.get("primary_supporting_evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        elif not isinstance(evidence, list):
            evidence = []

        evidence = [str(x).strip() for x in evidence if str(x).strip()]

        try:
            rank = int(item.get("rank", pos))
        except Exception:
            rank = pos

        try:
            confidence_score = float(item.get("confidence_score"))
            if not (0.0 <= confidence_score <= 1.0):
                confidence_score = None
        except (TypeError, ValueError):
            confidence_score = None

        normalized_items.append({
            "rank": rank,
            "diagnosis": diagnosis,
            "confidence_score": confidence_score,
            "primary_supporting_evidence": evidence,
            "adjudication_reasoning": str(item.get("adjudication_reasoning", "") or "").strip(),
        })


    normalized_items.sort(key=lambda x: x.get("rank", 999))
    normalized_items = normalized_items[:5]

    for idx, item in enumerate(normalized_items, start=1):
        item["rank"] = idx

    return {
        "agent": "agent_d",
        "role": "final_diagnostic_adjudicator",
        "diagnostic_synthesis": str(parsed.get("diagnostic_synthesis", "") or "").strip(),
        "top5_diagnoses": normalized_items,
    }


def build_full_success_result(
    item: Dict[str, Any],
    structured_case: Dict[str, Any],
    kg_context: str,
    kg_debug: Dict[str, Any],
    shared_context: Dict[str, Any],
    agent_d_output: Dict[str, Any],
) -> Dict[str, Any]:
    patient_id = str(shared_context.get("patient_id") or extract_patient_id(item)).strip()

    agent_outputs = {
        "agent_a": None,
        "agent_b": None,
        "agent_c": None,
        "agent_d_final": agent_d_output,
        "kg_context": kg_context,
    }

    return {
        "patient_id": patient_id,
        "status": "success",
        "case_mode": structured_case.get("case_mode", "prospective"),
        "kg_retrieved_diseases": summarize_retrieved_kgs(kg_debug or {}),
        "agent_a": None,
        "agent_b": None,
        "agent_c": None,
        "agent_d_final": agent_d_output,
        "_internal_raw": {
            "input_item": item,
            "structured_case": structured_case,
            "kg_context": kg_context,
            "kg_debug": kg_debug,
            "shared_context": shared_context,
            "agent_outputs": agent_outputs,
        },
    }


def build_full_error_result(
    item: Dict[str, Any],
    failed_stage: str,
    exc: Exception,
    structured_case: Optional[Dict[str, Any]] = None,
    kg_context: Optional[str] = None,
    kg_debug: Optional[Dict[str, Any]] = None,
    shared_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kg_debug = kg_debug if isinstance(kg_debug, dict) else None
    return {
        "patient_id": extract_patient_id(item),
        "status": "error",
        "case_mode": (structured_case or {}).get("case_mode", "prospective"),
        "kg_retrieved_diseases": summarize_retrieved_kgs(kg_debug or {}),
        "agent_a": None,
        "agent_b": None,
        "agent_c": None,
        "agent_d_final": None,
        "failed_stage": failed_stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "_internal_raw": {
            "input_item": item,
            "structured_case": structured_case,
            "kg_context": kg_context,
            "kg_debug": kg_debug,
            "shared_context": shared_context,
            "traceback": traceback.format_exc(),
        },
    }


def load_successful_case_cache(result_dir: Path) -> Dict[str, Dict[str, Any]]:
    result_dir = Path(result_dir)
    cache: Dict[str, Dict[str, Any]] = {}
    if not result_dir.exists():
        return cache

    for result_path in sorted(result_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue
        if str(data.get("status", "")).lower() != "success":
            continue

        patient_id = str(data.get("patient_id", "") or "").strip()
        if not patient_id or patient_id.lower() == "unknown":
            continue

        if patient_id not in cache:
            cache[patient_id] = {"result": data, "path": str(result_path)}

    return cache


def write_run_status(
    status_path: Path,
    total_records: int,
    completed_results: List[Dict[str, Any]],
    output_path: Path,
    result_dir: Path,
) -> None:
    success = sum(1 for x in completed_results if x.get("status") == "success")
    error = sum(1 for x in completed_results if x.get("status") == "error")
    other = len(completed_results) - success - error
    save_json_atomic(
        {
            "total_records": total_records,
            "completed_records": len(completed_results),
            "success": success,
            "error": error,
            "other": other,
            "final_summary_file": str(output_path),
            "single_case_result_dir": str(result_dir),
        },
        status_path,
    )



def run_single_inference(
    sample_id: int,
    item: Dict[str, Any],
    client: OpenAI,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    stage = "build_structured_case"
    structured_case: Optional[Dict[str, Any]] = None
    kg_context: Optional[str] = None
    kg_debug: Optional[Dict[str, Any]] = None
    shared_context: Optional[Dict[str, Any]] = None

    try:
        structured_case = build_structured_case(item)

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


        if not str(shared_context.get("patient_id", "")).strip() or str(shared_context.get("patient_id", "")).lower() == "unknown":
            shared_context["patient_id"] = extract_patient_id(item, sample_id)

        stage = "call_single_model"
        user_content = build_agent_user_content(shared_context)
        system_prompt = build_agent_system_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            shared_context,
        )

        request_params = {
            "model": args.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": args.temperature,
            "response_format": {"type": "json_object"},
        }

        response = client.chat.completions.create(**request_params)
        if not getattr(response, "choices", None):
            raise RuntimeError(f"Invalid API response: {response}")

        raw_response = response.choices[0].message.content
        if not raw_response:
            raise ValueError("Empty response from API")

        stage = "parse_model_output"
        parsed = extract_json_object(raw_response)
        agent_d_output = normalize_agent_d_output(parsed)

        stage = "build_result"
        return build_full_success_result(
            item=item,
            structured_case=structured_case,
            kg_context=kg_context,
            kg_debug=kg_debug or {},
            shared_context=shared_context,
            agent_d_output=agent_d_output,
        )

    except Exception as exc:
        return build_full_error_result(
            item=item,
            failed_stage=stage,
            exc=exc,
            structured_case=structured_case,
            kg_context=kg_context,
            kg_debug=kg_debug,
            shared_context=shared_context,
        )




def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "未提供 API key。请设置环境变量 OPENAI_API_KEY，或通过 --api_key 传入。"
        )
    if not args.base_url:
        raise ValueError("未提供 base_url。请设置 OPENAI_BASE_URL，或通过 --base_url 传入。")


    print("Validating KG files/schema with project config...")
    validate_kg_files()
    print(
        "KG configuration: "
        f"top_k={CONFIG.get('KG_TOP_K', 3)}, "
        f"dynamic_top_k={CONFIG.get('KG_DYNAMIC_TOP_K', True)}, "
        f"max_dynamic_top_k={CONFIG.get('KG_MAX_DYNAMIC_TOP_K', 5)}, "
        f"graph_edge_score={CONFIG.get('KG_INCLUDE_GRAPH_EDGE_SCORE', True)}, "
        f"numeric_facts={CONFIG.get('KG_INCLUDE_NUMERIC_FACTS', True)}"
    )

    output_path = Path(args.output_path)
    result_dir = Path(args.result_dir) if args.result_dir else output_path.with_suffix("")
    status_path = output_path.with_suffix(".run_status.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print("Loading dataset...")
    records = load_dataset(args.data_path)

    if args.sample_size is not None and args.sample_size > 0:
        records = records[: args.sample_size]

    total = len(records)
    print(f"Total selected samples: {total}")

    ordered_summary: List[Optional[Dict[str, Any]]] = [None] * total

    success_cache = load_successful_case_cache(result_dir)
    reused_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.fork_nums)) as executor:
        future_to_index = {}

        for idx, item in enumerate(records):
            patient_id = extract_patient_id(item, idx)
            cached = success_cache.get(patient_id) if patient_id.lower() != "unknown" else None

            if cached is not None:
                compact = build_compact_result_view(cached["result"])
                compact["input_index"] = idx
                compact["source_result_file"] = cached["path"]
                ordered_summary[idx] = compact
                reused_count += 1
                continue

            future = executor.submit(run_single_inference, idx, item, client, args)
            future_to_index[future] = idx

        print(
            f"Resume cache: reused {reused_count} successful cases; "
            f"running {len(future_to_index)} cases through {args.model_name} + KG."
        )

        cached_summary = [x for x in ordered_summary if x is not None]
        if cached_summary:
            save_json_atomic(cached_summary, output_path)
            write_run_status(status_path, total, cached_summary, output_path, result_dir)

        progress = tqdm(total=len(future_to_index), desc="Single GPT + KG inference")

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            item = records[idx]

            try:
                full_result = future.result()
            except Exception as exc:
                full_result = build_full_error_result(
                    item=item,
                    failed_stage="batch_future",
                    exc=exc,
                    structured_case=None,
                )

            patient_id = full_result.get("patient_id") or extract_patient_id(item, idx)
            safe_id = sanitize_filename(patient_id)
            single_result_path = result_dir / f"{idx + 1:05d}_{safe_id}.json"

            save_json_atomic(full_result, single_result_path)

            compact = build_compact_result_view(full_result)
            compact["input_index"] = idx
            compact["source_result_file"] = str(single_result_path)
            ordered_summary[idx] = compact

            completed_summary = [x for x in ordered_summary if x is not None]
            save_json_atomic(completed_summary, output_path)
            write_run_status(status_path, total, completed_summary, output_path, result_dir)

            progress.update(1)

        progress.close()

    final_summary = [x for x in ordered_summary if x is not None]
    save_json_atomic(final_summary, output_path)
    write_run_status(status_path, total, final_summary, output_path, result_dir)

    success_count = sum(1 for x in final_summary if x.get("status") == "success")
    error_count = sum(1 for x in final_summary if x.get("status") == "error")

    print(f"Done. Success={success_count}, Error={error_count}")
    print(f"Final compact summary saved to: {output_path}")
    print(f"Full per-case results saved to: {result_dir}")
    print(f"Run status saved to: {status_path}")


if __name__ == "__main__":
    main()
