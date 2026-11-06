
import json
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def get_first_non_empty(
    item: Dict[str, Any],
    keys: List[str],
    default: str = "",
) -> str:
    """
    按顺序从 dict 中取第一个非空值。
    """
    if not isinstance(item, dict):
        return default

    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)

    return default


def extract_input_patient_id(item: Dict[str, Any]) -> str:
    """
    从原始输入中提取 patient_id。
    和 case_processing.py / main.py 保持一致。
    """
    return get_first_non_empty(
        item,
        ["patient_id", "id", "Patient ID", "ID", "subject_id", "hadm_id"],
        "unknown",
    )


def remove_json_line_comments(text: str) -> str:
    """
    移除大模型偶尔输出的 // 单行注释。
    注意：
    - 只移除整行注释，不尝试移除字符串内部的 //。
    """
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)


def remove_trailing_commas(json_str: str) -> str:
    """
    移除 JSON 中对象或数组结束前的尾随逗号。
    """
    return re.sub(r",\s*([\]}])", r"\1", json_str)


def extract_json_from_markdown_fence(text: str) -> List[str]:
    candidates = []

    if "```" not in text:
        return candidates

    parts = text.split("```")

    for part in parts:
        candidate = part.strip()

        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

        if candidate.startswith("{"):
            candidates.append(candidate)

    return candidates


def find_balanced_json_object(text: str) -> Optional[str]:
    """
    从文本中寻找第一个括号平衡的 JSON object。
    比简单 text[text.find("{"):] 更安全。

    返回：
    - 找到则返回 JSON 字符串
    - 找不到则返回 None
    """
    start = text.find("{")
    if start == -1:
        return None

    stack = []
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

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
            continue

        if char == "{":
            stack.append("{")
        elif char == "}":
            if not stack:
                return None

            stack.pop()

            if not stack:
                return text[start:idx + 1]

    return None


def repair_truncated_json_conservatively(json_str: str) -> str:
    """
    保守修复被截断的 JSON。

    注意：
    这个函数只做轻量修复：
    - 删除尾随逗号
    - 如果括号明显没闭合，则补齐括号
    - 不尝试修复未闭合字符串，因为那样容易制造假内容

    如果字符串未闭合，仍然让 json.loads 报错。
    """
    json_str = json_str.strip()
    json_str = re.sub(r"[\s,:\-\+]+$", "", json_str)
    json_str = remove_trailing_commas(json_str)

    stack = []
    in_string = False
    escape = False

    for char in json_str:
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
            continue

        if char in ("{", "["):
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    if in_string:
        return json_str

    while stack:
        open_bracket = stack.pop()
        if open_bracket == "{":
            json_str += "}"
        elif open_bracket == "[":
            json_str += "]"

    return json_str


def try_parse_json(candidate: str, allow_repair: bool = True) -> Dict[str, Any]:
    """
    尝试解析单个 JSON candidate。
    """
    cleaned = remove_json_line_comments(candidate).strip()
    cleaned = remove_trailing_commas(cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        if not allow_repair:
            raise

        repaired = repair_truncated_json_conservatively(cleaned)
        parsed = json.loads(remove_trailing_commas(repaired))

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object/dict, got {type(parsed).__name__}"
        )

    return parsed


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    从大模型输出中提取 JSON object。

    解析顺序：
    1. 整段文本直接当 JSON 解析
    2. markdown ```json 代码块
    3. 文本中的第一个括号平衡 JSON object
    4. 最后才尝试保守修复截断 JSON

    设计原则：
    - 尽量容忍模型偶尔输出 markdown / 尾随逗号
    - 不要过度修复，避免把严重截断的输出伪装成合法 JSON
    """
    if not text:
        raise ValueError("Empty model output")

    original_text = str(text).strip()
    cleaned_text = remove_json_line_comments(original_text)

    parse_errors = []

    try:
        return try_parse_json(cleaned_text, allow_repair=False)
    except Exception as e:
        parse_errors.append(f"direct_parse: {e}")

    for candidate in extract_json_from_markdown_fence(cleaned_text):
        try:
            return try_parse_json(candidate, allow_repair=True)
        except Exception as e:
            parse_errors.append(f"markdown_candidate: {e}")

    balanced = find_balanced_json_object(cleaned_text)
    if balanced:
        try:
            return try_parse_json(balanced, allow_repair=True)
        except Exception as e:
            parse_errors.append(f"balanced_object: {e}")

    start_idx = cleaned_text.find("{")
    if start_idx != -1:
        raw_json = cleaned_text[start_idx:]
        try:
            return try_parse_json(raw_json, allow_repair=True)
        except Exception as e:
            parse_errors.append(f"truncated_repair: {e}")

    error_preview = cleaned_text[:500].replace("\n", "\\n")
    raise ValueError(
        "Failed to parse JSON object from model output. "
        f"Errors: {parse_errors}. "
        f"Output preview: {error_preview}"
    )


def load_dataset(file_path: str | Path) -> List[Dict[str, Any]]:
    """只读取 patient_id 和 combined_text，绝不读取金标准/评估标签。"""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    suffix = path.suffix.lower()
    allowed_columns = ["patient_id", "combined_text"]
    if suffix in (".csv", ".zip"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=allowed_columns)
        except ValueError as exc:
            raise ValueError("数据集必须包含 patient_id 和 combined_text 两列") from exc
        return df[allowed_columns].to_dict(orient="records")
    if suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, dtype=str, keep_default_na=False, usecols=allowed_columns)
        except ValueError as exc:
            raise ValueError("数据集必须包含 patient_id 和 combined_text 两列") from exc
        return df[allowed_columns].to_dict(orient="records")
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 数据集应是对象列表")
    elif suffix == ".jsonl":
        data = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise ValueError(f"不支持的输入文件格式: {suffix}")
    records=[]
    for idx, record in enumerate(data, 1):
        if not isinstance(record, dict) or not all(k in record for k in allowed_columns):
            raise ValueError(f"第 {idx} 条记录缺少 patient_id 或 combined_text")
        records.append({k:record[k] for k in allowed_columns})
    return records



def save_json_atomic(data: Any, path: Path) -> None:
    """
    原子写入 JSON 文件。
    先写 .tmp，再 replace，降低中途中断造成坏文件的概率。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tmp_path.replace(path)


def recursive_has_key(obj: Any, banned_keys: set) -> Optional[str]:
    """
    递归检查对象中是否存在某些 key。
    可用于检测输出里是否带了不该有的字段。
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in banned_keys:
                return key

            found = recursive_has_key(value, banned_keys)
            if found:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = recursive_has_key(item, banned_keys)
            if found:
                return found

    return None


WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def sanitize_filename(filename: Any, max_length: int = 120) -> str:
    """
    将任意 patient_id 转换为安全文件名。
    """
    if filename is None:
        filename = "unknown"

    filename = str(filename).strip()

    if not filename:
        filename = "unknown"

    filename = re.sub(r'[\\/*?:"<>|\s]+', "_", filename)

    filename = filename.strip("._ ")

    if not filename:
        filename = "unknown"

    if filename.upper() in WINDOWS_RESERVED_FILENAMES:
        filename = f"{filename}_file"

    if len(filename) > max_length:
        filename = filename[:max_length].rstrip("._ ")

    return filename or "unknown"


def _build_batch_exception_result(
    item: Dict[str, Any],
    error: Exception,
    stage: str = "batch_future",
) -> Dict[str, Any]:
    """
    极端兜底：process_single_patient() 理论上会自己捕获错误并返回 error result。
    这个函数只处理 future.result() 本身抛异常的情况，避免批处理直接中断。
    """
    patient_id = extract_input_patient_id(item)
    return {
        "patient_id": patient_id,
        "status": "error",
        "failed_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "_internal_raw": {
            "input_item": item,
            "traceback": traceback.format_exc(),
        },
    }


def _count_status(summary_results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"total": len(summary_results), "success": 0, "error": 0, "other": 0}
    for item in summary_results:
        status = str(item.get("status", "")).lower() if isinstance(item, dict) else ""
        if status == "success":
            counts["success"] += 1
        elif status == "error":
            counts["error"] += 1
        else:
            counts["other"] += 1
    return counts


def _write_run_status(
    *,
    status_path: Path,
    total_records: int,
    completed_results: List[Dict[str, Any]],
    clinical_output_path: Path,
    clinical_result_dir: Path,
) -> None:
    """
    额外写一个轻量状态文件，便于中途查看运行到哪里。
    不影响最终汇总文件 clinical_results_v2.json 的 list 结构。
    """
    counts = _count_status(completed_results)
    save_json_atomic(
        {
            "total_records": total_records,
            "completed_records": len(completed_results),
            "success": counts["success"],
            "error": counts["error"],
            "other": counts["other"],
            "final_summary_file": str(clinical_output_path),
            "single_case_result_dir": str(clinical_result_dir),
        },
        status_path,
    )


def load_successful_case_cache(result_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    扫描已有的逐病例结果文件，并按 patient_id 建立成功结果缓存。

    规则：
    - 只复用 status == "success" 的结果；
    - error / 非法 JSON / 缺少 patient_id 的结果不进入缓存；
    - patient_id == "unknown" 不缓存，避免不同无 ID 病例被误认为同一病例；
    - 同一个 patient_id 如果存在多个成功结果，优先使用最近修改的文件。

    返回格式：
    {
        "patient_id": {
            "result": <完整旧结果 dict>,
            "path": <旧结果文件路径字符串>,
        }
    }
    """
    result_dir = Path(result_dir)
    cache: Dict[str, Dict[str, Any]] = {}

    if not result_dir.exists():
        return cache

    result_files = sorted(
        result_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for result_path in result_files:
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        status = str(data.get("status", "")).strip().lower()
        patient_id = str(data.get("patient_id", "")).strip()

        if status != "success":
            continue
        if not patient_id or patient_id.lower() == "unknown":
            continue

        if patient_id not in cache:
            cache[patient_id] = {
                "result": data,
                "path": str(result_path),
            }

    return cache


def run_all_cases(
    input_path: Optional[str | Path] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    批量运行病例，并恢复“最终汇总文件”的清晰职责。

    输出规则：
    1. 单病例完整结果：
       ./HRDSS2.0-results/{RUN_NAME}/clinical_results_v2/{序号}_{patient_id}.json
       这里保留 agent_a/b/c/d、_internal_raw、kg_debug 等完整调试信息。

    2. 最终汇总文件：
       ./HRDSS2.0-results/{RUN_NAME}/clinical_results_v2.json
       这里只保存 compact summary，不保存 _internal_raw，方便统计 top5_diagnoses 检出率。

    3. 运行状态文件：
       ./HRDSS2.0-results/{RUN_NAME}/clinical_results_v2.run_status.json
       只记录 completed/success/error 数量。
    """
    from config import (
        CONFIG,
        ensure_output_dirs,
        validate_config,
        file_path as configured_file_path,
        clinical_result_dir,
        clinical_output_path,
    )
    from pipeline import process_single_patient
    from results import build_compact_result_view

    ensure_output_dirs()
    validate_config()

    actual_input_path = Path(input_path) if input_path is not None else Path(configured_file_path)
    records = load_dataset(actual_input_path)

    configured_max_samples = CONFIG.get("MAX_SAMPLES")
    if max_samples is None:
        max_samples = configured_max_samples

    if max_samples is not None:
        records = records[: int(max_samples)]

    total = len(records)
    max_workers = max(1, int(CONFIG.get("MAX_WORKERS", 1)))

    ordered_summary: List[Optional[Dict[str, Any]]] = [None] * total
    status_path = clinical_output_path.with_suffix(".run_status.json")

    success_cache = load_successful_case_cache(clinical_result_dir)
    reused_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}

        for idx, item in enumerate(records):
            patient_id = extract_input_patient_id(item)
            cached = None

            if patient_id and patient_id.lower() != "unknown":
                cached = success_cache.get(patient_id)

            if cached is not None:
                cached_result = cached["result"]
                cached_path = cached["path"]

                compact = build_compact_result_view(cached_result)
                compact["input_index"] = idx
                compact["source_result_file"] = cached_path
                ordered_summary[idx] = compact
                reused_count += 1
                continue

            future = executor.submit(process_single_patient, item)
            future_to_index[future] = idx

        print(
            f"Resume cache: reused {reused_count} successful cases; "
            f"running {len(future_to_index)} cases through the model."
        )

        cached_summary = [x for x in ordered_summary if x is not None]
        if cached_summary:
            save_json_atomic(cached_summary, clinical_output_path)
            _write_run_status(
                status_path=status_path,
                total_records=total,
                completed_results=cached_summary,
                clinical_output_path=clinical_output_path,
                clinical_result_dir=clinical_result_dir,
            )

        completed_count = reused_count
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            item = records[idx]
            fallback_patient_id = extract_input_patient_id(item)

            try:
                full_result = future.result()
            except Exception as e:
                full_result = _build_batch_exception_result(item, e)
            completed_count += 1
            print(
                f"[进度] 已完成 {completed_count}/{total} 个病例 "
                f"({completed_count / total * 100:.1f}%)",
                flush=True,
            )


            if not isinstance(full_result, dict):
                full_result = {
                    "patient_id": fallback_patient_id,
                    "status": "error",
                    "failed_stage": "batch_result_validation",
                    "error_type": "InvalidResultType",
                    "error_message": f"process_single_patient returned {type(full_result).__name__}, expected dict.",
                    "_internal_raw": {"input_item": item},
                }

            patient_id = full_result.get("patient_id") or fallback_patient_id
            safe_id = sanitize_filename(patient_id)
            single_result_path = clinical_result_dir / f"{idx + 1:05d}_{safe_id}.json"

            save_json_atomic(full_result, single_result_path)

            compact = build_compact_result_view(full_result)
            compact["input_index"] = idx
            compact["source_result_file"] = str(single_result_path)
            ordered_summary[idx] = compact

            completed_summary = [x for x in ordered_summary if x is not None]
            save_json_atomic(completed_summary, clinical_output_path)
            _write_run_status(
                status_path=status_path,
                total_records=total,
                completed_results=completed_summary,
                clinical_output_path=clinical_output_path,
                clinical_result_dir=clinical_result_dir,
            )

    final_summary = [x for x in ordered_summary if x is not None]
    save_json_atomic(final_summary, clinical_output_path)
    _write_run_status(
        status_path=status_path,
        total_records=total,
        completed_results=final_summary,
        clinical_output_path=clinical_output_path,
        clinical_result_dir=clinical_result_dir,
    )

    print(f"Done. Final compact summary saved to: {clinical_output_path}")
    print(f"Full per-case results saved to: {clinical_result_dir}")
    print(f"Run status saved to: {status_path}")

    return final_summary


if __name__ == "__main__":
    run_all_cases()
