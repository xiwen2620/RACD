import json
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
    if not isinstance(item, dict):
        return default
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)

    return default


def extract_input_patient_id(item: Dict[str, Any]) -> str:
    return get_first_non_empty(
        item,
        ["patient_id", "id", "Patient ID", "ID", "subject_id", "hadm_id"],
        "unknown",
    )


def remove_json_line_comments(text: str) -> str:
    return re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)


def remove_trailing_commas(json_str: str) -> str:
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
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input dataset file not found at: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
        return df.to_dict(orient="records")

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype=str).fillna("")
        return df.to_dict(orient="records")

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be a list of records.")

    if suffix == ".jsonl":
        records = []

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSONL at line {line_no}: {e}"
                    ) from e

                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL line {line_no} must be a JSON object."
                    )

                records.append(record)

        return records

    raise ValueError(f"Unsupported input file format: {suffix}")


def save_json_atomic(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def recursive_has_key(obj: Any, banned_keys: set) -> Optional[str]:
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

            # 总汇总：只放 compact view，不放 _internal_raw。
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

