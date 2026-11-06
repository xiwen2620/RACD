
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def get_first_non_empty(item: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return default


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model output")
    text = text.strip()

    text = re.sub(r'^\s*//.*$', '', text, flags=re.MULTILINE)

    def remove_trailing_commas(json_str: str) -> str:
        return re.sub(r',\s*([\]}])', r'\1', json_str)

    def repair_truncated_json(json_str: str) -> str:
        json_str = json_str.strip()
        json_str = re.sub(r'[\s,:\-\+]+$', '', json_str)

        stack = []
        for char in json_str:
            if char in ('{', '['):
                stack.append(char)
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()

        while stack:
            open_bracket = stack.pop()
            if open_bracket == '{':
                json_str += '}'
            elif open_bracket == '[':
                json_str += ']'
        return json_str

    try:
        return json.loads(remove_trailing_commas(text))
    except json.JSONDecodeError:
        pass

    if "```" in text:
        for part in text.split("```"):
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(remove_trailing_commas(candidate))
                except json.JSONDecodeError:
                    try:
                        repaired = repair_truncated_json(candidate)
                        return json.loads(remove_trailing_commas(repaired))
                    except json.JSONDecodeError:
                        pass

    start_idx = text.find("{")
    if start_idx != -1:
        raw_json = text[start_idx:]
        try:
            repaired = repair_truncated_json(raw_json)
            return json.loads(remove_trailing_commas(repaired))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse and repair JSON. Original error: {e}")

    raise ValueError("No JSON object found in text")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def extract_input_patient_id(item: Dict[str, Any]) -> str:
    return get_first_non_empty(item, ["patient_id", "id", "Patient ID", "ID", "subject_id", "hadm_id"], "unknown")


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


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', '_', filename)
