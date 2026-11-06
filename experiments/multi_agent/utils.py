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
        # 如果最后是以逗号或者冒号等未完结符号结尾，先清理掉尾巴
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
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset file not found at: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
        return df.to_dict(orient="records")
    if suffix == ".xlsx":
        df = pd.read_excel(path, dtype=str)
        return df.to_dict(orient="records")
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be a list of records.")
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    raise ValueError(f"Unsupported input file format: {suffix}")


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