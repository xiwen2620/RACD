import argparse
import json
import os
import re
import time
from collections import defaultdict
from statistics import mean, pstdev
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from tqdm import tqdm



GPT4O_API_KEY = os.getenv("GPT4O_API_KEY", "YOUR_API_KEY")
GPT4O_BASE_URL = os.getenv("GPT4O_BASE_URL", "https://YOUR_API_ENDPOINT/v1")
GPT4O_MODEL_NAME = os.getenv("GPT4O_MODEL_NAME", "gpt-4o")
DEFAULT_INPUT_PATH = "/PATH/TO/YOUR/RESULTS.json"
DEFAULT_DATASET_PATH = "/PATH/TO/YOUR/mimic1000.csv"
DEFAULT_OUTPUT_PATH = "/PATH/TO/YOUR/RESULTS_report.json"
DEFAULT_MAX_SAMPLES = None
DEFAULT_MAX_WORKERS = 25
DEFAULT_API_MAX_WORKERS = 10
TOPK_VALUES = [1, 3, 5]
EVALUATION_SCHEMA_VERSION = "mimic6000-target-diseases-dataset-only-highrisk-detection-v3"

TARGET_DISEASES: Dict[str, Dict[str, Any]] = {
    "SEPSIS": {
        "name": "Sepsis",
        "aliases": ["Sepsis"],
        "patterns": [r"\bsepsis\b", r"\bsepticemia\b", r"\bsepticaemia\b", r"\burosepsis\b"],
        "exclude_patterns": [r"\bseptic shock\b"],
    },
    "SEPTIC_SHOCK": {
        "name": "Septic shock",
        "aliases": ["Septic shock"],
        "patterns": [r"\bseptic shock\b", r"\bsepsis[- ]induced shock\b", r"\bsepsis with shock\b"],
        "exclude_patterns": [],
    },
    "SEVERE_INFECTION_WITH_ORGAN_DYSFUNCTION": {
        "name": "Severe infection with organ dysfunction",
        "aliases": ["Severe infection with organ dysfunction", "Severe infection with acute organ dysfunction"],
        "patterns": [
            r"\bsevere infection\b.*\borgan dysfunction\b",
            r"\bsevere infection\b.*\bacute organ dysfunction\b",
            r"\binfection[- ]associated acute organ dysfunction\b",
            r"\binfection[- ]associated organ dysfunction\b",
            r"\bsevere infection with organ dysfunction\b",
            r"\bsevere infection with acute organ dysfunction\b",
        ],
        "exclude_patterns": [],
    },
    "ACUTE_MYOCARDIAL_INFARCTION": {
        "name": "Acute myocardial infarction",
        "aliases": ["Acute myocardial infarction", "AMI", "STEMI", "NSTEMI"],
        "patterns": [
            r"\bacute myocardial infarction\b", r"\bacute mi\b", r"\bstemi\b", r"\bnstemi\b",
            r"\btype\s*1\s*myocardial infarction\b", r"\bmyocardial infarction\b",
        ],
        "exclude_patterns": [
            r"\bold myocardial infarction\b", r"\bhistory of myocardial infarction\b",
            r"\bpersonal history of myocardial infarction\b", r"\bprior myocardial infarction\b",
            r"\bremote myocardial infarction\b",
        ],
    },
    "ACUTE_CORONARY_SYNDROME": {
        "name": "Acute coronary syndrome",
        "aliases": ["Acute coronary syndrome", "ACS", "unstable angina", "STEMI", "NSTEMI"],
        "patterns": [r"\bacute coronary syndrome\b", r"\bacs\b", r"\bunstable angina\b", r"\bnste[- ]acs\b", r"\bstemi\b", r"\bnstemi\b"],
        "exclude_patterns": [r"\bacs service\b", r"\bacute care surgery\b"],
    },
    "PULMONARY_EMBOLISM": {
        "name": "Pulmonary embolism",
        "aliases": ["Pulmonary embolism", "PE", "pulmonary embolus"],
        "patterns": [r"\bpulmonary embolism\b", r"\bpulmonary embolus\b", r"\bpe\b"],
        "exclude_patterns": [r"\bpersonal history of venous thrombosis and embolism\b", r"\bhistory of pulmonary embolism\b", r"\bprior pulmonary embolism\b"],
    },
    "AORTIC_DISSECTION": {
        "name": "Aortic dissection",
        "aliases": ["Aortic dissection", "acute aortic dissection"],
        "patterns": [r"\baortic dissection\b", r"\bdissection of aorta\b", r"\bacute aortic syndrome\b.*\bdissection\b", r"\btype\s*a\s*aortic dissection\b", r"\btype\s*b\s*aortic dissection\b"],
        "exclude_patterns": [],
    },
    "ISCHEMIC_STROKE": {
        "name": "Ischemic stroke",
        "aliases": ["Ischemic stroke", "acute ischemic stroke", "cerebral infarction"],
        "patterns": [r"\bischemic stroke\b", r"\bischaemic stroke\b", r"\bacute ischemic stroke\b", r"\bcerebral infarction\b", r"\bbrain infarction\b", r"\bischemic cva\b", r"\bcva\b.*\bischemic\b"],
        "exclude_patterns": [r"\bhistory of stroke\b", r"\bpersonal history of stroke\b", r"\bold stroke\b"],
    },
    "INTRACRANIAL_HEMORRHAGE": {
        "name": "Intracranial hemorrhage",
        "aliases": ["Intracranial hemorrhage", "ICH", "hemorrhagic stroke", "subarachnoid hemorrhage", "subdural hematoma"],
        "patterns": [r"\bintracranial hemorrhage\b", r"\bintracerebral hemorrhage\b", r"\bbrain hemorrhage\b", r"\bhemorrhagic stroke\b", r"\bsubarachnoid hemorrhage\b", r"\bsubdural hematoma\b", r"\bepidural hematoma\b", r"\bich\b", r"\bsah\b", r"\bsdh\b", r"\bedh\b"],
        "exclude_patterns": [r"\bhistory of\b", r"\bold\b"],
    },
    "MENINGITIS": {
        "name": "Meningitis",
        "aliases": ["Meningitis", "bacterial meningitis", "meningococcal disease"],
        "patterns": [r"\bmeningitis\b", r"\bmeningococcal disease\b"],
        "exclude_patterns": [],
    },
    "ENCEPHALITIS": {
        "name": "Encephalitis",
        "aliases": ["Encephalitis", "HSV encephalitis", "meningoencephalitis"],
        "patterns": [r"\bencephalitis\b", r"\bmeningoencephalitis\b", r"\bhsv encephalitis\b"],
        "exclude_patterns": [r"\btoxic encephalopathy\b", r"\bmetabolic encephalopathy\b"],
    },
    "BOWEL_PERFORATION": {
        "name": "Bowel perforation",
        "aliases": ["Bowel perforation", "intestinal perforation", "perforated bowel", "perforated viscus"],
        "patterns": [r"\bbowel perforation\b", r"\bintestinal perforation\b", r"\bperforated bowel\b", r"\bperforated intestine\b", r"\bperforated viscus\b", r"\bhollow viscus perforation\b", r"\bgastrointestinal perforation\b", r"\bgi perforation\b"],
        "exclude_patterns": [],
    },
}
TARGET_ORDER = list(TARGET_DISEASES.keys())
NAME_TO_ID: Dict[str, str] = {}
for _did, _spec in TARGET_DISEASES.items():
    NAME_TO_ID[normalize_name := _spec["name"].lower()] = _did
    for _alias in _spec.get("aliases", []):
        NAME_TO_ID[_alias.lower()] = _did



def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)



def save_json_atomic(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)



def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("≥", ">=").replace("≤", "<=")
    text = re.sub(r"[\u3000\xa0]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()




def split_diagnosis_items(text: Any) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*;\s*|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]




def _matches_patterns(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)




def match_target_diseases_in_text(text: Any, *, split_items: bool = False) -> Set[str]:
    items = split_diagnosis_items(text) if split_items else [str(text or "")]
    matched: Set[str] = set()
    for raw_item in items:
        item = normalize_text(raw_item)
        if not item:
            continue
        for disease_id, spec in TARGET_DISEASES.items():
            if spec.get("exclude_patterns") and _matches_patterns(item, spec["exclude_patterns"]):
                continue
            if _matches_patterns(item, spec["patterns"]):
                matched.add(disease_id)
    return matched




def disease_ids_to_names(ids: Iterable[str]) -> List[str]:
    ids_set = set(ids)
    return [TARGET_DISEASES[d]["name"] for d in TARGET_ORDER if d in ids_set]



def parse_exact_target_diseases(value: Any) -> Set[str]:
    if value is None:
        return set()

    # 设置字段“raw_items”对应的配置、疾病信息或输出值。
    raw_items: List[Any]
    # 判断条件“isinstance(value, (list, tuple, set))”是否成立，成立时执行下方分支。
    if isinstance(value, (list, tuple, set)):
        # 计算并保存变量 raw_items，供后续步骤读取或继续处理。
        raw_items = list(value)
    # 当前面的条件均不成立时，执行下面的备用处理逻辑。
    else:
        # 计算并保存变量 text，供后续步骤读取或继续处理。
        text = str(value).strip()
        # 判断条件“not text or text.lower() in {"nan", "none", "null"}”是否成立，成立时执行下方分支。
        if not text or text.lower() in {"nan", "none", "null"}:
            # 将“set()”作为当前函数的返回结果交给调用方。
            return set()
        # 计算并保存变量 raw_items，供后续步骤读取或继续处理。
        raw_items = re.split(r"\s*;\s*|\s*\|\s*|\n+", text)

    # 设置字段“matched”对应的配置、疾病信息或输出值。
    matched: Set[str] = set()
    # 遍历“item in raw_items”中的元素，并逐项执行循环体。
    for item in raw_items:
        # 计算并保存变量 did，供后续步骤读取或继续处理。
        did = normalize_returned_disease_id(item)
        # 判断条件“did”是否成立，成立时执行下方分支。
        if did:
            # 调用 matched.add，执行该函数封装的处理逻辑。
            matched.add(did)
    # 将“matched”作为当前函数的返回结果交给调用方。
    return matched


def normalize_patient_id(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"\.0$", "", text)



def normalize_returned_disease_id(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    upper = text.upper().replace(" ", "_").replace("-", "_")
    if upper in TARGET_DISEASES:
        return upper
    low = text.lower().strip()
    if low in NAME_TO_ID:
        return NAME_TO_ID[low]
    for did, spec in TARGET_DISEASES.items():
        if low == spec["name"].lower():
            return did
        for alias in spec.get("aliases", []):
            if low == alias.lower():
                return did
    return None



def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = str(text).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    candidate = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None



def extract_gold_standard(
    result: Dict[str, Any],
    original_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Set[str], str, str, str]:
    del result
    original_data = original_data or {}

    if "target_diseases" not in original_data:
        raise ValueError(
            "Original dataset row is missing required field 'target_diseases'; "
            "Gold cannot be constructed from any fallback source."
        )

    raw_value = original_data.get("target_diseases")
    gold_ids = parse_exact_target_diseases(raw_value)
    return gold_ids, "dataset.target_diseases", str(raw_value or ""), ""



def extract_top5_diagnosis_names(result: Dict[str, Any]) -> List[str]:
    names = result.get("top5_diagnosis_names")
    if isinstance(names, list):
        return [str(x).strip() for x in names if str(x).strip()]

    for key in ["top5_diagnoses", "top_diagnoses"]:
        items = result.get(key)
        if isinstance(items, list):
            out = []
            for item in items:
                if isinstance(item, dict):
                    diagnosis = item.get("diagnosis")
                    if diagnosis is not None and str(diagnosis).strip():
                        out.append(str(diagnosis).strip())
                elif item is not None and str(item).strip():
                    out.append(str(item).strip())
            if out:
                return out

    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    items = nested.get("top5_diagnoses") if isinstance(nested, dict) else None
    if isinstance(items, list):
        out = []
        for item in items:
            if isinstance(item, dict):
                diagnosis = item.get("diagnosis")
                if diagnosis is not None and str(diagnosis).strip():
                    out.append(str(diagnosis).strip())
            elif item is not None and str(item).strip():
                out.append(str(item).strip())
        return out
    return []


def extract_top5_evidence(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = result.get("top5_evidence")
    if isinstance(evidence, list):
        return evidence

    evidence_out = []
    # 计算并保存变量 top5，供后续步骤读取或继续处理。
    top5 = result.get("top5_diagnoses")
    # 判断条件“isinstance(top5, list)”是否成立，成立时执行下方分支。
    if isinstance(top5, list):
        # 遍历“item in top5”中的元素，并逐项执行循环体。
        for item in top5:
            # 判断条件“not isinstance(item, dict)”是否成立，成立时执行下方分支。
            if not isinstance(item, dict):
                # 跳过本次循环剩余代码，直接处理下一个元素。
                continue
            # 计算并保存变量 diagnosis，供后续步骤读取或继续处理。
            diagnosis = str(item.get("diagnosis", "")).strip()
            # 计算并保存变量 ev_list，供后续步骤读取或继续处理。
            ev_list = item.get("primary_supporting_evidence", [])
            # 判断条件“not isinstance(ev_list, list)”是否成立，成立时执行下方分支。
            if not isinstance(ev_list, list):
                # 计算并保存变量 ev_list，供后续步骤读取或继续处理。
                ev_list = [ev_list] if ev_list else []
            # 调用 evidence_out.append，执行该函数封装的处理逻辑。
            evidence_out.append({
                # 设置字段“rank”对应的配置、疾病信息或输出值。
                "rank": item.get("rank"),
                # 设置字段“diagnosis”对应的配置、疾病信息或输出值。
                "diagnosis": diagnosis,
                # 设置字段“evidence_quote_count”对应的配置、疾病信息或输出值。
                "evidence_quote_count": len(ev_list),
                # 设置字段“evidence”对应的配置、疾病信息或输出值。
                "evidence": [{"quote": str(e), "source": "top5_diagnoses.primary_supporting_evidence"} for e in ev_list if str(e).strip()],
                # 设置字段“adjudication_reasoning”对应的配置、疾病信息或输出值。
                "adjudication_reasoning": item.get("adjudication_reasoning", ""),
            # 执行本行代码：“})”。
            })

    # 兼容旧结构 result.top5_evidence_claims
    # 计算并保存变量 nested，供后续步骤读取或继续处理。
    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    # 计算并保存变量 old_claims，供后续步骤读取或继续处理。
    old_claims = nested.get("top5_evidence_claims") if isinstance(nested, dict) else None
    # 判断条件“not evidence_out and isinstance(old_claims, list)”是否成立，成立时执行下方分支。
    if not evidence_out and isinstance(old_claims, list):
        # 遍历“claims in old_claims”中的元素，并逐项执行循环体。
        for claims in old_claims:
            # 判断条件“not isinstance(claims, dict)”是否成立，成立时执行下方分支。
            if not isinstance(claims, dict):
                # 跳过本次循环剩余代码，直接处理下一个元素。
                continue
            # 计算并保存变量 diagnosis，供后续步骤读取或继续处理。
            diagnosis = str(claims.get("diagnosis", "")).strip()
            # 计算并保存变量 ev_list，供后续步骤读取或继续处理。
            ev_list = claims.get("evidence", [])
            # 判断条件“not isinstance(ev_list, list)”是否成立，成立时执行下方分支。
            if not isinstance(ev_list, list):
                # 计算并保存变量 ev_list，供后续步骤读取或继续处理。
                ev_list = [ev_list] if ev_list else []
            # 调用 evidence_out.append，执行该函数封装的处理逻辑。
            evidence_out.append({
                # 设置字段“rank”对应的配置、疾病信息或输出值。
                "rank": claims.get("rank"),
                # 设置字段“diagnosis”对应的配置、疾病信息或输出值。
                "diagnosis": diagnosis,
                # 设置字段“evidence_quote_count”对应的配置、疾病信息或输出值。
                "evidence_quote_count": len(ev_list),
                # 设置字段“evidence”对应的配置、疾病信息或输出值。
                "evidence": [{"quote": str(e), "source": "result.top5_evidence_claims.evidence"} for e in ev_list if str(e).strip()],
                # 设置字段“adjudication_reasoning”对应的配置、疾病信息或输出值。
                "adjudication_reasoning": claims.get("adjudication_reasoning", ""),
            # 执行本行代码：“})”。
            })
    # 将“evidence_out”作为当前函数的返回结果交给调用方。
    return evidence_out


def extract_kg_target_ids(result: Dict[str, Any]) -> Set[str]:
    ids = result.get("kg_retrieved_disease_ids")
    if isinstance(ids, list):
        return {str(x).strip().upper() for x in ids if str(x).strip().upper() in TARGET_DISEASES}
    out = set()
    kg_items = result.get("kg_retrieved_diseases")
    if isinstance(kg_items, list):
        for item in kg_items:
            if isinstance(item, dict):
                did = str(item.get("disease_id", "")).strip().upper()
                if did in TARGET_DISEASES:
                    out.add(did)
    return out



def load_original_dataset(path: Optional[str | Path]) -> Dict[str, Dict[str, Any]]:
    if not path:
        raise ValueError(
            "--dataset_path is required because Gold may only come from the original dataset target_diseases column."
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_path not found: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"patient_id", "target_diseases"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required column(s): " + ", ".join(missing)
            + ". Expected the CSV generated by mimic6000.py."
        )
    df["patient_id"] = df["patient_id"].map(normalize_patient_id)
    if (df["patient_id"] == "").any():
        raise ValueError("Dataset contains empty patient_id values")
    if df["patient_id"].duplicated().any():
        duplicate_ids = df.loc[df["patient_id"].duplicated(keep=False), "patient_id"].head(10).tolist()
        raise ValueError(f"Dataset contains duplicate patient_id values, examples: {duplicate_ids}")
    return df.set_index("patient_id").to_dict(orient="index")


def match_exact_gold_disease_name(
    predicted_diagnosis: Any,
    gold_target_ids: Set[str],
) -> Set[str]:
    predicted = str(predicted_diagnosis or "").strip().casefold()
    if not predicted:
        return set()

    return {
        disease_id
        for disease_id in gold_target_ids
        if predicted == TARGET_DISEASES[disease_id]["name"].casefold()
    }


def api_match_single_prediction(
    predicted_diagnosis: str,
    gold_target_ids: Set[str],
    client: Optional[Any],
    model: str,
    max_retries: int = 2,
) -> Tuple[Set[str], Dict[str, Any]]:
    rule_ids = match_exact_gold_disease_name(predicted_diagnosis, gold_target_ids)

    debug = {
        "predicted_diagnosis": predicted_diagnosis,
        "rule_matched_disease_ids": [d for d in TARGET_ORDER if d in rule_ids],
        "api_matched_disease_ids": [],
        "api_reason": "",
        "api_error": "",
        "matching_method": "case_insensitive_exact_rule_only" if client is None else "highrisk_disease_detection_api_plus_exact_rule",
    }

    if not gold_target_ids or not str(predicted_diagnosis or "").strip():
        return set(), debug

    if client is None:
        return rule_ids, debug

    target_payload = [
        {
            "disease_id": did,
            "disease_name": TARGET_DISEASES[did]["name"],
        }
        for did in TARGET_ORDER if did in gold_target_ids
    ]

    prompt = f"""
You are evaluating whether a diagnostic model recognized a patient's high-risk disease.

For each item in [GOLD DISEASES], decide whether the [MODEL PREDICTED DIAGNOSIS] clearly identifies that high-risk disease, even when the wording or diagnostic granularity differs.

[GOLD DISEASES]
{json.dumps(target_payload, ensure_ascii=False, indent=2)}

[MODEL PREDICTED DIAGNOSIS]
"{predicted_diagnosis}"

Matching rules:
1. Match when the prediction clearly states the Gold disease itself.
2. Standard synonyms, accepted abbreviations, spelling variants, and equivalent medical terminology match.
3. A clear, clinically recognized subtype or more specific form of the Gold disease matches. Example: NSTEMI or STEMI matches ACUTE_MYOCARDIAL_INFARCTION; subarachnoid hemorrhage matches INTRACRANIAL_HEMORRHAGE; urosepsis matches SEPSIS.
4. A broader diagnosis may match a more specific Gold disease only when the prediction still clearly establishes that the patient has the same high-risk disease and does not omit a defining feature that changes the disease identity. Example: acute myocardial infarction may match NSTEMI for high-risk disease detection. However, sepsis must NOT match SEPTIC_SHOCK because shock is a defining high-risk state that sepsis alone does not establish.
5. Do NOT match merely similar, related, overlapping, or commonly confused diseases. A shared organ, symptom pattern, treatment, mechanism, or clinical context is insufficient.
6. Do NOT match a risk factor, symptom, manifestation, cause, complication, common comorbidity, or differential diagnosis instead of the Gold disease.
7. Do NOT match suspected, possible, ruled-out, historical, or unsupported diagnoses to an active Gold disease.
8. Examples of non-matches: ACUTE_CORONARY_SYNDROME does not automatically match ACUTE_MYOCARDIAL_INFARCTION; pneumonia does not match SEPSIS; meningitis does not match ENCEPHALITIS; ischemic stroke does not match INTRACRANIAL_HEMORRHAGE.
9. If the prediction does not clearly establish that the patient has the Gold high-risk disease, do not match it.
10. Return ONLY valid JSON in this exact schema:
{{"matched_disease_ids": ["DISEASE_ID"], "reason": "brief explanation of why the high-risk disease was or was not recognized"}}
""".strip()

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=220,
            )
            content = response.choices[0].message.content.strip()
            data = extract_json_object(content) or {}
            api_ids: Set[str] = set()
            returned_ids = data.get("matched_disease_ids", [])
            if not isinstance(returned_ids, list):
                returned_ids = []
            for item in returned_ids:
                did = normalize_returned_disease_id(item)
                if did and did in gold_target_ids:
                    api_ids.add(did)

            debug["api_matched_disease_ids"] = [d for d in TARGET_ORDER if d in api_ids]
            debug["api_reason"] = str(data.get("reason", ""))
            return api_ids | rule_ids, debug
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))

    debug["api_error"] = last_error
    debug["matching_method"] = "case_insensitive_exact_rule_fallback_after_api_error"
    return rule_ids, debug


def build_topk_detection(
    top5_names: List[str],
    gold_ids: Set[str],
    client: Optional[Any],
    model: str,
    k_values: Iterable[int] = TOPK_VALUES,
) -> Tuple[Dict[int, Set[str]], List[Dict[str, Any]]]:
    per_prediction_matches: List[Dict[str, Any]] = []
    matched_by_rank: Dict[int, Set[str]] = {}

    for rank, diagnosis in enumerate(top5_names[:max(k_values)], start=1):
        matched_ids, debug = api_match_single_prediction(diagnosis, gold_ids, client, model)
        matched_by_rank[rank] = matched_ids
        per_prediction_matches.append({
            "rank": rank,
            "predicted_diagnosis": diagnosis,
            "matched_gold_target_disease_ids": [d for d in TARGET_ORDER if d in matched_ids],
            "matched_gold_target_disease_names": disease_ids_to_names(matched_ids),
            **debug,
        })


    detected_at_k: Dict[int, Set[str]] = {}
    for k in k_values:
        ids: Set[str] = set()
        for rank in range(1, k + 1):
            ids |= matched_by_rank.get(rank, set())
        detected_at_k[k] = ids & gold_ids

    return detected_at_k, per_prediction_matches



def validate_evidence(items: List[Dict[str, Any]], combined_text: str, client: Optional[Any], model: str) -> List[
    Dict[str, Any]]:
    if not items:
        return []
    validation_results = []
    for item in items:
        evidence_text = item.get("text", "") or item.get("quote", "")
        source = item.get("source", "")
        if not evidence_text:
            continue

        if client is None or not str(combined_text).strip():
            validation_results.append({
                "text": evidence_text,
                "source": source,
                "is_valid": None,
                "reason": "skipped: API client unavailable or patient record/test results empty",
                "judgment": "skipped",
            })
            continue

        prompt = f"""You are a medical expert verifying clinical evidence.
[CLINICAL RECORD]
{combined_text}

[EVIDENCE]
"{evidence_text}"

Please verify if the evidence is supported by records. Return JSON ONLY:
{{"judgment": "valid/invalid/unclear", "reason": "brief explanation"}}
"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=220,
            )
            result_text = response.choices[0].message.content.strip()
            rj = extract_json_object(result_text) or {}
            judgment = str(rj.get("judgment", "")).lower()
            validation_results.append({
                "text": evidence_text,
                "source": source,
                "is_valid": judgment == "valid",
                "reason": rj.get("reason", ""),
                "judgment": judgment,
            })
        except Exception as e:
            validation_results.append({
                "text": evidence_text,
                "source": source,
                "is_valid": None,
                "reason": f"api_error: {e}",
                "judgment": "api_error",
            })
    return validation_results


def build_evidence_validation(
    top5_evidence: List[Dict[str, Any]],
    combined_text: str,
    client: Optional[Any],
    model: str,
    enabled: bool,
) -> List[Dict[str, Any]]:
    evidence_validations = []
    if not enabled:
        return evidence_validations

    for claims in top5_evidence:
        if not isinstance(claims, dict):
            continue
        diag = claims.get("diagnosis", "")
        raw_evidence_list = claims.get("evidence", [])
        support_items = []
        if isinstance(raw_evidence_list, list):
            for e in raw_evidence_list:
                if isinstance(e, dict):
                    quote = e.get("quote", "") or e.get("text", "")
                    source = e.get("source", "top5_evidence")
                else:
                    quote = str(e)
                    source = "top5_evidence"
                if str(quote).strip():
                    support_items.append({"text": str(quote), "source": source, "agent_source": "model"})

        evidence_validations.append({
            "diagnosis": diag,
            "supporting_evidence_validation": validate_evidence(support_items, combined_text, client, model),
            "opposing_evidence_validation": [],
        })
    return evidence_validations



def evaluate_single_case(
    result: Dict[str, Any],
    original_data: Dict[str, Any],
    match_client: Optional[Any],
    evidence_client: Optional[Any],
    gpt_model: str,
    enable_evidence_validation: bool,
) -> Dict[str, Any]:
    patient_id = normalize_patient_id(result.get("patient_id", ""))
    status = str(result.get("status", "")).strip()

    combined_text = original_data.get("combined_text", "")
    gold_ids, gold_source, gold_target_diseases_text, gold_diagnosis = extract_gold_standard(result, original_data)
    top5_names = extract_top5_diagnosis_names(result)
    top5_evidence = extract_top5_evidence(result)
    detected_at_k, prediction_matches = build_topk_detection(top5_names, gold_ids, match_client, gpt_model)

    kg_ids = extract_kg_target_ids(result)
    kg_gold_ids = gold_ids & kg_ids

    evidence_validations = build_evidence_validation(
        top5_evidence=top5_evidence,
        combined_text=str(combined_text),
        client=evidence_client,
        model=gpt_model,
        enabled=enable_evidence_validation,
    )

    total_supporting = sum(len(v.get("supporting_evidence_validation", [])) for v in evidence_validations)
    valid_supporting = sum(
        sum(1 for e in v.get("supporting_evidence_validation", []) if e.get("is_valid") is True)
        for v in evidence_validations
    )

    gold_count = len(gold_ids)

    topk_report: Dict[str, Dict[str, Any]] = {}
    for k in TOPK_VALUES:
        ids = detected_at_k.get(k, set())
        topk_report[f"top{k}"] = {
            "detected_disease_ids": [d for d in TARGET_ORDER if d in ids],
            "detected_disease_names": disease_ids_to_names(ids),
            "detected_disease_count": len(ids),
            "missed_disease_ids": [d for d in TARGET_ORDER if d in (gold_ids - ids)],
            "missed_disease_names": disease_ids_to_names(gold_ids - ids),
            "detection_rate": round(len(ids) / gold_count, 4) if gold_count else None,
        }

    return {
        "patient_id": patient_id,
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "status": status,
        "input_index": result.get("input_index"),
        "source_result_file": result.get("source_result_file", ""),
        "gold_standard_source": gold_source,
        "gold_target_diseases_text": gold_target_diseases_text,
        "gold_diagnosis_text": gold_diagnosis,
        "gold_target_disease_ids": [d for d in TARGET_ORDER if d in gold_ids],
        "gold_target_disease_names": disease_ids_to_names(gold_ids),
        "gold_target_disease_count": gold_count,
        "top5_diagnosis_names": top5_names,
        "prediction_target_matches_by_rank": prediction_matches,
        "topk_detection": topk_report,
        "detected_diseases": [
            {"disease": TARGET_DISEASES[d]["name"], "disease_id": d, "predicted_diagnosis": "API_or_rule_verified"}
            for d in TARGET_ORDER if d in detected_at_k.get(5, set())
        ],
        "detection_rate": topk_report["top5"]["detection_rate"],
        "has_gold_target_disease": gold_count > 0,
        "kg_retrieved_disease_ids": [d for d in TARGET_ORDER if d in kg_ids],
        "kg_retrieved_disease_names": disease_ids_to_names(kg_ids),
        "kg_gold_recalled_disease_ids": [d for d in TARGET_ORDER if d in kg_gold_ids],
        "kg_gold_recalled_disease_names": disease_ids_to_names(kg_gold_ids),
        "kg_gold_recall_rate": round(len(kg_gold_ids) / gold_count, 4) if gold_count else None,
        "top5_evidence": top5_evidence,
        "evidence_validation": evidence_validations,
        "evidence_accuracy": {
            "total_supporting_evidence": total_supporting,
            "valid_supporting_evidence": valid_supporting,
            "supporting_evidence_accuracy": round(valid_supporting / total_supporting, 4) if total_supporting > 0 else None,
            "total_opposing_evidence": 0,
            "valid_opposing_evidence": 0,
        },
    }


def generate_summary(case_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    disease_stats = defaultdict(lambda: {
        "gold_cases": 0,
        "top1_detected_cases": 0,
        "top3_detected_cases": 0,
        "top5_detected_cases": 0,
        "kg_recalled_cases": 0,
    })

    totals = {
        "gold": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "kg": 0,
    }
    cases_with_hit = {"top1": 0, "top3": 0, "top5": 0, "kg": 0}
    cases_with_gold = 0
    total_supporting_evidence = 0
    valid_supporting_evidence = 0
    cases_with_evidence_validation = 0



    # 遍历“case in case_reports”中的元素，并逐项执行循环体。
    for case in case_reports:
        gold_ids = set(case.get("gold_target_disease_ids", []))
        if gold_ids:
            cases_with_gold += 1
        totals["gold"] += len(gold_ids)

        kg_ids = set(case.get("kg_gold_recalled_disease_ids", []))
        totals["kg"] += len(kg_ids)
        if kg_ids:
            cases_with_hit["kg"] += 1


        topk = case.get("topk_detection", {})
        for k in TOPK_VALUES:
            key = f"top{k}"
            ids = set(topk.get(key, {}).get("detected_disease_ids", []))
            totals[key] += len(ids)
            if ids:
                cases_with_hit[key] += 1

        for d in gold_ids:
            disease_stats[d]["gold_cases"] += 1
        for d in set(topk.get("top1", {}).get("detected_disease_ids", [])):
            disease_stats[d]["top1_detected_cases"] += 1
        for d in set(topk.get("top3", {}).get("detected_disease_ids", [])):
            disease_stats[d]["top3_detected_cases"] += 1
        for d in set(topk.get("top5", {}).get("detected_disease_ids", [])):
            disease_stats[d]["top5_detected_cases"] += 1
        for d in kg_ids:
            disease_stats[d]["kg_recalled_cases"] += 1

        evidence_validation = case.get("evidence_validation", [])
        if evidence_validation:
            cases_with_evidence_validation += 1
        for ev in evidence_validation:
            supporting = ev.get("supporting_evidence_validation", [])
            total_supporting_evidence += len(supporting)
            valid_supporting_evidence += sum(1 for e in supporting if e.get("is_valid") is True)


    disease_detection_rates: Dict[str, Dict[str, Any]] = {}
    for d in TARGET_ORDER:
        stats = disease_stats[d]
        gold_cases = stats["gold_cases"]
        disease_detection_rates[d] = {
            "disease_name": TARGET_DISEASES[d]["name"],
            "gold_cases": gold_cases,
            "top1_detected_cases": stats["top1_detected_cases"],
            "top1_detection_rate": round(stats["top1_detected_cases"] / gold_cases, 4) if gold_cases else None,
            "top3_detected_cases": stats["top3_detected_cases"],
            "top3_detection_rate": round(stats["top3_detected_cases"] / gold_cases, 4) if gold_cases else None,
            "top5_detected_cases": stats["top5_detected_cases"],
            "top5_detection_rate": round(stats["top5_detected_cases"] / gold_cases, 4) if gold_cases else None,
            "kg_recalled_cases": stats["kg_recalled_cases"],
            "kg_recall_rate": round(stats["kg_recalled_cases"] / gold_cases, 4) if gold_cases else None,
        }


    return {
        "status": "summary_ready",
        "total_cases": len(case_reports),
        "success_cases": sum(1 for c in case_reports if str(c.get("status", "")).lower() == "success"),
        "cases_with_gold_target_disease": cases_with_gold,
        "cases_without_gold_target_disease": len(case_reports) - cases_with_gold,
        "total_gold_target_diseases": totals["gold"],
        "top1_detected_gold_target_diseases": totals["top1"],
        "top1_detection_rate": round(totals["top1"] / totals["gold"], 4) if totals["gold"] else None,
        "top3_detected_gold_target_diseases": totals["top3"],
        "top3_detection_rate": round(totals["top3"] / totals["gold"], 4) if totals["gold"] else None,
        "top5_detected_gold_target_diseases": totals["top5"],
        "top5_detection_rate": round(totals["top5"] / totals["gold"], 4) if totals["gold"] else None,
        "overall_detection_rate": round(totals["top5"] / totals["gold"], 4) if totals["gold"] else None,
        "cases_with_top1_hit": cases_with_hit["top1"],
        "cases_with_top3_hit": cases_with_hit["top3"],
        "cases_with_top5_hit": cases_with_hit["top5"],
        "total_kg_recalled_gold_target_diseases": totals["kg"],
        "overall_kg_gold_recall_rate": round(totals["kg"] / totals["gold"], 4) if totals["gold"] else None,
        "cases_with_kg_gold_hit": cases_with_hit["kg"],
        "evidence_validation_summary": {
            "cases_with_evidence_validation": cases_with_evidence_validation,
            "total_supporting_evidence_items": total_supporting_evidence,
            "valid_supporting_evidence_items": valid_supporting_evidence,
            "evidence_accuracy": round(valid_supporting_evidence / total_supporting_evidence, 4) if total_supporting_evidence else None,
        },
        "disease_detection_rates": disease_detection_rates,
    }


def is_case_report_compatible(case: Dict[str, Any]) -> bool:
    if not isinstance(case, dict):
        return False
    topk = case.get("topk_detection")
    if not isinstance(topk, dict):
        return False
    return (
        case.get("evaluation_schema_version") == EVALUATION_SCHEMA_VERSION
        and all(f"top{k}" in topk for k in TOPK_VALUES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HRDSS clinical_results_v3.json with API semantic matching and top1/top3/top5 detection rates.")
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--api_max_workers", type=int, default=DEFAULT_API_MAX_WORKERS)
    parser.add_argument("--disable_api_matching", action="store_true", help="Disable API semantic matching; use regex/rule matching only.")
    parser.add_argument("--disable_evidence_validation", action="store_true", help="Disable API evidence validation.")
    parser.add_argument("--no_resume", action="store_true", help="Do not reuse existing case_reports in output_path.")
    args = parser.parse_args()

    print("=" * 70)
    print("Starting HRDSS API top1/top3/top5 evaluation")
    print("=" * 70)
    print(f"Input result file: {args.input_path}")
    print(f"Output report file: {args.output_path}")
    results = load_json(args.input_path)
    if not isinstance(results, list):
        raise ValueError("Input clinical results JSON must be a list of case result objects.")
    if args.max_samples is not None:
        results = results[: int(args.max_samples)]
    print(f"Clinical results loaded: {len(results)} cases")

    dataset = load_original_dataset(args.dataset_path)
    print(f"Original dataset loaded: {len(dataset)} patients")

    result_patient_ids = {normalize_patient_id(r.get("patient_id", "")) for r in results}
    missing_patient_ids = sorted(pid for pid in result_patient_ids if pid and pid not in dataset)
    if missing_patient_ids:
        raise ValueError(
            "Some result patient_id values are absent from the original dataset; "
            f"Gold cannot be constructed. Examples: {missing_patient_ids[:10]}"
        )

    api_key_available = bool(GPT4O_API_KEY.strip())
    openai_available = OpenAI is not None
    use_api_matching = (not args.disable_api_matching) and api_key_available and openai_available
    enable_evidence_validation = (not args.disable_evidence_validation) and api_key_available and openai_available

    if not api_key_available:
        print("Warning: GPT4O_API_KEY is empty. API matching/evidence validation disabled; using rules only.")
    if not openai_available:
        print("Warning: openai package is not installed. API matching/evidence validation disabled; using rules only.")

    match_client = OpenAI(api_key=GPT4O_API_KEY, base_url=GPT4O_BASE_URL) if use_api_matching else None
    evidence_client = OpenAI(api_key=GPT4O_API_KEY, base_url=GPT4O_BASE_URL) if enable_evidence_validation else None
    print(f"API semantic matching enabled: {use_api_matching}")
    print(f"API evidence validation enabled: {enable_evidence_validation}")


    existing_case_reports: List[Dict[str, Any]] = []
    completed_ids: Set[str] = set()
    if (not args.no_resume) and Path(args.output_path).exists():
        try:
            existing_report = load_json(args.output_path)
            raw_cases = existing_report.get("case_reports", []) if isinstance(existing_report, dict) else []
            for case in raw_cases:
                if not is_case_report_compatible(case):
                    continue
                pid = str(case.get("patient_id", "")).strip()
                if pid:
                    existing_case_reports.append(case)
                    completed_ids.add(pid)
            print(f"Found reusable evaluated cases: {len(completed_ids)}")
        except Exception as e:
            print(f"Warning: Could not load existing evaluation report: {e}")

    results_to_evaluate = [r for r in results if normalize_patient_id(r.get("patient_id", "")) not in completed_ids]

    if completed_ids:
        print(f"Skipping {len(results) - len(results_to_evaluate)} already evaluated compatible cases. Remaining: {len(results_to_evaluate)}")

    new_case_reports: List[Dict[str, Any]] = []
    if results_to_evaluate:
        print(f"Evaluating {len(results_to_evaluate)} new cases...")
        workers = max(1, min(int(args.max_workers), int(args.api_max_workers) if (use_api_matching or enable_evidence_validation) else int(args.max_workers)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for r in results_to_evaluate:
                pid = normalize_patient_id(r.get("patient_id", ""))
                futures[executor.submit(
                    evaluate_single_case,
                    r,
                    dataset.get(pid, {}),
                    match_client,
                    evidence_client,
                    GPT4O_MODEL_NAME,
                    enable_evidence_validation,
                )] = r

            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    new_case_reports.append(future.result())
                except Exception as e:
                    r = futures[future]
                    new_case_reports.append({
                        "patient_id": normalize_patient_id(r.get("patient_id", "")),
                        "status": "evaluation_error",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    })

                checkpoint_case_reports = existing_case_reports + new_case_reports
                checkpoint_case_reports.sort(
                    key=lambda x: (
                        x.get("input_index") is None,
                        x.get("input_index")
                        if x.get("input_index") is not None
                        else 10 ** 12,
                        str(x.get("patient_id", "")),
                    ))
                checkpoint_summary = generate_summary(checkpoint_case_reports)
                checkpoint_report = {
                    "summary": checkpoint_summary,
                    "case_reports": checkpoint_case_reports,
                    "target_diseases": [{"disease_id": d, "disease_name": TARGET_DISEASES[d]["name"]} for d in
                                        TARGET_ORDER],
                    "settings": {
                        "api_semantic_matching_enabled": use_api_matching,
                        "api_evidence_validation_enabled": enable_evidence_validation,
                        "model": GPT4O_MODEL_NAME,
                        "topk_values": TOPK_VALUES,
                        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                        "gold_standard_source": "dataset.target_diseases_only",
                        "rule_matching_policy": "case_insensitive_exact_standard_disease_name_only",
                        "api_matching_policy": "same_clinical_disease_only; no related/similar/parent-child/complication matches",
                    },
                }
                save_json_atomic(checkpoint_report, args.output_path)
                print(
                    f"\n[CHECKPOINT SAVED] "
                    f"saved_cases={len(checkpoint_case_reports)}"
                )

    case_reports = existing_case_reports + new_case_reports
    case_reports.sort(key=lambda x: (x.get("input_index") is None, x.get("input_index") if x.get("input_index") is not None else 10**12, str(x.get("patient_id", ""))))

    summary = generate_summary(case_reports)
    report = {
        "summary": summary,
        "case_reports": case_reports,
        "target_diseases": [{"disease_id": d, "disease_name": TARGET_DISEASES[d]["name"]} for d in TARGET_ORDER],
        "settings": {
            "api_semantic_matching_enabled": use_api_matching,
            "api_evidence_validation_enabled": enable_evidence_validation,
            "model": GPT4O_MODEL_NAME,
            "topk_values": TOPK_VALUES,
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "gold_standard_source": "dataset.target_diseases_only",
            "rule_matching_policy": "case_insensitive_exact_standard_disease_name_only",
            "api_matching_policy": "same_clinical_disease_only; no related/similar/parent-child/complication matches",
        },
    }
    save_json_atomic(report, args.output_path)

    print("\nEvaluation completed.")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Cases with target gold disease: {summary['cases_with_gold_target_disease']}")
    print(f"Total gold target diseases: {summary['total_gold_target_diseases']}")
    print(f"Top1 detected gold target diseases: {summary['top1_detected_gold_target_diseases']} | rate={summary['top1_detection_rate']}")
    print(f"Top3 detected gold target diseases: {summary['top3_detected_gold_target_diseases']} | rate={summary['top3_detection_rate']}")
    print(f"Top5 detected gold target diseases: {summary['top5_detected_gold_target_diseases']} | rate={summary['top5_detection_rate']}")
    print(f"Overall KG gold recall rate: {summary['overall_kg_gold_recall_rate']}")
    print(f"Evidence accuracy: {summary['evidence_validation_summary']['evidence_accuracy']}")
    print(f"Report saved to: {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
