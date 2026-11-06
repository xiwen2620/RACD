
import os
import json
from pathlib import Path
from typing import Any, Dict, List


CONFIG = {
    "HARDCODED_API_KEY": os.environ.get("CLINICAL_API_KEY", "YOUR_API_KEY"),
    "API_BASE": "https://YOUR_API_ENDPOINT/v1",

    "INPUT_PATH": "/PATH/TO/YOUR/mimic.csv",

    "KG_DIR": str(Path(__file__).resolve().parent / "kg"),

    "USE_CASE_MODE": "precleaned",

    "RUN_NAME": "mimic6000_inference_only",

    "MAX_WORKERS": 25,
    "MAX_RETRIES": 3,
    "RETRY_DELAY": 5,
    "MAX_SAMPLES": None,

    "AGENT_MODELS": {
        "agent_a": "gpt-5.1",
        "agent_b": "gpt-5.1",
        "agent_c": "gpt-5.5",
        "agent_d_final": "gpt-5.5",
    },

    "AGENT_A_MAX_DDX": 20,
    "AGENT_B_MAX_HIGH_RISK_DIAGNOSES": 12,
    "AGENT_B_MAX_COMPETING_MODELS": 8,
    "MAX_PRIMARY_EXPLANATORY_MODELS": 3,
    "MAX_ADJUDICATED_EXPLANATORY_MODELS": 8,
    "MAX_RULED_OUT_EXPLANATORY_MODELS": 10,


    "KG_TOP_K": 3,
    "KG_MIN_SCORE": 1.0,
    "KG_MAX_TERMS_PER_GROUP": 8,

    "KG_STRICT_FILE_LOADING": True,
    "KG_DYNAMIC_TOP_K": True,
    "KG_MAX_DYNAMIC_TOP_K": 5,
    "KG_INCLUDE_GRAPH_EDGE_SCORE": True,
    "KG_INCLUDE_NUMERIC_FACTS": True,
    "KG_REDFLAG_FORCE_RECALL_SCORE": 2.5,
    "KG_MAX_GRAPH_EDGE_SCORE": 6.0,
    "KG_PER_DISEASE_MIN_SCORE": {
        "SEPSIS": 1.0,
        "SEPTIC_SHOCK": 1.0,
        "PULMONARY_EMBOLISM": 1.0,
        "ACUTE_MYOCARDIAL_INFARCTION": 1.0,
        "AORTIC_DISSECTION": 1.0,
        "ISCHEMIC_STROKE": 1.0,
        "INTRACRANIAL_HEMORRHAGE": 1.0,
        "MENINGITIS": 1.0,
        "ENCEPHALITIS": 1.0,
        "BOWEL_PERFORATION": 1.0,
    },

    "VALIDATE_KG_FILES_ON_STARTUP": True,
    "VALIDATE_KG_SCHEMA_ON_STARTUP": True,
    "AUTO_CREATE_OUTPUT_DIRS": True,
}

KG_FILES = [
    "acute_coronary_syndrome.json",
    "acute_myocardial_infarction.json",
    "aortic_dissection.json",
    "pulmonary_embolism.json",
    "ischemic_stroke.json",
    "intracranial_hemorrhage.json",
    "meningitis.json",
    "encephalitis.json",
    "sepsis.json",
    "septic_shock.json",
    "severe_infection_with_organ_dysfunction.json",
    "bowel_perforation.json",
]

CACHE_SCHEMA_VERSION = "v3"

file_path = Path(CONFIG["INPUT_PATH"])
kg_dir = Path(CONFIG["KG_DIR"])

output_base_dir = Path(f"./HRDSS2.0-results/{CONFIG['RUN_NAME']}")
clinical_result_dir = output_base_dir / f"clinical_results_{CACHE_SCHEMA_VERSION}"
history_dir = output_base_dir / f"dp_history_{CACHE_SCHEMA_VERSION}"
clinical_output_path = output_base_dir / f"kgtest600_{CACHE_SCHEMA_VERSION}.json"


def ensure_output_dirs() -> None:
    for p in [output_base_dir, clinical_result_dir, history_dir]:
        p.mkdir(parents=True, exist_ok=True)

def validate_config() -> None:
    if not CONFIG.get("HARDCODED_API_KEY"):
        raise ValueError("CONFIG['HARDCODED_API_KEY'] is empty.")
    if not CONFIG.get("API_BASE"):
        raise ValueError("CONFIG['API_BASE'] is empty.")

    if CONFIG.get("MAX_WORKERS", 1) <= 0:
        raise ValueError("CONFIG['MAX_WORKERS'] must be a positive integer.")

    if CONFIG.get("MAX_RETRIES", 0) < 0:
        raise ValueError("CONFIG['MAX_RETRIES'] must be >= 0.")
    if CONFIG.get("RETRY_DELAY", 0) < 0:
        raise ValueError("CONFIG['RETRY_DELAY'] must be >= 0.")

    if CONFIG.get("MAX_SAMPLES") is not None and int(CONFIG["MAX_SAMPLES"]) <= 0:
        raise ValueError("CONFIG['MAX_SAMPLES'] must be positive or None.")

    required_agents = ["agent_a", "agent_b", "agent_c", "agent_d_final"]
    missing_agents = [
        name for name in required_agents
        if name not in CONFIG.get("AGENT_MODELS", {})
    ]
    if missing_agents:
        raise ValueError(f"Missing AGENT_MODELS config for: {missing_agents}")
    if CONFIG.get("VALIDATE_KG_FILES_ON_STARTUP", True):
        validate_kg_files()

def validate_kg_files() -> None:
    if not kg_dir.exists():
        raise FileNotFoundError(f"KG directory not found: {kg_dir.resolve()}")
    missing_files = []
    for file_name in KG_FILES:
        path = kg_dir / file_name
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        missing_text = "\n".join(f"- {p}" for p in missing_files)
        raise FileNotFoundError(
            "Some KG files are missing:\n"
            f"{missing_text}"
        )

    if CONFIG.get("VALIDATE_KG_SCHEMA_ON_STARTUP", True):
        validate_kg_schema_files()


def _validate_non_empty_string(data: Dict[str, Any], key: str, file_name: str, errors: List[str]) -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{file_name}: missing or empty '{key}'")


def _validate_list(data: Dict[str, Any], key: str, file_name: str, errors: List[str], allow_empty: bool = True) -> None:
    value = data.get(key)
    if not isinstance(value, list):
        errors.append(f"{file_name}: '{key}' must be a list")
    elif not allow_empty and len(value) == 0:
        errors.append(f"{file_name}: '{key}' must not be empty")


def validate_single_kg_schema(path: Path) -> List[str]:
    errors: List[str] = []
    file_name = path.name

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"{file_name}: invalid JSON: {e}"]

    if not isinstance(data, dict):
        return [f"{file_name}: root must be a JSON object"]

    for key in ["disease_id", "disease_name_en"]:
        _validate_non_empty_string(data, key, file_name, errors)

    _validate_list(data, "graph_nodes", file_name, errors, allow_empty=False)
    _validate_list(data, "graph_edges", file_name, errors, allow_empty=True)

    case_norm = data.get("case_input_normalization", {})
    if not isinstance(case_norm, dict):
        errors.append(f"{file_name}: 'case_input_normalization' must be an object")
    else:
        rules = case_norm.get("normalization_rules", [])
        if not isinstance(rules, list):
            errors.append(f"{file_name}: normalization_rules must be a list")
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                errors.append(f"{file_name}: normalization_rules[{idx}] must be an object")
                continue
            has_trigger = bool(rule.get("input_terms") or rule.get("input_variants"))
            has_output = bool(rule.get("normalized_case_fact") or rule.get("normalized_node_id") or rule.get("normalized_node"))
            if not has_trigger and not has_output:
                errors.append(
                    f"{file_name}: normalization_rules[{idx}] has neither input terms nor normalized output"
                )

    supported_node_types = {
        "DiseaseCandidate", "DiseaseSubtype", "ClinicalContext", "Concept", "Assessment",
        "Symptom", "Sign", "RiskFactor", "Finding", "DiagnosticData",
        "LaboratoryTest", "MicrobiologyTest", "ImagingTest", "DiagnosticCriterion",
        "SyndromeComponent", "OrganDysfunction", "RiskScore", "TreatmentTarget",
        "MedicationClassOrAgent", "ManagementAction", "DifferentialDiagnosis",
    }

    node_ids = set()
    for idx, node in enumerate(data.get("graph_nodes", []) or []):
        if not isinstance(node, dict):
            errors.append(f"{file_name}: graph_nodes[{idx}] must be an object")
            continue
        node_id = node.get("node_id")
        node_type = node.get("node_type")
        name = node.get("name")
        if not isinstance(node_id, str) or not node_id.strip():
            errors.append(f"{file_name}: graph_nodes[{idx}] missing node_id")
        else:
            node_ids.add(node_id)
        if node_type not in supported_node_types:
            errors.append(f"{file_name}: graph_nodes[{idx}] unsupported node_type={node_type!r}")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{file_name}: graph_nodes[{idx}] missing name")

    for idx, edge in enumerate(data.get("graph_edges", []) or []):
        if not isinstance(edge, dict):
            errors.append(f"{file_name}: graph_edges[{idx}] must be an object")
            continue
        relation = edge.get("relation")
        if not isinstance(relation, str) or not relation.strip():
            errors.append(f"{file_name}: graph_edges[{idx}] missing relation")
        for endpoint in ["source", "target"]:
            value = edge.get(endpoint)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{file_name}: graph_edges[{idx}] missing {endpoint}")

    has_diagnostic_or_lab_node = any(
        isinstance(node, dict) and node.get("node_type") in {"DiagnosticData", "LaboratoryTest", "MicrobiologyTest", "ImagingTest"}
        for node in data.get("graph_nodes", []) or []
    )
    has_guideline_pathway = isinstance(data.get("guideline_verification_pathway"), list) and len(data.get("guideline_verification_pathway", [])) > 0
    if not has_diagnostic_or_lab_node and not has_guideline_pathway:
        errors.append(f"{file_name}: no diagnostic/lab/imaging nodes and no guideline_verification_pathway")

    return errors


def validate_kg_schema_files() -> None:
    all_errors: List[str] = []
    for path in get_kg_paths():
        if path.exists():
            all_errors.extend(validate_single_kg_schema(path))

    if all_errors:
        error_text = "\n".join(f"- {e}" for e in all_errors[:80])
        if len(all_errors) > 80:
            error_text += f"\n- ... and {len(all_errors) - 80} more schema errors"
        raise ValueError(
            "KG schema validation failed. Fix the KG files before running retrieval:\n"
            f"{error_text}"
        )


def get_kg_paths() -> list[Path]:
    """
    返回所有 KG 文件路径。
    kg_retrieval.py 后面可以直接调用这个函数，避免重复定义 KG_FILES。
    """
    return [kg_dir / file_name for file_name in KG_FILES]


if CONFIG.get("AUTO_CREATE_OUTPUT_DIRS", True):
    ensure_output_dirs()
