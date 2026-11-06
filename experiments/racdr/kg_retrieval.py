
import json

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable


DEFAULT_KG_FILES = [
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

try:
    from config import CONFIG
except Exception:
    CONFIG = {}

try:
    from config import KG_FILES as CONFIG_KG_FILES
except Exception:
    CONFIG_KG_FILES = DEFAULT_KG_FILES


KG_DIR = Path(CONFIG.get("KG_DIR", "./kg"))
KG_FILES = CONFIG_KG_FILES

KG_MIN_SCORE = float(CONFIG.get("KG_MIN_SCORE", 1.0))
KG_MAX_TERMS_PER_GROUP = int(CONFIG.get("KG_MAX_TERMS_PER_GROUP", 8))
KG_STRICT_FILE_LOADING = bool(CONFIG.get("KG_STRICT_FILE_LOADING", True))
KG_DYNAMIC_TOP_K = bool(CONFIG.get("KG_DYNAMIC_TOP_K", True))
KG_MAX_DYNAMIC_TOP_K = int(CONFIG.get("KG_MAX_DYNAMIC_TOP_K", max(5, int(CONFIG.get("KG_TOP_K", 3)))))
KG_INCLUDE_GRAPH_EDGE_SCORE = bool(CONFIG.get("KG_INCLUDE_GRAPH_EDGE_SCORE", True))
KG_INCLUDE_NUMERIC_FACTS = bool(CONFIG.get("KG_INCLUDE_NUMERIC_FACTS", True))
KG_PER_DISEASE_MIN_SCORE = CONFIG.get("KG_PER_DISEASE_MIN_SCORE", {}) or {}
KG_REDFLAG_FORCE_RECALL_SCORE = float(CONFIG.get("KG_REDFLAG_FORCE_RECALL_SCORE", 2.5))


CLINICAL_SYNONYM_EXPANSIONS = {
    "dyspnea": [
        "shortness of breath",
        "short of breath",
        "sob",
        "difficulty breathing",
        "breathlessness",
    ],
    "shortness of breath": [
        "dyspnea",
        "sob",
        "difficulty breathing",
        "breathlessness",
    ],
    "chest pressure, heaviness, tightness, or pain": [
        "chest pain",
        "chest pressure",
        "chest tightness",
        "chest heaviness",
        "substernal pain",
        "retrosternal pain",
    ],
    "pain radiating to arm, jaw, neck, shoulder, back, or epigastrium": [
        "radiating pain",
        "pain radiates",
        "radiation to arm",
        "radiates to arm",
        "jaw pain",
        "arm pain",
        "left arm pain",
        "shoulder pain",
        "back pain",
        "epigastric pain",
    ],
    "nausea or vomiting": [
        "nausea",
        "vomiting",
        "emesis",
    ],
    "lightheadedness or syncope": [
        "lightheadedness",
        "syncope",
        "fainting",
        "passed out",
        "presyncope",
    ],
    "hemodynamic instability": [
        "hypotension",
        "shock",
        "unstable vital signs",
        "hemodynamically unstable",
    ],
    "signs of poor perfusion": [
        "poor perfusion",
        "cool extremities",
        "mottled",
        "altered mental status",
        "low urine output",
    ],
    "pulmonary embolism": [
        "pe",
        "pulmonary embolus",
    ],
    "acute coronary syndrome": [
        "acs",
        "unstable angina",
        "nstemi",
        "stemi",
        "nste-acs",
    ],
    "acute myocardial infarction": [
        "ami",
        "myocardial infarction",
        "heart attack",
        "mi",
        "nstemi",
        "stemi",
    ],
    "intracranial hemorrhage": [
        "ich",
        "brain bleed",
        "intracerebral hemorrhage",
        "subarachnoid hemorrhage",
        "sah",
    ],
    "ischemic stroke": [
        "stroke",
        "cva",
        "cerebrovascular accident",
        "acute ischemic stroke",
    ],
    "sepsis": [
        "septic",
        "systemic infection",
        "bloodstream infection",
        "urosepsis",
    ],
    "septic shock": [
        "shock from sepsis",
        "sepsis with hypotension",
        "vasopressor requirement",
    ],
    "aortic dissection": [
        "dissection",
        "aortic tear",
    ],
    "bowel perforation": [
        "perforated bowel",
        "intestinal perforation",
        "free air",
        "pneumoperitoneum",
    ],
}

FACT_SYNONYMS = {
    "tachycardia": ["tachycardia", "tachycardic", "fast heart rate", "rapid heart rate", "hr high", "pulse high"],
    "severe_tachycardia": ["marked tachycardia", "severe tachycardia"],
    "tachypnea": ["tachypnea", "tachypneic", "rapid respirations", "rapid respiratory rate", "rr high"],
    "hypotension": ["hypotension", "hypotensive", "low blood pressure", "low bp", "sbp low", "map low"],
    "shock_physiology": ["shock", "hemodynamic instability", "hemodynamically unstable", "poor perfusion"],
    "hypoxemia": ["hypoxemia", "hypoxaemia", "hypoxic", "low oxygen saturation", "low spo2", "desaturation"],
    "fever_or_hypothermia": ["fever", "febrile", "pyrexia", "high temperature", "hypothermia", "low temperature"],
    "fever": ["fever", "febrile", "pyrexia", "high temperature"],
    "hypothermia": ["hypothermia", "low temperature"],
    "leukocytosis_or_leukopenia": ["leukocytosis", "leukopenia", "abnormal wbc", "white count high", "white count low"],
    "thrombocytopenia": ["thrombocytopenia", "low platelets", "platelets low"],
    "renal_dysfunction": ["renal dysfunction", "acute kidney injury", "aki", "creatinine elevated", "elevated creatinine"],
    "liver_dysfunction": ["hepatic dysfunction", "hyperbilirubinemia", "bilirubin elevated", "elevated bilirubin"],
    "elevated_lactate": ["elevated lactate", "high lactate", "lactate high", "lactic acidosis", "hyperlactatemia"],
    "lactate_gt_2": ["lactate >2", "lactate greater than 2", "lactate over 2"],
    "lactate_ge_4": ["lactate >=4", "lactate 4", "lactate greater than 4", "lactate over 4"],
    "metabolic_acidosis": ["metabolic acidosis", "acidemia", "low ph"],
    "altered_mental_status": ["altered mental status", "confusion", "confused", "encephalopathy", "delirium", "ams"],
    "infection_suspected": ["suspected infection", "infection", "infected", "pneumonia", "uti", "cellulitis", "bacteremia", "septic"],
    "vasopressor_requirement": ["vasopressor", "pressors", "pressor support", "norepinephrine", "levophed", "noradrenaline"],
}


NEGATION_BEFORE_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bfree of\b",
]

NEGATION_AFTER_PATTERNS = [
    r"\babsent\b",
    r"\bdenied\b",
    r"\bnegative\b",
    r"\bruled out\b",
]

UNCERTAINTY_BEFORE_PATTERNS = [
    r"\bpossible\b",
    r"\bprobable\b",
    r"\bsuspected\b",
    r"\bconcern for\b",
    r"\bevaluate for\b",
    r"\bworkup for\b",
    r"\brule out\b",
    r"\br/o\b",
    r"\?\s*$",
]

HISTORICAL_BEFORE_PATTERNS = [
    r"\bhistory of\b",
    r"\bhx of\b",
    r"\bprior\b",
    r"\bprevious\b",
    r"\bremote\b",
]

def load_kg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: Any) -> str:
    text = str(text).lower()
    text = text.replace("≥", ">=").replace("≤", "<=")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_token(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_text(text)).strip("_")


def flatten_case(structured_case: Dict[str, Any]) -> str:
    fields = flatten_case_fields(structured_case)
    return normalize_text(" ".join(item["text"] for item in fields))


def flatten_case_fields(obj: Any, prefix: str = "case") -> List[Dict[str, str]]:
    fields: List[Dict[str, str]] = []

    if obj is None:
        return fields
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}"
            fields.extend(flatten_case_fields(value, child_prefix))
        return fields

    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]"
            fields.extend(flatten_case_fields(value, child_prefix))
        return fields

    text = str(obj).strip()
    if text:
        fields.append(
            {
                "field": prefix,
                "text": text,
                "norm_text": normalize_text(text),
            }
        )

    return fields


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _temp_to_celsius(value: float, unit: str = "") -> float:
    unit_l = unit.lower()
    if unit_l.startswith("f") or value > 45:
        return (value - 32.0) * 5.0 / 9.0
    return value


def _clean_unit(unit: str) -> str:
    return re.sub(r"\s+", " ", unit or "").strip()


def _add_fact(
    facts: List[Dict[str, Any]],
    *,
    fact_id: str,
    canonical: str,
    label: str,
    value: Optional[float],
    unit: str,
    field: str,
    text: str,
    start: int,
    end: int,
    polarity: str,
    severity: str = "present",
    source: str = "numeric_rule",
) -> None:
    context = get_context_window(text, start, end)
    matched_text = text[start:end]
    dedup_key = (canonical, field, round(value, 4) if isinstance(value, (int, float)) else None, matched_text, polarity)
    for item in facts:
        existing_key = (
            item.get("canonical"),
            item.get("field"),
            round(item.get("value"), 4) if isinstance(item.get("value"), (int, float)) else None,
            item.get("matched_text"),
            item.get("polarity"),
        )
        if existing_key == dedup_key:
            return
    facts.append(
        {
            "fact_id": fact_id,
            "canonical": canonical,
            "label": label,
            "value": value,
            "unit": _clean_unit(unit),
            "field": field,
            "matched_text": matched_text,
            "polarity": polarity,
            "severity": severity,
            "context": context,
            "source": source,
        }
    )


def _iter_regex_matches(text: str, patterns: Iterable[str]) -> Iterable[re.Match]:
    for pattern in patterns:
        yield from re.finditer(pattern, text, flags=re.IGNORECASE)


def _extract_numeric_facts_from_field(field_name: str, text: str) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []

    hr_patterns = [
        r"\b(?:hr|heart\s*rate|pulse)\s*[:=]?\s*(\d{2,3})(?:\s*(?:bpm|/min))?\b",
        r"\b(?:p)\s*[:=]\s*(\d{2,3})(?:\s*(?:bpm|/min))?\b",
    ]
    for m in _iter_regex_matches(text, hr_patterns):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value >= 100:
            _add_fact(facts, fact_id="FACT_TACHYCARDIA", canonical="tachycardia", label="Tachycardia", value=value, unit="bpm", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if value >= 120 else "present")
        if value >= 120:
            _add_fact(facts, fact_id="FACT_SEVERE_TACHYCARDIA", canonical="severe_tachycardia", label="Severe tachycardia", value=value, unit="bpm", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe")

    rr_patterns = [
        r"\b(?:rr|resp(?:iratory)?\s*rate|respirations?)\s*[:=]?\s*(\d{1,3})(?:\s*(?:/min|breaths/min|bpm))?\b",
    ]
    for m in _iter_regex_matches(text, rr_patterns):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value >= 22:
            _add_fact(facts, fact_id="FACT_TACHYPNEA", canonical="tachypnea", label="Tachypnea", value=value, unit="breaths/min", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if value >= 30 else "present")

    bp_patterns = [
        r"\b(?:bp|blood\s*pressure)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b",
        r"\b(?:sbp)\s*[:=]?\s*(\d{2,3})\b",
    ]
    for m in _iter_regex_matches(text, bp_patterns):
        sbp = _safe_float(m.group(1))
        dbp = _safe_float(m.group(2)) if len(m.groups()) >= 2 else None
        if sbp is None:
            continue
        map_value = None
        if dbp is not None:
            map_value = (sbp + 2 * dbp) / 3
        polarity = detect_polarity(text, m.start(), m.end())
        if sbp < 90 or (map_value is not None and map_value < 65):
            _add_fact(facts, fact_id="FACT_HYPOTENSION", canonical="hypotension", label="Hypotension", value=sbp, unit="mmHg SBP", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if sbp < 80 else "present")
            _add_fact(facts, fact_id="FACT_SHOCK_PHYSIOLOGY", canonical="shock_physiology", label="Shock physiology / hemodynamic instability", value=sbp, unit="mmHg SBP", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if sbp < 80 else "present")

    for m in _iter_regex_matches(text, [r"\bmap\s*[:=]?\s*(\d{2,3})(?:\s*mm\s*hg)?\b"]):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value < 65:
            _add_fact(facts, fact_id="FACT_HYPOTENSION", canonical="hypotension", label="Hypotension", value=value, unit="mmHg MAP", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")
            _add_fact(facts, fact_id="FACT_SHOCK_PHYSIOLOGY", canonical="shock_physiology", label="Shock physiology / hemodynamic instability", value=value, unit="mmHg MAP", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")

    spo2_patterns = [
        r"\b(?:spo2|sp\s*o2|o2\s*sat(?:uration)?|oxygen\s*saturation|sat)\s*[:=]?\s*(\d{2,3})\s*%?\b",
    ]
    for m in _iter_regex_matches(text, spo2_patterns):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value < 90:
            _add_fact(facts, fact_id="FACT_HYPOXEMIA", canonical="hypoxemia", label="Hypoxemia", value=value, unit="%", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe")
        elif value <= 94:
            _add_fact(facts, fact_id="FACT_HYPOXEMIA", canonical="hypoxemia", label="Hypoxemia / reduced oxygen saturation", value=value, unit="%", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="mild")

    temp_patterns = [
        r"\b(?:temp(?:erature)?|tmax|t)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(°?\s*[cf])?\b",
    ]
    for m in _iter_regex_matches(text, temp_patterns):
        raw = _safe_float(m.group(1))
        if raw is None:
            continue
        unit = m.group(2) or "C_or_F"
        value_c = _temp_to_celsius(raw, unit)
        polarity = detect_polarity(text, m.start(), m.end())
        if value_c >= 38.0:
            _add_fact(facts, fact_id="FACT_FEVER", canonical="fever", label="Fever", value=round(value_c, 2), unit="C", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")
            _add_fact(facts, fact_id="FACT_FEVER_OR_HYPOTHERMIA", canonical="fever_or_hypothermia", label="Fever or hypothermia", value=round(value_c, 2), unit="C", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")
        elif value_c < 36.0:
            _add_fact(facts, fact_id="FACT_HYPOTHERMIA", canonical="hypothermia", label="Hypothermia", value=round(value_c, 2), unit="C", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")
            _add_fact(facts, fact_id="FACT_FEVER_OR_HYPOTHERMIA", canonical="fever_or_hypothermia", label="Fever or hypothermia", value=round(value_c, 2), unit="C", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")

    for m in _iter_regex_matches(text, [r"\b(?:wbc|white\s*(?:blood\s*)?(?:cell\s*)?(?:count)?)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:k|x10\^?9/l|10\^?3/ul|/ul)?\b"]):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        value_k = value / 1000 if value > 1000 else value
        if value_k > 12 or value_k < 4:
            _add_fact(facts, fact_id="FACT_ABNORMAL_WBC", canonical="leukocytosis_or_leukopenia", label="Leukocytosis or leukopenia", value=round(value_k, 2), unit="K/uL", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")

    for m in _iter_regex_matches(text, [r"\b(?:platelets?|plt)\s*[:=]?\s*(\d{1,4}(?:\.\d+)?)\s*(?:k|x10\^?9/l|10\^?3/ul|/ul)?\b"]):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        value_k = value / 1000 if value > 1000 else value
        polarity = detect_polarity(text, m.start(), m.end())
        if value_k < 150:
            _add_fact(facts, fact_id="FACT_THROMBOCYTOPENIA", canonical="thrombocytopenia", label="Thrombocytopenia", value=round(value_k, 2), unit="K/uL", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if value_k < 100 else "present")

    for m in _iter_regex_matches(text, [r"\b(?:creatinine|creat|cr)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:mg/dl|umol/l|µmol/l)?\b"]):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value >= 1.5:
            _add_fact(facts, fact_id="FACT_RENAL_DYSFUNCTION", canonical="renal_dysfunction", label="Renal dysfunction / elevated creatinine", value=value, unit="mg/dL_or_reported", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if value >= 2.0 else "present")

    for m in _iter_regex_matches(text, [r"\b(?:bilirubin|total\s*bili|t\.?\s*bili|tbili)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:mg/dl|umol/l|µmol/l)?\b"]):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value >= 2.0:
            _add_fact(facts, fact_id="FACT_LIVER_DYSFUNCTION", canonical="liver_dysfunction", label="Liver dysfunction / hyperbilirubinemia", value=value, unit="mg/dL_or_reported", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")

    for m in _iter_regex_matches(text, [r"\b(?:lactate|lactic\s*acid)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:mmol/l|mg/dl)?\b"]):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value > 2:
            _add_fact(facts, fact_id="FACT_ELEVATED_LACTATE", canonical="elevated_lactate", label="Elevated lactate", value=value, unit="mmol/L_or_reported", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if value >= 4 else "present")
            _add_fact(facts, fact_id="FACT_LACTATE_GT_2", canonical="lactate_gt_2", label="Lactate > 2", value=value, unit="mmol/L_or_reported", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="present")
        if value >= 4:
            _add_fact(facts, fact_id="FACT_LACTATE_GE_4", canonical="lactate_ge_4", label="Lactate >= 4", value=value, unit="mmol/L_or_reported", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe")
            _add_fact(facts, fact_id="FACT_SHOCK_PHYSIOLOGY", canonical="shock_physiology", label="Shock physiology / tissue hypoperfusion", value=value, unit="lactate mmol/L_or_reported", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe")

    for m in _iter_regex_matches(text, [r"\bpH\s*[:=]?\s*(7\.\d{1,3}|6\.\d{1,3})\b"]):
        value = _safe_float(m.group(1))
        if value is None:
            continue
        polarity = detect_polarity(text, m.start(), m.end())
        if value < 7.35:
            _add_fact(facts, fact_id="FACT_METABOLIC_ACIDOSIS", canonical="metabolic_acidosis", label="Acidemia / possible metabolic acidosis", value=value, unit="pH", field=field_name, text=text, start=m.start(), end=m.end(), polarity=polarity, severity="severe" if value < 7.2 else "present")

    return facts


def _extract_textual_facts_from_field(field_name: str, text: str) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for canonical, phrases in FACT_SYNONYMS.items():
        for phrase in phrases:
            pattern = build_variant_regex(normalize_text(phrase))
            for m in pattern.finditer(text):
                polarity = detect_polarity(text, m.start(), m.end())
                _add_fact(
                    facts,
                    fact_id=f"FACT_{normalize_token(canonical).upper()}",
                    canonical=canonical,
                    label=canonical.replace("_", " ").title(),
                    value=None,
                    unit="",
                    field=field_name,
                    text=text,
                    start=m.start(),
                    end=m.end(),
                    polarity=polarity,
                    severity="present",
                    source="textual_fact_rule",
                )
    return facts


def extract_clinical_facts(case_fields: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not KG_INCLUDE_NUMERIC_FACTS:
        return []

    facts: List[Dict[str, Any]] = []
    seen = set()

    for field_item in case_fields:
        field = field_item.get("field", "case")
        text = field_item.get("text", "")
        if not text:
            continue

        for fact in _extract_numeric_facts_from_field(field, text) + _extract_textual_facts_from_field(field, text):
            key = (
                fact.get("canonical"),
                fact.get("field"),
                fact.get("matched_text"),
                fact.get("polarity"),
            )
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)

    return facts


def extract_node_terms(kg: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {
        "aliases": [],
        "clinical_contexts": [],
        "symptoms": [],
        "signs": [],
        "risk_factors": [],
        "findings": [],
        "critical_tests": [],
        "organ_dysfunction": [],
        "diagnostic_criteria": [],
        "treatments": [],
        "management_actions": [],
        "mimics": [],
    }

    disease_name = kg.get("disease_name_en", "")
    if isinstance(disease_name, str) and disease_name.strip():
        groups["aliases"].append(
            {
                "term": disease_name.strip(),
                "node_id": "disease_name_en",
                "node_type": "DiseaseCandidate",
                "node_name": disease_name.strip(),
            }
        )

    for alias in kg.get("aliases", []):
        if isinstance(alias, str) and alias.strip():
            groups["aliases"].append(
                {
                    "term": alias.strip(),
                    "node_id": "top_level_alias",
                    "node_type": "Alias",
                    "node_name": alias.strip(),
                }
            )

    node_type_to_group = {
        "DiseaseCandidate": "aliases",
        "DiseaseSubtype": "aliases",
        "ClinicalContext": "clinical_contexts",
        "Concept": "clinical_contexts",
        "Assessment": "clinical_contexts",
        "Symptom": "symptoms",
        "Sign": "signs",
        "RiskFactor": "risk_factors",
        "Finding": "findings",
        "DiagnosticData": "critical_tests",
        "LaboratoryTest": "critical_tests",
        "MicrobiologyTest": "critical_tests",
        "ImagingTest": "critical_tests",
        "DiagnosticCriterion": "diagnostic_criteria",
        "SyndromeComponent": "diagnostic_criteria",
        "OrganDysfunction": "organ_dysfunction",
        "RiskScore": "diagnostic_criteria",
        "TreatmentTarget": "treatments",
        "MedicationClassOrAgent": "treatments",
        "ManagementAction": "management_actions",
        "DifferentialDiagnosis": "mimics",
    }

    for node in kg.get("graph_nodes", []):
        if not isinstance(node, dict):
            continue

        node_type = node.get("node_type", "")
        node_id = node.get("node_id", "")
        node_name = node.get("name", "")
        synonyms = node.get("synonyms", [])

        group = node_type_to_group.get(node_type)
        if not group:
            continue

        raw_terms = [node_name] + synonyms
        terms = [str(t).strip() for t in raw_terms if isinstance(t, str) and str(t).strip()]

        for term in terms:
            groups[group].append(
                {
                    "term": term,
                    "node_id": node_id,
                    "node_type": node_type,
                    "node_name": node_name,
                }
            )

    normalization_rules = kg.get("case_input_normalization", {}).get("normalization_rules", [])
    for rule in normalization_rules:
        if not isinstance(rule, dict):
            continue

        normalized_case_fact = str(rule.get("normalized_case_fact", "") or "").strip()
        normalized_node_id = str(rule.get("normalized_node_id", "") or "").strip()
        normalized_node = str(rule.get("normalized_node", "") or "").strip()
        normalized_node_name = normalized_node or normalized_node_id or normalized_case_fact or "normalization_rule"

        raw_terms = []
        raw_terms.extend(rule.get("input_terms", []) or [])
        raw_terms.extend(rule.get("input_variants", []) or [])

        for t in [normalized_case_fact, normalized_node, normalized_node_id.replace("_", " ")]:
            if t:
                raw_terms.append(t)
        if normalized_case_fact:
            raw_terms.extend(FACT_SYNONYMS.get(normalized_case_fact, []))

        for term in raw_terms:
            if not isinstance(term, str) or not term.strip():
                continue

            groups["clinical_contexts"].append(
                {
                    "term": term.strip(),
                    "node_id": normalized_node_id or str(rule.get("rule_id", "")),
                    "node_type": "NormalizationRule",
                    "node_name": normalized_node_name,
                }
            )

    for key in groups:
        seen = set()
        cleaned = []
        for item in groups[key]:
            term_key = normalize_text(item.get("term", ""))
            if term_key and term_key not in seen:
                cleaned.append(item)
                seen.add(term_key)
        groups[key] = cleaned

    return groups


def get_term_variants(term: str) -> List[str]:
    variants = []
    term_norm = normalize_text(term)
    if term_norm:
        variants.append(term_norm)

    for extra in CLINICAL_SYNONYM_EXPANSIONS.get(term_norm, []):
        extra_norm = normalize_text(extra)
        if extra_norm and extra_norm not in variants:
            variants.append(extra_norm)

    token = normalize_token(term_norm)
    for extra in FACT_SYNONYMS.get(token, []):
        extra_norm = normalize_text(extra)
        if extra_norm and extra_norm not in variants:
            variants.append(extra_norm)

    return variants


def build_variant_regex(variant_norm: str) -> re.Pattern:
    escaped = re.escape(variant_norm)
    escaped = escaped.replace(r"\ ", r"\s+")
    pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def get_context_window(text: str, start: int, end: int, window: int = 80) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def detect_polarity(text: str, start: int, end: int) -> str:
    before = normalize_text(text[max(0, start - 100):start])
    after = normalize_text(text[end:min(len(text), end + 80)])

    for pattern in NEGATION_BEFORE_PATTERNS:
        if re.search(pattern + r"[^.;:\n]{0,80}$", before):
            return "negative"

    for pattern in NEGATION_AFTER_PATTERNS:
        if re.search(r"^[^.;:\n]{0,50}" + pattern, after):
            return "negative"
    for pattern in UNCERTAINTY_BEFORE_PATTERNS:
        if re.search(pattern + r"[^.;:\n]{0,80}$", before):
            return "uncertain"

    for pattern in HISTORICAL_BEFORE_PATTERNS:
        if re.search(pattern + r"[^.;:\n]{0,80}$", before):
            return "historical"

    return "positive"


def polarity_multiplier(polarity: str) -> float:
    if polarity == "positive":
        return 1.0
    if polarity == "uncertain":
        return 0.5
    if polarity == "historical":
        return 0.25
    return 0.0


def find_term_matches(
    term_item: Dict[str, str],
    case_fields: List[Dict[str, str]],
    max_matches_per_term: int = 5,
) -> List[Dict[str, Any]]:
    term = term_item.get("term", "")
    variants = get_term_variants(term)

    matches: List[Dict[str, Any]] = []
    seen = set()

    for field_item in case_fields:
        field_name = field_item["field"]
        text = field_item["text"]

        for variant in variants:
            if not variant:
                continue

            pattern = build_variant_regex(variant)
            for match in pattern.finditer(text):
                matched_text = match.group(0)
                polarity = detect_polarity(text, match.start(), match.end())
                context = get_context_window(text, match.start(), match.end())
                dedup_key = (field_name, normalize_text(matched_text), variant, polarity, context)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                matches.append(
                    {
                        "term": term,
                        "matched_text": matched_text,
                        "term_variant": variant,
                        "field": field_name,
                        "polarity": polarity,
                        "context": context,
                        "node_id": term_item.get("node_id", ""),
                        "node_type": term_item.get("node_type", ""),
                        "node_name": term_item.get("node_name", ""),
                        "match_source": "term_or_synonym",
                    }
                )
                if len(matches) >= max_matches_per_term:
                    return matches
    return matches


def build_node_index(kg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index = {}
    for node in kg.get("graph_nodes", []) or []:
        if isinstance(node, dict) and node.get("node_id"):
            index[str(node["node_id"])] = node
    return index


def _node_tokens_for_match(node: Dict[str, Any]) -> List[str]:
    raw = [node.get("node_id", ""), node.get("name", "")]
    raw.extend(node.get("synonyms", []) or [])
    tokens = []
    for item in raw:
        if not item:
            continue
        norm = normalize_token(item)
        if norm:
            tokens.append(norm)
        txt = normalize_text(item)
        if txt:
            tokens.append(txt)
    return list(dict.fromkeys(tokens))


def _canonical_variants(canonical: str) -> List[str]:
    variants = [normalize_token(canonical), normalize_text(canonical).replace("_", " ")]
    variants.extend(normalize_text(x) for x in FACT_SYNONYMS.get(canonical, []))
    if canonical == "lactate_gt_2":
        variants.extend(["lactate gt 2", "lactate > 2", "lactate greater than 2", "lactate 2"])
    if canonical == "lactate_ge_4":
        variants.extend(["lactate ge 4", "lactate >= 4", "lactate greater than 4", "lactate 4"])
    return [v for v in dict.fromkeys(variants) if v]


def fact_matches_node_or_rule(fact: Dict[str, Any], kg: Dict[str, Any], node: Optional[Dict[str, Any]] = None, rule: Optional[Dict[str, Any]] = None) -> bool:
    canonical = str(fact.get("canonical", ""))
    variants = _canonical_variants(canonical)

    if node and node.get("node_type") in {"DiseaseCandidate", "DiseaseSubtype", "DifferentialDiagnosis"}:
        return False

    if node and node.get("node_type") in {"ManagementAction", "TreatmentTarget", "MedicationClassOrAgent"}:
        if canonical not in {"vasopressor_requirement"}:
            return False

    haystacks: List[str] = []
    if node:
        haystacks.extend(_node_tokens_for_match(node))
    if rule:
        for key in ["normalized_case_fact", "normalized_node_id", "normalized_node", "rule_id"]:
            value = str(rule.get(key, "") or "")
            if value:
                haystacks.append(normalize_token(value))
                haystacks.append(normalize_text(value).replace("_", " "))
        for term in (rule.get("input_terms", []) or []) + (rule.get("input_variants", []) or []):
            haystacks.append(normalize_text(term))
            haystacks.append(normalize_token(term))

    haystacks = [h for h in haystacks if h]
    for variant in variants:
        v_token = normalize_token(variant)
        v_text = normalize_text(variant).replace("_", " ")
        for hay in haystacks:
            hay_token = normalize_token(hay)
            hay_text = normalize_text(hay).replace("_", " ")
            if not hay_token:
                continue
            if v_token == hay_token:
                return True
            if len(v_token) >= 4 and v_token in hay_token:
                return True
            if len(v_text) >= 4 and v_text in hay_text:
                return True
    return False


def match_clinical_facts_to_kg(kg: Dict[str, Any], clinical_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    node_index = build_node_index(kg)
    rules = kg.get("case_input_normalization", {}).get("normalization_rules", []) or []
    matches: List[Dict[str, Any]] = []
    seen = set()

    for fact in clinical_facts:
        if fact.get("polarity") == "negative":
            multiplier = 0.0
        else:
            multiplier = polarity_multiplier(str(fact.get("polarity", "positive")))

        for node_id, node in node_index.items():
            if not fact_matches_node_or_rule(fact, kg, node=node):
                continue
            key = (fact.get("canonical"), fact.get("field"), fact.get("matched_text"), node_id)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "term": fact.get("label") or fact.get("canonical"),
                    "matched_text": fact.get("matched_text", ""),
                    "term_variant": fact.get("canonical", ""),
                    "field": fact.get("field", ""),
                    "polarity": fact.get("polarity", "positive"),
                    "context": fact.get("context", ""),
                    "node_id": node_id,
                    "node_type": node.get("node_type", "ClinicalFact"),
                    "node_name": node.get("name", ""),
                    "match_source": fact.get("source", "clinical_fact"),
                    "fact_id": fact.get("fact_id", ""),
                    "canonical_fact": fact.get("canonical", ""),
                    "value": fact.get("value"),
                    "unit": fact.get("unit", ""),
                    "severity": fact.get("severity", "present"),
                    "score_multiplier": multiplier,
                }
            )

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if not fact_matches_node_or_rule(fact, kg, rule=rule):
                continue
            normalized_node_id = str(rule.get("normalized_node_id", "") or "")
            key = (fact.get("canonical"), fact.get("field"), fact.get("matched_text"), normalized_node_id or rule.get("rule_id"))
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "term": fact.get("label") or fact.get("canonical"),
                    "matched_text": fact.get("matched_text", ""),
                    "term_variant": fact.get("canonical", ""),
                    "field": fact.get("field", ""),
                    "polarity": fact.get("polarity", "positive"),
                    "context": fact.get("context", ""),
                    "node_id": normalized_node_id,
                    "node_type": "NormalizationRuleFact",
                    "node_name": str(rule.get("normalized_node") or rule.get("normalized_case_fact") or normalized_node_id),
                    "match_source": fact.get("source", "clinical_fact"),
                    "fact_id": fact.get("fact_id", ""),
                    "canonical_fact": fact.get("canonical", ""),
                    "value": fact.get("value"),
                    "unit": fact.get("unit", ""),
                    "severity": fact.get("severity", "present"),
                    "score_multiplier": multiplier,
                }
            )

    return matches


RELATION_MULTIPLIERS = {
    "defined_by": 1.40,
    "core_clinical_criterion_for": 1.50,
    "required_context_for": 1.35,
    "has_red_flag": 1.30,
    "has_common_or_red_flag_symptom": 1.15,
    "has_clinical_context": 0.90,
    "may_present_with": 0.70,
    "supports_hypoperfusion_or_shock_if_elevated_in": 1.30,
    "supports_organ_dysfunction_if_abnormal_in": 1.20,
    "requires_assessment_of": 0.35,
    "management_requires": 0.25,
    "guideline_recommends": 0.20,
    "guideline_recommends_when_indicated": 0.20,
    "recommends": 0.20,
    "cannot_independently_rule_in_or_rule_out": 0.00,
    "differential_diagnosis": 0.15,
    "is_subset_of": 0.00,
    "is_severe_subtype_of": 0.00,
}

STRENGTH_MULTIPLIERS = {
    "definitional": 1.50,
    "critical": 1.35,
    "strong": 1.20,
    "important": 1.05,
    "moderate": 0.85,
    "supportive": 0.70,
    "weak": 0.45,
    "unspecified": 1.0,
    "": 1.0,
}

RESTRICTIVE_RELATION_HINTS = [
    "cannot_independently_rule_in_or_rule_out",
    "requires_assessment_of",
    "management_requires",
    "follow_up",
]


def _match_node_ids_from_matches(kg: Dict[str, Any], matched_terms: Dict[str, List[Dict[str, Any]]]) -> set:
    node_index = build_node_index(kg)
    node_ids = set()
    for group_name, matches in matched_terms.items():
        if group_name in {"aliases", "graph_edges"}:
            continue
        for match in matches:
            if match.get("polarity") not in ["positive", "uncertain"]:
                continue
            node_id = str(match.get("node_id", "") or "")
            if not node_id:
                continue
            node = node_index.get(node_id, {})
            if node.get("node_type") in {"DiseaseCandidate", "DiseaseSubtype", "DifferentialDiagnosis"}:
                continue
            if node_id in {"disease_name_en", "top_level_alias"}:
                continue
            node_ids.add(node_id)
    return node_ids


def graph_edge_score(
    kg: Dict[str, Any],
    matched_terms: Dict[str, List[Dict[str, Any]]],
) -> Tuple[float, List[Dict[str, Any]]]:
    if not KG_INCLUDE_GRAPH_EDGE_SCORE:
        return 0.0, []

    matched_node_ids = _match_node_ids_from_matches(kg, matched_terms)
    if not matched_node_ids:
        return 0.0, []

    edge_hits: List[Dict[str, Any]] = []
    score = 0.0

    for edge in kg.get("graph_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "") or "")
        target = str(edge.get("target", "") or "")
        if source not in matched_node_ids and target not in matched_node_ids:
            continue

        relation = str(edge.get("relation", "") or "")
        strength = str(edge.get("strength", "") or "")
        weight = _safe_float(edge.get("weight"))
        if weight is None:
            weight = 0.5

        relation_multiplier = RELATION_MULTIPLIERS.get(relation, 0.75)
        strength_multiplier = STRENGTH_MULTIPLIERS.get(strength, 1.0)

        restrictive = any(h in relation for h in RESTRICTIVE_RELATION_HINTS)
        if restrictive:
            relation_multiplier = min(relation_multiplier, 0.35)

        contribution = max(0.0, float(weight) * relation_multiplier * strength_multiplier)
        if contribution <= 0:
            continue

        score += contribution
        edge_hits.append(
            {
                "edge_id": edge.get("edge_id", ""),
                "source": source,
                "relation": relation,
                "target": target,
                "strength": strength,
                "weight": weight,
                "contribution": round(contribution, 4),
                "evidence_id": edge.get("evidence_id", ""),
                "clinical_note": edge.get("clinical_note", ""),
                "restrictive_or_assessment_edge": restrictive,
            }
        )

    score = min(score, float(CONFIG.get("KG_MAX_GRAPH_EDGE_SCORE", 6.0)))
    edge_hits = sorted(edge_hits, key=lambda e: e.get("contribution", 0), reverse=True)
    return score, edge_hits[:20]


def score_kg_with_fields(
    kg: Dict[str, Any],
    case_fields: List[Dict[str, str]],
) -> Tuple[float, Dict[str, List[Dict[str, Any]]]]:
    terms = extract_node_terms(kg)
    clinical_facts = extract_clinical_facts(case_fields)

    matched: Dict[str, List[Dict[str, Any]]] = {
        "aliases": [],
        "clinical_contexts": [],
        "symptoms": [],
        "signs": [],
        "risk_factors": [],
        "findings": [],
        "critical_tests": [],
        "organ_dysfunction": [],
        "diagnostic_criteria": [],
        "treatments": [],
        "management_actions": [],
        "mimics": [],
        "clinical_facts": [],
        "graph_edges": [],
    }

    weights = {
        "aliases": 3.0,
        "clinical_contexts": 1.3,
        "symptoms": 1.8,
        "signs": 2.0,
        "findings": 2.2,
        "critical_tests": 1.4,
        "organ_dysfunction": 2.4,
        "diagnostic_criteria": 2.2,
        "risk_factors": 0.8,
        "treatments": 0.7,
        "management_actions": 0.35,
        "mimics": 0.2,
        "clinical_facts": 2.0,
    }

    score = 0.0
    matched_node_ids_seen_by_group: Dict[str, set] = {k: set() for k in matched.keys()}

    for group, group_terms in terms.items():
        for term_item in group_terms:
            term_matches = find_term_matches(term_item, case_fields)
            if not term_matches:
                continue

            node_id = str(term_item.get("node_id", "") or term_item.get("term", ""))
            matched[group].extend(term_matches)
            best_multiplier = max(polarity_multiplier(m["polarity"]) for m in term_matches)
            if node_id not in matched_node_ids_seen_by_group[group]:
                score += weights.get(group, 1.0) * best_multiplier
                matched_node_ids_seen_by_group[group].add(node_id)

    fact_matches = match_clinical_facts_to_kg(kg, clinical_facts)
    if fact_matches:
        matched["clinical_facts"].extend(fact_matches)
        fact_node_seen = set()
        for m in fact_matches:
            if m.get("polarity") not in ["positive", "uncertain"]:
                continue
            key = m.get("node_id") or m.get("canonical_fact")
            if key in fact_node_seen:
                continue
            fact_node_seen.add(key)
            severity_boost = 1.25 if m.get("severity") == "severe" else 1.0
            score += weights["clinical_facts"] * polarity_multiplier(m.get("polarity", "positive")) * severity_boost

    g_score, edge_hits = graph_edge_score(kg, matched)
    if edge_hits:
        matched["graph_edges"] = edge_hits
        score += g_score

    strong_signal_groups = [
        "aliases",
        "clinical_contexts",
        "symptoms",
        "signs",
        "findings",
        "critical_tests",
        "organ_dysfunction",
        "diagnostic_criteria",
        "clinical_facts",
    ]

    strong_signal_count = 0
    for group in strong_signal_groups:
        for item in matched[group]:
            if item.get("polarity") in ["positive", "uncertain"]:
                strong_signal_count += 1

    if strong_signal_count == 0:
        score = 0.0

    return round(score, 4), matched


def score_kg(
    kg: Dict[str, Any],
    case_text: str,
) -> Tuple[float, Dict[str, List[Dict[str, Any]]]]:
    case_fields = [
        {
            "field": "case",
            "text": str(case_text),
            "norm_text": normalize_text(case_text),
        }
    ]
    return score_kg_with_fields(kg, case_fields)


def _limit(items: List[Any], n: int = 8) -> List[Any]:
    return items[:n] if items else []


def _term_names(term_items: List[Dict[str, str]], n: int = 8) -> List[str]:
    names = []
    seen = set()
    for item in term_items:
        term = item.get("term", "")
        key = normalize_text(term)
        if term and key not in seen:
            names.append(term)
            seen.add(key)
        if len(names) >= n:
            break
    return names


def _format_match_line(match: Dict[str, Any]) -> str:
    term = match.get("term", "")
    matched_text = match.get("matched_text", "")
    field = match.get("field", "")
    polarity = match.get("polarity", "")
    context = match.get("context", "")
    source = match.get("match_source", "")
    value = match.get("value", None)
    unit = match.get("unit", "")
    node_id = match.get("node_id", "")

    value_text = ""
    if value is not None:
        value_text = f" | value={value} {unit}".rstrip()
    source_text = f" | source={source}" if source else ""
    node_text = f" | node_id={node_id}" if node_id else ""

    return (
        f"{term} | matched='{matched_text}' | "
        f"field={field} | polarity={polarity}{value_text}{node_text}{source_text} | context=\"{context}\""
    )


def _format_edge_line(edge: Dict[str, Any]) -> str:
    return (
        f"{edge.get('edge_id', '')}: {edge.get('source', '')} "
        f"-{edge.get('relation', '')}-> {edge.get('target', '')} | "
        f"strength={edge.get('strength', '')} | weight={edge.get('weight', '')} | "
        f"score+={edge.get('contribution', '')}"
    )


def split_supporting_and_non_supporting_matches(
    matched_terms: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[str], List[str]]:
    supporting = []
    non_supporting = []
    for group, matches in matched_terms.items():
        if group == "graph_edges":
            continue
        for match in matches:
            line = f"{group}: {_format_match_line(match)}"
            if match.get("polarity") in ["positive", "uncertain"]:
                supporting.append(line)
            else:
                non_supporting.append(line)
    return supporting, non_supporting


def get_missing_critical_tests(
    kg: Dict[str, Any],
    matched_terms: Dict[str, List[Dict[str, Any]]],
    max_items: int = 6,
) -> List[str]:
    terms = extract_node_terms(kg)
    critical_terms = _term_names(terms.get("critical_tests", []), n=20)

    matched_critical = set()
    for item in matched_terms.get("critical_tests", []):
        if item.get("polarity") in ["positive", "uncertain"]:
            matched_critical.add(normalize_text(item.get("term", "")))
    for item in matched_terms.get("clinical_facts", []):
        if item.get("polarity") in ["positive", "uncertain"]:
            matched_critical.add(normalize_text(item.get("node_name", "")))
            matched_critical.add(normalize_text(item.get("term", "")))

    missing = []
    for term in critical_terms:
        if normalize_text(term) not in matched_critical:
            missing.append(term)
        if len(missing) >= max_items:
            break
    return missing

def build_kg_profile(
    kg: Dict[str, Any],
    matched_terms: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    score: Optional[float] = None,
) -> str:
    matched_terms = matched_terms or {}

    safety = kg.get("safety_constraints", {})
    must_do = safety.get("must_do", [])
    must_not = safety.get("must_not", [])
    terms = extract_node_terms(kg)

    disease_name = kg.get("disease_name_en", "Unknown disease")
    disease_id = kg.get("disease_id", "")
    aliases = kg.get("aliases", [])
    clinical_contexts = _term_names(terms.get("clinical_contexts", []), 6)
    symptoms = _term_names(terms.get("symptoms", []), KG_MAX_TERMS_PER_GROUP)
    signs = _term_names(terms.get("signs", []), 6)
    findings = _term_names(terms.get("findings", []), 6)
    critical_tests = _term_names(terms.get("critical_tests", []), 6)
    organ_dysfunction = _term_names(terms.get("organ_dysfunction", []), 6)
    diagnostic_criteria = _term_names(terms.get("diagnostic_criteria", []), 6)
    mimics = _term_names(terms.get("mimics", []), 6)

    supporting_matches, non_supporting_matches = split_supporting_and_non_supporting_matches(matched_terms)
    edge_lines = [_format_edge_line(e) for e in matched_terms.get("graph_edges", [])]

    supporting_text = "\n- ".join(_limit(supporting_matches, 16)) if supporting_matches else "No positive or uncertain matched case signals recorded."
    non_supporting_text = "\n- ".join(_limit(non_supporting_matches, 8)) if non_supporting_matches else "No negative or historical mentions detected."
    graph_edge_text = "\n- ".join(_limit(edge_lines, 10)) if edge_lines else "No graph-edge evidence path used in scoring."

    missing_critical_tests = get_missing_critical_tests(kg, matched_terms)
    missing_critical_text = "\n- ".join(missing_critical_tests) if missing_critical_tests else "No obvious missing critical verification data from KG profile."

    must_do_text = "\n- ".join(_limit(must_do, 8)) if must_do else "No mandatory guideline items available."
    must_not_text = "\n- ".join(_limit(must_not, 8)) if must_not else "No prohibited logic items available."
    score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"

    return f"""
[Clinical KG: {disease_name}]
Disease ID:
- {disease_id if disease_id else "Unknown"}

Retrieval score:
- {score_text}

Aliases:
- {", ".join(aliases) if aliases else "None"}

Positive / uncertain matched case signals:
- {supporting_text}

Graph-edge evidence paths used for retrieval scoring:
- {graph_edge_text}

Negative / historical mentions detected:
- {non_supporting_text}

High-yield symptoms:
- {"; ".join(symptoms) if symptoms else "None"}

Clinical contexts / normalization triggers:
- {"; ".join(clinical_contexts) if clinical_contexts else "None"}

High-yield signs:
- {"; ".join(signs) if signs else "None"}

Supporting findings:
- {"; ".join(findings) if findings else "None"}

Critical tests / verification data:
- {"; ".join(critical_tests) if critical_tests else "None"}

Organ dysfunction signals:
- {"; ".join(organ_dysfunction) if organ_dysfunction else "None"}

Diagnostic criteria / syndrome components:
- {"; ".join(diagnostic_criteria) if diagnostic_criteria else "None"}

Potentially missing critical verification data:
- {missing_critical_text}

Important mimics / competing diagnoses:
- {"; ".join(mimics) if mimics else "None"}

Mandatory Guidelines:
- {must_do_text}

Prohibited Logic:
- {must_not_text}

Important interpretation rule:
- Numeric clinical facts are retrieval signals, not autonomous diagnoses.
- Graph-edge paths adjust retrieval priority only when a matched case signal supports a graph node.
- Edge relations such as requires_assessment_of or cannot_independently_rule_in_or_rule_out are treated as restrictive/verification context, not ordinary positive evidence.
- Do not treat negative or historical mentions as positive evidence for the acute diagnosis.
- Missing critical verification data should be considered as missing information, not invented as present.
""".strip()


def _kg_min_score_for(kg: Dict[str, Any]) -> float:
    disease_id = str(kg.get("disease_id", ""))
    disease_name = str(kg.get("disease_name_en", ""))
    for key in [disease_id, disease_name, disease_id.lower(), disease_name.lower()]:
        if key in KG_PER_DISEASE_MIN_SCORE:
            try:
                return float(KG_PER_DISEASE_MIN_SCORE[key])
            except Exception:
                return KG_MIN_SCORE
    return KG_MIN_SCORE


def _resolve_kg_path(file_name: str) -> Path:
    return KG_DIR / file_name

def _assert_kg_file_available(file_name: str, path: Path) -> None:
    if path.exists():
        return
    if KG_STRICT_FILE_LOADING:
        raise FileNotFoundError(
            f"KG file listed in KG_FILES is missing: {path}. "
            "Set CONFIG['KG_STRICT_FILE_LOADING']=False only for exploratory runs."
        )

def _effective_top_k(requested_top_k: int, scored_count: int) -> int:
    if not KG_DYNAMIC_TOP_K:
        return requested_top_k
    return min(max(requested_top_k, KG_MAX_DYNAMIC_TOP_K), scored_count)


def retrieve_kg_structured(
    structured_case: Dict[str, Any],
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    if top_k is None:
        top_k = int(CONFIG.get("KG_TOP_K", 3))

    case_fields = flatten_case_fields(structured_case)
    clinical_facts = extract_clinical_facts(case_fields)
    scored_kgs = []
    skipped_files = []

    for file_name in KG_FILES:
        path = _resolve_kg_path(file_name)
        try:
            _assert_kg_file_available(file_name, path)
        except FileNotFoundError:
            raise

        if not path.exists():
            skipped_files.append({"file_name": file_name, "path": str(path), "reason": "missing"})
            continue

        kg = load_kg(path)
        score, matched_terms = score_kg_with_fields(kg, case_fields)
        min_score = _kg_min_score_for(kg)

        has_redflag_fact = any(
            m.get("severity") == "severe" and m.get("polarity") in ["positive", "uncertain"]
            for m in matched_terms.get("clinical_facts", [])
        )
        has_critical_edge = any(
            e.get("strength") in ["critical", "definitional"]
            for e in matched_terms.get("graph_edges", [])
        )
        force_recall = (has_redflag_fact or has_critical_edge) and score >= KG_REDFLAG_FORCE_RECALL_SCORE

        if score >= min_score or force_recall:
            kg_profile = build_kg_profile(kg=kg, matched_terms=matched_terms, score=score)
            scored_kgs.append(
                {
                    "file_name": file_name,
                    "path": str(path),
                    "disease_id": kg.get("disease_id", ""),
                    "disease_name_en": kg.get("disease_name_en", ""),
                    "score": score,
                    "min_score_used": min_score,
                    "force_recall": force_recall,
                    "matched_terms": matched_terms,
                    "extracted_clinical_facts": clinical_facts,
                    "missing_critical_tests": get_missing_critical_tests(kg, matched_terms),
                    "kg_profile": kg_profile,
                }
            )

    scored_kgs = sorted(scored_kgs, key=lambda item: item["score"], reverse=True)
    effective_top_k = _effective_top_k(int(top_k), len(scored_kgs))
    scored_kgs = scored_kgs[:effective_top_k]

    return {
        "kg_dir": str(KG_DIR),
        "top_k_requested": top_k,
        "top_k_effective": effective_top_k,
        "min_score": KG_MIN_SCORE,
        "per_disease_min_score": KG_PER_DISEASE_MIN_SCORE,
        "strict_file_loading": KG_STRICT_FILE_LOADING,
        "skipped_files": skipped_files,
        "extracted_clinical_facts": clinical_facts,
        "retrieved_kgs": scored_kgs,
    }


def retrieve_kg(
    structured_case: Dict[str, Any],
    top_k: int = 3,
) -> str:
    result = retrieve_kg_structured(structured_case, top_k=top_k)
    retrieved_kgs = result.get("retrieved_kgs", [])
    if not retrieved_kgs:
        return "No specific guidelines retrieved."
    kg_texts = [item["kg_profile"] for item in retrieved_kgs]
    return "\n\n---\n\n".join(kg_texts)

def retrieve_kg_with_debug(
    structured_case: Dict[str, Any],
    top_k: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    debug = retrieve_kg_structured(structured_case, top_k=top_k)
    retrieved_kgs = debug.get("retrieved_kgs", [])
    if not retrieved_kgs:
        return "No specific guidelines retrieved.", debug
    kg_context = "\n\n---\n\n".join(item["kg_profile"] for item in retrieved_kgs)
    return kg_context, debug
