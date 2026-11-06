from __future__ import annotations
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import pandas as pd

MIMIC_ROOT = Path(r"/PATH/TO/YOUR/MIMIC.csv")
OUTPUT_PATH = Path(r"/PATH/TO/YOUR/DATABASE.csv")

DISEASE_NAMES = [
    "Sepsis",
    "Septic shock",
    "Severe infection with organ dysfunction",
    "Acute myocardial infarction",
    "Acute coronary syndrome",
    "Pulmonary embolism",
    "Aortic dissection",
    "Ischemic stroke",
    "Intracranial hemorrhage",
    "Meningitis",
    "Encephalitis",
    "Bowel perforation",
]


ICD_CODE_PATTERNS = {
    "Sepsis": {
        "9": [r"038[A-Z0-9]*", r"99591", r"99592"],
        "10": [r"A40[A-Z0-9]*", r"A41[A-Z0-9]*"],
    },
    "Septic shock": {
        "9": [r"78552"],
        "10": [r"R6521", r"T8112[A-Z0-9]*"],
    },
    "Severe infection with organ dysfunction": {
        "9": [r"99592"],
        "10": [r"R652", r"R6520", r"R6521"],
    },
    "Acute myocardial infarction": {
        "9": [r"410[A-Z0-9]*"],
        "10": [r"I21[A-Z0-9]*", r"I22[A-Z0-9]*"],
    },
    "Acute coronary syndrome": {
        "9": [r"410[A-Z0-9]*", r"4111", r"41181", r"41189"],
        "10": [r"I200", r"I2000", r"I2001", r"I2009", r"I21[A-Z0-9]*", r"I22[A-Z0-9]*", r"I24[A-Z0-9]*"],
    },
    "Pulmonary embolism": {
        "9": [r"4151[A-Z0-9]*"],
        "10": [r"I26[A-Z0-9]*"],
    },
    "Aortic dissection": {
        "9": [r"4410[A-Z0-9]*"],
        "10": [r"I710[A-Z0-9]*"],
    },
    "Ischemic stroke": {
        "9": [r"433[0-9]1", r"434[0-9]1"],
        "10": [r"I63[A-Z0-9]*"],
    },
    "Intracranial hemorrhage": {
        "9": [r"430", r"431", r"432[A-Z0-9]*"],
        "10": [r"I60[A-Z0-9]*", r"I61[A-Z0-9]*", r"I62[A-Z0-9]*"],
    },
    "Meningitis": {
        "9": [r"320[A-Z0-9]*", r"321[A-Z0-9]*", r"322[A-Z0-9]*"],
        "10": [r"G00[A-Z0-9]*", r"G01", r"G02", r"G03[A-Z0-9]*"],
    },
    "Encephalitis": {
        "9": [r"049[A-Z0-9]*", r"062[A-Z0-9]*", r"063[A-Z0-9]*", r"064[A-Z0-9]*", r"066[A-Z0-9]*", r"323[A-Z0-9]*"],
        "10": [r"A83[A-Z0-9]*", r"A84[A-Z0-9]*", r"A85[A-Z0-9]*", r"A86", r"G04[A-Z0-9]*"],
    },
    "Bowel perforation": {
        "9": [r"56983"],
        "10": [r"K631"],
    },
}

COMPILED_ICD_CODE_PATTERNS = {
    disease: {
        version: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for version, patterns in version_rules.items()
    }
    for disease, version_rules in ICD_CODE_PATTERNS.items()
}

DICTIONARY_TITLE_PATTERNS = {
    "Sepsis": [
        re.compile(r".*(?:\bsepsis\b|\bsepticemia\b).*", re.IGNORECASE),
    ],
    "Meningitis": [
        re.compile(r".*\bmeningitis\b.*", re.IGNORECASE),
    ],
    "Encephalitis": [
        re.compile(r".*(?:\bencephalitis\b|\bmeningoencephalitis\b).*", re.IGNORECASE),
    ],
    "Bowel perforation": [
        re.compile(
            r".*(?:\bperforation\b|\bperforated\b).*"
            r"(?:\bintestine\b|\bintestinal\b|\bbowel\b|\bcolon\b|"
            r"\bcolonic\b|\bcecum\b|\bcecal\b|\bdiverticul\w*\b|"
            r"\bappendi\w*\b|\bgastrointestinal\b).*",
            re.IGNORECASE,
        ),
        re.compile(
            r".*(?:\bintestine\b|\bintestinal\b|\bbowel\b|\bcolon\b|"
            r"\bcolonic\b|\bcecum\b|\bcecal\b|\bdiverticul\w*\b|"
            r"\bappendi\w*\b|\bgastrointestinal\b).*"
            r"(?:\bperforation\b|\bperforated\b).*",
            re.IGNORECASE,
        ),
    ],
}


def normalize_id_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_code_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.upper()
    )


def valid_text(x) -> bool:
    if pd.isna(x):
        return False

    text = str(x).strip()
    if not text:
        return False
    if text.lower() in {"nan", "none"}:
        return False
    if re.fullmatch(r"[|;,\- ]+", text):
        return False

    return True


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    text = str(x).replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def safe_join(values: Iterable, max_n: int = 100) -> str:
    result = []
    seen = set()

    for value in values:
        text = clean_text(value)
        if not valid_text(text) or text in seen:
            continue

        seen.add(text)
        result.append(text)
        if len(result) >= max_n:
            break

    return " ; ".join(result)


def remove_chief_complaint(text):
    if pd.isna(text) or not isinstance(text, str):
        return text

    pattern = (
        r"Chief Complaint:.*?"
        r"(?=(Major Surgical|History of Present Illness|Past Medical History|"
        r"Social History|Family History|Physical Exam))"
    )
    result = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", result).strip()


def match_target_diseases_from_codes(
    code_version_pairs: Iterable[tuple[str, str]],
    disease_code_sets: dict[str, set[tuple[str, str]]],
) -> list[str]:
    normalized_pairs = {
        (str(code).strip().replace(".", "").upper(), str(version).strip())
        for code, version in code_version_pairs
        if code is not None and version is not None
    }

    matched = {
        disease
        for disease in DISEASE_NAMES
        if normalized_pairs & disease_code_sets[disease]
    }
    return [disease for disease in DISEASE_NAMES if disease in matched]


def resolve_file(base: Path, relative_with_nested_csv: str, simple_relative: str) -> Path:
    first = base / relative_with_nested_csv
    if first.exists():
        return first

    second = base / simple_relative
    if second.exists():
        return second

    raise FileNotFoundError(f"File not found: {first} or {second}")




def iter_complete_hadm_groups(diag_path: Path, chunksize: int = 500_000):
    usecols = ["hadm_id", "icd_code", "icd_version"]
    pending = pd.DataFrame(columns=usecols)

    for chunk in pd.read_csv(
        diag_path,
        usecols=usecols,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["hadm_id"] = normalize_id_series(chunk["hadm_id"])
        chunk["icd_code"] = normalize_code_series(chunk["icd_code"])
        chunk["icd_version"] = normalize_id_series(chunk["icd_version"])
        chunk = chunk[chunk["hadm_id"] != ""]

        if not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)

        if chunk.empty:
            pending = pd.DataFrame(columns=usecols)
            continue

        last_hadm_id = chunk["hadm_id"].iloc[-1]
        complete = chunk[chunk["hadm_id"] != last_hadm_id]
        pending = chunk[chunk["hadm_id"] == last_hadm_id].copy()

        for hadm_id, group in complete.groupby("hadm_id", sort=False):
            yield hadm_id, group

    if not pending.empty:
        for hadm_id, group in pending.groupby("hadm_id", sort=False):
            yield hadm_id, group


def build_diagnosis_maps(diag_dict_path: Path):
    diag_dict = pd.read_csv(
        diag_dict_path,
        usecols=["icd_code", "icd_version", "long_title"],
        dtype="string",
        low_memory=False,
    )
    diag_dict["icd_code"] = normalize_code_series(diag_dict["icd_code"])
    diag_dict["icd_version"] = normalize_id_series(diag_dict["icd_version"])
    diag_dict["long_title"] = diag_dict["long_title"].fillna("").astype(str)

    title_map = {
        (row.icd_code, row.icd_version): row.long_title
        for row in diag_dict.itertuples(index=False)
    }
    disease_code_sets = {disease: set() for disease in DISEASE_NAMES}

    for row in diag_dict.itertuples(index=False):
        key = (row.icd_code, row.icd_version)
        title = row.long_title

        for disease in DISEASE_NAMES:
            by_code = any(
                pattern.fullmatch(row.icd_code)
                for pattern in COMPILED_ICD_CODE_PATTERNS[disease].get(row.icd_version, [])
            )

            by_title = any(
                pattern.fullmatch(title)
                for pattern in DICTIONARY_TITLE_PATTERNS.get(disease, [])
            )

            # 排除明确写着“without perforation”的条目。
            if disease == "Bowel perforation" and re.search(
                r"\bwithout perforation\b", title, flags=re.IGNORECASE
            ):
                by_title = False

            if by_code or by_title:
                disease_code_sets[disease].add(key)

    print("\nICD dictionary coverage:")
    for disease in DISEASE_NAMES:
        keys = disease_code_sets[disease]
        print(f"  {disease}: {len(keys)} ICD entries")
        for code, version in sorted(keys, key=lambda x: (x[1], x[0]))[:5]:
            print(f"      ICD-{version} {code}: {title_map.get((code, version), '')}")

    missing = [disease for disease in DISEASE_NAMES if not disease_code_sets[disease]]
    if missing:
        raise RuntimeError(
            "No ICD dictionary entries were found for: " + ", ".join(missing)
        )

    return title_map, disease_code_sets


def collect_diagnosis_candidates(
    diag_path: Path,
    title_map: dict,
    disease_code_sets: dict[str, set[tuple[str, str]]],
    target_count: int,
    min_per_disease: int,
    candidate_pool_size: int,
    candidate_per_disease: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    print("\nScanning diagnoses with early-stop counters...")
    print(f"  Final target: {target_count}")
    print(f"  Minimum per disease: {min_per_disease}")
    print(f"  Candidate pool target: {candidate_pool_size}")
    print(f"  Candidate buffer per disease: {candidate_per_disease}")

    candidates = []
    candidate_counts = Counter()
    scanned_admissions = 0

    for hadm_id, group in iter_complete_hadm_groups(diag_path, chunksize=chunksize):
        scanned_admissions += 1

        titles = []
        for row in group.itertuples(index=False):
            title = title_map.get((row.icd_code, row.icd_version), "")
            if valid_text(title):
                titles.append(title)

        code_version_pairs = [
            (row.icd_code, row.icd_version)
            for row in group.itertuples(index=False)
        ]
        matched_diseases = match_target_diseases_from_codes(code_version_pairs, disease_code_sets)
        if not matched_diseases:
            continue

        # 前 candidate_pool_size 个目标病例全部保留；达到总候选量后，
        # 只继续保留能补充尚未达到缓冲计数的疾病病例。
        contributes_to_shortage = any(
            candidate_counts[disease] < candidate_per_disease
            for disease in matched_diseases
        )
        if len(candidates) >= candidate_pool_size and not contributes_to_shortage:
            continue

        diagnosis_text = safe_join(titles, max_n=200)
        if not valid_text(diagnosis_text):
            continue

        candidates.append(
            {
                "hadm_id": hadm_id,
                "diagnosis": diagnosis_text,
                "target_diseases": " ; ".join(matched_diseases),
            }
        )
        for disease in matched_diseases:
            candidate_counts[disease] += 1

        if len(candidates) % 1000 == 0:
            shortage = {
                disease: max(0, candidate_per_disease - candidate_counts[disease])
                for disease in DISEASE_NAMES
            }
            print(
                f"  Candidates: {len(candidates):,}; "
                f"scanned admissions: {scanned_admissions:,}; "
                f"remaining buffer deficits: {shortage}"
            )

        enough_total = len(candidates) >= candidate_pool_size
        enough_each_disease = all(
            candidate_counts[disease] >= candidate_per_disease
            for disease in DISEASE_NAMES
        )
        if enough_total and enough_each_disease:
            print("  Candidate counters reached. Stopping diagnosis scan early.")
            break

    print(f"\nDiagnosis candidate pool: {len(candidates):,}")
    print(f"Admissions examined before stopping: {scanned_admissions:,}")
    print("Candidate disease counts (multi-label cases count for every matched disease):")
    for disease in DISEASE_NAMES:
        print(f"  {disease}: {candidate_counts[disease]}")

    if len(candidates) < target_count:
        raise RuntimeError(
            f"Only {len(candidates)} diagnosis candidates were found; "
            f"at least {target_count} are required."
        )

    return pd.DataFrame(candidates).drop_duplicates("hadm_id", keep="first")



def aggregate_text_for_ids(
    filepath: Path,
    wanted_ids: set[str],
    description: str,
    max_n: int,
    chunksize: int = 200_000,
) -> dict[str, str]:
    """单次分块扫描，只累计候选 hadm_id 的文本，且每个ID最多保留 max_n 条。"""
    print(f"\nReading {description} once: {filepath.name}")
    values = defaultdict(list)
    seen = defaultdict(set)
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        filepath,
        usecols=["hadm_id", "text"],
        chunksize=chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk["hadm_id"] = normalize_id_series(chunk["hadm_id"])
        chunk = chunk[chunk["hadm_id"].isin(wanted_ids)]
        matched_rows += len(chunk)

        for row in chunk.itertuples(index=False):
            hadm_id = row.hadm_id
            if len(values[hadm_id]) >= max_n:
                continue

            text = clean_text(row.text)
            if not valid_text(text) or text in seen[hadm_id]:
                continue

            seen[hadm_id].add(text)
            values[hadm_id].append(text)

    print(f"  Rows scanned: {total_rows:,}; matching rows: {matched_rows:,}")
    print(f"  Candidate admissions with {description}: {len(values):,}")
    return {hadm_id: " ; ".join(texts) for hadm_id, texts in values.items()}


def aggregate_labs_for_ids(
    labs_path: Path,
    lab_dict_path: Path,
    wanted_ids: set[str],
    max_n: int = 150,
    chunksize: int = 500_000,
) -> dict[str, str]:
    print(f"\nReading laboratory events once: {labs_path.name}")

    lab_dict = pd.read_csv(
        lab_dict_path,
        usecols=["itemid", "label"],
        dtype="string",
        low_memory=False,
    )
    lab_dict["itemid"] = normalize_id_series(lab_dict["itemid"])
    label_map = dict(zip(lab_dict["itemid"], lab_dict["label"].fillna("")))

    header = pd.read_csv(labs_path, nrows=0).columns.tolist()
    requested_cols = ["hadm_id", "itemid", "value", "valuenum", "valueuom"]
    usecols = [column for column in requested_cols if column in header]

    required = {"hadm_id", "itemid"}
    if not required.issubset(usecols):
        raise RuntimeError("labevents.csv is missing hadm_id or itemid.")

    values = defaultdict(list)
    seen = defaultdict(set)
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        labs_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk["hadm_id"] = normalize_id_series(chunk["hadm_id"])
        chunk = chunk[chunk["hadm_id"].isin(wanted_ids)].copy()
        matched_rows += len(chunk)

        if chunk.empty:
            continue

        chunk["itemid"] = normalize_id_series(chunk["itemid"])
        chunk["label"] = chunk["itemid"].map(label_map).fillna("")

        if "value" in chunk.columns:
            lab_value = chunk["value"].fillna("").astype(str).str.strip()
        else:
            lab_value = pd.Series("", index=chunk.index, dtype="string")

        if "valuenum" in chunk.columns:
            numeric_value = chunk["valuenum"].fillna("").astype(str).str.strip()
            lab_value = lab_value.mask(lab_value == "", numeric_value)

        if "valueuom" in chunk.columns:
            unit = chunk["valueuom"].fillna("").astype(str).str.strip()
        else:
            unit = pd.Series("", index=chunk.index, dtype="string")

        chunk["lab_text"] = (
            chunk["label"].fillna("").astype(str).str.strip()
            + ": "
            + lab_value
            + " "
            + unit
        )
        chunk["lab_text"] = (
            chunk["lab_text"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip(" :")
        )

        for row in chunk[["hadm_id", "lab_text"]].itertuples(index=False):
            hadm_id = row.hadm_id
            if len(values[hadm_id]) >= max_n:
                continue

            text = clean_text(row.lab_text)
            if not valid_text(text) or text in seen[hadm_id]:
                continue

            seen[hadm_id].add(text)
            values[hadm_id].append(text)

    print(f"  Rows scanned: {total_rows:,}; matching rows: {matched_rows:,}")
    print(f"  Candidate admissions with labs: {len(values):,}")
    return {hadm_id: " ; ".join(texts) for hadm_id, texts in values.items()}




def build_complete_candidates(root: Path, diagnosis_candidates: pd.DataFrame) -> pd.DataFrame:
    hosp = root / "hosp"

    patients_path = resolve_file(hosp, "patients.csv/patients.csv", "patients.csv")
    admissions_path = resolve_file(hosp, "admissions.csv/admissions.csv", "admissions.csv")
    labs_path = resolve_file(hosp, "labevents.csv/labevents.csv", "labevents.csv")
    lab_dict_path = resolve_file(hosp, "d_labitems.csv/d_labitems.csv", "d_labitems.csv")
    discharge_path = resolve_file(root, "discharge.csv/discharge.csv", "discharge.csv")
    radiology_path = resolve_file(root, "radiology.csv/radiology.csv", "radiology.csv")

    print("\nLoading patient and admission tables...")
    patients = pd.read_csv(
        patients_path,
        usecols=["subject_id", "anchor_age", "gender"],
        low_memory=False,
    )
    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id"],
        low_memory=False,
    )

    patients["subject_id"] = normalize_id_series(patients["subject_id"])
    admissions["subject_id"] = normalize_id_series(admissions["subject_id"])
    admissions["hadm_id"] = normalize_id_series(admissions["hadm_id"])

    base = diagnosis_candidates.merge(
        admissions.drop_duplicates("hadm_id", keep="first"),
        on="hadm_id",
        how="left",
    )
    base = base.merge(
        patients.drop_duplicates("subject_id", keep="first"),
        on="subject_id",
        how="left",
    )
    base = base.rename(columns={"anchor_age": "age", "gender": "gender"})

    base = base[
        base["age"].apply(valid_text)
        & base["gender"].apply(valid_text)
    ].copy()

    wanted_ids = set(base["hadm_id"].astype(str))
    print(f"Candidates with age and gender: {len(base):,}")

    discharge_map = aggregate_text_for_ids(
        discharge_path,
        wanted_ids,
        description="discharge reports",
        max_n=1,
    )
    radiology_map = aggregate_text_for_ids(
        radiology_path,
        wanted_ids,
        description="radiology reports",
        max_n=3,
    )
    lab_map = aggregate_labs_for_ids(
        labs_path,
        lab_dict_path,
        wanted_ids,
        max_n=150,
    )

    base["case_report"] = base["hadm_id"].map(discharge_map).fillna("")
    base["case_report"] = base["case_report"].apply(remove_chief_complaint)
    base["imaging_text"] = base["hadm_id"].map(radiology_map).fillna("")
    base["lab_text"] = base["hadm_id"].map(lab_map).fillna("")

    complete_mask = (
        base["case_report"].apply(valid_text)
        & base["imaging_text"].apply(valid_text)
        & base["lab_text"].apply(valid_text)
        & base["diagnosis"].apply(valid_text)
        & base["target_diseases"].apply(valid_text)
    )
    complete = base[complete_mask].copy()

    complete["test_results"] = (
        "Imaging: "
        + complete["imaging_text"]
        + " | Laboratory: "
        + complete["lab_text"]
    )
    complete["patient_id"] = complete["hadm_id"]
    complete["source"] = "mimic"

    complete = complete[
        [
            "patient_id",
            "source",
            "age",
            "gender",
            "case_report",
            "test_results",
            "diagnosis",
            "target_diseases",
        ]
    ].drop_duplicates("patient_id", keep="first")

    print(f"\nComplete candidates with discharge + imaging + laboratory: {len(complete):,}")
    return complete.reset_index(drop=True)



def disease_set(text: str) -> set[str]:
    if not valid_text(text):
        return set()
    return {part.strip() for part in str(text).split(";") if part.strip() in DISEASE_NAMES}


def disease_distribution(df: pd.DataFrame) -> Counter:
    counts = Counter()
    for text in df["target_diseases"]:
        counts.update(disease_set(text))
    return counts


def select_balanced_cases(
    df: pd.DataFrame,
    target_count: int = 6000,
    min_per_disease: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(df) < target_count:
        raise RuntimeError(
            f"Only {len(df)} complete unique cases remain, fewer than target {target_count}. "
            "The dataset cannot provide 6000 complete cases under the current field requirements. "
            "Increase the diagnosis candidate pool or relax completeness requirements."
        )

    rng = random.Random(random_state)
    labels_by_index = {
        index: disease_set(text)
        for index, text in df["target_diseases"].items()
    }
    disease_to_indices = {
        disease: {
            index
            for index, labels in labels_by_index.items()
            if disease in labels
        }
        for disease in DISEASE_NAMES
    }
    availability = {
        disease: len(disease_to_indices[disease])
        for disease in DISEASE_NAMES
    }
    first_stage_targets = {
        disease: min(min_per_disease, availability[disease])
        for disease in DISEASE_NAMES
    }

    print("\nComplete-case disease availability and first-stage targets:")
    for disease in DISEASE_NAMES:
        note = "" if availability[disease] >= min_per_disease else " (all available cases will be used)"
        print(
            f"  {disease}: available={availability[disease]}, "
            f"first-stage target={first_stage_targets[disease]}{note}"
        )

    selected: set[int] = set()
    counts = Counter()

    disease_order = sorted(DISEASE_NAMES, key=lambda d: (availability[d], DISEASE_NAMES.index(d)))

    for disease in disease_order:
        target = first_stage_targets[disease]
        while counts[disease] < target:
            choices = list(disease_to_indices[disease] - selected)
            if not choices:
                # 多标签计数与去重后理论上可能提前耗尽；此时已有病例即为最大可实现量。
                print(
                    f"  Warning: {disease} stopped at {counts[disease]} selected labels "
                    f"against target {target}; no unique cases remain."
                )
                break

            rng.shuffle(choices)

            def first_stage_score(index: int):
                labels = labels_by_index[index]
                weighted_deficit_coverage = 0.0
                uncovered_labels = 0
                for label in labels:
                    deficit = max(0, first_stage_targets[label] - counts[label])
                    if deficit <= 0:
                        continue
                    remaining_for_label = max(1, len(disease_to_indices[label] - selected))
                    weighted_deficit_coverage += deficit / remaining_for_label
                    uncovered_labels += 1
                return weighted_deficit_coverage, uncovered_labels

            chosen = max(choices, key=first_stage_score)
            selected.add(chosen)
            counts.update(labels_by_index[chosen])

    print(f"\nFirst-stage unique cases selected: {len(selected):,}")
    print("Disease counts after first stage (multi-label counting):")
    for disease in DISEASE_NAMES:
        print(f"  {disease}: {counts[disease]}")

    while len(selected) < target_count:
        remaining_by_disease = {
            disease: disease_to_indices[disease] - selected
            for disease in DISEASE_NAMES
        }
        active_diseases = [
            disease for disease in DISEASE_NAMES
            if remaining_by_disease[disease]
        ]
        if not active_diseases:
            break

        # 先补当前计数最低的疾病；并列时优先剩余病例更多的疾病。
        chosen_disease = min(
            active_diseases,
            key=lambda d: (
                counts[d],
                -len(remaining_by_disease[d]),
                DISEASE_NAMES.index(d),
            ),
        )

        choices = list(remaining_by_disease[chosen_disease])
        rng.shuffle(choices)

        min_active_count = min(counts[d] for d in active_diseases)

        def fill_score(index: int):
            labels = labels_by_index[index]
            low_count_coverage = sum(
                1 for label in labels
                if label in active_diseases and counts[label] <= min_active_count
            )
            total_inverse_count = sum(1.0 / (counts[label] + 1) for label in labels)
            return low_count_coverage, total_inverse_count

        chosen = max(choices, key=fill_score)
        selected.add(chosen)
        counts.update(labels_by_index[chosen])

    if len(selected) < target_count:
        raise RuntimeError(
            f"Only {len(selected)} unique complete cases could be selected after exhausting "
            f"all disease-specific candidates; target is {target_count}."
        )

    final_df = df.loc[sorted(selected)].copy().reset_index(drop=True)
    final_counts = disease_distribution(final_df)

    failed = [
        disease
        for disease in DISEASE_NAMES
        if final_counts[disease] < first_stage_targets[disease]
    ]
    if len(final_df) != target_count or failed:
        raise RuntimeError(
            f"Final validation failed: rows={len(final_df)}, failed diseases={failed}"
        )

    print(f"\nBalanced selection completed: {len(final_df)} unique cases")
    print("Final disease counts (one multi-label case is counted for every listed disease):")
    for disease in DISEASE_NAMES:
        print(
            f"  {disease}: {final_counts[disease]} "
            f"(available={availability[disease]}, first-stage target={first_stage_targets[disease]})"
        )

    return final_df




DELETE_SECTIONS = {
    "Service",
    "Attending",
    "Major Surgical or Invasive Procedure",
    "Pertinent Results",
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
    "Admission Diagnosis",
    "Brief Hospital Course",
}

HEADERS = {
    "Name": ["Name"],
    "Unit No": ["Unit No"],
    "Admission Date": ["Admission Date"],
    "Discharge Date": ["Discharge Date"],
    "Date of Birth": ["Date of Birth"],
    "Sex": ["Sex"],
    "Service": ["Service"],
    "Allergies": ["Allergies"],
    "Attending": ["Attending"],
    "Major Surgical or Invasive Procedure": ["Major Surgical or Invasive Procedure"],
    "Admission Diagnosis": ["Admission Diagnosis"],
    "History of Present Illness": ["History of Present Illness", "HPI", "History of Presenting Illness"],
    "Admission History and Physical": ["Admission History and Physical"],
    "Review of Systems": ["Review of Systems", "Review of sytems", "Review of system", "ROS"],
    "Past Medical History": ["Past Medical History", "Past Medical Hx", "Medical History"],
    "Past Surgical History": [
        "Past Surgical History", "PSH", "PSHx", "Surgical History",
        "Past Surgical Hx", "Surgical Hx", "Surg Hx", "Surg",
    ],
    "Social History": ["Social History"],
    "Family History": ["Family History"],
    "Physical Exam": ["Physical Exam", "Physical Examination"],
    "Pertinent Results": ["Pertinent Results"],
    "Brief Hospital Course": ["Brief Hospital Course", "Hospital Course"],
    "Medications on Admission": [
        "Medications on Admission", "Medications prior to admission", "Home Medications",
    ],
    "Discharge Medications": ["Discharge Medications"],
    "Discharge Disposition": ["Discharge Disposition"],
    "Discharge Diagnosis": ["Discharge Diagnosis", "Discharge Diagnoses"],
    "Discharge Condition": ["Discharge Condition"],
    "Discharge Instructions": ["Discharge Instructions"],
    "Followup Instructions": ["Followup Instructions", "Follow-up Instructions", "Follow up Instructions"],
}


def norm(text):
    return " ".join(text.casefold().split())


ALIASES = {
    norm(variant): canonical
    for canonical, variants in HEADERS.items()
    for variant in variants
}

# Longest variants first; e.g. "Discharge Diagnoses" ahead of shorter aliases.
ALL_TITLES = sorted(
    {variant for variants in HEADERS.values() for variant in variants},
    key=len,
    reverse=True,
)
PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<title>" +
    "|".join(re.escape(variant) for variant in ALL_TITLES) +
    r")\s*[:：]",
    re.IGNORECASE,
)


BAD_PREFIXES = {
    "Attending": re.compile(r"\b(?:per|primary|onc|ep|cardiology|surgical|consulting|by)\s+(?:\w+\s+){0,2}$", re.I),
    "Service": re.compile(r"\b(?:to|from|by|cardiology|surgery|vascular|medicine|neurology|on|consulting)\s+(?:\w+\s+){0,3}$", re.I),
    "Physical Exam": re.compile(r"\b(?:admission|discharge|preoperative|postoperative|pre-op|post-op|on|during)\s+$", re.I),
    "Past Medical History": re.compile(r"\b(?:other|additional|relevant)\s+$", re.I),
    "Pertinent Results": re.compile(r"\b(?:other|additional|previous|no)\s+$", re.I),
    "Past Surgical History": re.compile(r"\b(?:transplant|general|cardiac|vascular)\s+$", re.I),
    "Name": re.compile(r"\b(?:proxy|contact|patient|surrogate|spouse|first|last)\s+$", re.I),
    "Brief Hospital Course": re.compile(r"\b(?:his|her|their|the|in|during|of|subsequent|prior|previous|on|following|last|remaining|entire|after)\s+$", re.I),
}


def is_actual_heading(text, match, section):
    preceding = text[max(0, match.start() - 80):match.start()]
    # A hash marker normally signals a problem-oriented subsection, e.g.
    # "# Allergies:", not a top-level discharge-summary section.
    if re.search(r"#\s*$", preceding):
        return False
    if section == "History of Present Illness" and re.search(r"\bCC\s*/\s*$", preceding, re.I):
        return False
    bad = BAD_PREFIXES.get(section)
    if bad and bad.search(preceding):
        return False
    return True


def split_sections(text):
    matches = []
    # Demographic keys are single-occurrence fields in this discharge-summary
    # format. A later "Name:" in an appended operative/echo report is not a new
    # top-level field. This restriction does NOT impose order across sections.
    DEMOGRAPHIC_FIELDS = {
        "Name", "Unit No", "Admission Date", "Discharge Date",
        "Date of Birth", "Sex", "Service", "Allergies", "Attending",
    }
    seen_demographic = set()
    for match in PATTERN.finditer(text):
        canonical = ALIASES[norm(match.group("title"))]
        if canonical in DEMOGRAPHIC_FIELDS and canonical in seen_demographic:
            continue
        if is_actual_heading(text, match, canonical):
            matches.append((match.start(), canonical, match.group("title")))
            if canonical in DEMOGRAPHIC_FIELDS:
                seen_demographic.add(canonical)

    sections = []
    for i, (start, canonical, literal) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        sections.append((start, end, canonical, literal))
    return sections


def clean_case_report(text):
    if not isinstance(text, str) or not text.strip():
        return text, [], []

    sections = split_sections(text)
    if not sections:
        return text, [], []

    pieces = []
    # Do not discard text before the first recognized heading.
    if sections[0][0] > 0:
        pieces.append(text[:sections[0][0]])

    detected = []
    deleted = []
    for start, end, title, literal in sections:
        detected.append(title)
        if title in DELETE_SECTIONS:
            deleted.append(title)
        else:
            pieces.append(text[start:end])


    cleaned = "\n".join(piece.strip() for piece in pieces if piece.strip())
    return cleaned, detected, deleted



def build_dataset(MIMIC_ROOT: Path):

    TARGET_COUNT = 6000
    MIN_PER_DISEASE = 500

    CANDIDATE_POOL_SIZE = 15_000
    CANDIDATE_PER_DISEASE = 1_000

    hosp = MIMIC_ROOT / "hosp"
    diag_path = resolve_file(hosp, "diagnoses_icd.csv/diagnoses_icd.csv", "diagnoses_icd.csv")
    diag_dict_path = resolve_file(hosp, "d_icd_diagnoses.csv/d_icd_diagnoses.csv", "d_icd_diagnoses.csv")

    print("=" * 70)
    print("Build balanced MIMIC-IV dataset: aim for 500 per disease, then evenly fill to 6000")
    print("=" * 70)

    title_map, disease_code_sets = build_diagnosis_maps(diag_dict_path)

    diagnosis_candidates = collect_diagnosis_candidates(
        diag_path=diag_path,
        title_map=title_map,
        disease_code_sets=disease_code_sets,
        target_count=TARGET_COUNT,
        min_per_disease=MIN_PER_DISEASE,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        candidate_per_disease=CANDIDATE_PER_DISEASE,
    )

    complete_candidates = build_complete_candidates(
        root=MIMIC_ROOT,
        diagnosis_candidates=diagnosis_candidates,
    )

    final_df = select_balanced_cases(
        complete_candidates,
        target_count=TARGET_COUNT,
        min_per_disease=MIN_PER_DISEASE,
        random_state=42,
    )

    final_df = final_df.sort_values(
        ["target_diseases", "diagnosis", "patient_id"]
    ).reset_index(drop=True)

    return final_df



def merge_columns(row):
    parts = []
    for column in ("age", "gender"):
        value = row[column]
        if pd.notna(value) and str(value).strip():
            parts.append(f"{column}: {value}")
    for column in ("case_report", "test_results"):
        value = row[column]
        if pd.notna(value) and str(value).strip():
            parts.append(str(value))
    return "\n".join(parts)


def main():
    if str(MIMIC_ROOT) == "请填写MIMIC-IV根目录" or str(OUTPUT_PATH) == "请填写最终CSV输出路径.csv":
        raise ValueError("请先填写 MIMIC_ROOT 和 OUTPUT_PATH")
    if not MIMIC_ROOT.is_dir():
        raise FileNotFoundError(f"MIMIC-IV 目录不存在: {MIMIC_ROOT}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


    df = build_dataset(MIMIC_ROOT)


    audit_rows = []
    totals = Counter()
    cleaned_texts = []
    for row_id, text in enumerate(df["case_report"]):
        cleaned, detected, deleted = clean_case_report(text)
        cleaned_texts.append(cleaned)
        totals.update(deleted)
        audit_rows.append({
            "row_number": row_id + 1,
            "patient_id": df.iloc[row_id]["patient_id"] if "patient_id" in df.columns else "",
            "all_detected_headers_in_text_order": " | ".join(detected),
            "deleted_headers_in_text_order": " | ".join(deleted),
            "n_detected": len(detected),
            "repeated_section_types": " | ".join(
                sorted(title for title, count in Counter(detected).items() if count > 1)
            ),
            "n_deleted": len(deleted),
            "n_chars_before": len(text) if isinstance(text, str) else 0,
            "n_chars_after": len(cleaned) if isinstance(cleaned, str) else 0,
            "needs_manual_review": (
                len(cleaned.strip()) < 400
                or not any(h in detected for h in
                           ("History of Present Illness", "Admission History and Physical"))
            ),
            "possible_residual_deleted_header_matches": sum(
                1 for s in split_sections(cleaned) if s[2] in DELETE_SECTIONS
            ),
        })
    df["case_report"] = cleaned_texts

    # 阶段三：四列合并，每段换行，age/gender 带列名
    required = ["age", "gender", "case_report", "test_results"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"缺少合并所需列: {missing}")
    df["combined_text"] = df.apply(merge_columns, axis=1)
    df = df.drop(columns=required)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    audit_path = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_audit.csv")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"finish：{len(df)} cases")
    print(f"final_database：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
