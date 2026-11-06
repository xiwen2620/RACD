# RACDR: Risk-Aware Collaborative Diagnostic Reasoning Framework for High-Risk Disease Identification

Code accompanying the RACDR study. This repository organizes the proposed knowledge-graph-assisted, multi-agent diagnostic reasoning framework and three comparison settings. The implementation is intended for research and reproducibility, not for clinical deployment or autonomous patient care.

## Repository layout

```text
RACDR/
├── experiments/
│   ├── single_agent/         # Single-agent baseline
│   │   └── single_agent.py
│   ├── single_agent_kg/      # Single-agent + knowledge graph
│   │   └── single_agent_kg.py
│   ├── multi_agent/          # Multi-agent baseline without KG
│   │   ├── main.py
│   │   ├── agents.py
│   │   ├── prompts.py
│   │   ├── pipeline.py
│   │   ├── config.py
│   │   ├── results.py
│   │   └── utils.py
│   └── racdr/                # Proposed multi-agent + KG framework
│       ├── main.py
│       ├── agents.py
│       ├── prompts.py
│       ├── pipeline.py
│       ├── config.py
│       ├── case_processing.py
│       ├── kg_retrieval.py
│       ├── results.py
│       ├── utils.py
│       └── kg/               # 12 disease-specific JSON knowledge graphs
├── data_preparation/
│   └── build_mimic6000.py
├── evaluation/
│   └── evaluate.py
├── README.md
└── requirements.txt
```

The single-agent scripts reuse modules from `experiments/racdr/` via a local Python import path. Run the four experimental conditions in **separate Python processes** to avoid collisions between modules with identical names.

## Requirements and installation

Python 3.10 is the development environment used for the repository preparation. Create an isolated environment and install the direct third-party dependencies:

```bash
conda create -n racdr python=3.10 -y
conda activate racdr
python -m pip install -r requirements.txt
```

The repository uses the `openai` Python client to call configurable OpenAI-compatible model endpoints. Model identifiers and endpoint compatibility depend on your provider and account access. An installation alone does not provide model access.

## Data availability and preparation

The study uses a high-risk cohort of 6,000 cases prepared from MIMIC-IV. **MIMIC-IV patient-level records and derived patient-level clinical text are not distributed in this repository.** Researchers must obtain appropriate access to MIMIC-IV through its authorized distribution process and comply with the dataset's terms and applicable privacy requirements.

The cohort-construction script is `data_preparation/build_mimic6000.py`. Before running it, review and configure its `MIMIC_ROOT` and `OUTPUT_PATH` values for your local data layout. The script includes ICD-based disease selection rules for 12 high-risk diagnostic categories. Its inputs depend on the locally available MIMIC-IV files and preprocessing arrangement; review the script before execution.

The experiment scripts expect a locally prepared CSV with a `combined_text` clinical-input field. Configure all four conditions to use the same intended cohort CSV for a controlled comparison. The placeholder filenames in the code are not downloadable datasets.

## Model and data configuration

**Do not commit real API keys, private endpoints, or patient-level datasets.** The repository uses placeholder configuration values such as `YOUR_API_KEY`, `https://YOUR_API_ENDPOINT/v1`, and `/PATH/TO/YOUR/...`.

- `experiments/single_agent/single_agent.py` and `experiments/single_agent_kg/single_agent_kg.py`: supply `--data_path`, `--api_key`, and `--base_url`, or use the supported environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`).
- `experiments/multi_agent/config.py`: set `CONFIG['INPUT_PATH']`, API settings, model identifiers, and other run settings. This configuration also supports `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- `experiments/racdr/config.py`: set `CONFIG['INPUT_PATH']`, API settings, and model identifiers. The API key can be provided through `CLINICAL_API_KEY`; the API base URL is configured in `CONFIG['API_BASE']`.
- `evaluation/evaluate.py`: configure the input result, dataset, and output paths through command-line arguments. If API-based matching/validation is enabled, configure its `GPT4O_API_KEY`, `GPT4O_BASE_URL`, and `GPT4O_MODEL_NAME` settings.

Check the model IDs in each experiment's source/configuration and ensure they are available from the configured endpoint. The code's default model identifiers are not a guarantee of public availability.

## Running the experiments

Run the following commands from the repository root after configuring paths and model access. These commands can trigger paid model API calls; consider testing on a small, authorized sample first.

### 1. Single-agent baseline

```bash
python experiments/single_agent/single_agent.py \
  --data_path /PATH/TO/YOUR/mimic6000.csv \
  --output_path /PATH/TO/YOUR/single_agent_results.json
```

### 2. Single-agent + KG baseline

```bash
python experiments/single_agent_kg/single_agent_kg.py \
  --data_path /PATH/TO/YOUR/mimic6000.csv \
  --output_path /PATH/TO/YOUR/single_agent_kg_results.json
```

This baseline imports the KG retrieval and case-processing modules from `experiments/racdr/`. Keep that directory and its `kg/` JSON files intact.

### 3. Multi-agent baseline (without KG)

Edit `experiments/multi_agent/config.py`, including `INPUT_PATH` and API settings, then run:

```bash
cd experiments/multi_agent
python main.py
```

### 4. RACDR (multi-agent + KG)

Edit `experiments/racdr/config.py`, including `INPUT_PATH` and API settings, then run:

```bash
cd experiments/racdr
python main.py
```

The RACDR configuration references the `kg/` directory relative to its own `config.py` file. The KG contains 12 disease-specific JSON graphs. The program includes KG file validation on startup. Outputs for the two multi-agent pipelines are controlled by their `config.py` files and are written under their configured run output directories.

Return to the repository root before running the evaluation command below.

## Evaluation

The evaluation entry point is `evaluation/evaluate.py`. It accepts an experiment-result JSON, the corresponding cohort CSV, and an output report path:

```bash
python evaluation/evaluate.py \
  --input_path /PATH/TO/YOUR/experiment_results.json \
  --dataset_path /PATH/TO/YOUR/mimic6000.csv \
  --output_path /PATH/TO/YOUR/evaluation_report.json
```

For a rules-only evaluation without external semantic-matching and evidence-validation calls, use the supported flags:

```bash
python evaluation/evaluate.py \
  --input_path /PATH/TO/YOUR/experiment_results.json \
  --dataset_path /PATH/TO/YOUR/mimic6000.csv \
  --output_path /PATH/TO/YOUR/evaluation_report.json \
  --disable_api_matching \
  --disable_evidence_validation
```

**Important:** The rules-only setting is a different evaluation configuration from API-assisted matching/validation. Do not treat its results as directly interchangeable with the original study's reported metrics without confirming methodological equivalence.

The study's **DetectionRate@K** is a *case-level hit rate*: a case is counted as detected if at least one ground-truth high-risk target diagnosis is matched within the top K predictions. For a 6,000-case evaluation cohort:

```text
DetectionRate@K = (number of cases with >=1 matched target diagnosis in top K) / 6000
```

The code evaluates K = 1, 3, and 5. This is **not** per-label recall. The evaluation script also includes target-disease matching and optional API-assisted evidence assessment; inspect its configuration to reproduce the intended assessment protocol.

Ensure that the result JSON structure is compatible with the evaluation script and that the result and cohort files refer to the same cases. Do not modify the original scoring logic when reproducing reported results.

## Reproducibility notes

- The four experimental conditions are deliberately kept in separate directories to preserve their original prompts and inference logic.
- The multi-agent baseline and RACDR have different case-processing configuration modes (`raw_combined_text` and `precleaned`, respectively); use the study's validated input preparation when comparing them.
- Model responses may vary with model availability, provider behavior, sampling settings, and API updates.
- Output files may contain sensitive derived clinical text. Store them securely and do not upload patient-level outputs to public repositories.
- The source archive used to prepare this documentation has not been verified through an end-to-end 6,000-case run in this environment.

## Citation and license

If you use this code, please cite the associated RACDR manuscript once its bibliographic details are available. Add the final article DOI and citation here after publication. A code license should be selected and added as a separate `LICENSE` file before public release; no license is implied by this README.
