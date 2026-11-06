import os
from pathlib import Path

CONFIG = {
    "HARDCODED_API_KEY": os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"),
    "API_BASE": os.getenv("OPENAI_BASE_URL", "https://YOUR_API_ENDPOINT/v1"),
    "INPUT_PATH": "/PATH/TO/YOUR/mimic6000.csv",
    "USE_CASE_MODE": "raw_combined_text",
    "RUN_NAME": "YOUR_RUN_NAME",

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

    "AUTO_CREATE_OUTPUT_DIRS": True,
}

CACHE_SCHEMA_VERSION = "YOUR_VERSION"

file_path = Path(CONFIG["INPUT_PATH"])
output_base_dir = Path(f"./HRDSS2.0-results/{CONFIG['RUN_NAME']}")
clinical_result_dir = output_base_dir / f"clinical_results_{CACHE_SCHEMA_VERSION}"
history_dir = output_base_dir / f"dp_history_{CACHE_SCHEMA_VERSION}"
clinical_output_path = output_base_dir / f"clinical_results_{CACHE_SCHEMA_VERSION}.json"


def ensure_output_dirs() -> None:
    for p in [output_base_dir, clinical_result_dir, history_dir]:
        p.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """Validate only the resources required by the no-KG experiment."""
    if not CONFIG.get("HARDCODED_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is empty. Set the environment variable OPENAI_API_KEY before running."
        )
    if not CONFIG.get("API_BASE"):
        raise ValueError("CONFIG['API_BASE'] is empty.")
    if int(CONFIG.get("MAX_WORKERS", 1)) <= 0:
        raise ValueError("CONFIG['MAX_WORKERS'] must be a positive integer.")
    if int(CONFIG.get("MAX_RETRIES", 0)) < 0:
        raise ValueError("CONFIG['MAX_RETRIES'] must be >= 0.")
    if float(CONFIG.get("RETRY_DELAY", 0)) < 0:
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


if CONFIG.get("AUTO_CREATE_OUTPUT_DIRS", True):
    ensure_output_dirs()
