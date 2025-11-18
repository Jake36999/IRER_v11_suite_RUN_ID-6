"""
settings.py
CLASSIFICATION: V11.0 Central Configuration File
GOAL: Acts as the single source of truth for all configuration parameters.
STATUS: REMEDIATED (Run 14 Gold Master + API Consolidation)
"""
import os
from pathlib import Path

BASE_DIR = Path(os.getcwd())

# --- DIRECTORY CONFIGURATION (Pathlib - Modern) ---
PROVENANCE_DIR = BASE_DIR / "provenance_reports"
DATA_DIR = BASE_DIR / "simulation_data"
CONFIG_DIR = BASE_DIR / "input_configs"
LOG_DIR = BASE_DIR / "logs"
LEDGER_FILE = BASE_DIR / "simulation_ledger.csv"

# --- SCRIPT POINTERS ---
WORKER_SCRIPT = "worker_sncgl_sdg.py"
VALIDATOR_SCRIPT = "validation_pipeline.py"

# --- EVOLUTIONARY ALGORITHM DEFAULTS ---
DEFAULT_NUM_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 10
DEFAULT_GRID_SIZE = 64
DEFAULT_T_STEPS = 200
DEFAULT_DT = 0.01
LAMBDA_FALSIFIABILITY = 0.1
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.05

# --- RESOURCE LIMITS ---
JOB_TIMEOUT_SECONDS = 600
VALIDATOR_TIMEOUT_SECONDS = 300

# --- DATA CONTRACT KEYS (BACKEND) ---
# MANDATE: "job_uuid" is required to prevent pipeline deadlocks.
HASH_KEY = "job_uuid"  
SSE_METRIC_KEY = "log_prime_sse"
STABILITY_METRIC_KEY = "sdg_h_norm_l2"
METRIC_BLOCK_SPECTRAL = "spectral_fidelity"

# --- API KEYS (FRONTEND INTERFACE) ---
API_KEY_HUNT_STATUS = "hunt_status"
API_KEY_LAST_EVENT = "last_event"
API_KEY_LAST_SSE = "last_sse"
API_KEY_LAST_STABILITY = "last_h_norm"
API_KEY_FINAL_RESULT = "final_result"