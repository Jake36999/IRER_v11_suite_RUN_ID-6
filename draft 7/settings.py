"""
settings.py
CLASSIFICATION: V11.0 Central Configuration File
GOAL: Acts as the single source of truth for all configuration parameters,
script pointers, and data contract keys for the entire V11.0 suite.
REMEDIATION: This version includes the API_KEY constants to fix the
V11.0 "Magic String" audit gap.

REMEDIATION (V11.0 Bug Fix):
This version now includes default physics and timeout parameters to
fix the "Configuration Violation" (hardcoded "magic numbers")
identified in the V11 audits.
"""

import os

# --- FILE PATHS AND DIRECTORIES ---
BASE_DIR = os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "input_configs")
DATA_DIR = os.path.join(BASE_DIR, "simulation_data")
PROVENANCE_DIR = os.path.join(BASE_DIR, "provenance_reports")
STATUS_FILE = os.path.join(BASE_DIR, "status.json")
LEDGER_FILE = os.path.join(BASE_DIR, "simulation_ledger.csv")

# --- V11.0 SCRIPT POINTERS ---
WORKER_SCRIPT = "worker_sncgl_sdg.py"
VALIDATOR_SCRIPT = "validation_pipeline.py"

# --- EVOLUTIONARY ALGORITHM PARAMETERS ---
LAMBDA_FALSIFIABILITY = 0.1
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 0.05

# --- DATA CONTRACT KEYS (WORKER <-> VALIDATOR <-> HUNTER) ---
HASH_KEY = "config_hash"  # As explicitly requested by the subtask
SSE_METRIC_KEY = "log_prime_sse"
STABILITY_METRIC_KEY = "sdg_h_norm_l2"

# --- DATA CONTRACT KEYS (BACKEND <-> FRONTEND) ---
# REMEDIATION for Audit Gap
API_KEY_HUNT_STATUS = "hunt_status"
API_KEY_LAST_EVENT = "last_event"
API_KEY_LAST_SSE = "last_sse"
API_KEY_LAST_STABILITY = "last_h_norm"
API_KEY_FINAL_RESULT = "final_result"

# --- REMEDIATION (V11.0 Bug Fix): Centralized Parameters ---
# Fixes the "Configuration Violation"
# These are now the "single source of truth" for the hunt.
DEFAULT_NUM_GENERATIONS = 10
DEFAULT_POPULATION_SIZE = 10
DEFAULT_GRID_SIZE = 64
DEFAULT_T_STEPS = 200
DEFAULT_DT = 0.01

# Fixes the hardcoded timeout in core_engine.py
JOB_TIMEOUT_SECONDS = 600
VALIDATOR_TIMEOUT_SECONDS = 300
