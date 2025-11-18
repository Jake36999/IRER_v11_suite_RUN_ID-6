"""
core_engine.py
CLASSIFICATION: V11.0 Data Plane Orchestrator
GOAL: Encapsulates the blocking, long-running evolutionary hunt logic.
This is a module, not an executable. It is imported by `app.py` and
run in a background thread, fixing the V10.x "Blocking Server" failure.

REMEDIATION: This version implements the deterministic `hashlib` (Variant A)
hashing mandate, replacing the non-deterministic `uuid.uuid4()` (Variant B)
to ensure reproducibility.
"""

import os
import sys
import json
import subprocess
import hashlib # REMEDIATION: Use hashlib for deterministic hashing
import logging
import time
from typing import Dict, Any, List, Optional

import settings
from aste_hunter import Hunter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CoreEngine] - %(message)s')

def generate_deterministic_hash(params: dict) -> str:
   """
   REMEDIATION: Implements the "Unified Hashing Mandate" (Variant A)
   using a content-based SHA1 hash.
   This ensures that
   identical parameters *always* produce an identical ID, guaranteeing
   reproducibility.
   """
   param_str = json.dumps(params, sort_keys=True).encode('utf-8')
   return hashlib.sha1(param_str).hexdigest()

# REMEDIATION: Function now accepts parameters to fix hardcoding
def _generate_config_file(job_uuid: str, params: Dict, gen: int, i: int, grid_size: int, t_steps: int, dt: float) -> str:
   """Generates a unique JSON config file for a specific job."""
   config = {
       settings.HASH_KEY: job_uuid,
       "generation": gen,
       "seed": (gen * 1000) + i,

       # REMEDIATION: Use arguments instead of hardcoded values
       "N_grid": grid_size,
       "T_steps": t_steps,
       "dt": dt,
       # -----------------------------------------------------

       **params # Evolutionary params
   }

   config_path = os.path.join(settings.CONFIG_DIR, f"config_{job_uuid}.json")
   with open(config_path, 'w') as f:
       json.dump(config, f, indent=2)
   return config_path

def _run_simulation_job(job_uuid: str, config_path: str) -> bool:
   """Runs a single Worker + Validator job as a subprocess."""

   # --- 1. Run Worker (Data Plane) ---
   worker_cmd = [sys.executable, settings.WORKER_SCRIPT, "--job_uuid", job_uuid, "--params", config_path]
   try:
       logging.info(f"Job {job_uuid[:8]}: Starting Worker...")
       # REMEDIATION: Use centralized timeout from settings
       subprocess.run(worker_cmd, check=True, capture_output=True, text=True, timeout=settings.JOB_TIMEOUT_SECONDS)
   except subprocess.CalledProcessError as e:
       logging.error(f"Job {job_uuid[:8]}: WORKER FAILED.\nSTDERR: {e.stderr}")
       return False
   except subprocess.TimeoutExpired:
       logging.error(f"Job {job_uuid[:8]}: WORKER TIMED OUT.")
       return False

   # --- 2. Run Validator (Analysis Plane) ---
   validator_cmd = [sys.executable, settings.VALIDATOR_SCRIPT, "--job_uuid", job_uuid]
   try:
       logging.info(f"Job {job_uuid[:8]}: Starting Validator...")
       # REMEDIATION: Use centralized timeout from settings
       subprocess.run(validator_cmd, check=True, capture_output=True, text=True, timeout=settings.VALIDATOR_TIMEOUT_SECONDS)
   except subprocess.CalledProcessError as e:
       logging.error(f"Job {job_uuid[:8]}: VALIDATOR FAILED.\nSTDERR: {e.stderr}")
       return False
   except subprocess.TimeoutExpired:
       logging.error(f"Job {job_uuid[:8]}: VALIDATOR TIMED OUT.")
       return False

   logging.info(f"Job {job_uuid[:8]}: Run SUCCEEDED.")
   return True

# REMEDIATION: Function now accepts parameters from app.py
def execute_hunt(
    num_generations: int,
    population_size: int,
    grid_size: int,
    t_steps: int
) -> Dict[str, Any]:
   """
   The main evolutionary hunt loop. This function is designed to be
   called by app.py in a background thread.
   """
   logging.info(f"--- V11.0 HUNT STARTING ---")
   logging.info(f"Gens: {num_generations}, Pop: {population_size}, Grid: {grid_size}, Steps: {t_steps}")

   for d in [settings.CONFIG_DIR, settings.DATA_DIR, settings.PROVENANCE_DIR]:
       os.makedirs(d, exist_ok=True)

   hunter = Hunter()
   final_best_run: Optional[Dict[str, Any]] = None

   for gen in range(num_generations):
       logging.info(f"--- GENERATION {gen}/{num_generations-1} ---")

       param_batch = hunter.breed_next_generation(population_size)
       job_contexts = []
       jobs_completed_this_gen = 0

       for i, params in enumerate(param_batch):
           job_uuid = generate_deterministic_hash(params)

           # REMEDIATION: Pass physics params to config file generator
           config_path = _generate_config_file(
               job_uuid, params, gen, i,
               grid_size=grid_size,
               t_steps=t_steps,
               dt=settings.DEFAULT_DT
           )
           job_contexts.append({"uuid": job_uuid, "params": params, "config": config_path})

           run_data = {"generation": gen, settings.HASH_KEY: job_uuid, **params}
           if not any(r[settings.HASH_KEY] == job_uuid for r in hunter.population):
               hunter.population.append(run_data)

       hunter._save_ledger()

       for i, job in enumerate(job_contexts):
           logging.info(f"Gen {gen}, Job {i}: Spawning run {job['uuid'][:8]}...")
           if _run_simulation_job(job["uuid"], job["config"]):
               jobs_completed_this_gen += 1

           # REMEDIATION: Check if the file *actually* exists before proceeding
           # This check is the core of the V11.0 Deadlock Fix
           # We check here to provide early feedback to the hunter.
           provenance_path = os.path.join(settings.PROVENANCE_DIR, f"provenance_{job['uuid']}.json")
           if not os.path.exists(provenance_path):
               logging.warning(f"Job {job['uuid'][:8]}: SUCCEEDED but FAILED TO PRODUCE PROVENANCE.json.")
               # The hunter will handle this gracefully, but this log is critical.

       logging.info(f"--- Gen {gen} Complete. Processing results... ---")
       hunter.process_generation_results()

       best_run = hunter.get_best_run()
       if best_run:
           final_best_run = best_run
           logging.info(f"Current Best: {final_best_run[settings.HASH_KEY][:8]} (Fitness: {final_best_run.get('fitness', 0):.4f})")

       # REMEDIATION: If no jobs in a generation produced a file, the hunt has failed.
       if jobs_completed_this_gen == 0:
           logging.error("CRITICAL DEADLOCK: Generation completed with 0 successful jobs. Aborting hunt.")
           raise Exception("Generation failed to produce any valid provenance artifacts.")

   logging.info(f"--- V11.0 HUNT COMPLETE ---")
   return final_best_run if final_best_run else {}
