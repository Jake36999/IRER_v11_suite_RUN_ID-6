"""
app.py
CLASSIFICATION: V11.0 Control Plane Server
GOAL: Provides a persistent, web-based meta-orchestration layer.

REMEDIATIONS:
1. OOM Bug Fix : The `update_status` function no longer appends
   to a `found_files` list, fixing the unbounded memory leak.
2. Data Contract Fix : This server now imports `settings.py` to
   use canonical API keys when writing `status.json` and serves
   these keys to the UI via a new `/api/get-constants` endpoint.
"""

import os
import json
import logging
import threading
from flask import Flask, render_template, jsonify, request
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import settings
import core_engine

# --- Configuration & Global State ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ControlHub] - %(message)s')
PROVENANCE_DIR = settings.PROVENANCE_DIR
STATUS_FILE = settings.STATUS_FILE
HUNT_RUNNING_LOCK = threading.Lock()
g_hunt_in_progress = False

app = Flask(__name__, template_folder="templates")

# --- State Management (REMEDIATED) ---
def update_status(new_data: dict = {})
   """
   Thread-safe function to update the central status.json file.
   REMEDIATION : The `found_files` list and
   its associated `.append()` logic have been *removed*.
   """
   with HUNT_RUNNING_LOCK:
       status = {
           settings.API_KEY_HUNT_STATUS: "Idle",
           settings.API_KEY_LAST_EVENT: "-",
           settings.API_KEY_LAST_SSE: "-",
           settings.API_KEY_LAST_STABILITY: "-",
           settings.API_KEY_FINAL_RESULT: {}
       }
       if os.path.exists(STATUS_FILE):
           try:
               with open(STATUS_FILE, 'r') as f:
                   status = json.load(f)
           except json.JSONDecodeError:
               pass # Overwrite corrupted file

       status.update(new_data)

       with open(STATUS_FILE, 'w') as f:
           json.dump(status, f, indent=2)

# --- Watchdog Service (WatcherThread - REMEDIATED) ---
class ProvenanceWatcher(FileSystemEventHandler):
   """Watches for new provenance.json files and updates the status."""

   def on_created(self, event):
       if not event.is_directory and event.src_path.endswith('.json'):
           logging.info(f"Watcher: Detected new artifact: {event.src_path}")

           try:
               with open(event.src_path, 'r') as f:
                   data = json.load(f)

               job_uuid = data.get(settings.HASH_KEY, "unknown")
               # The validation_pipeline.py now puts metrics directly under 'metrics'
               metrics = data.get("metrics", {})
               sse = metrics.get(settings.SSE_METRIC_KEY, -1.0)
               h_norm = metrics.get(settings.STABILITY_METRIC_KEY, -1.0)

               # REMEDIATION :
               # Use canonical API keys from settings.py, not
               # "magic strings" , to write the status update.
               status_data = {
                   settings.API_KEY_LAST_EVENT: f"Analyzed {job_uuid[:8]}...",
                   settings.API_KEY_LAST_SSE: f"{sse:.6f}",
                   settings.API_KEY_LAST_STABILITY: f"{h_norm:.6f}"
               }
               update_status(new_data=status_data)

           except Exception as e:
               logging.error(f"Watcher: Failed to process {event.src_path}: {e}")

def start_watcher_service():
   """Initializes and starts the watchdog observer daemon."""
   os.makedirs(PROVENANCE_DIR, exist_ok=True)
   event_handler = ProvenanceWatcher()
   observer = Observer()
   observer.schedule(event_handler, PROVENANCE_DIR, recursive=False)
   observer.daemon = True
   observer.start()
   logging.info(f"Watcher Service: Started monitoring {PROVENANCE_DIR}")

# --- Core Engine Runner (HuntThread - REMEDIATED) ---
def run_hunt_in_background(num_generations, population_size, grid_size, t_steps):
   """
   Target function for the non-blocking HuntThread.
   """
   global g_hunt_in_progress

   # This lock ensures the g_hunt_in_progress flag is set atomically
   with HUNT_RUNNING_LOCK:
       if g_hunt_in_progress:
           logging.warning("Hunt Thread: Hunt start requested, but already running.")
           return
       g_hunt_in_progress = True

   logging.info(f"Hunt Thread: Starting hunt (Gens: {num_generations}, Pop: {population_size}).")
   final_run_data = {}
   error_message = None

   try:
       # REMEDIATION : Use canonical keys
       update_status(new_data={
           settings.API_KEY_HUNT_STATUS: "Running",
           settings.API_KEY_LAST_EVENT: "Initializing hunt...",
           settings.API_KEY_FINAL_RESULT: {}
       })

       # REMEDIATION: Pass physics parameters to the core engine
       final_run_data = core_engine.execute_hunt(
           num_generations, population_size, grid_size, t_steps
       )

   except Exception as e:
       logging.error(f"Hunt Thread: CRITICAL FAILURE: {e}", exc_info=True)
       error_message = str(e)
   finally:
       # --- REMEDIATION: Deadlock Fix ---
       # This `finally` block is now the authority on hunt completion.
       with HUNT_RUNNING_LOCK:
           if error_message:
               # Case 1: The core_engine.execute_hunt() raised an exception
               update_status(new_data={
                   settings.API_KEY_HUNT_STATUS: f"Error: {error_message}",
                   settings.API_KEY_FINAL_RESULT: {"error": error_message}
               })
           else:
               # Case 2: The hunt finished without an exception.
               update_status(new_data={
                   settings.API_KEY_HUNT_STATUS: "Completed",
                   settings.API_KEY_FINAL_RESULT: final_run_data
               })

           # This flag is the "all clear" signal.
           g_hunt_in_progress = False
           logging.info("Hunt Thread: Hunt finished.")
       # ----------------------------------------------------

# --- Flask API Endpoints (REMEDIATED) ---
@app.route('/')
def index():
   """Serves the main Control Hub UI."""
   return render_template('index')

@app.route('/api/start-hunt', methods=['POST'])
def api_start_hunt():
   """
   Non-blocking endpoint to start a new hunt.
   Spawns the HuntThread and returns 202 immediately.
   """
   if g_hunt_in_progress:
       return jsonify({"status": "error", "message": "A hunt is already in progress."}), 409

   data = request.json or {}

   # REMEDIATION: Read physics params from UI, with fallbacks to settings
   generations = data.get('generations', settings.DEFAULT_NUM_GENERATIONS)
   population = data.get('population', settings.DEFAULT_POPULATION_SIZE)
   grid_size = data.get('grid_size', settings.DEFAULT_GRID_SIZE)
   t_steps = data.get('t_steps', settings.DEFAULT_T_STEPS)

   # Clean up old artifacts before starting
   for d in [settings.CONFIG_DIR, settings.DATA_DIR, settings.PROVENANCE_DIR]:
       if os.path.exists(d):
           for f in os.listdir(d):
               os.remove(os.path.join(d, f))
   if os.path.exists(settings.LEDGER_FILE):
       os.remove(settings.LEDGER_FILE)
   if os.path.exists(settings.STATUS_FILE):
       os.remove(settings.STATUS_FILE)

   # REMEDIATION: Pass all parameters to the background thread
   thread = threading.Thread(
       target=run_hunt_in_background,
       args=(generations, population, grid_size, t_steps)
   )
   thread.daemon = True
   thread.start()

   return jsonify({"status": "ok", "message": "Hunt started."}), 202

@app.route('/api/get-status')
def api_get_status():
   """Asynchronous polling endpoint for the UI."""
   if not os.path.exists(STATUS_FILE):
       return jsonify({settings.API_KEY_HUNT_STATUS: "Idle"})
   try:
       with open(STATUS_FILE, 'r') as f:
           return jsonify(json.load(f))
   except Exception as e:
       logging.warning(f"Failed to read status file: {e}")
       return jsonify({settings.API_KEY_HUNT_STATUS: "Error: Status file read failed."}) # Corrected return


@app.route('/api/get-constants')
def api_get_constants():
   """
   REMEDIATION: New endpoint to serve UI keys.
   This provides the JavaScript frontend with a dynamic data contract,
   eliminating "magic strings".
   """
   return jsonify({
       "HUNT_STATUS": settings.API_KEY_HUNT_STATUS,
       "LAST_EVENT": settings.API_KEY_LAST_EVENT,
       "LAST_SSE": settings.API_KEY_LAST_SSE,
       "LAST_STABILITY": settings.API_KEY_LAST_STABILITY,
       "FINAL_RESULT": settings.API_KEY_FINAL_RESULT
   })

if __name__ == '__main__':
   if not os.path.exists("templates"):
       os.makedirs("templates")
       print("Created 'templates' directory.")

   update_status() # Initialize status file
   start_watcher_service()
   app.run(host='0.0.0.0', port=8080)
