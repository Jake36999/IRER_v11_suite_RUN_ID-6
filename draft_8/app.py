import os
import json
import logging
import threading
import shutil  # Added for safer directory cleaning
from flask import Flask, render_template, jsonify, request
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Import Local Modules ---
# Ensure settings.py and core_engine.py exist in the same folder
import settings
import core_engine

# --- Configuration & Global State ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ControlHub] - %(message)s')
PROVENANCE_DIR = settings.PROVENANCE_DIR
STATUS_FILE = settings.STATUS_FILE
HUNT_RUNNING_LOCK = threading.Lock()
g_hunt_in_progress = False

app = Flask(__name__, template_folder="templates")

# --- State Management ---
def update_status(new_data: dict = None):
    """
    Thread-safe function to update the central status.json file.
    Includes new keys for real-time progress.
    """
    if new_data is None:
        new_data = {}

    with HUNT_RUNNING_LOCK:
        status = {
            settings.API_KEY_HUNT_STATUS: "Idle",
            settings.API_KEY_LAST_EVENT: "-",
            settings.API_KEY_LAST_SSE: "-",
            settings.API_KEY_LAST_STABILITY: "-",
            settings.API_KEY_FINAL_RESULT: {},
            "current_gen": 0, # NEW: Current generation count
            "total_gens": 0  # NEW: Total generations configured
        }
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, 'r') as f:
                    existing_status = json.load(f)
                    # Merge existing status with defaults to ensure new keys are present
                    status.update(existing_status)
            except json.JSONDecodeError:
                pass # Overwrite corrupted file

        status.update(new_data)

        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)

# --- Watchdog Service ---
class ProvenanceWatcher(FileSystemEventHandler):
    """Watches for new provenance.json files and updates the status."""

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            logging.info(f"Watcher: Detected new artifact: {event.src_path}")

            try:
                with open(event.src_path, 'r') as f:
                    data = json.load(f)

                job_uuid = data.get(settings.HASH_KEY, "unknown")
                metrics = data.get("metrics", {})
                sse = metrics.get(settings.SSE_METRIC_KEY, -1.0)
                h_norm = metrics.get(settings.STABILITY_METRIC_KEY, -1.0)

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

# --- Core Engine Runner ---
def run_hunt_in_background(num_generations, population_size, grid_size, t_steps):
    """
    Target function for the non-blocking HuntThread.
    """
    global g_hunt_in_progress

    with HUNT_RUNNING_LOCK:
        if g_hunt_in_progress:
            logging.warning("Hunt Thread: Hunt start requested, but already running.")
            return
        g_hunt_in_progress = True

    logging.info(f"Hunt Thread: Starting hunt (Gens: {num_generations}, Pop: {population_size}).")
    final_run_data = {}
    error_message = None

    try:
        # Initial status update including total generations
        update_status(new_data={
            settings.API_KEY_HUNT_STATUS: "Running",
            settings.API_KEY_LAST_EVENT: "Initializing hunt...",
            settings.API_KEY_FINAL_RESULT: {},
            "current_gen": 0,
            "total_gens": num_generations
        })

        # Execute the simulation
        final_run_data = core_engine.execute_hunt(
            num_generations, population_size, grid_size, t_steps
        )

    except Exception as e:
        logging.error(f"Hunt Thread: CRITICAL FAILURE: {e}", exc_info=True)
        error_message = str(e)
    finally:
        with HUNT_RUNNING_LOCK:
            if error_message:
                update_status(new_data={
                    settings.API_KEY_HUNT_STATUS: f"Error: {error_message}",
                    settings.API_KEY_FINAL_RESULT: {"error": error_message}
                })
            else:
                update_status(new_data={
                    settings.API_KEY_HUNT_STATUS: "Completed",
                    settings.API_KEY_FINAL_RESULT: final_run_data
                })

            g_hunt_in_progress = False
            logging.info("Hunt Thread: Hunt finished.")

# --- Flask API Endpoints ---
@app.route('/')
def index():
    """Serves the main Control Hub UI."""
    # Check if template exists to avoid 500 error
    if not os.path.exists(os.path.join("templates", "index.html")):
        return "<h1>Control Hub Running</h1><p>Please create templates/index.html</p>"
    return render_template('index.html')

@app.route('/api/start-hunt', methods=['POST'])
def api_start_hunt():
    """
    Non-blocking endpoint to start a new hunt.
    """
    if g_hunt_in_progress:
        return jsonify({"status": "error", "message": "A hunt is already in progress."}), 409

    # Safer JSON retrieval
    data = request.get_json(silent=True) or {}

    generations = data.get('generations', settings.DEFAULT_NUM_GENERATIONS)
    population = data.get('population', settings.DEFAULT_POPULATION_SIZE)
    grid_size = data.get('grid_size', settings.DEFAULT_GRID_SIZE)
    t_steps = data.get('t_steps', settings.DEFAULT_T_STEPS)

    # CLEANUP LOGIC FIXED
    for d in [settings.CONFIG_DIR, settings.DATA_DIR, settings.PROVENANCE_DIR]:
        if os.path.exists(d):
            for filename in os.listdir(d):
                file_path = os.path.join(d, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) # Recursive delete for folders
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}. Reason: {e}")

    # Clean specific files
    for f_path in [settings.LEDGER_FILE, settings.STATUS_FILE]:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except OSError:
                pass

    thread = threading.Thread(
        target=run_hunt_in_background,
        args=(generations, population, grid_size, t_steps)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "ok", "message": "Hunt started."}), 202

@app.route('/api/get-status')
def api_get_status():
    if not os.path.exists(STATUS_FILE):
        return jsonify({settings.API_KEY_HUNT_STATUS: "Idle"})
    try:
        with open(STATUS_FILE, 'r') as f:
            # Only return the core status keys
            data = json.load(f)
            return jsonify({
                settings.API_KEY_HUNT_STATUS: data.get(settings.API_KEY_HUNT_STATUS, "Unknown"),
                settings.API_KEY_LAST_EVENT: data.get(settings.API_KEY_LAST_EVENT, "-"),
                settings.API_KEY_LAST_SSE: data.get(settings.API_KEY_LAST_SSE, "-"),
                settings.API_KEY_LAST_STABILITY: data.get(settings.API_KEY_LAST_STABILITY, "-"),
                settings.API_KEY_FINAL_RESULT: data.get(settings.API_KEY_FINAL_RESULT, {})
            })
    except Exception as e:
        logging.warning(f"Failed to read status file: {e}")
        return jsonify({settings.API_KEY_HUNT_STATUS: "Error: Status file read failed."})

@app.route('/api/get-progress')
def api_get_progress():
    """
    New endpoint to fetch real-time progress for the loading bar.
    """
    if not os.path.exists(STATUS_FILE):
        return jsonify({"current_gen": 0, "total_gens": 0})
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
            return jsonify({
                "current_gen": data.get("current_gen", 0),
                "total_gens": data.get("total_gens", 0)
            })
    except Exception as e:
        logging.warning(f"Failed to read status file for progress: {e}")
        return jsonify({"current_gen": 0, "total_gens": 0})

@app.route('/api/get-constants')
def api_get_constants():
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

    update_status() 
    start_watcher_service()
    app.run(host='0.0.0.0', port=8080)