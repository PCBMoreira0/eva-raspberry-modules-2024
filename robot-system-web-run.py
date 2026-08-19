from flask import Flask, render_template, request
import atexit
import os
import signal
import subprocess
import threading

# Modules are started relative to this file, not to the current directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODULE_COMMANDS = {
    "cv": ["python3", "eva-cv-module/eva-cv.py"],
    "tts": ["python3", "eva-tts-module/eva-tts.py"],
    # All libraries from the TER module are in the global python dir.
    "ter": ["/usr/bin/python", "eva-ter-module/eva-ter.py"],
    "leds": ["python3", "eva-leds-module/eva-leds.py"],
    "light": ["python3", "eva-light-module/eva-light.py"],
    "audio": ["python3", "eva-audio-module/eva-audio.py"],
    "motion": ["python3", "eva-motion-module/eva-motion.py"],
    "stt": ["python3", "eva-stt-module/eva-stt.py"],
    "display": ["python3", "eva-display-module/eva-display.py"],
}

TERMINATION_TIMEOUT = 5.0

processes = {name: None for name in MODULE_COMMANDS}
processes_lock = threading.RLock()

app = Flask(__name__)


# ----------------------------------------------------------------------
# Process control
# ----------------------------------------------------------------------

def is_running(name):
    """A finished process is reaped here, so it never counts as running."""

    process = processes.get(name)

    if process is None:
        return False

    if process.poll() is not None:
        processes[name] = None
        return False

    return True


def start_module(name):
    with processes_lock:
        if is_running(name):
            print(
                "Module already running:",
                name,
                "PID:",
                processes[name].pid,
            )
            return False

        try:
            # start_new_session puts the module in its own process group,
            # so every process it spawns can be signalled together.
            process = subprocess.Popen(
                MODULE_COMMANDS[name],
                cwd=BASE_DIR,
                start_new_session=True,
            )

        except Exception as error:
            print(
                "Unable to start module:",
                name,
                str(error),
            )
            return False

        processes[name] = process

        print(
            "Running module:",
            name,
            "PID:",
            process.pid,
        )

        return True


def terminate_process_tree(process):
    try:
        group_id = os.getpgid(process.pid)
    except OSError:
        group_id = None

    try:
        if group_id is not None:
            os.killpg(group_id, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass

    try:
        process.wait(timeout=TERMINATION_TIMEOUT)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if group_id is not None:
            os.killpg(group_id, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass

    try:
        process.wait(timeout=TERMINATION_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(
            "Process did not terminate. PID:",
            process.pid,
        )


def stop_module(name):
    with processes_lock:
        process = processes.get(name)

        if process is None:
            print("Module is not running:", name)
            return False

        if process.poll() is not None:
            processes[name] = None
            print("Module already finished:", name)
            return False

        print(
            "Killing module:",
            name,
            "PID:",
            process.pid,
        )

        terminate_process_tree(process)
        processes[name] = None

        return True


def start_all_modules():
    with processes_lock:
        for name in MODULE_COMMANDS:
            start_module(name)


def stop_all_modules():
    with processes_lock:
        for name in MODULE_COMMANDS:
            stop_module(name)


def modules_status():
    with processes_lock:
        return {
            name: is_running(name)
            for name in MODULE_COMMANDS
        }


def handle_module_request(name):
    if "btn_run_" + name + "_module" in request.form:
        start_module(name)
    elif "btn_kill_" + name + "_module" in request.form:
        stop_module(name)

    return render_template(
        "index.html",
        modules_status=modules_status(),
    )


# ----------------------------------------------------------------------
# Shutdown handling
# ----------------------------------------------------------------------

shutdown_done = threading.Event()


def shutdown(signal_number=None, frame=None):
    if shutdown_done.is_set():
        return

    shutdown_done.set()

    print("Shutting down. Killing all modules.")
    stop_all_modules()

    if signal_number is not None:
        # Restores the default behaviour so the exit status is correct.
        signal.signal(signal_number, signal.SIG_DFL)
        os.kill(os.getpid(), signal_number)


atexit.register(shutdown)
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGHUP, shutdown)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        modules_status=modules_status(),
    )


@app.route("/eva_cv_module", methods=["POST"])
def eva_cv_module():
    return handle_module_request("cv")


@app.route("/eva_tts_module", methods=["POST"])
def eva_tts_module():
    return handle_module_request("tts")


@app.route("/eva_ter_module", methods=["POST"])
def eva_ter_module():
    return handle_module_request("ter")


@app.route("/eva_leds_module", methods=["POST"])
def eva_leds_module():
    return handle_module_request("leds")


@app.route("/eva_light_module", methods=["POST"])
def eva_light_module():
    return handle_module_request("light")


@app.route("/eva_audio_module", methods=["POST"])
def eva_audio_module():
    return handle_module_request("audio")


@app.route("/eva_motion_module", methods=["POST"])
def eva_motion_module():
    return handle_module_request("motion")


@app.route("/eva_stt_module", methods=["POST"])
def eva_stt_module():
    return handle_module_request("stt")


@app.route("/eva_display_module", methods=["POST"])
def eva_display_module():
    return handle_module_request("display")


@app.route("/eva_all_modules", methods=["POST"])
def eva_all_modules():
    if "btn_run_all_modules" in request.form:
        print("Running all modules.")
        start_all_modules()

    elif "btn_kill_all_modules" in request.form:
        print("Killing all modules.")
        stop_all_modules()

    return render_template(
        "index.html",
        modules_status=modules_status(),
    )


# ----------------------------------------------------------------------
# Application start
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # The reloader would run this file in two processes, duplicating
        # every module and losing track of the started ones.
        app.run(
            host="0.0.0.0",
            debug=True,
            use_reloader=False,
            threaded=True,
        )
    finally:
        shutdown()