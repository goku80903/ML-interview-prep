"""
Flask backend for the interview practice app.

Runs entirely on your local machine, so whatever's `pip install`-ed in
this environment (numpy, torch, ...) is what actually gets exercised
when you hit "Run tests" -- unlike a browser sandbox, there's no
restriction on which frameworks are available.

Run with:
    pip install -r requirements.txt
    python app.py
then open http://127.0.0.1:5050

(Port 5050, not 5000 -- on macOS, 5000 is claimed by the AirPlay Receiver
service, which causes silent connection failures for some clients.)
"""
import json
import subprocess
import sys
import tempfile
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from questions import CONCEPTS, CONCEPTS_BY_ID
from study import STUDY_SECTIONS, STUDY_CATEGORIES

app = Flask(__name__)

TIMEOUT_SECONDS = 15

# Progress lives on disk (not just the browser) so it survives clearing site
# data, switching browsers, or restarting the machine. This file is
# per-user local state, not source -- keep it out of git (see .gitignore).
PROGRESS_PATH = Path(__file__).parent / "data" / "progress.json"
_progress_lock = threading.Lock()


def _load_progress() -> dict:
    try:
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(PROGRESS_PATH)

HARNESS = {
    "numpy": '''import numpy as np, json, traceback
_results = []
def _check(name, fn):
    try:
        fn()
        _results.append({"name": name, "passed": True, "message": ""})
    except NotImplementedError:
        _results.append({"name": name, "passed": False, "message": "Not implemented yet"})
    except Exception:
        tb = traceback.format_exc()
        _results.append({"name": name, "passed": False, "message": tb})
''',
    "torch": '''import torch, torch.nn as nn, json, traceback
_results = []
def _check(name, fn):
    try:
        fn()
        _results.append({"name": name, "passed": True, "message": ""})
    except NotImplementedError:
        _results.append({"name": name, "passed": False, "message": "Not implemented yet"})
    except Exception:
        tb = traceback.format_exc()
        _results.append({"name": name, "passed": False, "message": tb})
''',
}


def run_in_subprocess(source: str):
    """Write `source` to a temp file and execute it with this same
    Python interpreter, so it sees whatever packages are installed
    in this environment. Returns (results_list_or_None, error_str_or_None,
    console_str_or_None).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name

    def _clean(text):
        # The traceback's file path is a meaningless temp path -- swap it for
        # something that reads like an actual terminal error.
        return text.replace(path, "your_submission.py") if text else text

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, f"Timed out after {TIMEOUT_SECONDS}s (infinite loop?)", None
    finally:
        os.unlink(path)

    stdout_lines = proc.stdout.splitlines()
    # The harness's `print(json.dumps(_results))` is always the last line of
    # stdout; everything before it is the student's own print() output.
    console = _clean("\n".join(stdout_lines[:-1]).strip()[-4000:]) or None

    if proc.returncode != 0 and not proc.stdout.strip():
        # Crashed before printing anything (e.g. a SyntaxError in the student's code)
        return None, _clean(proc.stderr.strip()[-4000:]), console

    last_line = stdout_lines[-1].strip() if stdout_lines else ""
    try:
        results = json.loads(last_line)
        for r in results:
            r["message"] = _clean(r.get("message", ""))
        return results, None, console
    except json.JSONDecodeError:
        return None, _clean((proc.stderr.strip() or proc.stdout.strip())[-4000:]), console


@app.route("/")
def index():
    # Only ship what the frontend needs -- omit `tests` (kept server-side).
    concepts = []
    for c in CONCEPTS:
        entry = {"id": c["id"], "title": c["title"], "category": c.get("category", "Misc")}
        for lang in ("numpy", "torch"):
            v = c[lang]
            entry[lang] = None if v is None else {
                "prompt": v["prompt"], "stub": v["stub"], "solution": v["solution"]
            }
        concepts.append(entry)
    return render_template(
        "index.html",
        concepts_json=json.dumps(concepts),
        study_sections_json=json.dumps(STUDY_SECTIONS),
        study_categories_json=json.dumps(STUDY_CATEGORIES),
    )


@app.route("/api/progress")
def api_progress():
    return jsonify(_load_progress())


@app.route("/api/save", methods=["POST"])
def api_save():
    """Persist just the code for an id/lang, e.g. while the student is
    still typing (hasn't run tests yet). Doesn't touch pass/fail state."""
    data = request.get_json(force=True)
    concept_id = data.get("id")
    lang = data.get("lang")
    code = data.get("code", "")
    key = f"{concept_id}::{lang}"

    with _progress_lock:
        progress = _load_progress()
        entry = progress.setdefault(key, {})
        entry["code"] = code
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_progress(progress)
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear saved exercise progress. With {id, lang}, clears just that one
    exercise's saved code and pass/fail state. With no body, wipes every
    exercise -- but leaves Study Guide read-progress alone (see
    /api/study/read), since that's a separate concern from coding exercises."""
    data = request.get_json(silent=True) or {}
    concept_id = data.get("id")
    lang = data.get("lang")

    with _progress_lock:
        progress = _load_progress()
        if concept_id and lang:
            progress.pop(f"{concept_id}::{lang}", None)
        else:
            progress = {k: v for k, v in progress.items() if k.startswith("study::")}
        _save_progress(progress)
    return jsonify({"ok": True})


@app.route("/api/study/read", methods=["POST"])
def api_study_read():
    """Persist a Study Guide section's read/unread state on disk, namespaced
    under 'study::<id>' in the same progress store as the exercises."""
    data = request.get_json(force=True)
    section_id = data.get("id")
    read = bool(data.get("read"))
    key = f"study::{section_id}"

    with _progress_lock:
        progress = _load_progress()
        if read:
            progress[key] = {"read": True, "updated_at": datetime.now(timezone.utc).isoformat()}
        else:
            progress.pop(key, None)
        _save_progress(progress)
    return jsonify({"ok": True})


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True)
    concept_id = data.get("id")
    lang = data.get("lang")
    code = data.get("code", "")

    concept = CONCEPTS_BY_ID.get(concept_id)
    if concept is None:
        return jsonify({"error": f"Unknown question id '{concept_id}'"}), 400
    version = concept.get(lang)
    if version is None:
        return jsonify({"error": f"No {lang} version for '{concept_id}'"}), 400

    # Normalize any literal tab characters to 4 spaces. Mixed tabs/spaces
    # look identical in most editors but make Python raise IndentationError,
    # so this is a safety net regardless of what produced the code.
    code = code.expandtabs(4)

    source = HARNESS[lang] + "\n" + code + "\n" + version["tests"] + "\nprint(json.dumps(_results))"
    results, error, console = run_in_subprocess(source)

    key = f"{concept_id}::{lang}"
    passed = error is None and bool(results) and all(r["passed"] for r in results)
    with _progress_lock:
        progress = _load_progress()
        entry = progress.setdefault(key, {})
        entry["code"] = code
        entry["ran_once"] = True
        entry["passed"] = passed
        entry["results"] = results
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_progress(progress)

    if error is not None:
        return jsonify({"results": None, "error": error, "console": console})
    return jsonify({"results": results, "error": None, "console": console})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
