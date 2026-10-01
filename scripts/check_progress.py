#!/usr/bin/env python3
"""
Daily interview-prep progress check, run locally via cron (see scripts/README.md).

Reads data/progress.json directly (local-only, gitignored -- this script must
run on this machine, not in a cloud sandbox), compares against yesterday's
snapshot, and fires a native macOS notification with the day-over-day delta.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROGRESS_PATH = PROJECT_DIR / "data" / "progress.json"
SNAPSHOT_PATH = PROJECT_DIR / "data" / ".notify_snapshot.json"

sys.path.insert(0, str(PROJECT_DIR))
from questions import CONCEPTS  # noqa: E402
from study import STUDY_SECTIONS  # noqa: E402


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def compute_stats():
    progress = load_json(PROGRESS_PATH)
    exercises_total = sum(1 for c in CONCEPTS for lang in ("numpy", "torch") if c.get(lang))
    exercises_passed = sum(
        1 for key, entry in progress.items()
        if not key.startswith("study::") and entry.get("ran_once") and entry.get("passed")
    )
    study_total = len(STUDY_SECTIONS)
    study_read = sum(
        1 for key, entry in progress.items()
        if key.startswith("study::") and entry.get("read")
    )
    return {
        "exercises_passed": exercises_passed,
        "exercises_total": exercises_total,
        "study_read": study_read,
        "study_total": study_total,
    }


def notify(title, message):
    # osascript's AppleScript string literals: escape backslashes then quotes.
    def esc(s):
        return s.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{esc(message)}" with title "{esc(title)}"'
    subprocess.run(["osascript", "-e", script], check=False)


def main():
    stats = compute_stats()
    prev = load_json(SNAPSHOT_PATH)

    lines = [
        f"Exercises: {stats['exercises_passed']}/{stats['exercises_total']} passed",
        f"Study guide: {stats['study_read']}/{stats['study_total']} read",
    ]
    if prev:
        d_ex = stats["exercises_passed"] - prev.get("exercises_passed", stats["exercises_passed"])
        d_st = stats["study_read"] - prev.get("study_read", stats["study_read"])
        if d_ex or d_st:
            deltas = []
            if d_ex:
                deltas.append(f"{'+' if d_ex > 0 else ''}{d_ex} exercises")
            if d_st:
                deltas.append(f"{'+' if d_st > 0 else ''}{d_st} sections")
            lines.append("Since yesterday: " + ", ".join(deltas))
        else:
            lines.append("No progress since yesterday")
    else:
        lines.append("(first check-in)")

    notify("Tessera Interview Prep", "  |  ".join(lines))

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
