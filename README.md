# ML Interview Practice Runner

A local Flask app I built for my own ML Research Scientist interview prep
(Tessera Labs, 2025). Posting it here in case it's useful to anyone else
practicing for similar roles. It runs entirely on your machine — code you
submit executes with your real local NumPy/PyTorch install, not a sandboxed
browser interpreter, so the grading is against the real thing.

## Modes

The home page is a choice between two modes:

### 1. Practice coding exercises

43 concepts / 82 exercises (NumPy and/or PyTorch versions of each), in an
online-judge-style layout: problem description on the left, a VS Code-style
(Monaco) editor and test output on the right.

- **Run tests** executes your code against hidden tests in a real subprocess
  and reports pass/fail per check, with full terminal-style tracebacks on
  failure (not just a generic error).
- **Reveal solution** / **Reset to stub** swap the editor contents.
- A left-hand nav groups exercises by category, with per-exercise and
  per-category pass counts, and a dashboard view showing overall progress.
- Progress (your saved code + pass/fail state per exercise) is persisted to
  a local `data/progress.json`, not just browser storage — it survives
  clearing site data, switching browsers, or restarting the machine.

Categories: Activations & Losses, Layers & Normalization, Classical ML &
Training, CNNs, Sequence Models, Transformers, RL Foundations, Value-Based
RL, Policy-Based RL, RL for LLMs (RLHF), Misc — spanning softmax and
backprop-by-hand up through attention, GAE, PPO, and GRPO.

### 2. Study guide

21 sections of conceptual reference material (foundations, scaling,
reasoning, evaluation, extras), each with a written explainer, curated
YouTube videos, and a self-check quiz. Read/unread state per section is
also persisted server-side. This mode exists for the conceptual and
discussion-style parts of an interview loop that pure coding exercises
don't cover (scaling laws, verifiable rewards, agent memory, long-horizon
reliability, etc.).

## Structure

```
interview_app/
  app.py                 Flask backend -- serves the page, runs submitted code, persists progress
  questions.py            The coding exercise bank (prompts, stubs, solutions, hidden tests)
  study.py                 The study guide content (sections, videos, quizzes)
  requirements.txt
  templates/
    index.html            Page shell only -- embeds concepts/study data as JSON, loads app.js
  static/
    css/style.css          App styling (Dracula theme, OJ-style split-pane practice layout)
    js/app.js               Frontend SPA: routing, rendering, Monaco editor, calling the API
  scripts/
    check_progress.py       Standalone script for a daily local progress notification (optional, see below)
  data/                    Gitignored -- your local progress.json lives here, never committed
```

Monaco (VS Code's editor) is loaded lazily from a CDN the first time you
open a coding exercise; everything else is self-contained.

## Setup

```bash
cd interview_app
pip install -r requirements.txt
pip install numpy          # for the NumPy exercises
pip install torch          # for the PyTorch exercises (see pytorch.org for your platform)
```

## Run

```bash
python app.py
```

Then open **http://127.0.0.1:5050** in your browser.

> Why 5050 and not 5000? On macOS, port 5000 is claimed by the AirPlay
> Receiver service (System Settings &rarr; General &rarr; AirDrop & Handoff),
> which causes some clients (notably Safari) to fail to load the page at
> all, often with no visible error.

## Optional: daily progress notification

`scripts/check_progress.py` compares the current `data/progress.json`
against a snapshot and, if anything changed, fires a native macOS
notification summarizing the day's progress. It's meant to be run once a
day from `cron` (it needs local filesystem access, so it can't run as a
cloud-based scheduled task):

```bash
crontab -e
# add a line like:
0 20 * * * /path/to/python /path/to/interview_app/scripts/check_progress.py
```

## Notes

- This executes arbitrary code you type as a real local subprocess. That's
  expected and fine for your own practice on your own machine — just be
  aware it's not sandboxed (don't paste in code from strangers and run it
  through here).
- `data/` (your personal progress) is gitignored on purpose — it's
  per-user local state, not something to commit or share.
