#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Error: activate your cse-471 Conda environment first." >&2
    exit 1
fi

PYTHON="$CONDA_PREFIX/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python not found at $PYTHON" >&2
    exit 1
fi

for script in research_agent.py verify_answer.py render_answer.py; do
    if [[ ! -f "$script" ]]; then
        echo "Error: missing $script" >&2
        exit 1
    fi
done

if [[ ! -f ".env" ]]; then
    echo "Error: .env is missing from the project folder." >&2
    exit 1
fi

# Existing scripts use timestamp-based output folders.
# Record completed runs so we can identify the newly created one.
BEFORE_RUNS="$("$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("output")
runs = (
    sorted(str(p.resolve()) for p in root.iterdir()
           if p.is_dir() and (p / "trace.json").exists())
    if root.exists() else []
)
print(json.dumps(runs))
PY
)"

echo
echo "=== 1/3: Research ==="
"$PYTHON" research_agent.py

RUN_DIR="$(BEFORE_RUNS="$BEFORE_RUNS" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

before = set(json.loads(os.environ["BEFORE_RUNS"]))
root = Path("output")

after = {
    str(p.resolve())
    for p in root.iterdir()
    if p.is_dir() and (p / "trace.json").exists()
}

new_runs = sorted(after - before)

if len(new_runs) != 1:
    raise SystemExit(
        "Expected exactly one new completed research run. "
        "Do not run multiple research processes concurrently."
    )

print(new_runs[0])
PY
)"

echo
echo "Research run: $RUN_DIR"

# The verifier starts a new Python process and does not share the
# research process's in-memory rate limiter.
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-65}"

echo
echo "Waiting ${COOLDOWN_SECONDS}s before verification..."
sleep "$COOLDOWN_SECONDS"

echo
echo "=== 2/3: Verify claims ==="
"$PYTHON" verify_answer.py "$RUN_DIR"

echo
echo "=== 3/3: Render final answer ==="
"$PYTHON" render_answer.py "$RUN_DIR"

if [[ ! -s "$RUN_DIR/final_answer.md" ]]; then
    echo "Error: final_answer.md was not created or is empty." >&2
    exit 1
fi

echo
echo "=== FINAL ANSWER ==="
cat "$RUN_DIR/final_answer.md"

echo
echo "=== Pipeline complete ==="
echo "All outputs: $RUN_DIR"