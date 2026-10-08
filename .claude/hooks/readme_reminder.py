"""PostToolUse hook (Edit|Write|NotebookEdit): after a pipeline/**/*.py
file or a notebook changes, remind Claude to bring the matching README(s)
up to date. Configured in .claude/settings.json."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def emit(message):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": message,
        }
    }))


data = json.load(sys.stdin)
tool_input = data.get("tool_input") or {}
file_path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
path = Path(file_path).resolve()

try:
    rel = path.relative_to(ROOT)
except ValueError:
    sys.exit(0)

if path.suffix == ".ipynb" and ".ipynb_checkpoints" not in rel.parts:
    emit(
        f"{rel} was just changed. Notebooks (especially notebooks/sandbox.ipynb) "
        "are the usage reference for the pipeline. Before finishing, check "
        "whether the usage examples in pipeline/README.md ('Two ways to run the "
        "pipeline', 'Typical run') and the per-package READMEs still match how "
        "the notebook now calls the steps (class names, arguments, run order). "
        "Update them only if the notebook shows a real change in how the API is "
        "used; leave them alone for experiment-only changes (parameter sweeps, "
        "plots, outputs)."
    )
    sys.exit(0)

if rel.parts[:1] != ("pipeline",) or path.suffix != ".py" or "__pycache__" in rel.parts:
    sys.exit(0)

readmes = []
package_readme = path.parent / "README.md"
if package_readme.exists():
    readmes.append(package_readme.relative_to(ROOT))
top_readme = ROOT / "pipeline" / "README.md"
if top_readme.exists() and top_readme.relative_to(ROOT) not in readmes:
    readmes.append(top_readme.relative_to(ROOT))

message = (
    f"{rel} was just changed. Before finishing, check whether "
    f"{' / '.join(str(r) for r in readmes)} still describes it correctly "
    "(signatures, defaults, behavior, file listing) and update the README if "
    "not. pipeline/README.md only needs changes for cross-stage behavior "
    "(Pipeline, run modes, output paths). If the change doesn't affect "
    "anything documented, leave the README alone."
)
emit(message)
