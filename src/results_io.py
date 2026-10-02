"""Writes every headline number the README quotes to outputs/results.json.

Why this module exists
---------------------
Before this, every metric in the README was typed in by hand. That is the exact
failure mode this portfolio has hit before: documentation describing results the
code never produced, discovered only when someone re-ran it. A number that is
typed can drift from the code silently; a number that is read from a file the
code writes cannot.

The rule that goes with it: the README quotes this file, and nothing else. If a
number is not in here, it does not belong in the README.

Provenance is recorded alongside the metrics deliberately. "val AUC 0.7622" on
its own is unfalsifiable. "val AUC 0.7622, from commit d74c298, real data,
61,503 validation rows, numpy 2.x, on this date" can be checked.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "results.json"


def _git_commit() -> str | None:
    """Short commit hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy"):
        module = {"scikit-learn": "sklearn"}.get(name, name)
        try:
            versions[name] = __import__(module).__version__
        except Exception:
            versions[name] = None
    return versions


def write_results(metrics: dict, *, data_source: str = "real", notes: str | None = None) -> Path:
    """Write metrics plus provenance to outputs/results.json.

    Parameters
    ----------
    metrics
        Headline numbers only, the ones the README actually quotes. Keep this
        small; a results file with 200 keys stops being read.
    data_source
        "real" or "synthetic". Never omit this. A metric computed on synthetic
        data and reported as though it were real is the worst error this file
        exists to prevent.
    notes
        Anything a reader needs in order not to misread the numbers, for example
        a known limitation of the run.
    """
    if data_source not in {"real", "synthetic"}:
        raise ValueError(f"data_source must be 'real' or 'synthetic', got {data_source!r}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "data_source": data_source,
        "python": platform.python_version(),
        "packages": _package_versions(),
        "metrics": {k: v for k, v in metrics.items() if v is not None},
    }
    if notes:
        payload["notes"] = notes

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return OUTPUT_PATH


def read_results() -> dict:
    """Load results.json. Notebooks and the README build script use this so that
    no number is ever transcribed by hand."""
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PATH} does not exist. Run `python src/run_pipeline.py` first; "
            "the metrics are written by the pipeline, never typed in."
        )
    return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
