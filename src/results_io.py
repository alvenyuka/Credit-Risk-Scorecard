"""Writes every headline number the README quotes to outputs/results.json.

A number typed into a README can drift from the code silently; a number read
from a file the code writes cannot. The rule that goes with this module: the
README and METHODOLOGY quote this file (and outputs/business_impact.json), and
nothing else.

Provenance is recorded alongside the metrics: the commit, whether the data was
real or synthetic, the Python and package versions and the time of the run, so
any figure can be checked rather than taken on trust. The file is strict JSON
(allow_nan=False), so any parser can read it.
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
    """Installed versions of the libraries the results depend on."""
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pyarrow"):
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
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
