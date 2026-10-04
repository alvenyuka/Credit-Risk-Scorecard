"""
Charts for the README, drawn from the holdout scores that run_pipeline.py saves
(outputs/val_scores.parquet), so they describe exactly the model whose metrics are
in outputs/results.json and need no second fit.

Two views a credit committee reads first: how far apart the score distributions of
repaid and defaulted applicants sit, and how the default rate falls as the score
rises. Writes figures/score_distribution.png and figures/default_rate_by_band.png.
The plotting functions return the figure, so the notebook can show them inline.

Run: python src/run_pipeline.py, then python src/make_figures.py   (seconds)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_scratch import default_rate_by_score_decile

OUT = Path(__file__).resolve().parent.parent / "figures"
REPAID, DEFAULTED, INK = "#2b6cb0", "#c0392b", "#2d3748"


def _finish(fig, path):
    """Tidy the layout and, when `path` is given, save the figure and close it."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def score_distribution(scores, y, path=None):
    """Overlaid histograms of the scores of repayers and defaulters (saved when `path` is given)."""
    import matplotlib.pyplot as plt

    scores, y = pd.Series(np.asarray(scores, dtype=float)), np.asarray(y)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(np.floor(scores.min() / 10) * 10, scores.max() + 10, 10)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.55, color=REPAID,
            label=f"Repaid (mean {scores[y == 0].mean():.1f})")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.55, color=DEFAULTED,
            label=f"Defaulted (mean {scores[y == 1].mean():.1f})")
    ax.set_xlabel("Scorecard points (600 = 20:1 good:bad odds)")
    ax.set_ylabel("Share of applicants")
    ax.set_yticks([])
    ax.set_title(f"Score distributions, {len(scores):,} held-out applicants", color=INK, loc="left")
    ax.legend(frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return _finish(fig, path)


def default_rate_by_band(scores, y, path=None):
    """Default rate in each score decile against the average (saved when `path` is given)."""
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float)
    table = default_rate_by_score_decile(scores, y)
    rate = table["default_rate"].to_numpy() * 100
    labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(table["score_min"], table["score_max"])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(rate)), rate, color=DEFAULTED, alpha=0.8)
    ax.axhline(y.mean() * 100, color=INK, lw=1, ls="--")
    ax.text(len(rate) - 0.5, y.mean() * 100, f" average {y.mean() * 100:.1f}%", va="bottom",
            ha="right", color=INK, fontsize=9)
    for b, v in zip(bars, rate):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Score band (deciles of held-out applicants)")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate falls as the score rises", color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return _finish(fig, path)


def main():
    """Draw both README charts from the saved holdout scores."""
    import matplotlib

    matplotlib.use("Agg")
    from run_pipeline import load_validation_scores

    df = load_validation_scores()
    OUT.mkdir(exist_ok=True)
    score_distribution(df["score"], df["TARGET"], OUT / "score_distribution.png")
    default_rate_by_band(df["score"], df["TARGET"], OUT / "default_rate_by_band.png")
    table = default_rate_by_score_decile(df["score"], df["TARGET"])
    print(f"default rate, lowest score decile: {table['default_rate'].iloc[0]:.1%}  "
          f"highest: {table['default_rate'].iloc[-1]:.1%}")
    print(f"wrote {OUT / 'score_distribution.png'} and {OUT / 'default_rate_by_band.png'}")


if __name__ == "__main__":
    main()
