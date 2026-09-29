"""
Charts for the README, drawn from the same model that writes outputs/results.json.

Two views a credit committee reads first: how far apart the score distributions of
repaid and defaulted applicants sit, and how the default rate falls as the score
rises. Writes figures/score_distribution.png and figures/default_rate_by_band.png.

Run: python src/make_figures.py   (needs the 8 Home Credit CSVs in data/, ~7 minutes)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from week8_full import build_validation_scores  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
REPAID, DEFAULTED, INK = "#2b6cb0", "#c0392b", "#2d3748"


def score_distribution(scores, y, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(np.floor(scores.min() / 10) * 10, scores.max() + 10, 10)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.55, color=REPAID,
            label=f"Repaid (mean {scores[y == 0].mean():.1f})")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.55, color=DEFAULTED,
            label=f"Defaulted (mean {scores[y == 1].mean():.1f})")
    ax.set_xlabel("Scorecard points")
    ax.set_ylabel("Share of applicants")
    ax.set_yticks([])
    ax.set_title(f"Score distributions, {len(scores):,} held-out applicants", color=INK, loc="left")
    ax.legend(frameon=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def default_rate_by_band(scores, y, path):
    bands = pd.qcut(scores, 10)
    table = pd.DataFrame({"band": bands, "default": y.values}).groupby("band", observed=True)["default"]
    rate = table.mean() * 100
    labels = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in rate.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(rate)), rate.values, color=DEFAULTED, alpha=0.8)
    ax.axhline(y.mean() * 100, color=INK, lw=1, ls="--")
    ax.text(len(rate) - 0.5, y.mean() * 100, f" average {y.mean() * 100:.1f}%", va="bottom",
            ha="right", color=INK, fontsize=9)
    for b, v in zip(bars, rate.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Score band (deciles of held-out applicants)")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate falls as the score rises", color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return rate


def main():
    _, _, _, y_val, scores = build_validation_scores()
    OUT.mkdir(exist_ok=True)
    score_distribution(scores, y_val, OUT / "score_distribution.png")
    rate = default_rate_by_band(scores, y_val, OUT / "default_rate_by_band.png")
    print(f"default rate, lowest score decile: {rate.iloc[0]:.1f}%  highest: {rate.iloc[-1]:.1f}%")
    print(f"wrote {OUT / 'score_distribution.png'} and {OUT / 'default_rate_by_band.png'}")


if __name__ == "__main__":
    main()
