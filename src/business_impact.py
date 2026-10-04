"""
Business impact of the scorecard: what a lender gains by using it to set a cut-off.

On the held-out applicants (61,503 in the published run), approve only the highest-scoring share (100%, 90%,
80%, 70%) and measure, for each policy, the default rate among approved loans, the
credit loss those defaults would cause, and how many good borrowers are turned away.

Credit loss = credit amount of each defaulted, approved loan x loss given default.
LGD is an assumption, not an output: 45%, the Basel foundation-IRB supervisory value
for senior unsecured exposures. Home Credit does not state the currency of AMT_CREDIT,
so amounts are in the dataset's own currency units.

Reads the holdout scores that run_pipeline.py saves (outputs/val_scores.parquet) and
the points table (outputs/scorecard_points.json), so it describes the same fit as
outputs/results.json without refitting. Writes outputs/business_impact.json (strict
JSON), figures/lending_policy.png and figures/iv_top15.png (the 15 selected features
with the highest information value).

Run: python src/run_pipeline.py, then python src/business_impact.py   (seconds)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LGD = 0.45
APPROVAL_RATES = (1.0, 0.9, 0.8, 0.7)
ROOT = Path(__file__).resolve().parent.parent


def lending_policy_table(scores, defaulted, exposure, lgd=LGD, approval_rates=APPROVAL_RATES):
    """One row per approval rate: approve the highest scores, report outcomes among them.

    scores     higher = safer (scorecard points)
    defaulted  1 if the applicant defaulted, else 0
    exposure   credit amount of each application
    """
    scores = np.asarray(scores, dtype=float)
    defaulted = np.asarray(defaulted, dtype=int)
    exposure = np.asarray(exposure, dtype=float)
    order = np.argsort(-scores, kind="stable")  # safest first
    n = len(scores)
    n_good = int((defaulted == 0).sum())
    base_loss = float((exposure * defaulted).sum() * lgd)

    rows = []
    for rate in approval_rates:
        k = int(round(rate * n))
        approved = np.zeros(n, dtype=bool)
        approved[order[:k]] = True
        loss = float((exposure[approved] * defaulted[approved]).sum() * lgd)
        rows.append({
            "approval_rate": rate,
            "approved": k,
            "cut_off_score": float(scores[order[k - 1]]) if 0 < k < n else None,
            "default_rate_approved": float(defaulted[approved].mean()) if k else 0.0,
            "credit_loss": loss,
            "loss_avoided_vs_approve_all": base_loss - loss,
            "loss_avoided_pct": (base_loss - loss) / base_loss if base_loss else 0.0,
            "good_borrowers_declined": int(((~approved) & (defaulted == 0)).sum()),
            "good_borrowers_declined_pct": float(((~approved) & (defaulted == 0)).sum() / n_good) if n_good else 0.0,
            "credit_approved": float(exposure[approved].sum()),
        })
    return pd.DataFrame(rows)


def policies_to_records(table: pd.DataFrame) -> list:
    """Rows as plain dicts with missing values as None (JSON null), never NaN."""
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def plot(table, path):
    """Bar chart of credit loss under each approval policy, saved to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = [f"{r:.0%}" for r in table["approval_rate"]]
    bars = ax.bar(x, table["credit_loss"] / 1e9, color="#2b6cb0")
    for bar, pct, dr in zip(bars, table["loss_avoided_pct"], table["default_rate_approved"]):
        label = f"default rate {dr:.1%}" + (f"\nloss -{pct:.0%}" if pct > 0 else "")
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, color="#2d3748")
    ax.set_xlabel("Share of applicants approved (highest scores first)")
    ax.set_ylabel("Credit loss, billions of dataset currency units")
    n = int(table["approved"].max())
    ax.set_title(f"Credit loss on {n:,} held-out applicants by approval policy (LGD 45%)", loc="left", fontsize=11)
    ax.set_ylim(0, table["credit_loss"].max() / 1e9 * 1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_iv(feature_iv: dict, path, top=15):
    """Horizontal bar chart of the `top` selected features by information value, saved to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kept_cols = list(feature_iv)
    iv = sorted(feature_iv.items(), key=lambda t: (t[1], t[0]))[-top:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([c for c, _ in iv], [v for _, v in iv], color="#2b6cb0")
    ax.axvline(0.1, color="#718096", ls="--", lw=1)
    ax.text(0.102, 0.2, "0.10: medium", fontsize=8, color="#4a5568")
    ax.set_xlabel("Information value")
    ax.set_title(f"Top {top} of {len(kept_cols)} selected features by information value", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Build the lending-policy table from the saved holdout scores and write the JSON and both charts."""
    from run_pipeline import POINTS_PATH, load_validation_scores

    df = load_validation_scores()
    table = lending_policy_table(df["score"].to_numpy(), df["TARGET"].to_numpy(), df["AMT_CREDIT"].to_numpy())

    out = ROOT / "outputs" / "business_impact.json"
    out.write_text(json.dumps({
        "assumptions": {
            "lgd": LGD,
            "lgd_basis": "Basel foundation-IRB supervisory LGD for senior unsecured exposures",
            "exposure": "AMT_CREDIT, the credit amount of the application",
            "currency": "not stated in the Home Credit data; figures are in its own currency units",
            "population": f"the {len(df):,}-applicant holdout used for every other metric",
        },
        "policies": policies_to_records(table),
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "figures").mkdir(exist_ok=True)
    plot(table, ROOT / "figures" / "lending_policy.png")
    points = json.loads(POINTS_PATH.read_text(encoding="utf-8"))
    plot_iv({f["feature"]: f["iv"] for f in points}, ROOT / "figures" / "iv_top15.png")
    print(table.to_string(index=False))
    print(f"wrote {out.relative_to(ROOT)}, figures/lending_policy.png and figures/iv_top15.png")


if __name__ == "__main__":
    main()
