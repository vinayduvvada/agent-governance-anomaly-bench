"""Phase 5 of the pipeline: paper figures from results/*.json.

Figures (figures/):
  fig1_coverage_gap_heatmap.png   class x detector recall at the headline budget (+ AUC annotation)
  fig2_roc_curves.png             per-class ROC overview for the best detector per class
  fig3_lead_time.png              detection lead-time distributions (violin/box) per detector
  fig4_prevalence_sensitivity.png recall/AUC vs anomaly prevalence per detector
  fig5_seed_stability.png         per-seed rank stability + AUC deltas

All figures are regenerated deterministically from saved results; matplotlib only.
Fully synthetic benchmark; see README disclosure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation_constants import ANOMALY_CLASS_ORDER, FIGURES_DIR, HEADLINE_FP_BUDGET, RESULTS_DIR  # noqa: E402

DETECTOR_NAMES = ("D1", "D2", "D3", "D4")
DETECTOR_LABELS = {
    "D1": "D1 static rules",
    "D2": "D2 per-agent baseline",
    "D3": "D3 isolation forest",
    "D4": "D4 Markov sequence",
}
CLASS_LABELS = {
    "scope_creep": "Scope creep",
    "privilege_escalation_attempts": "Privilege-escalation attempts",
    "runaway_loop": "Runaway loops",
    "data_exfiltration": "Data exfiltration",
    "prompt_injection_compromise": "Prompt-injection compromise",
    "cost_anomaly": "Cost anomalies",
    "silent_failure_masking": "Silent failure masking",
    "approval_gate_circumvention": "Approval-gate circumvention",
    "cross_agent_collusion": "Cross-agent collusion",
}


def load(name: str) -> dict:
    with open(RESULTS_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def _budget_key() -> str:
    return str(HEADLINE_FP_BUDGET)


# ---------------------------------------------------------------------------
# Figure 1: coverage-gap heatmap
# ---------------------------------------------------------------------------


def fig_coverage_gap(cg: dict) -> None:
    classes = cg["classes"]
    dets = cg["detectors"]
    recall = np.array([[cg["recall_at_headline"][d][i] for d in dets] for i in range(len(classes))], dtype=float)
    auc = np.array([[cg["auc_roc"][d][i] for d in dets] for i in range(len(classes))], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(recall, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(dets)), [DETECTOR_LABELS[d] for d in dets], rotation=15, ha="right")
    ax.set_yticks(range(len(classes)), [CLASS_LABELS[c] for c in classes])
    for i in range(len(classes)):
        for j in range(len(dets)):
            r, a = recall[i, j], auc[i, j]
            txt = f"{r:.2f}\n(AUC {a:.2f})" if np.isfinite(r) and np.isfinite(a) else "n/a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="black" if r < 0.65 else "white")
    ax.set_title(f"Coverage gaps: recall per misbehavior class at {HEADLINE_FP_BUDGET} FP/agent-day\n"
                 "(annotation: ROC-AUC of the same detector on the same class)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="recall at operating point")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_coverage_gap_heatmap.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: ROC curves
# ---------------------------------------------------------------------------


def fig_roc(ev: dict, scores: "pd.DataFrame") -> None:
    from sklearn.metrics import roc_curve

    top_classes = [
        "data_exfiltration", "runaway_loop", "privilege_escalation_attempts",
        "approval_gate_circumvention", "prompt_injection_compromise", "scope_creep",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharex=True, sharey=True)
    for ax, cid in zip(axes.ravel(), top_classes):
        y = (scores["label"] == cid).astype(int).to_numpy()
        for d in DETECTOR_NAMES:
            if y.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y, scores[d].to_numpy())
            ax.plot(fpr, tpr, lw=1.4, label=f"{d} (AUC {ev['per_class'][cid][d]['auc_roc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_title(CLASS_LABELS[cid], fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("false-positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("true-positive rate")
    fig.suptitle("ROC curves by misbehavior class (one-vs-benign, evaluation split)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "fig2_roc_curves.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: lead time
# ---------------------------------------------------------------------------


def fig_lead_time(lt: "pd.DataFrame") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    data = []
    labels = []
    for d in DETECTOR_NAMES:
        sub = lt[(lt["detector"] == d) & (lt["pre_violation"])]["lead_time_s"].dropna().to_numpy()
        data.append(sub)
        labels.append(DETECTOR_LABELS[d])
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#c8d7ea", "#b7e0c2", "#ecd9a8", "#e5b8b7"]):
        patch.set_facecolor(color)
    ax.set_ylabel("lead time before violation (s)")
    ax.set_title("Detection lead time (pre-violation detections only)", fontsize=9.5)
    ax.axhline(0, color="k", lw=0.7)

    ax2 = axes[1]
    classes = [c for c in ANOMALY_CLASS_ORDER if c in set(lt["anomaly_class"])]
    rates = {d: [] for d in DETECTOR_NAMES}
    for cid in classes:
        for d in DETECTOR_NAMES:
            sub = lt[(lt["detector"] == d) & (lt["anomaly_class"] == cid)]
            rates[d].append(float(sub["pre_violation"].mean()) if len(sub) else np.nan)
    x = np.arange(len(classes))
    w = 0.19
    for k, d in enumerate(DETECTOR_NAMES):
        ax2.bar(x + (k - 1.5) * w, rates[d], width=w, label=d)
    ax2.set_xticks(x, [CLASS_LABELS[c][:18] for c in classes], rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel("pre-violation rate")
    ax2.set_title("Fraction of injections flagged before completion", fontsize=9.5)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_lead_time.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: prevalence sensitivity
# ---------------------------------------------------------------------------


def fig_prevalence(sens: dict) -> None:
    sweep = sens["prevalence_sweep"]
    prevs = [row["prevalence"] for row in sweep["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    for d in DETECTOR_NAMES:
        aucs = [row["overall"][d]["auc_roc"] for row in sweep["rows"]]
        ax.plot(np.array(prevs) * 100, aucs, marker="o", ms=3.5, lw=1.4, label=d)
    ax.set_xscale("log")
    ax.set_xlabel("anomaly prevalence (%)")
    ax.set_ylabel("overall ROC-AUC")
    ax.set_title("Detection quality vs prevalence", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for d in DETECTOR_NAMES:
        rec = [row["overall"][d]["recall"] for row in sweep["rows"]]
        ax.plot(np.array(prevs) * 100, rec, marker="s", ms=3.5, lw=1.4, label=d)
    ax.set_xscale("log")
    ax.set_xlabel("anomaly prevalence (%)")
    ax.set_ylabel(f"recall @ {HEADLINE_FP_BUDGET} FP/agent-day")
    ax.set_title("Recall at fixed alert budget vs prevalence", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_prevalence_sensitivity.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: seed stability
# ---------------------------------------------------------------------------


def fig_seed_stability(sens: dict) -> None:
    stab = sens["seed_stability"]
    seeds = stab["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    for d in DETECTOR_NAMES:
        drops = stab["rank_stability"][d]["recall_spearman"]
        ax.bar(np.arange(len(seeds)) + (DETECTOR_NAMES.index(d) - 1.5) * 0.19, drops, width=0.19, label=d)
    ax.set_xticks(np.arange(len(seeds)), [str(s) for s in seeds])
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("seed")
    ax.set_ylabel("Spearman(recall ranking vs principal)")
    ax.set_title("Coverage-gap ranking stability across seeds", fontsize=9.5)
    ax.legend(fontsize=7)

    ax = axes[1]
    for d in DETECTOR_NAMES:
        deltas = stab["rank_stability"][d]["auc_max_abs_delta"]
        ax.bar(np.arange(len(seeds)) + (DETECTOR_NAMES.index(d) - 1.5) * 0.19, deltas, width=0.19, label=d)
    ax.set_xticks(np.arange(len(seeds)), [str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("max |AUC delta| vs principal (per class)")
    ax.set_title("Class-level AUC drift across seeds", fontsize=9.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_seed_stability.png", dpi=200)
    plt.close(fig)


def main() -> None:
    import pandas as pd

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ev = load("evaluation.json")
    cg = load("coverage_gap.json")
    sens = load("sensitivity.json")
    scores = pd.read_csv(RESULTS_DIR / "scores.csv", dtype={"label": str})
    lt = pd.read_csv(RESULTS_DIR / "lead_times.csv", dtype={"anomaly_class": str, "detector": str})

    fig_coverage_gap(cg)
    print("  fig1_coverage_gap_heatmap.png")
    fig_roc(ev, scores)
    print("  fig2_roc_curves.png")
    fig_lead_time(lt)
    print("  fig3_lead_time.png")
    fig_prevalence(sens)
    print("  fig4_prevalence_sensitivity.png")
    fig_seed_stability(sens)
    print("  fig5_seed_stability.png")
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
