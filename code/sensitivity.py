"""Phase 4 of the pipeline: sensitivity analysis of the benchmark results.

Three experiments:

1. Prevalence sweep (principal seed): the corpus is regenerated at anomaly
   prevalences {0.5%, 1%, 2%, 5%, 10%}. Benign sessions are identical across
   prevalences (only which sessions are mutated changes), so detector fits and
   operating thresholds are shared; the sweep isolates how detection quality
   scales with how much misbehavior is actually present.

2. Seed stability: the full pipeline (agents -> telemetry -> injection ->
   detection) is re-run under seeds 101 / 202 / 303. For each seed we record
   the class-level recall matrix and measure rank stability (Spearman) of the
   coverage-gap heatmap versus the principal run, plus per-detector deltas.

3. Threshold perturbation: the headline operating thresholds are scaled by
   ±30%; per-class and overall recall are recomputed post hoc from saved
   evaluation scores. Reports the elasticity of recall to threshold missetting.

Output: results/sensitivity.json (machine-readable, deterministic ordering).

Fully synthetic benchmark; see README disclosure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import DETECTOR_NAMES, run_evaluation  # noqa: E402
from generate_telemetry import generate_dataset  # noqa: E402
from simulation_constants import (  # noqa: E402
    ANOMALY_CLASS_ORDER,
    FP_BUDGETS_PER_AGENT_DAY,
    HEADLINE_FP_BUDGET,
    PREVALENCE_PRINCIPAL,
    PREVALENCE_SWEEP,
    RESULTS_DIR,
    SEEDS_SENSITIVITY,
    THRESHOLD_PERTURBATIONS,
    write_json,
)

BUDGET_KEY = str(HEADLINE_FP_BUDGET)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _recall_vector(result: dict, detector: str) -> list[float]:
    return [
        result["per_class"].get(cid, {}).get(detector, {}).get("recall", {}).get(BUDGET_KEY, float("nan"))
        for cid in ANOMALY_CLASS_ORDER
    ]


def _auc_vector(result: dict, detector: str) -> list[float]:
    return [
        result["per_class"].get(cid, {}).get(detector, {}).get("auc_roc", float("nan"))
        for cid in ANOMALY_CLASS_ORDER
    ]


def perturbed_thresholds(result: dict, factor: float) -> dict[str, float]:
    return {d: result["operating_points"][d][HEADLINE_FP_BUDGET]["threshold"] * factor for d in DETECTOR_NAMES}


def threshold_perturbation_table(result: dict) -> dict:
    """Recall (overall + per class) at thresholds scaled by each perturbation factor."""
    feats_eval, scores_eval, classes_eval = result["_frames"]["scores"]
    benign = classes_eval == "benign"
    out = {}
    for factor in THRESHOLD_PERTURBATIONS:
        thr = perturbed_thresholds(result, factor)
        entry = {
            "factor": factor,
            "thresholds": thr,
            "realized_fp_per_agent_day": {},
            "overall_recall": {},
            "per_class_recall": {},
        }
        for d in DETECTOR_NAMES:
            s = scores_eval[d]
            flagged = s >= thr[d]
            n_pos = int((~benign).sum())
            entry["overall_recall"][d] = float((flagged & ~benign).sum() / max(1, n_pos))
            fp = int((flagged & benign).sum())
            entry["realized_fp_per_agent_day"][d] = float(
                fp / max(1, int(benign.sum())) * result["meta"]["sessions_per_agent_day"]
            )
            entry["per_class_recall"][d] = {
                cid: float((flagged & (classes_eval == cid)).sum() / max(1, int((classes_eval == cid).sum())))
                for cid in ANOMALY_CLASS_ORDER
                if (classes_eval == cid).any()
            }
        out[f"{factor:.2f}"] = entry
    return out


def run_prevalence_sweep(seed_agents: int = 42, verbose: bool = True, prevalences=None) -> dict:
    prevalences = list(PREVALENCE_SWEEP) if prevalences is None else list(prevalences)
    rows = []
    for p in prevalences:
        if verbose:
            print(f"  prevalence {p:.1%} ...")
        ds = generate_dataset(prevalence=p, seed_agents=seed_agents, seed_injection=seed_agents, verbose=False)
        res = run_evaluation(
            ds["sessions"], ds["labels"], ds["agents"],
            seed_agents=seed_agents, include_lead_time=False,
            threshold_factor=1.0, verbose=False,
        )
        row = {
            "prevalence": p,
            "n_anomalous": res["meta"]["n_anomalous"],
            "overall": {d: {"auc_roc": res["overall"][d]["auc_roc"],
                            "recall": res["overall"][d]["at_budget"][BUDGET_KEY]["recall"],
                            "precision": res["overall"][d]["at_budget"][BUDGET_KEY]["precision"],
                            "fp_per_agent_day": res["operating_points"][d][HEADLINE_FP_BUDGET]["fp_per_agent_day_eval"]}
                        for d in DETECTOR_NAMES},
            "per_class_recall": {d: {cid: res["per_class"].get(cid, {}).get(d, {}).get("recall", {}).get(BUDGET_KEY)
                                     for cid in ANOMALY_CLASS_ORDER} for d in DETECTOR_NAMES},
            "per_class_auc": {d: {cid: res["per_class"].get(cid, {}).get(d, {}).get("auc_roc")
                                  for cid in ANOMALY_CLASS_ORDER} for d in DETECTOR_NAMES},
        }
        rows.append(row)
        if verbose:
            best = max(DETECTOR_NAMES, key=lambda d: row["overall"][d]["auc_roc"])
            print(f"    best overall AUC: {best} ({row['overall'][best]['auc_roc']:.3f})")
    return {"prevalences": list(prevalences), "rows": rows}


def run_seed_runs(seeds=SEEDS_SENSITIVITY, verbose: bool = True) -> dict[int, dict]:
    """Regenerate + evaluate the corpus once per seed (reused by both analyses)."""
    results = {}
    for seed in seeds:
        if verbose:
            print(f"  seed {seed} ...")
        ds = generate_dataset(seed_agents=seed, seed_injection=seed, verbose=False)
        results[seed] = run_evaluation(
            ds["sessions"], ds["labels"], ds["agents"], seed_agents=seed,
            include_lead_time=True, verbose=False,
        )
    return results


def run_seed_stability(principal: dict, seed_results: dict[int, dict]) -> dict:
    out = {"seeds": list(seed_results), "runs": [], "rank_stability": {}}
    principal_recall = {d: np.array(_recall_vector(principal, d)) for d in DETECTOR_NAMES}
    principal_auc = {d: np.array(_auc_vector(principal, d)) for d in DETECTOR_NAMES}

    recall_by_seed = {d: [] for d in DETECTOR_NAMES}
    auc_by_seed = {d: [] for d in DETECTOR_NAMES}
    for seed, res in seed_results.items():
        run = {
            "seed": seed,
            "overall": {d: {"auc_roc": res["overall"][d]["auc_roc"],
                            "recall": res["overall"][d]["at_budget"][BUDGET_KEY]["recall"],
                            "fp_per_agent_day": res["operating_points"][d][HEADLINE_FP_BUDGET]["fp_per_agent_day_eval"]}
                        for d in DETECTOR_NAMES},
            "per_class_recall": {d: _recall_vector(res, d) for d in DETECTOR_NAMES},
            "per_class_auc": {d: _auc_vector(res, d) for d in DETECTOR_NAMES},
            "lead_time": {d: {cid: res["lead_time"].get(d, {}).get("by_class", {}).get(cid, {})
                              for cid in ANOMALY_CLASS_ORDER} for d in DETECTOR_NAMES},
        }
        out["runs"].append(run)
        for d in DETECTOR_NAMES:
            recall_by_seed[d].append(np.array(run["per_class_recall"][d], dtype=float))
            auc_by_seed[d].append(np.array(run["per_class_auc"][d], dtype=float))

    # Rank stability of the coverage-gap heatmap: Spearman(principal, seed)
    stab = {}
    for d in DETECTOR_NAMES:
        stab[d] = {
            "recall_spearman": [spearman(principal_recall[d], v) for v in recall_by_seed[d]],
            "auc_spearman": [spearman(principal_auc[d], v) for v in auc_by_seed[d]],
            "recall_max_abs_delta": [float(np.nanmax(np.abs(principal_recall[d] - v))) for v in recall_by_seed[d]],
            "auc_max_abs_delta": [float(np.nanmax(np.abs(principal_auc[d] - v))) for v in auc_by_seed[d]],
        }
    out["rank_stability"] = stab
    return out


def run_threshold_perturbation(principal: dict, seed_results: dict[int, dict]) -> dict:
    """Perturbation table for the principal run + each seed run (reuses seed_results)."""
    tables = {"principal": threshold_perturbation_table(principal)}
    deltas = {}
    for seed, res in seed_results.items():
        tables[f"seed_{seed}"] = threshold_perturbation_table(res)
    # Summarize the elasticity: overall recall at 0.7x / 1.3x vs 1.0x, per detector
    for name, tab in tables.items():
        deltas[name] = {}
        for d in DETECTOR_NAMES:
            r10 = tab["1.00"]["overall_recall"][d]
            r07 = tab["0.70"]["overall_recall"][d]
            r13 = tab["1.30"]["overall_recall"][d]
            deltas[name][d] = {"recall_0.7x": r07, "recall_1.0x": r10, "recall_1.3x": r13,
                               "delta_0.7x": r07 - r10, "delta_1.3x": r13 - r10}
    return {"tables": tables, "summary": deltas}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensitivity analysis for the anomaly benchmark.")
    ap.add_argument("--fast", action="store_true", help="reduced sweep for development smoke testing")
    args = ap.parse_args()

    prevalences = (0.02,) if args.fast else PREVALENCE_SWEEP
    seeds = (101,) if args.fast else SEEDS_SENSITIVITY

    out = {
        "meta": {
            "principal_seed": 42,
            "seeds": list(seeds),
            "prevalences": list(prevalences),
            "threshold_perturbations": list(THRESHOLD_PERTURBATIONS),
            "headline_fp_budget": HEADLINE_FP_BUDGET,
            "note": "Benign sessions are identical across prevalence levels (only mutation coverage changes), so operating thresholds are shared across the sweep. Seed runs are generated once and reused by both the stability and threshold-perturbation analyses.",
        }
    }

    print("Principal run (seed 42) for reference ...")
    ds = generate_dataset(verbose=False)
    principal = run_evaluation(ds["sessions"], ds["labels"], ds["agents"], include_lead_time=False, verbose=False)

    print("Prevalence sweep ...")
    out["prevalence_sweep"] = run_prevalence_sweep(prevalences=prevalences, verbose=True)

    print("Seed runs ...")
    seed_results = run_seed_runs(seeds=seeds)

    print("Seed stability ...")
    out["seed_stability"] = run_seed_stability(principal, seed_results)

    print("Threshold perturbation ...")
    out["threshold_perturbation"] = run_threshold_perturbation(principal, seed_results)

    write_json(RESULTS_DIR / "sensitivity.json", out)
    print(f"Wrote {RESULTS_DIR / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
