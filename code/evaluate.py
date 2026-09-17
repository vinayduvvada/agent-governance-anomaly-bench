"""Phase 3 of the pipeline: evaluate detectors D1-D4 against the ground truth.

Metrics
-------
- per-class ROC-AUC and PR-AUC (class vs benign evaluation sessions)
- operating points: detector thresholds calibrated on benign *calibration*
  sessions to a target alert budget (false positives per agent-day), then
  realized on held-out benign evaluation sessions
- per-class recall/precision at the operating points
- detection lead time: for every anomalous session, the session is rescored on
  growing prefixes with the *same* feature function and operating threshold;
  lead time = violation_ts (from the injection log) - first-crossing ts.
  Positive = flagged before the violating action completed.
- coverage-gap matrix: class x detector recall at the headline budget + AUCs.

Outputs (results/):  evaluation.json, coverage_gap.json, operating_points.json,
scores.csv, lead_times.csv

Fully synthetic benchmark; thresholds and budgets are illustrative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as F  # noqa: E402
from detectors import (  # noqa: E402
    D1StaticRules,
    D2PerAgentBaseline,
    D3IsolationForest,
    D4Markov,
    threshold_for_budget,
)
from features import FEATURE_NAMES, TOOL_IDX  # noqa: E402
from generate_agents import build_agents  # noqa: E402
from generate_telemetry import session_layout_with_rates  # noqa: E402
from simulation_constants import (  # noqa: E402
    AGENTS_METADATA_PATH,
    ANOMALY_CLASS_ORDER,
    CALIB_FIT_FRACTION,
    D3_PREFIX_SAMPLES,
    FP_BUDGETS_PER_AGENT_DAY,
    HEADLINE_FP_BUDGET,
    LABELS_PATH,
    N_DAYS,
    RESULTS_DIR,
    SEED_AGENTS,
    SESSIONS_PATH,
    spawn_rng,
    write_json,
)

DETECTOR_NAMES = ("D1", "D2", "D3", "D4")


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------


def session_id_map(layout: dict, n_agents: int, n_days: int) -> dict[tuple, str]:
    """Replicate the generator's session-id counter: agent-major, day, session."""
    mapping = {}
    counter = 0
    for a in range(n_agents):
        for d in range(n_days):
            for s in range(layout["counts"][a][d]):
                counter += 1
                mapping[(a, d, s)] = f"S{counter:06d}"
    return mapping


def _aucs(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, scores)), float(average_precision_score(y_true, scores))


def _precision_recall(scores, y_true, threshold):
    flagged = scores >= threshold
    tp = int((flagged & (y_true == 1)).sum())
    fp_benign = 0  # computed by caller context; here only TP/flagged counts
    n_flagged = int(flagged.sum())
    n_pos = int(y_true.sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / n_flagged if n_flagged else float("nan")
    return recall, precision, tp, n_flagged


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    sessions: pd.DataFrame,
    labels: pd.DataFrame,
    agents: list[dict],
    seed_agents: int = SEED_AGENTS,
    n_days: int = N_DAYS,
    include_lead_time: bool = True,
    threshold_factor: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Run the full detector evaluation for one synthetic corpus."""
    n_agents = len(agents)
    arrays, scope_vocab, session_agent = F.build_session_arrays(sessions)
    scope_idx_map = {s: i for i, s in enumerate(scope_vocab)}
    scope_families = F.build_scope_families(scope_vocab)
    ctx = F.build_agent_contexts(agents, scope_idx_map)

    layout = session_layout_with_rates(seed_agents, agents, n_days)
    id_map = session_id_map(layout, n_agents, n_days)
    calib_ids = [id_map[tuple((a,) + slot)] for a in range(n_agents) for slot in layout["calib_slots"][a]]
    eval_ids = [id_map[tuple((a,) + slot)] for a in range(n_agents) for slot in layout["eval_slots"][a]]
    calib_ids = [sid for sid in calib_ids if sid in arrays]
    eval_ids = [sid for sid in eval_ids if sid in arrays]

    labels_by_session = labels.set_index("session_id")
    assert not set(calib_ids) & set(labels_by_session.index), "anomalies found in calibration split"

    n_total_sessions = len(arrays)
    agent_days = n_agents * n_days
    sessions_per_agent_day = n_total_sessions / agent_days

    # --- fit / threshold-calibration split ----------------------------------
    # Detectors are fit on 75% of the calibration sessions and thresholds are
    # selected on the remaining 25% (out-of-sample scores under the very same
    # models). The evaluation split is then scored by those models, so the
    # operating-point comparison is in-sample nowhere.
    perm = spawn_rng(seed_agents, "calibsplit").permutation(len(calib_ids))
    n_fit = int(round(len(calib_ids) * CALIB_FIT_FRACTION))
    fit_ids = [calib_ids[i] for i in sorted(perm[:n_fit])]
    thr_ids = [calib_ids[i] for i in sorted(perm[n_fit:])]

    agent_task_class = {a["agent_id"]: a["task_class"] for a in agents}
    agent_permitted = {a["agent_id"]: set(TOOL_IDX[t] for t in a["tool_permissions"]) for a in agents}
    d4 = D4Markov().fit(arrays, session_agent, fit_ids, agent_task_class, agent_permitted)

    feats_all = F.extract_features_frame(arrays, session_agent, ctx, scope_families, chain_model=d4)
    feats_fit = feats_all.loc[fit_ids]
    feats_thr = feats_all.loc[thr_ids]
    feats_eval = feats_all.loc[eval_ids]

    det_d2 = D2PerAgentBaseline().fit(feats_fit)
    d4.calibrate(feats_fit)

    calib_cap = min(len(fit_ids), 20000)
    d3_sessions = list(spawn_rng(seed_agents, "d3cap").permutation(len(fit_ids))[:calib_cap])
    d3_train_sessions = [fit_ids[i] for i in d3_sessions]
    prefix_rows = F.extract_prefix_rows(
        arrays, session_agent, ctx, scope_families, d4, d3_train_sessions, D3_PREFIX_SAMPLES, seed_agents
    )
    det_d3 = D3IsolationForest().fit(prefix_rows)

    det_d1 = D1StaticRules()
    detectors = {"D1": det_d1, "D2": det_d2, "D3": det_d3, "D4": d4}

    # --- scores ------------------------------------------------------------
    scores_calib = {name: det.score(feats_thr) for name, det in detectors.items()}  # out-of-sample
    scores_eval = {name: det.score(feats_eval) for name, det in detectors.items()}

    classes_eval = np.array(
        [labels_by_session.loc[sid, "anomaly_class"] if sid in labels_by_session.index else "benign" for sid in eval_ids]
    )
    y_any = (classes_eval != "benign").astype(int)
    benign_mask = y_any == 0
    n_benign_eval = int(benign_mask.sum())

    # --- operating points ----------------------------------------------------
    operating_points = {}
    for name in DETECTOR_NAMES:
        operating_points[name] = {}
        for budget in FP_BUDGETS_PER_AGENT_DAY:
            thr = threshold_for_budget(scores_calib[name], len(thr_ids) / sessions_per_agent_day, budget) * threshold_factor
            fp_eval = int((scores_eval[name][benign_mask] >= thr).sum())
            fp_per_ad = fp_eval / max(1, n_benign_eval) * sessions_per_agent_day
            operating_points[name][budget] = {
                "threshold": float(thr),
                "fp_per_agent_day_eval": float(fp_per_ad),
                "n_flagged_benign_eval": fp_eval,
            }

    headline_thr = {name: operating_points[name][HEADLINE_FP_BUDGET]["threshold"] for name in DETECTOR_NAMES}

    # --- overall + per-class metrics ----------------------------------------
    overall = {}
    for name in DETECTOR_NAMES:
        auc_roc, auc_pr = _aucs(scores_eval[name], y_any)
        entry = {"auc_roc": auc_roc, "auc_pr": auc_pr, "at_budget": {}}
        for budget in FP_BUDGETS_PER_AGENT_DAY:
            thr = operating_points[name][budget]["threshold"]
            recall, precision, tp, n_flagged = _precision_recall(scores_eval[name], y_any, thr)
            entry["at_budget"][str(budget)] = {
                "recall": recall, "precision": precision, "n_true_positives": tp, "n_flagged": n_flagged,
            }
        overall[name] = entry

    per_class = {}
    for cid in ANOMALY_CLASS_ORDER:
        pos = classes_eval == cid
        if not pos.any():
            continue
        per_class[cid] = {}
        y_cls = pos.astype(int)
        for name in DETECTOR_NAMES:
            auc_roc, auc_pr = _aucs(scores_eval[name], y_cls)
            entry = {"auc_roc": auc_roc, "auc_pr": auc_pr, "n_sessions": int(pos.sum()), "recall": {}, "precision": {}}
            for budget in FP_BUDGETS_PER_AGENT_DAY:
                thr = operating_points[name][budget]["threshold"]
                recall, precision, tp, n_flagged = _precision_recall(scores_eval[name], y_cls, thr)
                # precision against false alarms on benign sessions only (one-vs-benign convention)
                fp_benign = int((scores_eval[name][benign_mask] >= thr).sum())
                precision_vs_benign = tp / max(1, tp + fp_benign)
                entry["recall"][str(budget)] = recall
                entry["precision"][str(budget)] = precision_vs_benign
            per_class[cid][name] = entry

    # --- lead time -----------------------------------------------------------
    lead_times_df = pd.DataFrame()
    lead_time_summary = {}
    if include_lead_time and len(labels):
        anom_ids = [sid for sid in eval_ids if sid in labels_by_session.index]
        rows = []
        row_meta = []
        for sid in anom_ids:
            arr = arrays[sid]
            n = len(arr["ts"])
            for k in range(1, n + 1):
                aid = session_agent[sid]
                rows.append(F.extract_features(arr, ctx[aid], scope_families, k, chain_model=d4))
                row_meta.append((sid, k))
        prefix_feats = pd.DataFrame(rows)
        meta_df = pd.DataFrame(row_meta, columns=["session_id", "k"])
        prefix_feats["agent_id"] = [session_agent[sid] for sid, _ in row_meta]
        prefix_feats["task_class"] = [ctx[session_agent[sid]]["task_class"] for sid, _ in row_meta]

        prefix_scores = {name: det.score(prefix_feats) for name, det in detectors.items()}
        meta_df["ts"] = [arrays[sid]["ts"][k - 1] for sid, k in row_meta]

        recs = []
        for name in DETECTOR_NAMES:
            sc = pd.Series(prefix_scores[name], index=meta_df.index)
            thr = headline_thr[name]
            for sid in anom_ids:
                sub = meta_df[meta_df["session_id"] == sid]
                sub_scores = sc.loc[sub.index]
                hit = sub_scores[sub_scores >= thr]
                vts = float(labels_by_session.loc[sid, "violation_ts"])
                if len(hit):
                    first_idx = hit.index[0]
                    detection_ts = float(sub.loc[first_idx, "ts"])
                    lead = vts - detection_ts
                    recs.append({"session_id": sid, "anomaly_class": labels_by_session.loc[sid, "anomaly_class"],
                                 "detector": name, "detected": True, "detection_ts": detection_ts,
                                 "violation_ts": vts, "lead_time_s": lead, "pre_violation": bool(lead > 0)})
                else:
                    recs.append({"session_id": sid, "anomaly_class": labels_by_session.loc[sid, "anomaly_class"],
                                 "detector": name, "detected": False, "detection_ts": None,
                                 "violation_ts": vts, "lead_time_s": None, "pre_violation": False})
        lead_times_df = pd.DataFrame(recs)

        for name in DETECTOR_NAMES:
            sub = lead_times_df[lead_times_df["detector"] == name]
            lt = sub.dropna(subset=["lead_time_s"])
            lead_time_summary[name] = {
                "n_anomalies": int(len(sub)),
                "detected": int(sub["detected"].sum()),
                "detection_rate": float(sub["detected"].mean()),
                "pre_violation": int(sub["pre_violation"].sum()),
                "pre_violation_rate": float(sub["pre_violation"].mean()),
                "median_lead_among_detected_s": float(lt["lead_time_s"].median()) if len(lt) else None,
                "median_lead_pre_violation_s": float(lt[lt["pre_violation"]]["lead_time_s"].median()) if len(lt) and lt["pre_violation"].any() else None,
                "by_class": {},
            }
            for cid in ANOMALY_CLASS_ORDER:
                sc_sub = sub[sub["anomaly_class"] == cid]
                if not len(sc_sub):
                    continue
                lt_c = sc_sub.dropna(subset=["lead_time_s"])
                lead_time_summary[name]["by_class"][cid] = {
                    "n": int(len(sc_sub)),
                    "detected": int(sc_sub["detected"].sum()),
                    "detection_rate": float(sc_sub["detected"].mean()),
                    "pre_violation_rate": float(sc_sub["pre_violation"].mean()),
                    "median_lead_pre_violation_s": float(lt_c[lt_c["pre_violation"]]["lead_time_s"].median())
                    if len(lt_c) and lt_c["pre_violation"].any() else None,
                }

    # --- coverage-gap matrices ----------------------------------------------
    coverage = {
        "headline_budget": HEADLINE_FP_BUDGET,
        "classes": list(ANOMALY_CLASS_ORDER),
        "detectors": list(DETECTOR_NAMES),
        "recall_at_headline": {
            name: [per_class.get(cid, {}).get(name, {}).get("recall", {}).get(str(HEADLINE_FP_BUDGET), float("nan"))
                   for cid in ANOMALY_CLASS_ORDER] for name in DETECTOR_NAMES},
        "auc_roc": {name: [per_class.get(cid, {}).get(name, {}).get("auc_roc", float("nan")) for cid in ANOMALY_CLASS_ORDER]
                    for name in DETECTOR_NAMES},
        "auc_pr": {name: [per_class.get(cid, {}).get(name, {}).get("auc_pr", float("nan")) for cid in ANOMALY_CLASS_ORDER]
                   for name in DETECTOR_NAMES},
    }

    result = {
        "meta": {
            "n_agents": n_agents,
            "n_days": n_days,
            "n_sessions": n_total_sessions,
            "n_events": int(len(sessions)),
            "n_calibration_sessions": len(calib_ids),
            "n_eval_sessions": len(eval_ids),
            "n_eval_benign": n_benign_eval,
            "n_anomalous": int(y_any.sum()),
            "seed_agents": seed_agents,
            "sessions_per_agent_day": sessions_per_agent_day,
            "fp_budgets_per_agent_day": list(FP_BUDGETS_PER_AGENT_DAY),
            "headline_fp_budget": HEADLINE_FP_BUDGET,
            "threshold_factor": threshold_factor,
            "threshold_calibration": "detectors fit on 75% of the calibration split; thresholds selected on the held-out 25% (out-of-sample), eval split scored by the same models",
        },
        "operating_points": operating_points,
        "overall": overall,
        "per_class": per_class,
        "lead_time": lead_time_summary,
        "coverage_gap": coverage,
    }
    result["_frames"] = {"scores": (feats_eval, scores_eval, classes_eval), "lead_times": lead_times_df,
                         "detectors": detectors}
    if verbose:
        print(f"  eval sessions: {len(eval_ids):,} (benign {n_benign_eval:,}, anomalous {int(y_any.sum()):,})")
        for name in DETECTOR_NAMES:
            print(f"  {name}: overall AUC-ROC {overall[name]['auc_roc']:.3f}  PR-AUC {overall[name]['auc_pr']:.3f}  "
                  f"FP/agent-day@0.1: {operating_points[name][HEADLINE_FP_BUDGET]['fp_per_agent_day_eval']:.3f}  "
                  f"recall@0.1: {overall[name]['at_budget'][str(HEADLINE_FP_BUDGET)]['recall']:.3f}")
        if include_lead_time:
            for name in DETECTOR_NAMES:
                s = lead_time_summary.get(name, {})
                if s:
                    print(f"  {name} lead time: detection {s['detection_rate']:.2f}  pre-violation {s['pre_violation_rate']:.2f}  "
                          f"median (pre-violation) {s['median_lead_pre_violation_s']}")
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_results(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = {k: v for k, v in result.items() if not k.startswith("_")}
    write_json(out_dir / "evaluation.json", ev)
    write_json(out_dir / "coverage_gap.json", result["coverage_gap"])
    write_json(out_dir / "operating_points.json", result["operating_points"])

    feats_eval, scores_eval, classes_eval = result["_frames"]["scores"]
    scores_df = pd.DataFrame({"session_id": feats_eval.index, "label": classes_eval})
    for name in DETECTOR_NAMES:
        scores_df[name] = scores_eval[name]
    scores_df.to_csv(out_dir / "scores.csv", index=False)

    lt = result["_frames"]["lead_times"]
    if len(lt):
        lt.to_csv(out_dir / "lead_times.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate D1-D4 on the synthetic benchmark.")
    ap.add_argument("--smoke", action="store_true", help="evaluate the smoke dataset in data/smoke/")
    args = ap.parse_args()

    if args.smoke:
        sessions_path = SESSIONS_PATH.parent / "smoke" / "synthetic_sessions.csv"
        labels_path = SESSIONS_PATH.parent / "smoke" / "anomaly_labels.csv"
    else:
        sessions_path = SESSIONS_PATH
        labels_path = LABELS_PATH

    print(f"Evaluating {sessions_path.name} ...")
    sessions = F.load_sessions(sessions_path)
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    agents, _, _ = build_agents(SEED_AGENTS)

    if args.smoke:
        result = run_evaluation(sessions, labels, agents[:60], n_days=2)
    else:
        result = run_evaluation(sessions, labels, agents)
    if not args.smoke:
        write_results(result, RESULTS_DIR)
        print(f"Wrote {RESULTS_DIR}/evaluation.json (+ coverage_gap.json, operating_points.json, scores.csv, lead_times.csv)")
    else:
        print("Smoke evaluation complete (no files written).")


if __name__ == "__main__":
    main()
