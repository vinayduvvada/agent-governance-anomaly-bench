"""Detectors D1-D4 with a common interface, plus threshold calibration.

    D1  static rule thresholds (interpretable operational baseline)
    D2  per-agent statistical baselining (robust z / EWMA-style drift on signals)
    D3  Isolation Forest on session feature vectors (mixed-length training rows
        make the model usable online, i.e. evaluable on session prefixes)
    D4  Markov transition-likelihood scoring over per-agent tool-call sequences

Interface: fit(calibration data) -> self; score(features: DataFrame) -> np.ndarray
of anomaly scores (higher = more anomalous). Scores from all detectors are
comparable within a detector across time, which is what the lead-time protocol
relies on: thresholds are calibrated on complete calibration sessions and then
applied to growing prefixes of evaluation sessions.

Fully synthetic benchmark; see README disclosure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FEATURE_NAMES, LOG_FEATURES, TOOL_NAMES  # noqa: E402
from simulation_constants import (  # noqa: E402
    D1_RULES,
    D2_SIGNAL_FLOORS,
    D2_SIGNALS,
    D2_Z_CLIP,
    D3_MAX_SAMPLES,
    D3_N_ESTIMATORS,
    D4_ALPHA,
    D4_MIN_CALIBRATION_SESSIONS,
)


class BaseDetector:
    name = "base"

    def fit(self, *args, **kwargs) -> "BaseDetector":
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# D1: static rule thresholds
# ---------------------------------------------------------------------------


def _rel_margin(x, thr):
    """Relative exceedance: max(0, x/thr - 1). Zero when the rule is not tripped."""
    return np.maximum(0.0, np.asarray(x, dtype=float) / thr - 1.0)


class D1StaticRules(BaseDetector):
    """Weighted sum of normalized exceedances over hand-set operational rules.

    Rules mirror the taxonomy signals: scope discipline, denied-request bursts,
    retry-without-escalation, egress share, budget, repetition, and gate bypass.
    """

    name = "D1"

    def score(self, features: pd.DataFrame) -> np.ndarray:
        f = features
        margins = [
            _rel_margin(f["out_of_scope_ratio"], D1_RULES["out_of_scope_ratio"]),
            _rel_margin(f["denied_permission_count"], D1_RULES["denied_permission_count"]),
            _rel_margin(f["retry_ratio"], D1_RULES["silent_retry_ratio"])
            * (f["n_retries"].to_numpy() >= D1_RULES["silent_retries_min"])
            * (f["escalation_count"].to_numpy() == 0),
            _rel_margin(f["egress_token_share"], D1_RULES["egress_token_share"]),
            _rel_margin(f["cost_ratio_vs_baseline"], D1_RULES["cost_ratio_vs_baseline"]),
            _rel_margin(f["max_pattern_repetition"], D1_RULES["max_pattern_repetition"]),
            _rel_margin(f["gate_bypass_count"], D1_RULES["gate_bypass_count"]),
        ]
        return np.sum(margins, axis=0)


# ---------------------------------------------------------------------------
# D2: per-agent statistical baselining
# ---------------------------------------------------------------------------


class D2PerAgentBaseline(BaseDetector):
    """Robust z-scores of each governance signal against the agent's own history.

    Scale = max((q90 - q50)/1.2816, per-signal floor); agents with fewer than
    MIN_SESSIONS calibration sessions fall back to task-class statistics. The
    session score is the maximum positive z across signals, clipped, which acts
    as an EWMA-style drift alarm on whichever signal moves first.
    """

    name = "D2"
    MIN_SESSIONS = 15

    def __init__(self) -> None:
        self.agent_params: dict[str, dict[str, tuple[float, float]]] = {}
        self.class_params: dict[str, dict[str, tuple[float, float]]] = {}

    @staticmethod
    def _params_for(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
        out = {}
        for sig in D2_SIGNALS:
            x = df[sig].to_numpy(dtype=float)
            q50 = float(np.quantile(x, 0.50))
            q90 = float(np.quantile(x, 0.90))
            scale = max((q90 - q50) / 1.2816, D2_SIGNAL_FLOORS[sig])
            out[sig] = (q50, scale)
        return out

    def fit(self, calib_features: pd.DataFrame) -> "D2PerAgentBaseline":
        for cls, g in calib_features.groupby("task_class"):
            self.class_params[cls] = self._params_for(g)
        for aid, g in calib_features.groupby("agent_id"):
            if len(g) >= self.MIN_SESSIONS:
                self.agent_params[aid] = self._params_for(g)
        # safety net for task classes with very few sessions
        self.class_params.setdefault("__global__", self._params_for(calib_features))
        return self

    def _lookup(self, agent_id: str, task_class: str, sig: str) -> tuple[float, float]:
        p = self.agent_params.get(agent_id)
        if p is None:
            p = self.class_params.get(task_class) or self.class_params["__global__"]
        return p[sig]

    def score(self, features: pd.DataFrame) -> np.ndarray:
        n = len(features)
        agents = features["agent_id"].to_numpy()
        classes = features["task_class"].to_numpy()
        best = np.zeros(n)
        for sig in D2_SIGNALS:
            q50 = np.empty(n)
            scale = np.empty(n)
            for i in range(n):
                q50[i], scale[i] = self._lookup(agents[i], classes[i], sig)
            x = features[sig].to_numpy(dtype=float)
            z = np.clip((x - q50) / scale, 0.0, D2_Z_CLIP)
            best = np.maximum(best, z)
        return best


# ---------------------------------------------------------------------------
# D3: Isolation Forest on session feature vectors
# ---------------------------------------------------------------------------


class D3IsolationForest(BaseDetector):
    """Unsupervised outlier detection over the full feature vector.

    Training rows are mixed-length session-prefix vectors (see
    features.extract_prefix_rows), so the fitted model is meaningful when
    applied to partial sessions for lead-time evaluation.
    """

    name = "D3"

    def __init__(self) -> None:
        self._scaler = None
        self._model = None
        self._col_idx = [FEATURE_NAMES.index(c) for c in FEATURE_NAMES]

    def _prepare(self, features: pd.DataFrame, fit: bool = False) -> np.ndarray:
        x = features[list(FEATURE_NAMES)].to_numpy(dtype=float)
        log_idx = [FEATURE_NAMES.index(c) for c in LOG_FEATURES]
        x[:, log_idx] = np.log1p(np.maximum(0.0, x[:, log_idx]))
        if fit:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler().fit(x)
        return self._scaler.transform(x)

    def fit(self, train_rows: pd.DataFrame) -> "D3IsolationForest":
        from sklearn.ensemble import IsolationForest

        x = self._prepare(train_rows, fit=True)
        self._model = IsolationForest(
            n_estimators=D3_N_ESTIMATORS,
            max_samples=min(D3_MAX_SAMPLES, len(x)),
            random_state=42,
            n_jobs=1,  # small model; single-thread avoids process-spawn overhead and is deterministic
        ).fit(x)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        x = self._prepare(features)
        return -self._model.decision_function(x)  # higher = more anomalous


# ---------------------------------------------------------------------------
# D4: Markov transition-likelihood model
# ---------------------------------------------------------------------------


class D4Markov(BaseDetector):
    """Per-agent first-order Markov chain over tool-call sequences.

    Transition and start counts are collected from calibration sessions only;
    probabilities use Dirichlet (add-alpha) smoothing over the agent's permitted
    tool alphabet, so unpermitted/unseen tools receive a floor probability and
    the mean log-likelihood is the anomaly statistic (lower likelihood = higher
    score). Agents with too little history fall back to their task-class chain.
    """

    name = "D4"

    def __init__(self) -> None:
        self.agent_id_of: dict[str, str] = {}
        self.agent_task_class: dict[str, str] = {}
        self.agent_logP: dict[str, np.ndarray] = {}
        self.agent_log_start: dict[str, np.ndarray] = {}
        self.class_logP: dict[str, np.ndarray] = {}
        self.class_log_start: dict[str, np.ndarray] = {}
        self.agent_sessions: dict[str, int] = {}
        self.n_tools = len(TOOL_NAMES)

    @staticmethod
    def _counts_from_sequences(sequences: list[list[int]], k_tools: int) -> tuple[np.ndarray, np.ndarray]:
        c = np.zeros((k_tools, k_tools), dtype=float)
        start = np.zeros(k_tools, dtype=float)
        for seq in sequences:
            if not seq:
                continue
            start[seq[0]] += 1.0
            for a, b in zip(seq, seq[1:]):
                c[a, b] += 1.0
        return c, start

    @staticmethod
    def _log_probs(c: np.ndarray, start: np.ndarray, k_alphabet: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        n = c.shape[0]
        logp = np.full((n, n), -25.0)  # floor for impossible/degenerate rows
        for t in range(n):
            row = c[t]
            denom = row.sum() + alpha * k_alphabet
            if denom > 0:
                logp[t] = np.log((row + alpha) / denom)
        sdenom = start.sum() + alpha * k_alphabet
        log_start = np.log((start + alpha) / sdenom) if sdenom > 0 else np.full(n, -25.0)
        return logp, log_start

    def fit(
        self,
        session_arrays: dict[str, dict],
        session_agent: pd.Series,
        calib_ids,
        agent_task_class: dict[str, str],
        agent_permitted: dict[str, set[int]],
    ) -> "D4Markov":
        self.agent_task_class = dict(agent_task_class)
        per_agent_seqs: dict[str, list[list[int]]] = {}
        per_class_seqs: dict[str, list[list[int]]] = {}
        for sid in calib_ids:
            aid = session_agent[sid]
            arr = session_arrays[sid]
            seq = arr["tool"][arr["etype"] == 0].tolist()
            if not seq:
                continue
            per_agent_seqs.setdefault(aid, []).append(seq)
            per_class_seqs.setdefault(self.agent_task_class.get(aid, "__global__"), []).append(seq)

        default_alphabet = len(TOOL_NAMES)
        for cls, seqs in per_class_seqs.items():
            c, start = self._counts_from_sequences(seqs, self.n_tools)
            logp, log_start = self._log_probs(c, start, default_alphabet, D4_ALPHA)
            self.class_logP[cls] = logp
            self.class_log_start[cls] = log_start

        for aid, seqs in per_agent_seqs.items():
            self.agent_sessions[aid] = len(seqs)
            if len(seqs) < D4_MIN_CALIBRATION_SESSIONS:
                continue
            k = len(agent_permitted.get(aid, set(range(self.n_tools))))
            c, start = self._counts_from_sequences(seqs, self.n_tools)
            logp, log_start = self._log_probs(c, start, max(k, 1), D4_ALPHA)
            self.agent_logP[aid] = logp
            self.agent_log_start[aid] = log_start
        return self

    def _model_for(self, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
        if agent_id in self.agent_logP:
            return self.agent_logP[agent_id], self.agent_log_start[agent_id]
        cls = self.agent_task_class.get(agent_id, "__global__")
        return self.class_logP.get(cls, next(iter(self.class_logP.values()))), self.class_log_start.get(
            cls, next(iter(self.class_log_start.values()))
        )

    def session_mean_logprob(self, agent_id: str, tools: list[int]) -> float:
        """Mean log-probability of the tool-call sequence (start + transitions)."""
        if not tools:
            return 0.0
        logp, log_start = self._model_for(agent_id)
        total = log_start[tools[0]]
        for a, b in zip(tools, tools[1:]):
            total += logp[a, b]
        return float(total / len(tools))

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return -features["transition_logprob_mean"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def threshold_for_budget(scores: np.ndarray, agent_days: float, budget_per_agent_day: float) -> float:
    """Score threshold whose expected false-positive count matches the budget.

    budget_per_agent_day x agent_days = allowed false positives on the
    calibration set; the threshold is the corresponding order statistic.
    """
    k = max(1, int(round(budget_per_agent_day * agent_days)))
    srt = np.sort(np.asarray(scores, dtype=float))[::-1]
    k = min(k, len(srt))
    return float(srt[k - 1])


def realized_fp_per_agent_day(scores: np.ndarray, threshold: float, agent_days: float) -> float:
    fp = int((np.asarray(scores) >= threshold).sum())
    return fp / max(1e-9, agent_days)
