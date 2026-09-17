"""Shared per-session feature extraction for all detectors (D1-D4).

Design contract
---------------
Every detector consumes the same per-session feature vector produced here, and
the lead-time evaluator rescoring session *prefixes* uses the very same
function, so a score computed mid-session is directly comparable to the
threshold calibrated on complete sessions. Features are therefore restricted to
quantities that are well defined on a growing prefix of events.

Sessions are converted once into compact numpy arrays (build_session_arrays);
extract_features() then computes the vector for any prefix length k.

Feature groups
--------------
volume/cost, latency, permission behavior, retries/escalations, scope
discipline, gate discipline (approval gaps / post-hoc approvals / proxy
bypasses), repetition structure, and sequence likelihood under the agent's own
calibrated Markov chain (supplied by detectors.D4Markov; see chain_model).

Fully synthetic benchmark; see README disclosure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation_constants import EGRESS_TOOLS, GATED_TOOLS, TOOLS  # noqa: E402

EVENT_TYPES = ("tool_call", "permission_request", "escalation", "retry", "output")
ETYPE_IDX = {t: i for i, t in enumerate(EVENT_TYPES)}
TOOL_NAMES = list(TOOLS.keys())
TOOL_IDX = {t: i for i, t in enumerate(TOOL_NAMES)}
GATED_TOOL_IDX = np.array([TOOL_IDX[t] for t in GATED_TOOLS], dtype=np.int16)
EGRESS_TOOL_IDX = set(TOOL_IDX[t] for t in EGRESS_TOOLS)
TOOL_FAMILIES = {name: set(meta["families"]) for name, meta in TOOLS.items()}

FEATURE_NAMES = (
    "n_events",
    "n_calls",
    "n_unique_tools",
    "tool_entropy",
    "tool_entropy_shift",
    "duration_s",
    "cost_total",
    "cost_ratio_vs_baseline",
    "tokens_total",
    "tokens_per_call",
    "cost_per_call",
    "cost_per_call_over_baseline",
    "latency_mean_ms",
    "latency_p95_ms",
    "n_permission_requests",
    "denied_permission_count",
    "denied_permission_ratio",
    "n_retries",
    "retry_ratio",
    "escalation_count",
    "escalation_to_retry_ratio",
    "out_of_scope_count",
    "out_of_scope_ratio",
    "novel_scope_count",
    "egress_count",
    "egress_token_share",
    "egress_token_volume",
    "max_pattern_repetition",
    "gate_bypass_count",
    "approval_gap_count",
    "post_hoc_approval_count",
    "proxy_bypass_count",
    "approval_gap_ratio",
    "gated_call_count",
    "new_tool_count",
    "transition_logprob_mean",
    "transition_min3_logprob_mean",
    "transition_drop",
)

# Heavy-tailed non-negative features are log1p-transformed before D3 scaling.
LOG_FEATURES = (
    "duration_s", "cost_total", "tokens_total", "tokens_per_call", "cost_per_call",
    "latency_mean_ms", "latency_p95_ms", "egress_token_volume",
)


# ---------------------------------------------------------------------------
# Loading and array conversion
# ---------------------------------------------------------------------------


def load_sessions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "session_id": str, "agent_id": str, "event_type": str,
            "tool_name": str, "target_scope": str, "approved": str,
        },
        keep_default_na=False,
    )


def build_session_arrays(sessions: pd.DataFrame) -> tuple[dict[str, dict], list[str], pd.Series]:
    """Convert the event table into per-session numpy arrays (zero-copy views).

    The file is sorted by (session_id, ts), so each session is a contiguous
    block; block slicing avoids a per-group pandas round trip.

    Returns (arrays_by_session, scope_vocab, session_agent).
    Scope vocabulary order is the deterministic order of first appearance.
    """
    scope_vocab = list(pd.unique(sessions["target_scope"]))
    scope_idx_map = {s: i for i, s in enumerate(scope_vocab)}

    cols = {
        "ts": sessions["ts"].to_numpy(np.float64),
        "etype": sessions["event_type"].map(ETYPE_IDX).to_numpy(np.int8),
        "tool": sessions["tool_name"].map(lambda t: TOOL_IDX.get(t, -1)).to_numpy(np.int16),
        "latency": sessions["latency_ms"].to_numpy(np.float64),
        "tokens": sessions["tokens"].to_numpy(np.float64),
        "cost": sessions["cost"].to_numpy(np.float64),
        "approved": sessions["approved"].map({"true": 1, "false": 0, "": -1}).to_numpy(np.int8),
        "scope": sessions["target_scope"].map(scope_idx_map).to_numpy(np.int16),
    }
    sid_col = sessions["session_id"].to_numpy()
    agent_col = sessions["agent_id"].to_numpy()

    change = np.flatnonzero(sid_col[1:] != sid_col[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(sid_col)]])

    arrays: dict[str, dict] = {}
    session_agent = {}
    for st, en in zip(starts, ends):
        sid = str(sid_col[st])
        arrays[sid] = {name: col[st:en] for name, col in cols.items()}
        session_agent[sid] = str(agent_col[st])
    return arrays, scope_vocab, pd.Series(session_agent, name="agent_id")


def build_scope_families(scope_vocab: list[str]) -> np.ndarray:
    return np.array([s.split(":")[0] for s in scope_vocab], dtype=object)


def build_agent_contexts(agents: list[dict], scope_idx_map: dict[str, int]) -> dict[str, dict]:
    """Per-agent lookup used by feature extraction (declared scopes, permissions, baselines)."""
    ctx = {}
    for a in agents:
        ctx[a["agent_id"]] = {
            "agent_id": a["agent_id"],
            "task_class": a["task_class"],
            "cost_baseline": float(a["cost_baseline"]),
            "cost_per_call_baseline": float(a["cost_per_call_baseline"]),
            "declared_scope_idx": np.array(sorted(scope_idx_map[s] for s in a["declared_scopes"] if s in scope_idx_map), dtype=np.int16),
            "declared_scopes": set(a["declared_scopes"]),
            "permitted_idx": set(TOOL_IDX[t] for t in a["tool_permissions"]),
        }
    return ctx


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def max_pattern_repetition(tools: list[int]) -> int:
    """Longest consecutive repetition of a period-p pattern (p in 1..3), in repeats."""
    n = len(tools)
    if n == 0:
        return 0
    best = 1
    for p in (1, 2, 3):
        if n <= p:
            continue
        run = 0
        best_run = 0
        for j in range(p, n):
            if tools[j] == tools[j - p]:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        best = max(best, 1 + best_run // p)
    return best


def extract_features(arr: dict, agent_ctx: dict, scope_families: np.ndarray, k: int | None = None, chain_model=None) -> dict:
    """Compute the feature vector for the first k events of a session (None = all)."""
    n = len(arr["ts"]) if k is None else min(k, len(arr["ts"]))
    ts = arr["ts"][:n]
    et = arr["etype"][:n]
    tool = arr["tool"][:n]
    lat = arr["latency"][:n]
    tok = arr["tokens"][:n]
    cost = arr["cost"][:n]
    appr = arr["approved"][:n]
    scope = arr["scope"][:n]

    call_mask = et == 0
    n_calls = int(call_mask.sum())
    calls_tool = tool[call_mask]
    calls_scope = scope[call_mask]

    denied_count = int(((et == 1) & (appr == 0)).sum())
    n_req = int((et == 1).sum())
    n_retries = int((et == 3).sum())
    n_escal = int((et == 2).sum())

    # Scope discipline
    declared = agent_ctx["declared_scope_idx"]
    if n_calls:
        oos_mask = ~np.isin(calls_scope, declared) if len(declared) else np.ones(n_calls, bool)
        oos_count = int(oos_mask.sum())
        novel_scopes = len(set(calls_scope[oos_mask].tolist()))
    else:
        oos_count = 0
        novel_scopes = 0

    # Egress accounting
    if n_calls:
        egr_mask = np.isin(calls_tool, list(EGRESS_TOOL_IDX)) if EGRESS_TOOL_IDX else np.zeros(n_calls, bool)
        egress_count = int(egr_mask.sum())
        egress_tokens = float(tok[call_mask][egr_mask].sum())
    else:
        egress_count = 0
        egress_tokens = 0.0

    tokens_total = float(tok.sum())
    cost_total = float(cost.sum())
    tokens_per_call = tokens_total / max(1, n_calls)
    calls_cost = cost[call_mask | (et == 3)].sum()
    cost_per_call = float(calls_cost) / max(1, n_calls)

    # Latency over completed tool calls
    if n_calls:
        lat_calls = lat[call_mask]
        latency_mean = float(lat_calls.mean())
        latency_p95 = float(np.percentile(lat_calls, 95))
    else:
        latency_mean = 0.0
        latency_p95 = 0.0

    # Entropy over tool distribution + split-half shift
    if n_calls:
        counts = np.bincount(calls_tool[calls_tool >= 0], minlength=len(TOOL_NAMES))
        tool_entropy = _entropy(counts)
        half = n_calls // 2
        if half >= 2:
            h1 = _entropy(np.bincount(calls_tool[:half][calls_tool[:half] >= 0], minlength=len(TOOL_NAMES)))
            h2 = _entropy(np.bincount(calls_tool[half:][calls_tool[half:] >= 0], minlength=len(TOOL_NAMES)))
            tool_entropy_shift = abs(h1 - h2)
        else:
            tool_entropy_shift = 0.0
        n_unique_tools = int(len(set(calls_tool.tolist())))
    else:
        tool_entropy = 0.0
        tool_entropy_shift = 0.0
        n_unique_tools = 0

    # Gate discipline: for each gated call, look for an approved request for the
    # same tool before it (ok), after it (post-hoc), or nowhere (approval gap).
    approval_gap = 0
    post_hoc = 0
    gated_calls = 0
    proxy_bypass = 0
    req_idx_by_tool: dict[int, list[int]] = {}
    for i in np.where((et == 1) & (appr == 1))[0]:
        req_idx_by_tool.setdefault(int(tool[i]), []).append(int(i))
    for i in np.where(call_mask)[0]:
        t = int(tool[i])
        if t in GATED_TOOL_IDX:
            gated_calls += 1
            lst = req_idx_by_tool.get(t, [])
            if not any(j < i for j in lst):
                if any(j > i for j in lst):
                    post_hoc += 1
                else:
                    approval_gap += 1
        else:
            fam = scope_families[int(scope[i])] if n else ""
            if fam and fam not in TOOL_FAMILIES[TOOL_NAMES[t]]:
                proxy_bypass += 1  # ungated tool acting on a scope outside its families
    gate_bypass = approval_gap + post_hoc + proxy_bypass

    new_tool_count = int(sum(1 for t in calls_tool if int(t) not in agent_ctx["permitted_idx"]))

    # Sequence likelihood under the agent's own chain (features shared with D4)
    if chain_model is not None and n_calls:
        transition_logprob_mean, transition_min3, transition_drop = chain_model.session_logprob_stats(
            agent_ctx["agent_id"], calls_tool.tolist()
        )
    else:
        transition_logprob_mean, transition_min3, transition_drop = 0.0, 0.0, 0.0

    return {
        "n_events": n,
        "n_calls": n_calls,
        "n_unique_tools": n_unique_tools,
        "tool_entropy": tool_entropy,
        "tool_entropy_shift": tool_entropy_shift,
        "duration_s": float(ts[-1] - ts[0]) if n else 0.0,
        "cost_total": cost_total,
        "cost_ratio_vs_baseline": cost_total / max(1e-9, agent_ctx["cost_baseline"]),
        "tokens_total": tokens_total,
        "tokens_per_call": tokens_per_call,
        "cost_per_call": cost_per_call,
        "cost_per_call_over_baseline": cost_per_call / max(1e-9, agent_ctx["cost_per_call_baseline"]),
        "latency_mean_ms": latency_mean,
        "latency_p95_ms": latency_p95,
        "n_permission_requests": n_req,
        "denied_permission_count": denied_count,
        "denied_permission_ratio": denied_count / max(1, n_req),
        "n_retries": n_retries,
        "retry_ratio": n_retries / max(1, n_calls),
        "escalation_count": n_escal,
        "escalation_to_retry_ratio": n_escal / max(1, n_retries),
        "out_of_scope_count": oos_count,
        "out_of_scope_ratio": oos_count / max(1, n_calls),
        "novel_scope_count": novel_scopes,
        "egress_count": egress_count,
        "egress_token_share": egress_tokens / max(1e-9, tokens_total),
        "egress_token_volume": egress_tokens,
        "max_pattern_repetition": max_pattern_repetition(calls_tool.tolist()),
        "gate_bypass_count": gate_bypass,
        "approval_gap_count": approval_gap,
        "post_hoc_approval_count": post_hoc,
        "proxy_bypass_count": proxy_bypass,
        "approval_gap_ratio": gate_bypass / max(1, n_calls),
        "gated_call_count": gated_calls,
        "new_tool_count": new_tool_count,
        "transition_logprob_mean": transition_logprob_mean,
        "transition_min3_logprob_mean": transition_min3,
        "transition_drop": transition_drop,
    }


def extract_features_frame(
    arrays: dict[str, dict],
    session_agent: pd.Series,
    agent_ctx: dict[str, dict],
    scope_families: np.ndarray,
    chain_model=None,
    session_ids=None,
    progress_label: str | None = None,
) -> pd.DataFrame:
    """Feature vectors for complete sessions (rows indexed by session_id)."""
    ids = list(arrays.keys()) if session_ids is None else list(session_ids)
    rows = []
    for sid in ids:
        aid = session_agent[sid]
        rows.append(extract_features(arrays[sid], agent_ctx[aid], scope_families, None, chain_model))
    df = pd.DataFrame(rows, index=pd.Index(ids, name="session_id"))
    df["agent_id"] = [session_agent[sid] for sid in ids]
    df["task_class"] = [agent_ctx[session_agent[sid]]["task_class"] for sid in ids]
    return df


def sample_prefix_lengths(n: int, rng: np.random.Generator, n_samples: int) -> list[int]:
    """Prefix lengths for mixed-length calibration rows (includes the full session)."""
    if n <= 1:
        return [n]
    ks = set(int(rng.integers(1, n + 1)) for _ in range(n_samples))
    ks.add(n)
    return sorted(ks)


def extract_prefix_rows(
    arrays: dict[str, dict],
    session_agent: pd.Series,
    agent_ctx: dict[str, dict],
    scope_families: np.ndarray,
    chain_model,
    session_ids,
    n_samples_per_session: int,
    seed: int,
) -> pd.DataFrame:
    """Mixed-length feature rows over calibration sessions (used to fit D3)."""
    from simulation_constants import spawn_rng

    rows = []
    idx = []
    aids = []
    for sid in session_ids:
        aid = session_agent[sid]
        n = len(arrays[sid]["ts"])
        rng = spawn_rng(seed, "prefix", sid)
        for k in sample_prefix_lengths(n, rng, n_samples_per_session):
            rows.append(extract_features(arrays[sid], agent_ctx[aid], scope_families, k, chain_model))
            idx.append((sid, k))
            aids.append(aid)
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["session_id", "prefix_k"]))
    df["agent_id"] = aids
    df["task_class"] = [agent_ctx[a]["task_class"] for a in aids]
    return df
