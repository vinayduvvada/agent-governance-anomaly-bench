"""Phase 1 of the pipeline: build the synthetic agent population and write
data/generation_metadata.json (seeds, tools, task-class profiles, agents).

Each agent receives:
  - a task class and a personalized declared-scope set,
  - a per-agent Markov transition matrix over its permitted tools (the class
    template perturbed with Dirichlet noise, then masked to permissions),
  - a session rate (sessions/day) and an analytic cost baseline (USD/session),
  - style (session-length) and speed (latency) multipliers.

Fully synthetic: all parameters are illustrative (see
code/simulation_constants.py and the README disclosure).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation_constants import (  # noqa: E402
    AGENTS_METADATA_PATH,
    N_AGENTS,
    SEED_AGENTS,
    SESSION_LEN_NEGBIN_P,
    SESSION_LEN_NEGBIN_R,
    SESSION_STYLE_SIGMA,
    TASK_CLASS_PROFILES,
    TOOLS,
    WORKSPACE_NAMES,
    r6,
    write_json,
)

TASK_CLASS_ORDER = list(TASK_CLASS_PROFILES.keys())


# ---------------------------------------------------------------------------
# Markov-chain construction
# ---------------------------------------------------------------------------


def class_base_matrix(palette: list[str], workflow_template: list[str]) -> np.ndarray:
    """Build a class-level transition matrix from the canonical workflow template.

    Consecutive template pairs get +1.0 count, self-loops +0.35, and a 0.5
    pseudo-count prior keeps every transition reachable. Rows are normalized.
    """
    idx = {t: i for i, t in enumerate(palette)}
    n = len(palette)
    counts = np.full((n, n), 0.5, dtype=float)
    for a, b in zip(workflow_template, workflow_template[1:]):
        counts[idx[a], idx[b]] += 1.0
    for t in palette:
        counts[idx[t], idx[t]] += 0.35
    return counts / counts.sum(axis=1, keepdims=True)


def class_start_distribution(palette: list[str]) -> np.ndarray:
    """Slight preference for opening a session with a read-only tool."""
    w = np.array([1.5 if TOOLS[t]["category"] == "read" else 1.0 for t in palette], dtype=float)
    return w / w.sum()


def perturb_matrix(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-agent chain: 85% class template + 15% Dirichlet(0.5) draw."""
    n = base.shape[0]
    noise = rng.dirichlet(np.full(n, 0.5), size=n)
    m = 0.85 * base + 0.15 * noise
    return m / m.sum(axis=1, keepdims=True)


def mask_to_permissions(matrix: np.ndarray, palette: list[str], permitted: list[str]) -> np.ndarray:
    """Restrict a chain to permitted tools and renormalize each row."""
    keep = [i for i, t in enumerate(palette) if t in permitted]
    m = matrix[np.ix_(keep, keep)]
    m = m / m.sum(axis=1, keepdims=True)
    return m, keep


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Power iteration for the stationary distribution of a row-stochastic matrix."""
    n = matrix.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(200):
        nxt = pi @ matrix
        if np.max(np.abs(nxt - pi)) < 1e-12:
            pi = nxt
            break
        pi = nxt
    return pi / pi.sum()


# ---------------------------------------------------------------------------
# Cost baselines
# ---------------------------------------------------------------------------


def analytic_cost_baseline(matrix: np.ndarray, palette_masked: list[str], style_length_multiplier: float) -> tuple[float, float]:
    """Expected USD per session and per tool call under the agent's own chain.

    per-call cost = tokens_median/1000 * price + call fee, averaged under the
    chain's stationary distribution; session cost adds the retry overhead and
    the terminal output event. This is the declared cost baseline recorded in
    metadata and used by D1's budget rule and by the budget-based anomaly
    injectors (illustrative, analytic).
    """
    pi = stationary_distribution(matrix)
    per_call = np.array(
        [
            TOOLS[t]["tokens_median"] / 1000.0 * TOOLS[t]["price_per_1k"] + TOOLS[t]["call_fee"]
            for t in palette_masked
        ]
    )
    expected_per_call = float(np.sum(pi * per_call))
    expected_calls = 7.0 * style_length_multiplier
    retry_overhead = 1.02
    output_cost = 2200 / 1000.0 * 0.015
    session_cost = expected_calls * expected_per_call * retry_overhead + output_cost
    return session_cost, expected_per_call


# ---------------------------------------------------------------------------
# Population builder
# ---------------------------------------------------------------------------


def build_agents(seed: int = SEED_AGENTS) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build the full agent population deterministically.

    Returns (agents, class_matrices, class_start_dists). Per-agent chains are
    derivable from this function alone; metadata stores the agent-level
    attributes needed by detectors and the generator.
    """
    class_matrices: dict[str, np.ndarray] = {}
    class_starts: dict[str, np.ndarray] = {}
    for cls, profile in TASK_CLASS_PROFILES.items():
        class_matrices[cls] = class_base_matrix(profile["palette"], profile["workflow_template"])
        class_starts[cls] = class_start_distribution(profile["palette"])

    agents: list[dict] = []
    for i in range(N_AGENTS):
        cls = TASK_CLASS_ORDER[i % len(TASK_CLASS_ORDER)]
        profile = TASK_CLASS_PROFILES[cls]
        palette = profile["palette"]
        rng = np.random.default_rng([seed, i])  # per-agent stream, order-independent

        base = class_matrices[cls]
        agent_chain_full = perturb_matrix(base, rng)

        # Permissions: drop non-core tools with p=0.15; always keep the three
        # tools carrying the most stationary mass (the agent's core workflow).
        pi = stationary_distribution(agent_chain_full)
        core = set(np.array(palette)[np.argsort(pi)[::-1][:3]].tolist())
        permitted = [t for t in palette if (t in core or rng.random() > 0.15)]
        if len(permitted) < 3:
            permitted = palette[:3]

        chain, _ = mask_to_permissions(agent_chain_full, palette, permitted)

        ws = f"{WORKSPACE_NAMES[i % len(WORKSPACE_NAMES)]}-{i:03d}"
        sb = f"sb-{i:03d}"
        declared_scopes = [s.format(ws=ws, sb=sb) for s in profile["scope_template"]]

        lo, hi = profile["session_rate_range"]
        session_rate = float(lo + (hi - lo) * rng.random())
        style_len = float(np.exp(rng.normal(0.0, SESSION_STYLE_SIGMA)))
        speed = float(np.exp(rng.normal(0.0, 0.12)))

        cost_baseline, cost_per_call = analytic_cost_baseline(chain, permitted, style_len)

        agents.append(
            {
                "agent_id": f"AGT-{i + 1:04d}",
                "agent_index": i,
                "task_class": cls,
                "session_rate": r6(session_rate),
                "cost_baseline": r6(cost_baseline),
                "cost_per_call_baseline": r6(cost_per_call),
                "tool_permissions": permitted,
                "declared_scopes": declared_scopes,
                "style_length_multiplier": r6(style_len),
                "speed_multiplier": r6(speed),
            }
        )
    return agents, class_matrices, class_starts


def agent_chain_for(agent: dict, class_matrices: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct a single agent's masked chain deterministically."""
    i = agent["agent_index"]
    cls = agent["task_class"]
    palette = TASK_CLASS_PROFILES[cls]["palette"]
    rng = np.random.default_rng([SEED_AGENTS, i])
    full = perturb_matrix(class_matrices[cls], rng)
    chain, _ = mask_to_permissions(full, palette, agent["tool_permissions"])
    return chain


def main() -> None:
    agents, _, _ = build_agents(SEED_AGENTS)

    # Rebuild the exact per-agent ordering of permitted palettes for metadata.
    metadata = {
        "artifact": "generation_metadata",
        "schema_version": "1.0.0",
        "disclosure": "Fully synthetic benchmark. Illustrative parameters only; no product internals, customer data, or proprietary telemetry.",
        "principal_seeds": {"agents": SEED_AGENTS},
        "n_agents": N_AGENTS,
        "session_length_model": {
            "distribution": "negative_binomial (clipped to [3, 18])",
            "r": SESSION_LEN_NEGBIN_R,
            "p": SESSION_LEN_NEGBIN_P,
            "style_sigma": SESSION_STYLE_SIGMA,
        },
        "tools": {name: {k: (list(v) if isinstance(v, list) else v) for k, v in meta.items()} for name, meta in TOOLS.items()},
        "task_classes": {
            cls: {
                "palette": p["palette"],
                "workflow_template": p["workflow_template"],
                "scope_template": p["scope_template"],
                "session_rate_range": list(p["session_rate_range"]),
            }
            for cls, p in TASK_CLASS_PROFILES.items()
        },
        "agents": agents,
    }
    write_json(AGENTS_METADATA_PATH, metadata)

    by_class: dict[str, int] = {}
    for a in agents:
        by_class[a["task_class"]] = by_class.get(a["task_class"], 0) + 1
    total_sessions = sum(a["session_rate"] for a in agents) * 10
    print(f"Wrote {AGENTS_METADATA_PATH}")
    print(f"  agents: {len(agents)} across {len(by_class)} task classes: {by_class}")
    print(f"  expected sessions over 10 days: ~{total_sessions:,.0f}")
    print(f"  mean cost baseline: ${np.mean([a['cost_baseline'] for a in agents]):.4f}/session")


if __name__ == "__main__":
    main()
