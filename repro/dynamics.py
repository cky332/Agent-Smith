#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics.py — Reproduce the *infection dynamics* of Agent Smith at scale
(up to one million agents) using the parameters (alpha, beta) measured on the
real model by measure.py.

The paper models randomized pairwise chat as an SIS-type epidemic. Each round
the N agents are split into N/2 questioners and N/2 answerers; a virus-carrying
questioner infects its (benign) answerer with probability beta. Because each
agent is a questioner with probability 1/2, the *effective* per-contact infection
rate is beta/2, which yields the mean-field recursion

        p_{t+1} = p_t + (beta/2) * p_t * (1 - p_t) - gamma * p_t          (*)

with carrier ratio p_t and recovery (defense) rate gamma. Its non-zero fixed
point is p* = 1 - 2*gamma/beta, giving the paper's phase transition:

        beta > 2*gamma  -> endemic, p_inf = 1 - 2*gamma/beta
        beta = 2*gamma  -> p_inf = 0   (critical)
        beta < 2*gamma  -> p_inf = 0   exponentially fast   (DEFENSE WORKS)

We additionally track the *symptom* (actually-jailbroken) ratio = alpha * p_t,
which is what "exhibits harmful behavior" means in the paper.

Two engines:
  - mean_field(): deterministic recursion (*).
  - agent_based(): explicit, vectorized Monte-Carlo with real random pairing,
    valid up to N = 1e6, used to check the mean-field constant (the 1/2).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Mean-field recursion (*)
# ---------------------------------------------------------------------------
def mean_field(beta, gamma=0.0, p0=1e-6, T=60):
    p = np.empty(T + 1)
    p[0] = p0
    for t in range(T):
        x = p[t]
        x = x + (beta / 2.0) * x * (1 - x) - gamma * x
        p[t + 1] = min(1.0, max(0.0, x))
    return p


# ---------------------------------------------------------------------------
# Agent-based, vectorized; explicit randomized pairwise chat
# ---------------------------------------------------------------------------
def agent_based(N, beta, gamma=0.0, n0=1, T=60, seed=0):
    """N agents, n0 initial carriers. Returns carrier-ratio series length T+1."""
    rng = np.random.default_rng(seed)
    carrier = np.zeros(N, dtype=bool)
    carrier[rng.choice(N, size=n0, replace=False)] = True
    series = np.empty(T + 1)
    series[0] = carrier.mean()
    half = N // 2
    for t in range(T):
        perm = rng.permutation(N)
        q = perm[:half]            # questioners
        a = perm[half:2 * half]    # answerers (paired q[k] <-> a[k])
        # transmission: carrier questioner -> benign answerer w.p. beta
        mask = carrier[q] & (~carrier[a]) & (rng.random(half) < beta)
        new_inf = a[mask]
        # recovery (defense): each current carrier recovers w.p. gamma
        if gamma > 0:
            rec = np.where(carrier & (rng.random(N) < gamma))[0]
            carrier[rec] = False
        carrier[new_inf] = True
        series[t + 1] = carrier.mean()
    return series


def rounds_to(series, thr=0.99):
    """First round index reaching >= thr; -1 if never."""
    idx = np.where(np.asarray(series) >= thr)[0]
    return int(idx[0]) if len(idx) else -1


# ---------------------------------------------------------------------------
# Experiment drivers (used by plots.py / run_all.py)
# ---------------------------------------------------------------------------
def exp_ideal(N=1_000_000, T=45):
    """E1: ideal infectious image (beta=1, gamma=0). Reproduce ~100% fast."""
    mf = mean_field(beta=1.0, gamma=0.0, p0=1.0 / N, T=T)
    ab = agent_based(N=N, beta=1.0, gamma=0.0, n0=1, T=T, seed=1)
    return {"mean_field": mf, "agent_based": ab,
            "rounds_mf": rounds_to(mf, 0.99), "rounds_ab": rounds_to(ab, 0.99),
            "rounds_mf_999": rounds_to(mf, 0.999),
            "rounds_ab_999": rounds_to(ab, 0.999)}


def exp_beta_sweep(N=1_000_000, betas=(1.0, 0.7, 0.4, 0.2, 0.1), T=80):
    """E2: how transmission probability (adversarial budget) sets the speed."""
    out = {}
    for b in betas:
        mf = mean_field(beta=b, gamma=0.0, p0=1.0 / N, T=T)
        out[b] = {"mean_field": mf, "rounds": rounds_to(mf, 0.99)}
    return out


def exp_defense(N=1_000_000, beta=1.0,
                gammas=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8), T=120):
    """E3: defense sweep gamma -> phase transition at beta = 2*gamma."""
    out = {}
    for g in gammas:
        mf = mean_field(beta=beta, gamma=g, p0=0.5, T=T)  # start high to see decay
        pstar = max(0.0, 1 - 2 * g / beta) if beta > 0 else 0.0
        out[g] = {"mean_field": mf, "p_star_theory": pstar,
                  "p_final": float(mf[-1])}
    return out


def exp_failure_alpha(N=1_000_000, beta=1.0, alphas=(1.0, 0.6, 0.3, 0.1), T=45):
    """E4: low alpha -> symptom (harmful) ratio stays low even as carriers ->1.

    Carrier ratio is alpha-independent here (transmission already folded into
    beta); the *symptom* ratio = alpha * carrier_ratio is what the paper calls
    'exhibiting harmful behavior'. This is the paper's stated failure mode.
    """
    carrier = mean_field(beta=beta, gamma=0.0, p0=1.0 / N, T=T)
    out = {"carrier": carrier}
    for a in alphas:
        out[a] = {"symptom": a * carrier,
                  "peak_symptom": float((a * carrier).max())}
    return out


def dump_summary(path="results/dynamics_summary.json"):
    import json, os
    e1 = exp_ideal()
    e3 = exp_defense()
    # rounds-to-99% as a function of the effective growth rate r
    def rounds_for_r(r, N=1_000_000):
        p = [1.0 / N]
        for _ in range(300):
            x = p[-1]; p.append(min(1.0, x + r * x * (1 - x)))
        return rounds_to(p, 0.99)
    band = {f"{r:.2f}": rounds_for_r(r) for r in
            [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]}
    out = {
        "E1_ideal_N1e6": {
            "rounds_mf_99": e1["rounds_mf"], "rounds_ab_99": e1["rounds_ab"],
            "rounds_mf_999": e1["rounds_mf_999"],
            "paper_band": [27, 31],
            "note": "1 partition/round => r=beta/2 <= 0.5 => >=41 rounds; cannot reach 27-31",
        },
        "rounds_vs_effective_rate": band,
        "E3_defense_fixedpoint": {
            f"{g:.2f}": {"p_final": round(d["p_final"], 4),
                         "p_star_theory": round(d["p_star_theory"], 4)}
            for g, d in e3.items()},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    return out


if __name__ == "__main__":
    e1 = exp_ideal()
    print("E1 ideal (beta=1, gamma=0), N=1e6:")
    print(f"  mean-field rounds to 99%   = {e1['rounds_mf']}")
    print(f"  agent-based rounds to 99%  = {e1['rounds_ab']}")
    print(f"  mean-field rounds to 99.9% = {e1['rounds_mf_999']}")
    print(f"  agent-based rounds to 99.9%= {e1['rounds_ab_999']}")
    print(f"  paper reports ~100% in 27-31 rounds")
    e3 = exp_defense()
    print("\nE3 defense (beta=1): p_final vs theory p*=1-2g/b")
    for g, d in e3.items():
        print(f"  gamma={g:.2f}  p_final={d['p_final']:.3f}  "
              f"theory p*={d['p_star_theory']:.3f}")
    dump_summary()
    print("\nwrote results/dynamics_summary.json")
