#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plots.py — Generate reproduction figures into figs/.

Reads results/measure.json and results/realsim.json if present; always produces
the analytic dynamics figures from dynamics.py.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dynamics as dyn
from realrun import PAYLOAD_ORDER

FIG = "figs"
os.makedirs(FIG, exist_ok=True)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Fig 1: infection dynamics at scale + the rounds-to-100% metric mismatch
# ---------------------------------------------------------------------------
def fig_dynamics():
    N = 1_000_000
    T = 50
    e1 = dyn.exp_ideal(N=N, T=T)
    # factor-2 variant (each agent asks AND answers each round)
    def mf2(beta, p0, T):
        p = [p0]
        for _ in range(T):
            x = p[-1]; x = x + beta * x * (1 - x); p.append(min(1, x))
        return np.array(p)
    f2_b1 = mf2(1.0, 1.0 / N, T)
    f2_b075 = mf2(0.75, 1.0 / N, T)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    t = np.arange(T + 1)
    ax[0].plot(t, e1["mean_field"], "-", lw=2, label="mean-field, 1 partition/round (r=β/2), β=1")
    ax[0].plot(np.arange(len(e1["agent_based"])), e1["agent_based"], "o", ms=3,
               label="agent-based MC, N=1e6 (random pairing)")
    ax[0].plot(t, f2_b1, "--", lw=2, label="2 contacts/round (r=β), β=1")
    ax[0].plot(t, f2_b075, ":", lw=2, label="2 contacts/round, β=0.75")
    ax[0].axvspan(27, 31, color="green", alpha=0.15, label="paper: 27–31 rounds")
    ax[0].set_xlabel("chat round t"); ax[0].set_ylabel("carrier (infection) ratio")
    ax[0].set_title("Infection dynamics, N=1,000,000")
    ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=0.3)

    # rounds-to-99% as a function of effective rate r
    rs = np.arange(0.4, 1.01, 0.02)
    rr = []
    for r in rs:
        p = [1.0 / N]
        for _ in range(200):
            x = p[-1]; p.append(min(1, x + r * x * (1 - x)))
        rr.append(dyn.rounds_to(p, 0.99))
    ax[1].plot(rs, rr, "-o", ms=3)
    ax[1].axhspan(27, 31, color="green", alpha=0.15, label="paper band 27–31")
    ax[1].axvline(0.5, color="red", ls="--", lw=1,
                  label="max for 1 partition/round (β=1)")
    ax[1].set_xlabel("effective growth rate r"); ax[1].set_ylabel("rounds to 99%")
    ax[1].set_title("Rounds-to-100% is highly assumption-sensitive")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_dynamics.png", dpi=130)
    print("wrote fig1_dynamics.png")


# ---------------------------------------------------------------------------
# Fig 2: defense phase transition (beta vs 2*gamma)
# ---------------------------------------------------------------------------
def fig_defense():
    e3 = dyn.exp_defense(beta=1.0, gammas=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8), T=120)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for g, d in e3.items():
        ax[0].plot(d["mean_field"], label=f"γ={g:.2f} (p*={d['p_star_theory']:.2f})")
    ax[0].set_xlabel("chat round t"); ax[0].set_ylabel("carrier ratio")
    ax[0].set_title("Defense (recovery γ), β=1; dies out when β<2γ ⇔ γ>0.5")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    gammas = np.linspace(0, 0.9, 19)
    sim = [dyn.mean_field(1.0, g, 0.5, 200)[-1] for g in gammas]
    th = [max(0, 1 - 2 * g / 1.0) for g in gammas]
    ax[1].plot(gammas, sim, "o", label="simulated p_final")
    ax[1].plot(gammas, th, "-", label="theory p*=1−2γ/β")
    ax[1].axvline(0.5, color="red", ls="--", label="threshold γ=β/2")
    ax[1].set_xlabel("recovery rate γ"); ax[1].set_ylabel("endemic carrier ratio")
    ax[1].set_title("Phase transition reproduced exactly")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_defense.png", dpi=130)
    print("wrote fig2_defense.png")


# ---------------------------------------------------------------------------
# Fig 3: measured alpha/beta spectrum on the real model (failure cases)
# ---------------------------------------------------------------------------
def fig_measure():
    m = load("results/measure.json")
    if not m:
        print("skip fig3 (no measure.json)"); return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, dname in zip(axes, ["undefended", "defended"]):
        block = m["defenses"][dname]["conditions"]
        labels = [k for k in PAYLOAD_ORDER if k in block]
        aA = [block[k]["alphaA"] for k in labels]
        aQ = [block[k]["alphaQ"] for k in labels]
        be = [block[k]["beta"] for k in labels]
        x = np.arange(len(labels)); w = 0.27
        ax.bar(x - w, aA, w, label="α^A (answerer symptom)")
        ax.bar(x, aQ, w, label="α^Q (questioner symptom)")
        ax.bar(x + w, be, w, label="β (transmission)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(f"defense = {dname}")
        ax.axhline(m["defenses"][dname]["control"]["fp_symptom"], color="gray",
                   ls=":", label="control FP")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Measured α/β on DeepSeek-V3.2 across payload budgets")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_measure.png", dpi=130)
    print("wrote fig3_measure.png")


# ---------------------------------------------------------------------------
# Fig 4: low-alpha => harmful (symptom) ratio fails to reach 100%
# ---------------------------------------------------------------------------
def fig_failure_alpha():
    m = load("results/measure.json")
    e4 = dyn.exp_failure_alpha(beta=1.0, alphas=(1.0, 0.6, 0.3, 0.1), T=50)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(e4["carrier"], "k-", lw=2, label="carrier ratio (virus in album)")
    for a in (1.0, 0.6, 0.3, 0.1):
        ax[0].plot(e4[a]["symptom"], "--",
                   label=f"symptom ratio, α={a} (peak {e4[a]['peak_symptom']:.2f})")
    ax[0].set_xlabel("chat round t"); ax[0].set_ylabel("ratio")
    ax[0].set_title("Low α ⇒ harmful-behavior ratio capped at α\n(paper's stated failure mode)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # overlay the measured alphas as horizontal caps, if available
    if m:
        und = m["defenses"]["undefended"]["conditions"]
        names = [k for k in PAYLOAD_ORDER if k in und]
        caps = [min(und[k]["alphaA"], und[k]["alphaQ"]) for k in names]
        ax[1].bar(range(len(names)), caps)
        ax[1].set_xticks(range(len(names)))
        ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax[1].set_ylim(0, 1.05)
        ax[1].set_title("Measured effective α cap = min(α^A, α^Q)\nundefended DeepSeek-V3.2")
        ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_failure_alpha.png", dpi=130)
    print("wrote fig4_failure_alpha.png")


# ---------------------------------------------------------------------------
# Fig 5: real multi-round runs (DeepSeek) + round-dependence of alpha/beta
# ---------------------------------------------------------------------------
def fig_realsim():
    r = load("results/realsim.json")
    if not r:
        print("skip fig5 (no realsim.json)"); return
    runs = r["runs"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for tag, s in runs.items():
        t = np.arange(len(s["carrier"]))
        ax[0].plot(t, s["carrier"], "-o", ms=3, label=f"{tag} carrier")
        ax[0].plot(t, s["symptom"], "--", alpha=0.7, label=f"{tag} symptom")
    ax[0].set_xlabel("chat round t"); ax[0].set_ylabel("ratio")
    ax[0].set_title(f"Real DeepSeek run (N={r['N']})")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    for tag, s in runs.items():
        t = np.arange(len(s["beta"]))
        bb = [np.nan if v is None else v for v in s["beta"]]
        ax[1].plot(t, bb, "-o", ms=3, label=f"{tag} β_t")
    ax[1].set_xlabel("chat round t"); ax[1].set_ylabel("β_t")
    ax[1].set_title("Round-dependence of transmission β_t\n(paper assumes ~constant)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3); ax[1].set_ylim(-0.05, 1.05)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_realsim.png", dpi=130)
    print("wrote fig5_realsim.png")


if __name__ == "__main__":
    fig_dynamics()
    fig_defense()
    fig_measure()
    fig_failure_alpha()
    fig_realsim()
    print("done.")
