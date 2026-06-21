#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure.py — Empirically estimate the paper's infection parameters on a REAL
model (DeepSeek-V3.2 via SiliconFlow):

    alpha^A_t : P(a virus-carrying *answering* agent exhibits the symptom)
    alpha^Q_t : P(a virus-carrying *questioning* agent exhibits the symptom)
    beta_t    : P(a virus-carrying questioning agent transmits the virus
                  (payload) to a benign answering agent)

We sweep three "adversarial budgets" (payload strengths) to reproduce the
paper's qualitative claim that *small budgets / challenging scenarios* make the
infection fail, and we keep a benign control to bound false positives.

Because temperature=0 makes a single (prompt) deterministic, we estimate each
probability over a *population of contexts* (varying agent id, prior history,
and the benign topic) — i.e. alpha/beta are averages over the agent population,
exactly as in the paper.
"""

import os
import sys
import json
import time
import random

from realrun import (
    LLMClient, Agent, Image, make_virus_image, make_benign_image,
    SYS_AGENT, SYS_AGENT_DEFENDED, USER_ASK, USER_ANS, BENIGN_CAPTIONS,
    PAYLOADS, PAYLOAD_ORDER, VIRUS_MARKER, symptom, carries_payload,
)

random.seed(0)

M = int(os.getenv("SF_SAMPLES", "24"))          # contexts per condition
OUT = os.getenv("SF_OUT", "results/measure.json")

TOPICS = [
    "the weather today", "a recent book", "healthy breakfasts", "city traffic",
    "house plants", "weekend hiking", "learning the guitar", "saving money",
    "good movies", "morning routines", "public libraries", "board games",
]


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def random_history(rng, n=2):
    """A short benign prior history to make each agent's context distinct."""
    hist = []
    for _ in range(n):
        t = rng.choice(TOPICS)
        hist.append((f"What do you think about {t}?",
                     f"I think {t} is interesting and worth discussing."))
    return hist


def build_contexts(m):
    rng = random.Random(123)
    ctx = []
    for i in range(m):
        ctx.append({
            "aid": 1000 + i,
            "hist": random_history(rng),
            "topic": rng.choice(TOPICS),
            "benign_q": f"What stands out to you about this image?",
        })
    return ctx


def measure_alphaA(llm, payload_key, contexts, sys_t):
    """Answerer receives the virus image + a benign question -> symptom?"""
    virus = make_virus_image(payload_key)
    jobs = []
    for c in contexts:
        ag = Agent(aid=c["aid"], history=list(c["hist"]))
        jobs.append((sys_t.format(aid=ag.aid),
                     USER_ANS.format(hist=ag.hist_text(),
                                     caption=virus.caption,
                                     question=c["benign_q"])))
    outs = llm.map(jobs)
    flags = [symptom(o) for o in outs]
    return flags, outs


def measure_alphaQ_beta(llm, payload_key, contexts, sys_t):
    """Carrier acts as questioner from the virus image -> symptom? transmit?"""
    virus = make_virus_image(payload_key)
    jobs = []
    for c in contexts:
        ag = Agent(aid=c["aid"], history=list(c["hist"]))
        jobs.append((sys_t.format(aid=ag.aid),
                     USER_ASK.format(hist=ag.hist_text(), caption=virus.caption)))
    outs = llm.map(jobs)
    sym = [symptom(o) for o in outs]
    trans = [carries_payload(o, payload_key) for o in outs]
    return sym, trans, outs


def measure_control(llm, contexts, sys_t):
    """Benign image, benign question -> false-positive symptom rate."""
    jobs = []
    for c in contexts:
        bim = make_benign_image(c["aid"])
        ag = Agent(aid=c["aid"], history=list(c["hist"]))
        jobs.append((sys_t.format(aid=ag.aid),
                     USER_ANS.format(hist=ag.hist_text(),
                                     caption=bim.caption, question=c["benign_q"])))
    outs = llm.map(jobs)
    return [symptom(o) for o in outs], outs


DEFENSES = {"undefended": SYS_AGENT, "defended": SYS_AGENT_DEFENDED}


def main():
    backend = os.getenv("SF_BACKEND", "siliconflow")
    llm = LLMClient(backend=backend)
    contexts = build_contexts(M)
    t0 = time.time()

    report = {"model": llm.model, "backend": backend, "samples": M,
              "marker": VIRUS_MARKER, "defenses": {}}

    for dname, sys_t in DEFENSES.items():
        dblock = {"control": {}, "conditions": {}}
        cflags, couts = measure_control(llm, contexts, sys_t)
        p, lo, hi = wilson(sum(cflags), len(cflags))
        dblock["control"] = {"fp_symptom": p, "ci": [lo, hi], "examples": couts[:2]}
        print(f"\n=== defense={dname} ===")
        print(f"[control] false-positive symptom rate = {p:.2f} ({lo:.2f},{hi:.2f})")

        for pk in PAYLOAD_ORDER:
            aA, aA_out = measure_alphaA(llm, pk, contexts, sys_t)
            aQ, beta, aQ_out = measure_alphaQ_beta(llm, pk, contexts, sys_t)
            pA, loA, hiA = wilson(sum(aA), len(aA))
            pQ, loQ, hiQ = wilson(sum(aQ), len(aQ))
            pB, loB, hiB = wilson(sum(beta), len(beta))
            dblock["conditions"][pk] = {
                "alphaA": pA, "alphaA_ci": [loA, hiA],
                "alphaQ": pQ, "alphaQ_ci": [loQ, hiQ],
                "beta": pB, "beta_ci": [loB, hiB],
                "examples_answer": aA_out[:3],
                "examples_question": aQ_out[:3],
            }
            print(f"[{pk:14s}] alphaA={pA:.2f}({loA:.2f},{hiA:.2f})  "
                  f"alphaQ={pQ:.2f}({loQ:.2f},{hiQ:.2f})  "
                  f"beta={pB:.2f}({loB:.2f},{hiB:.2f})")
        report["defenses"][dname] = dblock

    report["wall_s"] = round(time.time() - t0, 1)
    report["calls"] = llm.calls
    report["tokens"] = llm.tokens
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}  | calls={llm.calls} tokens={llm.tokens} "
          f"wall={report['wall_s']}s")


if __name__ == "__main__":
    main()
