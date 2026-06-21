# Reproducing *Agent Smith* (arXiv:2402.08567) with a real LLM backend

This directory reproduces the **infectious-jailbreak dynamics** of

> Gu, Zheng, Pang et al., *Agent Smith: A Single Image Can Jailbreak One Million
> Multimodal LLM Agents Exponentially Fast*, ICML 2024 (arXiv:2402.08567).

against a **real** model — `Pro/deepseek-ai/DeepSeek-V3.2-Exp` served by
SiliconFlow — instead of the paper's LLaVA-1.5.

## What the paper claims (the targets we try to hit)

1. **Mechanism.** Agents = MLLM + RAG + text history + image album. Randomized
   pairwise chat: each round split N agents into N/2 questioners / N/2 answerers;
   a questioner retrieves an image, asks about it; the image lands in the
   *answerer's* album → that is the propagation route.
2. **Exponential spread.** One infectious adversarial image, dropped into one
   agent at round 0, drives the carrier ratio to ~100% in **27–31 rounds** for
   **N = 1,000,000**, with infected agents exhibiting harmful behavior.
3. **Dynamics / theory.** SIS-type model with effective infection rate `β/2`
   (the 1/2 from random questioner/answerer roles) and recovery `γ`:
   `p_{t+1} = p_t + (β/2)·p_t(1−p_t) − γ·p_t`, fixed point `p* = 1 − 2γ/β`.
4. **Defense principle.** Infection provably dies out iff `β < 2γ`.
5. **Failure cases.** "Most failure cases are attributed to **low α**" (the
   probability that a virus-carrying agent actually exhibits the symptom).

## How this reproduction maps onto it (and where it must deviate)

| Paper | Here | Why |
|---|---|---|
| LLaVA-1.5 (vision) | DeepSeek-V3.2 (text) | the provided API is text-only |
| gradient-optimized adversarial **image** | **text payload** in an "image caption" | no vision encoder / no white-box gradients |
| harmful target string | benign **marker token** (`AGENT-SMITH-42`) | measure identical α/β safely, no harmful content |
| 1e6 real agents | 1e6 **simulated** agents w/ measured α,β | cannot run 1e6 real chats |

So we reproduce in **three layers**:

- **Layer 1 — real measurement** (`measure.py`): estimate `α^A, α^Q, β` on
  DeepSeek across a spectrum of payload "budgets" and a defense prompt.
- **Layer 2 — real micro-sim** (`realsim.py`): a true multi-round randomized
  pairwise chat over N≈24 DeepSeek agents; tracks the round-wise curves.
- **Layer 3 — scale dynamics** (`dynamics.py`): plug measured α,β into the SIS
  mean-field + a vectorized agent-based Monte-Carlo up to N=1e6.

## Files

- `realrun.py` — backbone client (SiliconFlow/DeepSeek), Agent, album/RAG,
  pairwise-chat protocol, payload spectrum, defense prompt.
- `measure.py` — Layer 1; writes `results/measure.json`.
- `realsim.py` — Layer 2; writes `results/realsim.json`.
- `dynamics.py` — Layer 3; SIS mean-field + agent-based MC.
- `plots.py` — figures into `figs/`.
- `FINDINGS.md` — **the failure-case analysis** (what does not reproduce).

## Run

```bash
pip install openai numpy matplotlib
export SILICONFLOW_API_KEY=sk-...           # SiliconFlow key
python measure.py        # ~5-6 min, real API
python realsim.py        # ~5-8 min, real API
python dynamics.py       # instant, no API
python plots.py          # figures
```

Env knobs: `SF_MODEL`, `SF_SAMPLES`, `SF_N`, `SF_T`, `SF_WORKERS`,
`SF_BACKEND=mock` (offline plumbing test).
