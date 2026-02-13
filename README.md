# Senior Design — Variational Adversarial Imitation Learning (VAIRL)


**What this demonstrates**
- Applying research ideas (AIRL, VAIRL/VAIL) in a small, testable environment.
- Building end‑to‑end training loops with meaningful metrics and visual checks.
- Debugging and integration with a real ML library stack (`imitation`, `stable-baselines3`).
- Keeping experiments interpretable with clear structure and logging.

**Core idea**
Adversarial Imitation Learning trains a policy against a discriminator that distinguishes expert behavior from generated behavior. VAIRL adds a **variational information bottleneck** to the discriminator to improve generalization by limiting how much state information can pass through the discriminator.

## Repo Structure
- `2d-vairl-toy/vairl.py` — baseline VAIL/VDB-style imitation learning, custom training loop.
- `2d-vairl-toy/vairl2.py` — AIRL/VAIRL‑style discriminator with hand‑rolled training.
- `2d-vairl-toy/vairl3.py` — AIRL base class from `imitation` + VAIRL reward network + same metrics/outputs as `vairl2.py`.
- `2d-vairl-toy/lets-do-it-vairl.py` — single‑file VAIL experiment with plots.
- `2d-vairl-toy/run.py` — minimal AIRL‑style experiment for intuition.
- `2d-vairl-toy/metrics.py` — logging + CSV + plotting.
- `2d-vairl-toy/display.py` — expert vs generator visualization.

## Quickstart
Create a venv and install dependencies from the project requirements:

```bash
python -m venv 2d-vairl-toy/venv
2d-vairl-toy/venv/bin/pip install -r 2d-vairl-toy/requirements.txt
```

If you plan to run `vairl3.py` (AIRL via `imitation`), make sure SB3 versions are compatible:

```bash
2d-vairl-toy/venv/bin/pip install -U "stable-baselines3==2.7.1" "sb3-contrib==2.7.1"
```

Run the main experiment (VAIRL on the 2D grid world):

```bash
2d-vairl-toy/venv/bin/python 2d-vairl-toy/vairl3.py
```

You will be prompted to choose the expert path type:
- `s` → S‑curve
- `c` → C‑curve (with wall enabled in environment)
- anything else → straight line

## Experiment Outputs
The training loop prints per‑episode metrics and periodically writes:
- `2d-vairl-toy/metrics_out/training_metrics.csv`
- `2d-vairl-toy/metrics_out/plots/*.png`

It also visualizes expert vs generator trajectories at intervals.

## How `vairl3.py` Differs from `vairl2.py`
- **Uses `imitation` AIRL base class**: reward wrapping, buffering, and generator training are handled by the library.
- **VAIRL reward network**: discriminator implements
  `f(s,a,s') = g(z_g(s,a)) + γ h(z_h(s')) − h(z_h(s))`
  with variational encoders and KL penalty.
- **Same terminal outputs and metrics** as `vairl2.py`, to keep comparisons simple.
- **Expert trajectories** are generated analytically (same as `vairl2.py`) to ensure consistent curves and goal-reaching behavior.

## AI Experiment Design Choices
- **Small, interpretable env**: 2D point mass with a goal and optional wall.
- **Deterministic baseline + stochastic expert**: expert data is nearly deterministic but technically stochastic to satisfy AIRL assumptions.
- **Metrics first**: training stats + reward summaries + trajectory plots are prioritized for fast iteration.

## How I Use AI in My Role (What “AI Experimentation” Means Here)
- I use AI tools (LLMs, copilots, assistants) to **accelerate engineering work**: drafting, refactoring, testing, and documentation.\n+- I treat AI outputs as **starting points**, then validate with measurements, constraints, and code review.\n+- I run **small, controlled experiments** to evaluate AI‑assisted approaches before adopting them broadly.\n+- I focus on **repeatability and debuggability**, so AI‑assisted work remains production‑grade and auditable.\n+- When applicable, I also explore ML models directly—this repo is one such example.

## Notes
- AIRL assumes a **stochastic expert policy**. In this repo, the expert is almost deterministic with tiny noise (`EXPERT_NOISE_STD`) to satisfy that assumption.
- The environment can terminate early if the goal is reached, so AIRL is run with `allow_variable_horizon=True`.

## Questions or Demos
If you want a quick walkthrough, start with:
- `2d-vairl-toy/vairl3.py` for the full AIRL‑based VAIRL implementation
- `2d-vairl-toy/vairl2.py` for the simpler hand‑rolled version
