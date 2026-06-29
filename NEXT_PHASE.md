# Next Phase — RL Training Plan

This document is the team's working plan for the RL portion of the project. It assumes the simulation infrastructure (physics, multi-car, raycasts, collisions, lap timing, Gymnasium env scaffold) is essentially complete, and it focuses **only** on the reinforcement learning side: observation, reward, algorithm, training protocol, evaluation, and the final report.

Every methodology choice in this plan is tied to a specific course lecture (1–14). The mapping is in the **Sources** section at the bottom — when we write the report, the Methodology section will literally cite these slides.

## Completion status (as of 2026-06-23, seed 42)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | RL contract (obs/action/reward) | ✓ complete |
| 2 | PPO baseline on Sprint Circuit | ✓ complete |
| 3 | Reward shaping iteration | ✓ complete (3 seeds) |
| 4 | Domain randomization + held-out eval | ✓ complete (3 seeds, 5M steps each) |
| 5 | Multi-car curriculum (5b→5c→5d) | ✓ seed 42 complete |
| 6 | Self-play opponent pool | ✓ implemented as **Phase 5e** (see note below) |
| 7 | Advanced techniques | 7a/7b/7d tooling implemented; full experiment runs pending |
| 8 | Final evaluation protocol | tooling implemented; full evaluation runs pending |
| 9 | Report + presentation | pending |

> **Phase 5e = Phase 6.** The AlphaZero-style checkpoint pool self-play described in Phase 6 was implemented as the final curriculum stage `5e`, keeping the warm-start chain (5d → 5e) intact. The implementation is identical to the Phase 6 spec: 200k-step snapshots, 10-snapshot rolling pool, physics-driven `PolicyOpponent` with its own raycast sensors. Best model: `models/phase5/v3/seed42/5e_v3_seed42/best_model.zip` (val reward 11,217 at 1.25M/3M steps).

---

---

## 0. Quick orientation (read before starting)

### What is Frenet, and why does it come up?

The "Frenet frame" is a way of describing the car's position **relative to the track centerline** instead of in absolute world coordinates. It gives the agent four numbers: how far along the lap it is (`s`), how far off-center sideways (`e_y`), how rotated relative to the track direction (`e_psi`), and how curved the track is right here (`kappa`), plus a lookahead of future curvatures.

The tradeoff is exactly what you intuited: **Frenet assumes the agent already knows the full track map.** It's like giving the driver a perfect GPS + curvature preview. A raycast-only agent has to figure out the track shape from its "vision" the same way a human driver does on a first lap. Raycast-only is the human-like setup; Frenet+raycast is the "studied the onboards" setup.

We are **defaulting to raycast-only** in this plan. The Frenet code stays in the repo (it's still useful for reward computation — measuring forward progress along the centerline — and for the optional ablation). See the **Team decisions** section at the end.

### Tooling

- **Framework:** Stable-Baselines3 (SB3). The course grades algorithm choice and tuning, not whether we re-implemented PPO. SB3 lets us focus the report on the design choices that actually move the rubric.
- **Compute:** mix of local GPUs + Colab Pro. Train multiple seeds in parallel across machines.
- **Logging:** TensorBoard (built into SB3). Save learning curves as PNG for the report.

---

## Phase 1 — Observation, action, and reward (no training yet)

**Goal:** lock down the RL contract — what the agent sees, what it does, what it gets rewarded for. Get one team member to extend `src/env/racing_env.py` to match this contract. Everything downstream depends on this being right.

### Observation space (raycast-only baseline)

Concatenate per step:

1. **30 normalized raycast distances** — `RayCaster.get_normalized(distances)` → values in `[0, 1]`. (24 forward + 6 mirror, already implemented in `src/sensors/sensor.py`.)
2. **Ego car proprioception** (small, but crucial — the rays don't tell the agent its own speed):
   - normalized speed (`speed / 95.0`)
   - normalized lateral velocity (sideways drift — important so the agent can detect when it's losing grip)
   - last throttle, last steering (so it knows what it just did — helps with smooth control)

Total: **34-dim observation**.

3. **Frame stacking:** wrap the env with SB3's `VecFrameStack(n_stack=4)`. The course explicitly addresses **partial observability**: a single raycast frame doesn't tell the agent whether an opponent car is approaching or receding. Stacking the last 4 frames lets the policy infer relative velocity from the data itself, without us hand-engineering it. → maps to **Lecture 10** (deep RL with stacked frames, the Atari trick).

### Action space (already correct)

Continuous `[throttle, steering]` in `[-1, 1]`. Keep it.

### Reward function (v1 — single-car driving)

```
r_t = + 1.0 * ds                     # forward progress along centerline (meters this step)
      + 0.01 * speed                 # mild speed bonus
      - 0.1  * |e_y| / half_width    # stay near the racing line
      - 0.05 * |e_psi| / pi          # face the right way
      - 0.5  * |Δsteering|           # smoothness penalty (NEW — punishes jittery steering)
      - 10.0 * wall_hit_this_step    # hard penalty per wall contact (use collision_handler)
      - 50.0 * went_off_track        # terminal penalty
```

The `ds` and `e_y`/`e_psi` terms use the **Frenet projection** internally — we use Frenet as a reward signal even though we don't feed it into the observation. That's the "human-like driving but the trainer has a stopwatch" setup.

### Termination conditions

- Off-track (already implemented)
- 3+ wall hits in a row without moving forward (prevents the agent from learning to wedge against a wall)
- Max 6000 steps (~100s)

→ This whole phase corresponds to **Lecture 1** (defining the MDP — states, actions, rewards, transitions). The report's "Problem Formulation" section maps 1:1 to this phase.

---

## Phase 2 — PPO baseline on a single track

**Goal:** confirm the agent can learn to drive at all. If this doesn't work, nothing else will. Use **only the Sprint Circuit**, **single car**, **empty track**.

### Algorithm: PPO (Stable-Baselines3)

PPO is an **Actor-Critic with a Gaussian policy** for continuous control. Two heads on a shared MLP trunk: the Actor outputs `(μ, σ)` for throttle and steering and samples from `N(μ, σ²)`; the Critic outputs a scalar value estimate. → **Lecture 9** (policy-based RL, Gaussian policies for continuous actions, PPO clipped objective acting as a "trust region" via KL constraint).

The clipped objective is the safety leash: it bounds how much the policy can change per update, which prevents the "catastrophic forgetting" failure mode where one bad batch destroys a working policy.

### Starting hyperparameters (SB3 defaults, tweaked for our setup)

```python
PPO(
    "MlpPolicy", env,
    policy_kwargs=dict(net_arch=[256, 256]),
    n_steps=2048,        # rollout length per env
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,     # GAE — this is the n-step / multi-step bias-variance knob
    clip_range=0.2,
    ent_coef=0.01,       # entropy bonus encourages exploration
    learning_rate=3e-4,
    verbose=1,
)
```

### Training plan

- 1M timesteps, 3 random seeds (for variance bars in the report).
- Save model every 100k steps. Render every 50k for sanity-checking.
- Success criterion: completes laps without crashing on the Sprint Circuit.

→ Maps to **Lecture 9** (algorithm) + **Lecture 10** (deep function approximation, learning curves).

---

## Phase 3 — Reward shaping iteration

**Goal:** the v1 reward will produce broken behavior. The report's "Discussion" section will be much stronger if we **document the unintended behaviors and how we fixed them.** This is exactly what the rubric rewards under "Depth of Exploration."

Watch for and fix these classic exploits:

- **Reward hacking via reversing:** if `ds` is signed, the agent might learn to drive backwards through the start line. Fix: clip `ds` to `max(0, ds)` or terminate on backwards-s for too many steps.
- **Wall-scraping for speed bonus:** agent learns it's faster to slide along the wall than to brake for corners. Fix: increase wall-hit penalty, or make the off-track penalty more severe than any speed bonus could compensate for.
- **Spinning in circles:** if entropy bonus is too high, the agent stays exploratory forever. Fix: anneal `ent_coef` over training.
- **Camping:** agent learns that not moving avoids penalties. Fix: small time penalty per step (`-0.001`) to make idleness costly.

Each fix gets a paragraph in the report, with the before/after learning curve. → **Lecture 1 / RL Book** on reward design pitfalls.

---

## Phase 4 — Domain randomization (the generalization story)

**Goal:** train on a *distribution* of tracks so the agent learns "raycast pattern → driving action," not "track #1 → memorized turns." This is the headline contribution: zero-shot transfer to tracks it has never seen.

### Implementation

In `racing_env.reset()`, generate a **new track every episode** using a randomized version of `Track.create_complex_track`:

```python
def random_track(rng):
    n = 300
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = (rng.uniform(150, 500)
              + rng.uniform(50, 200) * np.cos(2*angles + rng.uniform(0, 2*np.pi))
              + rng.uniform(20, 100) * np.sin(rng.choice([3,4,5]) * angles)
              + rng.uniform(10, 60)  * np.cos(rng.choice([5,7,9]) * angles))
    width = rng.uniform(18, 28)
    return Track(np.column_stack([radius*np.cos(angles), radius*np.sin(angles)]), width=width)
```

Hold out a small, fixed set of **evaluation tracks** that are *never* seen during training:
- Sprint Circuit (existing)
- Grand Prix Circuit (existing)
- 3 procedurally-generated tracks with fixed seeds (so eval is reproducible)

### Training

Re-train from scratch with the same PPO config from Phase 2, but `n_envs=8` parallel randomized environments, for **5M timesteps**. SB3's `VecEnv` makes this trivial.

→ Maps to **Lecture 14** ("distribution of related environments" → forces generalized abstraction in the hidden layers).

---

## Phase 5 — Curriculum learning (multi-car ramp-up) ✓ COMPLETE (seed 42)

**Goal:** dropping the agent into a 20-car race from day one will fail. Phase the difficulty.

| Curriculum stage | Setup | Switch criterion |
|---|---|---|
| **5a — Solo driving** | Random tracks, empty | Mean lap completion > 80% over 100 eval episodes |
| **5b — Static obstacle** | + 1 stationary car placed at a random `s` | Mean lap completion > 70% (obstacle present) |
| **5c — Slow opponent** | + 1 car running on centerline at 50% target speed (already implemented as the "static control car") | Mean lap completion > 70% |
| **5d — Multi-opponent** | + 2–3 slow opponents | Mean lap completion > 60% |
| **5e — Self-play** | Opponents are prior snapshots of the agent itself (see Phase 6) | Final stage |

Initialize each stage from the previous stage's weights (`PPO.load(prev_model)`). This is **transfer learning across curriculum stages** — the early skill (driving) carries forward into the harder stage (avoiding).

→ Maps to **Lecture 14** (curriculum / task distributions) + course materials on transfer learning.

---

## Phase 6 — Self-play with opponent pool (the AlphaZero trick) ✓ COMPLETE as Phase 5e (seed 42)

**Goal:** opponents that learn alongside the agent create a non-stationary environment. Training against a *pool* of past versions of yourself stabilizes this.

### Implementation

- Every 200k training steps, save a snapshot of the current policy into a pool (cap at 10 snapshots).
- At episode reset, sample opponents uniformly from the pool.
- This forces the agent to be robust against a *range* of skill levels, not just the current opponent.

This is the AlphaGo/AlphaZero methodology — explicitly covered in the course's deep RL section. → **Lecture 10 / 14**.

---

## Phase 7 — Advanced techniques (grade maximizers, pick 1–2)

These are the "depth of exploration" amplifiers. Each one is a separate self-contained experiment with its own ablation. **Pick at least one. Two would be ideal. Three is overkill given the report length cap.**

### Implementation status

Phase 7a/7b/7d are wired into the training/evaluation tooling:

- `train.py --aux-raycast-prediction` uses a PPO policy with an auxiliary
  next-raycast prediction head and trains it from rollout transitions.
- `train.py --gae-lambda <value>` runs PPO with a chosen GAE / n-step lambda.
- `train.py --vec-normalize-reward` enables SB3 `VecNormalize` reward normalization and saves `<run>_final_vecnormalize.pkl`.
- `evaluate.py --vec-normalize <stats.pkl>` loads saved normalization stats for evaluation.
- `phase7_ablation.py` builds or executes the full GAE sweep manifest, with optional 7a/7d flags.

Suggested command for the Phase 7b sweep:

```bash
.venv/bin/python phase7_ablation.py \
  --gae-values 0.0 0.5 0.9 0.95 1.0 \
  --seeds 42 43 44 \
  --track-mode random \
  --reward-profile v2 \
  --timesteps 5000000 \
  --n-envs 8 \
  --manifest results/phase7/phase7_gae_manifest.json
```

Add `--aux-raycast-prediction` to include Phase 7a, and add
`--vec-normalize-reward` to combine the sweep with Phase 7d. Add `--execute`
only when you are ready to launch the training jobs.

### 7a — Auxiliary tasks (Lecture 14 — the "secret weapon")

Implemented as an opt-in PPO extension:

- `AuxRaycastActorCriticPolicy` adds a second prediction head to the policy network that predicts **the next raycast vector** given the current observation + action.
- `AuxRaycastPredictionCallback` trains that head from consecutive rollout frames with loss coefficient `--aux-loss-coef`.
- Use `train.py --aux-raycast-prediction` for a direct run, or add that flag to `phase7_ablation.py` to include it in the GAE sweep.

**Why it helps:** the policy representation is forced to learn features that are useful for *predicting the world*, not just for the current reward. This yields more transferable representations and should improve zero-shot performance on unseen tracks. The course calls this "general value functions" / "auxiliary tasks." The implementation stays SB3-compatible through a custom policy and rollout callback.

→ **Lecture 14** (auxiliary tasks, GVFs).

### 7b — n-step returns (Lecture 5 & 10)

PPO/GAE already does this implicitly via `gae_lambda`. But we can make it explicit: ablate `gae_lambda ∈ {0.0, 0.5, 0.9, 0.95, 1.0}` and discuss the **bias-variance tradeoff**. Show that pure 1-step (λ=0) is biased and slow; pure Monte Carlo (λ=1) is unbiased but high variance.

This gives us a tight, defensible paragraph in the report: "We chose λ=0.95 after the ablation in Fig. X, balancing bias and variance as discussed in Lecture 10."

→ **Lecture 5** (n-step prediction) + **Lecture 10** (deadly triad, bias-variance).

### 7d — Adaptive target normalization (Lecture 14)

Reward magnitudes shift dramatically across curriculum stages and randomized tracks. Maintain a running mean/variance of returns and normalize the critic's targets. SB3 has `normalize_advantage=True` (default) for PPO; use `VecNormalize` to normalize observations and rewards.

→ **Lecture 14** (adaptive normalization for shifting value scales).

---

## Phase 8 — Evaluation protocol

**Goal:** rigorous, reproducible numbers for the report.

### Metrics (computed on the held-out track set)

- **Zero-shot success rate:** % of episodes where the agent completes a full lap on a track it never trained on. Headline number.
- **Mean lap time** on each held-out track.
- **Wall hit count per lap** (already tracked in `collision_handler`).
- **Car-collision count per lap** (multi-agent stages).
- **Smoothness:** mean `|Δsteering|` per step (lower = more human-like).
- **Trajectory plots:** overlay the agent's path over the track for the report.

### Procedure

For each model (Phase-2 baseline, Phase-4 domain-randomized, Phase-5 multi-agent, Phase-7 advanced techniques):

- Freeze the weights.
- Run 100 evaluation episodes per held-out track (5 tracks × 100 = 500 episodes per model).
- Compute mean ± std for each metric.
- Save trajectory PNGs.

### Required figures for the report

1. Learning curves: episode return vs timesteps with seed shaded bands.
2. Ablation bar chart: Phase-2 vs Phase-4 vs Phase-7 zero-shot success rates.
3. Reward-shaping iteration: before/after curves for each fix in Phase 3.
4. Trajectory overlay: agent path on an unseen track.
5. Raycast heatmap: average ray distance vs track curvature (sanity check that the agent is using its sensors).

### Implementation status

Phase 9 is wired into the tooling:

- The env now tracks **lap time** (`lap_times`, `mean_lap_time` in `info`,
  derived from per-lap completion step marks × the 1/60 s physics timestep) and
  exposes `car_x`/`car_y`, `kappa`, and `ray_mean_distance` for trajectory and
  raycast analysis.
- `evaluate.py` aggregates lap time across episodes (`aggregate.lap_time`) and
  accepts `--record-trajectories N` to dump the first N car paths per track to a
  `<output>.trajectories.json` sidecar.
- `phase9_figures.py` renders the report figures from those artifacts via
  subcommands: `trajectory` (path overlay on a held-out track), `success`
  (zero-shot lap-success bar chart across model groups), `curves` (learning
  curves with seed bands from `evaluations.npz`), and `raycast` (mean ray
  distance vs absolute curvature heatmap). Matplotlib runs headless (`Agg`).

Example figure commands:

```bash
# Held-out evaluation with trajectory capture (one path per track)
.venv/bin/python evaluate.py models/phase4/v2/seed42/phase4_v2_seed42/best_model.zip \
  --tracks held-out --episodes 100 --record-trajectories 1 \
  --output results/phase9/phase4_v2_seed42_heldout.json

# Trajectory overlay on an unseen track
.venv/bin/python phase9_figures.py trajectory \
  results/phase9/phase4_v2_seed42_heldout.trajectories.json \
  --track grand-prix --out results/phase9/figures/traj_grand-prix.png

# Zero-shot success bar chart (Phase 2 vs Phase 4)
.venv/bin/python phase9_figures.py success \
  --group "Phase 4" results/phase4/phase4_v2_seed4*_heldout_final.json \
  --out results/phase9/figures/success.png

# Learning curves with seed bands (PPO)
.venv/bin/python phase9_figures.py curves \
  --group PPO logs/phase4/v2/seed4*/phase4_v2_seed4*/evaluations.npz \
  --out results/phase9/figures/curves.png

# Raycast-vs-curvature heatmap
.venv/bin/python phase9_figures.py raycast \
  models/phase4/v2/seed42/phase4_v2_seed42/best_model.zip \
  --tracks held-out --episodes 5 --out results/phase9/figures/raycast.png
```

All figures are produced from the existing PPO artifacts.

---

## Phase 9 — Report + presentation

The 8–20 page report is structured exactly as the rubric dictates:

1. **Project Overview** (~1 page) — F1 RL motivation, what we built, what we showed.
2. **Problem Formulation** (~2 pages) — pulled directly from Phase 1: state/action/reward formal definitions, transition dynamics (cite Box2D + the F1 physics in `car.py`).
3. **Methodology** (~5–7 pages) — algorithm choice (PPO) with Lecture-9 citations, curriculum design with Lecture-14 citations, GAE ablation with Lecture-5/10 citations, hyperparameter tuning narrative.
4. **Results** (~4–5 pages) — all five figures above + tables.
5. **Discussion** (~2 pages) — reward-shaping war stories from Phase 3, limitations, what didn't work, what we'd do with more compute.
6. **Conclusion** (~1 page).

### Presentation (15 min, 10% of grade)

- 5 min: motivation + problem formulation (animated track + raycasts).
- 5 min: methodology highlights (the curriculum diagram, the auxiliary task figure).
- 3 min: **video demo** — record the final agent battling 2 opponents through the Grand Prix esses section, with raycast colors visible in real time. This is the "showed me the policy is actually learning" moment.
- 2 min: results + Q&A buffer.

---

## Team decisions — discuss before starting Phase 1

These are the open questions where reasonable people will disagree. Decide as a team and commit:

1. **Frenet in observation or not?** Default in this plan is **raycast-only** (human-like). The alternative is to run all of Phase 4 onward as a **three-way ablation** (raycast-only vs Frenet-only vs combined) — this is a massive grade-maximizer under "Depth of Exploration" but costs ~3× the training compute and ~3× the writeup effort. **Pros of raycast-only:** clean story, novel framing, less compute, more "the agent figured the track out itself." **Pros of ablation:** report has a strong empirical claim about what perception modality matters most; very hard to argue with three trained models and a table. **Frenet still lives in the codebase either way** because we use `e_y`/`e_psi`/`ds` for the reward signal.

2. **Which advanced technique from Phase 7?** Pick 1–2:
   - **Auxiliary tasks (7a)** — highest expected grade impact, most aligned with Lecture 14, hardest to implement (requires stepping outside SB3).
   - **n-step / GAE ablation (7b)** — easiest, tight Lecture 5/10 connection, mostly a hyperparam sweep.
   - **Adaptive normalization (7d)** — almost free if we use `VecNormalize`, modest theoretical payoff.

3. **Compute budget per phase.** With Colab Pro unlimited + local GPUs, this is less constrained, but we should still agree: e.g., "Phase 4: 5M steps × 3 seeds × 2 algorithms = 30M total transitions." Estimate wall-clock and divide across machines.

5. **Reward weights.** The Phase-1 v1 weights are a starting guess. Plan a half-day calibration session after Phase 2 finishes to retune based on observed failure modes.

6. **Single ego car or symmetric self-play?** In self-play (Phase 6), do we train one network and clone it for opponents, or train multiple agents simultaneously (MARL)? Single-network self-play is much simpler and is what AlphaZero does. MARL is harder but more academically interesting — only do this if time allows after the rest is done.

---

## Sources — which lecture each choice came from

This maps every methodological choice in the plan back to the course material, so the report's citations are pre-traced. Format: `[Lecture X.pdf]`.

| Choice | Lecture | What the lecture covers |
|---|---|---|
| MDP formulation (states/actions/rewards/transitions) — Phase 1 | **Lecture 1** | Foundations: defining a problem as an MDP |
| Reward shaping pitfalls + agent exploiting subgoals — Phase 3 | **Lecture 1 / RL textbook** | Reward design, why dense rewards help and how they backfire |
| n-step returns, multi-step prediction — Phase 7b | **Lecture 5** | TD vs Monte Carlo, n-step targets |
| Experience replay framed as Dyna (model-based view) — Phase 7c | **Lecture 8** | Integrating learning and planning; replay as a non-parametric world model |
| Actor-Critic for continuous actions — Phase 2 | **Lecture 9** | Policy-based RL, why value-based fails for continuous |
| Gaussian policy `N(μ, σ²)` over continuous actions — Phase 2 | **Lecture 9** | Defining the policy distribution; parameterizing μ and σ with a NN |
| PPO clipped objective ("trust region") — Phase 2 | **Lecture 9** | KL-bounded policy updates to prevent catastrophic forgetting |
| Deep function approximation, frame stacking — Phase 1 obs, Phase 2 net | **Lecture 10** | DQN-era tricks: stacked frames for partial observability, replay for decorrelation |
| Bias-variance tradeoff, deadly triad — Phase 7b discussion | **Lecture 10** | Why n-step / GAE helps stabilize deep RL |
| Distribution of environments → generalization — Phase 4 | **Lecture 14** | Training on related task distributions instead of one fixed task |
| Auxiliary tasks / General Value Functions — Phase 7a | **Lecture 14** | Predicting extra targets to shape shared representations |
| Adaptive target normalization — Phase 7d | **Lecture 14** | Handling shifting reward/value scales online |
| Self-play opponent pool (AlphaZero-style) — Phase 6 | **Lecture 10 / 14** | Stabilizing non-stationary multi-agent training |
| Curriculum learning across difficulty stages — Phase 5 | **Lecture 14** | Easier-then-harder task ordering, transfer between stages |

The corresponding lectures should be re-read by whoever owns each phase, and the citations belong in the Methodology section of the final report.
