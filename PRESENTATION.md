# Presentation — RL Racing Project

Working plan for the 15-minute academic presentation that closes out Phases 1–4 of the project. Iterates from `NEXT_PHASE.md` (the methodology plan) and `CLAUDE.md` (the verified results).

## Narrative spine

> We built a human-like racing agent (rays only, no GPS), discovered it cheated by reversing for reward, fixed it, then showed it generalizes 100% to tracks it has never seen.

Three rubric-friendly beats: clean MDP formulation (Lecture 1), a documented reward-shaping failure with a fix (the "Depth of Exploration" gold), and a strong empirical generalization claim (Lecture 14).

## Format

- **Total length:** 15:00 (target ~14:55, ~5s buffer inside the cap)
- **Speakers:** 3 (segmented into contiguous blocks — no mid-slide handoffs)
- **Audience:** Mixed — CS grads, not all RL. PPO primer stays in.
- **Demos:** Pre-recorded video embedded in slides (no live demo risk).

## Slide-by-slide timeline

| # | Slide | Time | Purpose |
|---|---|---|---|
| 1 | Title + headline | 30s | Project, three names, one-line result: "PPO agent achieves 100% lap success on 5 unseen tracks." |
| 2 | The problem & framing | 70s | "Can an agent learn to race using only on-board sensors?" Car + 30-ray viz. Raycast-only = human-like; Frenet = GPS. |
| 3 | MDP formulation | 70s | State (34-dim) / Action / Reward — three-column compact slide. **Lecture 1.** |
| 4 | Algorithm: PPO | 70s | Actor-Critic, Gaussian policy on continuous actions, clipped objective as a trust region. **Lecture 9.** |
| 5 | Frame stacking & partial observability | 60s | `VecFrameStack(n_stack=4)` — single frame doesn't tell the agent its own velocity; stacking lets the policy infer motion. **Lecture 10.** |
| 6 | Phase 2 baseline | 50s | Sprint Circuit, 1M steps, 3 seeds, PPO. Curve goes up, agent moves — but lap success was 0%. Cliffhanger into slide 7. |
| 7 | Reward hacking war story ⭐ | 150s | v1 agent learned to **reverse** through the start/finish line for positive `ds` reward. **Play 10s v1 reversing gag clip.** Before/after table: v1 0% / 29 wall hits → v2 100% / 0.5 wall hits. List three v2 fixes (progress/speed zeroed during wall contact, time penalty, reverse termination). **Lecture 1 + RL Book.** |
| 8 | Phase 4 — domain randomization | 90s | 4 sampled procedural tracks in a 2×2 grid. Training distribution: radius 150–500m, width 18–28m, validated. Held-out set: Sprint + GP + procedural seeds 1001/1002/1003. **Lecture 14.** |
| 9 | Headline result ⭐ | 80s | **100% lap success across 3 seeds × 5 unseen tracks × 20 episodes = 300 episodes.** Mean ± seed-std table. Honest Sprint wall-hit caveat (14m vs 18–28m training range). |
| 10 | Demo medley ⭐ | 150s | Pre-recorded narrated clips: 30s Sprint (familiar) + 45s Grand Prix + 45s procedural held-out (the generalization moment) + 30s tight-cornering close-up. Rays visible throughout. |
| 11 | Limitations | 45s | Sprint width gap; single-car only; Phase 5+ planned (multi-car curriculum, SAC baseline). |
| 12 | What we learned about RL | 60s | Three meta-bullets: reward design is adversarial (L1); frame stacking solves partial observability cheaply (L10); training on a distribution generalizes (L14). |
| 13 | Thanks + Q&A | 30s | — |
|   | **Total** | **14:55** | |

## Speaker segmentation

Contiguous blocks. Handoffs happen on transition slides, not mid-thought. Rehearse the exact handoff sentences.

| Speaker | Slides | Block theme | Speaking time |
|---|---|---|---|
| **A — The Setup** | 1–5 | Problem, MDP, PPO, frame stacking | ~5:00 |
| **B — Training & Failure** | 6–7 | Phase 2 baseline + reward war story | ~3:20 |
| **C — Generalization & Reflection** | 8–13 | Phase 4 + headline + demo + reflection + Q&A | ~6:35 |

Speaker C looks heavy on paper, but 2:30 of that block is the demo medley (narration, not dense delivery). Actual sustained speaking is roughly balanced across all three.

### Handoff cues (scripted)

- **A → B** (end of slide 5): *"With the env, the algorithm, and the observation pipeline locked down, we ran the first training experiments — over to [B]."*
- **B → C** (end of slide 7): *"v2 worked on Sprint, but the real question was whether it could drive a track it had never seen — [C]."*

## Lecture citation map (surface on slides)

Small "[Lecture N]" tag in the corner of:

- Slide 3 — Lecture 1
- Slide 4 — Lecture 9
- Slide 5 — Lecture 10
- Slide 7 — Lecture 1 / RL Book
- Slide 8 — Lecture 14
- Slide 12 — Lectures 1, 10, 14 (one per bullet)

## Production assets to prepare

Roughly in order of lead time:

1. **Demo medley clips** (slide 10) — 4 segments totalling 2:30, rays visible. Pre-render.
2. **v1 reversing gag clip** (slide 7) — 10s, exaggerated for effect.
3. **Phase 4 track grid** (slide 8) — 2×2 thumbnails of sampled procedural tracks.
4. **Before/after results table** (slide 7) — v1 vs v2 numbers from `CLAUDE.md`.
5. **Headline results table** (slide 9) — held-out evaluation summary.
6. **Frame-stacking visual** (slide 5) — 1-frame vs 4-frame side-by-side.
7. **Learning curve PNG** — v1 vs v2 or Phase 4 returns (slot into slide 6 or 7).

## Slide content (draft v1)

Each block below is the *on-slide* content — what the audience reads. Speakers fill in the rest. Keep slide text sparse; aim for ≤25 words of body copy per slide so the audience listens instead of reading.

### Slide 1 — Title (30s)

**Layout:** Full-bleed dark background. Centered single rendered frame of the car with rays fanned out as a hero image. Title overlaid in the upper third.

- **Title:** *Learning to Race from Rays Alone*
- **Subtitle:** A PPO agent that generalizes to unseen Formula 1 circuits
- **Authors:** [Speaker A] · [Speaker B] · [Speaker C]
- **Footer:** University of Haifa · MSc Reinforcement Learning · 2026

### Slide 2 — The problem & framing (70s)

**Layout:** Left half — full-frame screenshot of car with 30 rays visible. Right half — the question + the framing tradeoff.

- **Heading:** *Can an agent learn to race using only on-board sensors?*
- **Two-column tradeoff:**
  - *Raycast-only* — "What a human driver sees on lap 1." 30 distance readings, no track map.
  - *Frenet (state-based)* — "GPS + curvature preview." Knows where it is on the track.
- **Our choice:** raycast-only. The hard, transferable, human-like version.

### Slide 3 — MDP formulation (70s)

**Layout:** One-line MDP refresher at top. Three vertical columns below: State / Action / Reward.

- **Heading:** *The problem as an MDP* — at each step the agent sees a state, picks an action, gets a reward.

| **State (34-dim)** | **Action (2-dim, continuous)** | **Reward (per step)** |
|---|---|---|
| 30 raycast distances ∈ [0,1] | Throttle ∈ [−1, +1] | + progress along centerline |
| Speed (normalized) | Steering ∈ [−1, +1] | + speed bonus |
| Lateral velocity | | − lateral / heading error |
| Last throttle, last steering | | − wall hits, − off-track |

- **Bottom-right tag:** [Lecture 1]

### Slide 4 — Algorithm: PPO (70s)

**Layout:** Left — small actor-critic diagram (shared MLP trunk → policy head μ,σ + value head V). Right — the three reasons PPO fits.

- **Heading:** *Why PPO?*
- **Three bullets:**
  - **Continuous actions** → Gaussian policy `a ~ N(μ, σ²)` on `[throttle, steering]`
  - **On-policy stability** → clipped objective acts as a trust region; no catastrophic policy collapse
  - **Mature library support** → Stable-Baselines3, lets us focus the report on design choices
- **Hyperparameters (one line, small font):** `net_arch=[256,256], γ=0.99, GAE λ=0.95, clip=0.2, lr=3e-4`
- **Bottom-right tag:** [Lecture 9]

### Slide 5 — Frame stacking & partial observability (60s)

**Layout:** Top — the problem in one sentence. Center — two side-by-side diagrams: "1 frame" vs "4 frames stacked." Bottom — the fix.

- **Heading:** *One frame can't tell you how fast you're moving.*
- **Diagram caption:** A single raycast vector says *where* the walls are, not *whether they're approaching.*
- **Fix:** Wrap with `VecFrameStack(n_stack=4)` → 136-dim policy input. The network learns to read motion from frame deltas.
- **Aside:** This is the same trick that made DQN work on Atari.
- **Bottom-right tag:** [Lecture 10]

### Slide 6 — Phase 2 baseline (50s)

**Layout:** Left — learning curve (episode return vs timesteps, 3 seeds, shaded band). Right — setup + the punchline.

- **Heading:** *Phase 2 — PPO on the Sprint Circuit*
- **Setup:** Single car · Sprint Circuit only · 1M timesteps · 3 seeds
- **Curve reads:** Reward goes up. Episode length goes up. The agent is "learning."
- **But:** lap success rate = **0%**.
- **Closing line (verbal cue for B):** *"…so we watched what it was actually doing."*

### Slide 7 — Reward hacking war story ⭐ (150s)

**Layout:** Top-left — the 10s gag clip (autoplay on slide entry). Top-right — the exploit explained in three lines. Bottom — the before/after table.

- **Heading:** *The agent learned to drive backwards.*
- **Exploit (right of clip):**
  - The `+ds` term rewarded centerline progress in *either* direction
  - Reversing through the start/finish line registered as new progress
  - Result: ever-increasing reward, zero valid laps
- **Three v2 fixes:**
  - Zero out progress + speed reward while a wall is being touched
  - Add `−0.001 / step` time cost to make camping costly
  - Terminate with `−25` after 120 steps of sustained reverse driving

**Phase 3 — v1 vs v2 on Sprint (mean ± seed std):**

| Metric | v1 control | v2 shaped |
|---|---:|---:|
| Lap success | 0% ± 0% | **100% ± 0%** |
| Laps per episode | 0 ± 0 | 8 ± 0 |
| Progress fraction | −8.01 ± 0.14 | +8.49 ± 0.07 |
| Wall hits | 29.4 ± 1.7 | **0.5 ± 0.7** |

- **Bottom-right tag:** [Lecture 1 / RL Book — reward design]

### Slide 8 — Phase 4: domain randomization (90s)

**Layout:** Left — 2×2 grid of four sampled procedural tracks rendered at the same scale. Right — the distribution definition + held-out set.

- **Heading:** *Training on a distribution, not a track.*
- **Training distribution (per episode reset):**
  - Base radius ∈ [150m, 500m]
  - Width ∈ [18m, 28m]
  - Fourier harmonics k ∈ {3,4,5,7,9} with randomized phases
  - Rejected: invalid polygons, curvature κ > 0.2 m⁻¹
- **Held-out set (never seen during training):** Sprint · Grand Prix · procedural seeds 1001 / 1002 / 1003
- **Training:** 5M timesteps · 8 parallel envs · 3 seeds
- **Bottom-right tag:** [Lecture 14]

### Slide 9 — Headline result ⭐ (80s)

**Layout:** Top — the headline number in display type. Middle — the results table. Bottom — one-line caveat.

- **Headline (large):** **100% lap success across 300 held-out episodes**
- **(small under it):** 3 seeds × 5 unseen tracks × 20 episodes = 300

**Held-out evaluation (mean ± seed std):**

| Metric | Result |
|---|---:|
| Lap success | **100% ± 0%** |
| Progress fraction | 3.83 ± 0.05 |
| Mean speed | 78.4 ± 0.2 m/s |
| Wall hits | 11.6 ± 4.9 |

- **Caveat (small):** Sprint is the outlier — its 14m width is narrower than the 18–28m training distribution, so it averages 38 wall hits while completing all laps.

### Slide 10 — Demo medley ⭐ (150s)

**Layout:** Full-bleed video. No slide text — the speaker narrates over the playback.

- **Heading (small, top-left, fades after 5s):** *Trained v2 agent · held-out tracks · rays visible*
- **Clip order (2:30 total):**
  1. **0:00–0:30** — Sprint Circuit (familiar from slide 6, "look how it changed")
  2. **0:30–1:15** — Grand Prix Circuit (longer, more varied)
  3. **1:15–2:00** — Procedural held-out (seed 1001 or 1002) — *the generalization moment*
  4. **2:00–2:30** — Tight cornering close-up with ray compression visible

### Slide 11 — Limitations (45s)

**Layout:** Three bullets, equal weight. No images — keep attention on the speaker.

- **Heading:** *What we didn't solve yet.*
- **Sprint width gap** — the agent collects ~38 wall hits on Sprint because the training distribution starts at 18m and Sprint is 14m. Easy fix: widen the training range.
- **Single car only** — no opponents, no overtaking, no race tactics.
- **Phase 5+ on the roadmap** — multi-car curriculum, self-play opponent pool, SAC comparison.

### Slide 12 — What we learned about RL (60s)

**Layout:** Three lines, each with a small lecture tag. Treat as the takeaway slide.

- **Heading:** *Three lessons from this project.*
- **1. Reward design is adversarial against your own agent.** The agent will find whatever the reward technically rewards. [Lecture 1]
- **2. Frame stacking solves partial observability cheaply.** No recurrent network, no engineered velocity feature — just stack four frames. [Lecture 10]
- **3. Training on a distribution generalizes; training on one track memorizes.** Same algorithm, same hyperparameters — different track-set discipline, different outcome. [Lecture 14]

### Slide 13 — Thanks + Q&A (30s)

**Layout:** Single-line thanks centered. Names, repo link, contact below. Optional: small "questions?" prompt at bottom.

- **Heading:** *Thank you — questions?*
- **Team:** [Speaker A] · [Speaker B] · [Speaker C]
- **Repo:** `github.com/<org>/racing-RL` *(or whatever the public link will be)*
- **Course:** University of Haifa · MSc Reinforcement Learning · 2026

## To do next on this document

- ~~Draft slide-level content (titles, bullet text, table layouts) for each slide~~ ✓
- Add a rehearsal / timing-check protocol
- Per-speaker script notes for the tighter slides (3, 4, 5, 9, 12)
