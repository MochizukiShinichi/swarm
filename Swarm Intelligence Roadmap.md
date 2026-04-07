# Swarm Intelligence — Future Improvement Roadmap

> **Project:** [swarm](https://github.com/MochizukiShinichi/swarm) — Decentralized object transport via evolutionary swarm intelligence
> **Goal:** Grow the swarm from barely-functional to real-world-transferable intelligence, with a gradual, auditable history of improvement
> **Hardware:** RTX 2080 (potentially 3080), cloud GPU on demand, 2D sim (transfers to ground robots/planar drones)

---

## Core Design: Metric-Gated Progression

Every complexity increase is **earned, not scheduled**. The swarm must prove competence before the world gets harder or the brain gets more complex.

### Three Mastery Metrics

Measured over a **rolling window of 50 generations**:

| Metric | Symbol | What it measures | Computation |
|--------|--------|-----------------|-------------|
| **Success Rate** | SR | Can the swarm reliably deliver? | `mean(env_success)` — fraction of envs reaching target |
| **Efficiency** | EF | Does it do it without wasting time? | `1 - mean(success_time / EPISODE_STEPS)` for successful envs |
| **Consistency** | CO | Is it reliable, not lucky? | `1 - std(fitness) / mean(fitness)` — low variance = generalization |

### Gate Rule

A stage is **mastered** when, over 50 consecutive generations, **all three** metrics meet their thresholds simultaneously.

### Regression Protocol

When a complexity increase causes metrics to drop:

1. **Hold** — No further increases for 200 generations
2. **Recover** — If metrics recover to gate thresholds, continue
3. **Rollback** — If no recovery after 500 generations, revert last complexity increase, boost sigma by 20%, adapt for 300 generations, then retry

---

## SR Thresholds by Phase

Early stages demand the **highest** SR because the tasks are easiest. The bar drops only when the task gets qualitatively harder.

| Phase | SR Baseline | Justification |
|-------|------------|---------------|
| Phase 1 — Learn to Push | **≥ 0.95** | Simple push, no obstacles, no noise. Near-perfect or it's not learned. |
| Phase 2 — Learn to Think | **≥ 0.90** | Obstacles and noise are real challenges, small margin allowed. |
| Phase 3 — Learn to Coordinate | **≥ 0.85** | Multi-object splits the swarm's resources. Genuinely hard. |
| Phase 4 — Generalize | **≥ 0.80** | Adversarial randomization. 80% here = battle-tested. |

---

## Phase 1: Learn to Push

> The swarm can barely nudge the object today. Fix the foundations before anything else.

| Stage | Track | Change | SR | EF | CO |
|-------|-------|--------|-----|-----|-----|
| **1.0** | Brain | **Training fixes** — Fix ES loop: proper fitness normalization, weight decay, sigma adaptation (CMA-style), longer warmup at low difficulty | ≥ 0.95 | ≥ 0.50 | ≥ 0.60 |
| **1.1** | Brain | **Sensor cleanup** — Replace the 3 dead inputs (`0.0, 0.0, 1.0`) with: distance-to-object (scalar), object velocity direction, agent density in forward cone | ≥ 0.95 | ≥ 0.60 | ≥ 0.70 |
| 1.2 | World | **Unlock curriculum** — Metric-gated difficulty: difficulty increases only when SR ≥ 0.95 at *current* level, not on a fixed timer | ≥ 0.95 | ≥ 0.65 | ≥ 0.70 |

### Phase 1 V&V Protocol (Mandatory)
*   **[Automated] Unit Parity:** Run `tests/parity_check.py` to ensure Python `step()` and JS `usePhysics.ts` logic produce identical state transitions (within 1e-4 tolerance) for a fixed weight set.
*   **[Empirical] 1K Marathon:** Run `main.py` for 1,000 generations. **Pass Criteria:** `SR` must reach ≥ 0.90 at Difficulty 0.0 and ≥ 0.50 at Difficulty 0.3 within this window.
*   **[Visual] Straight-Line Test:** In the web evaluator, the object trail must deviate < 15% from the shortest-path vector.

---

## Phase 2: Learn to Think


> The brain can push. Give it memory and communication.

| Stage | Track | Change | SR | EF | CO |
|-------|-------|--------|-----|-----|-----|
| **2.0** | Brain | **Recurrent memory (GRU cell)** — MLP+GRU, hidden state size 16. Agents remember the last 10-20 steps. Enables anticipation. | ≥ 0.90 | ≥ 0.55 | ≥ 0.60 |
| **2.1** | Brain | **Neighbor messaging** — Each agent broadcasts a 4-dim vector to neighbors within comm radius. Receive average neighborhood message. Foundation of coordination. | ≥ 0.90 | ≥ 0.60 | ≥ 0.65 |
| **2.2** | World | **Static obstacles** — 2-4 circular pillars between object and target. | ≥ 0.90 | ≥ 0.45 | ≥ 0.55 |
| 2.3 | World | **Sensor noise + motor delay** — Gaussian noise σ=0.05 on sensors. 2-step motor delay. First real-world constraint. | ≥ 0.90 | ≥ 0.40 | ≥ 0.50 |

### Phase 2 V&V Protocol (Mandatory)
*   **[Automated] Memory Test:** Run `tests/memory_check.py`. Agents must demonstrate "delayed reaction" (moving toward a target that has disappeared) if hidden states are working.
*   **[Automated] Messaging Test:** Run `tests/comm_check.py`. Check if agent A's output message appears in agent B's input message in the next step.
*   **[Empirical] 2K Marathon:** Run `main.py` for 2,000 generations. **Pass Criteria:** `SR` must reach ≥ 0.90 at Difficulty 0.6 (obstacles active) within this window.
*   **[Visual] Formation Test:** In the web evaluator, agents must form a clear "pushing arc" that persists even if the object is temporarily obscured.

---

## Phase 3: Learn to Coordinate


> The swarm can push, think, and communicate. Create pressure for specialization.

| Stage | Track | Change | SR | EF | CO |
|-------|-------|--------|-----|-----|-----|
| **3.0** | Brain | **Attention over neighbors** — 4-head attention replacing average-message. Agents selectively listen to specific neighbors. | ≥ 0.85 | ≥ 0.50 | ≥ 0.60 |
| **3.1** | World | **Multi-object transport** — 2 objects, 2 targets. Forces role differentiation. | ≥ 0.85 (both) | ≥ 0.35 | ≥ 0.50 |
| **3.2** | World | **Moving targets** — Target drifts slowly. Swarm must track, not just deliver. | ≥ 0.85 | ≥ 0.45 | ≥ 0.55 |
| **3.3** | Brain | **Energy budget** — Finite energy per agent. Depleted agents stop. Forces efficient swarm usage. | ≥ 0.85 | ≥ 0.50 | ≥ 0.55 |

**What you'll see:** The swarm splits into teams. Relay behavior — fresh agents rotating in. Leader-follower dynamics. Deliberate path optimization.

---

## Phase 4: Generalize (Sim-to-Real Bridge)

| Stage | Track | Change | SR | EF | CO |
|-------|-------|--------|-----|-----|-----|
| **4.0** | World | **Domain randomization** — Randomize agent mass ±20%, friction ±30%, object shape (polygons), communication dropout 10% | ≥ 0.80 | ≥ 0.40 | ≥ 0.50 |
| **4.1** | Brain | **Policy distillation** — Smaller deployment-ready network (≤5K params), runs on microcontroller | Distilled ≥ 0.90 × full SR | — | — |
| **4.2** | World | **ROS2 bridge** — Velocity commands in Twist format. Sim accepts odometry. Compatible with TurtleBot / Crazyflie. | Manual: runs in Gazebo | — | — |

**What you'll see:** Swarm handles novel situations. Distilled policy fits on real hardware. One cable away from a real robot swarm.

---

## Current Codebase Issues

> These need fixing before Stage 1.0 begins.

### Bugs (Correctness)

- [ ] **Wrong push penalty is inverted** — `swarm_sim.py:161`: `push_f.dot(to_t) > 0` penalizes when spring force on agent points toward target. But reaction force on object is `-push_f`, so this penalizes *good* pushing. Likely a major reason the swarm is struggling.
- [ ] **Web evaluator applies force only on X axis** — `usePhysics.ts:102`: `outY` is computed but never applied to `agent.vy`. The demo runs with half a brain.
- [ ] **Web local consensus is global, not grid-local** — `usePhysics.ts:66-68`: Python uses 3×3 grid neighborhood, JS averages all 100 agents. Breaks the "1:1 parity" claim.
- [ ] **Three dead sensor inputs** — `swarm_sim.py:104`: `inp[14], inp[15], inp[16]` are hardcoded `0.0, 0.0, 1.0`. 17% of the brain's sensory bandwidth is wasted.

### Performance

- [ ] **O(N²) collision per env** — `swarm_sim.py:131`: Fine at N=100, blocks scaling to 200+ agents. Grid-based broadphase would make it O(N). Fix before Phase 3.
- [ ] **Full weight transfer every generation** — `swarm_sim.py:236-240`: CPU→GPU copy of all weights for 256 envs. Could apply noise directly on GPU in-kernel.

### Structural

- [ ] **Single-file architecture** — `swarm_sim.py` = 296 lines, everything in one file. Needs separation (physics / neural net / ES optimizer / curriculum / logging) before Phase 2.

---

## Visual / Web Roadmap

Each phase should produce a visible upgrade to the browser demo:

| Phase | Visual Upgrade |
|-------|---------------|
| 1 | The swarm actually works reliably. Add trail lines showing object path. |
| 2 | Visualize agent communication links. Show obstacle avoidance in real-time. |
| 3 | Color-code agents by role (pusher/scout/relay). Show energy bars. Multi-object split visible. |
| 4 | "Chaos mode" button — random perturbations the swarm handles gracefully. |
