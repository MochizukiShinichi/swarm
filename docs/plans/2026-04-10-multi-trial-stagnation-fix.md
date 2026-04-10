# Stagnation & Multi-Trial Training Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix gradient dilution and rank compression by decoupling population size from evaluation environments. Implement multi-trial averaging (4 trials per candidate) and automated stagnation recovery.

**Architecture:** Use 1024 parallel environments to evaluate a population of 256 candidates, with each candidate running in 4 independent environments (different random seeds). Average fitness across trials before ranking to suppress environmental noise. Implement a temporal stagnation monitor to penalize "trapped" states.

**Tech Stack:** Taichi (GPU Physics), NumPy, OpenAI-ES (Adam).

---

### Task 1: Decoupling Population from Parallelism

**Files:**
- Modify: `engine.py`
- Modify: `main.py`

**Step 1: Update Constants in `engine.py`**
- Set `NUM_ENVS = 1024`.
- Add `POP_SIZE = 256`.
- Add `TRIALS_PER_CANDIDATE = 4`.
- Ensure `NUM_ENVS == POP_SIZE * TRIALS_PER_CANDIDATE`.

**Step 2: Update Weight Loading in `main.py`**
- Initialize optimizer with `engine.POP_SIZE` instead of `engine.NUM_ENVS`.
- In the training loop, get population from `optimizer.ask()`.
- **Tile population:** Create `eval_population = np.repeat(population, engine.TRIALS_PER_CANDIDATE, axis=0)`.
- Load tiled weights: `engine.load_weights(eval_population)`.

**Step 3: Average Fitness in `main.py`**
- After the simulation, get raw fitness: `raw_fitness, raw_reached = engine.get_fitness()`.
- **Reduce fitness:** Reshape and mean:
  ```python
  fitness = raw_fitness.reshape(engine.POP_SIZE, engine.TRIALS_PER_CANDIDATE).mean(axis=1)
  reached = raw_reached.reshape(engine.POP_SIZE, engine.TRIALS_PER_CANDIDATE).mean(axis=1)
  ```
- Pass the averaged `fitness` to `optimizer.tell(fitness, gen)`.

**Step 4: Commit**
```bash
git add engine.py main.py
git commit -m "feat: decouple population size from eval envs with multi-trial averaging"
```

---

### Task 2: Stagnation Detection & Trap Penalty

**Files:**
- Modify: `main.py`

**Step 1: Implement Position Tracking**
- Inside the generation loop, initialize `last_obj_pos = engine.obj_pos.to_numpy()`.
- Initialize `stagnation_counters = np.zeros(engine.NUM_ENVS)`.
- Initialize `stuck_envs = np.zeros(engine.NUM_ENVS)`.

**Step 2: Add Temporal Check in Step Loop**
- Every 50 steps (if `t % 50 == 0` and `t > 200`):
    - Calculate `dist_moved = norm(curr_obj_pos - last_obj_pos)`.
    - If `dist_moved < 2.0`, increment `stagnation_counters`.
    - Else, reset `stagnation_counters` to 0.
    - If `stagnation_counters >= 3` (150 steps stuck), set `stuck_envs = 1.0`.

**Step 3: Apply Trap Penalty to Raw Fitness**
- Before averaging: `raw_fitness -= stuck_envs * 500.0`.

**Step 4: Commit**
```bash
git add main.py
git commit -m "feat: implement stagnation detection and trap penalty"
```

---

### Task 3: Mastery-Based Gating

**Files:**
- Modify: `main.py`

**Step 1: Implement Rolling Metrics**
- Add rolling windows (size 50) for `Success Rate` (averaged `reached`), `Efficiency` (averaged time), and `Consistency` (1.0 - std/mean fitness).

**Step 2: Add Mastery Logic**
- If `mean_sr >= 0.75` AND `mean_ef >= 0.50` AND `mean_co >= 0.30`:
    - Increment `difficulty += 0.1`.
    - Reset rolling windows and plateau counters.
    - Print: `>>> MASTERY REACHED! Difficulty Increased to {difficulty} <<<`.

**Step 3: Commit**
```bash
git add main.py
git commit -m "feat: add mastery-based difficulty gating"
```

---

### Task 4: Final Validation Run

**Step 1: Start Marathon Training**
- Run: `uv run python main.py`.
- Verify that `SR` represents the average of 4 trials per candidate.
- Observe if the cleaner gradient leads to faster convergence.
