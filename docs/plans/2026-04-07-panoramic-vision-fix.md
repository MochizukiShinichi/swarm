# Panoramic Vision & Stagnation Fix Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve SR >= 0.90 at Difficulty 0.4 by fixing obstacle blindness and penalizing trapped states.

**Architecture:** 
1. Expand agent input from 22-D to 30-D to include all 4 obstacles.
2. Implement a "Stagnation early-exit" in the training loop.
3. Update the JS evaluator to maintain mathematical parity.

**Tech Stack:** Taichi (Python), NumPy, React/TypeScript (Web).

---

### Task 1: Engine Upgrade (Panoramic Vision)

**Files:**
- Modify: `engine.py`

**Step 1: Expand input dimensions and sensors**
- Update `IN_DIM` to 30.
- Update `PARAM_COUNT` calculation.
- Rewrite `step()` kernel sensor section to loop through all 4 obstacles and provide relative vectors + distances.

**Step 2: Add Obstacle Penalty**
- Update `obstacle_penalty` logic in `step()` or `integration` loop to penalize when the object center is within `OBJ_SIZE + 20.0` of a pillar.

**Step 3: Update load/export logic**
- Update `load_weights` and `export_policy` to handle the new `IN_DIM`.

**Step 4: Verify via Parity Test**
- Update `tests/parity_check.py` to match the new 30-D input.
- Run: `$env:PYTHONPATH='.'; uv run python tests/parity_check.py`

### Task 2: Training Hardening (Stagnation)

**Files:**
- Modify: `main.py`

**Step 1: Implement Stagnation Detection**
- In the generation loop, track `last_obj_pos` every 50 steps.
- If total movement < 2px over 50 steps, increment a per-env stagnation counter.
- If counter >= 3 (150 steps of no progress), flag the env as inactive.

**Step 2: Update Fitness for Stagnation**
- Deduct a "Trap Penalty" from fitness for envs that stagnation-exited.

**Step 3: Reset Gating**
- Ensure `difficulty` starts at 0.4 for this hardening session.

### Task 3: Web Parity Upgrade

**Files:**
- Modify: `web/src/hooks/usePhysics.ts`

**Step 1: Update Sensory logic**
- Mirror the 30-D panoramic sensor logic exactly from `engine.py`.

**Step 2: Update Inference logic**
- Adjust the MLP 1 loop to handle 30 inputs.

**Step 3: Verification**
- Refresh browser and verify no NaN errors in console.

### Task 4: Recovery Marathon

**Step 1: Run Training**
- Run: `uv run python main.py`
- Target: SR >= 0.90 at Difficulty 0.4.
- Stop when Mastery is reached.
