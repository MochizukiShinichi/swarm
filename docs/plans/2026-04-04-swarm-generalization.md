# Swarm Generalization Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generalize the swarm transport system to handle random object shapes and navigate static obstacles using a neural local policy.

**Architecture:** A decentralized swarm of particles where each particle's velocity is governed by a local Multi-Layer Perceptron (MLP) policy. The policy takes relative vectors to the target, the object, and nearest neighbors/obstacles as input. Training is done via Evolutionary Strategies (ES) across 16+ parallel GPU environments.

**Tech Stack:** Python, Taichi (GPU Physics), NumPy (ES/Optimization).

---

### Task 1: Random Polygon Target Objects

**Files:**
- Modify: `swarm_sim.py` (Update `is_in_object` and vertex calculation)

**Step 1: Define Polygon Data Structures**

```python
MAX_VERTS = 8
obj_num_verts = ti.field(dtype=ti.i32, shape=NUM_ENVS)
obj_verts = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_ENVS, MAX_VERTS))
```

**Step 2: Implement Random Polygon Generation in `initialize_env`**

Generate random convex polygons for each environment to test generalization.

**Step 3: Update `is_in_object` for Arbitrary Polygons**

Use the Ray Casting or Winding Number algorithm for point-in-polygon tests.

**Step 4: Commit**

```bash
git add swarm_sim.py
git commit -m "feat: support random polygon objects"
```

---

### Task 2: Static Obstacles (Pillars)

**Files:**
- Modify: `swarm_sim.py` (Add `pillars` field and collision logic)

**Step 1: Define Pillar Data**

```python
NUM_PILLARS = 4
pillar_pos = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_ENVS, NUM_PILLARS))
pillar_radius = 0.05
```

**Step 2: Add Pillar Collision in `step`**

Particles and the target object must bounce off pillars.

**Step 3: Render Pillars in GUI**

**Step 4: Commit**

```bash
git add swarm_sim.py
git commit -m "feat: add static pillars to environment"
```

---

### Task 3: Neural Local Policy (MLP)

**Files:**
- Modify: `swarm_sim.py` (Implement MLP layers and inference)

**Step 1: Define MLP Weights Fields**

```python
INPUT_DIM = 6 # target_rel, obj_rel, nearest_pillar_rel
HIDDEN_DIM = 16
OUTPUT_DIM = 2 # force vector

W1 = ti.field(dtype=ti.f32, shape=(NUM_ENVS, INPUT_DIM, HIDDEN_DIM))
b1 = ti.field(dtype=ti.f32, shape=(NUM_ENVS, HIDDEN_DIM))
W2 = ti.field(dtype=ti.f32, shape=(NUM_ENVS, HIDDEN_DIM, OUTPUT_DIM))
b2 = ti.field(dtype=ti.f32, shape=(NUM_ENVS, OUTPUT_DIM))
```

**Step 2: Implement MLP Inference in `step`**

Replace the current heuristic force with a forward pass of the MLP for each particle.

**Step 3: Update Weight Initialization**

Initialize MLP weights from a flattened vector passed during `initialize_env`.

**Step 4: Commit**

```bash
git add swarm_sim.py
git commit -m "feat: implement neural local policy (MLP)"
```

---

### Task 4: Advanced Fitness & ES Loop

**Files:**
- Modify: `swarm_sim.py` (Update `get_fitness` and ES main loop)

**Step 1: Refine Fitness Function**

Include penalty for object-pillar collisions and a small bonus for group coherence.

**Step 2: Implement Multi-Step Evolution**

Allow training for hundreds of generations with decreasing noise (simulated annealing).

**Step 3: Final Validation Run**

Verify that the swarm can push a random star-shaped object through a pillar-filled arena.

**Step 4: Commit**

```bash
git add swarm_sim.py
git commit -m "feat: update fitness and ES loop for generalization"
```
