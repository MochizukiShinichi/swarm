# 🐝 Swarm Intelligence: Decentralized Object Transport (ES-Marathon)

[![Deploy to GitHub Pages](https://github.com/MochizukiShinichi/swarm/actions/workflows/deploy.yml/badge.svg)](https://github.com/MochizukiShinichi/swarm/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://mochizukishinichi.github.io/swarm/)

A high-fidelity GPU-accelerated swarm simulation implemented in **Python (Taichi)** and **React (Vite)**. This project evolves a group of 100 autonomous agents to collaboratively move a rigid object to a target using **OpenAI Evolutionary Strategies (OpenAI-ES)**.

---

## 🧪 Experiment Setting

The experiment is designed to solve a complex multi-agent coordination problem where no single agent can move the object alone. They must learn to synchronize their vectors to overcome the object's mass and inertia.

### 📐 Physical Layout
```mermaid
graph TD
    subgraph "Simulation Space (800x500)"
    T((Target)) -- "Goal" --- O[Square Object]
    A1((Agent)) -. "Restorative Force" .-> O
    A2((Agent)) -. "Restorative Force" .-> O
    A3((Agent)) -. "Restorative Force" .-> O
    A4((Agent)) -. "Restorative Force" .-> O
    end
    
    style T fill:#f97316,stroke:#fff,stroke-width:2px
    style O fill:#fb923c,stroke:#fff,stroke-width:4px
    style A1 fill:#22d3ee,stroke:#fff
    style A2 fill:#22d3ee,stroke:#fff
    style A3 fill:#22d3ee,stroke:#fff
    style A4 fill:#22d3ee,stroke:#fff
```

### 📋 Key Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Agent Count** | 100 | Autonomous particles with 18-D sensors |
| **Parallel Envs** | 256 | High-throughput GPU training episodes |
| **Physics Model** | Hooke's Law | Continuous spring-based soft collisions |
| **Optimizer** | Adam | Gradient estimation via mirrored noise (Antithetic) |
| **Brain Architecture** | 18 → 64 → 2 | MLP controller determining force vectors |

---

## 🧠 Training Workflow

The simulation leverages **Taichi Lang** to run the physics and neural network kernels directly on the GPU, achieving thousands of frames per second.

```mermaid
sequenceDiagram
    participant GPU as Parallel GPU Envs (256)
    participant ES as OpenAI-ES Optimizer
    participant Policy as Neural Controller (18-D)

    loop Every Generation
        ES->>Policy: Inject Mirrored Noise (+ε / -ε)
        Policy->>GPU: Execute 500 Steps
        GPU-->>ES: Return Fitness Scores
        ES->>ES: Estimate Gradient via Adam
        ES->>Policy: Update Master Weights
    end
```

---

## 🚀 Getting Started

### 1. Training (Python)
Ensure you have a GPU supporting CUDA, Vulkan, or Metal. We recommend using `uv` for dependency management.
```bash
# Install dependencies and start the ES-Marathon
uv run python swarm_sim.py
```

### 2. Live Evaluation (Web)
The web interface features a **1:1 Physics Mirror** in Native JavaScript. It evaluates the trained `policy.json` with mathematical parity to the Python trainer.
```bash
cd web
npm install
npm run dev
```

---

## 🏆 Emergent behaviors
Through the Evolutionary process, the swarm discovers sophisticated coordination tactics:
*   **C-Shaped Wrapping**: Agents learn to form a concave shell around the object to prevent lateral sliding.
*   **Dynamic Braking**: Agents on the target side learn to yield or provide counter-pressure to stop the object precisely.
*   **Local Consensus**: Agents sense the velocity of their neighbors to stay grouped (flocking behavior).

---

## 📂 Repository Structure
*   `swarm_sim.py`: The core GPU trainer (Taichi + Adam Optimizer).
*   `web/`: React + Vite web evaluator with custom physics parity.
*   [**ALGORITHM.md**](./ALGORITHM.md): Deep dive into Hooke's Law, Reward Shaping, and 1:1 Physics Parity.

---
*Created by MochizukiShinichi - Distributed under the MIT License.*
