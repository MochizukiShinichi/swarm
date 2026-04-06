# Swarm Intelligence: Decentralized Object Transport (ES-Marathon)

A high-fidelity GPU-accelerated 2D swarm simulation implemented in **Python** using the **Taichi** programming language. This project demonstrates how a group of simple, autonomous agents can learn to collaborate and move a rigid object to a target location through **OpenAI Evolutionary Strategies (OpenAI-ES)**.

## Project Overview
The upgraded simulation now features **Spring-Based Physics** and an **Adam-optimized** training loop. 256 environments are simulated in parallel on the GPU to evolve a 10,000+ parameter policy that allows 100 agents to coordinate and push objects with precision.

-   **High Fidelity Physics:** Continuous spring physics (Hooke's Law) and $O(N^2)$ collision detection.
-   **Advanced Training:** OpenAI-ES with Adam Optimizer, Antithetic Sampling, and Auto-Curriculum learning.
-   **Self-Correcting:** Built-in plateau detection and learning rate decay to escape local minima.
-   **Web Parity:** A custom 1:1 Native JavaScript physics mirror runs in the browser, completely replacing external libraries (like Matter.js) for mathematically perfect policy evaluation.

## Getting Started

### Prerequisites
-   Python 3.8+ (using `uv` is recommended for dependency management)
-   A GPU supporting CUDA, Vulkan, or Metal (Taichi's default backends)
-   2GB GPU VRAM (configured in `swarm_sim.py`)
-   Node.js (for the web interface)

### Installation & Training
1. Install dependencies and run the training marathon:
   ```bash
   uv run python swarm_sim.py
   ```

2. Run the Web Evaluator:
   ```bash
   cd web
   npm install
   npm run dev
   ```

### Training Output
-   The trained policy is exported to `web/public/policy.json`.
-   Real-time progress is logged to the console:
    ```text
    Gen 00100 | Max Fit: 389503.4 | Success: 118/256 | LR: 0.0500 | Sig: 0.100
    ```

## Algorithm Details (V2)
For a deep dive into Hooke's Law penalty physics, OpenAI-ES gradient estimation, reward shaping, and 1:1 Physics Parity, see:
👉 [**ALGORITHM.MD**](./ALGORITHM.md)

## Repository Structure
-   `swarm_sim.py`: The core simulation and training logic (Taichi + NumPy + Adam Optimizer).
-   `ALGORITHM.md`: Detailed documentation of the physics and training innovations.
-   `web/`: React + Vite web application featuring a custom Native JS physics engine for evaluating the policies.
-   `plan.txt`: The original project vision and roadmap.
