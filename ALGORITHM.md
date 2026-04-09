# Algorithm Details: Deep Rigid Swarm (V3 - Coordination)

The system has been upgraded to a high-fidelity "Marathon" trainer using **Recurrent Neural Networks (GRU)**, **Neighborhood Attention**, and optimized **Grid-based Physics**.

## 1. Physics Engine: Optimized Grid Broadphase
The simulation uses **Penalty-Based Contact** (Hooke's Law) for realistic momentum transfer. To support scaling to larger swarms, the collision system has been optimized.

### Collision Handling
-   **Grid-Based Broadphase:** Instead of brute-force $O(N^2)$, the arena is divided into a 40x40 spatial grid. Agents only check for collisions with neighbors in adjacent grid cells ($O(N)$ scaling).
-   **Soft Collisions:** Overlaps generate restorative spring forces (`F = overlap * K`).
-   **Newtonian Parity:** Reaction forces are applied equally to agents and objects (Newton's Third Law), ensuring conservation of momentum during pushes.

---

## 2. Neural Architecture: Recurrent Attention Controller (22-D)
The agent "brain" has evolved from a simple MLP to a sophisticated temporal processor capable of selective coordination.

### Multi-Stage Processing
1.  **Sensation (MLP 1):** Processes 18 spatial sensors + 4 coordination signals.
2.  **Selective Attention (Stage 3.0):** Agents generate a **Query** vector to probe their local neighborhood. They perform **Dot-Product Attention** over neighbors' broadcasted **Key/Value** messages, allowing them to selectively "listen" to relevant signals (e.g., ignoring noise to follow a scout).
3.  **Memory (GRU Cell):** A 16-unit Gated Recurrent Unit (GRU) maintains internal state across time steps. This enables anticipation, path-planning through obstacles, and persistence of intent.
4.  **Motor Output (MLP 2):** Decodes the GRU state into 2D force vectors and a new 4D message to broadcast.

---

## 3. World Constraints: Sim-to-Real Hardening
To prepare for physical robot deployment, the trainer now includes real-world "friction":
-   **Sensor Noise:** Gaussian noise ($\sigma=0.05$) is injected into all inputs when `difficulty > 0.6`.
-   **Actuator Delay:** A 2-step motor delay is implemented, forcing the brain to learn "dead reckoning" and predictive control.
-   **Static Obstacles:** Pillars appear between the object and target, requiring the swarm to split and navigate gaps.

---

## 4. Training: Robust OpenAI-ES
The "Marathon" trainer uses a gradient-estimated Evolution Strategy with the **Adam Optimizer**.

### Key Training Features:
-   **Metric-Gated Curriculum:** Difficulty only increases when the swarm reaches **Mastery** (Success Rate $\ge 0.95$, Efficiency $\ge 0.50$, Consistency $\ge 0.60$).
-   **Rank-Based Fitness:** Fitness is transformed into percentiles to prevent outliers from distorting the gradient estimate.
-   **Weight Decay:** L2 regularization prevents weights from exploding, ensuring smooth, transferrable policies.

---

## 5. Emergent Coordination
-   **Role Differentiation:** The Attention mechanism allows agents to naturally specialize. Some act as **Pushers** (high-force, low-maneuver), while others act as **Scouts** (low-force, high-signal).
-   **Arc Formations:** The swarm learns to form persistent "pushing arcs" that can deform to squeeze through gaps between obstacles without losing control of the object.
-   **Leader-Follower Dynamics:** Agents learn to relay signals from the "front" of the swarm to the "back" via neighbor messaging.
