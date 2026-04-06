# Algorithm Details: Deep Rigid Swarm (V2)

The system has been upgraded to a high-fidelity "Marathon" trainer using advanced Evolution strategies and spring-based physics, with a perfectly synchronized web evaluator.

## 1. Physics Engine: Continuous Spring Physics
The simulation now uses **Penalty-Based Contact** (Hooke's Law) instead of simple position snapping. This provides smoother, more realistic momentum transfer.

### Soft Collisions (Hooke's Law)
-   **Agent-Agent:** When two agents overlap, a restorative spring force is applied: `F = overlap * K_AGENT`.
-   **Agent-Object:** When an agent penetrates the object's bounds, a force proportional to the penetration depth (`K_OBJ`) is applied to both the agent and the object (Newton's Third Law).
-   **Brute Force Stability:** To ensure perfect stability and eliminate grid-based artifacts, the simulation performs brute-force $O(N^2)$ collision checks for the 100 agents within each environment.

---

## 2. Neural Network: 18-D Controller
The "brain" has been expanded to 18 dimensions to provide better spatial awareness.

### Expanded Inputs (18-dimensions)
-   **Target/Object Vectors:** Normalized directions to the target and object.
-   **Local Consensus (`lv`):** The average velocity of neighbors in the immediate grid cells.
-   **Inverse Proximity Walls:** Distances to the boundaries calculated as `1/(1+dist)`, giving the agents a non-linear "fear" of walls as they get closer.
-   **Dynamics:** Individual velocity and absolute speed.

---

## 3. Training: OpenAI Evolution Strategy (OpenAI-ES)
The trainer has been upgraded from a basic "pick the best" approach to a gradient-based optimization method.

### Key Innovations:
1.  **Adam Optimizer:** We estimate the gradient of the fitness landscape and use the Adam optimizer (with momentum and second-order moment estimation) to update weights.
2.  **Antithetic Sampling (Mirroring):** For every noise perturbation $\epsilon$, we also evaluate its mirror $-\epsilon$. This significantly reduces gradient variance and stabilizes training.
3.  **Auto-Curriculum Learning:** The simulation starts with the object near the target and gradually increases the distance (`difficulty`) as the swarm succeeds.
4.  **Reward Shaping:** 
    -   **Time Bonus:** Rewards finishing the task quickly.
    -   **Wrong Push Penalty:** Agents are penalized heavily if they apply a force that moves the object *away* from the target.
    -   **Success Multiplier:** A massive bonus when the object reaches the target radius.
5.  **Plateau Detection:** If fitness stops improving, the system automatically decays the learning rate and injects additional noise (`sigma`) to "kick" the population out of local minima.

---

## 4. 1:1 Web Parity (The Golden Standard)
To solve the "Exploding Physics" problem, external physics libraries (like `Matter.js`) were entirely removed from the frontend.
-   **Custom JS Engine:** The `web/src/hooks/usePhysics.ts` file now implements a custom Native JavaScript `requestAnimationFrame` loop that runs the **exact same math** as the Taichi Python script.
-   **Synchronized Integration:** The integration order (Force -> Acceleration -> Velocity -> Air Friction -> Position) is strictly mirrored, ensuring that the policy evolved in Python transfers flawlessly to the browser.

---

## 5. Emergent Tactics
-   **C-Shaped Wrapping:** Agents learn to form a concave shell around the object to prevent it from sliding sideways.
-   **Dynamic Braking:** Agents on the target side of the object learn to move out of the way or provide counter-pressure to stop the object exactly on the goal.
