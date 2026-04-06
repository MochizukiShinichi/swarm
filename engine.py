import taichi as ti
import numpy as np
import json
import os

# Use GPU, disable unrolling limits for fast compilation of big loops
ti.init(arch=ti.gpu, device_memory_GB=2.0)

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800.0, 500.0
NUM_ENVS, NUM_PARTICLES = 256, 100
DT = 1.0
EPISODE_STEPS = 600

# Continuous Contact Physics (Hooke's Law)
FRICTION_AIR = 0.05
AGENT_RADIUS, COLLISION_RADIUS = 5.0, 10.0
AGENT_MASS, OBJ_SIZE, OBJ_MASS = 0.05026, 40.0, 6.4
K_AGENT = 0.02 # Agent-Agent spring stiffness
K_OBJ = 0.05   # Agent-Object spring stiffness

# Brain (18 -> 64 -> 2)
IN_DIM, HIDDEN_DIM, OUT_DIM = 18, 64, 2
PARAM_COUNT = (IN_DIM * HIDDEN_DIM + HIDDEN_DIM) + (HIDDEN_DIM * OUT_DIM + OUT_DIM)

# --- FIELDS ---
x, v = ti.Vector.field(2, ti.f32, (NUM_ENVS, NUM_PARTICLES)), ti.Vector.field(2, ti.f32, (NUM_ENVS, NUM_PARTICLES))
obj_pos, target_pos = ti.Vector.field(2, ti.f32, NUM_ENVS), ti.Vector.field(2, ti.f32, NUM_ENVS)
obj_v = ti.Vector.field(2, ti.f32, NUM_ENVS)

env_active = ti.field(ti.i32, NUM_ENVS)
env_success = ti.field(ti.i32, NUM_ENVS)
success_time = ti.field(ti.i32, NUM_ENVS)

total_progress = ti.field(ti.f32, NUM_ENVS)
wrong_push_penalty = ti.field(ti.f32, NUM_ENVS)
agent_contact_reward = ti.field(ti.f32, NUM_ENVS)

# Weights
w1 = ti.field(ti.f32, (NUM_ENVS, IN_DIM, HIDDEN_DIM))
b1 = ti.field(ti.f32, (NUM_ENVS, HIDDEN_DIM))
w2 = ti.field(ti.f32, (NUM_ENVS, HIDDEN_DIM, OUT_DIM))
b2 = ti.field(ti.f32, (NUM_ENVS, OUT_DIM))

# Sensors
grid_v_sum = ti.Vector.field(2, ti.f32, (NUM_ENVS, 40, 40))
grid_num = ti.field(ti.i32, (NUM_ENVS, 40, 40))

@ti.kernel
def initialize_episode(difficulty: ti.f32):
    for e in range(NUM_ENVS):
        obj_pos[e] = ti.Vector([100.0 + ti.random() * 600.0, 100.0 + ti.random() * 300.0])
        obj_v[e] = ti.Vector([0.0, 0.0])
        
        # Difficulty scales distance from 50px to 300px
        offset_mag = 50.0 + difficulty * 250.0
        offset = ti.Vector([ti.random()-0.5, ti.random()-0.5]).normalized() * offset_mag
        target_pos[e] = obj_pos[e] + offset
        
        # Spawn agents clustered on the opposite side of the object from the target
        sc = obj_pos[e] - offset.normalized() * 60.0
        for i in range(NUM_PARTICLES):
            x[e, i] = sc + ti.Vector([(ti.random()-0.5)*80, (ti.random()-0.5)*80])
            v[e, i] = ti.Vector([0.0, 0.0])
        
        env_active[e] = 1
        env_success[e] = 0
        success_time[e] = EPISODE_STEPS
        total_progress[e] = 0.0
        wrong_push_penalty[e] = 0.0
        agent_contact_reward[e] = 0.0

@ti.kernel
def update_grid():
    grid_num.fill(0)
    grid_v_sum.fill(0)
    for e, i in x:
        if env_active[e]:
            c = (x[e, i] / 20.0).cast(ti.i32)
            c = ti.max(0, ti.min(39, c))
            ti.atomic_add(grid_num[e, c.x, c.y], 1)
            grid_v_sum[e, c.x, c.y] += v[e, i]

@ti.kernel
def step(t: ti.i32):
    for e, i in x:
        if env_active[e]:
            # 1. Local Consensus
            c = (x[e, i] / 20.0).cast(ti.i32)
            v_s, v_c = ti.Vector([0.0, 0.0]), 0
            for dx, dy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                nc = ti.max(0, ti.min(39, c + ti.Vector([dx, dy])))
                v_s += grid_v_sum[e, nc.x, nc.y]
                v_c += grid_num[e, nc.x, nc.y]
            lv = v_s / ti.max(1, v_c)

            # 2. Sensors
            to_t = (target_pos[e] - obj_pos[e]).normalized()
            d_o = (obj_pos[e] - x[e, i])
            dist_o = d_o.norm()
            to_o = d_o.normalized()
            rp = (x[e, i] - obj_pos[e]) / 100.0
            wl, wr = 1.0/(1.0+x[e, i].x), 1.0/(1.0+WIDTH-x[e, i].x)
            wt, wb = 1.0/(1.0+x[e, i].y), 1.0/(1.0+HEIGHT-x[e, i].y)
            
            # Stage 1.1: distance-to-object, object velocity direction, agent density in forward cone
            inp = ti.Vector([to_t.x, to_t.y, to_o.x, to_o.y, rp.x, rp.y, v[e, i].x, v[e, i].y, 
                             lv.x, lv.y, wl, wr, wt, wb, dist_o/100.0, obj_v[e].x, obj_v[e].y, v[e, i].norm()])
            
            # 3. Brain MLP
            h1 = ti.Vector([0.0] * HIDDEN_DIM)
            for j in range(HIDDEN_DIM):
                val = b1[e, j]
                for k in ti.static(range(IN_DIM)): val += inp[k] * w1[e, k, j]
                h1[j] = ti.tanh(val)
                
            out = ti.Vector([0.0] * OUT_DIM)
            for j in range(OUT_DIM):
                val = b2[e, j]
                for k in range(HIDDEN_DIM): val += h1[k] * w2[e, k, j]
                out[j] = ti.tanh(val)
            
            # Agent Force
            f_agent = out * 0.02
            v[e, i] += (f_agent / AGENT_MASS) * DT

    # 4. Continuous Spring Physics (Penalty-Based Contact)
    for e, i in x:
        if env_active[e]:
            # Agent-Agent Soft Collision
            c = (x[e, i] / 20.0).cast(ti.i32)
            for j in range(NUM_PARTICLES):
                if i != j:
                    diff = x[e, i] - x[e, j]
                    dist = diff.norm() + 1e-5
                    if dist < COLLISION_RADIUS:
                        overlap = COLLISION_RADIUS - dist
                        push_f = (diff / dist) * (overlap * K_AGENT)
                        v[e, i] += (push_f / AGENT_MASS) * DT
            
            # Agent-Object Soft Collision
            d_obj = x[e, i] - obj_pos[e]
            if ti.abs(d_obj.x) < OBJ_SIZE + AGENT_RADIUS and ti.abs(d_obj.y) < OBJ_SIZE + AGENT_RADIUS:
                # Calculate penetration vector
                dx = OBJ_SIZE + AGENT_RADIUS - ti.abs(d_obj.x)
                dy = OBJ_SIZE + AGENT_RADIUS - ti.abs(d_obj.y)
                
                push_f = ti.Vector([0.0, 0.0])
                if dx < dy:
                    s = 1.0 if d_obj.x >= 0 else -1.0
                    push_f.x = s * dx * K_OBJ
                else:
                    s = 1.0 if d_obj.y >= 0 else -1.0
                    push_f.y = s * dy * K_OBJ
                
                # Apply spring forces
                v[e, i] += (push_f / AGENT_MASS) * DT
                obj_v[e] -= (push_f / OBJ_MASS) * DT
                
                agent_contact_reward[e] += 0.01
                
                # Reward Shaping: Penalize counter-pushing
                to_t = (target_pos[e] - obj_pos[e]).normalized()
                # If push_f.dot(to_t) > 0, the agent is between object and target (bad).
                if push_f.dot(to_t) > 0: 
                    wrong_push_penalty[e] += 0.05

            # Agent Integration
            v[e, i] *= (1.0 - FRICTION_AIR)
            x[e, i] += v[e, i] * DT
            
            # Boundaries
            if x[e, i].x < 0: x[e, i].x = 0; v[e, i].x *= -0.5
            if x[e, i].x > WIDTH: x[e, i].x = WIDTH; v[e, i].x *= -0.5
            if x[e, i].y < 0: x[e, i].y = 0; v[e, i].y *= -0.5
            if x[e, i].y > HEIGHT: x[e, i].y = HEIGHT; v[e, i].y *= -0.5

    # Object Integration
    for e in range(NUM_ENVS):
        if env_active[e]:
            old_dist = (obj_pos[e] - target_pos[e]).norm()
            
            obj_v[e] *= (1.0 - FRICTION_AIR)
            obj_pos[e] += obj_v[e] * DT
            
            # Object Boundaries
            if obj_pos[e].x < OBJ_SIZE: obj_pos[e].x = OBJ_SIZE; obj_v[e].x *= -0.5
            if obj_pos[e].x > WIDTH - OBJ_SIZE: obj_pos[e].x = WIDTH - OBJ_SIZE; obj_v[e].x *= -0.5
            if obj_pos[e].y < OBJ_SIZE: obj_pos[e].y = OBJ_SIZE; obj_v[e].y *= -0.5
            if obj_pos[e].y > HEIGHT - OBJ_SIZE: obj_pos[e].y = HEIGHT - OBJ_SIZE; obj_v[e].y *= -0.5
            
            new_dist = (obj_pos[e] - target_pos[e]).norm()
            total_progress[e] += (old_dist - new_dist)
            
            # Success Detection
            if new_dist < 40.0:
                env_active[e] = 0
                env_success[e] = 1
                success_time[e] = t

def load_weights(population):
    w1.from_numpy(population[:, :IN_DIM*HIDDEN_DIM].reshape(NUM_ENVS, IN_DIM, HIDDEN_DIM))
    ptr = IN_DIM*HIDDEN_DIM
    b1.from_numpy(population[:, ptr:ptr+HIDDEN_DIM]); ptr += HIDDEN_DIM
    w2.from_numpy(population[:, ptr:ptr+HIDDEN_DIM*OUT_DIM].reshape(NUM_ENVS, HIDDEN_DIM, OUT_DIM)); ptr += HIDDEN_DIM*OUT_DIM
    b2.from_numpy(population[:, ptr:ptr+OUT_DIM])

def export_policy(weights, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ptr = 0
    w1_v = weights[ptr:ptr+IN_DIM*HIDDEN_DIM].reshape(IN_DIM, HIDDEN_DIM).tolist(); ptr += IN_DIM*HIDDEN_DIM
    b1_v = weights[ptr:ptr+HIDDEN_DIM].tolist(); ptr += HIDDEN_DIM
    w2_v = weights[ptr:ptr+HIDDEN_DIM*OUT_DIM].reshape(HIDDEN_DIM, OUT_DIM).tolist(); ptr += HIDDEN_DIM*OUT_DIM
    b2_v = weights[ptr:ptr+OUT_DIM].tolist()
    with open(path, "w") as f: 
        json.dump({"W1": w1_v, "b1": b1_v, "W2": w2_v, "b2": b2_v}, f)

def get_fitness():
    reached = env_success.to_numpy()
    t_bonus = (EPISODE_STEPS - success_time.to_numpy()) * 2.0
    prog = total_progress.to_numpy() * 100.0
    penalty = wrong_push_penalty.to_numpy() * 20.0
    contact = agent_contact_reward.to_numpy() * 0.5
    dist_final = np.linalg.norm(obj_pos.to_numpy() - target_pos.to_numpy(), axis=1)
    
    fitness = reached * 5000.0 + t_bonus + prog + contact - penalty - dist_final * 5.0
    return fitness, reached
