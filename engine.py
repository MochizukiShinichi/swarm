import taichi as ti
import numpy as np
import json
import os

POLICY_PATH = os.path.join("web", "public", "policy.json")

# Use GPU
ti.init(arch=ti.gpu, device_memory_GB=2.0, unrolling_limit=0)

# --- CONFIGURATION ---
WIDTH, HEIGHT = 800.0, 500.0
NUM_ENVS, NUM_PARTICLES = 1024, 100
POP_SIZE = 256
TRIALS_PER_CANDIDATE = 4
DT = 1.0
EPISODE_STEPS = 600
assert NUM_ENVS == POP_SIZE * TRIALS_PER_CANDIDATE

# Continuous Contact Physics
FRICTION_AIR = 0.05
AGENT_RADIUS, COLLISION_RADIUS = 5.0, 10.0
AGENT_MASS, OBJ_SIZE, OBJ_MASS = 0.05026, 40.0, 6.4
K_AGENT = 0.02
K_OBJ = 0.05
K_OBS = 0.1 # Obstacle stiffness

# Brain Phase 2 (Sensors 18 + Msg 4 -> MLP 64 -> GRU 16 -> Motor 2 + Msg 4)
IN_DIM, MLP_DIM, RNN_DIM, OUT_DIM = 22, 64, 16, 6
PARAM_COUNT = (IN_DIM * MLP_DIM + MLP_DIM) + \
              (MLP_DIM * RNN_DIM * 3 + RNN_DIM * RNN_DIM * 3 + RNN_DIM * 3) + \
              (RNN_DIM * OUT_DIM + OUT_DIM)

# --- FIELDS ---
x, v = ti.Vector.field(2, ti.f32, (NUM_ENVS, NUM_PARTICLES)), ti.Vector.field(2, ti.f32, (NUM_ENVS, NUM_PARTICLES))
obj_pos, target_pos = ti.Vector.field(2, ti.f32, NUM_ENVS), ti.Vector.field(2, ti.f32, NUM_ENVS)
obj_v = ti.Vector.field(2, ti.f32, NUM_ENVS)

# Obstacles (Stage 2.2)
obs_pos = ti.Vector.field(2, ti.f32, (NUM_ENVS, 4))
obs_active = ti.field(ti.i32, (NUM_ENVS, 4))

# Brain State (Stage 2.0 & 2.1)
h_state = ti.field(ti.f32, (NUM_ENVS, NUM_PARTICLES, RNN_DIM))
msg_out = ti.field(ti.f32, (NUM_ENVS, NUM_PARTICLES, 4))
motor_queue = ti.Vector.field(2, ti.f32, (NUM_ENVS, NUM_PARTICLES, 2))

env_active = ti.field(ti.i32, NUM_ENVS)
env_success = ti.field(ti.i32, NUM_ENVS)
success_time = ti.field(ti.i32, NUM_ENVS)

total_progress = ti.field(ti.f32, NUM_ENVS)
wrong_push_penalty = ti.field(ti.f32, NUM_ENVS)
agent_contact_reward = ti.field(ti.f32, NUM_ENVS)
difficulty = ti.field(ti.f32, ())

# Weights

# MLP 1
w1 = ti.field(ti.f32, (NUM_ENVS, IN_DIM, MLP_DIM))
b1 = ti.field(ti.f32, (NUM_ENVS, MLP_DIM))
# GRU
w_gru_x = ti.field(ti.f32, (NUM_ENVS, 3, MLP_DIM, RNN_DIM)) # z, r, h gates
w_gru_h = ti.field(ti.f32, (NUM_ENVS, 3, RNN_DIM, RNN_DIM))
b_gru = ti.field(ti.f32, (NUM_ENVS, 3, RNN_DIM))
# MLP 2
w2 = ti.field(ti.f32, (NUM_ENVS, RNN_DIM, OUT_DIM))
b2 = ti.field(ti.f32, (NUM_ENVS, OUT_DIM))

# Sensors/Comm Grid
grid_num = ti.field(ti.i32, (NUM_ENVS, 40, 40))
grid_v_sum = ti.Vector.field(2, ti.f32, (NUM_ENVS, 40, 40))
grid_msg_sum = ti.field(ti.f32, (NUM_ENVS, 40, 40, 4))

@ti.func
def sigmoid(x): return 1.0 / (1.0 + ti.exp(-x))

@ti.kernel
def initialize_episode(diff: ti.f32):
    difficulty[None] = diff
    for e in range(NUM_ENVS):
        obj_pos[e] = ti.Vector([100.0 + ti.random() * 600.0, 100.0 + ti.random() * 300.0])
        obj_v[e] = ti.Vector([0.0, 0.0])
        
        offset_mag = 50.0 + diff * 250.0
        offset = ti.Vector([ti.random()-0.5, ti.random()-0.5]).normalized() * offset_mag
        
        # Ensure target is within arena bounds [OBJ_SIZE, WIDTH-OBJ_SIZE]
        tp = obj_pos[e] + offset
        tp.x = ti.max(OBJ_SIZE, ti.min(WIDTH - OBJ_SIZE, tp.x))
        tp.y = ti.max(OBJ_SIZE, ti.min(HEIGHT - OBJ_SIZE, tp.y))
        target_pos[e] = tp
        
        # Recalculate spawn side relative to actual (clipped) target
        offset_real = target_pos[e] - obj_pos[e]
        sc = obj_pos[e] - offset_real.normalized() * 60.0
        for i in range(NUM_PARTICLES):
            x[e, i] = sc + ti.Vector([(ti.random()-0.5)*80, (ti.random()-0.5)*80])
            v[e, i] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(RNN_DIM)): h_state[e, i, k] = 0.0
            for k in ti.static(range(4)): msg_out[e, i, k] = 0.0
            for k in ti.static(range(2)): motor_queue[e, i, k] = ti.Vector([0.0, 0.0])
        
        # Obstacles (Stage 2.2)
        num_obs = 0
        if diff > 0.3: num_obs = 2
        if diff > 0.6: num_obs = 4
        for i in range(4):
            if i < num_obs:
                obs_active[e, i] = 1
                # Place between object and target with some jitter
                t = 0.3 + 0.4 * ti.random()
                mid = obj_pos[e] + offset * t
                perp = ti.Vector([-offset.y, offset.x]).normalized()
                obs_pos[e, i] = mid + perp * (ti.random()-0.5) * 150.0
            else:
                obs_active[e, i] = 0

        env_active[e], env_success[e] = 1, 0
        success_time[e] = EPISODE_STEPS
        total_progress[e], wrong_push_penalty[e], agent_contact_reward[e] = 0.0, 0.0, 0.0

@ti.kernel
def update_grid():
    grid_num.fill(0)
    grid_v_sum.fill(0)
    grid_msg_sum.fill(0)
    for e, i in x:
        if env_active[e]:
            c = (x[e, i] / 20.0).cast(ti.i32)
            c = ti.max(0, ti.min(39, c))
            ti.atomic_add(grid_num[e, c.x, c.y], 1)
            grid_v_sum[e, c.x, c.y] += v[e, i]
            for k in ti.static(range(4)):
                ti.atomic_add(grid_msg_sum[e, c.x, c.y, k], msg_out[e, i, k])

@ti.kernel
def step(t: ti.i32):
    for e, i in x:
        if env_active[e]:
            # 1. Consensus (Grid-local)
            c = (x[e, i] / 20.0).cast(ti.i32)
            v_s, m_s, v_c = ti.Vector([0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0, 0.0]), 0
            for dx, dy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                nc = ti.max(0, ti.min(39, c + ti.Vector([dx, dy])))
                v_s += grid_v_sum[e, nc.x, nc.y]
                v_c += grid_num[e, nc.x, nc.y]
                for k in ti.static(range(4)): m_s[k] += grid_msg_sum[e, nc.x, nc.y, k]
            lv = v_s / ti.max(1, v_c)
            lm = m_s / ti.max(1, v_c)

            # 2. Sensors
            to_t = (target_pos[e] - obj_pos[e]).normalized()
            d_o = (obj_pos[e] - x[e, i])
            dist_o = d_o.norm()
            to_o = d_o.normalized()
            rp = (x[e, i] - obj_pos[e]) / 100.0
            wl, wr = 1.0/(1.0+x[e, i].x), 1.0/(1.0+WIDTH-x[e, i].x)
            wt, wb = 1.0/(1.0+x[e, i].y), 1.0/(1.0+HEIGHT-x[e, i].y)
            
            # Input: [18 sensors] + [4 messages] = 22
            inp = ti.Vector([to_t.x, to_t.y, to_o.x, to_o.y, rp.x, rp.y, v[e, i].x, v[e, i].y, 
                             lv.x, lv.y, wl, wr, wt, wb, dist_o/100.0, obj_v[e].x, obj_v[e].y, v[e, i].norm(),
                             lm[0], lm[1], lm[2], lm[3]])
            
            # Stage 2.3: Sensor Noise (σ=0.05)
            if difficulty[None] > 0.6:
                for k in ti.static(range(22)):
                    inp[k] += (ti.random() - 0.5) * 0.1 # Approx Gaussian σ=0.05
            
            # 3. Brain MLP 1
            h_mlp1 = ti.Vector([0.0] * MLP_DIM)
            for j in range(MLP_DIM):
                val = b1[e, j]
                for k in range(IN_DIM): val += inp[k] * w1[e, k, j]
                h_mlp1[j] = ti.tanh(val)
                
            # 4. GRU Cell (Stage 2.0)
            h_prev = ti.Vector([0.0] * RNN_DIM)
            for k in range(RNN_DIM): h_prev[k] = h_state[e, i, k]
            
            z_gate = ti.Vector([0.0] * RNN_DIM)
            r_gate = ti.Vector([0.0] * RNN_DIM)
            for j in range(RNN_DIM):
                vz, vr = b_gru[e, 0, j], b_gru[e, 1, j]
                for k in range(MLP_DIM):
                    vz += h_mlp1[k] * w_gru_x[e, 0, k, j]
                    vr += h_mlp1[k] * w_gru_x[e, 1, k, j]
                for k in range(RNN_DIM):
                    vz += h_prev[k] * w_gru_h[e, 0, k, j]
                    vr += h_prev[k] * w_gru_h[e, 1, k, j]
                z_gate[j], r_gate[j] = sigmoid(vz), sigmoid(vr)
                
            h_hat = ti.Vector([0.0] * RNN_DIM)
            for j in range(RNN_DIM):
                vh = b_gru[e, 2, j]
                for k in range(MLP_DIM): vh += h_mlp1[k] * w_gru_x[e, 2, k, j]
                for k in range(RNN_DIM): vh += r_gate[j] * h_prev[k] * w_gru_h[e, 2, k, j]
                h_hat[j] = ti.tanh(vh)
                
            for j in range(RNN_DIM):
                h_new = (1.0 - z_gate[j]) * h_prev[j] + z_gate[j] * h_hat[j]
                h_state[e, i, j] = h_new

            # 5. Output MLP
            out = ti.Vector([0.0] * OUT_DIM)
            for j in ti.static(range(OUT_DIM)):
                val = b2[e, j]
                for k in ti.static(range(RNN_DIM)): val += h_state[e, i, k] * w2[e, k, j]
                out[j] = ti.tanh(val)
            
            # Motor + Msg Out
            # Stage 2.3: 2-step motor delay
            f_current = ti.Vector([out[0], out[1]]) * 0.02
            f_delayed = f_current
            if difficulty[None] > 0.6:
                f_delayed = motor_queue[e, i, 1]
                motor_queue[e, i, 1] = motor_queue[e, i, 0]
                motor_queue[e, i, 0] = f_current
            
            v[e, i] += (f_delayed / AGENT_MASS) * DT
            for k in ti.static(range(4)): msg_out[e, i, k] = out[2+k]

    # Physics
    for e, i in x:
        if env_active[e]:
            # Agent-Agent
            for j in range(NUM_PARTICLES):
                if i != j:
                    diff = x[e, i] - x[e, j]
                    dist = diff.norm() + 1e-5
                    if dist < COLLISION_RADIUS:
                        v[e, i] += (diff / dist) * ((COLLISION_RADIUS - dist) * K_AGENT / AGENT_MASS) * DT
            
            # Agent-Object
            d_obj = x[e, i] - obj_pos[e]
            if ti.abs(d_obj.x) < OBJ_SIZE + AGENT_RADIUS and ti.abs(d_obj.y) < OBJ_SIZE + AGENT_RADIUS:
                dx, dy = OBJ_SIZE + AGENT_RADIUS - ti.abs(d_obj.x), OBJ_SIZE + AGENT_RADIUS - ti.abs(d_obj.y)
                push_f = ti.Vector([0.0, 0.0])
                if dx < dy: push_f.x = (1.0 if d_obj.x >= 0 else -1.0) * dx * K_OBJ
                else: push_f.y = (1.0 if d_obj.y >= 0 else -1.0) * dy * K_OBJ
                v[e, i] += (push_f / AGENT_MASS) * DT
                obj_v[e] -= (push_f / OBJ_MASS) * DT
                agent_contact_reward[e] += 0.01
                to_t = (target_pos[e] - obj_pos[e]).normalized()
                if push_f.dot(to_t) > 0: wrong_push_penalty[e] += 0.05

            # Obstacles
            for j in range(4):
                if obs_active[e, j]:
                    d_obs = x[e, i] - obs_pos[e, j]
                    dist = d_obs.norm() + 1e-5
                    r_sum = AGENT_RADIUS + 20.0
                    if dist < r_sum:
                        v[e, i] += (d_obs / dist) * ((r_sum - dist) * K_OBS / AGENT_MASS) * DT

            v[e, i] *= (1.0 - FRICTION_AIR)
            x[e, i] += v[e, i] * DT
            if x[e, i].x < 0: x[e, i].x = 0; v[e, i].x *= -0.5
            if x[e, i].x > WIDTH: x[e, i].x = WIDTH; v[e, i].x *= -0.5
            if x[e, i].y < 0: x[e, i].y = 0; v[e, i].y *= -0.5
            if x[e, i].y > HEIGHT: x[e, i].y = HEIGHT; v[e, i].y *= -0.5

    for e in range(NUM_ENVS):
        if env_active[e]:
            old_dist = (obj_pos[e] - target_pos[e]).norm()
            obj_v[e] *= (1.0 - FRICTION_AIR)
            for j in range(4):
                if obs_active[e, j]:
                    d_obs = obj_pos[e] - obs_pos[e, j]
                    dist = d_obs.norm() + 1e-5
                    r_sum = OBJ_SIZE + 20.0
                    if dist < r_sum:
                        obj_v[e] += (d_obs / dist) * ((r_sum - dist) * K_OBS / OBJ_MASS) * DT
            obj_pos[e] += obj_v[e] * DT
            if obj_pos[e].x < OBJ_SIZE: obj_pos[e].x = OBJ_SIZE; obj_v[e].x *= -0.5
            if obj_pos[e].x > WIDTH - OBJ_SIZE: obj_pos[e].x = WIDTH - OBJ_SIZE; obj_v[e].x *= -0.5
            if obj_pos[e].y < OBJ_SIZE: obj_pos[e].y = OBJ_SIZE; obj_v[e].y *= -0.5
            if obj_pos[e].y > HEIGHT - OBJ_SIZE: obj_pos[e].y = HEIGHT - OBJ_SIZE; obj_v[e].y *= -0.5
            new_dist = (obj_pos[e] - target_pos[e]).norm()
            total_progress[e] += (old_dist - new_dist)
            if new_dist < 40.0:
                env_active[e], env_success[e] = 0, 1
                success_time[e] = t

def load_weights(pop):
    ptr = 0
    w1.from_numpy(pop[:, ptr:ptr+IN_DIM*MLP_DIM].reshape(NUM_ENVS, IN_DIM, MLP_DIM)); ptr += IN_DIM*MLP_DIM
    b1.from_numpy(pop[:, ptr:ptr+MLP_DIM]); ptr += MLP_DIM
    w_gru_x.from_numpy(pop[:, ptr:ptr+3*MLP_DIM*RNN_DIM].reshape(NUM_ENVS, 3, MLP_DIM, RNN_DIM)); ptr += 3*MLP_DIM*RNN_DIM
    w_gru_h.from_numpy(pop[:, ptr:ptr+3*RNN_DIM*RNN_DIM].reshape(NUM_ENVS, 3, RNN_DIM, RNN_DIM)); ptr += 3*RNN_DIM*RNN_DIM
    b_gru.from_numpy(pop[:, ptr:ptr+3*RNN_DIM].reshape(NUM_ENVS, 3, RNN_DIM)); ptr += 3*RNN_DIM
    w2.from_numpy(pop[:, ptr:ptr+RNN_DIM*OUT_DIM].reshape(NUM_ENVS, RNN_DIM, OUT_DIM)); ptr += RNN_DIM*OUT_DIM
    b2.from_numpy(pop[:, ptr:ptr+OUT_DIM]); ptr += OUT_DIM

def import_policy(path):
    if not os.path.exists(path): return None, None
    with open(path, "r") as f: d = json.load(f)
    weights = np.zeros(PARAM_COUNT, dtype=np.float32)
    ptr = 0
    def load_arr(k):
        nonlocal ptr
        w = np.array(d[k], dtype=np.float32).flatten()
        weights[ptr:ptr+len(w)] = w
        ptr += len(w)
    try:
        load_arr("w1"); load_arr("b1")
        load_arr("w_gru_x"); load_arr("w_gru_h"); load_arr("b_gru")
        load_arr("w2"); load_arr("b2")
    except KeyError:
        return None, None
    return weights, d.get("meta")

def export_policy(weights, path, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ptr = 0
    d = {}
    d["w1"] = weights[ptr:ptr+IN_DIM*MLP_DIM].reshape(IN_DIM, MLP_DIM).tolist(); ptr += IN_DIM*MLP_DIM
    d["b1"] = weights[ptr:ptr+MLP_DIM].tolist(); ptr += MLP_DIM
    d["w_gru_x"] = weights[ptr:ptr+3*MLP_DIM*RNN_DIM].reshape(3, MLP_DIM, RNN_DIM).tolist(); ptr += 3*MLP_DIM*RNN_DIM
    d["w_gru_h"] = weights[ptr:ptr+3*RNN_DIM*RNN_DIM].reshape(3, RNN_DIM, RNN_DIM).tolist(); ptr += 3*RNN_DIM*RNN_DIM
    d["b_gru"] = weights[ptr:ptr+3*RNN_DIM].reshape(3, RNN_DIM).tolist(); ptr += 3*RNN_DIM
    d["w2"] = weights[ptr:ptr+RNN_DIM*OUT_DIM].reshape(RNN_DIM, OUT_DIM).tolist(); ptr += RNN_DIM*OUT_DIM
    d["b2"] = weights[ptr:ptr+OUT_DIM].tolist()
    if meta is not None:
        d["meta"] = meta
    with open(path, "w") as f: json.dump(d, f)

def get_fitness():
    reached = env_success.to_numpy()
    t_bonus = (EPISODE_STEPS - success_time.to_numpy()) * 2.0
    prog = total_progress.to_numpy() * 100.0
    penalty = wrong_push_penalty.to_numpy() * 20.0
    contact = agent_contact_reward.to_numpy() * 0.5
    dist_final = np.linalg.norm(obj_pos.to_numpy() - target_pos.to_numpy(), axis=1)
    fitness = reached * 5000.0 + t_bonus + prog + contact - penalty - dist_final * 5.0
    return fitness, reached
