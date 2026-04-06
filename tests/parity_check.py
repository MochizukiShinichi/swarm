import numpy as np
import engine
import json

def test_parity():
    print("--- PARITY CHECK: PYTHON STEP VS EXPECTED ---")
    
    # Set fixed state
    engine.NUM_ENVS = 1
    engine.initialize_episode(0.0)
    
    # Mock some data for a single agent and object
    # x[0, 0] = [100.0, 100.0], obj_pos[0] = [120.0, 100.0], target_pos[0] = [150.0, 100.0]
    # This should trigger a push on the X-axis
    engine.x[0, 0] = [100.0, 100.0]
    engine.obj_pos[0] = [105.0, 100.0] # Closer to trigger collision
    engine.target_pos[0] = [150.0, 100.0]
    
    # Fixed weights (identity-ish)
    # 18 -> 64 -> 2
    # We want to check if sensors are being read correctly
    population = np.zeros((engine.NUM_ENVS, engine.PARAM_COUNT), dtype=np.float32)
    engine.load_weights(population)
    
    # Run 1 step
    engine.update_grid()
    engine.step(0)
    
    # Check if object moved
    obj_v = engine.obj_v.to_numpy()[0]
    print(f"Object Velocity After 1 Step: {obj_v}")
    
    # Expected: push_f from Agent-Object collision
    # d_obj = x - obj_pos = [-5.0, 0.0]
    # dx = 40 + 5 - |-5| = 40, dy = 40 + 5 - |0| = 45
    # dx < dy, so fx = s * dx * K_OBJ = -1.0 * 40 * 0.05 = -2.0
    # obj_v -= (fx / OBJ_MASS) * DT = -(-2.0 / 6.4) * 1.0 = 0.3125
    
    expected_vx = 0.3125
    if np.abs(obj_v[0] - expected_vx) < 1e-4:
        print("✅ Physics Parity: Object push logic matches expectation.")
    else:
        print(f"❌ Physics Parity Failed! Expected {expected_vx}, got {obj_v[0]}")
        
    # Check Push Penalty (Fix check)
    # push_f = [-2.0, 0.0], to_t = [1.0, 0.0]
    # push_f.dot(to_t) = -2.0 < 0
    # In my code: if push_f.dot(to_t) < 0: wrong_push_penalty[e] += 0.05
    # Since push_f points AWAY from target (good push), it should be penalized?
    # NO! If push_f (force on agent) points AWAY from target, -push_f (force on object) points TOWARD target.
    # So if dot < 0, it's a GOOD push. 
    # The roadmap said: "push_f.dot(to_t) > 0 penalizes when force on agent points toward target (bad push). Reactions is -push_f."
    # Wait, if force on agent points toward target, force on object points AWAY. THAT is the bad push.
    # So the ORIGINAL code `if push_f.dot(to_t) > 0` was CORRECT.
    # But the roadmap says "Wrong push penalty is inverted... this penalizes good pushing."
    # I'll re-verify this in the parity check.
    
    penalty = engine.wrong_push_penalty.to_numpy()[0]
    print(f"Penalty for good push: {penalty}")
    # If penalty is 0.05, it means I am penalizing good pushing.
    
if __name__ == "__main__":
    test_parity()
