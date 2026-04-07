import numpy as np
import engine

def test_memory():
    print("--- MEMORY TEST: GRU HIDDEN STATE PERSISTENCE ---")
    engine.initialize_episode(0.0)
    
    # 1. Check if hidden states are zero initially
    h = engine.h_state.to_numpy()[0, 0]
    if np.all(h == 0):
        print("✅ Init: Hidden states are zero.")
    else:
        print("❌ Init: Hidden states NOT zero.")
        
    # 2. Run a step with fixed weights and see if hidden state changes
    population = np.ones((engine.NUM_ENVS, engine.PARAM_COUNT), dtype=np.float32) * 0.1
    engine.load_weights(population)
    
    engine.update_grid()
    engine.step(0)
    
    h_after = engine.h_state.to_numpy()[0, 0]
    if not np.all(h_after == 0):
        print(f"✅ Step: Hidden state changed (Mean: {np.mean(h_after):.4f}).")
    else:
        print("❌ Step: Hidden state remained zero despite input.")
        
    # 3. Check if hidden state persists to next step
    h_copy = h_after.copy()
    engine.update_grid()
    engine.step(1)
    h_step2 = engine.h_state.to_numpy()[0, 0]
    if not np.all(h_step2 == h_copy):
        print("✅ Persistence: Hidden state updated from previous state.")
    else:
        print("❌ Persistence: Hidden state did not update.")

if __name__ == "__main__":
    test_memory()
