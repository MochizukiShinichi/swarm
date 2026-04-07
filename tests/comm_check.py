import numpy as np
import engine

def test_comm():
    print("--- COMM TEST: NEIGHBOR MESSAGING ---")
    engine.initialize_episode(0.0)
    
    # Place two agents close together
    engine.x[0, 0] = [100.0, 100.0]
    engine.x[0, 1] = [105.0, 100.0]
    
    # Set agent 0's message output manually
    engine.msg_out[0, 0, 0] = 1.0
    engine.msg_out[0, 0, 1] = 2.0
    engine.msg_out[0, 0, 2] = 3.0
    engine.msg_out[0, 0, 3] = 4.0
    
    # Update grid
    engine.update_grid()
    
    # Check if grid contains the message
    grid_msg = engine.grid_msg_sum.to_numpy()[0, 5, 5] # [100/20, 100/20]
    if np.sum(grid_msg) > 0:
        print(f"✅ Grid: Message registered in grid cell (Sum: {np.sum(grid_msg):.1f}).")
    else:
        print("❌ Grid: Message NOT found in grid cell.")
        
    # Check if agent 1 receives it in the next step
    # We need to run step() to see the inp being formed, but step() is a kernel.
    # We can check the Consensus logic inside step by mocking inputs.
    
if __name__ == "__main__":
    test_comm()
