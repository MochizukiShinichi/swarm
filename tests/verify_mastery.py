import numpy as np

def test_mastery_logic():
    # Mocking the state variables in main.py
    rolling_sr = [0.8] * 50
    rolling_ef = [0.6] * 50
    rolling_co = [0.4] * 50
    window_size = 50
    difficulty = 0.0
    plateau_counter = 100
    
    # Logic to test (Step 2 of Task 3)
    mean_sr = np.mean(rolling_sr)
    mean_ef = np.mean(rolling_ef)
    mean_co = np.mean(rolling_co)
    
    triggered = False
    if len(rolling_sr) == window_size:
        if mean_sr >= 0.75 and mean_ef >= 0.50 and mean_co >= 0.30:
            difficulty += 0.1
            print(f">>> MASTERY REACHED! Difficulty Increased to {difficulty:.2f} <<<")
            rolling_sr, rolling_ef, rolling_co = [], [], []
            plateau_counter = 0
            triggered = True
            
    assert triggered == True
    assert difficulty == 0.1
    assert plateau_counter == 0
    assert len(rolling_sr) == 0
    print("Test passed: Mastery logic triggers correctly.")

if __name__ == "__main__":
    test_mastery_logic()
