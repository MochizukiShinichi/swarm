import numpy as np
import engine
from optimizer import OpenAI_ES

def train():
    print(f"STARTING SWARM INTELLIGENCE TRAINING: {engine.NUM_ENVS} ENVS ({engine.POP_SIZE} POP x {engine.TRIALS_PER_CANDIDATE} TRIALS)")
    
    optimizer = OpenAI_ES(engine.PARAM_COUNT, engine.POP_SIZE)
    
    best_all_time_fit = -999999.0
    plateau_counter = 0
    difficulty = 0.0
    
    # Mastery Metrics (Stage 1.2)
    rolling_sr = []
    rolling_ef = []
    rolling_co = []
    window_size = 50

    for gen in range(1, 50001):
        # 1. Ask optimizer for population
        population = optimizer.ask()
        
        # 2. Tile population for multi-trial evaluation
        eval_population = np.repeat(population, engine.TRIALS_PER_CANDIDATE, axis=0)
        
        # 3. Load weights into engine
        engine.load_weights(eval_population)
        
        # 4. Run episode
        engine.initialize_episode(difficulty)
        for t in range(engine.EPISODE_STEPS):
            engine.update_grid()
            engine.step(t)
            
        # 5. Get raw results and reduce across trials
        raw_fitness, raw_reached = engine.get_fitness()
        fitness = raw_fitness.reshape(engine.POP_SIZE, engine.TRIALS_PER_CANDIDATE).mean(axis=1)
        reached = raw_reached.reshape(engine.POP_SIZE, engine.TRIALS_PER_CANDIDATE).mean(axis=1)
        
        sr = np.mean(reached)
        
        # Efficiency (Stage 1.2): 1 - mean(success_time / EPISODE_STEPS) for successful envs
        # Note: raw_reached used here to match original success_time indexing
        success_indices = np.where(raw_reached == 1)[0]
        ef = 0.0
        if len(success_indices) > 0:
            ef = 1.0 - np.mean(engine.success_time.to_numpy()[success_indices] / engine.EPISODE_STEPS)
            
        # Consistency (Stage 1.2): 1 - std(fitness) / mean(fitness)
        co = 0.0
        fit_mean = np.mean(fitness)
        if fit_mean != 0:
            co = 1.0 - np.std(fitness) / (np.abs(fit_mean) + 1e-8)
            
        # Update rolling metrics
        rolling_sr.append(sr)
        rolling_ef.append(ef)
        rolling_co.append(co)
        if len(rolling_sr) > window_size:
            rolling_sr.pop(0)
            rolling_ef.pop(0)
            rolling_co.pop(0)
            
        mean_sr = np.mean(rolling_sr)
        mean_ef = np.mean(rolling_ef)
        mean_co = np.mean(rolling_co)

        # 5. Tell optimizer the results
        optimizer.tell(fitness, gen)
        
        # 6. Logging & Checkpointing
        max_fit = np.max(fitness)
        if max_fit > best_all_time_fit:
            best_all_time_fit = max_fit
            plateau_counter = 0
        else:
            plateau_counter += 1

        if gen % 100 == 0:
            print(f"Gen {gen:05d} | SR: {mean_sr:.3f} | EF: {mean_ef:.3f} | CO: {mean_co:.3f} | Fit: {max_fit:.1f} | Diff: {difficulty:.2f}")
            engine.export_policy(optimizer.weights, "web/public/policy.json")

        # 7. Metric-Gated Curriculum (Stage 1.2 & 2.x)
        if len(rolling_sr) == window_size:
            # Phase thresholds change as difficulty increases
            target_sr = 0.95 if difficulty < 0.3 else 0.90
            target_ef = 0.50 if difficulty < 0.3 else 0.45
            target_co = 0.60 if difficulty < 0.3 else 0.55
            
            if mean_sr >= target_sr and mean_ef >= target_ef and mean_co >= target_co:
                if difficulty < 1.0:
                    difficulty += 0.1
                    print(f">>> MASTERY REACHED! Difficulty Increased to {difficulty:.2f} <<<")
                    rolling_sr, rolling_ef, rolling_co = [], [], []

        # 8. Plateau Detection
        if plateau_counter > 200:
            optimizer.adapt(plateau_counter)
            plateau_counter = 0
            
    print("TRAINING COMPLETE.")
    engine.export_policy(optimizer.weights, "web/src/assets/policy.json")

if __name__ == "__main__":
    train()
