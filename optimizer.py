import numpy as np

class OpenAI_ES:
    def __init__(self, param_count, num_envs, lr=0.05, sigma=0.1, weight_decay=0.01):
        self.param_count = param_count
        self.num_envs = num_envs
        self.lr = lr
        self.sigma = sigma
        self.weight_decay = weight_decay
        
        self.weights = np.random.randn(param_count).astype(np.float32) * 0.1
        self.m = np.zeros(param_count, dtype=np.float32)
        self.v = np.zeros(param_count, dtype=np.float32)
        self.beta1, self.beta2 = 0.9, 0.999
        self.epsilon = 1e-8
        
    def ask(self):
        half_pop = self.num_envs // 2
        self.noise = np.random.randn(half_pop, self.param_count).astype(np.float32)
        noise_matrix = np.vstack([self.noise, -self.noise])
        population = self.weights + self.sigma * noise_matrix
        return population
        
    def tell(self, fitness, gen):
        # Rank-based fitness transformation
        order = np.argsort(fitness)[::-1]
        ranks = np.zeros(self.num_envs, dtype=np.float32)
        ranks[order] = np.linspace(0.5, -0.5, self.num_envs)
        
        # Gradient Estimate
        noise_matrix = np.vstack([self.noise, -self.noise])
        gradient = (1.0 / (self.num_envs * self.sigma)) * np.dot(ranks, noise_matrix)
        
        # Weight Decay (Stage 1.0)
        gradient -= self.weight_decay * self.weights
        
        # Adam Update
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1.0 - self.beta1**gen)
        v_hat = self.v / (1.0 - self.beta2**gen)
        self.weights += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
    def adapt(self, plateau_counter):
        # Sigma Adaptation (Stage 1.0)
        print(f">>> Plateau Detected ({plateau_counter}). Adapting Hyperparams. <<<")
        self.lr = max(0.005, self.lr * 0.8)
        self.sigma = min(0.3, self.sigma * 1.2)
