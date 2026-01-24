import numpy as np
import gymnasium as gym
import random
import deep_q_learning_network_lib as dqn

# --- 1. Hyperparameters ---
ENV_NAME = "CartPole-v1"
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.95
ALPHA = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # How many episodes before syncing target net

# NN Structure: 4 inputs (state), two hidden layers of 24, 2 outputs (actions)
NN_STRUCTURE = [4, 24, 24, 2]
FUNCTIONS = [dqn.ReLU, dqn.ReLU, dqn.identity]
DERIVS = [dqn.deriv_ReLU, dqn.deriv_ReLU, dqn.deriv_identity]

# --- 2. Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, s, a, r, s_next, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s_next, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        # We need to transpose these so they become matrices (features, batch_size)
        s = np.array([x[0] for x in samples]).T
        a = np.array([x[1] for x in samples])
        r = np.array([x[2] for x in samples])
        s_next = np.array([x[3] for x in samples]).T
        done = np.array([x[4] for x in samples]).astype(int)
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)

# --- 3. Setup ---
env = gym.make(ENV_NAME)
memory = ReplayBuffer(10000)
epsilon = EPSILON_START

# Initialize Main and Target Networks
main_W, main_b = dqn.init_params(NN_STRUCTURE)
target_W, target_b = [w.copy() for w in main_W], [b.copy() for b in main_b]

# --- 4. Training Loop ---
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(500):
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # Reshape state to (4, 1) for your library
            s_vec = state.reshape(-1, 1)
            A_list, _ = dqn.forward_propogate(main_W, main_b, s_vec, FUNCTIONS)
            action = np.argmax(A_list[-1])

        # Step in Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Penality for failing (optional, helps CartPole learn faster)
        reward = reward if not terminated else -10
        
        # Save to Memory
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train Step
        if len(memory) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            main_W, main_b = dqn.train_step(
                main_W, main_b, target_W, target_b, 
                batch, FUNCTIONS, DERIVS, GAMMA, ALPHA
            )

        if done:
            break
    
    # Update Epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Sync Target Network weights
    if episode % TARGET_UPDATE_FREQ == 0:
        target_W = [w.copy() for w in main_W]
        target_b = [b.copy() for b in main_b]

    if episode % 10 == 0:
        print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()