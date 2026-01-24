import numpy as np
import gymnasium as gym
import random
import deep_q_learning_network_lib as dqn

# --- 1. Hyperparameters for LunarLander ---
ENV_NAME = "LunarLander-v3" # v2 is the standard; v3 is very new/identical in logic
EPISODES = 500            # LunarLander takes longer to learn than CartPole
BATCH_SIZE = 64            # Increased for more stable gradients
GAMMA = 0.99               # High gamma to care more about the final landing
ALPHA = 0.0005             # Slightly lower learning rate for stability
EPSILON_START = 1.0
EPSILON_END = 0.05         # Leave a bit more exploration
EPSILON_DECAY = 0.996      # Slower decay to allow more time to discover the landing pad
TARGET_UPDATE_FREQ = 10    

# NN Structure: 
# Input: 8 (x, y, v_x, v_y, angle, angular_v, leg_L, leg_R)
# Output: 4 (Do nothing, Main engine, Left engine, Right engine)
NN_STRUCTURE = [8, 128, 64, 128, 4] # Larger hidden layers for more complex physics
FUNCTIONS = [dqn.ReLU, dqn.ReLU, dqn.ReLU, dqn.identity]
DERIVS = [dqn.deriv_ReLU, dqn.deriv_ReLU, dqn.deriv_ReLU, dqn.deriv_identity]

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
memory = ReplayBuffer(50000) # Increased memory for more diverse experiences
epsilon = EPSILON_START
scores = []

# Initialize Main and Target Networks
main_W, main_b = dqn.init_params(NN_STRUCTURE)
target_W, target_b = [w.copy() for w in main_W], [b.copy() for b in main_b]

# --- 4. Training Loop ---
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    
    # LunarLander can take up to 1000 steps per episode
    for step in range(1000):
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            s_vec = state.reshape(-1, 1) # Reshaped to (8, 1)
            A_list, _ = dqn.forward_propogate(main_W, main_b, s_vec, FUNCTIONS)
            action = np.argmax(A_list[-1])

        # Step in Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # NOTE: For LunarLander, we don't need manual reward shaping.
        # The environment already gives -100 for crashing and +100 for landing.
        
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
        avg_score = np.mean(scores[-10:]) if scores else total_reward
        print(f"Episode: {episode}, Score: {total_reward:.2f}, Avg (10): {avg_score:.2f}, Epsilon: {epsilon:.2f}")
    
    scores.append(total_reward)

env.close()

# --- 5. Post-Training Victory Lap ---
print("\n--- Training Complete! Starting Victory Lap ---")
test_env = gym.make(ENV_NAME, render_mode="human")

for i in range(5):
    state, _ = test_env.reset()
    done = False
    score = 0
    while not done:
        s_vec = state.reshape(-1, 1)
        A_list, _ = dqn.forward_propogate(main_W, main_b, s_vec, FUNCTIONS)
        action = np.argmax(A_list[-1])
        state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        score += reward # type: ignore
    print(f"Victory Lap {i+1} Score: {score:.2f}")

test_env.close()