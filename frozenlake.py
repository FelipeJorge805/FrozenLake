import random
import gym
import numpy as np

# Function to run SARSA training for a given set of hyperparameters
def run_training(alpha, gamma, epsilon, episodes=10000):
    env = gym.make('FrozenLake-v1', is_slippery=False)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    q_table = np.zeros((state_size, action_size))
    total_reward = 0

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        action = np.random.choice(action_size) if np.random.rand() < epsilon else np.argmax(q_table[state])

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = np.random.choice(action_size) if random.uniform(0, 1) < epsilon else np.argmax(q_table[next_state])

            # SARSA update
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

            state = next_state
            action = next_action
            total_reward += reward

        # Decay epsilon
        epsilon = max(0.1, epsilon * 0.999)

    env.close()
    return total_reward

# Default values for gamma and epsilon
default_alpha = 0.1
default_gamma = 0.9
default_epsilon = 0.9

# Hyperparameter grids to test independently
alpha_values = [0.1, 0.2, 0.3]
gamma_values = [0.9, 0.95, 1.0]
epsilon_values = [0.9, 0.8, 0.7]

# Table to store results: columns = alpha, gamma, epsilon, and total reward
results = []

# Test different alpha values while keeping gamma and epsilon fixed at their defaults
for alpha in alpha_values:
    total_reward = run_training(alpha, default_gamma, default_epsilon)
    results.append(('alpha', alpha, total_reward))

# Test different gamma values while keeping alpha and epsilon fixed at their defaults
for gamma in gamma_values:
    total_reward = run_training(default_alpha, gamma, default_epsilon)
    results.append(('gamma', gamma, total_reward))

# Test different epsilon values while keeping alpha and gamma fixed at their defaults
for epsilon in epsilon_values:
    total_reward = run_training(default_alpha, default_gamma, epsilon)
    results.append(('epsilon', epsilon, total_reward))

# Display results in a table-like format
print("\nResults (parameter, value, total_reward):")
for result in results:
    print(f"{result[0].capitalize()}: {result[1]}, Total Reward: {result[2]}")
