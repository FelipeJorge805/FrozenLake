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

# Defaults
default_alpha = 0.1
default_gamma = 0.9
default_epsilon = 0.9

# Hyperparameter values to test
alpha_values = [0.12, 0.15, 0.17]
gamma_values = [0.85, 0.90, 0.95]
epsilon_values = [0.9, 0.85, 0.95]

# Number of times to repeat each configuration
repeats = 5

# Table to store results
results = []

# Test alpha values
for alpha in alpha_values:
    total = 0
    for i in range(repeats):
        total += run_training(alpha, default_gamma, default_epsilon)
        print(f"\rAlpha {alpha} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('alpha', alpha, avg))
    print()

# Test gamma values
for gamma in gamma_values:
    total = 0
    for i in range(repeats):
        total += run_training(default_alpha, gamma, default_epsilon)
        print(f"\rGamma {gamma} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('gamma', gamma, avg))
    print()

# Test epsilon values
for epsilon in epsilon_values:
    total = 0
    for i in range(repeats):
        total += run_training(default_alpha, default_gamma, epsilon)
        print(f"\rEpsilon {epsilon} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('epsilon', epsilon, avg))
    print()

# Print final results
print("\nResults (parameter, value, average_total_reward):")
for param, value, avg_reward in results:
    print(f"{param.capitalize()}: {value}, Average Total Reward: {avg_reward:.2f}")
