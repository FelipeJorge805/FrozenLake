import random
import gym
import numpy as np

# Function to run SARSA training for a given set of hyperparameters
def run_training(alpha=0.1, gamma=0.9, epsilon=0.9, seed=0, episodes=10000, is_slippery=False):
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)

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