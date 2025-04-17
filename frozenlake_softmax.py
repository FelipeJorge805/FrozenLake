import random
import gym
import numpy as np

def softmax(q_values, temperature=1.0):
    exp_q = np.exp((q_values - np.max(q_values)) / temperature)
    return exp_q / np.sum(exp_q)

# Function to run SARSA training for a given set of hyperparameters
def run_training_softmax(alpha=0.1, gamma=0.9, temperature=1.5, temperature_decay=0.999, seed=None, episodes=10000, is_slippery=False):
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
        probs = softmax(q_table[state], temperature=temperature)
        action = np.random.choice(action_size, p=probs)

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_probs = softmax(q_table[next_state], temperature=temperature)
            next_action = np.random.choice(action_size, p=next_probs)

            # SARSA update
            q_table[state, action] += alpha * (reward + (gamma * q_table[next_state, next_action]) - q_table[state, action])

            state = next_state
            action = next_action
            total_reward += reward

        # Decay temperature
        temperature = max(0.1, temperature * temperature_decay)

    env.close()
    return total_reward