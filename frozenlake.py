import random

import gym
import numpy as np

# Create the FrozenLake environment (v1)
env = gym.make('FrozenLake-v1', is_slippery=False)  # Set is_slippery=True for more challenging version
state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))
# Reset the environment to start a new episode
state = env.reset()

# To keep track of the total reward
total_reward = 0
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
episodes = 10000
epsilon = 0.9  # Exploration rate
min_epsilon = 0.1
epsilon_decay = 0.999

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    action = np.random.choice(action_size) if(np.random.rand() < epsilon) else np.argmax(q_table[state])

    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = np.random.choice(action_size) if(random.uniform(0, 1) < epsilon) else np.argmax(q_table[next_state])

        # SARSA update
        q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

        state = next_state
        action = next_action
        total_reward += reward

        print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}")

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()
# Display total reward at the end
print(f"Total Reward: {total_reward}")
