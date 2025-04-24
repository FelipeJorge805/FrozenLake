import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define the network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Hyperparameters
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LR = 1e-3
EPISODES = 500

# Init
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state)
        return policy_net(state_tensor).argmax().item()

# Training with 1 experience at a time
def train_step(state, action, reward, next_state, done):
    state_tensor = torch.FloatTensor(state)
    next_state_tensor = torch.FloatTensor(next_state)
    reward_tensor = torch.tensor(reward)
    done_tensor = torch.tensor(done)

    q_value = policy_net(state_tensor)[action]

    with torch.no_grad():
        max_next_q = target_net(next_state_tensor).max()
        target = reward_tensor if done else reward_tensor + GAMMA * max_next_q

    loss = loss_fn(q_value, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main loop
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        #env.render()
        action = select_action(state, EPSILON)
        next_state, reward, done, _, _ = env.step(action)
        train_step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {EPSILON:.3f}")

env.close()