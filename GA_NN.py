import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==== Genetic parameters ====
POP_SIZE = 10
GENERATIONS = 5
SEED = 42
EPISODES = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = gym.make("CartPole-v1")
env.seed(SEED)

# ==== Search space ====
HIDDEN_LAYER_OPTIONS = [8, 16, 32, 64]
MAX_LAYERS = 2


# ==== Model building ====
def build_model(layer_sizes):
    layers = []
    input_size = 4
    for size in layer_sizes:
        layers.append(nn.Linear(input_size, size))
        layers.append(nn.ReLU())
        input_size = size
    layers.append(nn.Linear(input_size, 2))  # Output layer
    return nn.Sequential(*layers)


# ==== Genome ====
def random_architecture():
    return [random.choice(HIDDEN_LAYER_OPTIONS) for _ in range(random.randint(1, MAX_LAYERS))]


# ==== Training & evaluation ====
def evaluate_model(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    total_rewards = []

    for _ in range(EPISODES):
        state = env.reset()[0]
        done = False
        total = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            next_state, reward, done, _, _ = env.step(action)
            total += reward
            state = next_state
        total_rewards.append(total)

    return np.mean(total_rewards)


# ==== Genetic algorithm ====
population = [random_architecture() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    print(f"\nGeneration {gen}")
    scored = []
    for genome in population:
        model = build_model(genome)
        score = evaluate_model(model)
        scored.append((genome, score))
        print(f"  Genome {genome} => Score: {score:.2f}")

    # Selection: top 50%
    scored.sort(key=lambda x: x[1], reverse=True)
    parents = [g for g, _ in scored[:POP_SIZE // 2]]

    # Crossover & mutation
    children = []
    while len(children) < POP_SIZE - len(parents):
        p1, p2 = random.sample(parents, 2)
        crossover_point = random.randint(1, min(len(p1), len(p2)) - 1)
        child = p1[:crossover_point] + p2[crossover_point:]
        if random.random() < 0.2:  # Mutation
            idx = random.randint(0, len(child) - 1)
            child[idx] = random.choice(HIDDEN_LAYER_OPTIONS)
        children.append(child)

    population = parents + children

env.close()

