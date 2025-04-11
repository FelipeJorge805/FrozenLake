import gym
import numpy as np
from frozenlake import run_training

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
        total += run_training(alpha=alpha)
        print(f"\rAlpha {alpha} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('alpha', alpha, avg))

# Test gamma values
for gamma in gamma_values:
    total = 0
    for i in range(repeats):
        total += run_training(gamma=gamma)
        print(f"\rGamma {gamma} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('gamma', gamma, avg))

# Test epsilon values
for epsilon in epsilon_values:
    total = 0
    for i in range(repeats):
        total += run_training(epsilon=epsilon)
        print(f"\rEpsilon {epsilon} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('epsilon', epsilon, avg))

# Print final results
print("\nResults (parameter, value, average_total_reward):")
for param, value, avg_reward in results:
    print(f"{param.capitalize()}: {value}, Average Total Reward: {avg_reward:.2f}")
