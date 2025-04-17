import gym
import numpy as np
from frozenlake import run_training
from frozenlake_softmax import run_training_softmax

# Hyperparameter values to test
alpha_values = [0.12, 0.15, 0.2]
gamma_values = [1.0, 0.95, 0.9]
#epsilon_values = [0.9, 0.85, 0.95]
temperature_decay_values = [0.998, 0.997, 0.996]
temperature_values = [0.3, 1.0, 2.0]

# Number of times to repeat each configuration
repeats = 5

# Table to store results
results = []

# Test alpha values
for alpha in alpha_values:
    total = 0
    for i in range(repeats):
        total += run_training_softmax(alpha=alpha,seed=i,is_slippery=True)
        print(f"\rAlpha {alpha} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('alpha', alpha, avg))

# Test gamma values
for gamma in gamma_values:
    total = 0
    for i in range(repeats):
        total += run_training_softmax(gamma=gamma,seed=i,is_slippery=True)
        print(f"\rGamma {gamma} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('gamma', gamma, avg))
'''
# Test epsilon values
for epsilon in epsilon_values:
    total = 0
    for i in range(repeats):
        total += run_training_softmax(epsilon=epsilon,seed=i,is_slippery=True)
        print(f"\rEpsilon {epsilon} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('epsilon', epsilon, avg))
'''
# Test temperature decay values
for temp_decay in temperature_decay_values:
    total = 0
    for i in range(repeats):
        total += run_training_softmax(temperature_decay=temp_decay,seed=i,is_slippery=True)
        print(f"\rTemperature Decay {temp_decay} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('temp_decay', temp_decay, avg))

# Test temperature values
for temperature in temperature_values:
    total = 0
    for i in range(repeats):
        total += run_training_softmax(temperature=temperature,seed=i,is_slippery=True)
        print(f"\rTemperature {temperature} - Repeat {i+1}/{repeats}", end='', flush=True)
    avg = total / repeats
    results.append(('temperature', temperature, avg))

# Print final results
f = open("data","a")
f.write("\nResults (parameter, value, average_total_reward):")
print("\nResults (parameter, value, average_total_reward):")
for param, value, avg_reward in results:
    s = f"{param.capitalize()}: {value}, Average Total Reward: {avg_reward:.2f}"
    f.write(s+"\n")
    print(s)

f.write("\n")
f.close()
