import random
import numpy as np
from frozenlake import run_training

# ---- GA Helper Functions ----

def generate_individual():
    """
    Create an individual.
    Replace these dummy ranges with the bounds for your hyperparameters.
    For example, if you have alpha, gamma, epsilon:
        'alpha': random.uniform(0.05, 0.3),
        'gamma': random.uniform(0.8, 1.0),
        'epsilon': random.uniform(0.05, 0.9)
    """
    return {
        'alpha': random.randint(5, 30)/100,
        'gamma': random.randint(80, 100)/100,
        'epsilon': random.randint(50, 99)/100,
        'epsilon_decay': random.uniform(0.8,0.9999)
    }


def fitness(individual):
    """
    Dummy fitness function.
    Replace this with a call to your SARSA training routine (e.g., run_training_softmax or your baseline)
    that returns a fitness score (e.g., average reward).

    For now, we use a placeholder: the lower the sum of hyperparameters, the higher the fitness.
    (Remember: you'll want to *maximize* reward, so adjust fitness accordingly.)
    """
    # Placeholder: modify with your actual evaluation.
    # For example:
    #   return run_training_softmax(**individual)
    return run_training(alpha=individual["alpha"], gamma=individual["gamma"], epsilon=individual["epsilon"], epsilon_decay=individual["epsilon_decay"], episodes=5000, is_slippery=True)


def selection(population, fitnesses):
    """
    Simple selection: choose the top half of the population.
    You can refine this to tournament selection or roulette-wheel selection later.
    """
    # Pair individuals with their fitness, then sort by fitness descending.
    paired = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    selected = [ind for ind, fit in paired[:len(paired) // 2]]
    return selected


def crossover(parent1, parent2):
    """
    Single-point crossover: for each hyperparameter, randomly choose from parent1 or parent2.
    """
    child = {}
    for key in parent1.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


def mutate(individual, mutation_rate=0.1):
    """
    Mutate an individual by adding a small random perturbation.
    The range of mutation can be fine-tuned.
    """
    new_ind = individual.copy()
    for key in new_ind:
        if random.random() < mutation_rate:
            perturbation = random.uniform(-0.05, 0.05)
            new_ind[key] = max(0, new_ind[key] + perturbation)
    return new_ind


# ---- GA Main Loop ----

def run_ga(generations=10, population_size=20):
    # Generate initial population
    population = [generate_individual() for _ in range(population_size)]

    for gen in range(generations):
        # Evaluate fitness for each individual
        fitnesses = [fitness(ind) for ind in population]
        print(f"Generation {gen}: Best fitness = {max(fitnesses):.4f}")

        # Selection: choose a subset of individuals to be parents
        parents = selection(population, fitnesses)

        # Create new population via crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return the best individual from the final population.
    fitnesses = [fitness(ind) for ind in population]
    best = population[np.argmax(fitnesses)]
    print(f"best fitness {best}")
    return best


# ---- Run the GA if this file is executed directly ----

if __name__ == '__main__':
    best_individual = run_ga()
    print("Best individual found:", best_individual)
