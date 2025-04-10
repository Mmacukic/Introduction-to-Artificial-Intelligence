import argparse
import random
import numpy as np


def load_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    features = data[:, :-1]
    target = data[:, -1]
    headers = open(file_path, 'r').readline().strip().split(',')
    return features, target, headers


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_weights(input_size, layer_sizes):
    layers = [input_size] + layer_sizes + [1]  # Include output layer with size 1
    weights = [np.random.normal(0, 0.01, (layers[i], layers[i + 1])) for i in range(len(layers) - 1)]
    biases = [np.random.normal(0, 0.01, layers[i + 1]) for i in range(len(layers) - 1)]
    return weights, biases


def flatten_weights(weights, biases):
    flat_weights = np.concatenate([w.flatten() for w in weights] + [b.flatten() for b in biases])
    return flat_weights


def unflatten_weights(flat_weights, layers):
    idx = 0
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        w_size = layers[i] * layers[i + 1]
        weights.append(flat_weights[idx:idx + w_size].reshape((layers[i], layers[i + 1])))
        idx += w_size
        biases.append(flat_weights[idx:idx + layers[i + 1]])
        idx += layers[i + 1]
    return weights, biases


def forward_pass(weights, biases, inputs):
    activation = inputs
    for i in range(len(weights) - 1):
        z = np.dot(activation, weights[i]) + biases[i]
        activation = sigmoid(z)
    final_output = np.dot(activation, weights[-1]) + biases[-1]
    return final_output[0]  # The final output is a single value


def compute_mse(weights, biases, features, targets):
    errors = []
    for feature, target in zip(features, targets):
        prediction = forward_pass(weights, biases, feature)
        error = (target - prediction) ** 2
        errors.append(error)
    mean_squared_error = np.mean(errors)
    return mean_squared_error


def evaluate_fitness(chromosome, layers, train_features, train_target):
    weights, biases = unflatten_weights(chromosome, layers)
    mse = compute_mse(weights, biases, train_features, train_target)
    return 1 / (mse + 1e-8)  # Add small value to avoid division by zero


def select_parents(population, fitnesses):
    total_fitness = np.sum(fitnesses)
    selection_probs = fitnesses / total_fitness
    selected_indices = np.random.choice(len(population), size=len(population), p=selection_probs)
    return [population[i] for i in selected_indices]


def crossover(parent1, parent2):
    return (np.array(parent1) + np.array(parent2)) / 2


def mutate(chromosome, p, K):
    mutation_mask = np.random.rand(len(chromosome)) < p
    chromosome += mutation_mask * np.random.normal(0, K, len(chromosome))
    return chromosome


def genetic_algorithm(train_features, train_target, input_dim, layer_config, popsize, elitism, p, K, iterations):
    layers = [input_dim] + [int(layer) for layer in layer_config.split('s') if layer] + [1]

    # Initialize population
    population = [flatten_weights(*initialize_weights(input_dim, layers[1:-1])) for _ in range(popsize)]

    # Initialize the best chromosome and its corresponding weights and biases
    best_chromosome = None
    best_weights = None
    best_biases = None
    best_fitness = float('-inf')

    for iter in range(1, iterations + 1):
        # Evaluate fitness
        fitnesses = np.array([evaluate_fitness(chromosome, layers, train_features, train_target) for chromosome in population])

        # Elitism: Select the best individuals to carry over to the next generation
        elite_indices = np.argsort(fitnesses)[-elitism:]
        elites = [population[i] for i in elite_indices]

        # Update the best chromosome
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_chromosome = population[max_fitness_idx]
            best_weights, best_biases = unflatten_weights(best_chromosome, layers)
            best_fitness = fitnesses[max_fitness_idx]

        # Selection: Select parents based on fitness
        parents = select_parents(population, fitnesses)

        # Crossover and mutation to create the next generation
        next_generation = []
        while len(next_generation) < popsize - elitism:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, p, K)
            next_generation.append(child)

        # Combine elites and the newly created individuals to form the next generation
        population = elites + next_generation

        # Every 2000 iterations, print the training error
        if iter % 2000 == 0:
            best_chromosome = population[np.argmax(fitnesses)]
            best_weights, best_biases = unflatten_weights(best_chromosome, layers)
            train_mse = compute_mse(best_weights, best_biases, train_features, train_target)
            print(f"[Train error @{iter}]: {train_mse}")

    best_chromosome = population[np.argmax(fitnesses)]
    best_weights, best_biases = unflatten_weights(best_chromosome, layers)

    return best_weights, best_biases


def main(args):
    train_features, train_target, train_headers = load_dataset(args.train)

    test_features, test_target, test_headers = load_dataset(args.test)

    input_dim = train_features.shape[1]

    best_weights, best_biases = genetic_algorithm(train_features, train_target, input_dim, args.nn, args.popsize, args.elitism, args.p, args.K, args.iter)
    test_mse = compute_mse(best_weights, best_biases, test_features, test_target)

    print(f"[Test error]: {test_mse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a neural network using a genetic algorithm")
    parser.add_argument('--train', type=str, required=True, help="Path to the training dataset (CSV file)")
    parser.add_argument('--test', type=str, required=True, help="Path to the testing dataset (CSV file)")
    parser.add_argument('--nn', type=str, required=True,
                        help="Neural network configuration (e.g., '5s' for a single hidden layer with 5 units)")
    parser.add_argument('--popsize', type=int, required=True, help="Population size for the genetic algorithm")
    parser.add_argument('--elitism', type=int, required=True, help="Number of elite chromosomes to retain")
    parser.add_argument('--p', type=float, required=True, help="Mutation probability for each weight")
    parser.add_argument('--K', type=float, required=True, help="Standard deviation of Gaussian noise for mutation")
    parser.add_argument('--iter', type=int, required=True, help="Number of iterations for the genetic algorithm")

    args = parser.parse_args()
    main(args)

