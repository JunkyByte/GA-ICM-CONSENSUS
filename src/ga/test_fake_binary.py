import numpy as np
import matplotlib.pyplot as plt
from ga import GA, Individual, roulette_selection, onep_crossover, invert_mutation, roulette_selection

# Individual class
class Binary(Individual):
    def initialize_genes(self):
        self.genes = np.array([np.random.choice([-1, 1], size=self.gene_shape)])


# Configs
pop_size = 1000
gene_size = 100
selection_size = 750

# Setup Genetic Algorithm class
ga = GA(pop_size, Binary, gene_size, selection_size, roulette_selection,
        onep_crossover, invert_mutation, elite_size=50)
#ga = GA(pop_size, Binary, gene_size, selection_size, roulette_selection,
#        onep_crossover, invert_mutation)
ga.initialize_pop()

# Fake problem solution
solution = np.random.choice([-1, 1], size=gene_size)

# Run configs
iterations = 10000
log_step = 10

# Iteration
for i in range(iterations):
    # Calculate fitness as distance from solution
    fitness = [np.count_nonzero(ind.genes == solution) for ind in ga.pop]

    log = False
    if i % log_step == 0:
        log = True

    m, mean, var = ga.simulate(fitness=fitness, log=log)
