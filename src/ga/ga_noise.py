import abc
import termplotlib as tpl
import os
import numpy as np
import random
from itertools import combinations
from .tensorboard_logging import Logger
import datetime


# Step 0: Generate Population of N individuals
# While not fitness >= desired fitness / Max generations
#   Step 1: Evaluate Population on task T generating fitness Fᵢ ∀i=1 to N
#   Step 2: Sample best individuals based on a sample criterion C
#   Step 3: Generate new individuals by doing crossover on selected
#   Step 4: Mutate new individuals by mutation criterion M
# Repeat


class Param:
    def __init__(self, value, is_float=False, name=None, end_value=None, iters=None, multiplier=1):
        self._value = value
        self.name = name
        self.end_value = end_value
        self.is_float = is_float
        self.iters = iters
        self.multiplier = multiplier
        self.anneal = self.end_value is not None
        self.counter = 0

        if self.anneal:
            self.step_size = (self.end_value - self._value) / self.iters

    @property
    def value(self):
        if not self.is_float:
            return int(self._value) * self.multiplier
        return self._value * self.multiplier

    @value.setter
    def value(self, x):
        self._value = x

    def step(self):
        if self.anneal and self.counter < self.iters:
            self._value += self.step_size
        self.counter += 1

    def __repr__(self):
        return '%s: %9.4f' % (self.name if self.name is not None else 'Value', self.value)


class GANoise:
    def __init__(self, pop_size, pop_class, gene_shape, selection_size, sigma,
                 LOG_PATH, log_step=0, consensus_step=None, consensus_optimal=False,
                 consensus_sel_size=None, elite_size=0):
        self.pop_size = pop_size
        self.pop_class = pop_class
        self.gene_shape = gene_shape
        self.selection_size = selection_size
        self.elite_size = elite_size
        self.sigma = sigma
        self.consensus_step = consensus_step
        self.consensus_optimal = consensus_optimal
        self.consensus_sel_size = consensus_sel_size
        self.pop = []
        self.selected = []
        self.counter = 0

        # Initialize plotting
        self.max_fit = []
        self.mean_fit = []
        self.std_fit = []
        self.logger = Logger(os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        self.logger.log_text('Pop size', self.pop_size, step=0)
        self.logger.log_text('Elite size', self.elite_size.value, step=0)
        self.logger.log_text('Consensus Step', self.consensus_step, step=0)
        self.logger.log_text('Consensus Optimal', self.consensus_optimal, step=0)
        self.logger.log_text('Sigma', self.sigma.value, step=0)

        self.logger.step = log_step

    def initialize_pop(self):
        for i in range(self.pop_size):
            self.pop.append(self.pop_class(self.gene_shape))
            self.pop[-1].initialize_genes()

    def selection(self, memories):
        fitness = np.array([p.fitness for p in self.pop])

        top_k = []
        if self.elite_size.value != 0:
            top_k = np.argpartition(fitness, -self.elite_size.value)[-self.elite_size.value:]

        selected = np.argpartition(fitness, -self.selection_size.value)[-self.selection_size.value:]
        return [p.genes for p in np.take(self.pop, top_k)], [(self.pop[idx], memories[idx]) for idx in selected]

    def assign_fitness(self, fitness):
        for i, individual in enumerate(self.pop):
            individual.fitness = fitness[i]

    def _mutation(self, gene):
        new_genes = []
        for p in gene:
            new_genes.append(p.copy() + self.sigma.value * np.random.normal(size=p.shape))
        return new_genes

    def mutation(self, parent):
        for i in range(self.pop_size):
            p = np.random.choice(parent)
            self.pop[i].genes = self._mutation(p.genes)
            self.pop[i].fitness = None

    def simulate(self, fitness=None, preselected=None, memories=None, log=False):
        if fitness is not None:  # if fitness is passed assign it
            self.assign_fitness(fitness)

        m, mean, std = None, None, None
        if log:  # The data is relative to last generation
            m, mean, std = self.log()

        # preselected contains index of preselected individuals
        preselected_genes = []
        if preselected is not None:
            for idx in preselected:
                preselected_genes.append(self.pop[idx].genes.copy())

        # Main cycle
        elite_genes, parent = self.selection(memories)
        parent_genes, parent_memories = zip(*parent)

        self.mutation(parent_genes)

        for i in range(self.elite_size.value):
            self.pop[self.pop_size - i - 1].genes = elite_genes[i].copy()
            self.pop[self.pop_size - i - 1].fitness = None

        if preselected is not None:
            for i, genes in enumerate(preselected_genes):
                self.pop[self.pop_size - i - self.elite_size.value].genes = genes.copy()
                self.pop[self.pop_size - i - self.elite_size.value].fitness = None

        # Update params
        self.selection_size.step()
        self.elite_size.step()
        self.sigma.step()

        self.counter += 1
        return m, mean, std

    def calculate_stats(self):
        f = [p.fitness for p in self.pop]
        return np.max(f), np.mean(f), np.std(f, dtype=np.float64)

    def clear_stats(self):
        self.max_fit = []
        self.mean_fit = []
        self.std_fit = []

    def log(self):
        m, mean, std = self.calculate_stats()
        self.max_fit.append(m)
        self.mean_fit.append(mean)
        self.std_fit.append(std)

        # Plotting
        x_range = int(self.counter / max(1, len(self.max_fit) - 1)) * np.arange(0, len(self.max_fit))

        #os.system('clear')
        fig = tpl.figure()
        fig.plot(x_range[-10:], self.max_fit[-10:], width=84, height=10)
        fig.plot(x_range[-10:], self.mean_fit[-10:], width=84, height=10)
        fig.plot(x_range[-10:], self.std_fit[-10:], width=84, height=10)
        fig.show()

        self.logger.log_scalar(tag='Max fitness', value=m)
        self.logger.log_scalar(tag='Mean fitness', value=mean)
        self.logger.log_scalar(tag='STD fitness', value=std)
        self.logger.step += 1

        print(self.gen_log(m, mean, std))
        print(self.param_log())
        return m, mean, std

    def gen_log(self, maxx, mean, std):
        return 'Gen: %5d | Max Fitness: %9.4f | Mean Fitness: %9.4f | Fitness STD: %9.4f' % (self.counter, maxx, mean, std)

    def param_log(self):
        return 'Pop Size: %5d | Sel Size: %5d | Sigma: %2.4f' % (self.pop_size, self.selection_size.value, self.sigma._value)


class Individual:
    def __init__(self, gene_shape):
        self.gene_shape = gene_shape
        self.genes = self.initialize_genes()
        self.fitness = None

    @abc.abstractmethod
    def initialize_genes(self):
        return
