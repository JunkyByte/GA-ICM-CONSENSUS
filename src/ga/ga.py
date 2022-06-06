import abc
import termplotlib as tpl
import os
import numpy as np
import random
from itertools import combinations
from numba import njit, vectorize
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
    def __init__(self, value, name=None, end_value=None, iters=None):
        self._value = value
        self.name = name
        self.end_value = end_value
        self.iters = iters
        self.anneal = self.end_value is not None
        self.counter = 0

        if self.anneal:
            self.step_size = (self.end_value - self.value) / self.iters

    @property
    def value(self):
        return int(self._value)

    @value.setter
    def value(self, x):
        self._value = x

    def step(self):
        if self.anneal and self.counter < self.iters:
            self._value += self.step_size
        self.counter += 1

    def __repr__(self):
        return '%s: %9.4f' % (self.name if self.name is not None else 'Value', self.value)


def rank_selection(pop, sel_size):
    if hasattr(pop[0], 'fitness'):
        fitness = np.array([p.fitness for p in pop])
    else:
        fitness = np.array([p[0].fitness for p in pop])

    probs = (np.argsort(-fitness) + 1) / ((len(fitness) + 1) * len(fitness) / 2)

    return np.array(pop)[np.random.choice(len(pop), size=sel_size, replace=False, p=probs)]


def roulette_selection(pop, sel_size):
    # Probability of picking an individual is proportional to its fitness (ƒ)
    # pᵢ = ƒᵢ / (∑ fⱼ where j=1 to pop size)
    fitness = np.array([p.fitness for p in pop])
    probs = fitness / np.sum(fitness)
    return np.random.choice(pop, size=sel_size, replace=False, p=probs)


def lw_crossover(a, b):
    new = []
    for i in range(len(a)):
        if a[i].ndim == 1:
            p = np.random.random()
            if p > 0.5:
                new.append(a[i].copy())
            else:
                new.append(b[i].copy())
        elif a[i].ndim == 2:  # Linear (OUT, IN)
            new.append(a[i].copy())
            for j in range(a[i].shape[1]):
                p = np.random.random()
                if p > 0.5:
                    v = a[i][:, j]
                else:
                    v = b[i][:, j]
                new[i][:, j] = v.copy()
        elif a[i].ndim == 4:  # Conv (OUT_C, IN_C, K_H, K_W)
            new.append(a[i].copy())
            for j in range(a[i].shape[0]):
                p = np.random.random()
                if p > 0.5:
                    v = a[i][j]
                else:
                    v = b[i][j]
                new[i][j] = v.copy()
    return new


def onep_crossover(a, b):
    # We assume is represented by a ndarray
    # Take a random point p in the gene and split into a[:p] - b[p:]
    assert a.shape == b.shape
    p = np.random.random()
    shape = a.shape
    flat = np.prod(shape)
    a_sel = a.flatten()[:int(p * flat)]
    b_sel = b.flatten()[int(p * flat):]
    return np.concatenate([a_sel, b_sel]).reshape(shape)


def consensus_crossover(a, b, ma, mb, optimal=False):
    if optimal:
        picked = a if a.fitness < b.fitness else b
        mem = mb if a.fitness < b.fitness else ma
    else:
        s = np.random.random() > 0.5
        picked = a if s else b
        mem = mb if s else ma

    loss = 0
    for i in range(32):
        loss += picked.train(mem)
    print(loss)

    w = picked.get_weights()
    picked.genes_to_weights()

    return w


@vectorize('int64(int64, float64)', nopython=True)
def _random_invert(x, p):
    if np.random.random() < p:
        return -x
    return x


@njit(['int64[:, :](int64[:, :])', 'int64[:](int64[:])'], parallel=True)
def invert_mutation(x):
    s = 1.
    for i in range(len(x.shape)):
        s *= x.shape[i]
    p = 1. / s
    out = _random_invert(x, p)
    return out


@vectorize('float64(float64, float64)', nopython=True)
def _gaussian_mutation(x, p):
    if np.random.random() < p:
        return x + np.random.normal()
    return x


@njit(parallel=True)
def gaussian_mutation(x, s=0.):
    if s == 0:
        s = 1.
        for i in range(len(x.shape)):
            s *= x.shape[i]
    p = 1. / s
    out = _gaussian_mutation(x, p)
    return out


def lw_invert_mutation(x):
    for i in range(len(x)):
        x[i] = invert_mutation(x[i])  # TODO: Vec size as under
    return x


def lw_gaussian_noise(x):
    s = 0
    for i in range(len(x)):
        s += np.prod(x[i].shape)

    for i in range(len(x)):
        x[i] = gaussian_mutation(x[i], s)
    return x


class GA:
    def __init__(self, pop_size, pop_class, gene_shape, selection_size, selection, crossover, mutation,
                 LOG_PATH, log_step=0, consensus_step=None, consensus_optimal=False, consensus_mutation=0.5,
                 consensus_sel_size=None, elite_size=0):
        self.pop_size = pop_size
        self.pop_class = pop_class
        self.gene_shape = gene_shape
        self.selection_size = selection_size
        self.selection_f = selection
        self.crossover_f = crossover
        self.mutation_f = mutation
        self.elite_size = elite_size
        self.consensus_step = consensus_step
        self.consensus_optimal = consensus_optimal
        self.consensus_mutation = consensus_mutation
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
        self.logger.log_text('Consensus Mutation', self.consensus_mutation, step=0)
        self.logger.log_text('Selection Function', self.selection_f.__name__, step=0)
        self.logger.log_text('Mutation Function', self.mutation_f.__name__, step=0)

        self.logger.step = log_step

    def initialize_pop(self):
        for i in range(self.pop_size):
            self.pop.append(self.pop_class(self.gene_shape))
            self.pop[-1].initialize_genes()

    def crossover(self, selected, memories=None, mutate=True):
        indices = random.choices(list(combinations(range(len(selected)), 2)), k=self.pop_size - self.elite_size.value)

        if self.consensus_step is not None and self.consensus_optimal is not None and self.counter == self.consensus_step:
            self.selection_size.value = self.consensus_sel_size
            self.selection_size.end_value = None

        cons_counter = 0
        cross_counter = 0
        for idx, (j, k) in enumerate(indices):
            consensus = False
            if (self.consensus_step is not None and self.counter > self.consensus_step and
                    min(selected[j].fitness, selected[k].fitness) > self.mean_fit[-1]):
                consensus = True
                child_genes = consensus_crossover(selected[j], selected[k],
                                                  memories[j], memories[k], optimal=self.consensus_optimal)
                cons_counter += 1
            else:
                child_genes = self.crossover_f(selected[j].genes, selected[k].genes)
                cross_counter += 1

            if mutate and (not consensus or np.random.random() > self.consensus_mutation):
                child_genes = self.mutation(child_genes)

            self.pop[idx].genes = child_genes.copy()

        for indiv in self.pop:
            indiv.fitness = None

        print('Used Conensus on: %5d | Crossover on: %5d' % (cons_counter, cross_counter))

    def selection(self, memories):
        top_k = []
        if self.elite_size.value != 0:
            fitness = np.array([p.fitness for p in self.pop])
            top_k = np.argpartition(fitness, -self.elite_size.value)[-self.elite_size.value:]

        selected = self.selection_f(list(zip(*[self.pop, memories])), self.selection_size.value)
        return [p.genes for p in np.take(self.pop, top_k)], selected

    def assign_fitness(self, fitness):
        for i, individual in enumerate(self.pop):
            individual.fitness = fitness[i]

    def mutation(self, gene):
        return self.mutation_f(gene)

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

        self.crossover(parent_genes, memories=parent_memories, mutate=True)

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
        return 'Pop Size: %5d | Selection Size: %5d | Elite Size: %5d' % (self.pop_size, self.selection_size.value, self.elite_size.value)


class Individual:
    def __init__(self, gene_shape):
        self.gene_shape = gene_shape
        self.genes = self.initialize_genes()
        self.fitness = None

    @abc.abstractmethod
    def initialize_genes(self):
        return
