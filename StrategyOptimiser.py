import numpy as np
import pandas as pd


class StrategyOptimizer:

    def __init__(self,
                 fitness_function,
                 n_generations,
                 generation_size,
                 n_genes,
                 gene_ranges,
                 mutation_probability,
                 gene_mutation_probability,
                 n_select_best):

        self.fitness_function = fitness_function,
        self.n_generations = n_generations,
        self.generation_size = generation_size,
        self.n_genes = n_genes,
        self.gene_ranges = gene_ranges,
        self.mutation_probability = mutation_probability,
        self.gene_mutation_probability = gene_mutation_probability,
        self.n_select_best = n_select_best

    def create_individual(self):
        """ Create individual """

        individual = []

        for i in range(self.n_genes):
            gene = np.random.randint(self.gene_ranges[i][0], self.gene_ranges[i][1])
            individual.append(gene)

        return individual

    def create_population(self, population_size):
        """ Creates a population """

        population = []

        for i in range(population_size):
            population.append(self.create_individual())

        return population

    def mate_parents(self, parents, n_offspring):
        """ Mates 'parents' and returns 'n_springs' individuals """

        n_parents = len(parents)

        offspring = []

        for i in range(n_offspring):
            random_dad = parents(np.random.randint(0, n_parents - 1))
            random_mom = parents(np.random.randint(0, n_parents - 1))

            dad_mask = np.random.randint(0, 2, size=np.array(random_dad).shape)
            mom_mask = np.logical_not(dad_mask)

            child = np.add(np.multiply(random_dad, dad_mask), np.multiply(random_mom, mom_mask))

            offspring.append(child)

        return offspring

    def mutate_individual(self, individual):
        """ Mutates 'individual' """

        new_individual = []

        for i in range(0, self.n_genes):
            gene = individual[i]

            if np.random.random() < self.gene_mutation_probability:
                # Mutate gene

                if np.random.random() < 0.5:
                    # Mutate brute force way

                    gene = np.random.randint(self.gene_ranges[i][0], self.gene_ranges[i][1])

                else:
                    # Mutate nicer way

                    left_range = self.gene_ranges[i][0]
                    right_range = self.gene_ranges[i][1]

                    gene_dist = right_range - left_range
                    # gene_mid = gene_dist / 2
                    x = individual[i] + gene_dist / 3 * (2 * np.random.random() - 1)

                    if x > right_range:
                        x = (x - left_range) % gene_dist + left_range
                    elif x < left_range:
                        x = (right_range - x) % gene_dist + left_range

                    gene = int(x)

            new_individual.append(gene)

    def mutate_population(self, population):
        """ Mutates 'population'. """

        mutated_pop = []

        for individual in population:
            new_individual = individual
            if np.random.random() < self.mutation_probability:
                new_individual = self.mutate_individual(individual)

            mutated_pop.append(new_individual)

        return mutated_pop

    def select_best(self, population, n_best):
        """ Selects the best 'n_best' individuals out of a population. """

        fitnesses = []

        for idx, individual in enumerate(population):
            individual_fitness = self.fitness_function(individual)
            fitnesses.append([idx, individual_fitness])

        costs_tmp = pd.DataFrame(fitnesses).sort_values(by=1, ascending=False).reset_index(drop=True)
        selected_parents_idx = list(costs_tmp.iloc[:n_best, 0])
        selected_parents = [parent for idx, parent in enumerate(population) if idx in selected_parents_idx]

        print('best is: {}, average: {}, and worst: {}'.format(
            costs_tmp[1].max(),
            costs_tmp[1].mean(),
            costs_tmp[1].min()
        ))

    def run_genetic_algo(self):
        """"""

        parent_gen = self.create_population(self.generations_size)

        for i in range(self.n_generations):
            parent_gen = self.select_best(parent_gen, self.n_select_best)

            parent_gen = self.select_best(parent_gen, self.generation_size)

            parent_gen = self.mutate_population(parent_gen)

        best_children = self.select_best(parent_gen, 10)
        return best_children
