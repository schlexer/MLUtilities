
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns

from itertools import chain, combinations

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Plotting settings
plt.rc('text', usetex=False)
font = {'size': 24}
plt.rc('font', **font)

sns.set_style('white')
sns.set_palette(sns.hls_palette(8, h=0.5, l=0.4, s=0.5))


def unique_set(iterable, feature_dim=2):
    """
    Find unique sets of n features.
    :param iterable: list of full possibilities.
    :param feature_dim: Combination size.
    :return: iterable chain with all unique (no duplicate) combinations.
    """
    s = list(iterable)  # allows duplicate elements
    ch = chain.from_iterable(combinations(s, r) for r in range(feature_dim, feature_dim + 1))
    sn = [x for x in ch]
    return sn


class GAFeatureSelection:
    """Feature combination (model) selection using a genetic algorithm (GA)"""
    def __init__(self,
                 X,
                 y,
                 clf=None,
                 n_features=2,
                 population_size=100,
                 offspring_size=10,
                 scoring='cv',
                 n_cv=4,
                 random_state=42,
                 starting_population=None,
                 verbose=0):
        """
        Instantiates GAFeatureSelection object.
        :param X: feature data set of training data.
        :param y: targets of training data set.
        :param clf: classifier, or regressior (e.g. sklearn regression object)
        :param n_features: Number of features that the resulting model should have.
        :param population_size: How many feature combinations to start with (the more the better).
        :param offspring_size: How many offsprings should be created by mating the best candidates.
        :param scoring: How to determine the fitness of the model, cv uses cross validation R2.
        :param n_cv: validation set split multiple.
        :param random_state: random state integer for reproducibility.
        :param starting_population: GAFeatureSelection object can start from
        a pre-converged/pre-computed population.
        :param verbose: Whether to allow printing output.
        """

        self.verbose = verbose
        self.random_state = random_state
        self.X = X
        self.y = y
        self.scoring = scoring
        self.n_cv = n_cv
        self.feature_names = list(self.X.columns)
        self.total_features = len(self.X.columns)
        if clf is None:
            self.clf = RandomForestRegressor(max_depth=5,
                                             random_state=42,
                                             n_estimators=50)
            print('Using RandomForestRegressor, because no classifier was given.')
        else:
            self.clf = clf

        self.n_features = n_features
        self.all_feature_combinations = None

        self.population_size = population_size
        self.offspring_size = offspring_size
        self.offspring_chromosomes = None

        self.offspring = None
        self.selected_features = None

        if starting_population is None:
            print('Creating random population.')
            self.genes = self.random_genes(n=self.population_size)
            print('Calculating population metrics.')
            self.population = self.get_fitness(chromosomes=self.genes)
        else:
            self.population = starting_population

        self.genes = [tup[0] for tup in self.population]
        self._feature_frequencies = self.feature_frequencies
        self.feature_frequency_evolution = {}

        self.evolution = None
        self.evolution_history = None

    @property
    def mean_population_fitness(self):
        """Computes average of population fitness metrics."""
        m = np.mean([tup[2] for tup in self.population])
        return m

    def population_percentile(self, percentile=75):
        """Computes percentile of population fitness metrics."""
        m = np.percentile([tup[2] for tup in self.population], percentile)
        return m

    @property
    def max_population_fitness(self):
        """Computes max of population fitness metrics."""
        m = np.max([tup[2] for tup in self.population])
        return m

    def random_genes(self, n=1):
        """
        From the unique sets of n_features combinations, returns the first n.
        :param n: Number of chromosomes to return.
        :return: List of unique random chromosomes of length n.
        """
        random.seed(self.random_state)
        if self.all_feature_combinations is None:
            unique_combinations = []
            for feature_combination in unique_set(list(range(self.total_features)), feature_dim=self.n_features):
                unique_combinations.append(list(feature_combination))
            random.shuffle(unique_combinations)
            self.all_feature_combinations = unique_combinations
        selection = self.all_feature_combinations.copy()
        random.shuffle(selection)
        return selection[:n]

    def get_chromosome_score(self, X_chromosome):
        """
        Computes fitness using the subset of data in X_chromosome.
        :param X_chromosome: subset of full data set, containing only a selection of the features.
        :return: mean R2 or keras history last column entry.
        """
        np.random.seed(self.random_state)
        # Use either cross validation
        if self.scoring == 'cv':
            scores = cross_val_score(self.clf, X_chromosome, np.array(self.y), cv=self.n_cv)
            return np.mean(scores)
        # Or keras history in the case of neural networks (based on keras/tensorflow)
        else:
            try:
                history = self.clf.fit(X_chromosome, np.array(self.y))
                return history.history[self.scoring][-1]
            except:
                raise ValueError('Use either "cv" or keras history metrics.')

    def get_fitness(self, chromosomes):
        """
        Compute the fitness (derived from the performance metric, but not equal to it,
        as it is updated in the cross over process.
        :param chromosomes: List of chromosomes, which in turn are list of features.
        :return: List of the chromosome, and the performance metric (twice),
        which is necessary to keep track of performance during cross over.
        """
        results = []
        for n_chromosome, chromosome in enumerate(chromosomes):
            if self.verbose:
                print('Evaluating chromosome '+str(n_chromosome+1)+'/'+str(len(chromosomes)))

            X_chromosome = self.X[self.X.columns[chromosome]]
            chromosome_score = self.get_chromosome_score(X_chromosome=X_chromosome)

            if self.verbose:
                print('Chromosome score: '+str(chromosome_score))

            results.append([chromosome, chromosome_score, chromosome_score])
        return sorted(results, reverse=True, key=lambda tup: tup[1])

    def crossover(self):
        """Parent gene mating. Out of the best m+1 parents, a pairwise crossover is generated as the offspring,
        with m being the offspring size. """
        parent_genes = [tup[0] for tup in self.population]
        crossover_point = int(len(parent_genes[0]) / 2)
        offspring_chromosomes = []
        for i in range(self.offspring_size):
            parent1 = parent_genes[i]
            random.shuffle(parent1)
            parent2 = parent_genes[i+i]
            random.shuffle(parent2)
            offspring_chromosomes.append(parent1[:crossover_point] + parent2[crossover_point:])

        self.offspring_chromosomes = self.mutate_duplicate_chromosomes(chromosome_list=offspring_chromosomes)

        # Get offspring fitness
        self.offspring = self.get_fitness(chromosomes=self.offspring_chromosomes)

        # Lower parent fitness for gene diversity
        parents_size = len(self.offspring_chromosomes) + 1
        probability_adjustment = self.population_percentile()
        for i in range(parents_size):
            self.population[i][1] = self.population[i][1]*probability_adjustment

        # Update population
        self.population = self.population[:len(self.population) - len(self.offspring)]
        self.population.extend(self.offspring)
        self.population = sorted(self.population, reverse=True, key=lambda tup: tup[1])
        self.genes = [tup[0] for tup in self.population]
        return self.population

    def mutate_duplicate_chromosomes(self, chromosome_list):
        """
        During cross-over duplicate genes can occur.
        These will be replaced by random genes, while avoiding duplicate generation.
        :param chromosome_list: list of chromosomes to check.
        :return: duplicate-free chromosome list.
        """
        for i, chromosome in enumerate(chromosome_list):
            if len(chromosome) != len(set(chromosome)):
                n_duplicates = len(chromosome) - len(list(set(chromosome)))
                new_candidates = list(set(range(self.total_features)) - set(chromosome))

                chromosome = list(set(chromosome))
                chromosome.extend(np.random.choice(new_candidates, size=n_duplicates, replace=False))
                chromosome_list[i] = chromosome
        return chromosome_list

    @property
    def homogeneity(self):
        """Measure of whether all possible features are still in the population."""
        gene_ensemble = set(list(np.array(self.genes).flatten()))
        return 1-(len(gene_ensemble)/self.total_features)

    def mutate(self):
        """Increase stochastic effects to enhance optimization."""
        random_mutation_percentage = random.randint(1, 20)/100
        n_mutations = int(len(self.genes)*random_mutation_percentage)
        genes_copy = self.genes.copy()
        for i in range(n_mutations):
            random_gene_position = random.sample(range(self.population_size), 1)[0]
            random_gene = genes_copy[random_gene_position]
            random_chromosome_position = random.sample(range(self.n_features), 1)[0]

            new_chromosome_candidates = list(set(range(self.total_features)) - set(random_gene))
            random_gene[random_chromosome_position] = random.sample(new_chromosome_candidates, 1)[0]
            genes_copy[random_gene_position] = random_gene
        self.genes = genes_copy
        return self.genes

    def evolve(self, generations=20):
        """
        Run genetic algorithm for generations number of generations.
        :param generations: number of generations
        :return: evolution history with evolution of performance metrics.
        """
        evolution = np.ndarray((generations, 4))
        for i in range(generations):
            print('Generation: ' + str(i))
            # start with stats from random population
            self.feature_frequency_evolution[len(self.feature_frequency_evolution.keys())] = self.feature_frequencies
            evolution[i][0] = i
            evolution[i][1] = self.max_population_fitness
            evolution[i][2] = self.mean_population_fitness
            evolution[i][3] = self.homogeneity

            self.crossover()
            self.mutate()

            print('Max fitness: %5.2f ' % self.max_population_fitness)
            print('Mean fitness: %5.2f ' % self.mean_population_fitness)

        df_ev = pd.DataFrame(evolution)
        df_ev.columns = ['generation', 'max_population_fitness', 'mean_population_fitness', 'homogeneity']

        self.evolution = df_ev
        if self.evolution_history is None:
            self.evolution_history = self.evolution
        else:
            self.evolution_history = pd.concat([self.evolution_history, self.evolution], axis=0).reset_index(drop=True)
            self.evolution_history.generation = self.evolution_history.index
        return self.evolution

    @property
    def feature_frequencies(self):
        """Compute how often every feature occurs in the population."""
        all_genes = np.array(self.genes).flatten()
        gene_occurrences = np.bincount(all_genes)
        features = list(self.X.columns)
        feature_occurrences = sorted(list(zip(features, gene_occurrences)), key=lambda tup: tup[1], reverse=True)
        occ = np.array(list(zip(*feature_occurrences))[1])
        self._feature_frequencies = dict(zip(list(list(zip(*feature_occurrences))[0]), occ/occ.sum()*100))
        return self._feature_frequencies

    def plot_features(self,
                      labels=None,
                      frequencies=None,
                      show=True):
        """Plots feature frequencies"""
        print('Plotting feature frequencies.')
        if labels is None:
            labels = list(self.feature_frequencies.keys())
        if frequencies is None:
            frequencies = list(self.feature_frequencies.values())
        fig, ax = plt.subplots(figsize=(12, int(len(labels)*0.6)))
        ax = sns.barplot(x=frequencies, y=labels)
        ax.set_xlabel(r'Occurrence probability (\%)')
        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_evolution_stats(self,
                             full_history=True,
                             show=True):
        """Plots evolution history."""
        print('Plotting evolution statistics.')
        if full_history:
            df_stats = self.evolution_history
        else:
            df_stats = self.evolution
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(df_stats.generation, df_stats.max_population_fitness, lw=3, ls='-', label=r'$\mathrm{R^{2}}$-cv max')
        ax.plot(df_stats.generation, df_stats.mean_population_fitness, lw=3, ls='--', label=r'$\mathrm{R^{2}}$-cv mean')
        ax.plot(df_stats.generation, df_stats.homogeneity, lw=3, ls='-.', label='Homogeneity')
        # ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
        ax.legend()
        ax.set_title('Evolution statistics')
        ax.set_xlabel('Generation')
        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax
