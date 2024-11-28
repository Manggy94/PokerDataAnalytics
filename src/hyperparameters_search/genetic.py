import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from numpy.random import choice, rand
from sklearn.model_selection import train_test_split
from src.hyperparameters_search.utils import make_parameters_df
from src.pipelines.preprocessing.features.models.neural_network import NeuralNetworkFeaturesPreprocessor
from src.pipelines.preprocessing.targets.models.neural_network import NeuralNetworkTargetPreprocessor


class GeneticSearch:
    """
    Class that performs a genetic search for the best hyperparameters for a given model
    """
    def __init__(self, model_class, hyperparameters, data, population_size=10, n_generations=10, mutation_rate=0.1):
        self.model_class = model_class
        self.hyperparameters = hyperparameters
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.population = self._generate_population()
        self.population_history = []
        self.best_individual_history = pd.DataFrame(index=range(1, n_generations+1), columns=hyperparameters.keys())
        self.fitness_history = pd.DataFrame(index=range(1, n_generations+1), columns=[["mean_fitness", "max_fitness"]])
        self.data = data
        self.features_preprocessor = NeuralNetworkFeaturesPreprocessor()
        self.targets_preprocessor = NeuralNetworkTargetPreprocessor(target_column="player_combo")
        self.X_train = self.y_train = self.X_test = self.y_test = None
        self._generate_data()
        self.fitness = None

    def _generate_data(self):
        X = self.features_preprocessor.fit_transform(self.data)
        y = self.targets_preprocessor.fit_transform(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def _generate_population(self) -> pd.DataFrame:
        return make_parameters_df(self.population_size, self.hyperparameters)

    def _evaluate_population(self):
        print("Current population:\n")
        print(self.population)
        fitness = []
        for i, individual in self.population.iterrows():
            print(f"Individual {i+1}/{self.population_size}:\n")
            fitness.append(self._evaluate_individual(individual.to_dict()))
            print(f"Fitness: {fitness[-1]:.2%}\n")
        self.fitness = pd.DataFrame({"fitness": fitness})
        generation_index = len(self.population_history)+1
        best_individual_index = self.fitness.fitness.idxmax()
        best_individual = self.population.loc[best_individual_index]
        mean_fitness = self.fitness.fitness.mean()
        max_fitness = self.fitness.fitness.max()
        print(f"Best individual:\n")
        print(best_individual)
        print(f"Mean fitness: {mean_fitness:.2%}\n")
        print(f"Max fitness: {max_fitness:.2%}\n")
        self.fitness_history.loc[generation_index, "min_fitness"] = mean_fitness
        self.fitness_history.loc[generation_index, "max_fitness"] = max_fitness
        self.best_individual_history.loc[generation_index] = best_individual



    def _select_parents(self) -> list[pd.DataFrame]:
        """
        Selects the parents for the next generation
        :return:
        """
        self.population["fitness"] = self.fitness
        self.population["proba"] = self.population["fitness"] / self.population["fitness"].sum()
        parent_indices = choice(self.population.index, size=(self.population_size-3,2), p=self.population["proba"])
        selected_parents = [pd.concat((self.population.loc[i], self.population.loc[j]), axis=1).T for (i, j) in parent_indices]
        return selected_parents

    def _evaluate_individual(self, individual) -> float:
        model = self.model_class(**individual, preprocessor=self.features_preprocessor)
        model.fit(self.X_train, self.y_train)
        return float(model.score(self.X_test, self.y_test))

    def _return_top_individuals(self) -> pd.DataFrame:
        """
        Returns the top 3 individuals based on their fitness
        :return: pd.DataFrame
        """
        top_individuals_fitness = self.fitness.nlargest(3, 'fitness')
        selected_individuals = self.population.loc[top_individuals_fitness.index]
        return selected_individuals

    def _crossover(self, parents):
        child = {}
        for parameter in self.hyperparameters.keys():
            child[parameter] = choice([parents[parameter].iloc[0], parents[parameter].iloc[1]])
        return child

    def mutate(self, child):
        for param, values in self.hyperparameters.items():
            if rand() < self.mutation_rate:
                child[param] = choice(values)
        return child

    def _create_next_generation(self):
        self.population_history.append(self.population)
        top_individuals = self._return_top_individuals()
        parents = self._select_parents()
        children = []
        for parent in parents:
            child = self._crossover(parent)
            child = self.mutate(child)
            children.append(child)
        children_df = pd.DataFrame(children)
        new_population = pd.concat((top_individuals, children_df), ignore_index=True)
        return new_population

    def run(self):
        print("Starting genetic search\n")
        print(f"Population size: {self.population_size}\n")
        print(f"Number of generations: {self.n_generations}\n")
        print(f"Mutation rate: {self.mutation_rate}\n")
        for generation in range(self.n_generations):
            print(f"Generation {generation+1}/{self.n_generations}\n")
            self._evaluate_population()
            print("Fitness evaluation complete\n")
            print(self.fitness)
            self.population = self._create_next_generation()
        self.plot_fitness_evolution()

    def plot_fitness_evolution(self):
        x_index = list(range(1, len(self.fitness_history)+1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_index, y=self.fitness_history["mean_fitness"], mode="lines", name="Mean Fitness"))
        fig.add_trace(go.Scatter(x=x_index, y=self.fitness_history["max_fitness"], mode="lines", name="Max Fitness"))
        fig.update_layout(title="Mean Fitness Evolution", xaxis_title="Generation", yaxis_title="Mean Fitness")
        fig.update_xaxes(tickvals=x_index)
        # Montrer l'axe y en pourcentage entre 0% et 100%
        fig.update_yaxes(tickvals=np.linspace(0, 1, 11), tickformat="%")
        fig.show()
