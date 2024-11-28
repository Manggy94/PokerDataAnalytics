import math
import numpy as np
import pandas as pd


def make_parameters_df(n_population: int, grid: dict) -> pd.DataFrame:
    n_values_list = [len(value) for value in grid.values()]
    max_population = math.prod(n_values_list)
    parameters_list = []
    while len(parameters_list) < min(n_population, max_population):
        parameters = {}
        for key, values in grid.items():
            parameters[key] = np.random.choice(values)
        if parameters not in parameters_list:
            parameters_list.append(parameters)
    return pd.DataFrame(parameters_list)