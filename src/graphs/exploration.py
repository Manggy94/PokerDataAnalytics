from functools import cached_property

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ExplorationGraphs:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.categorical_data = self.data.select_dtypes(include=["object", "category"])
        self.numerical_data = self.data.select_dtypes(include=[np.number])

    @cached_property
    def min_max_data(self):
        return self.min_max_scaler.fit_transform(self.numerical_data)

    @cached_property
    def standard_data(self):
        return self.standard_scaler.fit_transform(self.numerical_data)

    def plot_correlation_heatmap(self):
        fig = px.imshow(self.numerical_data.corr()**2)
        fig.show()

    def plot_min_max_standard_deviation(self):
        # Bar plot of the standard deviation of the min-max scaled data
        fig = go.Figure()
        fig.add_trace(go.Bar(x=self.numerical_data.columns, y=np.std(self.min_max_data, axis=0), name="Min-Max"))
        fig.update_layout(title="Standard Deviation of Min-Max Scaled Data")
        fig.show()

    def plot_missing_numerical_data(self):
        missing_data = self.numerical_data.isna().sum() / self.numerical_data.shape[0]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=missing_data.index, y=missing_data.values, name="Missing Data"))
        fig.update_layout(title="Missing Numerical Data")
        fig.show()

    def plot_missing_categorical_data(self):
        missing_data = self.categorical_data.isna().sum() / self.categorical_data.shape[0]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=missing_data.index, y=missing_data.values, name="Missing Data"))
        fig.update_layout(title="Missing Categorical Data")
        fig.show()




