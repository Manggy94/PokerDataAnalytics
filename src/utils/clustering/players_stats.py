import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.pipeline import Pipeline


class PlayersClusterFinder(BaseEstimator, TransformerMixin):

    def __init__(self, min_hands: int = 50):
        self.min_hands = min_hands
        self.min_confidence_ratio = 1-1/np.sqrt(min_hands)
        self.confirmed_players = None
        self.ms = None
        self.ms_labels = None
        self.ms_cluster_centers = None
        self.nb_clusters = None
        self.df_columns = None
        self.kmeans_labels = None
        self.kmeans_cluster_centers = None
        self.kmeans_classifier = None

    def fit(self, X: pd.DataFrame, y=None):
        self.confirmed_players = X[X.confidence_ratio >= self.min_confidence_ratio]
        self.df_columns = self.confirmed_players.columns
        bandwidth = estimate_bandwidth(self.confirmed_players, quantile=0.3, n_samples=500, random_state=42)
        self.ms = MeanShift(bandwidth=bandwidth)
        self.ms.fit(self.confirmed_players)
        self.ms_labels = self.ms.labels_
        self.ms_cluster_centers = pd.DataFrame(
            data=self.ms.cluster_centers_, columns=self.df_columns)
        self.nb_clusters = len(np.unique(self.ms_labels))
        self.kmeans_classifier = KMeans(n_clusters=self.nb_clusters, random_state=42, init=self.ms_cluster_centers, )
        self.kmeans_classifier.fit(X)
        self.kmeans_labels = self.kmeans_classifier.labels_
        self.kmeans_cluster_centers = pd.DataFrame(
            data=self.kmeans_classifier.cluster_centers_, columns=self.df_columns)
        return self

    def transform(self, X: pd.DataFrame):
        return pd.Series(self.kmeans_classifier.predict(X), name="cluster", index=X.index)

