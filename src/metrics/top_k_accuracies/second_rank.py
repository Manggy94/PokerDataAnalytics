from src.mappers.combos.categorical.second_rank import second_rank_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopkComboSecondRankAccuracy(TopKComboAccuracyBase):
    def __init__(self, k=1):
        super().__init__(k=k, name=f"Top {k} Combo 2nd rank Accuracy", matrix_fn=second_rank_matrix)