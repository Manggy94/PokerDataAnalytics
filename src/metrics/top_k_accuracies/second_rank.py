from src.mappers.combos.categorical.combos_to_second_rank import combos_second_rank_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopkComboSecondRankAccuracy(TopKComboAccuracyBase):
    def __init__(self, k=1):
        super().__init__(k=k, name=f"Top {k} Combo 2nd rank Accuracy", matrix_fn=combos_second_rank_matrix)