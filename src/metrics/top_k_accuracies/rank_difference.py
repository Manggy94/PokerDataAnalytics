from src.mappers.combos.categorical.rank_difference import rank_difference_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboRankDifferenceAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} rank_difference Accuracy', matrix_fn=rank_difference_matrix)
