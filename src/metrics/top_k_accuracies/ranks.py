from src.mappers.combos.categorical.ranks import ranks_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboRanksAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} ranks Accuracy', matrix_fn=ranks_matrix)