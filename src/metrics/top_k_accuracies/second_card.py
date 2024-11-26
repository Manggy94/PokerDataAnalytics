from src.mappers.combos.categorical.combos_to_first_rank import combos_first_rank_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboFirstRankAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} first ranks Accuracy', matrix_fn=combos_first_rank_matrix)
