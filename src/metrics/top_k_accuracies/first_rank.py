from src.mappers.combos.categorical.first_rank import first_rank_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboFirstRankAccuracy(TopKComboAccuracyBase):
    def __init__(self, k=1):
        super().__init__(k=k, name=f'Top {k} Combo 1st rank Accuracy', matrix_fn=first_rank_matrix)