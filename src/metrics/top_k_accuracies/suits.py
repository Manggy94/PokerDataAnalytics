from src.mappers.combos.categorical.combos_to_suits import combos_suits_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboSuitsAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} suits Accuracy', matrix_fn=combos_suits_matrix)