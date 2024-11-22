from src.mappers.combos.categorical.combos_to_hands import combos_hands_matrix
from src.metrics.top_k_combo_accuracy_base import TopKComboAccuracyBase


class TopKComboHandsAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} hands Accuracy', matrix_fn=combos_hands_matrix)
