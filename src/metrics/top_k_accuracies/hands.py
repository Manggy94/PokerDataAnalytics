from src.mappers.combos.categorical.hands import hands_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboHandsAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} hands Accuracy', matrix_fn=hands_matrix)
