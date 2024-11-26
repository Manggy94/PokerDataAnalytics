from src.mappers.combos.categorical.combos_to_second_card import combos_second_card_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboSecondCardAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} second cards Accuracy', matrix_fn=combos_second_card_matrix)
