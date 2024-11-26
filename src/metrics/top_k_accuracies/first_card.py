from src.mappers.combos.categorical.combos_to_first_card import combos_first_card_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboFirstCardAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} first cards Accuracy', matrix_fn=combos_first_card_matrix)
