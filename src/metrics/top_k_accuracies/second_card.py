from src.mappers.combos.categorical.second_card import second_card_matrix
from src.metrics.top_k_accuracies.base import TopKComboAccuracyBase


class TopKComboSecondCardAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} second cards Accuracy', matrix_fn=second_card_matrix)
