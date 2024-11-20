from src.mappers.combos_to_first_card import combos_first_card_matrix
from src.metrics.top_k_combo_accuracy_base import TopKComboAccuracyBase


class TopKComboFirstCardsAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} first_cards Accuracy', matrix_fn=combos_first_card_matrix)
