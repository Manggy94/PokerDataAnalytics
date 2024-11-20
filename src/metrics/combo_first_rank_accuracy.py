from src.mappers.combos_to_first_rank import combos_first_rank_matrix
from src.metrics.top_k_combo_accuracy_base import TopKComboAccuracyBase


class ComboFirstRankAccuracy(TopKComboAccuracyBase):
    def __init__(self):
        super().__init__(k=1, name='Combo 1st rank Accuracy', matrix_fn=combos_first_rank_matrix)