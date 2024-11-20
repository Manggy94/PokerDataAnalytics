from src.mappers.combos_to_second_rank import combos_second_rank_matrix
from src.metrics.top_k_combo_accuracy_base import TopKComboAccuracyBase


class ComboSecondRankAccuracy(TopKComboAccuracyBase):
    def __init__(self):
        super().__init__(k=1, name='Combo 2nd rank Accuracy', matrix_fn=combos_second_rank_matrix)