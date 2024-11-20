from src.mappers.combos_to_ranks import combos_ranks_matrix
from src.metrics.top_k_combo_accuracy_base import TopKComboAccuracyBase


class TopKComboRanksAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} ranks Accuracy', matrix_fn=combos_ranks_matrix)