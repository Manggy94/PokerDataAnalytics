import numpy as np
from src.metrics.top_k_combo_accuracy_base import TopKComboAccuracyBase


class TopKCombosAccuracy(TopKComboAccuracyBase):
    def __init__(self, k: int):
        super().__init__(k=k, name=f'Top {k} combos Accuracy', matrix_fn=lambda : np.eye(1326))
