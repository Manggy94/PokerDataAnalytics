import numpy as np
from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase


class CombosCombosCrossEntropy(CombosCrossEntropyBase):
    """
    Loss function that computes the cross entropy loss between the true and predicted combos of a combo.
    """

    def __init__(self, name="combos_combos_crossentropy"):
        super().__init__(matrix_fn=lambda: np.eye(1326), name=name)