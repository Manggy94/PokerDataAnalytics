from src.loss_functions.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos_to_hands import combos_hands_matrix


class CombosHandsCrossEntropy(CombosCrossEntropyBase):
    """
    Loss function that computes the cross entropy loss between the true and predicted hands of a combo.
    """

    def __init__(self, name="combos_hands_crossentropy"):
        super().__init__(combos_hands_matrix, name)