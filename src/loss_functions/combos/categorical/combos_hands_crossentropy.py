from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos.categorical.hands import hands_matrix


class CombosHandsCrossEntropy(CombosCrossEntropyBase):
    """
    Loss function that computes the cross entropy loss between the true and predicted hands of a combo.
    """

    def __init__(self, name="combos_hands_crossentropy"):
        super().__init__(hands_matrix, name)