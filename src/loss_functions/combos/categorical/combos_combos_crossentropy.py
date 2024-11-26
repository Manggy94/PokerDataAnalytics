from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos.categorical.combos_to_combos import combos_combos_matrix


class CombosCombosCrossEntropy(CombosCrossEntropyBase):
    """
    Loss function that computes the cross entropy loss between the true and predicted combos of a combo.
    """

    def __init__(self, name="combos_combos_crossentropy"):
        super().__init__(combos_combos_matrix, name)