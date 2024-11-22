from src.mappers.combos.categorical.combos_to_suits import combos_suits_matrix
from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase


class CombosSuitsCrossEntropy(CombosCrossEntropyBase):
    """
    Loss function that computes the crossentropy between the true and predicted suits of a combo.
    """
    def __init__(self, name="combos_suits_crossentropy"):
        super().__init__(combos_suits_matrix, name)
