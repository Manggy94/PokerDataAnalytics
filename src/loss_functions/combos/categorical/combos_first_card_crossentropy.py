from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos.categorical.combos_to_first_card import combos_first_card_matrix


class CombosFirstCardCrossEntropy(CombosCrossEntropyBase):


    def __init__(self, name="combos_cards_crossentropy"):
        super().__init__(combos_first_card_matrix, name)