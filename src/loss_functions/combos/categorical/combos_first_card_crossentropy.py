from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos.categorical.first_card import first_card_matrix


class CombosFirstCardCrossEntropy(CombosCrossEntropyBase):


    def __init__(self, name="combos_cards_crossentropy"):
        super().__init__(first_card_matrix, name)