from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos.categorical.second_card import second_card_matrix


class CombosSecondCardCrossEntropy(CombosCrossEntropyBase):


    def __init__(self, name="combos_cards_crossentropy"):
        super().__init__(second_card_matrix, name)