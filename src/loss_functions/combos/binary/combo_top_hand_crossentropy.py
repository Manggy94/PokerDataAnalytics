from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_top_hand import combos_is_top_hand_matrix

class CombosTopHandCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_top_hand_crossentropy"):
        super().__init__(combos_is_top_hand_matrix, name)