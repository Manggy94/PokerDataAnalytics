from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.top_hand import top_hand_matrix

class CombosTopHandCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_top_hand_crossentropy"):
        super().__init__(top_hand_matrix, name)