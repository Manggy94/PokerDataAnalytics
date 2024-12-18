from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.super_hand import super_hand_matrix

class CombosSuperHandCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_super_hand_crossentropy"):
        super().__init__(super_hand_matrix, name)