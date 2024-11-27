from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.super_premium import super_premium_matrix

class CombosSuperPremiumCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_super_premium_crossentropy"):
        super().__init__(super_premium_matrix, name)