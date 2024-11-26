from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_super_premium import combos_is_super_premium_matrix

class CombosSuperPremiumCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_super_premium_crossentropy"):
        super().__init__(combos_is_super_premium_matrix, name)