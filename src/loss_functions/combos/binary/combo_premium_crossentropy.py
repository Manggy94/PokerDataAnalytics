from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.premium import premium_matrix

class CombosPremiumCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_premium_crossentropy"):
        super().__init__(premium_matrix, name)