from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.offsuit import offsuit_matrix

class CombosOffsuitCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_offsuit_crossentropy"):
        super().__init__(offsuit_matrix, name)