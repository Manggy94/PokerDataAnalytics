from src.loss_functions.combos.binary.combos_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_offsuit import combos_is_offsuit_matrix

class CombosOffsuitCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_offsuit_crossentropy"):
        super().__init__(combos_is_offsuit_matrix, name)