from src.loss_functions.combos.binary.combos_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_one_gapper import combos_is_one_gapper_matrix

class CombosOneGapperCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_one_gapper_crossentropy"):
        super().__init__(combos_is_one_gapper_matrix, name)