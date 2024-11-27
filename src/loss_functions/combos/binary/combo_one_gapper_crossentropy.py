from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.one_gapper import one_gapper_matrix

class CombosOneGapperCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_one_gapper_crossentropy"):
        super().__init__(one_gapper_matrix, name)