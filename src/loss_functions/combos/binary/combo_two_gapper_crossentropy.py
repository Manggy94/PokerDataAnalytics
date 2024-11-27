from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.two_gapper import two_gapper_matrix

class CombosTwoGapperCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_two_gapper_crossentropy"):
        super().__init__(two_gapper_matrix, name)