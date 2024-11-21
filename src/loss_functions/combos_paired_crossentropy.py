from src.loss_functions.combos_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos_is_paired import combos_is_paired_matrix

class CombosPairedCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_paired_crossentropy"):
        super().__init__(combos_is_paired_matrix, name)