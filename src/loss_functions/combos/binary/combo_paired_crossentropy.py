from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.paired import paired_matrix

class CombosPairedCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_paired_crossentropy"):
        super().__init__(paired_matrix, name)