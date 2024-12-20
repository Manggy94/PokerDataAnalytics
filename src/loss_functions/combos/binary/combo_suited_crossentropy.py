from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.suited import suited_matrix

class CombosSuitedCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_suited_crossentropy"):
        super().__init__(suited_matrix, name)