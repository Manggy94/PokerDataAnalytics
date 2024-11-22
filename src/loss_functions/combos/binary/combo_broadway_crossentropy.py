from src.loss_functions.combos.binary.combos_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_broadway import combos_is_broadway_matrix

class CombosBroadwayCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_broadway_crossentropy"):
        super().__init__(combos_is_broadway_matrix, name)