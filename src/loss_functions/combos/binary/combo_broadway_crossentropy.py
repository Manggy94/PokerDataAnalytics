from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.broadway import broadway_matrix

class CombosBroadwayCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_broadway_crossentropy"):
        super().__init__(broadway_matrix, name)