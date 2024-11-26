from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_connector import combos_is_connector_matrix

class CombosConnectorCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_connector_crossentropy"):
        super().__init__(combos_is_connector_matrix, name)