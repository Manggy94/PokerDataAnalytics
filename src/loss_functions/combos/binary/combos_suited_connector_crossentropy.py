from src.loss_functions.combos.binary.combos_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_suited_connector import combos_is_suited_connector_matrix

class CombosSuitedConnectorCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_suited_connector_crossentropy"):
        super().__init__(combos_is_suited_connector_matrix, name)