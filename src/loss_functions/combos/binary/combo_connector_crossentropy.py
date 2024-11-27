from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.connector import connector_matrix

class CombosConnectorCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_connector_crossentropy"):
        super().__init__(connector_matrix, name)