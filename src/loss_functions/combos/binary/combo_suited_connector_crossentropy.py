from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.suited_connector import suited_connector_matrix

class CombosSuitedConnectorCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_suited_connector_crossentropy"):
        super().__init__(suited_connector_matrix, name)