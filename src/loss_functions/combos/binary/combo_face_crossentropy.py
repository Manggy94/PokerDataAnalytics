from src.loss_functions.combos.binary.combo_binary_crossentropy_base import CombosBinaryCrossEntropy
from src.mappers.combos.binary.combos_is_face import combos_is_face_matrix

class CombosFaceCrossEntropy(CombosBinaryCrossEntropy):
    def __init__(self, name="combos_face_crossentropy"):
        super().__init__(combos_is_face_matrix, name)