from src.loss_functions.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos_to_ranks import combos_ranks_matrix


class CombosRanksCrossEntropy(CombosCrossEntropyBase):


    def __init__(self, name="combos_ranks_crossentropy"):
        super().__init__(combos_ranks_matrix, name)