from src.loss_functions.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos_to_first_rank import combos_first_rank_matrix


class CombosFirstRankCrossEntropy(CombosCrossEntropyBase):


    def __init__(self, name="combos_ranks_crossentropy"):
        super().__init__(combos_first_rank_matrix, name)