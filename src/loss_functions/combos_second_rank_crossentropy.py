from src.loss_functions.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos_to_second_rank import combos_second_rank_matrix


class CombosSecondRankCrossEntropy(CombosCrossEntropyBase):


    def __init__(self, name="combos_ranks_crossentropy"):
        super().__init__(combos_second_rank_matrix, name)