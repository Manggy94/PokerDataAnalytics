from src.loss_functions.combos.categorical.combos_crossentropy_base import CombosCrossEntropyBase
from src.mappers.combos.categorical.rank_difference import rank_difference_matrix


class CombosRankDifferenceCrossEntropy(CombosCrossEntropyBase):

    def __init__(self, name="combos_rank_crossentropy"):
        super().__init__(rank_difference_matrix, name)