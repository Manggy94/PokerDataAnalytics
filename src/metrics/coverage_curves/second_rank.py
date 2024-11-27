from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.second_rank import second_rank_matrix


class CoverageCurveSecondRank(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=second_rank_matrix, target_name="Second Rank")
