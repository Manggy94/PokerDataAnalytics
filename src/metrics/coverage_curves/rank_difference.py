from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.rank_difference import rank_difference_matrix


class CoverageCurveRankDifference(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=rank_difference_matrix, target_name="RankDifferences")
