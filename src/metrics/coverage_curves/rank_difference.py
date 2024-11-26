from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_rank_difference import combos_rank_difference_matrix


class CoverageCurveRankDifference(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_rank_difference_matrix, target_name="RankDifferences")
