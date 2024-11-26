from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_second_rank import combos_second_rank_matrix


class CoverageCurveSecondRank(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_second_rank_matrix, target_name="Second Rank")
