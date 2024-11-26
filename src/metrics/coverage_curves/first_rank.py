from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_first_rank import combos_first_rank_matrix


class CoverageCurveFirstRank(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_first_rank_matrix, target_name="First Rank")
