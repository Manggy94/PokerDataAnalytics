from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.first_rank import first_rank_matrix


class CoverageCurveFirstRank(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=first_rank_matrix, target_name="First Rank")
