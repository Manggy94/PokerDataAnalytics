from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.ranks import ranks_matrix


class CoverageCurveRanks(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=ranks_matrix, target_name="Ranks")
