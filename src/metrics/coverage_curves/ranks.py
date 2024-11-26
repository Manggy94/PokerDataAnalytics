from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_ranks import combos_ranks_matrix


class CoverageCurveRanks(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_ranks_matrix, target_name="Ranks")
