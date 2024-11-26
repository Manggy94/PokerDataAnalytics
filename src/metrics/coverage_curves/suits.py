from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_suits import combos_suits_matrix


class CoverageCurveSuits(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_suits_matrix, target_name="Suits")
