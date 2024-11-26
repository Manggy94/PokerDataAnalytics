from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_hands import combos_hands_matrix


class CoverageCurveHands(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_hands_matrix, target_name="Hands")
