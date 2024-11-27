from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.hands import hands_matrix


class CoverageCurveHands(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=hands_matrix, target_name="Hands")
