from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.suits import suits_matrix


class CoverageCurveSuits(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=suits_matrix, target_name="Suits")
