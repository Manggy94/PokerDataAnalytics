from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.second_card import second_card_matrix


class CoverageCurveSecondCard(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=second_card_matrix, target_name="Second Card")
