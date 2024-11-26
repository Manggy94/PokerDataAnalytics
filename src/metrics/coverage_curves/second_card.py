from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_second_card import combos_second_card_matrix


class CoverageCurveSecondCard(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_second_card_matrix, target_name="Second Card")
