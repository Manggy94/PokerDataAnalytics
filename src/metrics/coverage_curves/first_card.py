from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_first_card import combos_first_card_matrix


class CoverageCurveFirstCard(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_first_card_matrix, target_name="First Card")
