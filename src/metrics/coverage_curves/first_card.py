from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.first_card import first_card_matrix


class CoverageCurveFirstCard(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=first_card_matrix, target_name="First Card")
