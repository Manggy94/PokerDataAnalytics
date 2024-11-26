from src.metrics.coverage_curves.coverage_curve import CoverageCurve
from src.mappers.combos.categorical.combos_to_combos import combos_combos_matrix


class CoverageCurveCombos(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=combos_combos_matrix, target_name="Combos")
