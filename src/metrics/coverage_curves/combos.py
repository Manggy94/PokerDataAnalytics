import numpy as np
from src.metrics.coverage_curves.coverage_curve import CoverageCurve


class CoverageCurveCombos(CoverageCurve):
    def __init__(self):
        super().__init__(matrix_fn=lambda: np.eye(1326), target_name="Combos")
