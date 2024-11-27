from src.mappers.combos.binary.broadway import broadway_matrix
from src.metrics.accuracies.base import BinaryAccuracyBase


class BroadwayAccuracy(BinaryAccuracyBase):
    def __init__(self, name="broadway_accuracy"):
        super().__init__(broadway_matrix, name)