import tensorflow as tf
from src.metrics.coverage_curves.base import CoverageCurveBase

class CoverageCurve(CoverageCurveBase):
    def __init__(self, matrix_fn, target_name=None):
        self.matrix = tf.constant(matrix_fn(), dtype=tf.float32)
        self.target_name = target_name
        super().__init__()

    def transform(self, vector):
        if vector.shape[1] != self.matrix.shape[0]:
            return vector
        transformed_vector = tf.matmul(tf.cast(vector, dtype=tf.float32), self.matrix)
        return transformed_vector

    def compute_coverage_curve(self, y_true, y_pred):
        transformed_y_true, transformed_y_pred = self.transform(y_true), self.transform(y_pred)
        return super().compute_coverage_curve(transformed_y_true, transformed_y_pred)

    def plot_coverage_curve(self, y_true, y_proba, dummy_proba):
        transformed_y_true, transformed_y_proba = self.transform(y_true), self.transform(y_proba)
        transformed_dummy_proba = self.transform(dummy_proba) if dummy_proba is not None else None
        return super().plot_coverage_curve(transformed_y_true, transformed_y_proba, target=self.target_name, dummy_proba=transformed_dummy_proba)




