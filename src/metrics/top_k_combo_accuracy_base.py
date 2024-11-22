import tensorflow as tf


class TopKComboAccuracyBase(tf.keras.metrics.TopKCategoricalAccuracy):
    """
    Base class for metrics that compute the top-k accuracy for combos related groups.
    """

    def __init__(self, name, matrix_fn, k=1):
        self.matrix = tf.constant(matrix_fn(), dtype=tf.float32)
        self.k = k
        super().__init__(k=k, name=name)

    def transform(self, vector):
        transformed_vector = tf.matmul(tf.cast(vector, dtype=tf.float32), self.matrix)
        return transformed_vector

    def update_state(self, y_true, y_pred, sample_weight=None):
        transformed_y_true, transformed_y_pred = self.transform(y_true), self.transform(y_pred)
        super().update_state(transformed_y_true, transformed_y_pred, sample_weight)

