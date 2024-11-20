import tensorflow as tf


class TopKComboAccuracyBase(tf.keras.metrics.TopKCategoricalAccuracy):
    """
    Base class for metrics that compute the top-k accuracy for combos related groups.
    """

    def __init__(self, name, matrix_fn, k=1):
        self.matrix = tf.constant(matrix_fn(), dtype=tf.float32)
        super().__init__(k=k, name=name)

    def update_state(self, y_true, y_pred, sample_weight=None):
        transformed_y_true = tf.matmul(tf.cast(y_true, dtype=tf.float32), self.matrix)
        transformed_y_pred = tf.matmul(tf.cast(y_pred, dtype=tf.float32), self.matrix)
        super().update_state(transformed_y_true, transformed_y_pred, sample_weight)