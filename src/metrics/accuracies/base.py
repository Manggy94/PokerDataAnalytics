import tensorflow as tf

class BinaryAccuracyBase(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, matrix_fn, name=None):
        self.matrix = tf.constant(matrix_fn(), dtype=tf.float32)
        super().__init__(name=name)

    def transform(self, vector):
        if vector.shape[1] == 1:
            return vector
        transformed_vector = tf.matmul(tf.cast(vector, dtype=tf.float32), self.matrix)
        return transformed_vector

    def update_state(self, y_true, y_pred, sample_weight=None):
        transformed_y_true, transformed_y_pred = self.transform(y_true), self.transform(y_pred)
        super().update_state(transformed_y_true, transformed_y_pred, sample_weight)