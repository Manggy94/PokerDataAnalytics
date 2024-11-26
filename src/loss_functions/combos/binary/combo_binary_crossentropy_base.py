import tensorflow as tf


class CombosBinaryCrossEntropy(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, matrix_fn, name=None):
        self.matrix = tf.constant(matrix_fn(), dtype=tf.float32)
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        transformed_true = tf.matmul(y_true, self.matrix)
        transformed_pred = tf.matmul(y_pred, self.matrix)
        return super().call(transformed_true, transformed_pred)
