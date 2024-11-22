import tensorflow as tf


class CombosCrossEntropyBase(tf.keras.losses.Loss):
    """
    Base class for loss functions that compute the crossentropy between the true and predicted group of a combo.
    The group is defined by the matrix attribute, and can be hands, suits, ranks, etc...
    """
    def __init__(self, matrix_fn, name):
        self.matrix = tf.constant(matrix_fn(), dtype=tf.float32)
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        transformed_true = tf.matmul(y_true, self.matrix)
        transformed_pred = tf.matmul(y_pred, self.matrix)
        crossentropy = tf.keras.losses.categorical_crossentropy(transformed_true, transformed_pred)
        return crossentropy