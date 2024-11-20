import tensorflow as tf
from src.mappers.combos_is_broadway import combos_is_broadway_matrix

class CombosBroadwayCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, name="combos_broadway_crossentropy"):
        self.matrix = tf.constant(combos_is_broadway_matrix(), dtype=tf.float32)
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        transformed_true = tf.matmul(y_true, self.matrix)
        transformed_pred = tf.matmul(y_pred, self.matrix)
        crossentropy = tf.keras.losses.binary_crossentropy(transformed_true, transformed_pred)
        return crossentropy