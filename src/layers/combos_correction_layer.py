import tensorflow as tf


class CombosCorrectionLayer(tf.keras.layers.Layer):
    """
    Layer that corrects the output of a neural network to match the constraints of a combo.
    """

    def call(self, y_pred, y_corrector):
        y_pred_corrected = y_pred * y_corrector
        y_pred_corrected = y_pred_corrected / tf.reduce_sum(y_pred_corrected, axis=-1, keepdims=True)
        return y_pred_corrected

