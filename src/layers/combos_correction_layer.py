import tensorflow as tf


class CombosCorrectionLayer(tf.keras.layers.Layer):
    """
    Layer that corrects the output of a neural network to match the constraints of a combo.
    """
    def __init__(self, **kwargs):
        super(CombosCorrectionLayer, self).__init__(**kwargs)
        self.y_corrector = kwargs.get('y_corrector', None)

    def call(self, inputs):
        y_pred, y_corrector = inputs
        y_pred_corrected = y_pred * y_corrector
        y_pred_corrected = y_pred_corrected / tf.reduce_sum(y_pred_corrected, axis=-1, keepdims=True)
        return y_pred_corrected

