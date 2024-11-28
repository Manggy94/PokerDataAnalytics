import tensorflow as tf
from src.loss_functions.combos.categorical import categorical_crossentropy_classes, categorical_factor_names
from src.loss_functions.combos.binary import binary_crossentropy_classes, binary_factor_names


class CombosCrossEntropy(tf.keras.losses.Loss):
    def __init__(
            self,
            name="combos_crossentropy",
            **kwargs
    ):
        binary_loss_functions = [loss_function() for loss_function in binary_crossentropy_classes]
        binary_facors = [1] * len(binary_loss_functions)
        categorical_loss_functions = [loss_function() for loss_function in categorical_crossentropy_classes]
        categorical_factors = [1] * len(categorical_loss_functions)
        for kwarg in kwargs:
            if kwarg in binary_factor_names:
                binary_facors[binary_factor_names.index(kwarg)] = kwargs[kwarg]
            if kwarg in categorical_factor_names:
                categorical_factors[categorical_factor_names.index(kwarg)] = kwargs[kwarg]
        self.factors = tf.constant(binary_facors + categorical_factors, dtype=tf.float32)
        self.loss_functions = binary_loss_functions + categorical_loss_functions

        super().__init__(name=name)

    def call(self, y_true, y_pred):
        losses = tf.stack([tf.reduce_mean(loss_function.call(y_true, y_pred)) for loss_function in self.loss_functions])
        total_loss = tf.reduce_sum(losses * self.factors)
        return total_loss