import tensorflow as tf
from src.loss_functions.combos_hands_crossentropy import CombosHandsCrossEntropy
from src.loss_functions.combos_ranks_crossentropy import CombosRanksCrossEntropy
from src.loss_functions.combos_suits_crossentropy import CombosSuitsCrossEntropy


class CombosCrossEntropy(tf.keras.losses.Loss):
    def __init__(
            self,
            name="combos_crossentropy",
            combos_factor = 1,
            hands_factor = 1,
            suits_factor = 1,
            ranks_factor = 1
    ):
        self.combos_factor = combos_factor
        self.hands_factor = hands_factor
        self.suits_factor = suits_factor
        self.ranks_factor = ranks_factor

        self.hands_loss_function = CombosHandsCrossEntropy()
        self.suits_loss_function = CombosSuitsCrossEntropy()
        self.ranks_loss_function = CombosRanksCrossEntropy()
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        combos_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
        hands_loss = tf.reduce_mean(self.hands_loss_function.call(y_true, y_pred))
        suits_loss = tf.reduce_mean(self.suits_loss_function.call(y_true, y_pred))
        ranks_loss = tf.reduce_mean(self.ranks_loss_function.call(y_true, y_pred))
        losses = tf.stack([
            combos_loss,
            hands_loss,
            suits_loss,
            ranks_loss])
        factors = tf.constant([
            self.combos_factor,
            self.hands_factor,
            self.suits_factor,
            self.ranks_factor], dtype=tf.float32)
        total_loss = tf.reduce_sum(losses * factors)
        return total_loss