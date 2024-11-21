import tensorflow as tf
from src.loss_functions.combo_broadway_crossentropy import CombosBroadwayCrossEntropy
from src.loss_functions.combos_connector_crossentropy import CombosConnectorCrossEntropy
from src.loss_functions.combos_first_card_crossentropy import CombosFirstCardCrossEntropy
from src.loss_functions.combos_first_rank_crossentropy import CombosFirstRankCrossEntropy
from src.loss_functions.combos_hands_crossentropy import CombosHandsCrossEntropy
from src.loss_functions.combos_paired_crossentropy import CombosPairedCrossEntropy
from src.loss_functions.combos_ranks_crossentropy import CombosRanksCrossEntropy
from src.loss_functions.combos_ranks_difference_crossentropy import CombosRankDifferenceCrossEntropy
from src.loss_functions.combos_second_card_crossentropy import CombosSecondCardCrossEntropy
from src.loss_functions.combos_second_rank_crossentropy import CombosSecondRankCrossEntropy
from src.loss_functions.combos_suits_crossentropy import CombosSuitsCrossEntropy


class CombosCrossEntropy(tf.keras.losses.Loss):
    def __init__(
            self,
            name="combos_crossentropy",
            combos_factor = 1,
            hands_factor = 1,
            suits_factor = 1,
            ranks_factor = 1,
            first_rank_factor = 1,
            first_card_factor = 1,
            second_rank_factor = 1,
            second_card_factor = 1,
            broadway_factor = 1,
            connectors_factor = 1,
            paired_factor = 1,
            rank_difference_factor = 1,
    ):
        self.factors = tf.constant([
            combos_factor,
            hands_factor,
            suits_factor,
            ranks_factor,
            first_rank_factor,
            first_card_factor,
            second_rank_factor,
            second_card_factor,
            broadway_factor,
            connectors_factor,
            paired_factor,
            rank_difference_factor,

        ], dtype=tf.float32)
        self.loss_functions = [
            tf.keras.losses.CategoricalCrossentropy(),
            CombosHandsCrossEntropy(),
            CombosSuitsCrossEntropy(),
            CombosRanksCrossEntropy(),
            CombosFirstRankCrossEntropy(),
            CombosFirstCardCrossEntropy(),
            CombosSecondRankCrossEntropy(),
            CombosSecondCardCrossEntropy(),
            CombosBroadwayCrossEntropy(),
            CombosConnectorCrossEntropy(),
            CombosPairedCrossEntropy(),
            CombosRankDifferenceCrossEntropy(),
        ]
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        losses = tf.stack([tf.reduce_mean(loss_function.call(y_true, y_pred)) for loss_function in self.loss_functions])
        total_loss = tf.reduce_sum(losses * self.factors)
        return total_loss