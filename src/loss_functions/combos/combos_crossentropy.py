import tensorflow as tf
from src.loss_functions.combos.binary.combo_broadway_crossentropy import CombosBroadwayCrossEntropy
from src.loss_functions.combos.binary.combos_connector_crossentropy import CombosConnectorCrossEntropy
from src.loss_functions.combos.binary.combos_face_crossentropy import CombosFaceCrossEntropy
from src.loss_functions.combos.categorical.combos_first_card_crossentropy import CombosFirstCardCrossEntropy
from src.loss_functions.combos.categorical.combos_first_rank_crossentropy import CombosFirstRankCrossEntropy
from src.loss_functions.combos.categorical.combos_hands_crossentropy import CombosHandsCrossEntropy
from src.loss_functions.combos.binary.combos_offsuit_crossentropy import CombosOffsuitCrossEntropy
from src.loss_functions.combos.binary.combos_one_gapper_crossentropy import CombosOneGapperCrossEntropy
from src.loss_functions.combos.binary.combos_paired_crossentropy import CombosPairedCrossEntropy
from src.loss_functions.combos.categorical.combos_ranks_crossentropy import CombosRanksCrossEntropy
from src.loss_functions.combos.categorical.combos_ranks_difference_crossentropy import CombosRankDifferenceCrossEntropy
from src.loss_functions.combos.categorical.combos_second_card_crossentropy import CombosSecondCardCrossEntropy
from src.loss_functions.combos.categorical.combos_second_rank_crossentropy import CombosSecondRankCrossEntropy
from src.loss_functions.combos.binary.combos_suited_connector_crossentropy import CombosSuitedConnectorCrossEntropy
from src.loss_functions.combos.binary.combos_suited_crossentropy import CombosSuitedCrossEntropy
from src.loss_functions.combos.categorical.combos_suits_crossentropy import CombosSuitsCrossEntropy
from src.loss_functions.combos.binary.combos_two_gapper_crossentropy import CombosTwoGapperCrossEntropy


class CombosSuitedConnectorCross:
    pass


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
            rank_difference_factor = 1,
    ):
        # Fixed factors
        broadway_factor = 1
        face_factor = 1
        offsuit_factor = 1
        suited_factor = 1
        paired_factor = 1
        connectors_factor = 1
        one_gapper_factor = 1
        two_gapper_factor = 1
        suited_connector_factor = 1

        self.factors = tf.constant([
            # Parameter factors
            combos_factor,
            hands_factor,
            suits_factor,
            ranks_factor,
            first_rank_factor,
            first_card_factor,
            second_rank_factor,
            second_card_factor,
            rank_difference_factor,
            # Fixed factors
            broadway_factor,
            face_factor,
            offsuit_factor,
            suited_factor,
            paired_factor,
            connectors_factor,
            one_gapper_factor,
            two_gapper_factor,
            suited_connector_factor,
            
            

        ], dtype=tf.float32)
        self.loss_functions = [
            # Parameter factor losses
            tf.keras.losses.CategoricalCrossentropy(),
            CombosHandsCrossEntropy(),
            CombosSuitsCrossEntropy(),
            CombosRanksCrossEntropy(),
            CombosFirstRankCrossEntropy(),
            CombosFirstCardCrossEntropy(),
            CombosSecondRankCrossEntropy(),
            CombosSecondCardCrossEntropy(),
            CombosRankDifferenceCrossEntropy(),
            # Fixed factor losses
            CombosBroadwayCrossEntropy(),
            CombosFaceCrossEntropy(),
            CombosOffsuitCrossEntropy(),
            CombosSuitedCrossEntropy(),
            CombosPairedCrossEntropy(),
            CombosConnectorCrossEntropy(),
            CombosOneGapperCrossEntropy(),
            CombosTwoGapperCrossEntropy(),
            CombosSuitedConnectorCrossEntropy(),
            
        ]
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        losses = tf.stack([tf.reduce_mean(loss_function.call(y_true, y_pred)) for loss_function in self.loss_functions])
        total_loss = tf.reduce_sum(losses * self.factors)
        return total_loss