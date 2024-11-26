from src.loss_functions.combos.categorical.combos_combos_crossentropy import CombosCombosCrossEntropy
from src.loss_functions.combos.categorical.combos_first_card_crossentropy import CombosFirstCardCrossEntropy
from src.loss_functions.combos.categorical.combos_first_rank_crossentropy import CombosFirstRankCrossEntropy
from src.loss_functions.combos.categorical.combos_hands_crossentropy import CombosHandsCrossEntropy
from src.loss_functions.combos.categorical.combos_ranks_crossentropy import CombosRanksCrossEntropy
from src.loss_functions.combos.categorical.combos_ranks_difference_crossentropy import CombosRankDifferenceCrossEntropy
from src.loss_functions.combos.categorical.combos_second_card_crossentropy import CombosSecondCardCrossEntropy
from src.loss_functions.combos.categorical.combos_second_rank_crossentropy import CombosSecondRankCrossEntropy
from src.loss_functions.combos.categorical.combos_suits_crossentropy import CombosSuitsCrossEntropy

categorical_crossentropy_classes = [
    CombosCombosCrossEntropy,
    CombosFirstCardCrossEntropy,
    CombosFirstRankCrossEntropy,
    CombosHandsCrossEntropy,
    CombosRanksCrossEntropy,
    CombosRankDifferenceCrossEntropy,
    CombosSecondCardCrossEntropy,
    CombosSecondRankCrossEntropy,
    CombosSuitsCrossEntropy,
]

categorical_factor_names = [
    "combos_factor",
    "first_card_factor",
    "first_rank_factor",
    "hands_factor",
    "ranks_factor",
    "rank_difference_factor",
    "second_card_factor",
    "second_rank_factor",
    "suits_factor",
]
