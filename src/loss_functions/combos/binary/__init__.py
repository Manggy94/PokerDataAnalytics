from src.loss_functions.combos.binary.combo_broadway_crossentropy import CombosBroadwayCrossEntropy
from src.loss_functions.combos.binary.combo_connector_crossentropy import CombosConnectorCrossEntropy
from src.loss_functions.combos.binary.combo_face_crossentropy import CombosFaceCrossEntropy
from src.loss_functions.combos.binary.combo_offsuit_crossentropy import CombosOffsuitCrossEntropy
from src.loss_functions.combos.binary.combo_one_gapper_crossentropy import CombosOneGapperCrossEntropy
from src.loss_functions.combos.binary.combo_paired_crossentropy import CombosPairedCrossEntropy
from src.loss_functions.combos.binary.combo_premium_crossentropy import CombosPremiumCrossEntropy
from src.loss_functions.combos.binary.combo_suited_connector_crossentropy import CombosSuitedConnectorCrossEntropy
from src.loss_functions.combos.binary.combo_suited_crossentropy import CombosSuitedCrossEntropy
from src.loss_functions.combos.binary.combo_super_hand_crossentropy import CombosSuperHandCrossEntropy
from src.loss_functions.combos.binary.combo_super_premium_crossentropy import CombosSuperPremiumCrossEntropy
from src.loss_functions.combos.binary.combo_top_hand_crossentropy import CombosTopHandCrossEntropy
from src.loss_functions.combos.binary.combo_two_gapper_crossentropy import CombosTwoGapperCrossEntropy

binary_crossentropy_classes = [
    CombosBroadwayCrossEntropy,
    CombosConnectorCrossEntropy,
    CombosFaceCrossEntropy,
    CombosOffsuitCrossEntropy,
    CombosOneGapperCrossEntropy,
    CombosPairedCrossEntropy,
    CombosPremiumCrossEntropy,
    CombosSuitedConnectorCrossEntropy,
    CombosSuitedCrossEntropy,
    CombosSuperHandCrossEntropy,
    CombosSuperPremiumCrossEntropy,
    CombosTopHandCrossEntropy,
    CombosTwoGapperCrossEntropy,
]

binary_factor_names = [
    "broadway_factor",
    "connector_factor",
    "face_factor",
    "offsuit_factor",
    "one_gapper_factor",
    "paired_factor",
    "premium_factor",
    "suited_connector_factor",
    "suited_factor",
    "super_hand_factor",
    "super_premium_factor",
    "top_hand_factor",
    "two_gapper_factor",
]

