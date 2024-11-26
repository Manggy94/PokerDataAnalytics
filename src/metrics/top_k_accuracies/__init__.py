from src.metrics.top_k_accuracies.combos import TopKCombosAccuracy
from src.metrics.top_k_accuracies.first_card import TopKComboFirstCardAccuracy
from src.metrics.top_k_accuracies.first_rank import TopKComboFirstRankAccuracy
from src.metrics.top_k_accuracies.hands import TopKComboHandsAccuracy
from src.metrics.top_k_accuracies.rank_difference import TopKComboRankDifferenceAccuracy
from src.metrics.top_k_accuracies.ranks import TopKComboRanksAccuracy
from src.metrics.top_k_accuracies.second_card import TopKComboSecondCardAccuracy
from src.metrics.top_k_accuracies.second_rank import TopkComboSecondRankAccuracy
from src.metrics.top_k_accuracies.suits import TopKComboSuitsAccuracy

top_k_accuracy_classes = [
    TopKCombosAccuracy,
    TopKComboFirstCardAccuracy,
    TopKComboFirstRankAccuracy,
    TopKComboHandsAccuracy,
    TopKComboRankDifferenceAccuracy,
    TopKComboRanksAccuracy,
    TopKComboSecondCardAccuracy,
    TopkComboSecondRankAccuracy,
    TopKComboSuitsAccuracy
]
