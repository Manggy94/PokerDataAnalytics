from src.metrics.coverage_curves.combos import CoverageCurveCombos
from src.metrics.coverage_curves.first_card import CoverageCurveFirstCard
from src.metrics.coverage_curves.first_rank import CoverageCurveFirstRank
from src.metrics.coverage_curves.hands import CoverageCurveHands
from src.metrics.coverage_curves.rank_difference import CoverageCurveRankDifference
from src.metrics.coverage_curves.ranks import CoverageCurveRanks
from src.metrics.coverage_curves.second_card import CoverageCurveSecondCard
from src.metrics.coverage_curves.second_rank import CoverageCurveSecondRank
from src.metrics.coverage_curves.suits import CoverageCurveSuits


coverage_classes = [
    CoverageCurveCombos,
    CoverageCurveFirstCard,
    CoverageCurveFirstRank,
    CoverageCurveHands,
    CoverageCurveRankDifference,
    CoverageCurveRanks,
    CoverageCurveSecondCard,
    CoverageCurveSecondRank,
    CoverageCurveSuits,
]