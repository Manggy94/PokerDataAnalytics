import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.player_hand_stats.player_hand_stats_hand_history_merger import HandStatsHandHistoryMerger
from src.transformers.player_hand_stats.player_hand_stats_general_merger import HandStatsGeneralMerger
from src.transformers.player_hand_stats.player_hand_stats_preflop_merger import HandStatsPreflopMerger
from src.transformers.player_hand_stats.bb_normalizer import BBNormalizer

class PlayerHandStatsPipeline(Pipeline):
    def __init__(
            self, hand_histories: pd.DataFrame,
            general_player_hand_stats: pd.DataFrame,
            preflop_player_hand_stats: pd.DataFrame):
        self.hand_histories = hand_histories
        self.general_player_hand_stats = general_player_hand_stats
        self.preflop_player_hand_stats = preflop_player_hand_stats
        super().__init__(steps=[
            ("hand_stats_hand_history_merger", HandStatsHandHistoryMerger(hand_histories)),
            ("hand_stats_general_merger", HandStatsGeneralMerger(general_player_hand_stats)),
            ("hand_stats_preflop_merger", HandStatsPreflopMerger(preflop_player_hand_stats)),
            ("bb_normalizer", BBNormalizer())
        ])