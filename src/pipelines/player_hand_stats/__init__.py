import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers.na_bool_filler import NaBoolFiller
from src.transformers.player_hand_stats.bb_normalizer import BBNormalizer
from src.transformers.player_hand_stats.player_hand_stats_hand_history_merger import HandStatsHandHistoryMerger
from src.transformers.player_hand_stats.player_hand_stats_general_merger import HandStatsGeneralMerger
from src.transformers.player_hand_stats.player_hand_stats_player_stats_merger import PlayerHandStatsPlayerStatsMerger
from src.transformers.player_hand_stats.player_hand_stats_street_merger import HandStatsStreetMerger
from src.transformers.player_hand_stats.category_transformer import CategoryTransformer

class PlayerHandStatsPipeline(Pipeline):
    def __init__(
            self,
            hand_histories: pd.DataFrame,
            player_stats: pd.DataFrame,
            general_player_hand_stats: pd.DataFrame,
            preflop_player_hand_stats: pd.DataFrame,
            flop_player_hand_stats: pd.DataFrame,
            turn_player_hand_stats: pd.DataFrame,
            river_player_hand_stats: pd.DataFrame
    ):
        self.hand_histories = hand_histories
        self.player_stats = player_stats
        self.general_player_hand_stats = general_player_hand_stats
        self.preflop_player_hand_stats = preflop_player_hand_stats
        self.flop_player_hand_stats = flop_player_hand_stats
        self.turn_player_hand_stats = turn_player_hand_stats
        self.river_player_hand_stats = river_player_hand_stats
        super().__init__(steps=[
            ("hand_stats_hand_history_merger", HandStatsHandHistoryMerger(hand_histories)),
            ("hand_stats_general_merger", HandStatsGeneralMerger(general_player_hand_stats)),
            ("hand_stats_preflop_merger", HandStatsStreetMerger(preflop_player_hand_stats, "preflop")),
            ("hand_stats_flop_merger", HandStatsStreetMerger(flop_player_hand_stats, "flop")),
            ("hand_stats_turn_merger", HandStatsStreetMerger(turn_player_hand_stats, "turn")),
            ("hand_stats_river_merger", HandStatsStreetMerger(river_player_hand_stats, "river")),
            ("player_hand_stats_player_stats_merger", PlayerHandStatsPlayerStatsMerger(player_stats)),
            ("category_transformer", CategoryTransformer()),
            ("bb_normalizer", BBNormalizer()),
            ("na_bool_filler", NaBoolFiller())
        ])