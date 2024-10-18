import pandas as pd
from sklearn.pipeline import Pipeline
from src.pipelines.player_stats.general import GeneralPlayerStatsPipeline
from src.pipelines.player_stats.postflop import PostflopPlayerStatsPipeline
from src.pipelines.player_stats.preflop import PreflopPlayerStatsPipeline
from src.transformers.player_stats.cnt_dropper import CountDropper
from src.transformers.player_stats.player_id_dropper import PlayerIdDropper
from src.transformers.objects_categorizer import ObjectsCategorizer


class PlayerStatsPipeline(Pipeline):

    def __init__(
            self,
            raw_general_player_stats: pd.DataFrame,
            raw_preflop_player_stats: pd.DataFrame,
            raw_flop_player_stats: pd.DataFrame,
            raw_turn_player_stats: pd.DataFrame,
            raw_river_player_stats: pd.DataFrame
    ):
        self.raw_general_player_stats = raw_general_player_stats
        self.raw_preflop_player_stats = raw_preflop_player_stats
        self.raw_flop_player_stats = raw_flop_player_stats
        self.raw_turn_player_stats = raw_turn_player_stats
        self.raw_river_player_stats = raw_river_player_stats
        super().__init__(steps=[
            ("general_player_stats_pipeline", GeneralPlayerStatsPipeline(
                raw_general_player_stats=raw_general_player_stats)),
            ("preflop_player_stats_pipeline", PreflopPlayerStatsPipeline(
                raw_preflop_player_stats=raw_preflop_player_stats)
             ),
            ("postflop_player_stats_pipeline", PostflopPlayerStatsPipeline(
                raw_flop_player_stats=raw_flop_player_stats,
                raw_turn_player_stats=raw_turn_player_stats,
                raw_river_player_stats=raw_river_player_stats)
             ),
            ('player_id_dropper', PlayerIdDropper()),
            ('objects_categorizer', ObjectsCategorizer()),
            ('count_dropper', CountDropper())
        ])