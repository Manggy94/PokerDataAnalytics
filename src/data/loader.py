from src.loaders.dynamic.hand_histories import HandHistoriesLoader
from src.loaders.dynamic.player_hand_stats import PlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.flop import FlopPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.general import GeneralPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.preflop import PreflopPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.river import RiverPlayerHandStatsLoader
from src.loaders.dynamic.player_hand_stats.turn import TurnPlayerHandStatsLoader
from src.loaders.dynamic.player_stats import PlayerStatsLoader
from src.loaders.dynamic.player_stats.general import GeneralPlayerStatsLoader
from src.loaders.dynamic.player_stats.postflop import PostflopPlayerStatsLoader
from src.loaders.dynamic.player_stats.preflop import PreflopPlayerStatsLoader
from src.loaders.dynamic.ref_tournaments import RefTournamentsLoader
from src.loaders.dynamic.tournaments import TournamentsLoader
from src.loaders.fixed.cards import CardsLoader
from src.loaders.fixed.combos import CombosLoader
from src.loaders.fixed.flops import FlopsLoader
from src.loaders.fixed.hands import HandsLoader
from src.loaders.fixed.positions import PositionsLoader


class DataLoader:

    @staticmethod
    def load_cards():
        return CardsLoader().fit_transform(None)

    @staticmethod
    def load_combos():
        return CombosLoader().fit_transform(None)

    @staticmethod
    def load_flops(self):
        return FlopsLoader().fit_transform(None)

    @staticmethod
    def load_hands(self):
       return HandsLoader().fit_transform(None)

    @staticmethod
    def load_positions():
       return PositionsLoader().fit_transform(None)

    @staticmethod
    def load_ref_tournaments():
        return RefTournamentsLoader().fit_transform(None)

    @staticmethod
    def load_tournaments():
        return TournamentsLoader().fit_transform(None)

    @staticmethod
    def load_hand_histories():
        return HandHistoriesLoader().fit_transform(None)

    @staticmethod
    def load_general_player_hand_stats():
        return GeneralPlayerHandStatsLoader().fit_transform(None)

    @staticmethod
    def load_preflop_player_hand_stats():
        return PreflopPlayerHandStatsLoader().fit_transform(None)

    @staticmethod
    def load_flop_player_hand_stats():
        return FlopPlayerHandStatsLoader().fit_transform(None)

    @staticmethod
    def load_turn_player_hand_stats():
        return TurnPlayerHandStatsLoader().fit_transform(None)

    @staticmethod
    def load_river_player_hand_stats():
        return RiverPlayerHandStatsLoader().fit_transform(None)

    @staticmethod
    def load_player_hand_stats():
        return PlayerHandStatsLoader().fit_transform(None)

    def load_showdown_hands(self):
        phs =  self.load_player_hand_stats()
        return phs[phs["flag_went_to_showdown"] == True]

    def load_villain_hands(self):
        phs =  self.load_player_hand_stats()
        return phs[phs["flag_is_hero"] == False]

    def  load_villain_showdown_hands(self):
        phs =  self.load_villain_hands()
        return phs[phs["flag_went_to_showdown"] == True]

    def load_revaled_hands(self):
        phs =  self.load_player_hand_stats()
        return phs.dropna(subset=["player_combo"])

    @staticmethod
    def load_general_player_stats():
        return GeneralPlayerStatsLoader().fit_transform(None)

    @staticmethod
    def load_preflop_player_stats():
        return PreflopPlayerStatsLoader().fit_transform(None)

    @staticmethod
    def load_postflop_player_stats():
        return PostflopPlayerStatsLoader().fit_transform(None)

    @staticmethod
    def load_player_stats():
        return PlayerStatsLoader().fit_transform(None).set_index('name').drop(columns=['player'])

    def load_villain_player_stats(self):
        return self.load_player_stats().drop(index='manggy94')