import pandas as pd
import numpy as np
from config.settings import ANALYTICS_DATA_DIR


class DataLoader:

    def __init__(self):
        self.ANALYTICS_DATA_DIR = ANALYTICS_DATA_DIR

    def load_raw_tournaments(self):
        tournaments = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/tournaments.csv', index_col=0)
        return tournaments

    def load_raw_ref_tournaments(self):
        ref_tournaments = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/ref_tournaments.csv', index_col=0)
        return ref_tournaments

    def load_raw_buy_ins(self):
        buy_ins = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/buy_ins.csv', index_col=0)
        return buy_ins

    def load_raw_tour_speeds(self):
        tour_speeds = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/tour_speeds.csv', index_col=0)
        return tour_speeds

    def load_raw_tour_types(self):
        tour_types = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/tour_types.csv', index_col=0)
        return tour_types

    def load_tournaments(self):
        raw_tournaments = self.load_raw_tournaments()
        raw_tournaments.tournament_id = raw_tournaments.tournament_id.astype("str")
        raw_tournaments['final_position'] = raw_tournaments['final_position'].fillna(raw_tournaments['total_players']).astype("int32")
        raw_ref_tournaments = self.load_raw_ref_tournaments()
        raw_buy_ins = self.load_raw_buy_ins()
        raw_tour_speeds = self.load_raw_tour_speeds()
        raw_tour_types = self.load_raw_tour_types()
        tournaments = raw_tournaments\
            .merge(raw_ref_tournaments, how='left', left_on='ref_tournament', right_on='id', suffixes=('', '_ref'))\
            .merge(raw_buy_ins, how='left', left_on='buy_in', right_on='id', suffixes=('', '_buy_in'))\
            .merge(raw_tour_speeds, how='left', left_on='speed', right_on='id', suffixes=('', '_speed'))\
            .merge(raw_tour_types, how='left', left_on='tournament_type', right_on='id', suffixes=('', '_type'))\
            .drop(columns=['id', 'ref_tournament', 'id_ref', 'buy_in', 'id_buy_in', 'speed', 'id_speed',
                           'tournament_type', 'id_type'])\
            .rename(columns={'name_speed': 'speed', 'name_type': 'tournament_type', "total": "buy_in_total"})
        tournaments['start_date'] = pd.to_datetime(tournaments['start_date'])
        tournaments = tournaments.sort_values('start_date')
        tournaments['total_won'] = tournaments['amount_won'] + tournaments['bounty_won']
        tournaments['total_investment'] = tournaments['nb_entries'] * tournaments['buy_in_total']
        tournaments['classic_investment'] = ((tournaments['prize_pool_contribution'] + tournaments['rake']) *
                                             tournaments['nb_entries'])
        tournaments["bounty_investment"] = tournaments["nb_entries"] * tournaments["bounty"]
        tournaments["profit"] = tournaments["total_won"] - tournaments["total_investment"]
        tournaments["freeze_out_profit"] = tournaments["amount_won"] - tournaments["classic_investment"]
        tournaments["bounty_profit"] = tournaments["bounty_won"] - tournaments["bounty_investment"]
        tournaments['roi'] = tournaments["profit"] / tournaments['total_investment']
        tournaments["freeze_out_roi"] = tournaments["freeze_out_profit"] / tournaments["classic_investment"]
        tournaments["bounty_roi"] = tournaments["bounty_profit"] / tournaments["bounty_investment"]
        tournaments['roi'] = tournaments['roi'].replace([np.inf, -np.inf], np.nan)
        tournaments["freeze_out_roi"] = tournaments["freeze_out_roi"].replace([np.inf, -np.inf], np.nan)
        tournaments["bounty_roi"] = tournaments["bounty_roi"].replace([np.inf, -np.inf], np.nan)
        tournaments["finish_percentage"] = np.round(tournaments["final_position"] / tournaments["total_players"]*100, 2)
        tournaments["ITM"] = tournaments["amount_won"] > 0
        return tournaments

    def load_raw_hand_histories(self):
        hand_histories = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/hand_histories.csv', index_col=0)
        hand_histories['hand_date'] = pd.to_datetime(hand_histories['hand_date'])
        return hand_histories

    def load_raw_player_hand_stats(self):
        """
        Load player hand stats from the database.
        """
        player_hand_stats = pd.concat(
            pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/player_hand_stats.csv', index_col=0, chunksize=10000))
        return player_hand_stats

    def load_raw_general_player_hand_stats(self):
        """
        Load general player hand stats from the database.
        """
        general_player_hand_stats = pd.concat(
            pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/general_player_hand_stats.csv', index_col=0, chunksize=10000))
        general_player_hand_stats = general_player_hand_stats.rename(
            columns={x: f"general_{x}" for x in general_player_hand_stats.columns})
        return general_player_hand_stats

    def load_raw_preflop_player_hand_stats(self):
        """
        Load preflop player hand stats from the database.
        """
        preflop_player_hand_stats = pd.concat(
            pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/preflop_player_hand_stats.csv', index_col=0, chunksize=10000))
        preflop_player_hand_stats = preflop_player_hand_stats.rename(
            columns={x: f"preflop_{x}" for x in preflop_player_hand_stats.columns})
        return preflop_player_hand_stats

    def load_raw_flop_player_hand_stats(self):
        """
        Load flop player hand stats from the database.
        """
        flop_player_hand_stats = pd.concat(
            pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/flop_player_hand_stats.csv', index_col=0, chunksize=10000))
        flop_player_hand_stats = flop_player_hand_stats.rename(
            columns={x: f"flop_{x}" for x in flop_player_hand_stats.columns})
        return flop_player_hand_stats

    def load_raw_turn_player_hand_stats(self):
        """
        Load turn player hand stats from the database.
        """
        turn_player_hand_stats = pd.concat(
            pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/turn_player_hand_stats.csv', index_col=0, chunksize=10000))
        turn_player_hand_stats = turn_player_hand_stats.rename(
            columns={x: f"turn_{x}" for x in turn_player_hand_stats.columns})
        return turn_player_hand_stats

    def load_raw_river_player_hand_stats(self):
        """
        Load river player hand stats from the database.
        """
        river_player_hand_stats = pd.concat(
            pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/river_player_hand_stats.csv', index_col=0, chunksize=10000))
        river_player_hand_stats = river_player_hand_stats.rename(
            columns={x: f"river_{x}" for x in river_player_hand_stats.columns})
        return river_player_hand_stats

    def load_player_hand_stats(self):
        raw_player_hand_stats = self.load_raw_player_hand_stats()
        raw_hand_histories = self.load_raw_hand_histories()
        raw_general_player_hand_stats = self.load_raw_general_player_hand_stats()
        raw_preflop_player_hand_stats = self.load_raw_preflop_player_hand_stats()
        raw_flop_player_hand_stats = self.load_raw_flop_player_hand_stats()
        raw_turn_player_hand_stats = self.load_raw_turn_player_hand_stats()
        raw_river_player_hand_stats = self.load_raw_river_player_hand_stats()
        player_hand_stats = raw_player_hand_stats\
            .merge(raw_hand_histories, how='left', left_on='hand_history', right_on='id', suffixes=('', '_hand'))\
            .merge(raw_general_player_hand_stats, how='left', left_on='general_stats', right_on='general_id',
                   suffixes=('', '_general'))\
            .merge(raw_preflop_player_hand_stats, how='left', left_on='preflop_stats', right_on='preflop_id')\
            .merge(raw_flop_player_hand_stats, how='left', left_on='flop_stats', right_on='flop_id')\
            .merge(raw_turn_player_hand_stats, how='left', left_on='turn_stats', right_on='turn_id')\
            .merge(raw_river_player_hand_stats, how='left', left_on='river_stats', right_on='river_id')\
            .drop(columns=['id', 'hand_history', 'id_hand', 'general_hand_history', 'general_stats', 'general_id',
                           'preflop_stats', 'preflop_id', 'flop_stats', 'flop_id', 'turn_stats', 'turn_id',
                           'river_stats', 'river_id'])
        return player_hand_stats

    def load_exploitable_hand_stats(self):
        general_player_hand_stats = self.load_raw_general_player_hand_stats()
        exploitable_filter = ~ general_player_hand_stats['general_position'].isna()
        general_player_hand_stats = general_player_hand_stats[exploitable_filter]