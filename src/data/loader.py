import pandas as pd
import numpy as np
from config.settings import ANALYTICS_DATA_DIR


class DataLoader:

    def __init__(self):
        self.ANALYTICS_DATA_DIR = ANALYTICS_DATA_DIR

    def load_cards(self):
        cards = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/cards.csv', index_col=0)
        return cards

    def load_combos(self):
        cards = self.load_cards()
        hands = self.load_hands()
        combos = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/combos.csv', index_col=0)
        # Merge combos and cards
        combos = combos\
            .merge(cards, how='left', left_on='first_card', right_on='id', suffixes=('', '_card1'))\
            .merge(cards, how='left', left_on='second_card', right_on='id', suffixes=('', '_card2'))
        # Drop columns
        columns_to_drop = ([c for c in combos.columns
                           if ("id_" in c or "symbol" in c)] +
                           ["name", "name_card2"] + ["first_card", "second_card"])

        combos = combos.drop(columns=columns_to_drop)
        # Rename columns on card1 to match the naming convention
        correction_dict = {
            "is_broadway": "is_broadway_card1",
            "is_face": "is_face_card1",
            "suit": "suit_card1",
            "rank": "rank_card1",
        }
        combos = combos.rename(columns=correction_dict)
        # Add prefix to columns
        renaming_dict = {c: f"combo_{c}" for c in combos.columns}
        combos = combos.rename(columns=renaming_dict)
        return combos

    def load_hands(self):
        hands = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/hands.csv', index_col=0)
        return hands

    def load_flops(self):
        cards = self.load_cards()
        flops = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/flops.csv', index_col=0)
        # Merge flops and cards
        flops = flops\
            .merge(cards, how='left', left_on='first_card', right_on='id', suffixes=('', '_card1'))\
            .merge(cards, how='left', left_on='second_card', right_on='id', suffixes=('', '_card2'))\
            .merge(cards, how='left', left_on='third_card', right_on='id', suffixes=('', '_card3'))
        columns_to_drop = ([c for c in flops.columns
                           if ("id_" in c or "symbol" in c)] +
                           ["name", "name_card2", "name_card3"] + ["first_card", "second_card", "third_card"])
        # Drop columns
        flops = flops.drop(columns=columns_to_drop)
        # Rename columns on card1 to match the naming convention
        correction_dict = {
            "is_broadway": "is_broadway_card1",
            "is_face": "is_face_card1",
            "suit": "suit_card1",
            "rank": "rank_card1",
        }
        flops = flops.rename(columns=correction_dict)
        # Add prefix to columns
        renaming_dict = {c: f"flop_{c}" for c in flops.columns}
        flops = flops.rename(columns=renaming_dict)
        return flops

    def load_levels(self):
        levels = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/levels.csv', index_col=0)
        return levels

    def load_positions(self):
        positions = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/positions.csv', index_col=0)
        positions = positions.drop(columns=["short_name", "symbol", "preflop_order", "postflop_order"])\
            .rename(columns={c: f"position_{c}" for c in positions.columns})
        return positions

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
        tournaments["prize_pool_percentage_won"] = tournaments["total_won"] / tournaments["prize_pool"]*100
        tournaments["freeze_out_percentage_won"] = tournaments["amount_won"] / tournaments["prize_pool"]*100
        tournaments["profit"] = tournaments["total_won"] - tournaments["total_investment"]
        tournaments["freeze_out_profit"] = tournaments["amount_won"] - tournaments["classic_investment"]
        tournaments["bounty_profit"] = tournaments["bounty_won"] - tournaments["bounty_investment"]
        tournaments['roi'] = tournaments["profit"] / tournaments['total_investment']
        tournaments["freeze_out_roi"] = tournaments["freeze_out_profit"] / tournaments["classic_investment"]
        tournaments["bounty_roi"] = tournaments["bounty_profit"] / tournaments["bounty_investment"]
        tournaments['roi'] = tournaments['roi'].replace([np.inf, -np.inf], np.nan)
        tournaments["freeze_out_roi"] = tournaments["freeze_out_roi"].replace([np.inf, -np.inf], np.nan)
        tournaments["bounty_roi"] = tournaments["bounty_roi"].replace([np.inf, -np.inf], np.nan)
        tournaments["finish_percentage"] = tournaments["final_position"] / tournaments["total_players"]*100
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
        combos = self.load_combos()
        positions = self.load_positions()
        general_player_hand_stats = pd.concat(
            pd.read_csv(
                f'{self.ANALYTICS_DATA_DIR}/general_player_hand_stats.csv', index_col=0, chunksize=10000))
        # Drop rows with missing position
        general_player_hand_stats = general_player_hand_stats.dropna(subset=["position"])
        #Merge with combos
        general_player_hand_stats = general_player_hand_stats\
            .merge(combos, how='left', left_on='combo', right_on='combo_id', suffixes=('', '_combo'))\
            .rename(columns={x: f"player_{x}" for x in combos.columns})
        # Merge with positions
        general_player_hand_stats = general_player_hand_stats\
            .merge(positions, how='left', left_on='position', right_on='position_id', suffixes=('', '_position'))\
            .drop(columns=["position_id", "position"])\
            .rename(columns={x: f"player_{x}" for x in positions.columns})
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
        levels = self.load_levels()
        combos = self.load_combos()
        cards = self.load_cards()
        flops = self.load_flops()
        raw_player_hand_stats = self.load_raw_player_hand_stats()
        raw_hand_histories = self.load_raw_hand_histories()
        raw_general_player_hand_stats = self.load_raw_general_player_hand_stats()
        raw_preflop_player_hand_stats = self.load_raw_preflop_player_hand_stats()
        raw_flop_player_hand_stats = self.load_raw_flop_player_hand_stats()
        raw_turn_player_hand_stats = self.load_raw_turn_player_hand_stats()
        raw_river_player_hand_stats = self.load_raw_river_player_hand_stats()
        # Merge player_hand_stats and hand_histories
        player_hand_stats = raw_player_hand_stats\
            .merge(raw_hand_histories, how='left', left_on='hand_history', right_on='id', suffixes=('', '_hand'))\
            .drop(columns=['id_hand'])
        # Merge player_hand_stats and level
        player_hand_stats = player_hand_stats\
            .merge(levels, how='left', left_on='level', right_on='id', suffixes=('', '_level'))\
        # Merge player_hand_stats and flop
        player_hand_stats = player_hand_stats\
            .merge(flops, how='left', left_on='flop', right_on='flop_id', suffixes=('', '_flop'))
        # Merge player_hand_stats and turn
        player_hand_stats = player_hand_stats\
            .merge(cards, how='left', left_on='turn', right_on='id', suffixes=('', '_turn'))\
            .rename(columns={x: f"turn_{x}" for x in cards.columns}).drop(columns=["turn_name", "turn_symbol"])
        # Merge player_hand_stats and river
        player_hand_stats = player_hand_stats\
            .merge(cards, how='left', left_on='river', right_on='id', suffixes=('', '_river'))\
            .rename(columns={x: f"river_{x}" for x in cards.columns}).drop(columns=["river_name", "river_symbol"])
        # Merge player_hand_stats and hero_combo
        player_hand_stats = player_hand_stats\
            .merge(combos, how='left', left_on='hero_combo', right_on='combo_id', suffixes=('', '_hero_combo'))\
            .rename(columns={x: f"hero_{x}" for x in combos.columns})
        # Drop some useless columns
        columns_to_drop = ["flop_id", "turn_id", "river_id", "hero_combo", "hand_history", "level", "id_level",
                           "hero_combo"]
        player_hand_stats = player_hand_stats.drop(columns=columns_to_drop)
        # Merge player_hand_stats and general_player_hand_stats
        player_hand_stats = player_hand_stats\
            .merge(raw_general_player_hand_stats, how='right', left_on='general_stats', right_on='general_id',
                     suffixes=('', '_general'))\
            .drop(columns=['general_player', 'general_hand_history', 'general_stats', 'general_id'])
        # Merge player_hand_stats and preflop, flop, turn, river player hand stats
        player_hand_stats = player_hand_stats\
            .merge(raw_preflop_player_hand_stats, how='left', left_on='preflop_stats', right_on='preflop_id')\
            .merge(raw_flop_player_hand_stats, how='left', left_on='flop_stats', right_on='flop_id')\
            .merge(raw_turn_player_hand_stats, how='left', left_on='turn_stats', right_on='turn_id')\
            .merge(raw_river_player_hand_stats, how='left', left_on='river_stats', right_on='river_id')\
            .drop(columns=['preflop_stats', 'preflop_id', 'flop_stats', 'flop_id', 'turn_stats', 'turn_id',
                           'river_stats', 'river_id'])
        return player_hand_stats

    def load_exploitable_hand_stats(self):
        general_player_hand_stats = self.load_player_hand_stats()
        exploitable_filter =~ general_player_hand_stats["general_player_combo_short_name"].isna()
        general_player_hand_stats = general_player_hand_stats[exploitable_filter]
        return general_player_hand_stats