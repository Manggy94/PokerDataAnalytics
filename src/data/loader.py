import pandas as pd
from config.settings import ANALYTICS_DATA_DIR
from src.pipelines.ref_tournaments import RefTournamentPipeline
from src.pipelines.tournaments import TournamentsPipeline
from src.transformers.combos.combos_cards_merger import CombosCardsMerger
from src.transformers.flops.flops_cards_merger import FlopsCardsMerger
from src.transformers.positions.columns_cleaner import PositionsColumnsCleaner



class DataLoader:

    def __init__(self):
        self.ANALYTICS_DATA_DIR = ANALYTICS_DATA_DIR

    def load_raw_cards(self):
        raw_cards = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/cards.csv', index_col=0)
        return raw_cards

    def load_raw_combos(self):
        raw_combos = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/combos.csv', index_col=0)
        return raw_combos

    def load_combos(self):
        raw_cards = self.load_raw_cards()
        raw_combos = self.load_raw_combos()
        merger = CombosCardsMerger(raw_cards)
        combos = merger.fit_transform(raw_combos)
        return combos

    def load_raw_hands(self):
        hands = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/hands.csv', index_col=0)
        return hands

    def load_raw_flops(self):
        flops = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/flops.csv', index_col=0)
        return flops

    def load_flops(self):
        raw_cards = self.load_raw_cards()
        raw_flops = self.load_raw_flops()
        merger = FlopsCardsMerger(raw_cards)
        flops = merger.fit_transform(raw_flops)
        return flops

    def load_raw_levels(self):
        levels = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/levels.csv', index_col=0)
        return levels

    def load_raw_positions(self):
        positions = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/positions.csv', index_col=0)
        return positions

    def load_positions(self):
        raw_positions = self.load_raw_positions()
        cleaner = PositionsColumnsCleaner()
        positions = cleaner.fit_transform(raw_positions)
        return positions

    def load_raw_tournaments(self):
        tournaments = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/tournaments.csv', index_col=0)
        return tournaments

    def load_raw_ref_tournaments(self):
        ref_tournaments = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/ref_tournaments.csv', index_col=0)
        return ref_tournaments

    def load_ref_tournaments(self):
        raw_ref_tournaments = self.load_raw_ref_tournaments()
        raw_buy_ins = self.load_raw_buy_ins()
        tour_types = self.load_raw_tour_types()
        tour_speeds = self.load_raw_tour_speeds()
        ref_tournaments_pipeline = RefTournamentPipeline(buy_ins=raw_buy_ins, tour_types=tour_types, tour_speeds=tour_speeds)
        tournaments = ref_tournaments_pipeline.fit_transform(raw_ref_tournaments)
        return tournaments

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
        raw_ref_tournaments = self.load_ref_tournaments()
        tournaments_pipeline = TournamentsPipeline(ref_tournaments=raw_ref_tournaments)
        tournaments = tournaments_pipeline.fit_transform(raw_tournaments)
        return tournaments

    def load_raw_hand_histories(self):
        raw_hand_histories = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/hand_histories.csv', index_col=0)
        return raw_hand_histories

    def load_hand_histories(self):
        raw_hand_histories = self.load_raw_hand_histories()
        hand_histories = raw_hand_histories
        hand_histories['hand_date'] = pd.to_datetime(hand_histories['hand_date'])
        return hand_histories

    def load_raw_player_hand_stats(self):
        """
        Load player hand stats from the database.
        """
        player_hand_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/player_hand_stats.csv', index_col=0)
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
        levels = self.load_raw_levels()
        combos = self.load_combos()
        cards = self.load_raw_cards()
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

    def load_raw_players(self):
        players = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/players.csv', index_col=0)
        return players

    def load_raw_player_stats(self):
        players = self.load_raw_players()
        stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/player_stats.csv', index_col=0)
        player_stats = players\
            .merge(stats, how='left', left_on='id', right_on='player', suffixes=('_player', ''))\
            .drop(columns=['id_player', 'player'])
        return player_stats

    def load_raw_general_player_stats(self):
        general_player_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/general_player_stats.csv', index_col=0)
        general_player_stats = general_player_stats.rename(columns={x: f"general_{x}" for x in general_player_stats.columns})
        return general_player_stats

    def load_raw_preflop_player_stats(self):
        preflop_player_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/preflop_player_stats.csv', index_col=0)
        preflop_player_stats = preflop_player_stats.rename(columns={x: f"preflop_{x}" for x in preflop_player_stats.columns})
        return preflop_player_stats

    def load_raw_flop_player_stats(self):
        flop_player_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/flop_player_stats.csv', index_col=0)
        flop_player_stats = flop_player_stats.rename(columns={x: f"flop_{x}" for x in flop_player_stats.columns})
        return flop_player_stats

    def load_raw_turn_player_stats(self):
        turn_player_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/turn_player_stats.csv', index_col=0)
        turn_player_stats = turn_player_stats.rename(columns={x: f"turn_{x}" for x in turn_player_stats.columns})
        return turn_player_stats

    def load_raw_river_player_stats(self):
        river_player_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/river_player_stats.csv', index_col=0)
        river_player_stats = river_player_stats.rename(columns={x: f"river_{x}" for x in river_player_stats.columns})
        return river_player_stats

    def load_player_stats(self):
        raw_player_stats = self.load_raw_player_stats()
        general_player_stats = self.load_raw_general_player_stats()
        preflop_player_stats = self.load_raw_preflop_player_stats()
        flop_player_stats = self.load_raw_flop_player_stats()
        turn_player_stats = self.load_raw_turn_player_stats()
        river_player_stats = self.load_raw_river_player_stats()
        player_stats = raw_player_stats\
            .merge(general_player_stats, how='left', left_on='general_stats', right_on='general_id', suffixes=('', '_general'))\
            .merge(preflop_player_stats, how='left', left_on='preflop_stats', right_on='preflop_id', suffixes=('', '_preflop'))\
            .merge(flop_player_stats, how='left', left_on='flop_stats', right_on='flop_id', suffixes=('', '_flop'))\
            .merge(turn_player_stats, how='left', left_on='turn_stats', right_on='turn_id', suffixes=('', '_turn'))\
            .merge(river_player_stats, how='left', left_on='river_stats', right_on='river_id', suffixes=('', '_river'))
        player_stats = player_stats.rename(columns={x: f"player_{x}" for x in player_stats.columns})
        return player_stats