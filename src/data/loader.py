import pandas as pd
from config.settings import ANALYTICS_DATA_DIR
from src.pipelines.cards import CardsPipeline
from src.pipelines.combos import CombosPipeline
from src.pipelines.flops import FlopsPipeline
from src.pipelines.hand_histories import HandHistoriesPipeline
from src.pipelines.hands import HandsPipeline
from src.pipelines.player_hand_stats.general import GeneralPlayerHandStatsPipeline
from src.pipelines.player_hand_stats.street import StreetPlayerHandStatsPipeline
from src.pipelines.ref_tournaments import RefTournamentPipeline
from src.pipelines.tournaments import TournamentsPipeline
from src.transformers.positions.columns_cleaner import PositionsColumnsCleaner
from src.pipelines.player_hand_stats import PlayerHandStatsPipeline


class DataLoader:

    def __init__(self):
        self.ANALYTICS_DATA_DIR = ANALYTICS_DATA_DIR

    def load_raw_suits(self):
        suits = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/suits.csv', index_col=0)
        return suits

    def load_raw_ranks(self):
        ranks = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/ranks.csv', index_col=0)
        return ranks

    def load_raw_shapes(self):
        shapes = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/shapes.csv', index_col=0)
        return shapes

    def load_raw_action_moves(self):
        action_moves = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/action_moves.csv', index_col=0)
        return action_moves

    def load_raw_actions_sequences(self):
        actions_sequences = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/actions_sequences.csv', index_col=0)
        return actions_sequences


    def load_raw_cards(self):
        raw_cards = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/cards.csv', index_col=0)
        return raw_cards

    def load_cards(self):
        raw_ranks = self.load_raw_ranks()
        raw_suits = self.load_raw_suits()
        raw_cards = self.load_raw_cards()
        cards_pipeline = CardsPipeline(ranks=raw_ranks, suits=raw_suits)
        cards = cards_pipeline.fit_transform(raw_cards)
        return cards

    def load_raw_combos(self):
        raw_combos = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/combos.csv', index_col=0)
        return raw_combos

    def load_combos(self):
        cards = self.load_cards()
        hands = self.load_hands()
        raw_combos = self.load_raw_combos()
        combos_pipeline = CombosPipeline(cards=cards, hands=hands)
        combos = combos_pipeline.fit_transform(raw_combos)
        return combos

    def load_raw_hands(self):
        hands = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/hands.csv', index_col=0)
        return hands

    def load_hands(self):
        raw_hands = self.load_raw_hands()
        ranks = self.load_raw_ranks()
        shapes = self.load_raw_shapes()
        hands_pipeline = HandsPipeline(ranks=ranks, shapes=shapes)
        hands = hands_pipeline.fit_transform(raw_hands)
        return hands

    def load_raw_flops(self):
        flops = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/flops.csv', index_col=0)
        return flops

    def load_flops(self):
        raw_flops = self.load_raw_flops()
        cards = self.load_cards()
        flops_pipeline = FlopsPipeline(cards=cards)
        flops = flops_pipeline.fit_transform(raw_flops)
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

    def load_raw_streets(self):
        streets = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/streets.csv', index_col=0)
        return streets

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
        cards = self.load_cards()
        combos = self.load_combos()
        flops = self.load_flops()
        raw_levels = self.load_raw_levels()
        hh_pipeline = HandHistoriesPipeline(
            levels=raw_levels, flops=flops, cards=cards, combos=combos)
        hand_histories = hh_pipeline.fit_transform(raw_hand_histories)
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
        general_player_hand_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/general_player_hand_stats.csv', index_col=0)
        return general_player_hand_stats

    def load_general_player_hand_stats(self):
        raw_general_player_hand_stats = self.load_raw_general_player_hand_stats()
        action_moves = self.load_raw_action_moves()
        combos = self.load_combos()
        positions = self.load_positions()
        streets = self.load_raw_streets()
        pipeline = GeneralPlayerHandStatsPipeline(
            combos=combos, positions=positions, action_moves=action_moves, streets=streets)
        general_player_hand_stats = pipeline.fit_transform(raw_general_player_hand_stats)
        return general_player_hand_stats

    def load_raw_preflop_player_hand_stats(self):
        """
        Load preflop player hand stats from the database.
        """
        preflop_player_hand_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/preflop_player_hand_stats.csv', index_col=0)
        return preflop_player_hand_stats

    def load_preflop_player_hand_stats(self):
        raw_preflop_player_hand_stats = self.load_raw_preflop_player_hand_stats()
        action_moves = self.load_raw_action_moves()
        sequences = self.load_raw_actions_sequences()
        pipeline = StreetPlayerHandStatsPipeline(
            action_moves=action_moves, sequences=sequences)
        preflop_player_hand_stats = pipeline.fit_transform(raw_preflop_player_hand_stats)
        return preflop_player_hand_stats

    def load_raw_flop_player_hand_stats(self):
        """
        Load flop player hand stats from the database.
        """
        flop_player_hand_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/flop_player_hand_stats.csv', index_col=0)
        return flop_player_hand_stats

    def load_flop_player_hand_stats(self):
        raw_flop_player_hand_stats = self.load_raw_flop_player_hand_stats()
        action_moves = self.load_raw_action_moves()
        sequences = self.load_raw_actions_sequences()
        pipeline = StreetPlayerHandStatsPipeline(
            action_moves=action_moves, sequences=sequences)
        flop_player_hand_stats = pipeline.fit_transform(raw_flop_player_hand_stats)
        return flop_player_hand_stats

    def load_raw_turn_player_hand_stats(self):
        """
        Load turn player hand stats from the database.
        """
        turn_player_hand_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/turn_player_hand_stats.csv', index_col=0)
        return turn_player_hand_stats

    def load_turn_player_hand_stats(self):
        raw_turn_player_hand_stats = self.load_raw_turn_player_hand_stats()
        action_moves = self.load_raw_action_moves()
        sequences = self.load_raw_actions_sequences()
        pipeline = StreetPlayerHandStatsPipeline(
            action_moves=action_moves, sequences=sequences)
        turn_player_hand_stats = pipeline.fit_transform(raw_turn_player_hand_stats)
        return turn_player_hand_stats

    def load_raw_river_player_hand_stats(self):
        """
        Load river player hand stats from the database.
        """
        river_player_hand_stats = pd.read_csv(f'{self.ANALYTICS_DATA_DIR}/river_player_hand_stats.csv', index_col=0)
        return river_player_hand_stats

    def load_river_player_hand_stats(self):
        raw_river_player_hand_stats = self.load_raw_river_player_hand_stats()
        action_moves = self.load_raw_action_moves()
        sequences = self.load_raw_actions_sequences()
        pipeline = StreetPlayerHandStatsPipeline(
            action_moves=action_moves, sequences=sequences)
        river_player_hand_stats = pipeline.fit_transform(raw_river_player_hand_stats)
        return river_player_hand_stats
    

    def load_player_hand_stats(self):
        raw_player_hand_stats = self.load_raw_player_hand_stats()
        hand_histories = self.load_hand_histories()
        general_player_hand_stats = self.load_general_player_hand_stats()
        preflop_player_hand_stats = self.load_preflop_player_hand_stats()
        flop_player_hand_stats = self.load_flop_player_hand_stats()
        turn_player_hand_stats = self.load_turn_player_hand_stats()
        river_player_hand_stats = self.load_river_player_hand_stats()
        pipeline = PlayerHandStatsPipeline(
            hand_histories=hand_histories,
            general_player_hand_stats=general_player_hand_stats,
            preflop_player_hand_stats=preflop_player_hand_stats,
            flop_player_hand_stats=flop_player_hand_stats,
            turn_player_hand_stats=turn_player_hand_stats,
            river_player_hand_stats=river_player_hand_stats

        )
        player_hand_stats = pipeline.fit_transform(raw_player_hand_stats)
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