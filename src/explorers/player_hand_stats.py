import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from functools import cached_property
from src.data.loader import DataLoader
from src.graphs import freq_comparison_bar_plot, comparison_bar_plot
from src.graphs.exploration import ExplorationGraphs


class PlayerHandStatsExplorer:
    def __init__(self, data: pd.DataFrame):
        self.data = data.drop(columns=["id"])
        self.graphs_explorer = ExplorationGraphs(data=self.data)
        target_cols = [col for col in self.data.columns if 'player_combo' in col]
        self.targets = self.data[target_cols]
        self.purged_targets = self.targets.dropna()
        self.combos = DataLoader().load_combos()

    def print_resume(self):
        print(f"This dataset has {self.data.shape[0]} rows and {self.data.shape[1]} columns.\n"
              f"It gathers information about the player's hand stats.\n"
              f"It has both numerical and categorical data types.\n"
              f"It also has a datetime column that enables time series analysis.")

    def display_num_data_info(self):
        self.graphs_explorer.plot_min_max_standard_deviation()
        self.graphs_explorer.plot_missing_numerical_data()

    def display_cat_data_info(self):
        self.graphs_explorer.plot_missing_categorical_data()

    def display_combos_distribution(self, n_combos=30, n_graphs=3):
        value_graph = self.purged_targets["player_combo"].value_counts(normalize=True).reset_index()
        for i in range(n_graphs):
            b1, b2 = i*n_combos, (i+1)*n_combos
            combos_fig = px.bar(
                value_graph.iloc[b1:b2],
                x="index", y="player_combo", title=f"Most observed Combos {b1+1}-{min(b2, 1326)}")
            combos_fig.add_shape(type='line', x0=-1, y0=1/1326, x1=n_combos, y1=1/1326, line=dict(color='red', width=2))
            combos_fig.show()

    def display_hands_distribution(self, n_graphs=1):
        value_graph = self.purged_targets["player_combo_hand"].value_counts(normalize=True).reset_index()
        n_hands = 169 // n_graphs + 1
        for i in range(n_graphs):
            b1, b2 = i*n_hands, (i+1)*n_hands
            hands_fig = px.bar(
                value_graph.iloc[b1:b2],
                x="index", y="player_combo_hand", title=f"Most observed Hands {b1+1}-{min(b2, 169)}")
            hands_fig.add_shape(type='line', x0=-1, y0=1/169, x1=n_hands, y1=1/169, line=dict(color='red', width=2))
            hands_fig.show()


    def display_frequency_comparison(self, col_name: str):
        col_name_txt = col_name.replace("_", " ").title()
        freq_comparison_bar_plot(
            s1=self.purged_targets[f"player_combo_{col_name}"],
            s2=self.combos[col_name],
            title=f"{col_name_txt} frequency comparison",
            x_title=col_name_txt,
            y_title="Frequency")

    def compare_frequencies(self):
        freq_cols = ["hand_shape", "hand_is_offsuit", "hand_is_suited", "hand_is_paired", "hand_is_broadway",
                         "hand_is_face", "hand_is_connector", "hand_is_one_gapper", "hand_is_two_gapper",
                         "hand_is_suited_connector", "first_card_rank", "second_card_rank", "first_card_is_broadway",
                         "first_card_is_face", "second_card_is_broadway", "second_card_is_face", "first_card_suit",
                         "second_card_suit", "hand_rank_difference"]
        for col_name in freq_cols:
            self.display_frequency_comparison(col_name)
            
    def display_targets_info(self):
        print("Targets Analyse...")
        print(self.targets.info())
        print("")
        self.display_combos_distribution()
        self.display_hands_distribution()
        self.compare_frequencies()


    def full_report(self):
        print("Starting the report")
        print(self.data.info())
        print("")
        self.print_resume()
        self.display_num_data_info()
        self.display_cat_data_info()
        self.display_targets_info()

        print("End of the report")





