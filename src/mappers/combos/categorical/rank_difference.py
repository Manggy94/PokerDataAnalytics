import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo


def rank_difference_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    ranks = source.hand_rank_difference.sort_values().unique()
    matrix = pd.DataFrame(
        index=combos,
        columns=ranks,
        data=0,
        dtype=int)
    for combo in Combo:
        x, y = f"{combo}", combo.rank_difference
        matrix.loc[x, y] = 1
    return matrix