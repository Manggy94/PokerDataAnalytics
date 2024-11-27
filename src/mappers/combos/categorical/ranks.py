import pandas as pd
from pkrcomponents.components.cards.combo import Combo
from itertools import combinations_with_replacement
from src.data.loader import DataLoader


def ranks_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    ranks = source.first_card_rank.cat.categories
    ranks_pairs = [f"{x[0]}{x[1]}" for x in combinations_with_replacement(ranks[::-1], 2)]
    matrix = pd.DataFrame(
        index=combos,
        columns=ranks_pairs,
        data=0,
        dtype=int)
    for combo in Combo:
        x = f"{combo}"
        y  = f"{combo.hand}"[:2]
        matrix.loc[x, y] = 1
    return matrix