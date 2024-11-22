import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo


def combos_second_rank_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    ranks = source.first_card_rank.cat.categories
    matrix = pd.DataFrame(
        index=combos,
        columns=ranks,
        data=0,
        dtype=int)
    for combo in Combo:
        x, y = f"{combo}", f"{combo.second.rank}"
        matrix.loc[x, y] = 1
    return matrix