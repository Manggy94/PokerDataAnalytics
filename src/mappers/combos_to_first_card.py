import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo


def combos_first_card_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    cards = source.first_card.cat.categories
    matrix = pd.DataFrame(
        index=combos,
        columns=cards,
        data=0,
        dtype=int)
    for combo in Combo:
        x, y = f"{combo}", f"{combo.first}"
        matrix.loc[x, y] = 1
    return matrix