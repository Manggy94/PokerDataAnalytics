import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo

def card_to_combos_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    cards = source.first_card.cat.categories
    matrix = pd.DataFrame(
        index=cards,
        columns=combos,
        data=0,
        dtype=int)
    for combo in Combo:
        x1 = combo.first.short_name
        x2 = combo.second.short_name
        y = f"{combo}"
        matrix.loc[x1, y] = 1
        matrix.loc[x2, y] = 1
    return matrix