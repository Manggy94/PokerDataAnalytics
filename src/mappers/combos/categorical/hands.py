import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo

def hands_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    hands = source.hand.cat.categories
    matrix = pd.DataFrame(
        index=combos,
        columns=hands,
        data=0,
        dtype=int)
    for combo in Combo:
        x, y = f"{combo}", f"{combo.hand}"
        matrix.loc[x, y] = 1
    return matrix
