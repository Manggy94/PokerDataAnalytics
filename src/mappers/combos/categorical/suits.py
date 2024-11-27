import pandas as pd
from pkrcomponents.components.cards.combo import Combo
from itertools import combinations_with_replacement
from src.data.loader import DataLoader


def suits_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    suits = source.first_card_suit.cat.categories
    suit_pairs = [f"{x[0]}{x[1]}" for x in combinations_with_replacement(suits, 2)]
    matrix = pd.DataFrame(
        index=combos,
        columns=suit_pairs,
        data=0,
        dtype=int)
    for combo in Combo:
        x = f"{combo}"
        y  = ''.join(sorted(f"{combo.first.suit}{combo.second.suit}"))
        matrix.loc[x, y] = 1
    return matrix

