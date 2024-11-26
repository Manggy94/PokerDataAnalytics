import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo

def combos_is_premium_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    premium_hands = [
        "AA", "KK", "QQ", "JJ", "AKs",
        "TT", "AKo", "AQs", "99", "AJs"
    ]
    matrix = pd.DataFrame(
        index=combos,
        columns=["is_premium"],
        data=0,
        dtype=int)
    for combo in Combo:
        x = f"{combo}"
        matrix.loc[x, "is_premium"] = int(x in premium_hands)
    return matrix