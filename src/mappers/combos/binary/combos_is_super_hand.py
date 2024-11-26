import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo

def combos_is_super_hand_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    premium_hands = [
        "AA", "KK", "QQ", "JJ", "AKs",
        "TT", "AKo", "AQs", "99", "AJs",
        "KQs", "ATs", "AQo", "88", "KJs"
    ]
    matrix = pd.DataFrame(
        index=combos,
        columns=["is_super_hand"],
        data=0,
        dtype=int)
    for combo in Combo:
        x = f"{combo}"
        matrix.loc[x, "is_super_hand"] = int(x in premium_hands)
    return matrix