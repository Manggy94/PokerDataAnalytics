import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo


def combos_is_paired_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    matrix = pd.DataFrame(
        index=combos,
        columns=["is_paired"],
        data=0,
        dtype=int)
    for combo in Combo:
        x = f"{combo}"
        matrix.loc[x, "is_paired"] = int(combo.is_pair)
    return matrix