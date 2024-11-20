import pandas as pd
from src.data.loader import DataLoader
from pkrcomponents.components.cards.combo import Combo


def combos_is_one_gapper_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    matrix = pd.DataFrame(
        index=combos,
        columns=["is_one_gapper"],
        data=0,
        dtype=int)
    for combo in Combo:
        x = f"{combo}"
        matrix.loc[x, "is_one_gapper"] = int(combo.is_one_gapper)
    return matrix