import pandas as pd
import numpy as np
from src.data.loader import DataLoader


def combos_combos_matrix() -> pd.DataFrame:
    source = DataLoader().load_combos()
    combos = source.short_name.cat.categories
    matrix = pd.DataFrame(
        index=combos,
        columns=combos,
        data=np.identity(len(combos)),
        dtype=int)
    return matrix