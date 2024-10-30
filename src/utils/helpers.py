import os
import pandas as pd
from config.settings import ANALYTICS_DATA_DIR


def map_id_dtype(df: pd.DataFrame):
    size = df["id"].unique().size
    if size < 256:
        return "uint8"
    if size < 65536:
        return "uint16"
    if size < 4294967296:
        return "uint32"
    return "uint64"
