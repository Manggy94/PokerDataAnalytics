import os
import numpy as np
import pandas as pd


def map_int_dtype(s: pd.Series):
    size = s.unique().size
    if size < 256//2:
        dtype = np.int8
    elif size < 65536//2:
        dtype = np.int16
    elif size < 4294967296//2:
        dtype = np.int32
    else:
        dtype = np.int64
    return s.fillna(-1).astype(dtype).replace(-1, np.nan)
