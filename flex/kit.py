from typing import List, Union
from pathlib import Path
import logging
import sqlalchemy
import numpy as np
import pandas as pd
from functools import wraps
import time


def get_logger(name, level=logging.DEBUG, file_name: Union[str, "Path"] = None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    if file_name:
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def performance_counter(func):
    @wraps(func)
    def wrapper(*args):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        exe_time = round(t_end - t_start, 3)
        logger.info(f"Timer: {func.__name__} - {exe_time}s.")
        return result

    return wrapper


def convert_datatype_py2sql(data_types: dict) -> dict:
    type_py2sql_dict: dict = {
        int: sqlalchemy.types.BigInteger,
        Union[int, None]: sqlalchemy.types.BigInteger,
        Union[List[int], None]: sqlalchemy.types.BigInteger,
        Union[np.ndarray, int, None]: sqlalchemy.types.BigInteger,
        str: sqlalchemy.types.Unicode,
        Union[str, None]: sqlalchemy.types.Unicode,
        Union[List[str], None]: sqlalchemy.types.Unicode,
        float: sqlalchemy.types.Float,
        Union[float, None]: sqlalchemy.types.Float,
        Union[List[float], None]: sqlalchemy.types.Float,
        Union[np.ndarray, None]: sqlalchemy.types.Float,
        Union[np.ndarray, float, None]: sqlalchemy.types.Float,
    }
    for key, value in data_types.items():
        data_types[key] = type_py2sql_dict[value]
    return data_types


def filter_df(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df_filtered = df.loc[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)]
    return df_filtered


def filter_df2s(df: pd.DataFrame, filter_dict: dict) -> pd.Series:
    df_filtered = filter_df(df, filter_dict)
    s = df_filtered.iloc[0]
    return s
