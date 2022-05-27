from typing import List, Union
import logging
import sqlalchemy
import numpy as np
from functools import wraps
import time


def get_logger(name, level=logging.INFO, file_name: str = None):
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    if file_name:
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger(__name__)


def performance_counter(func):
    @wraps(func)
    def wrapper(*args):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        logger.info(f"function >>{func.__name__}<< time for execution: {t_end - t_start}")
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


