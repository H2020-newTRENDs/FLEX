import os
from pathlib import Path
import pandas as pd


def read_source(file_name: str, sheet_name: str = None):
    source_folder = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"source/")
    if sheet_name is None:
        df = pd.read_excel(source_folder / Path(file_name + ".xlsx"))
    else:
        df = pd.read_excel(source_folder / Path(file_name + ".xlsx"), sheet_name=sheet_name)
    return df


def read_result(file_name: str, sheet_name: str = None):
    result_folder = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"result/")
    if sheet_name is None:
        df = pd.read_excel(result_folder / Path(file_name + ".xlsx"))
    else:
        df = pd.read_excel(result_folder / Path(file_name + ".xlsx"), sheet_name=sheet_name)
    return df


def save_result(df: pd.DataFrame, file_name: str):
    result_folder = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"result/")
    df.to_excel(result_folder / Path(file_name + ".xlsx"), index=False)

