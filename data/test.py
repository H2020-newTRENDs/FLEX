import os
from pathlib import Path

import pandas as pd


def save_data_excel(df: pd.DataFrame, file_name: str):
    df.to_excel(file_name + ".xlsx", index=False)


def read_data_json(file_name: str):
    path = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"{file_name}.json")
    return pd.read_json(path, orient="table")


df = read_data_json("SFH_building_data")
save_data_excel(df, "SFH_building_data")
