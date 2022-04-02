import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from flex.core.config import Config
from flex.core.db import create_db_conn

config = Config(db_name="pv_gis")
db = create_db_conn(config)


@dataclass
class Var:
    region = "region"
    year = "year"
    id_hour = "id_hour"
    pv_generation = "pv_generation"
    pv_generation_unit = "pv_generation_unit"
    pv_generation_unit_string = 'W/kW_peak'
    temperature = "temperature"
    temperature_unit = "temperature_unit"
    temperature_unit_string = "°C"
    south = "south"
    east = "east"
    west = "west"
    north = "north"
    radiation_prefix = "radiation_"
    radiation_south = "radiation_south"
    radiation_east = "radiation_east"
    radiation_west = "radiation_west"
    radiation_north = "radiation_north"
    radiation_unit = "radiation_unit"
    radiation_unit_string = "W"


def read_data_excel(file_name: str, sheet_name: str = None):
    folder = config.db_folder
    if sheet_name is None:
        df = pd.read_excel(folder / Path(file_name + ".xlsx"))
    else:
        df = pd.read_excel(folder / Path(file_name + ".xlsx"), sheet_name=sheet_name)
    return df


def save_data_excel(df: pd.DataFrame, file_name: str):
    folder = config.db_folder
    df.to_excel(folder / Path(file_name + ".xlsx"), index=False)


def read_data_sqlite(table_name: str):
    return db.read_dataframe(table_name)


def save_data_sqlite(df: pd.DataFrame, table_name: str, data_types: dict = None, if_exists: str = 'replace'):
    return db.write_dataframe(table_name, df, data_types=data_types, if_exists=if_exists)

