import pandas as pd
import numpy as np
from pathlib import Path

from utils.db import create_db_conn
from projects.main import get_config


COUNTRY_LIST = [
    "AUT",  
    "BEL", 
    # "POL",
    # "CYP", 
    # "PRT",
    # "DNK",
    # "FRA",
    # "CZE",  
    # "DEU", 
    # "HRV",
    # "HUN",
    # "ITA",  
    # "LUX",
    # "NLD",
    # "SVK",
    # "SVN",
    # 'IRL',
    # 'ESP',  
    # 'SWE',
    # 'GRC',
    # 'LVA',  
    # 'LTU',
    # 'MLT',
    # 'ROU',
    # 'BGR',  
    # 'FIN',
    # 'EST',
]

YEARS = [2020, 2030, 2040, 2050]

for year in YEARS:
    for country in COUNTRY_LIST:
        project_name = f"{country}_{year}"
        config = get_config(project_name=project_name)
        invert_file = config.input / f"INVERT_{country}_{year}.csv"
        df = pd.read_csv(invert_file, sep=",")

        # round the number of buildings
        df[["number_buildings_heat_pump_air", "number_buildings_heat_pump_ground"]] = df[["number_buildings_heat_pump_air", "number_buildings_heat_pump_ground"]].astype(float).round()
        # we are ignoring PV installations in this analysis, drop the PV columns:
        pv_columns = [c for c in df.columns if "PV_number" in c]
        df.drop(columns=pv_columns, inplace=True)
        # filter out buildings that have no hp installations
        df_hp = df.loc[(df["number_buildings_heat_pump_air"] != 0) &  (df["number_buildings_heat_pump_ground"] != 0), :].copy()

        # there are a lot of building archetypes with less than 10 heat pumps, more than half of of all:
        df_low = df_hp[(df_hp["number_buildings_heat_pump_air"]<=10) & (df_hp["number_buildings_heat_pump_ground"]<=10)]

        for (name), group in df_low.groupby(["name"]):
            group["name"].unique()
            len(group)

        print(f"exptected number of scenarios for {country} {year}: {len(df_hp)*2*2*2}")  # Heating Tank, DHW Tank, Battery


