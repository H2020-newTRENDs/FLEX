import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
from entsoe import EntsoePandasClient


class EnergyPrice:

    def __init__(self, grid_fee: float = 0.02):
        self.api_key = 'c06ee579-f827-486d-bc1f-8fa0d7ccd3da'
        self.grid_fee = grid_fee  # unit: ct/Wh
        self.energy_carrier_price = {  # unit: ct/Wh
            "feed_in_price": 0.00767,
            "district_heating": 0.02,
        }

    def get_electricity_price(self, country: str, year: int) -> pd.DataFrame:
        config = self.gen_config(country, year)
        country_code = config["country"]
        start = pd.Timestamp(config["start_time"], tz='CET')
        end = pd.Timestamp(config["end_time"], tz='CET')
        client = EntsoePandasClient(api_key=self.api_key)
        day_ahead_price: pd.Series = client.query_day_ahead_prices(country_code, start=start, end=end)  # unit: €/MWh
        electricity_price = day_ahead_price.values / 10_000 + self.grid_fee / 1_000  # unit: ct/Wh
        return electricity_price

    @staticmethod
    def country_exception_handler(country: str):
        exception = {"DE": "DE_LU"}
        if country in exception.keys():
            country = exception[country]
        return country

    def gen_config(self, country: str, year: int):
        config = {
            "country": self.country_exception_handler(country),
            "start_time": f"{year}0101",
            "end_time": f"{year + 1}0101",
        }
        return config

    @staticmethod
    def scalar2array(value):
        return [value for _ in range(0, 8760)]

    def gen_energy_price_df(self, country: str, year: int):
        electricity_price = self.get_electricity_price(country, year)
        electricity_price_mean = electricity_price.mean()
        energy_price = {
            'region': self.scalar2array(country),
            'year': self.scalar2array(2019),
            'id_hour': [i for i in range(1, 8761)],
            'unit': self.scalar2array('cent/Wh'),
            'electricity_var': electricity_price,
            'electricity_fix': self.scalar2array(electricity_price_mean),
        }
        for key, value in self.energy_carrier_price.items():
            energy_price[key] = self.scalar2array(value)
        energy_price_df = pd.DataFrame.from_dict(energy_price)
        return energy_price_df


def read_excel2df(file_name: str):
    file_path = Path(os.path.abspath(__file__)).parent.resolve() / Path(file_name)
    df = pd.read_excel(file_path)
    return df


def download_electricity_price(countries: List[str], year: int = 2019):
    ent = EnergyPrice()
    electricity_price_df_list: List[pd.DataFrame] = []
    missing_country = []
    for country in countries:
        print(f'Downloading price for {country}.')
        try:
            electricity_price_df_list.append(ent.gen_energy_price_df(country, year))
        except Exception as e:
            missing_country.append(country)
    full_electricity_price_df = pd.concat(electricity_price_df_list)
    full_electricity_price_df.to_excel(r'electricity_price_df.xlsx', index=False)
    print(f'missing country: {missing_country}')


if __name__ == "__main__":
    # country_list = read_excel2df("NUTS2021.xlsx")["nuts0"].unique()
    country_list = ['BE', 'AT']
    download_electricity_price(country_list)






