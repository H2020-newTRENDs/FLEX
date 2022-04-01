import urllib.error
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from flex.data import read_source, read_result, save_result


class PVGIS:

    def __init__(self, start_year: int = 2019, end_year: int = 2019):
        self.id_hour = np.arange(1, 8761)
        self.start_year = start_year
        self.end_year = end_year
        self.pv_calculation: int = 1
        self.peak_power: float = 1
        self.pv_loss: int = 14
        self.pv_tech: str = "crystSi"
        self.tracking_type: int = 0
        self.angle: int = 90
        self.optimal_inclination: int = 1
        self.optimal_angle: int = 1

    """
    pv_calculation: No = 0; Yes = 1
    peak_power: size of PV (in kW_peak)
    pv_loss: system losses in %
    pv_tech: "crystSi", "CIS", "CdTe" and "Unknown".
    tracking_type: type of sun-tracking used,
                    - fixed = 0
                    - single horizontal axis aligned north-south = 1
                    - two-axis tracking = 2
                    - vertical axis tracking = 3
                    - single horizontal axis aligned east-west = 4
                    - single inclined axis aligned north-south = 5
    angle: inclination angle from horizontal plane, which is set to 90° because we are looking at a vertical plane.
    optimal_inclination: Yes = 1, meaning to calculate the optimum inclination angle.
                         All other values (or no value) mean "no". Not relevant for 2-axis tracking.
    optimal_angle: Yes = 1, meaning to calculate the optimum inclination AND orientation angles.
                   All other values (or no value) mean "no". Not relevant for tracking planes.
    """

    @staticmethod
    def scalar2array(value):
        return [value for _ in range(0, 8760)]

    @staticmethod
    def get_url_region_geo_center(nuts_level: int):
        if nuts_level in [0, 1, 2, 3]:
            return f'https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/' \
                   f'NUTS_RG_60M_2021_3035_LEVL_{nuts_level}.geojson'
        else:
            raise Exception(f'Wrong input for NUTS level.')

    def get_geo_center(self, region: str):
        nuts_level = self.get_nuts_level(region)
        nuts = gpd.read_file(self.get_url_region_geo_center(nuts_level))
        transformer = Transformer.from_crs(CRS("EPSG:3035"), CRS("EPSG:4326"))
        point = nuts[nuts.NUTS_ID == region].centroid.values[0]
        lat, lon = transformer.transform(point.y, point.x)
        return lat, lon

    @staticmethod
    def get_nuts_level(region: str):
        return int(len(region) - 2)

    def get_pv_generation(self, region: str) -> np.array:
        pv_generation_dict = {}
        self.pv_calculation = 1
        self.optimal_inclination = 1
        self.optimal_angle = 1
        lat, lon = self.get_geo_center(region)
        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&" \
              f"startyear={self.start_year}&" \
              f"endyear={self.end_year}&" \
              f"pvcalculation={self.pv_calculation}&" \
              f"peakpower={self.peak_power}&" \
              f"loss={self.pv_loss}&" \
              f"pvtechchoice={self.pv_tech}&" \
              f"components={1}&" \
              f"trackingtype={self.tracking_type}&" \
              f"optimalinclination={self.optimal_inclination}&" \
              f"optimalangles={self.optimal_angle}"
        try:
            # Read the csv from api and use 20 columns to receive the data, because depending on the parameters,
            # the number of columns could vary. Empty columns are dropped afterwards:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            df = df.dropna().reset_index(drop=True)
            # set header to first row
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            df = df.reset_index(drop=True)
            pv_generation_dict["pv_generation"] = pd.to_numeric(df["P"]).to_numpy()  # unit: W
            pv_generation_dict["pv_generation_unit"] = self.scalar2array('W/kW_peak')
            return pv_generation_dict
        except urllib.error.HTTPError:
            print(f"pv_generation data is not available for region {region}.")

    def get_temperature_and_solar_radiation(self, region, aspect) -> pd.DataFrame:

        self.pv_calculation = 0
        self.optimal_inclination = 0
        self.optimal_angle = 0
        lat, lon = self.get_geo_center(region)
        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&" \
              f"startyear={self.start_year}&" \
              f"endyear={self.end_year}&" \
              f"pvcalculation={self.pv_calculation}&" \
              f"peakpower={self.peak_power}&" \
              f"loss={self.pv_loss}&" \
              f"pvtechchoice={self.pv_tech}&" \
              f"components={1}&" \
              f"trackingtype={self.tracking_type}&" \
              f"optimalinclination={self.optimal_inclination}&" \
              f"optimalangles={self.optimal_angle}&" \
              f"angle={self.angle}&" \
              f"aspect={aspect}"

        # Read the csv from api and use 20 columns to receive the data, because depending on the parameters,
        # the number of columns could vary. Empty columns are dropped afterwards:
        try:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            df = df.dropna().reset_index(drop=True)
            # set header to first row
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            return df
        except urllib.error.HTTPError:
            pass

    def get_temperature(self, region: str) -> dict:
        temperature_dict = {}
        try:
            df = self.get_temperature_and_solar_radiation(region, 0)
            temperature_dict["temperature"] = pd.to_numeric(df["T2m"].reset_index(drop=True)).values
            temperature_dict["temperature_unit"] = self.scalar2array("°C")
            return temperature_dict
        except Exception as e:
            print(f"Temperature data is not available for region {region}.")

    def get_radiation(self, region) -> dict:
        radiation_dict = {}
        celestial_direction_aspect = {
            "south": 0,
            "east": -90,
            "west": 90,
            "north": -180
        }
        try:
            for direction, aspect in celestial_direction_aspect.items():
                df = self.get_temperature_and_solar_radiation(region, aspect)
                radiation = pd.to_numeric(df["Gb(i)"]) + pd.to_numeric(df["Gd(i)"])
                radiation_dict['radiation_' + direction] = radiation.reset_index(drop=True).to_numpy()
            radiation_dict["radiation_unit"] = self.scalar2array('W')
            return radiation_dict
        except Exception as e:
            print(f"Radiation data is not available for region {region}.")

    def get_pv_gis_data(self, region) -> pd.DataFrame:
        result_dict = {
            "region": self.scalar2array(region),
            "year": self.start_year,
            "id_hour": self.id_hour,
        }
        pv_generation_dict = self.get_pv_generation(region)
        temperature_dict = self.get_temperature(region)
        radiation_dict = self.get_radiation(region)
        try:
            assert pv_generation_dict["pv_generation"].sum() != 0
            assert temperature_dict["temperature"].sum() != 0
            assert radiation_dict["radiation_south"].sum() != 0
            assert radiation_dict["radiation_east"].sum() != 0
            assert radiation_dict["radiation_west"].sum() != 0
            assert radiation_dict["radiation_north"].sum() != 0
            result_dict.update(pv_generation_dict)
            result_dict.update(temperature_dict)
            result_dict.update(radiation_dict)
            result_df = pd.DataFrame.from_dict(result_dict)
            return result_df
        except Exception as e:
            print(f"Part of the data is not right for {region}.")

    @staticmethod
    def gen_region_area_dict(country: str):
        df = read_source('NUTS_hotmaps', sheet_name="hotmaps")
        df_filter = df.loc[(df["nuts0"] == country) & (df["floor_area"] > 0)]
        region_area_dict = {}
        for index, row in df_filter.iterrows():
            region_area_dict[row["nuts3"]] = row["floor_area"]
        return region_area_dict


def download_pv_gis(countries: List[str]):
    pv_gis = PVGIS()
    nuts = read_source("NUTS2021")
    for country in countries:
        country_pv_gis_list = []
        nuts3 = nuts.loc[nuts["nuts0"] == country]["nuts3"].to_list()
        for region in nuts3:
            print(f'Downloading: {country} - {region}.')
            try:
                region_df = pv_gis.get_pv_gis_data(region)
                country_pv_gis_list.append(region_df)
            except Exception as e:
                pass
        country_pv_gis_df = pd.concat(country_pv_gis_list)
        save_result(country_pv_gis_df, "pv_gis_" + country + "_nuts3")


def process_pv_gis(countries: List[str]):
    pv_gis = PVGIS()
    country_no_pv_gis_data = []
    country_no_floor_area_data = []
    countries_pv_gis_df_list = []
    for country in countries:
        print(f'Processing: {country}.')
        try:
            country_table = read_result(country + '_nuts3')
            region_area_dict = pv_gis.gen_region_area_dict(country)
            country_result_dict = {
                "region": pv_gis.scalar2array(country),
                "year": pv_gis.start_year,
                "id_hour": pv_gis.id_hour
            }

            country_pv_generation = np.zeros(8760, )
            country_temperature = np.zeros(8760, )
            country_radiation_south = np.zeros(8760, )
            country_radiation_north = np.zeros(8760, )
            country_radiation_east = np.zeros(8760, )
            country_radiation_west = np.zeros(8760, )
            sum = 0
            if len(region_area_dict) > 0:
                for region, area in region_area_dict.items():
                    region_table = country_table.loc[country_table["region"] == region]
                    if len(region_table) > 0:
                        country_pv_generation += region_table["pv_generation"].to_numpy() * area
                        country_temperature += region_table["temperature"].to_numpy() * area
                        country_radiation_south += region_table["radiation_south"].to_numpy() * area
                        country_radiation_east += region_table["radiation_east"].to_numpy() * area
                        country_radiation_west += region_table["radiation_west"].to_numpy() * area
                        country_radiation_north += region_table["radiation_north"].to_numpy() * area
                        sum += area
                    else:
                        pass
            else:
                country_no_floor_area_data.append(country)
                sum += 1
                for count, region in enumerate(country_table["region"].unique()):
                    region_table = country_table.loc[country_table["region"] == region]
                    country_pv_generation += region_table["pv_generation"].to_numpy()
                    country_temperature += region_table["temperature"].to_numpy()
                    country_radiation_south += region_table["radiation_south"].to_numpy()
                    country_radiation_east += region_table["radiation_east"].to_numpy()
                    country_radiation_west += region_table["radiation_west"].to_numpy()
                    country_radiation_north += region_table["radiation_north"].to_numpy()
                    sum += count

            country_result_dict["pv_generation"] = country_pv_generation / sum
            country_result_dict["pv_generation_unit"] = pv_gis.scalar2array('W/kW_peak')
            country_result_dict["temperature"] = country_temperature / sum
            country_result_dict["temperature_unit"] = pv_gis.scalar2array("°C")
            country_result_dict["radiation_south"] = country_radiation_south / sum
            country_result_dict["radiation_east"] = country_radiation_east / sum
            country_result_dict["radiation_west"] = country_radiation_west / sum
            country_result_dict["radiation_north"] = country_radiation_north / sum
            country_result_dict["radiation_unit"] = pv_gis.scalar2array("W")
            countries_pv_gis_df_list.append(pd.DataFrame(country_result_dict))

        except FileNotFoundError as e:
            country_no_pv_gis_data.append(country)
    countries_pv_gis_df = pd.concat(countries_pv_gis_df_list)
    save_result(countries_pv_gis_df, "pv_gis")
    print(f'country_no_pv_gis_data: {country_no_pv_gis_data}')
    print(f'country_no_floor_area_data: {country_no_floor_area_data}')


if __name__ == "__main__":
    country_list = [
        'BE', 'BG', 'CZ',
        # 'DK',
        'DE', 'EE', 'IE', 'EL', 'ES', 'FR',
        'HR', 'IT', 'CY',
        # 'LV',
        'LT', 'LU', 'HU', 'MT', 'NL', 'AT',
        'PL', 'PT', 'RO', 'SI', 'SK', 'FI', 'SE', 'UK', 'IS', 'LI',
        'NO', 'CH', 'ME', 'MK', 'AL', 'RS', 'TR'
    ]
    download_pv_gis(country_list)
    process_pv_gis(country_list)
