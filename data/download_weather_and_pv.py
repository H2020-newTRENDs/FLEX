import os
import urllib.error
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pyproj import CRS, Transformer

import data.input_data_structure as input_data_structure
from basic.db import DB


class PVGIS:

    def __init__(self):
        self.id_hour = np.arange(1, 8761)

    @staticmethod
    def get_nuts_center(
            region_id,
            url_nuts0_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_0.geojson",
            url_nuts1_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_1.geojson",
            url_nuts2_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_2.geojson",
            url_nuts3_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_3.geojson"
    ):

        nuts = gpd.read_file(url_nuts3_poly)
        transformer = Transformer.from_crs(CRS("EPSG:3035"), CRS("EPSG:4326"))
        point = nuts[nuts.NUTS_ID == region_id].centroid.values[0]
        return transformer.transform(point.y, point.x)  # returns lat, lon

    def get_PV_generation(self, lat, lon, startyear, endyear, peakpower, nuts_id) -> np.array:
        # % JRC data
        # possible years are 2005 to 2017
        pvCalculation = 1  # 0 for no and 1 for yes
        peakPower = peakpower  # kWp
        pvLoss = 14  # system losses in %
        pvTechChoice = "crystSi"  # Choices are: "crystSi", "CIS", "CdTe" and "Unknown".
        trackingtype = 0  # Type of suntracking used, 0=fixed, 1=single horizontal axis aligned north-south,
        # 2=two-axis tracking, 3=vertical axis tracking, 4=single horizontal axis aligned east-west,
        # 5=single inclined axis aligned north-south.
        optimalInclination = 1  # Calculate the optimum inclination angle. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for 2-axis tracking.
        optimalAngles = 1  # Calculate the optimum inclination AND orientation angles. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for tracking planes.

        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&" \
              f"lon={lon}&" \
              f"startyear={startyear}&" \
              f"endyear={endyear}&" \
              f"pvcalculation={pvCalculation}&" \
              f"peakpower={peakPower}&" \
              f"loss={pvLoss}&" \
              f"pvtechchoice={pvTechChoice}&" \
              f"components={1}&" \
              f"trackingtype={trackingtype}&" \
              f"optimalinclination={optimalInclination}&" \
              f"optimalangles={optimalAngles}"

        try:
            # read the csv from api and set column names to list of 20 because depending on input parameters the number
            # of rows will vary. This way all parameters are included for sure, empty rows are dropped afterwards:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            # drop rows with nan:
            df = df.dropna().reset_index(drop=True)
            # set header to first column
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            df = df.reset_index(drop=True)
            PV_Profile = pd.to_numeric(df["P"]).to_numpy()  # W
            return PV_Profile
        except urllib.error.HTTPError:  # Error when nuts center is somewhere where there is no PVGIS data
            PV_Profile = None
            print("PV Data is not available for this location {}".format(nuts_id))
            return PV_Profile

    def get_temperature_and_solar_radiation(self, lat, lon, aspect, startyear, endyear, nuts_id) -> pd.DataFrame:
        # % JRC data
        # possible years are 2005 to 2017
        pvCalculation = 0  # 0 for no and 1 for yes
        peakPower = 1  # kWp
        pvLoss = 14  # system losses in %
        pvTechChoice = "crystSi"  # Choices are: "crystSi", "CIS", "CdTe" and "Unknown".
        trackingtype = 0  # Type of suntracking used, 0=fixed, 1=single horizontal axis aligned north-south,
        # 2=two-axis tracking, 3=vertical axis tracking, 4=single horizontal axis aligned east-west,
        # 5=single inclined axis aligned north-south.
        # angle is set to 90° because we are looking at a vertical plane
        angle = 90  # Inclination angle from horizontal plane
        optimalInclination = 0  # Calculate the optimum inclination angle. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for 2-axis tracking.
        optimalAngles = 0  # Calculate the optimum inclination AND orientation angles. Value of 1 for "yes".
        # All other values (or no value) mean "no". Not relevant for tracking planes.

        req = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&" \
              f"lon={lon}&" \
              f"startyear={startyear}&" \
              f"endyear={endyear}&" \
              f"pvcalculation={pvCalculation}&" \
              f"peakpower={peakPower}&" \
              f"loss={pvLoss}&" \
              f"pvtechchoice={pvTechChoice}&" \
              f"components={1}&" \
              f"trackingtype={trackingtype}&" \
              f"optimalinclination={optimalInclination}&" \
              f"optimalangles={optimalAngles}&" \
              f"angle={angle}&" \
              f"aspect={aspect}"

        # read the csv from api and set column names to list of 20 because depending on input parameters the number
        # of rows will vary. This way all parameters are included for sure, empty rows are dropped afterwards:
        nuts_not_working = []
        try:
            df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
            # drop rows with nan:
            df = df.dropna().reset_index(drop=True)
            # set header to first column
            header = df.iloc[0]
            df = df.iloc[1:, :]
            df.columns = header
            return df
        except urllib.error.HTTPError:  # Error when nuts center is somewhere where there is no PVGIS data
            nuts_not_working.append(nuts_id)
            print("{} does not have a valid location for PV-GIS".format(nuts_id))
            print(nuts_not_working)
            return None

    def get_PVGIS_data(self,
                       country: str,
                       nuts_id_list: list,
                       start_year: int,
                       end_year: int,
                       PV_sizes: list) -> None:
        """
        This function gets the solar radiation for every cilestial direction (North, East, West, South) on a vertical
        plane. This radiation will later be multiplied with the respective window areas to calculate the radiation
        gains. The function takes the latitude and longitude as input and the start and endyear. Possible years are
        2005 until 2017.

                    Explanation of header of returned dataframe:
                #  P: PV system power (W)
                #  Gb(i): Beam (direct) irradiance on the inclined plane (plane of the array) (W/m2)
                #  Gd(i): Diffuse irradiance on the inclined plane (plane of the array) (W/m2)
                #  Gr(i): Reflected irradiance on the inclined plane (plane of the array) (W/m2)
                #  H_sun: Sun height (degree)
                #  T2m: 2-m air temperature (degree Celsius)
                #  WS10m: 10-m total wind speed (m/s)
                #  Int: 1 means solar radiation values are reconstructed

        In addition the temperatur for different nuts_ids is saved as well as the produced PV profile for a certain PV
        peak power.
        The output is NUTS2_PV_Data, NUTS2_Radiation_Data and NUTS2_Temperature_Data in the SQLite Database.
        """
        nuts_level = "3"
        table_name_temperature = f"Temperature_NUTS{nuts_level}_{country}"
        table_name_radiation = f"Radiation_NUTS{nuts_level}_{country}"
        table_name_PV = f"PV_generation_NUTS{nuts_level}_{country}"
        number = 0
        print(nuts_id_list)
        for nuts in nuts_id_list:
            try:
                if number == 0:  # first run replace the existing tables
                    exists = "replace"
                else:
                    exists = "append"
                lat, lon = self.get_nuts_center(nuts)
                # CelestialDirections are "north", "south", "east", "west":
                # Orientation (azimuth) angle of the (fixed) plane, 0=south, 90=west, -90=east. Not relevant for tracking planes
                valid_nuts_id = True
                for CelestialDirection in ["south", "east", "west", "north"]:
                    if CelestialDirection == "south":
                        aspect = 0
                        df_south = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year,
                                                                            nuts)
                        if df_south is None:
                            valid_nuts_id = False
                            break
                        south_radiation = pd.to_numeric(df_south["Gb(i)"]) + pd.to_numeric(df_south["Gd(i)"])
                    elif CelestialDirection == "east":
                        aspect = -90
                        df_east = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
                        east_radiation = pd.to_numeric(df_east["Gb(i)"]) + pd.to_numeric(df_east["Gd(i)"])
                    elif CelestialDirection == "west":
                        aspect = 90
                        df_west = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
                        west_radiation = pd.to_numeric(df_west["Gb(i)"]) + pd.to_numeric(df_west["Gd(i)"])
                    elif CelestialDirection == "north":
                        aspect = -180
                        df_north = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year,
                                                                            nuts)
                        north_radiation = pd.to_numeric(df_north["Gb(i)"]) + pd.to_numeric(df_north["Gd(i)"])
                if not valid_nuts_id:
                    continue

                # save solar radiation
                radiation_table = {"nuts_id": np.full((8760,), nuts),
                                   "id_hour": self.id_hour,
                                   "south": south_radiation.reset_index(drop=True).to_numpy(),
                                   "east": east_radiation.reset_index(drop=True).to_numpy(),
                                   "west": west_radiation.reset_index(drop=True).to_numpy(),
                                   "north": north_radiation.reset_index(drop=True).to_numpy()}
                solar_radiation_table = pd.DataFrame(radiation_table)
                assert sorted(list(input_data_structure.RadiationData().__dict__.keys())) == sorted(list(
                    solar_radiation_table.columns))
                DB().write_dataframe(table_name=table_name_radiation,
                                     data_frame=solar_radiation_table,
                                     data_types=input_data_structure.RadiationData().__dict__,
                                     if_exists=exists
                                     )

                # save temperature
                outsideTemperature = pd.to_numeric(df_south["T2m"].reset_index(drop=True).rename("temperature"))
                temperature_dict = {"nuts_id": np.full((8760,), nuts),
                                    "id_hour": self.id_hour,
                                    "temperature": outsideTemperature.to_numpy()}
                # write table for outside temperature:
                temperature_table = pd.DataFrame(temperature_dict)
                assert sorted(list(input_data_structure.TemperatureData().__dict__.keys())) == sorted(list(
                    temperature_table.columns))
                DB().write_dataframe(table_name=table_name_temperature,
                                     data_frame=temperature_table,
                                     data_types=input_data_structure.TemperatureData().__dict__,
                                     if_exists=exists
                                     )

                # PV profiles:
                unit = np.full((len(self.id_hour),), "W")

                for index, peakPower in enumerate(PV_sizes):
                    if number == 0:  # first run replace the existing tables
                        exists = "replace"
                    else:
                        exists = "append"
                    if peakPower == 0:
                        PVProfile = np.full((8760,), 0)
                    else:
                        PVProfile = self.get_PV_generation(lat, lon, start_year, end_year, peakPower, nuts)

                    pv_type = np.full((len(PVProfile),), index + 1)
                    columns_pv = {"nuts_id": np.full((8760,), nuts),
                                  "ID_PV": pv_type,
                                  "id_hour": self.id_hour,
                                  "power": PVProfile,
                                  "unit": unit,
                                  "peak_power": np.full((8760,), peakPower),
                                  "peak_power_unit": np.full((8760,), "kWp")}
                    pv_table = pd.DataFrame(columns_pv)
                    assert sorted(list(pv_table.columns)) == sorted(list(
                        input_data_structure.PVData().__dict__.keys()))
                    DB().write_dataframe(table_name=table_name_PV,
                                         data_frame=pv_table,
                                         data_types=input_data_structure.PVData().__dict__,
                                         if_exists=exists
                                         )
                    number += 1

                print(f"{nuts} saved to intermediate table")
            except Exception as e:
                print(f'{nuts} not found.')

    def get_nuts_id_list(self, nuts_level: int, country: str) -> list:
        nuts = get_excel2df('NUTS.xlsx')
        nuts_ids = nuts.loc[nuts["nuts0"] == country][f'nuts{nuts_level}'].to_list()
        return nuts_ids

    def run(self,
            nuts_level: int,
            country_code: str,
            start_year: int,
            end_year: int,
            pv_sizes: list):

        self.get_PVGIS_data(country=country_code,
                            nuts_id_list=self.get_nuts_id_list(nuts_level, country_code),
                            start_year=start_year,
                            end_year=end_year,
                            PV_sizes=pv_sizes)


class CountryProfile:

    def __init__(self, country: str):
        self.country = country
        self.region_area_dict = self.gen_region_area_dict()

    def gen_region_area_dict(self):
        df = get_excel2df('NUTS_match.xlsx')
        df_filter = df.loc[(df["nuts0"] == self.country) & (df["floor_area"] > 0)]
        region_area_dict = {}
        for index, row in df_filter.iterrows():
            region_area_dict[row["nuts3"]] = row["floor_area"]
        return region_area_dict

    def get_floor_area(self):
        absolut_path = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"residential_floor_area.json")
        floor_area_total = pd.read_json(absolut_path, orient="table")
        mask = floor_area_total["nuts_id"].str.len() == 5
        floor_area = floor_area_total.loc[mask]
        floor_area = floor_area[floor_area["nuts_id"].str.startswith(self.country)].reset_index(drop=True)
        return floor_area

    def get_country_temperature_table(self):
        temperature_table = DB().read_dataframe(f"Temperature_NUTS3_" + self.country)
        return temperature_table

    def get_country_radiation_table(self):
        radiation_table = DB().read_dataframe(f"Radiation_NUTS3_" + self.country)
        return radiation_table

    def get_country_pv_table(self):
        pv_table = DB().read_dataframe(f"PV_generation_NUTS3_" + self.country)
        return pv_table

    def calc_country_pv_weighted_average_with_floor_area(self):
        country_pv_table = self.get_country_pv_table()
        pv_50 = np.zeros(8760, )
        pv_75 = np.zeros(8760, )
        pv_100 = np.zeros(8760, )
        area_sum = 0
        counter = 0
        for region, area in self.region_area_dict.items():
            print(f'counter: {self.country} - {counter + 1} of {len(self.region_area_dict)}')
            counter += 1
            region_pv_table = country_pv_table.loc[country_pv_table['nuts_id'] == region]
            if len(region_pv_table) == 8760 * 3:
                pv_50 += region_pv_table[region_pv_table["ID_PV"] == 1]['power'].to_numpy() * area
                pv_75 += region_pv_table[region_pv_table["ID_PV"] == 2]['power'].to_numpy() * area
                pv_100 += region_pv_table[region_pv_table["ID_PV"] == 3]['power'].to_numpy() * area
                area_sum += area
        pv_50 = pv_50 / area_sum
        pv_75 = pv_75 / area_sum
        pv_100 = pv_100 / area_sum

        result_dict = {
            'region': [self.country for _ in range(0, 8760)],
            'id_hour': [i for i in range(1, 8761)],
            'pv_50': pv_50,
            'pv_75': pv_75,
            'pv_100': pv_100,
            'unit': ['W' for _ in range(0, 8760)],
        }
        return result_dict

    def calc_country_weather_weighted_average_with_floor_area(self):
        country_temperature_table = self.get_country_temperature_table()
        country_radiation_table = self.get_country_radiation_table()
        country_temperature = np.zeros(8760, )
        country_radiation_south = np.zeros(8760, )
        country_radiation_north = np.zeros(8760, )
        country_radiation_east = np.zeros(8760, )
        country_radiation_west = np.zeros(8760, )
        area_sum = 0
        counter = 0
        for region, area in self.region_area_dict.items():
            print(f'counter: {self.country} - {counter + 1} of {len(self.region_area_dict)}')
            counter += 1
            region_temperature_table = country_temperature_table.loc[country_temperature_table['nuts_id'] == region]
            region_radiation_table = country_radiation_table.loc[country_temperature_table['nuts_id'] == region]
            region_temperature = region_temperature_table['temperature'].to_numpy()
            region_radiation_south = region_radiation_table['south'].to_numpy()
            region_radiation_north = region_radiation_table['north'].to_numpy()
            region_radiation_east = region_radiation_table['east'].to_numpy()
            region_radiation_west = region_radiation_table['west'].to_numpy()
            if len(region_temperature) == 8760 and len(region_radiation_south) == 8760 and area > 0:
                country_temperature += region_temperature * area
                country_radiation_south += region_radiation_south * area
                country_radiation_north += region_radiation_north * area
                country_radiation_east += region_radiation_east * area
                country_radiation_west += region_radiation_west * area
                area_sum += area
        country_temperature = country_temperature / area_sum
        country_radiation_south = country_radiation_south / area_sum
        country_radiation_north = country_radiation_north / area_sum
        country_radiation_east = country_radiation_east / area_sum
        country_radiation_west = country_radiation_west / area_sum

        result_dict = {
            'region': [self.country for _ in range(0, 8760)],
            'id_hour': [i for i in range(1, 8761)],
            'temperature': country_temperature,
            'south': country_radiation_south,
            'north': country_radiation_north,
            'east': country_radiation_east,
            'west': country_radiation_west,
        }
        return result_dict

    def calc_country_pv_geo_average(self):
        country_pv_table = self.get_country_pv_table()
        pv_50 = np.zeros(8760, )
        pv_75 = np.zeros(8760, )
        pv_100 = np.zeros(8760, )
        counter = 0
        nuts_id_list = country_pv_table["nuts_id"].unique()
        num_region = len(nuts_id_list)
        for region in nuts_id_list:
            print(f'counter: {self.country} - {counter + 1} of {len(self.region_area_dict)}')
            counter += 1
            region_pv_table = country_pv_table.loc[country_pv_table['nuts_id'] == region]
            if len(region_pv_table) == 8760 * 3:
                pv_50 += region_pv_table[region_pv_table["ID_PV"] == 1]['power'].to_numpy()
                pv_75 += region_pv_table[region_pv_table["ID_PV"] == 2]['power'].to_numpy()
                pv_100 += region_pv_table[region_pv_table["ID_PV"] == 3]['power'].to_numpy()
        pv_50 = pv_50 / num_region
        pv_75 = pv_75 / num_region
        pv_100 = pv_100 / num_region

        result_dict = {
            'region': [self.country for _ in range(0, 8760)],
            'id_hour': [i for i in range(1, 8761)],
            'pv_50': pv_50,
            'pv_75': pv_75,
            'pv_100': pv_100,
            'unit': ['W' for _ in range(0, 8760)],
        }
        return result_dict

    def calc_country_weather_geo_average(self):
        country_temperature_table = self.get_country_temperature_table()
        country_radiation_table = self.get_country_radiation_table()
        country_temperature = np.zeros(8760, )
        country_radiation_south = np.zeros(8760, )
        country_radiation_north = np.zeros(8760, )
        country_radiation_east = np.zeros(8760, )
        country_radiation_west = np.zeros(8760, )
        counter = 0
        nuts_id_list = country_temperature_table["nuts_id"].unique()
        num_region = len(nuts_id_list)
        for region in nuts_id_list:
            print(f'counter: {self.country} - {counter + 1} of {len(self.region_area_dict)}')
            counter += 1
            region_temperature_table = country_temperature_table.loc[country_temperature_table['nuts_id'] == region]
            region_radiation_table = country_radiation_table.loc[country_temperature_table['nuts_id'] == region]
            region_temperature = region_temperature_table['temperature'].to_numpy()
            region_radiation_south = region_radiation_table['south'].to_numpy()
            region_radiation_north = region_radiation_table['north'].to_numpy()
            region_radiation_east = region_radiation_table['east'].to_numpy()
            region_radiation_west = region_radiation_table['west'].to_numpy()
            if len(region_temperature) == 8760 and len(region_radiation_south) == 8760:
                country_temperature += region_temperature
                country_radiation_south += region_radiation_south
                country_radiation_north += region_radiation_north
                country_radiation_east += region_radiation_east
                country_radiation_west += region_radiation_west
        country_temperature = country_temperature / num_region
        country_radiation_south = country_radiation_south / num_region
        country_radiation_north = country_radiation_north / num_region
        country_radiation_east = country_radiation_east / num_region
        country_radiation_west = country_radiation_west / num_region

        result_dict = {
            'region': [self.country for _ in range(0, 8760)],
            'id_hour': [i for i in range(1, 8761)],
            'temperature': country_temperature,
            'south': country_radiation_south,
            'north': country_radiation_north,
            'east': country_radiation_east,
            'west': country_radiation_west,
        }
        return result_dict


def get_json2df(file: str):
    absolut_path = Path(os.path.abspath(__file__)).parent.resolve() / Path(file)
    df = pd.read_json(absolut_path, orient="table")
    return df


def get_excel2df(file: str):
    df = pd.read_excel(file)
    return df


def create_nuts_yaml2df():
    # from OpenEntrance project: https://github.com/openENTRANCE/openentrance/blob/main/definitions/region/nuts3.yaml
    info = []
    columns = ['country', 'nuts0', 'nuts1', 'nuts2', 'nuts3']
    with open("NUTS.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        nuts3 = data_loaded[0]["NUTS3"]
        for region in nuts3:
            for key, value in region.items():
                if key[-3:-1] == 'ZZ':
                    pass
                else:
                    info.append([value["country"], value["nuts1"][0:2], value["nuts1"], value["nuts2"], key])
        df = pd.DataFrame(info, columns=columns, index=None)
        df.to_excel(f'NUTS.xlsx', index=False)


def export_to_excel(file: str):
    df = get_json2df(file)
    df.to_excel(f'{file}.xlsx', index=False)


def match_heat_and_area():
    nuts = get_excel2df("NUTS.xlsx")
    nuts['heat_demand'] = pd.Series(dtype=float)
    nuts['floor_area'] = pd.Series(dtype=float)
    heat = get_json2df("heat_demand.json")
    heat['match'] = 0
    area = get_json2df("residential_floor_area.json")
    area['match'] = 0

    for index, row in nuts.iterrows():
        region = row["nuts3"]

        if len(heat.loc[heat["nuts_id"] == region]) > 0:
            heat_demand = heat.loc[heat["nuts_id"] == region]['sum'].values[0]
            heat.loc[heat["nuts_id"] == region, 'match'] = 1
            nuts.loc[nuts["nuts3"] == region, 'heat_demand'] = heat_demand

        if len(area.loc[area["nuts_id"] == region]) > 0:
            floor_area = area.loc[area["nuts_id"] == region]['sum'].values[0]
            area.loc[area["nuts_id"] == region, 'match'] = 1
            nuts.loc[nuts["nuts3"] == region, 'floor_area'] = floor_area

    nuts.to_excel(f'NUTS_match.xlsx', index=False)
    heat.to_excel(f'heat_demand.xlsx', index=False)
    area.to_excel(f'floor_area.xlsx', index=False)


def download_pv_gis_data(country_list: List[str]):
    for country in country_list:
        PVGIS().run(nuts_level=3,
                    country_code=country,
                    start_year=2019,
                    end_year=2019,
                    pv_sizes=[5, 7.5, 10])


def calc_country_weather_and_pv_weighted_with_floor_area(country_list: List[str]):
    no_floor_area_data = []
    no_weather_data = []
    country_weather_list = []
    country_pv_list = []
    for country in country_list:
        cp = CountryProfile(country)
        if len(cp.region_area_dict) == 0:
            no_floor_area_data.append(country)
        else:
            try:
                country_weather = pd.DataFrame.from_dict(cp.calc_country_weather_weighted_average_with_floor_area())
                country_weather_list.append(country_weather)
                country_pv = pd.DataFrame.from_dict(cp.calc_country_pv_weighted_average_with_floor_area())
                country_pv_list.append(country_pv)
            except Exception:
                no_weather_data.append(country)
    weather = pd.concat(country_weather_list)
    weather.to_excel(r'Weather_weighted_with_floor_area.xlsx', index=False)
    pv = pd.concat(country_pv_list)
    pv.to_excel(r'PV_weighted_with_floor_area.xlsx', index=False)

    print(f'no_floor_area_data: {no_floor_area_data}')
    print(f'no_weather_data: {no_weather_data}')


def calc_country_weather_and_pv_geo_average():
    country_list = ['IE', 'LT', 'ME', 'MK', 'AL', 'RS', 'TR']
    no_weather_data = []
    country_weather_list = []
    country_pv_list = []
    for country in country_list:
        print('h')
        cp = CountryProfile(country)
        try:
            country_weather = pd.DataFrame.from_dict(cp.calc_country_weather_geo_average())
            country_weather_list.append(country_weather)
            country_pv = pd.DataFrame.from_dict(cp.calc_country_pv_geo_average())
            country_pv_list.append(country_pv)
        except Exception:
            no_weather_data.append(country)
    weather = pd.concat(country_weather_list)
    weather.to_excel(r'Weather_geo_average.xlsx', index=False)
    pv = pd.concat(country_pv_list)
    pv.to_excel(r'PV_geo_average.xlsx', index=False)

    print(f'no_weather_data: {no_weather_data}')


if __name__ == "__main__":
    COUNTRY_LIST = get_excel2df('NUTS.xlsx')['nuts0'].unique()
    # download_pv_gis_data(COUNTRY_LIST)
    # calc_country_weather_and_pv(COUNTRY_LIST)
    calc_country_weather_and_pv_geo_average()

