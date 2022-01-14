import pandas as pd
import numpy as np
from pathlib import Path
import sqlalchemy.types
import geopandas as gpd
from pyproj import CRS, Transformer
import urllib.error

from _Refactor.basic.db import DB
from _Refactor.basic.reg import Table
import _Refactor.data.input_data_structure as input_data_structure
import _Refactor.basic.config as config


class GenerateDataForRoot:

    def __init__(self):
        self.id_hour = np.arange(1, 8761)
        self.SolarRadiationList = []
        self.TemperatureList = []
        self.PVPowerList = []

    def create_nuts_regions(self):
        # read nuts regions from excel (eurostat)
        path_to_file = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Refactor\data\inputdata\NUTS2021.xlsx"
        NUTS_codes = pd.read_excel(path_to_file,
                                   sheet_name="NUTS & SR 2021",
                                   engine="openpyxl"
                                   )["Code 2021"].dropna().to_numpy()
        NUTS1 = []
        NUTS2 = []
        NUTS3 = []
        # iterate through NUTS code and create tables for NUTS1, NUTS2 and NUTS3
        for code in NUTS_codes:
            if "Z" in code:  # drop extra nuts regions
                continue
            elif len(code) == 2:
                continue
            elif len(code) == 3:  # NUTS1
                NUTS1.append(code)
            elif len(code) == 4:  # NUTS2
                NUTS2.append(code)
            elif len(code) == 5:  # NUTS3
                NUTS3.append(code)

        nuts1_country_column = [nuts[:2] for nuts in NUTS1]
        nuts2_country_column = [nuts[:2] for nuts in NUTS2]
        nuts3_country_column = [nuts[:2] for nuts in NUTS3]

        nuts1_frame = pd.DataFrame(np.column_stack([nuts1_country_column, NUTS1]), columns=["country", "nuts_id"])
        nuts2_frame = pd.DataFrame(np.column_stack([nuts2_country_column, NUTS2]), columns=["country", "nuts_id"])
        nuts3_frame = pd.DataFrame(np.column_stack([nuts3_country_column, NUTS3]), columns=["country", "nuts_id"])
        # save to root:
        DB().write_dataframe(table_name="NUTS1",
                             data_frame=nuts1_frame,
                             data_types={"country": sqlalchemy.types.String, "nuts_id": sqlalchemy.types.String},
                             if_exists="replace"
                             )
        DB().write_dataframe(table_name="NUTS2",
                             data_frame=nuts2_frame,
                             data_types={"country": sqlalchemy.types.String, "nuts_id": sqlalchemy.types.String},
                             if_exists="replace"
                             )
        DB().write_dataframe(table_name="NUTS3",
                             data_frame=nuts3_frame,
                             data_types={"country": sqlalchemy.types.String, "nuts_id": sqlalchemy.types.String},
                             if_exists="replace"
                             )

    def get_nuts_center(
            self,
            region_id,
            url_nuts0_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_0.geojson",
            url_nuts1_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_1.geojson",
            url_nuts2_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_2.geojson",
            url_nuts3_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_3.geojson"
    ):
        """
        This function returns the latitude and longitude of the center of a NUTS region.
        """
        region_id = region_id.strip()
        if len(region_id) == 2:
            url = url_nuts0_poly
        elif len(region_id) == 3:
            url = url_nuts1_poly
        elif len(region_id) == 4:
            url = url_nuts2_poly
        elif len(region_id) > 4:
            url = url_nuts3_poly
        else:
            assert False, f"Error could not identify nuts level for {region_id}"
        nuts = gpd.read_file(url)
        transformer = Transformer.from_crs(CRS("EPSG:3035"), CRS("EPSG:4326"))
        point = nuts[nuts.NUTS_ID == region_id].centroid.values[0]
        return transformer.transform(point.y, point.x)  # returns lat, lon

    def get_PV_generation(self, lat, lon, startyear, endyear, peakpower, nuts_id) -> pd.DataFrame:
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

        req = f"https://re.jrc.ec.europa.eu/api/seriescalc?lat={lat}&" \
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
            PV_Profile = pd.to_numeric(df["P"]).to_numpy()   # W
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

        req = f"https://re.jrc.ec.europa.eu/api/seriescalc?lat={lat}&" \
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
            df = None
            return df

    def get_PVGIS_data(self,
                       country: str,
                       nuts_id_list: list,
                       start_year: int,
                       end_year: int) -> None:
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
        # determine if NUTS1,2 or 3:
        if len(nuts_id_list[0]) == 3:  # NUTS1
            nuts_level = "1"
        elif len(nuts_id_list[0]) == 4:
            nuts_level = "2"
        elif len(nuts_id_list[0]) == 5:
            nuts_level = "3"
        table_name_temperature = f"Temperature_NUTS{nuts_level}_{country}"
        table_name_radiation = f"Radiation_NUTS{nuts_level}_{country}"
        table_name_PV = f"PV_generation_NUTS{nuts_level}_{country}"
        number = 0
        for nuts in nuts_id_list:
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
                    df_south = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
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
                    df_north = self.get_temperature_and_solar_radiation(lat, lon, aspect, start_year, end_year, nuts)
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
            assert list(input_data_structure.RadiationData().__dict__.keys()).sort() == list(solar_radiation_table.columns).sort()
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
            assert list(input_data_structure.TemperatureData().__dict__.keys()).sort() == list(temperature_table.columns).sort()
            DB().write_dataframe(table_name=table_name_temperature,
                                 data_frame=temperature_table,
                                 data_types=input_data_structure.TemperatureData().__dict__,
                                 if_exists=exists
                                 )

            # PV profiles:
            PV_options = DB().read_dataframe(Table().pv)
            unit = np.full((len(self.id_hour),), "W")

            for index, peakPower in enumerate(PV_options.peak_power):
                if number == 0:  # first run replace the existing tables
                    exists = "replace"
                else:
                    exists = "append"
                if peakPower == 0:
                    continue  # 0-profile will not be saved
                else:
                    PVProfile = self.get_PV_generation(lat, lon, start_year, end_year, peakPower, nuts)

                    pv_type = np.full((len(PVProfile),), int(PV_options.ID_PV[index]))
                    columns_pv = {"nuts_id": np.full((8760,), nuts),
                                  "ID_PV": pv_type,
                                  "id_hour": self.id_hour,
                                  "power": PVProfile,
                                  "unit": unit}
                    pv_table = pd.DataFrame(columns_pv)
                    assert list(pv_table.columns).sort() == list(input_data_structure.PVGenerationData().__dict__.keys()).sort()
                    DB().write_dataframe(table_name=table_name_PV,
                                         data_frame=pv_table,
                                         data_types=input_data_structure.PVGenerationData().__dict__,
                                         if_exists=exists
                                         )
                    number += 1

            print(f"{nuts} saved to intermediate table")

    def calculate_single_profile_for_country(self, country: str, nuts_level: int) -> None:
        """nuts level 1, 2 or 3"""
        # weighted average over heat demand from hotmaps TODO get Hotmaps data through API
        heat_demand_AT = pd.read_excel(
            "C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Philipp/inputdata/AUT/AT_Heat_demand_total_NUTS3.xlsx",
            engine="openpyxl")
        total_heat_demand = heat_demand_AT.loc[:, "sum"].sum()

        # create numpy arrays that will be filled inside the loop:
        temperature_weighted_sum = np.zeros((8760, 1))
        radiation_weighted_sum = np.zeros((8760, 4))
        # get different PV id types:
        unique_PV_id_types = np.unique(
            DB().read_dataframe(f"PV_generation_NUTS{nuts_level}_{country}",
                                "ID_PV").to_numpy()  # *columns
        )
        # check if the PV types in the root are also there from PV GIS
        assert list(unique_PV_id_types).sort() == list(DB().read_dataframe(Table().pv).ID_PV).sort()

        # dictionary for different PV types
        PV_weighted_sum = {id_pv: np.zeros((8760, 1)) for id_pv in unique_PV_id_types}
        for index, row in heat_demand_AT.iterrows():
            nuts_id = row["nuts_id"]
            heat_demand_of_specific_region = row["sum"]
            # Temperature
            temperature_profile = DB().read_dataframe(f"Temperature_NUTS{nuts_level}_{country}",
                                                      "temperature",  # *columns
                                                      nuts_id=nuts_id).to_numpy()  # **kwargs
            temperature_weighted_sum += temperature_profile * heat_demand_of_specific_region / total_heat_demand

            # Solar radiation
            radiation_profile = DB().read_dataframe(f"Radiation_NUTS{nuts_level}_{country}",
                                                    "south", "east", "west", "north",  # *columns
                                                    nuts_id=nuts_id).to_numpy()  # **kwargs
            radiation_weighted_sum += radiation_profile * heat_demand_of_specific_region / total_heat_demand

            # PV
            # iterate through different PV types and add each weighted profiles to corresponding dictionary index
            for id_pv in unique_PV_id_types:
                PV_profile = DB().read_dataframe(f"PV_generation_NUTS{nuts_level}_{country}",
                                                 "power",  # *columns
                                                 nuts_id=nuts_id, ID_PV=id_pv).to_numpy()  # **kwargs
                PV_weighted_sum[id_pv] += PV_profile * heat_demand_of_specific_region / total_heat_demand

        # create table for saving to DB:
        # Temperature
        temperature_columns = {"nuts_id": np.full((8760, ), country),
                               "id_hour": self.id_hour,
                               "temperature": temperature_weighted_sum.flatten()}
        temperature_table = pd.DataFrame(temperature_columns)
        assert list(temperature_table.columns).sort() == list(input_data_structure.TemperatureData().__dict__.keys()).sort()
        # save weighted temperature average profile to database:
        DB().write_dataframe(table_name=Table().temperature,
                             data_frame=temperature_table,
                             data_types=input_data_structure.TemperatureData().__dict__,
                             if_exists="replace"
                             )

        # Radiation
        radiation_columns = {"nuts_id": np.full((8760, ), country),
                             "id_hour": self.id_hour,
                             "south": radiation_weighted_sum[:, 0],
                             "east": radiation_weighted_sum[:, 1],
                             "west": radiation_weighted_sum[:, 2],
                             "north": radiation_weighted_sum[:, 3]}
        radiation_table = pd.DataFrame(radiation_columns)
        assert list(radiation_table.columns).sort() == list(input_data_structure.RadiationData().__dict__.keys()).sort()

        # save to sql database:
        DB().write_dataframe(table_name=Table().radiation,
                             data_frame=radiation_table,
                             data_types=input_data_structure.RadiationData().__dict__,
                             if_exists="replace"
                             )

        # PV
        pv_columns = input_data_structure.PVGenerationData().__dict__.keys()
        # create single row on which tables are stacked up
        pv_table_numpy = np.zeros((1, len(pv_columns)))
        for key, values in PV_weighted_sum.items():
            # create the table for each pv type
            single_pv_table = np.column_stack([np.full((8760, ), country),  # nuts_id
                                               np.full((8760, ), key),  # ID_PV
                                               self.id_hour,  # id_hour
                                               values,  # power
                                               np.full((8760, 1), "W")])  # unit
            # stack the tables from dictionary to one large table
            pv_table_numpy = np.vstack([pv_table_numpy, single_pv_table])

        # remove the first row (zeros) from the pv_table
        pv_table_numpy = pv_table_numpy[1:]
        pv_table = pd.DataFrame(pv_table_numpy, columns=pv_columns)
        # save to sql database:
        DB().write_dataframe(table_name=Table().pv_generation,
                             data_frame=pv_table,
                             data_types=input_data_structure.PVGenerationData().__dict__,
                             if_exists="replace"
                             )

        print(f"mean tables for country {country} have been saved")

    def get_nuts_id_list(self, nuts_level: int, country: str) -> list:
        database_name = f"NUTS{nuts_level}"
        all_nuts_ids = DB().read_dataframe(database_name, country=country)["nuts_id"].to_numpy()
        return list(all_nuts_ids)

    def run(self):
        # delete old tables:
        DB().drop_table("Temperature")
        DB().drop_table("Radiation")
        DB().drop_table("PV_generation")
        # check if nuts regions exist in root:
        engine = config.root_connection.connect()
        if not engine.dialect.has_table(engine, "NUTS1") or not \
                engine.dialect.has_table(engine, "NUTS2") or not \
                engine.dialect.has_table(engine, "NUTS3"):
            self.create_nuts_regions()

        self.get_PVGIS_data(nuts_id_list=self.get_nuts_id_list(3, "AT"),
                            country="AT",
                            start_year=2010,
                            end_year=2010)  # TODO make this versatile for other countries
        # creating a single profile as mean profile for a country weighted by the heat demand of each nuts region:
        self.calculate_single_profile_for_country(country="AT", nuts_level=3)


if __name__ == "__main__":
    GenerateDataForRoot().run()
