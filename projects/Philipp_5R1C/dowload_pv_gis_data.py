import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from pyproj import CRS, Transformer
import urllib.error
import os


class PVGIS:

    def __init__(self, output_path: str):
        self.id_hour = np.arange(1, 8761)
        self.output_path = Path(output_path)

    def get_temperature_and_solar_radiation(self, lat, lon, aspect, startyear, endyear) -> pd.DataFrame:
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
            print(f"does not have a valid location for PV-GIS. \n lat: {lat} \n lon: {lon}")
            return None


    def get_PVGIS_data(self,
                       latitude: float,
                       longitude: float,
                       start_year: int,
                       end_year: int
                       ) -> None:
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
        # CelestialDirections are "north", "south", "east", "west":
        # Orientation (azimuth) angle of the (fixed) plane, 0=south, 90=west, -90=east. Not relevant for tracking planes
        for CelestialDirection in ["south", "east", "west", "north"]:
            if CelestialDirection == "south":
                aspect = 0
                df_south = self.get_temperature_and_solar_radiation(latitude, longitude, aspect, start_year, end_year)
                if df_south is None:
                    valid_nuts_id = False
                    break
                south_radiation = pd.to_numeric(df_south["Gb(i)"]) + pd.to_numeric(df_south["Gd(i)"])
            elif CelestialDirection == "east":
                aspect = -90
                df_east = self.get_temperature_and_solar_radiation(latitude, longitude, aspect, start_year, end_year)
                east_radiation = pd.to_numeric(df_east["Gb(i)"]) + pd.to_numeric(df_east["Gd(i)"])
            elif CelestialDirection == "west":
                aspect = 90
                df_west = self.get_temperature_and_solar_radiation(latitude, longitude, aspect, start_year, end_year)
                west_radiation = pd.to_numeric(df_west["Gb(i)"]) + pd.to_numeric(df_west["Gd(i)"])
            elif CelestialDirection == "north":
                aspect = -180
                df_north = self.get_temperature_and_solar_radiation(latitude, longitude, aspect, start_year, end_year)
                north_radiation = pd.to_numeric(df_north["Gb(i)"]) + pd.to_numeric(df_north["Gd(i)"])

        # save solar radiation
        radiation_table = {"id_hour": self.id_hour,
                           "south": south_radiation.reset_index(drop=True).to_numpy(),
                           "east": east_radiation.reset_index(drop=True).to_numpy(),
                           "west": west_radiation.reset_index(drop=True).to_numpy(),
                           "north": north_radiation.reset_index(drop=True).to_numpy()}
        solar_radiation_table = pd.DataFrame(radiation_table)

        # write to csv:
        solar_radiation_table.to_csv(self.output_path / Path("solar_radiation.csv"), sep=";")

        # save temperature
        outsideTemperature = pd.to_numeric(df_south["T2m"].reset_index(drop=True).rename("temperature"))
        temperature_dict = {"id_hour": self.id_hour,
                            "temperature": outsideTemperature.to_numpy()}
        # write table for outside temperature:
        temperature_table = pd.DataFrame(temperature_dict)

        # write to csv:
        temperature_table.to_csv(self.output_path / Path("temperature_pvgis.csv"), sep=";")


    def run(self,
            latitude: float,
            longitude: float,
            start_year: int,
            end_year: int):

        self.get_PVGIS_data(latitude,
                            longitude,
                            start_year=start_year,
                            end_year=end_year,
                            )


if __name__ == "__main__":
    output_path = r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\Philipp_5R1C\ouput_data"
    PVGIS(output_path).run(latitude=47.841, longitude=13.023, start_year=2018, end_year=2018)
