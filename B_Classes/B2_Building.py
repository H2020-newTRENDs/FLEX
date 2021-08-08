from A_Infrastructure.A1_CONS import CONS
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS, Transformer
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table



class Building:

    def __init__(self, para_series):
        # self.ID_BuildingType = para_series["ID_BuildingType"]
        # self.ID_BuildingAgeClass = para_series["ID_BuildingAgeClass"]
        # self.FloorArea = para_series["FloorArea"]

        self.ID = para_series['ID']
        self.f1 = para_series['f1']
        self.index = para_series['index']
        self.name = para_series['name']
        self.construction_period_start = para_series['construction_period_start']
        self.construction_period_end = para_series['construction_period_end']
        self.building_categories_index = para_series['building_categories_index']
        self.number_of_dwellings_per_building = para_series['number_of_dwellings_per_building']
        self.horizontal_shading_building = para_series['horizontal_shading_building']  #
        self.areawindows = para_series['areawindows']
        self.area_suitable_solar = para_series['area_suitable_solar']
        self.ued_dhw = para_series['ued_dhw']
        self.average_effective_area_wind_west_east_red_cool \
            = para_series['average_effective_area_wind_west_east_red_cool']
        self.average_effective_area_wind_south_red_cool \
            = para_series['average_effective_area_wind_south_red_cool']
        self.average_effective_area_wind_north_red_cool \
            = para_series['average_effective_area_wind_north_red_cool']
        self.spec_int_gains_cool_watt = para_series['spec_int_gains_cool_watt']
        self.building_categories_index = para_series['building_categories_index']
        self.Af = para_series['Af']
        self.Hop = para_series['Hop']
        self.Htr_w = para_series['Htr_w']
        self.Hve = para_series['Hve']
        self.CM_factor = para_series['CM_factor']
        self.Am_factor = para_series['Am_factor']
        self.hwb_norm1 = para_series['hwb_norm1']
        self.country_ID = para_series['country_ID']

        self.MaximalGridPower = para_series['MaximalGridPower']
        self.BuildingMassTemperatureStartValue = para_series['BuildingMassTemperatureStartValue']
        self.MaximalBuildingMassTemperature = para_series['MaximalBuildingMassTemperature']


class HeatingCooling_noDR:

    def __init__(self, ID_BuildingOption_dataframe):
        try:
            self.index = ID_BuildingOption_dataframe['index']
            self.name = ID_BuildingOption_dataframe['name']
            self.building_categories_index = ID_BuildingOption_dataframe['building_categories_index']
            self.number_of_dwellings_per_building = ID_BuildingOption_dataframe['number_of_dwellings_per_building']
            self.areawindows = ID_BuildingOption_dataframe['areawindows']
            self.area_suitable_solar = ID_BuildingOption_dataframe['area_suitable_solar']
            self.ued_dhw = ID_BuildingOption_dataframe['ued_dhw']
            self.AreaWindowEastWest = ID_BuildingOption_dataframe[
                'average_effective_area_wind_west_east_red_cool'].to_numpy()
            self.AreaWindowSouth = ID_BuildingOption_dataframe['average_effective_area_wind_south_red_cool'].to_numpy()
            self.AreaWindowNorth = ID_BuildingOption_dataframe['average_effective_area_wind_north_red_cool'].to_numpy()
            self.building_categories_index = ID_BuildingOption_dataframe['building_categories_index']
        except:
            pass
        self.InternalGains = ID_BuildingOption_dataframe['spec_int_gains_cool_watt'].to_numpy()
        self.Hop = ID_BuildingOption_dataframe['Hop'].to_numpy()
        self.Htr_w = ID_BuildingOption_dataframe['Htr_w'].to_numpy()
        self.Hve = ID_BuildingOption_dataframe['Hve'].to_numpy()
        self.CM_factor = ID_BuildingOption_dataframe['CM_factor'].to_numpy()
        self.Am_factor = ID_BuildingOption_dataframe['Am_factor'].to_numpy()
        # self.country_ID = ID_BuildingOption_dataframe['country_ID'].to_numpy()
        try:
            self.Af = ID_BuildingOption_dataframe['Af'].to_numpy()
        except:
            self.Af = ID_BuildingOption_dataframe["areafloor"].to_numpy()

    def ref_HeatingCooling(self, T_outside, Q_solar=None,
                           initial_thermal_mass_temp=20, T_air_min=20, T_air_max=26):
        """
        This function calculates the heating and cooling demand as well as the indoor temperature for every building
        category based in the 5R1C model. The results are hourls vectors for one year. Q_solar is imported from a CSV
        at the time!
        """
        if type(T_outside) is np.ndarray:
            pass
        else:
            T_outside = T_outside.loc[:, "Temperature"]

        if Q_solar is None:
            print("keine Solarstrahlung gegeben! \n Achtung falsches Ergebnis!")
            Q_sol_all = pd.read_csv(CONS().DatabasePath + "\\directRadiation_himmelsrichtung_GER.csv", sep=";")
            Q_sol_north = np.outer(Q_sol_all.loc[:, "RadiationNorth"].to_numpy(), self.AreaWindowNorth)
            Q_sol_south = np.outer(Q_sol_all.loc[:, "RadiationSouth"].to_numpy(), self.AreaWindowSouth)
            Q_sol_east = np.outer(Q_sol_all.loc[:, "RadiationEast"].to_numpy(), self.AreaWindowEastWest)
            Q_sol_west = np.outer(Q_sol_all.loc[:, "RadiationWest"].to_numpy(), self.AreaWindowEastWest)
            Q_solar = Q_sol_north + Q_sol_south + Q_sol_east + Q_sol_west
            print("Q_solar is calculated from csv file")
        else:
            pass
        # Oberflächeninhalt aller Flächen, die zur Gebäudezone weisen
        Atot = 4.5 * self.Af
        # Speicherkapazität J/K
        Cm = self.CM_factor * self.Af
        # wirksame Massenbezogene Fläche [m^2]
        Am = self.Am_factor * self.Af
        # internal gains
        Q_InternalGains = self.InternalGains * self.Af
        timesteps = np.arange(len(T_outside))

        # Kopplung Temp Luft mit Temp Surface Knoten s
        his = np.float_(3.45)  # 7.2.2.2
        # kopplung zwischen Masse und  zentralen Knoten s (surface)
        hms = np.float_(9.1)  # W / m2K from Equ.C.3 (from 12.2.2)
        Htr_ms = hms * Am  # from 12.2.2 Equ. (64)
        Htr_em = 1 / (1 / self.Hop - 1 / Htr_ms)  # from 12.2.2 Equ. (63)
        # thermischer Kopplungswerte W/K
        Htr_is = his * Atot
        Htr_1 = np.float_(1) / (np.float_(1) / self.Hve + np.float_(1) / Htr_is)  # Equ. C.6
        Htr_2 = Htr_1 + self.Htr_w  # Equ. C.7
        Htr_3 = 1 / (1 / Htr_2 + 1 / Htr_ms)  # Equ.C.8

        # Equ. C.1
        PHI_ia = 0.5 * Q_InternalGains

        Tm_t = np.zeros(shape=(len(timesteps), len(self.Hve)))
        T_sup = np.zeros(shape=(len(timesteps),))
        Q_Heating_noDR = np.zeros(shape=(len(timesteps), len(self.Hve)))
        Q_Cooling_noDR = np.zeros(shape=(len(timesteps), len(self.Hve)))
        T_Room_noDR = np.zeros(shape=(len(timesteps), len(self.Hve)))
        heating_power_10 = self.Af * 10

        for t in timesteps:  # t is the index for each timestep
            # # Equ. C.2
            # PHI_m = Am / Atot * (0.5 * Q_InternalGains + Q_solar[t, :])
            # # Equ. C.3
            # PHI_st = (1 - Am / Atot - self.Htr_w / 9.1 / Atot) * \
            #          (0.5 * Q_InternalGains + Q_solar[t, :])

            # For RefBuilding Thomas
            # Equ. C.2
            PHI_m = Am / Atot * (0.5 * Q_InternalGains + Q_solar[t])
            # Equ. C.3
            PHI_st = (1 - Am / Atot - self.Htr_w / 9.1 / Atot) * \
                     (0.5 * Q_InternalGains + Q_solar[t])

            # (T_sup = T_outside weil die Zuluft nicht vorgewärmt oder vorgekühlt wird)
            T_sup[t] = T_outside[t]

            # Equ. C.5
            PHI_mtot_0 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (((PHI_ia + 0) / self.Hve) + T_sup[t])) / \
                         Htr_2

            # Equ. C.5 with 10 W/m^2 heating power
            PHI_mtot_10 = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                    ((PHI_ia + heating_power_10) / self.Hve) + T_sup[t])) / Htr_2

            # Equ. C.5 with 10 W/m^2 cooling power
            PHI_mtot_10_c = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                    ((PHI_ia - heating_power_10) / self.Hve) + T_sup[t])) / Htr_2

            if t == 0:
                if type(initial_thermal_mass_temp) == int or type(initial_thermal_mass_temp) == float:
                    Tm_t_prev = np.array([initial_thermal_mass_temp] * len(self.Hve))
                else:  # initial temperature is already a vector
                    Tm_t_prev = initial_thermal_mass_temp
            else:
                Tm_t_prev = Tm_t[t - 1, :]

            # Equ. C.4
            Tm_t_0 = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_0) / \
                     (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.4 for 10 W/m^2 heating
            Tm_t_10 = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_10) / \
                      (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.4 for 10 W/m^2 cooling
            Tm_t_10_c = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_10_c) / \
                        (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.9
            T_m_0 = (Tm_t_0 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 heating
            T_m_10 = (Tm_t_10 + Tm_t_prev) / 2

            # Equ. C.9 for 10 W/m^2 cooling
            T_m_10_c = (Tm_t_10_c + Tm_t_prev) / 2

            # Euq. C.10
            T_s_0 = (Htr_ms * T_m_0 + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                     (T_sup[t] + (PHI_ia + 0) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

            # Euq. C.10 for 10 W/m^2 heating
            T_s_10 = (Htr_ms * T_m_10 + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                      (T_sup[t] + (PHI_ia + heating_power_10) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

            # Euq. C.10 for 10 W/m^2 cooling
            T_s_10_c = (Htr_ms * T_m_10_c + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                        (T_sup[t] + (PHI_ia - heating_power_10) / self.Hve)) / (Htr_ms + self.Htr_w + Htr_1)

            # Equ. C.11
            T_air_0 = (Htr_is * T_s_0 + self.Hve * T_sup[t] + PHI_ia + 0) / \
                      (Htr_is + self.Hve)

            # Equ. C.11 for 10 W/m^2 heating
            T_air_10 = (Htr_is * T_s_10 + self.Hve * T_sup[t] + PHI_ia + heating_power_10) / \
                       (Htr_is + self.Hve)

            # Equ. C.11 for 10 W/m^2 cooling
            T_air_10_c = (Htr_is * T_s_10_c + self.Hve * T_sup[t] + PHI_ia - heating_power_10) / \
                         (Htr_is + self.Hve)

            for i in range(len(self.Hve)):
                # Check if air temperature without heating is in between boundaries and calculate actual HC power:
                if T_air_0[i] >= T_air_min and T_air_0[i] <= T_air_max:
                    Q_Heating_noDR[t, i] = 0
                elif T_air_0[i] < T_air_min:  # heating is required
                    Q_Heating_noDR[t, i] = heating_power_10[i] * (T_air_min - T_air_0[i]) / (T_air_10[i] - T_air_0[i])
                elif T_air_0[i] > T_air_max:  # cooling is required
                    Q_Cooling_noDR[t, i] = heating_power_10[i] * (T_air_max - T_air_0[i]) / (T_air_10_c[i] - T_air_0[i])

            # now calculate the actual temperature of thermal mass Tm_t with Q_HC_real:
            # Equ. C.5 with actual heating power
            PHI_mtot_real = PHI_m + Htr_em * T_outside[t] + Htr_3 * (
                    PHI_st + self.Htr_w * T_outside[t] + Htr_1 * (
                    ((PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / self.Hve) + T_sup[t])) / Htr_2
            # Equ. C.4
            Tm_t[t, :] = (Tm_t_prev * (Cm / 3600 - 0.5 * (Htr_3 + Htr_em)) + PHI_mtot_real) / \
                         (Cm / 3600 + 0.5 * (Htr_3 + Htr_em))

            # Equ. C.9
            T_m_real = (Tm_t[t, :] + Tm_t_prev) / 2

            # Euq. C.10
            T_s_real = (Htr_ms * T_m_real + PHI_st + self.Htr_w * T_outside[t] + Htr_1 *
                        (T_sup[t] + (PHI_ia + Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / self.Hve)) / \
                       (Htr_ms + self.Htr_w + Htr_1)

            # Equ. C.11 for 10 W/m^2 heating
            T_Room_noDR[t, :] = (Htr_is * T_s_real + self.Hve * T_sup[t] + PHI_ia +
                                 Q_Heating_noDR[t, :] - Q_Cooling_noDR[t, :]) / (Htr_is + self.Hve)

        # fill nan
        Q_Cooling_noDR = np.nan_to_num(Q_Cooling_noDR, nan=0)
        Q_Heating_noDR = np.nan_to_num(Q_Heating_noDR, nan=0)
        T_Room_noDR = np.nan_to_num(T_Room_noDR, nan=0)
        Tm_t = np.nan_to_num(Tm_t, nan=0)

        return Q_Heating_noDR, Q_Cooling_noDR, T_Room_noDR, Tm_t


def get_solar_radiation_cilestial_direction(lat, lon, startyear=2010, endyear=2010):
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

    The output Dataframe has a multiIndex. First index is celestial direction, second index is the above explained.
    """

    def get_JRC(aspect):
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
        df = pd.read_csv(req, sep=",", header=None, names=range(20)).dropna(how="all", axis=1)
        # drop rows with nan:
        df = df.dropna().reset_index(drop=True)
        # set header to first column
        header = df.iloc[0]
        df = df.iloc[1:, :]
        df.columns = header
        return df

    # CelestialDirections are "north", "south", "east", "west":
    # Orientation (azimuth) angle of the (fixed) plane, 0=south, 90=west, -90=east. Not relevant for tracking planes.
    for CelestialDirection in ["south", "east", "west", "north"]:
        if CelestialDirection == "south":
            aspect = 0
            df_south = get_JRC(aspect)
        elif CelestialDirection == "east":
            aspect = -90
            df_east = get_JRC(aspect)
        elif CelestialDirection == "west":
            aspect = 90
            df_west = get_JRC(aspect)
        elif CelestialDirection == "north":
            aspect = -180
            df_north = get_JRC(aspect)
    bigFrame = pd.concat([df_south, df_east, df_west, df_north], axis=1)
    # create header with Multi Index:
    columnNames = df_north.columns.to_list()
    bigHeader = pd.MultiIndex.from_product([["south", "east", "west", "north"],
                                            columnNames],
                                           names=['CelestialDirection', 'columns'])
    bigFrame.columns = bigHeader
    return bigFrame


def getNutsCenter(nuts_id,
                  url_nuts0_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_0.geojson",
                  url_nuts1_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_1.geojson",
                  url_nuts2_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_2.geojson",
                  url_nuts3_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_3.geojson"):
    """
    This function returns the latitude and longitude of the center of a NUTS region.
    """
    nuts_id = nuts_id.strip()
    if len(nuts_id) == 2:
        url = url_nuts0_poly
    elif len(nuts_id) == 3:
        url = url_nuts1_poly
    elif len(nuts_id) == 4:
        url = url_nuts2_poly
    elif len(nuts_id) > 4:
        url = url_nuts3_poly
    else:
        assert False, f"Error could not identify nuts level for {nuts_id}"
    nuts = gpd.read_file(url)
    transformer = Transformer.from_crs(CRS("EPSG:3035"), CRS("EPSG:4326"))
    point = nuts[nuts.NUTS_ID == nuts_id].centroid.values[0]
    return transformer.transform(point.y, point.x)  # returns lat, lon

def calculate_hourly_COP(outsideTemperature):
    """
    This function calculates the hourly COP for space heating and DHW based on the table in the database which provides
    values for temperatures between -22 and 40 °C. Every value is interpolated between the temperature steps.
    Input must be the outside temperature vector for every hour of the year.
    Output are two numpy arrays with 8760 entries for every hours of the year.
    """
    # calculate COP of HP for heating and DHW:
    CONN = DB().create_Connection(CONS().RootDB)
    HeatPump_COP = DB().read_DataFrame(REG_Table().Sce_HeatPump_COPCurve, CONN)
    HeatPump_COP = HeatPump_COP[:-1]

    # create COP vector with linear interpolation:
    HeatPump_HourlyCOP = np.zeros((8760,))
    DHW_HourlyCOP = np.zeros((8760,))
    for index, temperature in enumerate(outsideTemperature):
        if temperature <= min(HeatPump_COP.TemperatureEnvironment):
            HeatPump_HourlyCOP[index] = min(HeatPump_COP.COP_SpaceHeating)
            DHW_HourlyCOP[index] = min(HeatPump_COP.COP_HotWater)
            continue
        if temperature >= 18:
            HeatPump_HourlyCOP[index] = max(HeatPump_COP.COP_SpaceHeating)
            DHW_HourlyCOP[index] = max(HeatPump_COP.COP_HotWater)
            continue
        for index2, temperatureEnvironment in enumerate(HeatPump_COP.TemperatureEnvironment):
            if temperatureEnvironment - temperature <= 0 and temperatureEnvironment - temperature >> -1:
                HeatPump_HourlyCOP[index] = HeatPump_COP.COP_SpaceHeating[index2] + \
                                            (HeatPump_COP.COP_SpaceHeating[index2 + 1] -
                                             HeatPump_COP.COP_SpaceHeating[
                                                 index2]) \
                                            * abs(temperatureEnvironment - temperature)
                DHW_HourlyCOP[index] = HeatPump_COP.COP_HotWater[index2] + \
                                       (HeatPump_COP.COP_HotWater[index2 + 1] - HeatPump_COP.COP_HotWater[index2]) * \
                                       abs(temperatureEnvironment - temperature)

    return HeatPump_HourlyCOP, DHW_HourlyCOP

if __name__ == "__main__":
    # example is austria
    # Mittelpunkt von NUTS0
    nuts0 = "AT"
    lat, lon = getNutsCenter(nuts0)
    df = get_solar_radiation_cilestial_direction(lat=lat, lon=lon)
