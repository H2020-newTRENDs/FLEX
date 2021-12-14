import urllib.error

from C_Model_Operation.C1_REG import REG_Table, REG_Var
from A_Infrastructure.A2_DB import DB
import numpy as np
import pandas as pd
from A_Infrastructure.A1_CONS import CONS
import geopandas as gpd
from pyproj import CRS, Transformer
from pathlib import Path
from abc import ABC, abstractmethod


class MotherTableGenerator(ABC):
    def __init__(self):
        self.conn = DB().create_Connection(CONS().RootDB)
        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.conn)
        self.id_hour = self.TimeStructure.ID_Hour.to_numpy()

    @abstractmethod
    def run(self):
        """runs all the functions of the class to generate the tables"""

    def gen_OBJ_ID_Table_1To1(self, target_table_name, table1):

        TargetTable_list = []
        TargetTable_columns = ["ID"]
        for table in [table1]:
            TargetTable_columns += list(table.keys())
        ID = 1
        for row1 in range(0, len(table1)):
            TargetTable_list.append([ID] + list(table1.iloc[row1].values))
            ID += 1
        DB().write_DataFrame(TargetTable_list, target_table_name, TargetTable_columns, self.conn)

        return None

    def gen_OBJ_ID_Table_2To1(self, target_table_name, table1, table2):

        TargetTable_list = []
        TargetTable_columns = ["ID"]
        for table in [table1, table2]:
            TargetTable_columns += list(table.keys())
        ID = 1
        for row1 in range(0, len(table1)):
            for row2 in range(0, len(table2)):
                TargetTable_list.append([ID] +
                                        list(table1.iloc[row1].values) +
                                        list(table2.iloc[row2].values))
                ID += 1
        DB().write_DataFrame(TargetTable_list, target_table_name, TargetTable_columns, self.Conn)

        return None

    def gen_OBJ_ID_Table_3To1(self, target_table_name, table1, table2, table3):

        TargetTable_list = []
        TargetTable_columns = ["ID"]
        for table in [table1, table2, table3]:
            TargetTable_columns += list(table.keys())
        ID = 1
        for row1 in range(0, len(table1)):
            for row2 in range(0, len(table2)):
                for row3 in range(0, len(table3)):
                    TargetTable_list.append([ID] +
                                            list(table1.iloc[row1].values) +
                                            list(table2.iloc[row2].values) +
                                            list(table3.iloc[row3].values))
                    ID += 1
        DB().write_DataFrame(TargetTable_list, target_table_name, TargetTable_columns, self.conn)
        return None


class TableGeneratorGenObj(MotherTableGenerator, ABC):
    """generates the GEN_OBJ tables through the ID_Tables"""

    def gen_OBJ_ID_Building(self) -> None:  # TODO create link to INVERT
        BuildingOption = DB().read_DataFrame(REG_Table().ID_BuildingOption,
                                             self.conn)  # TODO the function to create ID_Building option still has to be created
        BuildingMassTemperature = DB().read_DataFrame(REG_Table().ID_BuildingMassTemperature, self.conn)
        GridInfrastructure = DB().read_DataFrame(REG_Table().ID_GridInfrastructure, self.conn)
        self.gen_OBJ_ID_Table_3To1(REG_Table().Gen_OBJ_ID_Building,
                                   BuildingOption,
                                   BuildingMassTemperature,
                                   GridInfrastructure)
        return None

    def gen_OBJ_ID_ApplianceGroup(self):
        DishWasher = DB().read_DataFrame(REG_Table().ID_DishWasherType, self.conn)
        Dryer = DB().read_DataFrame(REG_Table().ID_DryerType, self.conn)
        WashingMachine = DB().read_DataFrame(REG_Table().ID_WashingMachineType, self.conn)
        self.gen_OBJ_ID_Table_3To1(REG_Table().Gen_OBJ_ID_ApplianceGroup,
                                   DishWasher,
                                   Dryer,
                                   WashingMachine)
        return None

    def gen_OBJ_ID_SpaceHeatingSystem(self) -> None:
        """creates the space heating system table in the Database"""
        heating_system = ["Air_HP", "Ground_HP"]  # TODO kan be given externally
        carnot_factor = [0.4, 0.35]  # for the heat pumps respectively
        columns = {REG_Var().ID_SpaceHeatingSystem: "INTEGER",
                   "Name_SpaceHeatingPumpType": "TEXT",
                   "HeatPumpMaximalThermalPower": "REAL",
                   "HeatPumpMaximalThermalPower_unit": "TEXT",
                   "HeatingElementPower": "REAL",
                   "HeatingElementPower_unit": "TEXT",
                   "CarnotEfficiencyFactor": "REAL"}

        table = np.column_stack([np.arange(1, len(heating_system) + 1),  # ID
                                 heating_system,  # Name_SpaceHeatingPumpType
                                 np.full((len(heating_system),), 15_000),  # HeatPumpMaximalThermalPower
                                 np.full((len(heating_system),), "W"),  # HeatPumpMaximalThermalPower_unit
                                 np.full((len(heating_system),), 7_500),  # HeatingElementPower
                                 np.full((len(heating_system),), "W"),  # HeatingElementPower_unit
                                 carnot_factor]  # CarnotEfficiencyFactor
                                )
        # save
        DB().write_DataFrame(table=table,
                             table_name=REG_Table().Gen_OBJ_ID_SpaceHeatingSystem,
                             conn=self.conn,
                             column_names=columns.keys(),
                             dtype=columns)

    def gen_OBJ_ID_heatingTank(self) -> None:
        """creates the space heating tank table in the Database"""
        tank_sizes = [500, 1_000, 1500]  # liter  TODO kan be given externally
        columns = {REG_Var().ID_SpaceHeatingTank: "INTEGER",
                   "TankSize": "REAL",
                   "TankSize_unit": "TEXT",
                   "TankSurfaceArea": "REAL",
                   "TankSurfaceArea_unit": "TEXT",
                   "TankLoss": "REAL",
                   "TankLoss_unit": "TEXT",
                   "TankStartTemperature": "REAL",
                   "TankMaximalTemperature": "REAL",
                   "TankMinimalTemperature": "REAL",
                   "TankSurroundingTemperature": "REAL"}
        # tank is designed as Cylinder with minimal surface area:
        radius = [(2 * volume / 1_000 / 4 / np.pi) ** (1. / 3) for volume in
                  tank_sizes]  # radius for cylinder with minimal surface
        surface_area = [2 * np.pi * r ** 2 + 2 * tank_sizes[i] / 1_000 / r for i, r in enumerate(radius)]
        table = np.column_stack([np.arange(1, len(tank_sizes) + 1),  # ID
                                 tank_sizes,  # tank size
                                 np.full((len(tank_sizes),), "kg"),  # TankSize_unit
                                 surface_area,  # TankSurfaceArea
                                 np.full((len(tank_sizes),), "m2"),  # TankSurfaceArea_unit
                                 np.full((len(tank_sizes),), 0.2),  # TankLoss
                                 np.full((len(tank_sizes),), "W/m2"),  # TankLoss_unit
                                 np.full((len(tank_sizes),), 28),  # TankStartTemperature
                                 np.full((len(tank_sizes),), 65),  # TankMaximalTemperature
                                 np.full((len(tank_sizes),), 28),  # TankMinimalTemperature
                                 np.full((len(tank_sizes),), 20)]  # TankSurroundingTemperature
                                )
        # save
        DB().write_DataFrame(table=table,
                             table_name=REG_Table().Gen_OBJ_ID_SpaceHeatingTank,
                             conn=self.conn,
                             column_names=columns.keys(),
                             dtype=columns)

    def gen_OBJ_ID_DHWTank(self) -> None:
        """creates the ID_DHWTank table in the Database"""
        tank_sizes = [200, 500, 1_000]  # liter  TODO kan be given externally
        columns = {REG_Var().ID_DHWTank: "INTEGER",
                   "DHWTankSize": "REAL",
                   "DHWTankSize_unit": "TEXT",
                   "DHWTankSurfaceArea": "REAL",
                   "DHWTankSurfaceArea_unit": "TEXT",
                   "DHWTankLoss": "REAL",
                   "DHWTankLoss_unit": "TEXT",
                   "DHWTankStartTemperature": "REAL",
                   "DHWTankMaximalTemperature": "REAL",
                   "DHWTankMinimalTemperature": "REAL",
                   "DHWTankSurroundingTemperature": "REAL"}
        # tank is designed as Cylinder with minimal surface area:
        radius = [(2 * volume / 1_000 / 4 / np.pi) ** (1. / 3) for volume in
                  tank_sizes]  # radius for cylinder with minimal surface
        surface_area = [2 * np.pi * r ** 2 + 2 * tank_sizes[i] / 1_000 / r for i, r in enumerate(radius)]
        table = np.column_stack([np.arange(1, len(tank_sizes) + 1),  # ID
                                 tank_sizes,  # tank size
                                 np.full((len(tank_sizes),), "kg"),  # TankSize_unit
                                 surface_area,  # TankSurfaceArea
                                 np.full((len(tank_sizes),), "m2"),  # TankSurfaceArea_unit
                                 np.full((len(tank_sizes),), 0.2),  # TankLoss
                                 np.full((len(tank_sizes),), "W/m2"),  # TankLoss_unit
                                 np.full((len(tank_sizes),), 28),  # TankStartTemperature
                                 np.full((len(tank_sizes),), 65),  # TankMaximalTemperature
                                 np.full((len(tank_sizes),), 28),  # TankMinimalTemperature
                                 np.full((len(tank_sizes),), 20)]  # TankSurroundingTemperature
                                )
        # save
        DB().write_DataFrame(table=table,
                             table_name=REG_Table().Gen_OBJ_ID_DHW_tank,
                             conn=self.conn,
                             column_names=columns.keys(),
                             dtype=columns)

    def gen_OBJ_ID_SpaceCooling(self) -> None:
        """creates the ID_SpaceCooling table in the Database"""
        cooling_power = [0, 10_000]  # W  TODO kan be given externally
        columns = {REG_Var().ID_SpaceCooling: "INTEGER",
                   "SpaceCoolingEfficiency": "REAL",
                   "SpaceCoolingPower": "REAL",
                   "SpaceCoolingPower_unit": "TEXT"}
        cooling_efficiency = np.full((len(cooling_power),), 3)  # TODO add other COPs?

        table = np.column_stack([np.arange(1, len(cooling_power) + 1),  # ID
                                 cooling_efficiency,  # SpaceCoolingEfficiency
                                 cooling_power,  # SpaceCoolingPower
                                 np.full((len(cooling_power),), "W")]  # SpaceCoolingPower_unit
                                )
        # save
        DB().write_DataFrame(table=table,
                             table_name=REG_Table().Gen_OBJ_ID_SpaceCooling,
                             conn=self.conn,
                             column_names=columns.keys(),
                             dtype=columns)

    def gen_OBJ_ID_Battery(self) -> None:
        """creates the ID_Battery table in the Database"""
        battery_capacity = [0, 10_000]  # W  TODO kan be given externally
        columns = {REG_Var().ID_Battery: "INTEGER",
                   "Capacity": "REAL",
                   "Capacity_unit": "TEXT",
                   "ChargeEfficiency": "REAL",
                   "DischargeEfficiency": "REAL",
                   "MaxChargePower": "REAL",
                   "MaxChargePower_unit": "TEXT",
                   "MaxDischargePower": "REAL",
                   "MaxDischargePower_unit": "TEXT"}

        table = np.column_stack([np.arange(1, len(battery_capacity) + 1),  # ID
                                 battery_capacity,  # Capacity
                                 np.full((len(battery_capacity),), "W"),  # Capacity_unit
                                 np.full((len(battery_capacity),), 0.95),  # ChargeEfficiency
                                 np.full((len(battery_capacity),), 0.95),  # DischargeEfficiency
                                 np.full((len(battery_capacity),), 4_500),  # MaxChargePower
                                 np.full((len(battery_capacity),), "W"),  # MaxChargePower_unit
                                 np.full((len(battery_capacity),), 4_500),  # MaxDischargePower
                                 np.full((len(battery_capacity),), "W")]  # MaxDischargePower_unit
                                )
        # save
        DB().write_DataFrame(table=table,
                             table_name=REG_Table().Gen_OBJ_ID_Battery,
                             conn=self.conn,
                             column_names=columns.keys(),
                             dtype=columns)

    def gen_OBJ_ID_PV(self):
        """saves the different PV types to the DB"""
        pv_power = [0, 5, 10]  # kWp  TODO kan be given externally
        columns = {REG_Var().ID_PV: "INTEGER",
                   "PVPower": "REAL",
                   "PVPower_unit": "TEXT"}

        table = np.column_stack([np.arange(1, len(pv_power) + 1),  # ID
                                 pv_power,  # PVPower
                                 np.full((len(pv_power),), "kWp")]  # MaxDischargePower_unit
                                )
        # save
        DB().write_DataFrame(table=table,
                             table_name=REG_Table().Gen_OBJ_ID_PV,
                             conn=self.conn,
                             column_names=columns.keys(),
                             dtype=columns)

    def gen_OBJ_ID_Household(self) -> None:
        """
        Creates big table for optimization with everthing included.
        """
        Country = DB().read_DataFrame(REG_Table().ID_Country, self.conn)
        HouseholdType = DB().read_DataFrame(REG_Table().ID_HouseholdType, self.conn)
        AgeGroup = DB().read_DataFrame(REG_Table().ID_AgeGroup, self.conn)
        Building = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Building, self.conn)
        # ApplianceGroup = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_ApplianceGroup, self.Conn)
        SpaceHeatingSystem = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_SpaceHeatingSystem, self.conn)
        SpaceHeatingTank = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_SpaceHeatingTank, self.conn)
        DHWTank = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_DHW_tank, self.conn)
        SpaceCooling = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_SpaceCooling, self.conn)
        # HotWater = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_HotWater, self.Conn)
        PV = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_PV, self.conn)
        Battery = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Battery, self.conn)

        input_table_list = [Country, HouseholdType, AgeGroup, Building, SpaceHeatingSystem, SpaceHeatingTank, DHWTank,
                            SpaceCooling, PV, Battery]
        total_rounds = 1
        for table in input_table_list:
            total_rounds *= len(table)
        TargetTable_list = []
        TargetTable_columns = ["ID"]
        for table in [Country, HouseholdType, AgeGroup]:
            TargetTable_columns += list(table.keys())
        TargetTable_columns += [REG_Var().ID_Building, REG_Var().ID_SpaceHeatingSystem, REG_Var().ID_SpaceHeatingTank,
                                REG_Var().ID_DHWTank, REG_Var().ID_SpaceCooling, REG_Var().ID_PV, REG_Var().ID_Battery]

        ID = 1
        # len(ApplianceGroup)

        for row1 in range(0, len(Country)):
            for row2 in range(0, len(HouseholdType)):
                for row3 in range(0, len(AgeGroup)):
                    for row4 in range(0, len(Building)):
                        for row5 in range(0, len(SpaceHeatingSystem)):
                            for row6 in range(0, len(SpaceHeatingTank)):
                                for row7 in range(0, len(DHWTank)):
                                    for row8 in range(0, len(SpaceCooling)):
                                        for row9 in range(0, len(PV)):
                                            for row10 in range(0, len(Battery)):
                                                TargetTable_list.append([ID] +
                                                                        list(Country.iloc[row1].values) +
                                                                        list(HouseholdType.iloc[row2].values) +
                                                                        list(AgeGroup.iloc[row3].values) +
                                                                        [Building.iloc[row4]["ID"]] +
                                                                        [SpaceHeatingSystem.iloc[row5][REG_Var().ID_SpaceHeatingSystem]] +
                                                                        [SpaceHeatingTank.iloc[row6][REG_Var().ID_SpaceHeatingTank]] +
                                                                        [DHWTank.iloc[row7][REG_Var().ID_DHWTank]] +
                                                                        [SpaceCooling.iloc[row8][REG_Var().ID_SpaceCooling]] +
                                                                        [PV.iloc[row9][REG_Var().ID_PV]] +
                                                                        [Battery.iloc[row10][REG_Var().ID_Battery]]
                                                                        )
                                                print("Round: " + str(ID) + "/" + str(total_rounds))
                                                ID += 1
        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_OBJ_ID_Household, TargetTable_columns, self.conn)

    def run(self):
        self.gen_OBJ_ID_Building()
        # self.gen_OBJ_ID_ApplianceGroup()
        self.gen_OBJ_ID_SpaceHeatingSystem()
        self.gen_OBJ_ID_heatingTank()
        self.gen_OBJ_ID_DHWTank()
        self.gen_OBJ_ID_SpaceCooling()
        self.gen_OBJ_ID_Battery()
        self.gen_OBJ_ID_PV()
        self.gen_OBJ_ID_Household()


class TableGeneratorID(MotherTableGenerator, ABC):
    def __init__(self):
        super().__init__()
        self.SolarRadiationList = []
        self.TemperatureList = []
        self.PVPowerList = []

    def getNutsCenter(self, region_id,
                      url_nuts0_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_0.geojson",
                      url_nuts1_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_1.geojson",
                      url_nuts2_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_2.geojson",
                      url_nuts3_poly="https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_60M_2021_3035_LEVL_3.geojson"):
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

    def get_PVGIS_data(self):
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
        # nuts_id_europe = DB().read_DataFrame("NUTS_IDs", conn=self.Conn)
        # table_name_temperature = "NUTS2_Temperature_Data"
        # table_name_radiation = "NUTS2_Radiation_Data"
        # table_name_PV = "NUTS2_PV_Data"
        # nuts = [nuts_id_europe.loc[:, "Code_2021"][i] for i in range(len(nuts_id_europe)) if
        #          len(nuts_id_europe.loc[:, "Code_2021"][i]) == 4]
        # NUTS2[128] muss noch gemacht werden! (funktioniert nicht für PV Power 10 kW

        # nuts3 for austria:
        nuts = ["AT111", "AT112", "AT113",
                "AT121", "AT122", "AT123", "AT124", "AT125", "AT126", "AT127",
                "AT130",
                "AT211", "AT212", "AT213",
                "AT221", "AT222", "AT223", "AT224", "AT225", "AT226",
                "AT311", "AT312", "AT313", "AT314", "AT315",
                "AT321", "AT322", "AT323",
                "AT331", "AT332", "AT333", "AT334", "AT335",
                "AT341", "AT342"]
        table_name_temperature = "NUTS3_Temperature_Data_AT"
        table_name_radiation = "NUTS3_Radiation_Data_AT"
        table_name_PV = "NUTS3_PV_Data_AT"

        def get_PV_generation(lat, lon, startyear, endyear, peakpower, nuts_id):
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
                PV_Profile = pd.to_numeric(df["P"]).to_numpy() / 1_000  # kW
                return PV_Profile
            except urllib.error.HTTPError:  # Error when nuts center is somewhere where there is no PVGIS data
                PV_Profile = None
                print("PV Data is not available for this location {}".format(nuts_id))
                return PV_Profile

        def get_temperature_and_solar_radiation(lat, lon, aspect, startyear, endyear, nuts_id):
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

        startyear = self.TimeStructure.ID_Year.iloc[0]
        endyear = self.TimeStructure.ID_Year.iloc[-1]
        for nuts_id in nuts:
            lat, lon = self.getNutsCenter(nuts_id)
            # CelestialDirections are "north", "south", "east", "west":
            # Orientation (azimuth) angle of the (fixed) plane, 0=south, 90=west, -90=east. Not relevant for tracking planes
            valid_nuts_id = True
            for CelestialDirection in ["south", "east", "west", "north"]:
                if CelestialDirection == "south":
                    aspect = 0
                    df_south = get_temperature_and_solar_radiation(lat, lon, aspect, startyear, endyear, nuts_id)
                    if df_south is None:
                        valid_nuts_id = False
                        break
                    south_radiation = pd.to_numeric(df_south["Gb(i)"]) + pd.to_numeric(df_south["Gd(i)"])
                elif CelestialDirection == "east":
                    aspect = -90
                    df_east = get_temperature_and_solar_radiation(lat, lon, aspect, startyear, endyear, nuts_id)
                    east_radiation = pd.to_numeric(df_east["Gb(i)"]) + pd.to_numeric(df_east["Gd(i)"])
                elif CelestialDirection == "west":
                    aspect = 90
                    df_west = get_temperature_and_solar_radiation(lat, lon, aspect, startyear, endyear, nuts_id)
                    west_radiation = pd.to_numeric(df_west["Gb(i)"]) + pd.to_numeric(df_west["Gd(i)"])
                elif CelestialDirection == "north":
                    aspect = -180
                    df_north = get_temperature_and_solar_radiation(lat, lon, aspect, startyear, endyear, nuts_id)
                    north_radiation = pd.to_numeric(df_north["Gb(i)"]) + pd.to_numeric(df_north["Gd(i)"])
            if not valid_nuts_id:
                continue
            ID_Country = pd.Series([nuts_id] * len(self.id_hour), name="ID_Country").to_numpy()
            SolarRadiationList = np.column_stack([ID_Country,
                                                  self.id_hour,
                                                  south_radiation.reset_index(drop=True).to_numpy(),
                                                  east_radiation.reset_index(drop=True).to_numpy(),
                                                  west_radiation.reset_index(drop=True).to_numpy(),
                                                  north_radiation.reset_index(drop=True).to_numpy()])
            # save solar radiation
            columns = {"ID_Country": "INTEGER",
                       "ID_Hour": "INTEGER",
                       "south": "REAL",
                       "east": "REAL",
                       "west": "REAL",
                       "north": "REAL"}
            DB().add_DataFrame(table=SolarRadiationList,
                               table_name=table_name_radiation,
                               column_names=columns.keys(),
                               conn=self.conn,
                               dtype=columns)

            dtypes = {"ID_Country": "TEXT",
                      "ID_Hour": "INTEGER",
                      "Temperature": "REAL"}
            # write table for outside temperature:
            outsideTemperature = pd.to_numeric(df_south["T2m"].reset_index(drop=True).rename("Temperature"))
            TemperatureList = np.column_stack([ID_Country,
                                               self.id_hour,
                                               outsideTemperature.to_numpy()])
            DB().add_DataFrame(table=TemperatureList,
                               table_name=table_name_temperature,
                               column_names=dtypes.keys(),
                               conn=self.conn,
                               dtype=dtypes)

            # PV profiles:
            PV_options = DB().read_DataFrame(REG_Table().ID_PVType, self.conn)
            columnNames = {"ID_Country": "INTEGER",
                           REG_Var().ID_PV: "INTEGER",
                           "ID_Hour": "INTEGER",
                           "PVPower": "REAL",
                           "Unit": "TEXT"}
            unit = np.full((len(self.id_hour),), "kW")
            for index, peakPower in enumerate(PV_options.PVPower):
                # for peakpower = 0 exception:
                if peakPower == 0:
                    continue  # 0-profile will not be saved
                else:
                    PVProfile = get_PV_generation(lat, lon, startyear, endyear, peakPower, nuts_id)

                pv_type = np.full((len(PVProfile),), int(PV_options.ID_PVType[index]))
                PVPowerList = np.column_stack([ID_Country, pv_type, self.id_hour, PVProfile, unit])

                DB().add_DataFrame(table=PVPowerList,
                                   table_name=table_name_PV,
                                   column_names=columnNames.keys(),
                                   conn=self.conn,
                                   dtype=columnNames)

            print("{} saved".format(nuts_id))

    def country_mean_PV_GIS(self, id_country):
        # weighted average over heat demand from hotmaps TODO get Hotmaps data through API
        heat_demand_AT = pd.read_excel(
            "C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Philipp/inputdata/AUT/AT_Heat_demand_total_NUTS3.xlsx",
            engine="openpyxl")
        total_heat_demand = heat_demand_AT.loc[:, "sum"].sum()
        # create numpy arrays that will be filled inside the loop:
        temperature_weighted_sum = np.zeros((8760, 1))
        radiation_weighted_sum = np.zeros((8760, 4))
        # get different PV id types:
        unique_PV_id_types = np.unique(DB().read_DataFrame("NUTS3_PV_Data_AT", self.conn, REG_Var().ID_PV).to_numpy())
        # dictionary for different PV types
        PV_weighted_sum = {id_pv: np.zeros((8760, 1)) for id_pv in unique_PV_id_types}
        for index, row in heat_demand_AT.iterrows():
            nuts_id = row["nuts_id"]
            heat_demand_of_specific_region = row["sum"]
            # Temperature
            temperature_profile = DB().read_DataFrame("NUTS3_Temperature_Data_AT",
                                                      self.conn,
                                                      "Temperature",
                                                      ID_Country=nuts_id).to_numpy()
            temperature_weighted_sum += temperature_profile * heat_demand_of_specific_region / total_heat_demand

            # Solar radiation
            radiation_profile = DB().read_DataFrame("NUTS3_Radiation_Data_AT",
                                                    self.conn,
                                                    "south", "east", "west", "north",  # *columns
                                                    ID_Country=nuts_id).to_numpy()  # **kwargs
            radiation_weighted_sum += radiation_profile * heat_demand_of_specific_region / total_heat_demand

            # PV
            # iterate through different PV types and add each weighted profiles to corresponding dictionary index
            for id_pv in unique_PV_id_types:
                PV_profile = DB().read_DataFrame("NUTS3_PV_Data_AT",
                                                 self.conn,
                                                 "PVPower",  # *columns
                                                 ID_Country=nuts_id, ID_PV=id_pv).to_numpy()  # **kwargs
                PV_weighted_sum[id_pv] += PV_profile * heat_demand_of_specific_region / total_heat_demand

        # create table for saving to DB:
        # Temperature
        temperature_table = np.column_stack([np.full((8760, 1), id_country),
                                             np.arange(8760),
                                             temperature_weighted_sum])
        temperature_columns = {"ID_Country": "TEXT",
                               "ID_Hour": "INTEGER",
                               "Temperature": "REAL"}
        # save weighted temperature average profile to database:
        DB().write_DataFrame(table=temperature_table,
                             table_name="Sce_Weather_Temperature",
                             column_names=temperature_columns.keys(),
                             conn=self.conn,
                             dtype=temperature_columns)

        # Radiation
        radiation_table = np.column_stack([np.full((8760, 1), id_country),
                                           np.arange(8760),
                                           radiation_weighted_sum])
        radiation_columns = {"ID_Country": "TEXT",
                             "ID_Hour": "INTEGER",
                             "south": "REAL", "east": "REAL", "west": "REAL", "north": "REAL"}
        # save to sql database:
        DB().write_DataFrame(table=radiation_table,
                             table_name="Gen_Sce_Weather_Radiation_SkyDirections",
                             column_names=radiation_columns.keys(),
                             conn=self.conn,
                             dtype=radiation_columns)

        # PV
        pv_columns = {"ID_Country": "TEXT",
                      REG_Var().ID_PV: "INTEGER",
                      "ID_Hour": "INTEGER",
                      "PVPower": "REAL",
                      "Unit": "TEXT"}
        # create single row on which tables are stacked up
        pv_table = np.zeros((1, len(pv_columns)))
        for key, values in PV_weighted_sum.items():
            # create the table for each pv type
            single_pv_table = np.column_stack([np.full((8760, 1), id_country),
                                               np.full((8760, 1), key),
                                               np.arange(8760),
                                               values,
                                               np.full((8760, 1), "kW")])
            # stack the tables from dictionary to one large table
            pv_table = np.vstack([pv_table, single_pv_table])

        # remove the first row (zeros) from the pv_table
        pv_table = pv_table[1:]
        # save to sql database:
        DB().write_DataFrame(table=pv_table,
                             table_name=REG_Table().Gen_Sce_PhotovoltaicProfile,
                             column_names=pv_columns.keys(),
                             conn=self.conn,
                             dtype=pv_columns)

        print("mean tables for country {} have been saved".format(id_country))

    def run(self):
        self.get_PVGIS_data()  # TODO make this versatile for other countries
        self.country_mean_PV_GIS("AT")


class TableGeneratorGenSce(MotherTableGenerator, ABC):

    def gen_Sce_Demand_DishWasherHours(self,
                                       NumberOfDishWasherProfiles):  # TODO if more than 1 dishwasher type is implemented, rewrite the function
        """
        creates Dish washer days where the dishwasher can be used on a random basis. Hours of the days where
        the Dishwasher has to be used are index = 1 and hours of days where the dishwasher is not used are indexed = 0.
        Dishwasher is not used during the night, only between 06:00 and 22:00.

        Dishwasher starting hours are randomly generated between 06:00 and 22:00 and a seperate dataframe is created
        """
        Demand_DishWasher = DB().read_DataFrame(REG_Table().Sce_Demand_DishWasher, self.conn)
        DishWasherDuration = int(Demand_DishWasher.DishWasherDuration)
        TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.conn)

        # Assumption: Dishwasher runs maximum once a day:
        UseDays = int(Demand_DishWasher.DishWasherCycle)
        TotalDays = TimeStructure.ID_Day.to_numpy()[-1]

        # create array with 0 or 1 for every hour a day and random choice of 0 and 1 but with correct numbers of 1
        # (UseDays)
        def rand_bin_array(usedays, totaldays):
            arr = np.zeros(totaldays)
            arr[:usedays] = 1
            np.random.shuffle(arr)
            arr = np.repeat(arr, 24, axis=0)
            return arr

        TargetTable = TimeStructure.ID_Hour.to_numpy()
        TargetTable = np.column_stack([TargetTable, TimeStructure.ID_DayHour.to_numpy()])
        TargetTable_columns = {"ID_Hour": "INTEGER",
                               "ID_DayHour": "INTEGER"}
        for i in range(NumberOfDishWasherProfiles + 1):
            TargetTable = np.column_stack([TargetTable, rand_bin_array(UseDays, TotalDays)])
            TargetTable_columns["DishWasherHours " + str(i)] = "INTEGER"

        # iterate through table and assign random values between 06:00 and 21:00 on UseDays:
        # this table is for reference scenarios or if the dishwasher is not optimized: First a random starting time is
        # specified and from there the timeslots for the duration of the dishwasher are set to 1:
        TargetTable2 = np.copy(TargetTable)
        for index in range(0, len(TargetTable2), 24):
            for column in range(2, TargetTable2.shape[1]):
                if TargetTable2[index, column] == 1:
                    HourOfTheDay = np.random.randint(low=6, high=22) - 1
                    TargetTable2[index + HourOfTheDay:index + HourOfTheDay + DishWasherDuration, column] = 1
                    TargetTable2[index:index + HourOfTheDay, column] = 0
                    TargetTable2[index + HourOfTheDay + DishWasherDuration:index + 24, column] = 0
        # write dataframe with starting hours to database:
        DB().write_DataFrame(TargetTable2, REG_Table().Gen_Sce_DishWasherStartingHours, TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)

        # set values to 0 when it is before 6 am:
        TargetTable[:, 2:][TargetTable[:, 1] < 6] = 0
        # set values to 0 when it is after 10 pm:
        TargetTable[:, 2:][TargetTable[:, 1] > 21] = 0

        # save arrays to database:
        DB().write_DataFrame(TargetTable, REG_Table().Gen_Sce_DishWasherHours, TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)

    def gen_Sce_Demand_WashingMachineHours(self, NumberOfWashingMachineProfiles):
        """
        same as dish washer function above
        washingmachine starting hours are between 06:00 and 20:00 because the dryer might be used afterwards
        Note: The last day is not being used! because in the optimization the dryer would run until
        """
        # Dryer has no own function because it will always be used after the washing machine
        Demand_WashingMachine = DB().read_DataFrame(REG_Table().Sce_Demand_WashingMachine, self.conn)
        WashingMachineDuration = int(Demand_WashingMachine.WashingMachineDuration)
        Demand_Dryer = DB().read_DataFrame(REG_Table().Sce_Demand_Dryer, self.conn)
        DryerDuration = int(Demand_Dryer.DryerDuration)
        TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.conn)

        # Assumption: Washing machine runs maximum once a day:
        UseDays = int(Demand_WashingMachine.WashingMachineCycle)
        TotalDays = TimeStructure.ID_Day.to_numpy()[-1] - 1  # last day is added later

        # create array with 0 or 1 for every hour a day and random choice of 0 and 1 but with correct numbers of 1
        # (UseDays)
        def rand_bin_array(usedays, totaldays):
            arr = np.zeros(totaldays)
            arr[:usedays] = 1
            np.random.shuffle(arr)
            arr = np.repeat(arr, 24, axis=0)
            return arr

        TargetTable = TimeStructure.ID_Hour.to_numpy()[:-24]
        TargetTable = np.column_stack([TargetTable, TimeStructure.ID_DayHour.to_numpy()[:-24]])
        TargetTable_columns = {"ID_Hour": "INTEGER",
                               "ID_DayHour": "INTEGER"}
        Dryer_columns = {"ID_Hour": "INTEGER",
                         "ID_DayHour": "INTEGER"}
        for i in range(NumberOfWashingMachineProfiles + 1):
            TargetTable = np.column_stack([TargetTable, rand_bin_array(UseDays, TotalDays)])
            TargetTable_columns["WashingMachineHours " + str(i)] = "INTEGER"
            Dryer_columns["DryerHours " + str(i)] = "INTEGER"

        # append the last day to the target table:
        lastDay_hours = TimeStructure.ID_Hour.to_numpy()[-24:]
        lastDay_IDHours = TimeStructure.ID_DayHour.to_numpy()[-24:]
        lastDay_zeros = np.zeros((len(lastDay_IDHours), TargetTable.shape[1] - 2))
        lastDay = np.column_stack([lastDay_hours, lastDay_IDHours, lastDay_zeros])

        # merge last day to target table:
        TargetTable = np.vstack([TargetTable, lastDay])

        # iterate through table and assign random values between 06:00 and 19:00 on UseDays (dryer has
        # to be used as well):
        # this table is for reference scenarios or if the dishwasher is not optimized:
        TargetTable_washmachine = np.copy(TargetTable)
        TargetTable_dryer = np.copy(TargetTable)  # for the dryer
        for index in range(0, len(TargetTable_washmachine), 24):
            for column in range(2, TargetTable_washmachine.shape[1]):
                if TargetTable_washmachine[index, column] == 1:
                    HourOfTheDay = np.random.randint(low=6,
                                                     high=21) - 1  # weil bei 0 zu zählen anfängt (Hour of day 6 ist 7 uhr)
                    TargetTable_washmachine[index + HourOfTheDay:index + HourOfTheDay + WashingMachineDuration,
                    column] = 1
                    TargetTable_washmachine[index:index + HourOfTheDay, column] = 0
                    TargetTable_washmachine[index + HourOfTheDay + WashingMachineDuration:index + 24, column] = 0

                    # Dryer always starts 1 hour after the washing machine:
                    TargetTable_dryer[
                    index + HourOfTheDay + WashingMachineDuration + 1:index + HourOfTheDay + WashingMachineDuration + 1 + DryerDuration,
                    column] = 1
                    TargetTable_dryer[index:index + HourOfTheDay + WashingMachineDuration + 1, column] = 0
                    TargetTable_dryer[index + HourOfTheDay + WashingMachineDuration + 1 + DryerDuration:index + 24,
                    column] = 0

        # write dataframe with starting hours to database:
        DB().write_DataFrame(TargetTable_washmachine, REG_Table().Gen_Sce_WashingMachineStartingHours,
                             TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)
        DB().write_DataFrame(TargetTable_dryer, REG_Table().Gen_Sce_DryerStartingHours, Dryer_columns.keys(),
                             self.conn, dtype=Dryer_columns)

        # set values to 0 when it is before 6 am:
        TargetTable[:, 2:][TargetTable[:, 1] < 6] = 0
        # set values to 0 when it is after 8 pm:
        TargetTable[:, 2:][TargetTable[:, 1] > 20] = 0
        # save starting days for optimization:
        DB().write_DataFrame(TargetTable, REG_Table().Gen_Sce_WashingMachineHours, TargetTable_columns.keys(),
                             self.conn, dtype=TargetTable_columns)

    def gen_sce_target_temperature(self):
        outsideTemperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.conn).Temperature.to_numpy()
        TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.conn)
        ID_Hour = TimeStructure.ID_Hour
        ID_DayHour = TimeStructure.ID_DayHour.to_numpy()
        # load min max indoor temperature from profile table
        targetTemperature = DB().read_DataFrame(REG_Table().Sce_ID_TargetTemperatureType, self.conn)

        T_min_young = targetTemperature.HeatingTargetTemperatureYoung
        T_max_young = targetTemperature.CoolingTargetTemperatureYoung

        T_min_old = targetTemperature.HeatingTargetTemperatureOld
        T_max_old = targetTemperature.CoolingTargetTemperatureOld

        # temperature with night reduction: lowering min temperature by 2°C from 22:00 to 06:00 every day:
        T_min_young_nightReduction = np.array([])
        T_min_old_nightReduction = np.array([])
        for hour in ID_DayHour:
            if hour <= 6:
                T_min_young_nightReduction = np.append(T_min_young_nightReduction, (float(T_min_young) - 2))
                T_min_old_nightReduction = np.append(T_min_old_nightReduction, (float(T_min_old) - 2))
            elif hour >= 22:
                T_min_young_nightReduction = np.append(T_min_young_nightReduction, (float(T_min_young) - 2))
                T_min_old_nightReduction = np.append(T_min_old_nightReduction, (float(T_min_old) - 2))
            else:
                T_min_young_nightReduction = np.append(T_min_young_nightReduction, float(T_min_young))
                T_min_old_nightReduction = np.append(T_min_old_nightReduction, float(T_min_old))

        # set temperature with smart system that adapts according to outside temperatuer to save energy:
        # the heating temperature will be calculated by a function, if the heating temperature is above 21°C,
        # it is capped. The maximum temperature will be the minimum temperature + 5°C in winter and when the minimum
        # temperature is above 22°C the maximum temperature is set to 27°C for cooling.
        # Fct: Tset = 20°C + (T_outside - 20°C + 12)/8
        T_min_smartHome = np.array([])
        T_max_smartHome = np.array([])
        for T_out in outsideTemperature:
            if 20 + (T_out - 20 + 12) / 8 <= 21:
                T_min_smartHome = np.append(T_min_smartHome, 20 + (T_out - 20 + 12) / 8)
                T_max_smartHome = np.append(T_max_smartHome, 20 + (T_out - 20 + 12) / 8 + 5)
            else:
                T_min_smartHome = np.append(T_min_smartHome, 21)
                T_max_smartHome = np.append(T_max_smartHome, 27)

        TargetFrame = np.column_stack([ID_DayHour,
                                       ID_Hour,
                                       [float(T_min_young)] * len(ID_Hour),
                                       [float(T_max_young)] * len(ID_Hour),
                                       [float(T_min_old)] * len(ID_Hour),
                                       [float(T_max_old)] * len(ID_Hour),
                                       T_min_young_nightReduction,
                                       T_min_old_nightReduction,
                                       T_min_smartHome,
                                       T_max_smartHome])
        column_names = {"ID_DayHour": "INTEGER",
                        "ID_Hour": "INTEGER",
                        "HeatingTargetTemperatureYoung": "REAL",
                        "CoolingTargetTemperatureYoung": "REAL",
                        "HeatingTargetTemperatureOld": "REAL",
                        "CoolingTargetTemperatureOld": "REAL",
                        "HeatingTargetTemperatureYoungNightReduction": "REAL",
                        "HeatingTargetTemperatureOldNightReduction": "REAL",
                        "HeatingTargetTemperatureSmartHome": "REAL",
                        "CoolingTargetTemperatureSmartHome": "REAL"}

        # write set temperature data to database
        DB().write_DataFrame(TargetFrame, REG_Table().Gen_Sce_TargetTemperature, column_names.keys(),
                             self.conn, dtype=column_names)

    def gen_Sce_AC_HourlyCOP(self):
        # constant COP of 3 is estimated:
        COP = 3
        COP_array = np.full((len(self.id_hour),), COP)
        TargetTable = np.column_stack([self.id_hour, COP_array])
        columns = {"ID_Hour": "INTEGER",
                   "ACHourlyCOP": "REAL"}
        DB().write_DataFrame(TargetTable, REG_Table().Gen_Sce_AC_HourlyCOP, columns.keys(), self.conn, dtype=columns)

    def gen_Sce_electricity_price(self):
        elec_price = pd.read_excel(
            "C:/Users/mascherbauer/PycharmProjects/NewTrends/Prosumager/_Philipp/inputdata/Elec_price_per_hour.xlsx",
            engine="openpyxl").to_numpy() / 10 + 15  # cent/kWh
        mean_price = elec_price.mean()
        max_price = elec_price.max()
        min_price = elec_price.min()
        median_price = np.median(elec_price)
        first_quantile = np.quantile(elec_price, 0.25)
        third_quantile = np.quantile(elec_price, 0.75)
        standard_deviation = np.std(elec_price)
        variance = np.var(elec_price)

        mean_price_vector = np.full((8760,), 20)  # cent/kWh

        variable_price_to_db = np.column_stack(
            [np.full((8760,), 1), np.full((8760,), "AT"), np.arange(8760), elec_price, np.full((8760,), "cent/kWh")])
        fixed_price_to_db = np.column_stack(
            [np.full((8760,), 2), np.full((8760,), "AT"), np.arange(8760), mean_price_vector,
             np.full((8760,), "cent/kWh")])
        price_to_db = np.vstack([variable_price_to_db, fixed_price_to_db])
        price_columns = {"ID_PriceType": "INT",
                         "ID_Country": "TEXT",
                         "ID_Hour": "INTEGER",
                         "ElectricityPrice": "REAL",
                         "Unit": "TEXT"}
        # save to database

        DB().write_DataFrame(price_to_db, "Gen_Sce_ElectricityProfile",
                             column_names=price_columns.keys(),
                             conn=self.conn,
                             dtype=price_columns)
        # FIT
        feed_in_tariff = np.column_stack(
            [np.full((8760,), 1), np.arange(8760), np.full((8760,), 7.67), np.full((8760,), "cent/kWh")])
        feed_in_columns = {"ID_FeedinTariffType": "INT",
                           "ID_Hour": "INTEGER",
                           "HourlyFeedinTariff": "REAL",
                           "HourlyFeedinTariff_unit": "TEXT"}

        DB().write_DataFrame(feed_in_tariff, "Sce_Price_HourlyFeedinTariff",
                             column_names=feed_in_columns.keys(),
                             conn=self.conn,
                             dtype=feed_in_columns)

    def gen_Sce_Demand_BaseElectricityProfile(self):
        baseload = pd.read_csv(Path().absolute().parent.resolve() / Path("_Philipp/inputdata/AUT/synthload2019.csv"),
                               sep=None, engine="python")
        baseload_h0 = baseload.loc[baseload["Typnummer"] == 1].set_index("Zeit", drop=True).drop(columns="Typnummer")
        baseload_h0.index = pd.to_datetime(baseload_h0.index)
        baseload_h0["Wert"] = pd.to_numeric(baseload_h0["Wert"].str.replace(",", "."))
        baseload_h0 = baseload_h0[3:].resample("1H").sum()
        baseload_h0 = baseload_h0.reset_index(drop=True).rename(columns={"Wert": "BaseElectricityProfile"})

        columns = {"BaseElectricityProfile": "REAL"}
        DB().write_DataFrame(table=baseload_h0, table_name="Sce_Demand_BaseElectricityProfile", conn=self.conn,
                             column_names=columns.keys(), dtype=columns)

    def gen_Sce_HotWaterProfile(self):
        hot_water = pd.read_excel(
            Path().absolute().parent.resolve() / Path("_Philipp/inputdata/AUT/Hot_water_profile.xlsx"),
            engine="openpyxl")
        hot_water_profile = np.column_stack([hot_water["Profile"].to_numpy(), np.full((8760,), "kWh")])
        columns = {"HotWater": "REAL", "Unit": "TEXT"}
        DB().write_DataFrame(hot_water_profile, "Gen_Sce_HotWaterProfile", columns.keys(), self.conn, dtype=columns)

    def gen_Sce_ID_Environment(self):

        ElectricityPriceType = DB().read_DataFrame(REG_Table().ID_ElectricityPrice, self.conn)
        FeedinTariffType = DB().read_DataFrame(REG_Table().Sce_ID_FeedinTariffType, self.conn)
        # HotWaterProfileType = DB().read_DataFrame(REG_Table().Sce_ID_HotWaterProfileType, self.Conn)
        # PhotovoltaicProfileType = DB().read_DataFrame(REG_Table().Sce_ID_PhotovoltaicProfileType, self.Conn)
        # TargetTemperatureType = DB().read_DataFrame(REG_Table().Sce_ID_TargetTemperatureType, self.Conn)
        # BaseElectricityProfileType = DB().read_DataFrame(REG_Table().Sce_ID_BaseElectricityProfileType, self.Conn)
        # EnergyCostType = DB().read_DataFrame(REG_Table().Sce_ID_EnergyCostType, self.Conn)

        TargetTable_list = []

        TargetTable_columns = ["ID"]

        # TargetTable_columns += ["ID_ElectricityPriceType", "ID_TargetTemperatureType", "ID_FeedinTariffType",
        #                         "ID_HotWaterProfileType", "ID_PhotovoltaicProfileType", "ID_BaseElectricityProfileType",
        #                         "ID_EnergyCostType"]
        TargetTable_columns += ["ID_ElectricityPriceType", "ID_FeedinTariffType"]

        ID = 1

        for row1 in range(0, len(ElectricityPriceType)):
            # for row2 in range(0, len(TargetTemperatureType)):
            for row3 in range(0, len(FeedinTariffType)):
                # for row4 in range(0, len(HotWaterProfileType)):
                #     for row5 in range(0, len(PhotovoltaicProfileType)):
                #         for row6 in range(0, len(BaseElectricityProfileType)):
                #             for row7 in range(0, len(EnergyCostType)):
                TargetTable_list.append([ID] +
                                        [ElectricityPriceType.iloc[row1][
                                             "ID_ElectricityPriceType"]] +
                                        # [TargetTemperatureType.iloc[row2][
                                        #      "ID_TargetTemperatureType"]] +
                                        [FeedinTariffType.iloc[row3]["ID_FeedinTariffType"]])  # +
                # [HotWaterProfileType.iloc[row4]["ID_HotWaterProfileType"]] +
                # [PhotovoltaicProfileType.iloc[row5][
                #      "ID_PhotovoltaicProfile"]] +
                # [BaseElectricityProfileType.iloc[row6][
                #      "ID_BaseElectricityProfileType"]] +
                # [EnergyCostType.iloc[row6]["ID_EnergyCostType"]]

                ID += 1

        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_ID_Environment, TargetTable_columns, self.conn)

    def run(self):
        self.gen_Sce_Demand_DishWasherHours(
            10)  # TODO this number should be externally set or we dont need the profiles anyways
        self.gen_Sce_Demand_WashingMachineHours(10)
        self.gen_sce_target_temperature()
        self.gen_Sce_AC_HourlyCOP()
        self.gen_Sce_electricity_price()
        self.gen_Sce_Demand_BaseElectricityProfile()
        self.gen_Sce_HotWaterProfile()
        self.gen_Sce_ID_Environment()


if __name__ == "__main__":
    ID_generator = TableGeneratorID()
    gensce_generator = TableGeneratorGenSce()
    genobj_generator = TableGeneratorGenObj()

    # ID_generator.run()
    # gensce_generator.run()
    genobj_generator.run()
