from C_Model_Operation.C1_REG import REG_Table
from A_Infrastructure.A2_DB import DB
import numpy as np
import pandas as pd
from A_Infrastructure.A1_CONS import CONS
import geopandas as gpd
from pyproj import CRS, Transformer

class TableGenerator:

    def __init__(self, conn):
        self.Conn = conn

    # ---------------------------------------
    # 1 Functions used to generate the tables
    # ---------------------------------------

    def gen_OBJ_ID_Table_1To1(self, target_table_name, table1):

        TargetTable_list = []
        TargetTable_columns = ["ID"]
        for table in [table1]:
            TargetTable_columns += list(table.keys())
        ID = 1
        for row1 in range(0, len(table1)):
            TargetTable_list.append([ID] + list(table1.iloc[row1].values))
            ID += 1
        DB().write_DataFrame(TargetTable_list, target_table_name, TargetTable_columns, self.Conn)

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
        DB().write_DataFrame(TargetTable_list, target_table_name, TargetTable_columns, self.Conn)

        return None

    def gen_Sce_ApplianceUseHours(self, OnDays):
        # use the cycles of technology and generates a yearly table with 365 days with 1 and 0

        Year = 365
        OffDays = Year - int(OnDays)
        rest = round(int(OnDays) / OffDays, 2)
        add = int(OnDays) / OffDays
        i = 1
        UseDays = []

        while i <= Year:
            if rest > 1:
                UseDays.append(1)
                rest = rest - 1
                i = i + 1
            elif rest < 1:
                UseDays.append(0)
                rest = rest + add
                i = i + 1

        TheorecitalOnHours = []
        for day in UseDays:
            if day == 1:
                TheorecitalOnHours.extend(
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

            elif day == 0:
                TheorecitalOnHours.extend(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        return TheorecitalOnHours  # returns list of UseHours with 8760 values

    # -------------------------
    # 2 Generate the OBJ tables
    # -------------------------

    def gen_OBJ_ID_Building(self):
        BuildingOption = DB().read_DataFrame(REG_Table().ID_BuildingOption, self.Conn)
        BuildingMassTemperature = DB().read_DataFrame(REG_Table().ID_BuildingMassTemperature, self.Conn)
        GridInfrastructure = DB().read_DataFrame(REG_Table().ID_GridInfrastructure, self.Conn)
        self.gen_OBJ_ID_Table_3To1(REG_Table().Gen_OBJ_ID_Building,
                                   BuildingOption,
                                   BuildingMassTemperature,
                                   GridInfrastructure)
        return None

    def gen_OBJ_ID_ApplianceGroup(self):
        DishWasher = DB().read_DataFrame(REG_Table().ID_DishWasherType, self.Conn)
        Dryer = DB().read_DataFrame(REG_Table().ID_DryerType, self.Conn)
        WashingMachine = DB().read_DataFrame(REG_Table().ID_WashingMachineType, self.Conn)
        self.gen_OBJ_ID_Table_3To1(REG_Table().Gen_OBJ_ID_ApplianceGroup,
                                   DishWasher,
                                   Dryer,
                                   WashingMachine)
        return None

    def gen_OBJ_ID_SpaceHeating(self):
        SpaceHeatingBoiler = DB().read_DataFrame(REG_Table().ID_SpaceHeatingBoilerType, self.Conn)
        SpaceHeatingTank = DB().read_DataFrame(REG_Table().ID_SpaceHeatingTankType, self.Conn)
        SpaceHeatingPump = DB().read_DataFrame(REG_Table().ID_SpaceHeatingPumpType, self.Conn)
        self.gen_OBJ_ID_Table_3To1(REG_Table().Gen_OBJ_ID_SpaceHeating,
                                   SpaceHeatingBoiler,
                                   SpaceHeatingTank,
                                   SpaceHeatingPump)
        return None

    def gen_OBJ_ID_SpaceCooling(self):
        SpaceCooling = DB().read_DataFrame(REG_Table().ID_SpaceCoolingType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG_Table().Gen_OBJ_ID_SpaceCooling,
                                   SpaceCooling)
        return None

    def gen_OBJ_ID_HotWater(self):
        HotWaterBoiler = DB().read_DataFrame(REG_Table().ID_HotWaterBoilerType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG_Table().Gen_OBJ_ID_HotWater, HotWaterBoiler)
        return None

    def gen_OBJ_ID_PV(self):
        PV = DB().read_DataFrame(REG_Table().ID_PVType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG_Table().Gen_OBJ_ID_PV, PV)
        return None

    def gen_OBJ_ID_Battery(self):
        Battery = DB().read_DataFrame(REG_Table().ID_BatteryType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG_Table().Gen_OBJ_ID_Battery, Battery)
        return None

    def gen_OBJ_ID_ElectricVehicle(self):
        ElectricVehicle = DB().read_DataFrame(REG_Table().ID_ElectricVehicleType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG_Table().Gen_OBJ_ID_ElectricVehicle, ElectricVehicle)
        return None

    def gen_OBJ_ID_Household(self):
        """
        Creates big table for optimization with everthing included.
        """

        Country = DB().read_DataFrame(REG_Table().ID_Country, self.Conn)
        HouseholdType = DB().read_DataFrame(REG_Table().ID_HouseholdType, self.Conn)
        AgeGroup = DB().read_DataFrame(REG_Table().ID_AgeGroup, self.Conn)
        Building = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Building, self.Conn)
        ApplianceGroup = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_ApplianceGroup, self.Conn)
        SpaceHeating = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_SpaceHeating, self.Conn)
        SpaceCooling = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_SpaceCooling, self.Conn)
        HotWater = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_HotWater, self.Conn)
        PV = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_PV, self.Conn)
        Battery = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Battery, self.Conn)
        ElectricVehicle = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_ElectricVehicle, self.Conn)

        TargetTable_list = []
        TargetTable_columns = ["ID"]
        for table in [Country, HouseholdType, AgeGroup]:
            TargetTable_columns += list(table.keys())
        TargetTable_columns += ["ID_Building", "ID_ApplianceGroup", "ID_SpaceHeating", "ID_SpaceCooling",
                                "ID_HotWater", "ID_PV", "ID_Battery", "ID_ElectricVehicle"]
        ID = 1
        TotalRounds = len(Country) * len(HouseholdType) * len(AgeGroup) * len(Building) * \
                      len(ApplianceGroup) * len(SpaceHeating) * len(SpaceCooling) * len(HotWater) * \
                      len(PV) * len(Battery) * len(ElectricVehicle)

        for row1 in range(0, len(Country)):
            for row2 in range(0, len(HouseholdType)):
                for row3 in range(0, len(AgeGroup)):
                    for row4 in range(0, len(Building)):
                        for row5 in range(0, len(ApplianceGroup)):
                            for row6 in range(0, len(SpaceHeating)):
                                for row7 in range(0, len(SpaceCooling)):
                                    for row8 in range(0, len(HotWater)):
                                        for row9 in range(0, len(PV)):
                                            for row10 in range(0, len(Battery)):
                                                for row11 in range(0, len(ElectricVehicle)):
                                                    TargetTable_list.append([ID] +
                                                                            list(Country.iloc[row1].values) +
                                                                            list(HouseholdType.iloc[row2].values) +
                                                                            list(AgeGroup.iloc[row3].values) +
                                                                            [Building.iloc[row4]["ID"]] +
                                                                            [ApplianceGroup.iloc[row5]["ID"]] +
                                                                            [SpaceHeating.iloc[row6]["ID"]] +
                                                                            [SpaceCooling.iloc[row7]["ID"]] +
                                                                            [HotWater.iloc[row8]["ID"]] +
                                                                            [PV.iloc[row9]["ID"]] +
                                                                            [Battery.iloc[row10]["ID"]] +
                                                                            [ElectricVehicle.iloc[row11]["ID"]])
                                                    print("Round: " + str(ID) + "/" + str(TotalRounds))
                                                    ID += 1
        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_OBJ_ID_Household, TargetTable_columns, self.Conn)

    # ------------------------------
    # 3 Generate the scenario tables
    # ------------------------------

    def gen_Sce_Demand_DishWasherHours(self):

        self.Demand_DishWasher = DB().read_DataFrame(REG_Table().Sce_Demand_DishWasher, self.Conn)
        Demand_DishWasher = self.Demand_DishWasher
        Cycle = Demand_DishWasher.DishWasherCycle
        print('this is the cycle' + str(Cycle))
        hours = self.gen_Sce_ApplianceUseHours(Cycle)
        print(hours)
        TargetTable_list = []
        for hour in range(0, len(hours)):
            TargetTable_list.append([hour + 1, hours[hour]])

        TargetTable_columns = ["ID_Hour", "DishWasherHours"]
        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_DishWasherHours, TargetTable_columns, self.Conn)
        pass

    def gen_Sce_Demand_WashingMachineHours(self):
        # trockner wird nicht extra ausgegeben weil er an waschmaschine hängt

        self.Demand_WashingMachine = DB().read_DataFrame(REG_Table().Sce_Demand_WashingMachine, self.Conn)
        Demand_WashingMachine = self.Demand_WashingMachine
        Cycle = Demand_WashingMachine.WashingMachineCycle
        print(Cycle)
        hours = self.gen_Sce_ApplianceUseHours(Cycle)
        print(hours)

        TargetTable_list = []
        for hour in range(0, len(hours)):
            TargetTable_list.append([hour + 1, hours[hour]])

        TargetTable_columns = ['ID_Hour', "WashingMachineHours"]
        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_WashingMachineHours, TargetTable_columns, self.Conn)
        pass

    def gen_Sce_ID_Environment(self):

        ElectricityPriceType = DB().read_DataFrame(REG_Table().Sce_ID_ElectricityPriceType, self.Conn)
        FeedinTariffType = DB().read_DataFrame(REG_Table().Sce_ID_FeedinTariffType, self.Conn)
        HotWaterProfileType = DB().read_DataFrame(REG_Table().Sce_ID_HotWaterProfileType, self.Conn)
        PhotovoltaicProfileType = DB().read_DataFrame(REG_Table().Sce_ID_PhotovoltaicProfileType, self.Conn)
        TargetTemperatureType = DB().read_DataFrame(REG_Table().Sce_ID_TargetTemperatureType, self.Conn)
        BaseElectricityProfileType = DB().read_DataFrame(REG_Table().Sce_ID_BaseElectricityProfileType, self.Conn)
        EnergyCostType = DB().read_DataFrame(REG_Table().Sce_ID_EnergyCostType, self.Conn)

        TargetTable_list = []

        TargetTable_columns = ["ID"]

        TargetTable_columns += ["ID_ElectricityPriceType", "ID_TargetTemperatureType", "ID_FeedinTariffType",
                                "ID_HotWaterProfileType", "ID_PhotovoltaicProfileType", "ID_BaseElectricityProfileType",
                                "ID_EnergyCostType"]

        ID = 1

        for row1 in range(0, len(ElectricityPriceType)):
            for row2 in range(0, len(TargetTemperatureType)):
                for row3 in range(0, len(FeedinTariffType)):
                    for row4 in range(0, len(HotWaterProfileType)):
                        for row5 in range(0, len(PhotovoltaicProfileType)):
                            for row6 in range(0, len(BaseElectricityProfileType)):
                                for row7 in range(0, len(EnergyCostType)):
                                    TargetTable_list.append([ID] +
                                                            [ElectricityPriceType.iloc[row1][
                                                                 "ID_ElectricityPriceType"]] +
                                                            [TargetTemperatureType.iloc[row2][
                                                                 "ID_TargetTemperatureType"]] +
                                                            [FeedinTariffType.iloc[row3]["ID_FeedinTariffType"]] +
                                                            [HotWaterProfileType.iloc[row4]["ID_HotWaterProfileType"]] +
                                                            [PhotovoltaicProfileType.iloc[row5][
                                                                 "ID_PhotovoltaicProfile"]] +
                                                            [BaseElectricityProfileType.iloc[row6][
                                                                 "ID_BaseElectricityProfileType"]] +
                                                            [EnergyCostType.iloc[row6]["ID_EnergyCostType"]]
                                                            )
                                ID += 1

        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_ID_Environment, TargetTable_columns, self.Conn)

    def gen_Sce_CarAtHomeHours(self):

        self.Demand_ElectricVehicleBehavior = DB().read_DataFrame(REG_Table().Sce_Demand_ElectricVehicleBehavior,
                                                                  self.Conn)
        ElectricVehicleBehavior = self.Demand_ElectricVehicleBehavior
        LeaveTime = int(ElectricVehicleBehavior.EVLeaveHomeClock)
        ArriveTime = int(ElectricVehicleBehavior.EVArriveHomeClock)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        TimeStructure = self.TimeStructure
        ID_DayType = TimeStructure.ID_DayType

        print(ID_DayType)
        print(type(ID_DayType))

        # list of ID_DayType has 8760 values, this is reduced to 365 values with the type of day

        j = 0
        DayType = []
        for i in ID_DayType:
            if i == 1:
                j = j + 1
                if j == 24:
                    DayType.append(1)
                    j = 0
            elif i == 2:
                j = j + 1
                if j == 24:
                    DayType.append(2)
                    j = 0
            elif i == 3:
                j = j + 1
                if j == 24:
                    DayType.append(3)
                    j = 0
        print(DayType)
        print(len(DayType))

        # generate from list daytype the hours of use of the car, weekend it is not used now

        CarStatus = []
        for i in DayType:
            if i == 1:
                for i in range(0, LeaveTime):
                    CarStatus.append(1)
                for i in range(LeaveTime, ArriveTime):
                    CarStatus.append(0)
                for i in range(ArriveTime, 24):
                    CarStatus.append((1))
            elif i == 2 or i == 3:
                for i in range(0, 24):
                    CarStatus.append(1)

        print(CarStatus)
        print(len(CarStatus))

        TargetTable_columns = ['ID_Hour', "CarAtHomeHours"]

        TargetTable_list = []
        for Hour in range(0, len(CarStatus)):
            TargetTable_list.append([Hour + 1, CarStatus[Hour]])

        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_CarAtHomeHours, TargetTable_columns, self.Conn)
        pass

    def gen_Sce_HeatPump_HourlyCOP(self):
        """
        This function calculates the hourly COP for space heating and DHW based on the table in the database which
        provides values for temperatures between -22 and 40 °C. Every value is interpolated between the temperature
        steps.
        Input must be the outside temperature vector for every hour of the year.
        Output are four numpy arrays with 8760 entries for every hours of the year.
        """
        # calculate COP of HP for heating and DHW:
        outsideTemperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn).Temperature.to_numpy()
        HeatPump_COP = DB().read_DataFrame(REG_Table().Sce_HeatPump_COPCurve, self.Conn)
        HeatPump_COP_Water = HeatPump_COP.loc[HeatPump_COP['ID_SpaceHeatingBoilerType'] == 2]  # outside heat changer to water
        HeatPump_COP_Air = HeatPump_COP.loc[HeatPump_COP['ID_SpaceHeatingBoilerType'] == 1]  # outside heat exchanger to air

        # create COP vector with linear interpolation:
        HeatPump_HourlyCOP_Air = np.zeros((8760,))
        DHW_HourlyCOP_Air = np.zeros((8760,))
        HeatPump_HourlyCOP_Water = np.full((8760,), HeatPump_COP_Water.COP_SpaceHeating)
        DHW_HourlyCOP_Water = np.full((8760,), HeatPump_COP_Water.COP_HotWater)
        for index, temperature in enumerate(outsideTemperature):
            if temperature <= min(HeatPump_COP_Air.TemperatureEnvironment):
                HeatPump_HourlyCOP_Air[index] = min(HeatPump_COP_Air.COP_SpaceHeating)
                DHW_HourlyCOP_Air[index] = min(HeatPump_COP_Air.COP_HotWater)
                continue
            if temperature >= 18:  # because the COP curve is capped above 18°C
                HeatPump_HourlyCOP_Air[index] = max(HeatPump_COP_Air.COP_SpaceHeating)
                DHW_HourlyCOP_Air[index] = max(HeatPump_COP_Air.COP_HotWater)
                continue
            for index2, temperatureEnvironment in enumerate(HeatPump_COP_Air.TemperatureEnvironment):
                if temperatureEnvironment - temperature <= 0 and temperatureEnvironment - temperature > -1:

                    HeatPump_HourlyCOP_Air[index] = HeatPump_COP_Air.COP_SpaceHeating[index2] + \
                                                (HeatPump_COP_Air.COP_SpaceHeating[index2 + 1] -
                                                 HeatPump_COP_Air.COP_SpaceHeating[
                                                     index2]) \
                                                * abs(temperatureEnvironment - temperature)

                    DHW_HourlyCOP_Air[index] = HeatPump_COP_Air.COP_HotWater[index2] + \
                                           (HeatPump_COP_Air.COP_HotWater[index2 + 1] - HeatPump_COP_Air.COP_HotWater[index2]) * \
                                           abs(temperatureEnvironment - temperature)

        TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        ID_Hour = TimeStructure.ID_Hour
        ID_SpaceHeatingBoilerTypeAir = list([1]) * len(ID_Hour)  # outside heat changer to air
        ID_SpaceHeatingBoilerTypeWater = list([2]) * len(ID_Hour)  # outside heat changer to water

        # combine list of both types for DB
        ID_SpaceHeatingBoilerType = ID_SpaceHeatingBoilerTypeAir + ID_SpaceHeatingBoilerTypeWater
        HeatPump_HourlyCOP_both = np.append(HeatPump_HourlyCOP_Air, HeatPump_HourlyCOP_Water)
        DHW_HourlyCOP_both = np.append(DHW_HourlyCOP_Air, DHW_HourlyCOP_Water)
        ID_Hour = np.append(ID_Hour, ID_Hour)
        Temperature = np.append(outsideTemperature, outsideTemperature)

        TargetTable_columns = ["ID_SpaceHeatingBoilerType",
                               "ID_Hour",
                               "Temperature",
                               "SpaceHeatingHourlyCOP",
                               "HotWaterHourlyCOP"]

        TargetTable = np.column_stack((ID_SpaceHeatingBoilerType,
                                    ID_Hour,
                                    Temperature,
                                    HeatPump_HourlyCOP_both,
                                    DHW_HourlyCOP_both))

        DB().write_DataFrame(TargetTable, REG_Table().Gen_Sce_HeatPump_HourlyCOP, TargetTable_columns, self.Conn)

    def gen_SolarRadiation_windows_and_outsideTemperature(self, nuts_id):  # TODO write for loop for multiple nuts IDs!
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


        def get_JRC(aspect, startyear, endyear):
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

        startyear = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn).ID_Year.iloc[0]
        endyear = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn).ID_Year.iloc[-1]
        lat, lon = getNutsCenter(nuts_id)
        # CelestialDirections are "north", "south", "east", "west":
        # Orientation (azimuth) angle of the (fixed) plane, 0=south, 90=west, -90=east. Not relevant for tracking planes
        for CelestialDirection in ["south", "east", "west", "north"]:
            if CelestialDirection == "south":
                aspect = 0
                df_south = get_JRC(aspect, startyear, endyear)
                south_radiation = pd.to_numeric(df_south["Gb(i)"]) + pd.to_numeric(df_south["Gd(i)"])
            elif CelestialDirection == "east":
                aspect = -90
                df_east = get_JRC(aspect, startyear, endyear)
                east_radiation = pd.to_numeric(df_east["Gb(i)"]) + pd.to_numeric(df_east["Gd(i)"])
            elif CelestialDirection == "west":
                aspect = 90
                df_west = get_JRC(aspect, startyear, endyear)
                west_radiation = pd.to_numeric(df_west["Gb(i)"]) + pd.to_numeric(df_west["Gd(i)"])
            elif CelestialDirection == "north":
                aspect = -180
                df_north = get_JRC(aspect, startyear, endyear)
                north_radiation = pd.to_numeric(df_north["Gb(i)"]) + pd.to_numeric(df_north["Gd(i)"])

        ID_Timestructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        ID_Hour = ID_Timestructure.ID_Hour
        ID_Country = pd.Series([nuts_id] * len(ID_Hour), name="ID_Country")  # TODO maybe change to corresponding number??
        bigFrame = pd.concat([ID_Country,
                              ID_Hour,
                              south_radiation.reset_index(drop=True),
                              east_radiation.reset_index(drop=True),
                              west_radiation.reset_index(drop=True),
                              north_radiation.reset_index(drop=True)],
                             axis=1)
        bigFrame.columns = ["ID_Country", "ID_Hour", "south", "east", "west", "north"]
        # write radiation data to database
        DB().write_DataFrame(bigFrame, REG_Table().Gen_Sce_Weather_Radiation_SkyDirections, bigFrame.columns, self.Conn)

        ## write table for outside temperature:
        outsideTemperature = pd.to_numeric(df_south["T2m"].reset_index(drop=True).rename("Temperature"))
        TemperatureFrame = pd.concat([ID_Country, ID_Timestructure, outsideTemperature], axis=1)
        # write temperature data to database
        DB().write_DataFrame(TemperatureFrame, REG_Table().Sce_Weather_Temperature, TemperatureFrame.columns, self.Conn)

    def run(self):
        # these have to be run before ID Household
        self.gen_OBJ_ID_Building()
        self.gen_OBJ_ID_ApplianceGroup()
        self.gen_OBJ_ID_SpaceHeating()
        self.gen_OBJ_ID_SpaceCooling()
        self.gen_OBJ_ID_HotWater()
        self.gen_OBJ_ID_PV()
        self.gen_OBJ_ID_Battery()
        self.gen_OBJ_ID_ElectricVehicle()
        ##
        self.gen_OBJ_ID_Household()

        # are independent from the rest (usually are not changed, therefore don't run them every time)
        self.gen_Sce_Demand_DishWasherHours()
        self.gen_Sce_Demand_WashingMachineHours()

        # gibt an welche Spalten aus Sce_ Tables verwendet werden
        self.gen_Sce_ID_Environment()
        # self.gen_Sce_CarAtHomeHours()
        self.gen_Sce_HeatPump_HourlyCOP()

        # +Radiation
        pass

if __name__=="__main__":
    CONN = DB().create_Connection(CONS().RootDB)
    A = TableGenerator(CONN)


    NUTS_ID = "AT"
    A.gen_SolarRadiation_windows_and_outsideTemperature(nuts_id=NUTS_ID)

    A.gen_Sce_HeatPump_HourlyCOP()  # is dependent on gen_SolarRadiation_windows_and_outsideTemperature