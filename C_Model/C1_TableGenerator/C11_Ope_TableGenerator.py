from A_Infrastructure.A2_REG import REG_Table
from A_Infrastructure.A3_DB import DB


class Ope_TableGenerator:

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

        TargetTable_columns += ["ID_ElectricityPriceType","ID_TargetTemperatureType", "ID_FeedinTariffType",
                                "ID_HotWaterProfileType", "ID_PhotovoltaicProfileType", "ID_BaseElectricityProfileType", "ID_EnergyCostType"]

        ID = 1

        for row1 in range(0, len(ElectricityPriceType)):
            for row2 in range(0, len(TargetTemperatureType)):
                for row3 in range(0, len(FeedinTariffType)):
                    for row4 in range(0, len(HotWaterProfileType)):
                        for row5 in range(0, len(PhotovoltaicProfileType)):
                            for row6 in range(0, len(BaseElectricityProfileType)):
                                for row7 in range(0, len(EnergyCostType)):

                                    TargetTable_list.append([ID] +
                                                        [ElectricityPriceType.iloc[row1]["ID_ElectricityPriceType"]] +
                                                        [TargetTemperatureType.iloc[row2]["ID_TargetTemperatureType"]] +
                                                        [FeedinTariffType.iloc[row3]["ID_FeedinTariffType"]] +
                                                        [HotWaterProfileType.iloc[row4]["ID_HotWaterProfileType"]] +
                                                        [PhotovoltaicProfileType.iloc[row5]["ID_PhotovoltaicProfile"]] +
                                                        [BaseElectricityProfileType.iloc[row6]["ID_BaseElectricityProfileType"]]+
                                                        [EnergyCostType.iloc[row6]["ID_EnergyCostType"]]
                                                            )
                                ID += 1

        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_ID_Environment, TargetTable_columns, self.Conn)

    def gen_Sce_CarAtHomeHours(self):

        self.Demand_ElectricVehicleBehavior = DB().read_DataFrame(REG_Table().Sce_Demand_ElectricVehicleBehavior, self.Conn)
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
                for i in range(0,24):
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
        self.Sce_HeatPump_COPCurve = DB().read_DataFrame(REG_Table().Sce_HeatPump_COPCurve, self.Conn)
        HeatPump_COPCurve = DB().read_DataFrame(REG_Table().Sce_HeatPump_COPCurve, self.Conn)
        Temperature2COPCurve = self.Sce_HeatPump_COPCurve.TemperatureEnvironment

        SpaceHeating_Air_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==1].loc[:,'COP_SpaceHeating'].to_numpy()
        SpaceHeating_Water_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==2].loc[:,'COP_SpaceHeating'].to_numpy()
        HotWater_Air_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==1].loc[:,'COP_HotWater'].to_numpy()
        HotWater_Water_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==2].loc[:,'COP_HotWater'].to_numpy()

        self.Sce_Weather_Temperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn)
        Temperature = self.Sce_Weather_Temperature.Temperature

        # Hourly list of COP Air for Space Heating
        SpaceHeatingAirHourlyCOP = []
        j = 0

        for i in range(0, len(list(Temperature))):
            for j in range(0, len(list(SpaceHeating_Air_COPCurve))):
                if Temperature2COPCurve[j] < Temperature[i]:
                    continue
                else:
                    SpaceHeatingAirHourlyCOP.append(SpaceHeating_Air_COPCurve[j])
                break

        # Hourly list of COP Air for HotWater
        HotWaterAirHourlyCOP = []
        j = 0

        for i in range(0, len(list(Temperature))):
            for j in range(0, len(list(HotWater_Air_COPCurve))):
                if Temperature2COPCurve[j] < Temperature[i]:
                    continue
                else:
                    HotWaterAirHourlyCOP.append(HotWater_Air_COPCurve[j])
                break

        # Hourly list of COP Water for Space Heating
        SpaceHeatingWaterHourlyCOP = list(SpaceHeating_Water_COPCurve) * len(Temperature)

        # Hourly list of COP Water for HotWater
        HotWaterWaterHourlyCOP = list(HotWater_Water_COPCurve) * len(Temperature)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        TimeStructure = self.TimeStructure
        ID_Hour = list(TimeStructure.ID_Hour)

        ID_SpaceHeatingBoilerTypeAir = list([1]) *len(Temperature)
        ID_SpaceHeatingBoilerTypeWater = list([2]) * len(Temperature)

        # combine list of both types for DB
        ID_SpaceHeatingBoilerType = ID_SpaceHeatingBoilerTypeAir + ID_SpaceHeatingBoilerTypeWater
        ID_Hour = ID_Hour + ID_Hour
        SpaceHeatingHourlyCOP = SpaceHeatingAirHourlyCOP + SpaceHeatingWaterHourlyCOP
        HotWaterHourlyCOP = HotWaterAirHourlyCOP + HotWaterWaterHourlyCOP
        Temperature = list(Temperature) + list(Temperature)


        TargetTable_columns = ["ID_SpaceHeatingBoilerType", "ID_Hour", "Temperature",
                               "SpaceHeatingHourlyCOP", "HotWaterHourlyCOP"]

        TargetTable_list = []
        for i in range(0, len(ID_Hour)):
            TargetTable_list.append([ID_SpaceHeatingBoilerType[i], ID_Hour[i], Temperature[i], SpaceHeatingHourlyCOP[i], HotWaterHourlyCOP[i]])

        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_HeatPump_HourlyCOP, TargetTable_columns, self.Conn)
        pass

    # this function is to be revised to add ID_Country
    def gen_Sce_HeatPump_HourlyCOP_test(self):

        Country = DB().read_DataFrame(REG_Table().ID_Country, self.Conn)

        self.Sce_HeatPump_COPCurve = DB().read_DataFrame(REG_Table().Sce_HeatPump_COPCurve, self.Conn)
        HeatPump_COPCurve = DB().read_DataFrame(REG_Table().Sce_HeatPump_COPCurve, self.Conn)
        Temperature2COPCurve = self.Sce_HeatPump_COPCurve.TemperatureEnvironment

        SpaceHeating_Air_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==1].loc[:,'COP_SpaceHeating'].to_numpy()
        SpaceHeating_Water_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==2].loc[:,'COP_SpaceHeating'].to_numpy()
        HotWater_Air_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==1].loc[:,'COP_HotWater'].to_numpy()
        HotWater_Water_COPCurve = HeatPump_COPCurve.loc[HeatPump_COPCurve['ID_SpaceHeatingBoilerType'] ==2].loc[:,'COP_HotWater'].to_numpy()

        self.Sce_Weather_Temperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn)
        Temperature = self.Sce_Weather_Temperature.Temperature

        # Hourly list of COP Air for Space Heating
        SpaceHeatingAirHourlyCOP = []
        j = 0

        for i in range(0, len(list(Temperature))):
            for j in range(0, len(list(SpaceHeating_Air_COPCurve))):
                if Temperature2COPCurve[j] < Temperature[i]:
                    continue
                else:
                    SpaceHeatingAirHourlyCOP.append(SpaceHeating_Air_COPCurve[j])
                break

        # Hourly list of COP Air for HotWater
        HotWaterAirHourlyCOP = []
        j = 0

        for i in range(0, len(list(Temperature))):
            for j in range(0, len(list(HotWater_Air_COPCurve))):
                if Temperature2COPCurve[j] < Temperature[i]:
                    continue
                else:
                    HotWaterAirHourlyCOP.append(HotWater_Air_COPCurve[j])
                break

        # Hourly list of COP Water for Space Heating
        SpaceHeatingWaterHourlyCOP = list(SpaceHeating_Water_COPCurve) * len(Temperature)

        # Hourly list of COP Water for HotWater
        HotWaterWaterHourlyCOP = list(HotWater_Water_COPCurve) * len(Temperature)

        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        TimeStructure = self.TimeStructure
        ID_Hour = list(TimeStructure.ID_Hour)

        ID_SpaceHeatingBoilerTypeAir = list([1]) *len(Temperature)
        ID_SpaceHeatingBoilerTypeWater = list([2]) * len(Temperature)

        # combine list of both types for DB
        ID_SpaceHeatingBoilerType = ID_SpaceHeatingBoilerTypeAir + ID_SpaceHeatingBoilerTypeWater
        ID_Hour = ID_Hour + ID_Hour
        SpaceHeatingHourlyCOP = SpaceHeatingAirHourlyCOP + SpaceHeatingWaterHourlyCOP
        HotWaterHourlyCOP = HotWaterAirHourlyCOP + HotWaterWaterHourlyCOP
        Temperature = list(Temperature) + list(Temperature)


        TargetTable_columns = ["ID_SpaceHeatingBoilerType", "ID_Hour", "Temperature",
                               "SpaceHeatingHourlyCOP", "HotWaterHourlyCOP"]

        TargetTable_list = []
        for i in range(0, len(ID_Hour)):
            TargetTable_list.append([ID_SpaceHeatingBoilerType[i], ID_Hour[i], Temperature[i], SpaceHeatingHourlyCOP[i], HotWaterHourlyCOP[i]])

        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_Sce_HeatPump_HourlyCOP, TargetTable_columns, self.Conn)
        pass


    def run(self):
        # self.gen_OBJ_ID_Building()
        # self.gen_OBJ_ID_ApplianceGroup()
        # self.gen_OBJ_ID_SpaceHeating()
        # self.gen_OBJ_ID_SpaceCooling()
        # self.gen_OBJ_ID_HotWater()
        # self.gen_OBJ_ID_PV()
        # self.gen_OBJ_ID_Battery()
        # self.gen_OBJ_ID_ElectricVehicle()
        # self.gen_OBJ_ID_Household()

        # self.gen_Sce_Demand_DishWasherHours()
        # self.gen_Sce_Demand_WashingMachineHours()

        # self.gen_Sce_ID_Environment()
        # self.gen_Sce_CarAtHomeHours()
        # self.gen_Sce_HeatPump_HourlyCOP()

        # +Radiation
        pass
