
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB


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

    # to be developed
    def gen_Sce_ApplianceUseDays(self):

        pass

    # -------------------------
    # 2 Generate the OBJ tables
    # -------------------------

    def gen_OBJ_ID_Building(self):
        BuildingOption = DB().read_DataFrame(REG().ID_BuildingOption, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG().Gen_OBJ_ID_Building,
                                   BuildingOption)
        return None

    def gen_OBJ_ID_ApplianceGroup(self):
        DishWasher = DB().read_DataFrame(REG().ID_DishWasherType, self.Conn)
        Dryer = DB().read_DataFrame(REG().ID_DryerType, self.Conn)
        WashingMachine = DB().read_DataFrame(REG().ID_WashingMachineType, self.Conn)
        self.gen_OBJ_ID_Table_3To1(REG().Gen_OBJ_ID_ApplianceGroup,
                                   DishWasher,
                                   Dryer,
                                   WashingMachine)
        return None

    def gen_OBJ_ID_SpaceHeating(self):
        SpaceHeatingBoiler = DB().read_DataFrame(REG().ID_SpaceHeatingBoilerType, self.Conn)
        SpaceHeatingTank = DB().read_DataFrame(REG().ID_SpaceHeatingTankType, self.Conn)
        SpaceHeatingPump = DB().read_DataFrame(REG().ID_SpaceHeatingPumpType, self.Conn)
        self.gen_OBJ_ID_Table_3To1(REG().Gen_OBJ_ID_SpaceHeating,
                                   SpaceHeatingBoiler,
                                   SpaceHeatingTank,
                                   SpaceHeatingPump)
        return None

    def gen_OBJ_ID_SpaceCooling(self):
        SpaceCooling = DB().read_DataFrame(REG().ID_SpaceCoolingType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG().Gen_OBJ_ID_SpaceCooling,
                                   SpaceCooling)
        return None

    def gen_OBJ_ID_HotWater(self):
        HotWaterBoiler = DB().read_DataFrame(REG().ID_HotWaterBoilerType, self.Conn)
        HotWaterTank = DB().read_DataFrame(REG().ID_HotWaterTankType, self.Conn)
        self.gen_OBJ_ID_Table_2To1(REG().Gen_OBJ_ID_HotWater,
                                   HotWaterBoiler,
                                   HotWaterTank)
        return None

    def gen_OBJ_ID_PV(self):
        PV = DB().read_DataFrame(REG().ID_PVType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG().Gen_OBJ_ID_PV,
                                   PV)
        return None

    def gen_OBJ_ID_Battery(self):
        Battery = DB().read_DataFrame(REG().ID_BatteryType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG().Gen_OBJ_ID_Battery,
                                   Battery)
        return None

    def gen_OBJ_ID_ElectricVehicle(self):
        ElectricVehicle = DB().read_DataFrame(REG().ID_ElectricVehicleType, self.Conn)
        self.gen_OBJ_ID_Table_1To1(REG().Gen_OBJ_ID_ElectricVehicle,
                                   ElectricVehicle)
        return None

    def gen_OBJ_ID_Household(self):

        Country = DB().read_DataFrame(REG().ID_Country, self.Conn)
        HouseholdType = DB().read_DataFrame(REG().ID_HouseholdType, self.Conn)
        AgeGroup = DB().read_DataFrame(REG().ID_AgeGroup, self.Conn)
        Building = DB().read_DataFrame(REG().Gen_OBJ_ID_Building, self.Conn)
        ApplianceGroup = DB().read_DataFrame(REG().Gen_OBJ_ID_ApplianceGroup, self.Conn)
        SpaceHeating = DB().read_DataFrame(REG().Gen_OBJ_ID_SpaceHeating, self.Conn)
        SpaceCooling = DB().read_DataFrame(REG().Gen_OBJ_ID_SpaceCooling, self.Conn)
        HotWater = DB().read_DataFrame(REG().Gen_OBJ_ID_HotWater, self.Conn)
        PV = DB().read_DataFrame(REG().Gen_OBJ_ID_PV, self.Conn)
        Battery = DB().read_DataFrame(REG().Gen_OBJ_ID_Battery, self.Conn)
        ElectricVehicle = DB().read_DataFrame(REG().Gen_OBJ_ID_ElectricVehicle, self.Conn)

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
        DB().write_DataFrame(TargetTable_list, REG().Gen_OBJ_ID_Household, TargetTable_columns, self.Conn)

    # ------------------------------
    # 3 Generate the scenario tables
    # ------------------------------

    # to be developed
    def gen_Sce_Demand_DishWasherUseDays(self):

        pass

    # to be developed
    def gen_Sce_Demand_DryerUseDays(self):

        pass

    # to be developed
    def gen_Sce_Demand_WashingMachineUseDays(self):

        pass





    def run(self):
        # self.gen_OBJ_ID_Building()
        # self.gen_OBJ_ID_ApplianceGroup()
        # self.gen_OBJ_ID_SpaceHeating()
        # self.gen_OBJ_ID_SpaceCooling()
        self.gen_OBJ_ID_HotWater()
        # self.gen_OBJ_ID_PV()
        # self.gen_OBJ_ID_Battery()
        # self.gen_OBJ_ID_ElectricVehicle()
        # self.gen_OBJ_ID_Household()
        pass



