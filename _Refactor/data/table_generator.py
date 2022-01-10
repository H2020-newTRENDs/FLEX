

import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod
import sqlalchemy.types

from _Refactor.basic.db import DB
from _Refactor.core.household.components import Region, Person, Building, Appliance, Boiler, SpaceHeatingTank, \
    AirConditioner, HotWaterTank, PV, Battery, Vehicle, Behavior, Demand
from _Refactor.basic.reg import Table
import import_building_data


class MotherTableGenerator(ABC):

    def get_dtypes_dict(self, class_dict):
        dtypes_dict = {}
        for key in class_dict.keys():
            if "id" in key.lower():  # check if it is id parameter
                value = sqlalchemy.types.Integer
            elif "_unit" in key.lower() or "name" in key.lower():
                value = sqlalchemy.types.String
            else:
                value = sqlalchemy.types.Float
            dtypes_dict[key] = value
        return dtypes_dict

    @abstractmethod
    def run(self):
        """runs all the functions of the household to generate the tables"""

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


class HouseholdComponentGenerator(MotherTableGenerator):
    """generates the Household tables"""

    def calculate_cylindrical_area_from_volume(self, tank_sizes: float) -> float:
        """calculates the minimal cylindicral area of a tank with a give volume"""
        radius = [(2 * volume / 1_000 / 4 / np.pi) ** (1. / 3) for volume in tank_sizes]
        surface_area = [2 * np.pi * r ** 2 + 2 * tank_sizes[i] / 1_000 / r for i, r in enumerate(radius)]
        return surface_area

    def gen_OBJ_ID_ApplianceGroup(self):
        DishWasher = DB().read_DataFrame(REG_Table().ID_DishWasherType, self.conn)
        Dryer = DB().read_DataFrame(REG_Table().ID_DryerType, self.conn)
        WashingMachine = DB().read_DataFrame(REG_Table().ID_WashingMachineType, self.conn)
        self.gen_OBJ_ID_Table_3To1(REG_Table().Gen_OBJ_ID_ApplianceGroup,
                                   DishWasher,
                                   Dryer,
                                   WashingMachine)

    def create_household_building(self) -> None:  # TODO create link to INVERT
        """reads building excel table and stores it to root"""
        building_mass_temperature_start = 15  # °C
        building_mass_temperature_max = 60  # °C
        grid_power_max = 21_000  # W
        building_data = import_building_data.load_building_data_from_excel()
        # rename columns:
        building_data = building_data.rename(
            columns={"index": "ID_Building",
                     "horizontal_shading_building": "horizontal_shading",
                     "areawindows": "area_windows",
                     "area_suitable_solar": "area_suitable_for_solar",
                     "average_effective_area_wind_west_east_red_cool": "effective_window_area_west_east",
                     "average_effective_area_wind_north_red_cool": "effective_window_area_north",
                     "average_effective_area_wind_south_red_cool": "effective_window_area_south",
                     "spec_int_gains_cool_watt": "internal_gains"
                     })
        building_data["building_mass_temperature_start"] = building_mass_temperature_start
        building_data["building_mass_temperature_max"] = building_mass_temperature_max
        building_data["grid_power_max"] = grid_power_max
        building_data["grid_power_max_unit"] = "W"
        assert list(Building().__dict__.keys()).sort() == list(building_data.columns).sort()
        columns = self.get_dtypes_dict(Building().__dict__)
        assert list(columns.keys()).sort() == list(building_data.keys()).sort()
        # save
        DB().write_dataframe(table_name=Table().boiler,
                             data_frame=building_data,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_household_boiler(self) -> None:
        """creates the space heating system table in the Database"""
        heating_system = ["Air_HP", "Ground_HP"]  # TODO kan be given externally
        carnot_factor = [0.4, 0.35]  # for the heat pumps respectively
        columns = self.get_dtypes_dict(Boiler().__dict__)
        table_dict = {"ID_Boiler": np.arange(1, len(heating_system) + 1),
                      "name": heating_system,
                      "thermal_power_max": np.full((len(heating_system),), 15_000),
                      "thermal_power_max_unit": np.full((len(heating_system),), "W"),
                      "heating_element_power": np.full((len(heating_system),), 7_500),
                      "heating_element_power_unit": np.full((len(heating_system),), "W"),
                      "carnot_efficiency_factor": carnot_factor
                      }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().boiler,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_household_space_heating_tank(self) -> None:
        """creates the space heating tank table in the Database"""
        tank_sizes = [500, 1_000, 1500]  # liter  TODO kan be given externally
        columns = self.get_dtypes_dict(SpaceHeatingTank().__dict__)
        # tank is designed as Cylinder with minimal surface area:
        surface_area = self.calculate_cylindrical_area_from_volume(tank_sizes)
        table_dict = {"ID_SpaceHeatingTank": np.arange(1, len(tank_sizes) + 1),  # ID
                      "size": tank_sizes,  # tank size
                      "size_unit": np.full((len(tank_sizes),), "kg"),  # TankSize_unit
                      "surface_area": surface_area,  # TankSurfaceArea
                      "surface_area_unit": np.full((len(tank_sizes),), "m2"),  # TankSurfaceArea_unit
                      "loss": np.full((len(tank_sizes),), 0.2),  # TankLoss
                      "loss_unit": np.full((len(tank_sizes),), "W/m2"),  # TankLoss_unit
                      "temperature_start": np.full((len(tank_sizes),), 28),  # TankStartTemperature
                      "temperature_max": np.full((len(tank_sizes),), 65),  # TankMaximalTemperature
                      "temperature_min": np.full((len(tank_sizes),), 28),  # TankMinimalTemperature
                      "temperature_surrounding": np.full((len(tank_sizes),), 20)  # TankSurroundingTemperature
                      }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().space_heating_tank,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_household_hot_water_tank(self) -> None:
        """creates the ID_DHWTank table in the Database"""
        tank_sizes = [200, 500, 1_000]  # liter  TODO kan be given externally
        columns = self.get_dtypes_dict(HotWaterTank().__dict__)
        # tank is designed as Cylinder with minimal surface area:
        surface_area = self.calculate_cylindrical_area_from_volume(tank_sizes)
        table_dict = {"ID_HotWaterTank": np.arange(1, len(tank_sizes) + 1),  # ID
                      "size": tank_sizes,  # tank size
                      "size_unit": np.full((len(tank_sizes),), "kg"),  # TankSize_unit
                      "surface_area": surface_area,  # TankSurfaceArea
                      "surface_area_unit": np.full((len(tank_sizes),), "m2"),  # TankSurfaceArea_unit
                      "loss": np.full((len(tank_sizes),), 0.2),  # TankLoss
                      "loss_unit": np.full((len(tank_sizes),), "W/m2"),  # TankLoss_unit
                      "temperature_start": np.full((len(tank_sizes),), 28),  # TankStartTemperature
                      "temperature_max": np.full((len(tank_sizes),), 65),  # TankMaximalTemperature
                      "temperature_min": np.full((len(tank_sizes),), 28),  # TankMinimalTemperature
                      "temperature_surrounding": np.full((len(tank_sizes),), 20)  # TankSurroundingTemperature
                      }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().hot_water_tank,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_household_air_conditioner(self) -> None:
        """creates the ID_SpaceCooling table in the Database"""
        cooling_power = [0, 10_000]  # W  TODO kan be given externally
        cooling_efficiency = np.full((len(cooling_power),), 3)  # TODO add other COPs?
        columns = self.get_dtypes_dict(AirConditioner().__dict__)
        table_dict = {"ID_AirConditioner": np.arange(1, len(cooling_power) + 1),
                      "efficiency": cooling_efficiency,
                      "power": cooling_power,
                      "power_unit": np.full((len(cooling_power),), "W")
                      }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().air_conditioner,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_household_battery(self) -> None:
        """creates the ID_Battery table in the Database"""
        battery_capacity = [0, 10_000]  # W  TODO kan be given externally
        columns = self.get_dtypes_dict(Battery().__dict__)
        table_dict = {"ID_Battery": np.arange(1, len(battery_capacity) + 1),
                      "capacity": battery_capacity,
                      "capacity_unit": np.full((len(battery_capacity),), "W"),
                      "charge_efficiency": np.full((len(battery_capacity),), 0.95),
                      "discharge_efficiency": np.full((len(battery_capacity),), 0.95),
                      "charge_power_max": np.full((len(battery_capacity),), 4_500),
                      "charge_power_max_unit": np.full((len(battery_capacity),), "W"),
                      "discharge_power_max": np.full((len(battery_capacity),), 4_500),
                      "discharge_power_max_unit": np.full((len(battery_capacity),), "W")
                      }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().battery,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_household_pv(self):
        """saves the different PV types to the DB"""
        pv_power = [0, 5, 10]  # kWp  TODO kan be given externally
        columns = self.get_dtypes_dict(PV().__dict__)
        table_dict = {"ID_PV": np.arange(1, len(pv_power) + 1),
                      "peak_power": pv_power,
                      "peak_power_unit": np.full((len(pv_power),), "kWp")
                      }
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().pv,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def generate_scenario_table(self) -> None:
        """
        Creates big table for optimization with everthing included.
        """
        Country = DB().read_DataFrame()
        HouseholdType = DB().read_DataFrame()
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
                                                                        [SpaceHeatingSystem.iloc[row5][
                                                                             REG_Var().ID_SpaceHeatingSystem]] +
                                                                        [SpaceHeatingTank.iloc[row6][
                                                                             REG_Var().ID_SpaceHeatingTank]] +
                                                                        [DHWTank.iloc[row7][REG_Var().ID_DHWTank]] +
                                                                        [SpaceCooling.iloc[row8][
                                                                             REG_Var().ID_SpaceCooling]] +
                                                                        [PV.iloc[row9][REG_Var().ID_PV]] +
                                                                        [Battery.iloc[row10][REG_Var().ID_Battery]]
                                                                        )
                                                print("Round: " + str(ID) + "/" + str(total_rounds))
                                                ID += 1
        DB().write_DataFrame(TargetTable_list, REG_Table().Gen_OBJ_ID_Household, TargetTable_columns, self.conn)

    def run(self):
        self.create_household_building()
        # self.gen_OBJ_ID_ApplianceGroup()
        self.create_household_boiler()
        self.create_household_pv()
        self.create_household_battery()
        self.create_household_air_conditioner()
        self.create_household_hot_water_tank()
        self.create_household_space_heating_tank()
        # self.generate_scenario_table()





class ProfileGenerator(MotherTableGenerator, ABC):

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
        columns = {REG_Var().HotWater: "REAL", "Unit": "TEXT"}
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


def main():
    HouseholdComponentGenerator().run()

    pass


if __name__ == "__main__":
    main()
