import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import sqlalchemy.types

from _Refactor.basic.db import DB
import _Refactor.core.household.components as components
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
        assert list(components.Building().__dict__.keys()).sort() == list(building_data.columns).sort()
        columns = self.get_dtypes_dict(components.Building().__dict__)
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
        columns = self.get_dtypes_dict(components.Boiler().__dict__)
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
        columns = self.get_dtypes_dict(components.SpaceHeatingTank().__dict__)
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
        columns = self.get_dtypes_dict(components.HotWaterTank().__dict__)
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
        columns = self.get_dtypes_dict(components.AirConditioner().__dict__)
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
        columns = self.get_dtypes_dict(components.Battery().__dict__)
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
        columns = self.get_dtypes_dict(components.PV().__dict__)
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





def main():
    HouseholdComponentGenerator().run()

    pass


if __name__ == "__main__":
    main()
