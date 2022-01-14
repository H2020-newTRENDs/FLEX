import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import sqlalchemy.types
import itertools

from _Refactor.basic.db import DB
import _Refactor.core.household.components as household_components
import _Refactor.core.environment.components as environment_components
from _Refactor.basic.reg import Table
import _Refactor.basic.config as config
import import_building_data
import input_data_structure as structure
import profile_generator


class HouseholdComponentGenerator:
    """generates the Household tables"""

    def calculate_cylindrical_area_from_volume(self, tank_sizes: list) -> list:
        """calculates the minimal cylindicral area of a tank with a give volume"""
        radius = [(2 * volume / 1_000 / 4 / np.pi) ** (1. / 3) for volume in tank_sizes]
        surface_area = [2 * np.pi * r ** 2 + 2 * tank_sizes[i] / 1_000 / r for i, r in enumerate(radius)]
        return surface_area

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
        columns = structure.BuildingData().__dict__
        assert list(columns.keys()).sort() == list(building_data.keys()).sort()
        # save
        DB().write_dataframe(table_name=Table().building,
                             data_frame=building_data,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_household_boiler(self) -> None:
        """creates the space heating system table in the Database"""
        heating_system = ["Air_HP", "Ground_HP"]  # TODO kan be given externally
        carnot_factor = [0.4, 0.35]  # for the heat pumps respectively
        columns = structure.BoilerData().__dict__
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
        columns = structure.SpaceHeatingTankData().__dict__
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
                      "temperature_max": np.full((len(tank_sizes),), 45),  # TankMaximalTemperature
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
        columns = structure.HotWaterTankData().__dict__
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
        columns = structure.AirConditionerData().__dict__
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
        columns = structure.BatteryData().__dict__
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
        columns = structure.PVData().__dict__
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

    def run(self):
        # delete existing tables so no old tables stay accidentally:
        for household_table in household_components.household_component_list:
            DB().drop_table(household_table.__name__)
        # create new tables
        self.create_household_building()
        self.create_household_boiler()
        self.create_household_pv()
        self.create_household_battery()
        self.create_household_air_conditioner()
        self.create_household_hot_water_tank()
        self.create_household_space_heating_tank()


class EnvironmentGenerator:

    def create_environment_electricity_price(self) -> None:
        columns_price = self.get_dtypes_dict(environment_components.ElectricityPrice().__dict__)
        electricity_price_types = ["variable", "fixed"]  # name  TODO should be provided externally
        electricity_price_ids = [i for i in range(1, len(electricity_price_types) + 1)]

        electricity_price_dict = {"ID_ElectricityPrice": electricity_price_ids,
                                  "name": electricity_price_types}
        table = pd.DataFrame(electricity_price_dict)
        assert list(table.columns).sort() == list(columns_price.keys()).sort()
        # save
        DB().write_dataframe(table_name=Table().electricity_price,
                             data_frame=table,
                             data_types=columns_price,
                             if_exists="replace"
                             )

    def create_environment_feed_in_tariff(self) -> None:
        columns_feed_in = self.get_dtypes_dict(environment_components.FeedInTariff().__dict__)
        electricity_feed_in_type = ["fixed"]  # name
        electricity_feed_in_ids = [i for i in range(1, len(electricity_feed_in_type) + 1)]

        electricity_price_dict = {"ID_FeedInTariff": electricity_feed_in_ids,
                                  "name": electricity_feed_in_type}
        table = pd.DataFrame(electricity_price_dict)
        assert list(table.columns).sort() == list(columns_feed_in.keys()).sort()
        # save
        DB().write_dataframe(table_name=Table().feedin_tariff,
                             data_frame=table,
                             data_types=columns_feed_in,
                             if_exists="replace"
                             )

    def run(self):
        # delete existing tables so no old tables stay accidentally:
        for environment_table in environment_components.environment_component_list:
            DB().drop_table(f"Environment{environment_table.__name__}")
        # create new tables
        self.create_environment_electricity_price()
        self.create_environment_feed_in_tariff()


def generate_scenarios_table() -> None:
    """
    Creates big table for optimization with everything included.
    """
    # check which of the tables exist for the project:
    engine = config.root_connection.connect()
    scenarios_columns = {}

    # iterate through household components
    for household_table in household_components.household_component_list:
        if engine.dialect.has_table(engine, f"Household{household_table.__name__}"):  # check if table exists
            # read table:
            table = DB().read_dataframe(f"Household{household_table.__name__}")
            # iterate through columns and get id name:
            for column_name in table.columns:
                if column_name.startswith("ID"):
                    scenarios_columns[column_name] = table[column_name].unique()
        else:
            print(f"{household_table.__name__} does not exist in root db")

    # iterate through household components
    for environment_table in environment_components.environment_component_list:
        if engine.dialect.has_table(engine, f"Environment{environment_table.__name__}"):  # check if table exists
            # read table:
            env_table = DB().read_dataframe(f"Environment{environment_table.__name__}")
            # iterate through columns and get id name:
            for column_name in env_table.columns:
                if column_name.startswith("ID"):
                    scenarios_columns[column_name] = env_table[column_name].unique()
        else:
            print(f"{environment_table.__name__} does not exist in root db")

    # create permutations of the columns dictionary to get all possible combinations:
    keys, values = zip(*scenarios_columns.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # create dataframe
    scenarios_table = pd.DataFrame(permutations_dicts)
    # dtypes are all "Integer":
    dtypes_dict = {name: sqlalchemy.types.Integer for name in scenarios_table.columns}
    # save to root db:
    DB().write_dataframe(table_name=Table().scenarios,
                         data_frame=scenarios_table,
                         data_types=dtypes_dict,
                         if_exists="replace"
                         )


def main():
    HouseholdComponentGenerator().run()
    EnvironmentGenerator().run()

    profile_generator.ProfileGenerator().run()

    generate_scenarios_table()


if __name__ == "__main__":
    main()
