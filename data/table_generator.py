import datetime
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy.types

import basic.config as config
import core.household.components as components
from basic.db import DB
from basic.reg import Table
from data import import_building_data
from data import input_data_structure as structure
from flex.data.setup_energy_price import get_entsoe_prices
from data.download_weather_and_pv import PVGIS
from data.profile_generator import ProfileGenerator


class InputDataGenerator:
    def __init__(self, configuration: config.Config, year: int):
        self.input = configuration
        self.id_hour = np.arange(1, 8761)
        self.year = year

    """generates the Household tables"""

    def calculate_cylindrical_area_from_volume(self, tank_sizes: list) -> list:
        """calculates the minimal cylindicral area of a tank with a give volume"""
        if 0 in tank_sizes:
            tank_sizes.remove(0)
            radius = [(2 * volume / 1_000 / 4 / np.pi) ** (1. / 3) for volume in tank_sizes]
            surface_area = [2 * np.pi * r ** 2 + 2 * tank_sizes[i] / 1_000 / r for i, r in enumerate(radius)]
            surface_area.append(0)
            tank_sizes.append(0)
            return surface_area
        else:
            radius = [(2 * volume / 1_000 / 4 / np.pi) ** (1. / 3) for volume in tank_sizes]
            surface_area = [2 * np.pi * r ** 2 + 2 * tank_sizes[i] / 1_000 / r for i, r in enumerate(radius)]
            return surface_area

    def adapt_hotmaps_profile_to_year(self, hotmaps_hot_water_profile: np.array) -> np.array:
        """
        Checks what the first day of the year is and changes the HOTMAPS profile accordingly.
        Args:
            year: int
            hotmaps_hot_water_profile: numpy array

        Returns: the profile with the right day of the week in the beginning
        """
        # Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
        first_day_of_year = datetime.datetime(self.year, 1, 1).weekday()
        # standardized profiles are shifted to the respective weekday
        # HotMaps profile starts on a friday:
        if first_day_of_year == 4:
            # return the profile without changing anything
            return hotmaps_hot_water_profile
        elif first_day_of_year == 5:  # year starts on saturday, add friday to the end of the year
            return np.append(hotmaps_hot_water_profile[24:], hotmaps_hot_water_profile[0:24])
        elif first_day_of_year == 6:  # sunday, add friday and saturday to the end of the year
            return np.append(hotmaps_hot_water_profile[48:], hotmaps_hot_water_profile[0:48])
        elif first_day_of_year == 3:  # thursday, add the friday profile to the beginning and delete the last day
            return np.append(hotmaps_hot_water_profile[0:24], hotmaps_hot_water_profile[:8736])
        elif first_day_of_year == 2:  # wednesday, add the monday+tuesday profile to beginning and delete last 2 days
            return np.append(hotmaps_hot_water_profile[72:120], hotmaps_hot_water_profile[:8712])
        elif first_day_of_year == 1:  # tuesday, add monday+tuesday+wednesday profile and delete last 3 days
            return np.append(hotmaps_hot_water_profile[72:144], hotmaps_hot_water_profile[:8688])
        elif first_day_of_year == 0:  # monday, add 4 days from following week and delete last 4 days
            return np.append(hotmaps_hot_water_profile[72:168], hotmaps_hot_water_profile[:8664])
        else:
            assert "no valid day"

    def create_air_conditioner_data(self) -> None:
        """creates the ID_SpaceCooling table in the Database"""
        cooling_power = self.input.air_conditioner_config["power"]  # W
        cooling_efficiency = np.full((len(cooling_power),), self.input.air_conditioner_config["efficiency"])
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
                             if_exists="replace")

    def create_appliance_data(self):
        pass

    def create_battery_data(self) -> None:
        """creates the ID_Battery table in the Database"""
        battery_capacity = self.input.battery_config["capacity"]  # W
        columns = structure.BatteryData().__dict__
        table_dict = {
            "ID_Battery": np.arange(1, len(battery_capacity) + 1),
            "capacity": battery_capacity,
            "capacity_unit": np.full((len(battery_capacity),), "W"),
            "charge_efficiency": np.full((len(battery_capacity),), self.input.battery_config["charge_efficiency"]),
            "discharge_efficiency": np.full((len(battery_capacity),),
                                            self.input.battery_config["discharge_efficiency"]),
            "charge_power_max": np.full((len(battery_capacity),), self.input.battery_config["charge_power_max"]),
            "charge_power_max_unit": np.full((len(battery_capacity),), "W"),
            "discharge_power_max": np.full((len(battery_capacity),), self.input.battery_config["discharge_power_max"]),
            "discharge_power_max_unit": np.full((len(battery_capacity),), "W")
        }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().battery,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace")

    def create_behavior_data(self):
        minimum_indoor_temperature, maximum_indoor_temperature = ProfileGenerator().generate_target_indoor_temperature_fixed(
            temperature_min=self.input.behavior_config["temperature_min"],
            temperature_max=self.input.behavior_config["temperature_max"],
            night_reduction=self.input.behavior_config["night_reduction"]
        )
        behaviour_dict = {"ID_Behavior": np.full((8760,), 1),
                          "indoor_set_temperature_min": minimum_indoor_temperature,
                          "indoor_set_temperature_max": maximum_indoor_temperature}
        behaviour_table = pd.DataFrame(behaviour_dict)
        assert sorted(list(behaviour_table.columns)) == sorted(list(structure.BehaviorData().__dict__.keys()))

        DB().write_dataframe(table_name=Table().behavior,
                             data_frame=behaviour_table,
                             data_types=structure.BehaviorData().__dict__,
                             if_exists="replace"
                             )

    def create_boiler_data(self) -> None:
        """creates the space heating system table in the Database"""
        heating_system = self.input.boiler_config["name"]
        carnot_factor = self.input.boiler_config["carnot_efficiency_factor"]
        columns = structure.BoilerData().__dict__
        table_dict = {
            "ID_Boiler": np.arange(1, len(heating_system) + 1),
            "name": heating_system,
            "thermal_power_max": np.full((len(heating_system),), self.input.boiler_config["thermal_power_max"]),
            "thermal_power_max_unit": np.full((len(heating_system),), "W"),
            "heating_element_power": np.full((len(heating_system),), self.input.boiler_config["heating_element_power"]),
            "heating_element_power_unit": np.full((len(heating_system),), "W"),
            "carnot_efficiency_factor": carnot_factor,
            "heating_supply_temperature": np.full((len(heating_system),),
                                                  self.input.boiler_config["heating_supply_temperature"]),
            "hot_water_supply_temperature": np.full((len(heating_system),),
                                                    self.input.boiler_config["hot_water_supply_temperature"])
        }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().boiler,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_building_data(self) -> None:
        """reads building excel table and stores it to root"""
        building_mass_temperature_max = self.input.building_config["building_mass_temperature_max"]  # °C
        grid_power_max = self.input.building_config["grid_power_max"]  # W
        building_data = import_building_data.load_building_data_from_json(
            country=self.input.region_config["country_code"]
        )
        # rename columns:
        building_data = building_data.rename(
            columns={"index": "ID_Building",
                     "areawindows": "area_windows",
                     "area_suitable_solar": "area_suitable_for_solar",
                     "average_effective_area_wind_west_east_red_cool": "effective_window_area_west_east",
                     "average_effective_area_wind_north_red_cool": "effective_window_area_north",
                     "average_effective_area_wind_south_red_cool": "effective_window_area_south",
                     "spec_int_gains_cool_watt": "internal_gains"
                     })
        building_data["building_mass_temperature_max"] = building_mass_temperature_max
        building_data["grid_power_max"] = grid_power_max
        building_data["grid_power_max_unit"] = "W"
        columns = structure.BuildingData().__dict__
        # assert sorted(list(columns.keys())) == sorted(list(building_data.keys()))
        # save
        DB().write_dataframe(table_name=Table().building,
                             data_frame=building_data,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_demand_data(self):
        # load json file
        synthetic_load_path = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"synthetic_load_household.json")
        baseload = pd.read_json(synthetic_load_path, orient="table")
        baseload = baseload.loc[:, self.input.demand_config["base_load_year"]].to_numpy()

        baseload_dict = {"ID_ElectricityDemand": np.full((8760,), 1),
                         "electricity_demand": baseload.flatten(),
                         "unit": np.full((8760,), "Wh")}
        baseload_table = pd.DataFrame(baseload_dict)
        assert sorted(list(baseload_table.columns)) == sorted(list(structure.ElectricityDemandData().__dict__.keys()))
        DB().write_dataframe(table_name=Table().electricity_demand,
                             data_frame=baseload_table,
                             data_types=structure.ElectricityDemandData().__dict__,
                             if_exists="replace"
                             )

        # Hot water demand profile
        hot_water_path = Path(os.path.abspath(__file__)).parent.resolve() / Path(f"hot_water_demand.json")
        hot_water = pd.read_json(hot_water_path, orient="table")
        hot_water_profile = self.adapt_hotmaps_profile_to_year(hot_water["Profile"].to_numpy())
        hot_water_dict = {"ID_HotWaterDemand": np.full((8760,), 1),
                          "hot_water_demand": hot_water_profile,
                          "unit": np.full((8760,), "Wh")}
        hot_water_table = pd.DataFrame(hot_water_dict)
        assert sorted(list(hot_water_table.columns)) == sorted(list(structure.HotWaterDemandData().__dict__.keys()))

        DB().write_dataframe(table_name=Table().hot_water_demand,
                             data_frame=hot_water_table,
                             data_types=structure.HotWaterDemandData().__dict__,
                             if_exists="replace")

    def create_electricity_price_data(self):
        # load electricity price # TODO implement possibility of more than 2 price scenarios
        variable_electricity_price = get_entsoe_prices(
            api_key=self.input.electricity_price_config["api_key"],
            start_time=self.input.electricity_price_config["start"],
            end_time=self.input.electricity_price_config["end"],
            country_code=self.input.electricity_price_config["country_code"],
            grid_fee=self.input.electricity_price_config["grid_fee"]
        )

        if self.input.electricity_price_config["fixed_price"] == "avg":
            # TODO make mean function of variable price
            pass
        else:
            fixed_price_vector = np.full((8760,), self.input.electricity_price_config["fixed_price"] / 1_000)  # cent/Wh

        variable_price_to_db = np.column_stack(
            [np.full((8760,), 1),  # ID
             self.id_hour,  # id_hour
             variable_electricity_price,
             np.full((8760,), "cent/Wh")]
        )
        fixed_price_to_db = np.column_stack(
            [np.full((8760,), 2),  # ID
             self.id_hour,
             fixed_price_vector,
             np.full((8760,), "cent/Wh")]
        )
        price_to_db = np.vstack([variable_price_to_db, fixed_price_to_db])
        price_table = pd.DataFrame(price_to_db, columns=list(structure.ElectricityPriceData().__dict__.keys()))
        # save to database

        DB().write_dataframe(table_name=Table().electricity_price,
                             data_frame=price_table,
                             data_types=structure.ElectricityPriceData().__dict__,
                             if_exists="replace"
                             )

    def create_feed_in_tariff_data(self):
        feed_in_table = pd.DataFrame(columns=["ID_FeedInTariff", "id_hour", "feed_in_tariff", "unit"])
        for i, fixed_feed_in in enumerate(self.input.feed_in_tariff_config["fixed_feed_in_tariff"]):
            feed_in_dict = {"ID_FeedInTariff": np.full((8760,), i + 1),
                            "id_hour": self.id_hour,
                            "feed_in_tariff": np.full((8760,), fixed_feed_in / 1_000),  # in ct/Wh
                            "unit": np.full((8760,), "cent/Wh")}

            table = pd.DataFrame(feed_in_dict)
            feed_in_table = pd.concat([feed_in_table, table], axis=0)

        assert sorted(list(feed_in_table.columns)) == sorted(list(structure.FeedInTariffData().__dict__.keys()))

        for j, variable_feed_in in enumerate(self.input.feed_in_tariff_config["variable_feed_in_tariff_path"]):
            # TODO read the table from a file from path
            pass

        DB().write_dataframe(table_name=Table().feedin_tariff,
                             data_frame=feed_in_table,
                             data_types=structure.FeedInTariffData().__dict__,
                             if_exists="replace"
                             )

    def create_hot_water_tank_data(self) -> None:
        """creates the ID_DHWTank table in the Database"""
        tank_sizes = self.input.hot_water_tank_config["size"]  # liter
        columns = structure.HotWaterTankData().__dict__
        # tank is designed as Cylinder with minimal surface area:
        surface_area = self.calculate_cylindrical_area_from_volume(tank_sizes)
        table_dict = {
            "ID_HotWaterTank": np.arange(1, len(tank_sizes) + 1),  # ID
            "size": tank_sizes,  # tank size
            "size_unit": np.full((len(tank_sizes),), "kg"),  # TankSize_unit
            "surface_area": surface_area,  # TankSurfaceArea
            "surface_area_unit": np.full((len(tank_sizes),), "m2"),  # TankSurfaceArea_unit
            "loss": np.full((len(tank_sizes),), self.input.hot_water_tank_config["loss"]),  # TankLoss
            "loss_unit": np.full((len(tank_sizes),), "W/m2"),  # TankLoss_unit
            "temperature_start": np.full((len(tank_sizes),), self.input.hot_water_tank_config["temperature_start"]),
            "temperature_max": np.full((len(tank_sizes),), self.input.hot_water_tank_config["temperature_max"]),
            "temperature_min": np.full((len(tank_sizes),), self.input.hot_water_tank_config["temperature_min"]),
            "temperature_surrounding": np.full((len(tank_sizes),),
                                               self.input.hot_water_tank_config["temperature_surrounding"])
        }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().hot_water_tank,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_person_data(self):
        pass

    def create_pv_data(self):
        # is created together with region data
        pass

    def create_region_data(self) -> None:
        PVGIS().run(nuts_level=self.input.region_config["nuts_level"],
                    country_code=self.input.region_config["country_code"],
                    start_year=self.input.region_config["start_year"],
                    end_year=self.input.region_config["end_year"],
                    pv_sizes=self.input.region_config["pv_size"])

    def create_space_heating_tank_data(self) -> None:
        """creates the space heating tank table in the Database"""
        tank_size = self.input.space_heating_tank_config["size"]  # liter
        columns = structure.SpaceHeatingTankData().__dict__
        # tank is designed as Cylinder with minimal surface area:
        surface_area = self.calculate_cylindrical_area_from_volume(tank_size)
        table_dict = {
            "ID_SpaceHeatingTank": np.arange(1, len(tank_size) + 1),  # ID
            "size": tank_size,  # tank size
            "size_unit": np.full((len(tank_size),), "kg"),  # TankSize_unit
            "surface_area": surface_area,  # TankSurfaceArea
            "surface_area_unit": np.full((len(tank_size),), "m2"),  # TankSurfaceArea_unit
            "loss": np.full((len(tank_size),), self.input.space_heating_tank_config["loss"]),  # TankLoss
            "loss_unit": np.full((len(tank_size),), "W/m2"),  # TankLoss_unit
            "temperature_start": np.full((len(tank_size),), self.input.space_heating_tank_config["temperature_start"]),
            "temperature_max": np.full((len(tank_size),), self.input.space_heating_tank_config["temperature_max"]),
            "temperature_min": np.full((len(tank_size),), self.input.space_heating_tank_config["temperature_min"]),
            "temperature_surrounding": np.full((len(tank_size),),
                                               self.input.space_heating_tank_config["temperature_surrounding"])
        }
        assert table_dict.keys() == columns.keys()
        table = pd.DataFrame(table_dict)
        # save
        DB().write_dataframe(table_name=Table().space_heating_tank,
                             data_frame=table,
                             data_types=columns,
                             if_exists="replace"
                             )

    def create_vehicle_data(self):
        pass

    def generate_scenarios_table(self) -> None:
        """
        Creates big table for optimization with everything included.
        """
        # check which of the tables exist for the project:
        engine = config.root_connection.connect()
        scenarios_columns = {}

        # iterate through household components
        for household_table in components.component_list:
            if engine.dialect.has_table(engine, household_table.__name__):  # check if table exists
                # read table:
                table = DB().read_dataframe(household_table.__name__)
                # iterate through columns and get id name:
                for column_name in table.columns:
                    if column_name.startswith("ID"):
                        scenarios_columns[column_name] = table[column_name].unique()
            else:
                print(f"{household_table.__name__} does not exist in root db")

        # create permutations of the columns dictionary to get all possible combinations:
        keys, values = zip(*scenarios_columns.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        # create dataframe
        scenarios_table = pd.DataFrame(permutations_dicts)
        scenarios_table["ID_Scenarios"] = np.arange(len(scenarios_table))
        # dtypes are all "Integer":
        dtypes_dict = {name: sqlalchemy.types.Integer for name in scenarios_table.columns}
        # save to root db:
        DB().write_dataframe(table_name=Table().scenarios,
                             data_frame=scenarios_table,
                             data_types=dtypes_dict,
                             if_exists="replace"
                             )

    def run(self, skip_region: bool = False):
        # TODO implement what to do when inputs are None
        for configuration_name in self.input.__dict__.keys():
            if skip_region == True:
                if configuration_name == "region_config":
                    print("skip region")
                    continue
            function_name = f"create_{configuration_name.replace('_config', '')}_data"
            method_to_call = getattr(self, function_name)
            method_to_call()

        self.generate_scenarios_table()


def main():
    config_list = [{'region_config': {"nuts_level": 3,
                                      "country_code": '',
                                      "start_year": 2019,
                                      "end_year": 2019,
                                      "pv_size": [5, 7.5, 10]}}]
    for country in ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
                    'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL',
                    'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
                    'CH', 'IS', 'LI', 'NO', 'UK', 'TR']:
        config_list[0]['region_config']['country_code'] = country
        configuration = Config(config_list)
        try:
            InputDataGenerator(configuration=configuration, year=2019).run(skip_region=False)
        except Exception as e:
            print(e)

    # create list of all configurations defined in configurations
    # config_list = [{config_name: value} for (config_name, value) in configurations.__dict__.items()
    #                if not config_name.startswith("__")]
    # define scenario:
    # configuration = Config(config_list)
    #
    # InputDataGenerator(configuration=configuration, year=2019).run(skip_region=False)


if __name__ == "__main__":
    from basic.config import Config

    main()
