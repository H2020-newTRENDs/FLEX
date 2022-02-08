from pathlib import Path
import sqlalchemy


def create_connection(db_folder: Path = Path(__file__).parent.parent / Path("data"),
                      db_name: str = "root"
                      ) -> sqlalchemy.engine.Engine:
    """if no database name and folder path are provided the root database is selected"""
    # assert (db_folder / Path(db_name + ".sqlite")).exists()
    return sqlalchemy.create_engine(f'sqlite:///{(str(db_folder / Path(db_name)) + ".sqlite")}')


root_connection = create_connection()  # when multiprocessing the engine has to be created for each process
results_connection = create_connection(db_name="results")


class Config:
    def __init__(self, list_of_dicts):
        self.region_config = None
        self.space_heating_tank_config = None
        self.hot_water_tank_config = None
        self.feed_in_tariff_config = None
        self.electricity_price_config = None
        self.demand_config = None
        self.building_config = None
        self.boiler_config = None
        self.behavior_config = None
        self.battery_config = None
        self.air_conditioner_config = None
        for dictionary in list_of_dicts:
            setattr(self, list(dictionary.keys())[0], list(dictionary.values())[0])


# # All Power inputs have to be provided in W !!!!
# air_conditioner_config = {
#     "efficiency": 3,
#     "power": [0, 10_000],
# }
#
# appliance_config = {
#     None: None
# }
#
# battery_config = {
#     "capacity": [0, 7_000],
#     "charge_efficiency":  0.95,
#     "discharge_efficiency": 0.95,
#     "charge_power_max": 4_500,
#     "discharge_power_max": 4_500
# }
#
# behavior_config = {
#     "temperature_min": 20,
#     "temperature_max": 27,
#     "night_reduction": 2
# }
#
# boiler_config = {
#     "name": ["Air_HP", "Ground_HP"],
#     "thermal_power_max": 15_000,   # W
#     "heating_element_power": 7_500,
#     "carnot_efficiency_factor": [0.4, 0.35],
#     "heating_supply_temperature": 35,
#     "hot_water_supply_temperature":  55
# }
#
# building_config = {
#     None: None
# }
#
# demand_config = {
#     "hot_water_demand_path": "C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\_Philipp\inputdata\AUT\Hot_water_profile.xlsx"  # absolute path to the profile
# }
#
# electricity_price_config = {
#     "entsoe_api_key": 'c06ee579-f827-486d-bc1f-8fa0d7ccd3da',
#     "start": "20190101",
#     "end": "20200101",
#     "country_code": "AT"
# }
#
# feed_in_tariff_config = {
#     "fixed_feed_in_tariff": [7.67],  # more feed in tarifs can be provided in list
#     "variable_feed_in_tariff_path": [None]  # for variable feed ins provide path(s)
# }
#
# hot_water_tank_config = {
#     "size": [0, 400],  # l
#     "loss": 0.2,  # W/m2K
#     "temperature_start": 28,  # °C
#     "temperature_max": 65,  # °C
#     "temperature_min": 28,  # °C
#     "temperature_surrounding": 20  # °C
# }
#
# person_config = {
#     None: None
# }
#
# pv_region_config = {
#     "nuts_level": 3,
#     "country_code": "AT",
#     "start_year": 2010,
#     "end_year": 2010
# }
#
# space_heating_tank_config = {
#     "size": [0, 750],
#     "loss": 0.2,  # W/m2K
#     "temperature_start": 28,  # °C
#     "temperature_max": 45,  # °C
#     "temperature_min": 28,
#     "temperature_surrounding": 20  # °C
# }
#
# vehicle_config = {
#     None: None
# }


