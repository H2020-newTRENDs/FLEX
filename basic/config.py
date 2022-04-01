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
        # self.space_heating_tank_config = None
        # self.hot_water_tank_config = None
        # self.feed_in_tariff_config = None
        # self.electricity_price_config = None
        # self.demand_config = None
        # self.building_config = None
        # self.boiler_config = None
        # self.behavior_config = None
        # self.battery_config = None
        # self.air_conditioner_config = None
        for dictionary in list_of_dicts:
            setattr(self, list(dictionary.keys())[0], list(dictionary.values())[0])



