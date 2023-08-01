import logging
from pathlib import Path
import itertools
from typing import Dict, List, TYPE_CHECKING
import sqlalchemy
import numpy as np
import pandas as pd
from flex.db import create_db_conn
from flex import kit

if TYPE_CHECKING:
    from flex.config import Config


class DatabaseInitializer:
    def __init__(self, config: "Config"):
        self.config = config
        self.db = create_db_conn(config)

    def load_behavior_table(self, table_name: str):
        logger = logging.getLogger(self.config.project_name)
        logger.info(f"Loading FLEX-Behavior table -> {table_name}")
        self.load_table(self.config.input_behavior, table_name=table_name)

    def load_operation_table(self, table_name: str):
        logger = logging.getLogger(self.config.project_name)
        logger.info(f"Loading FLEX-Operation table -> {table_name}")
        self.load_table(self.config.input_operation, table_name=table_name)

    def load_community_table(self, table_name: str):
        logger = logging.getLogger(self.config.project_name)
        logger.info(f"Loading FLEX-Community table -> {table_name}")
        self.load_table(self.config.input_community, table_name=table_name)

    def load_table(self, input_folder: "Path", table_name: str):
        extensions = {
            ".xlsx": pd.read_excel,
            ".csv": pd.read_csv
        }
        for ext, pd_read_func in extensions.items():
            file = input_folder.joinpath(table_name + ext)
            if file.exists():
                self.db.write_dataframe(
                    table_name=table_name,
                    data_frame=pd_read_func(file).dropna(axis=1, how="all").dropna(axis=0, how="all"),
                    if_exists="replace"
                )

    def generate_operation_scenario_table(self, scenario_table_name: str, component_table_names: Dict[str, str]):
        self.drop_table(scenario_table_name)
        logger = logging.getLogger(self.config.project_name)
        logger.info(f"Generating FLEX-Operation scenario table -> {scenario_table_name}")
        scenario_df = self.generate_params_combination_df(self.get_component_scenario_ids(component_table_names))
        scenario_ids = np.array(range(1, 1 + len(scenario_df)))
        scenario_df.insert(loc=0, column="ID_Scenario", value=scenario_ids)
        data_types = {name: sqlalchemy.types.Integer for name in scenario_df.columns}
        self.db.write_dataframe(
            scenario_table_name,
            scenario_df,
            data_types=data_types,
            if_exists="replace",
        )

    def get_component_scenario_ids(self, component_table_names: Dict[str, str]) -> Dict[str, List[int]]:
        component_scenario_ids = {}
        engine = self.db.get_engine().connect()
        for id_name, table_name in component_table_names.items():
            if engine.dialect.has_table(engine, table_name):
                component_scenario_ids[id_name] = self.db.read_dataframe(table_name)[id_name].unique()
        return component_scenario_ids

    @staticmethod
    def generate_params_combination_df(params_values: dict,
                                       excluded_keys: List[str]) -> pd.DataFrame:

        # check if the exclude keys have the same length,
        excluded_values = [params_values[k] for k in excluded_keys]
        if not all(len(element) == len(excluded_values[0]) for element in excluded_values):
            assert "values of excluded keys provided dont have the same length"
        # define the first key and its value which is used in the permutation:
        first_excluded_key = excluded_keys[0]
        # create new dictionary where the excluded list is the first element of the first excluded key
        new_dict = {first_excluded_key: excluded_values}
        for key, values in params_values.items():
            if key not in excluded_keys:
                new_dict[key] = values

        keys, values = zip(*new_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame(permutations_dicts)
        # expand the column with the excluded values with their key names:
        df[excluded_keys] = df[first_excluded_key].apply(pd.Series)
        return df

    def drop_table(self, table_name: str):
        logger = logging.getLogger(self.config.project_name)
        engine = self.db.get_engine().connect()
        if engine.dialect.has_table(engine, table_name):
            logger.info(f"Dropping table -> {table_name}")
            self.db.drop_table(table_name)


if __name__ == "__main__":
    from config import cfg
    test_df1 = pd.DataFrame(data=[1, 2, 3], columns=["A"])
    test_df2 = pd.DataFrame(data=[1, 2, 3], columns=["B"])
    test_df3 = pd.DataFrame(data=[1, 2, 3], columns=["C"])
    dbi = DatabaseInitializer(config=cfg)
    dictionary = {"a": [1, 1, 1],
                  "b": [2, 2, 2],
                  "c": [3, 3]}
    # dbi.\
    generate_params_combination_df(params_values=dictionary, excluded_keys=["a", "b"])

