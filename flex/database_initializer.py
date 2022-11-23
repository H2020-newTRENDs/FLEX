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

logger = kit.get_logger(__name__)


class DatabaseInitializer:
    def __init__(self, config: "Config"):
        self.config = config
        self.db = create_db_conn(config)

    def load_behavior_table(self, table_name: str):
        logger.info(f"Loading FLEX-Behavior table -> {table_name}")
        file_name = self.config.input_behavior / Path(table_name + ".xlsx")
        self.load_table(file_name=file_name, table_name=table_name)

    def load_operation_table(self, table_name: str):
        logger.info(f"Loading FLEX-Operation table -> {table_name}")
        file_name = self.config.input_operation / Path(table_name + ".xlsx")
        self.load_table(file_name=file_name, table_name=table_name)

    def load_community_table(self, table_name: str):
        logger.info(f"Loading FLEX-Community table -> {table_name}")
        file_name = self.config.input_community / Path(table_name + ".xlsx")
        self.load_table(file_name=file_name, table_name=table_name)

    def load_table(self, file_name: "Path", table_name: str):
        df = pd.read_excel(file_name, engine="openpyxl")
        self.db.write_dataframe(table_name, df, if_exists="replace")

    def generate_operation_scenario_table(self, scenario_table_name: str, component_table_names: Dict[str, str]):
        self.drop_table(scenario_table_name)
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
    def generate_params_combination_df(params_values: dict) -> pd.DataFrame:
        keys, values = zip(*params_values.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return pd.DataFrame(permutations_dicts)

    def drop_table(self, table_name: str):
        engine = self.db.get_engine().connect()
        if engine.dialect.has_table(engine, table_name):
            logger.info(f"Dropping table -> {table_name}")
            self.db.drop_table(table_name)
