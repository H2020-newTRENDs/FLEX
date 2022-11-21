from pathlib import Path
import itertools
from typing import Dict, List, TYPE_CHECKING, get_type_hints, Type
import sqlalchemy
import numpy as np
import pandas as pd
from enum import Enum
from basics.db import create_db_conn
from basics import kit

if TYPE_CHECKING:
    from basics.config import Config

logger = kit.get_logger(__name__)


class DatabaseInitializer:
    def __init__(
        self,
        config: "Config",
        input_folder: "Path",
        scenario_components: Type["Enum"] = None,
    ):
        self.config = config
        self.db = create_db_conn(config)
        self.input_folder = input_folder
        self.scenario_components = scenario_components

    def clear_db(self):
        logger.info(f"clearing database {self.config.project_name}.sqlite.")
        self.db.clear_database()

    def load_component_table(self, component: "Enum"):
        file = self.input_folder / Path(component.table_name + ".xlsx")
        logger.info(f"loading table -> {component.table_name}")
        df = pd.read_excel(file, engine="openpyxl").dropna(how="all")  # drop columns and rows that contain only nan
        data_types = kit.convert_datatype_py2sql(get_type_hints(component.class_var))
        assert all(
            key in list(data_types.keys()) for key in list(df.columns[1:])
        ), logger.error(
            f"Attributes of component {component.name} are inconsistent in the scenario table. "
            f"Please check and revise."
        )
        self.db.write_dataframe(
            component.table_name, df, data_types=data_types, if_exists="replace"
        )

    def load_table(self, table_enum: "Enum"):
        table_name = table_enum.value
        file = self.input_folder / Path(table_name + ".xlsx")
        logger.info(f"loading table -> {table_name}")
        df = pd.read_excel(file, engine="openpyxl")
        self.db.write_dataframe(table_name, df, if_exists="replace")

    def drop_table(self, table_enum: "Enum"):
        table_name = table_enum.value
        logger.info(f"dropping table -> {table_name}")
        self.db.drop_table(table_name)

    def get_component_scenario_ids(self) -> Dict[str, List[int]]:
        component_scenario_ids = {}
        engine = self.db.get_engine().connect()
        for name, item_enum in self.scenario_components.__members__.items():
            component_enum_table_name = item_enum.table_name
            component_enum_id = item_enum.id
            if engine.dialect.has_table(engine, component_enum_table_name):
                table = self.db.read_dataframe(component_enum_table_name)
                component_scenario_ids[component_enum_id] = table[
                    component_enum_id
                ].unique()
        return component_scenario_ids

    @staticmethod
    def generate_params_combination_df(params_values: dict) -> pd.DataFrame:
        keys, values = zip(*params_values.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return pd.DataFrame(permutations_dicts)

    def generate_scenario_table(self, table_name: str):
        self.db.drop_table(table_name)
        logger.info(f"generating table -> {table_name}")
        scenario_df = self.generate_params_combination_df(
            self.get_component_scenario_ids()
        )
        scenario_ids = np.array(range(1, 1 + len(scenario_df)))
        scenario_df.insert(
            loc=0, column="ID_Scenario", value=scenario_ids
        )
        data_types = {name: sqlalchemy.types.Integer for name in scenario_df.columns}
        self.db.write_dataframe(
            table_name,
            scenario_df,
            data_types=data_types,
            if_exists="replace",
        )
