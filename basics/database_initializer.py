from pathlib import Path
import itertools
from typing import Dict, TYPE_CHECKING, get_type_hints
import sqlalchemy
import numpy as np
import pandas as pd
from typing import ClassVar
from enum import Enum
from basics.db import create_db_conn
from basics import kit

if TYPE_CHECKING:
    from basics.config import Config

logger = kit.get_logger(__name__)


class DatabaseInitializer:

    def __init__(self, config: 'Config', enums: ClassVar['Enum']):
        self.config = config
        self.db = create_db_conn(config)
        self.enums = enums

    def clear_db(self):
        logger.info(f'clearing database {self.config.project_name}.sqlite.')
        self.db.clear_table_from_database()

    def load_component_scenario_tables(self):
        pass

    def load_component_table(self, component: ClassVar['Enum']):
        file = self.config.input_folder / Path(component.table_name + ".xlsx")
        logger.info(f'loading component table -> {component.table_name}')
        df = pd.read_excel(file, engine="openpyxl")
        data_types = kit.convert_datatype_py2sql(get_type_hints(component.class_var))
        assert all(key in list(data_types.keys()) for key in list(df.columns[1:])), \
            logger.error(f'Attributes of component {component.name} are inconsistent in the scenario table. '
                         f'Please check and revise.')
        self.db.write_dataframe(component.table_name, df, data_types=data_types, if_exists='replace')

    def load_other_source_tables(self):
        pass

    def load_source_table(self, table: ClassVar['Enum']):
        file = self.config.input_folder / Path(table.value + ".xlsx")
        logger.info(f'loading component table -> {table.value}')
        df = pd.read_excel(file, engine="openpyxl")
        self.db.write_dataframe(table.value, df, if_exists='replace')

    def get_component_scenario_ids(self) -> Dict[str, int]:
        component_scenario_ids = {}
        engine = self.db.get_engine().connect()
        for item in self.enums.__members__.items():
            component_enum_table_name = getattr(self.enums, item[0]).table_name
            component_enum_id = getattr(self.enums, item[0]).id
            if engine.dialect.has_table(engine, component_enum_table_name):
                table = self.db.read_dataframe(component_enum_table_name)
                component_scenario_ids[component_enum_id] = table[component_enum_id].unique()
            else:
                if item[0] == "Scenario":
                    pass
                else:
                    logger.error(f'Component {item[0]} is not initialized.')
        return component_scenario_ids

    @staticmethod
    def generate_params_combination_df(params_values: dict) -> pd.DataFrame:
        keys, values = zip(*params_values.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return pd.DataFrame(permutations_dicts)

    def setup_scenario(self):
        scenario_df = self.generate_params_combination_df(self.get_component_scenario_ids())
        scenario_ids = np.array(range(1, 1 + len(scenario_df)))
        scenario_df[self.enums.Scenario.id] = scenario_ids
        data_types = {name: sqlalchemy.types.Integer for name in scenario_df.columns}
        self.db.write_dataframe(self.enums.Scenario.table_name, scenario_df,
                                data_types=data_types, if_exists='replace')

    def run(self):
        self.clear_db()
        self.load_component_scenario_tables()
        self.load_other_source_tables()
        self.setup_scenario()
