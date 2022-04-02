import itertools
from abc import ABC, abstractmethod
from typing import List, Dict, Union, ClassVar, TYPE_CHECKING, get_type_hints

import numpy as np
import pandas as pd
import sqlalchemy

from flex.core.db import create_db_conn

if TYPE_CHECKING:
    from flex.core.config import Config
    from flex.core.component import Component


class DatabaseInitializer(ABC):

    def __init__(self, config: 'Config'):
        self.config = config
        self.db = create_db_conn(config)
        self.scenario_enum = self.get_scenario_enum()
        self.component_enum = self.get_component_enum()

    @abstractmethod
    def get_scenario_enum(self):
        pass

    @abstractmethod
    def get_component_enum(self):
        pass

    @staticmethod
    def convert_datatype_py2sql(data_types: dict) -> dict:
        type_py2sql_dict: dict = {
            int: sqlalchemy.types.BigInteger,
            Union[int, None]: sqlalchemy.types.BigInteger,
            Union[List[int], None]: sqlalchemy.types.BigInteger,
            Union[np.ndarray, int, None]: sqlalchemy.types.BigInteger,
            str: sqlalchemy.types.Unicode,
            Union[str, None]: sqlalchemy.types.Unicode,
            Union[List[str], None]: sqlalchemy.types.Unicode,
            float: sqlalchemy.types.Float,
            Union[float, None]: sqlalchemy.types.Float,
            Union[List[float], None]: sqlalchemy.types.Float,
            Union[np.ndarray, None]: sqlalchemy.types.Float,
            Union[np.ndarray, float, None]: sqlalchemy.types.Float,
        }
        for key, value in data_types.items():
            data_types[key] = type_py2sql_dict[value]
        return data_types

    @staticmethod
    def generate_params_combination_df(params_values: dict) -> pd.DataFrame:
        keys, values = zip(*params_values.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return pd.DataFrame(permutations_dicts)

    def load_table(self, table_name: str):
        print("done")
        self.db.load_source_table_to_project_db(table_name)

    @abstractmethod
    def load_source_table(self):
        pass

    def add_component(self, component_cls: ClassVar['Component'], component_scenarios: Dict[str, list]):
        name = component_cls.__name__
        table_name = self.component_enum.__dict__[name].table_name
        data_types = self.convert_datatype_py2sql(get_type_hints(component_cls))
        assert all(key in list(data_types.keys()) for key in list(component_scenarios.keys())), \
            print(f'ComponentAttributeNameError in db_initializer: {name}')
        component_scenario_df = self.generate_component_scenario_df(name, component_scenarios)
        self.db.write_dataframe(table_name, component_scenario_df, data_types=data_types, if_exists='replace')

    def generate_component_scenario_df(self, component_cls_name: str, component_scenarios: Dict[str, list]) -> pd.DataFrame:
        component_scenario_df = pd.DataFrame(component_scenarios)
        component_scenario_ids = np.array(range(1, 1 + len(component_scenario_df)))
        component_scenario_df.insert(0, self.component_enum.__dict__[component_cls_name].id, component_scenario_ids)
        return component_scenario_df

    @abstractmethod
    def setup_component_scenario(self):
        pass

    def get_component_scenario_ids(self) -> Dict[str, int]:
        component_scenario_ids = {}
        engine = self.db.get_engine().connect()
        for item in self.component_enum.__members__.items():
            component_enum_table_name = getattr(self.component_enum, item[0]).table_name
            component_enum_id = getattr(self.component_enum, item[0]).id
            if engine.dialect.has_table(engine, component_enum_table_name):
                table = self.db.read_dataframe(component_enum_table_name)
                component_scenario_ids[component_enum_id] = table[component_enum_id].unique()
            else:
                print(f"Component {item[0]} is not initialized.")
        return component_scenario_ids

    def setup_scenario(self):
        scenario_df = self.generate_params_combination_df(self.get_component_scenario_ids())
        scenario_ids = np.array(range(1, 1 + len(scenario_df)))
        scenario_df.insert(0, self.scenario_enum.Scenario.id, scenario_ids)
        data_types = {name: sqlalchemy.types.Integer for name in scenario_df.columns}
        self.db.write_dataframe(self.scenario_enum.Scenario.table_name, scenario_df,
                                data_types=data_types, if_exists='replace')

    def run(self):
        self.load_source_table()
        self.setup_component_scenario()
        self.setup_scenario()
