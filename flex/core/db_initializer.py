import itertools
import sqlalchemy
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Dict, ClassVar, Union, TYPE_CHECKING, get_type_hints

from flex.core.db import create_db_conn

if TYPE_CHECKING:
    from flex.core.config import Config
    from flex.core.component import Component


class DBInitializer:

    def __init__(self, config: 'Config'):
        self.config = config
        self.db = create_db_conn(config)
        self.component_table_prefix = "OperationScenarioComponent_"
        self.scenario_table_name = "OperationScenario"
        self.component_scenario_ids: Dict[str, np.ndarray] = {}

    @staticmethod
    def convert_datatype_py2sql(datatypes: dict) -> dict:
        type_py2sql_dict: dict = {
            int: sqlalchemy.types.BigInteger,
            Union[int, None]: sqlalchemy.types.BigInteger,
            str: sqlalchemy.types.Unicode,
            Union[str, None]: sqlalchemy.types.Unicode,
            float: sqlalchemy.types.Float,
            Union[float, None]: sqlalchemy.types.Float,
        }
        for key, value in datatypes.items():
            datatypes[key] = type_py2sql_dict[value]
        return datatypes

    @staticmethod
    def generate_params_combination_dataframe(params_values: dict) -> pd.DataFrame:
        keys, values = zip(*params_values.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return pd.DataFrame(permutations_dicts)

    def add_component(self, component_cls: ClassVar['Component'], component_scenarios: Dict[str, list]):
        table_name = self.component_table_prefix + component_cls.__name__
        data_types = self.convert_datatype_py2sql(get_type_hints(component_cls))
        assert data_types.keys() == component_scenarios.keys(), \
            print(f'ComponentAttributeNameError in db_initializer: {component_cls.__name__}')
        component_scenarios_dataframe = self.setup_component_scenario(component_cls.__name__, component_scenarios)
        self.db.write_dataframe(table_name, component_scenarios_dataframe, data_types=data_types, if_exists='replace')

    @abstractmethod
    def setup_component_scenarios(self):
        pass

    def register_component_scenario_ids(self, component_cls_name: str, component_scenario_ids: np.ndarray):
        self.component_scenario_ids['ID_' + component_cls_name] = component_scenario_ids

    def setup_component_scenario(self, component_cls_name: str, component_scenarios: Dict[str, list]) -> pd.DataFrame:
        component_scenario_dataframe = pd.DataFrame(component_scenarios)
        component_scenario_ids = np.array(range(1, 1 + len(component_scenario_dataframe)))
        component_scenario_dataframe.insert(0, 'ID_' + component_cls_name, component_scenario_ids)
        self.register_component_scenario_ids(component_cls_name, component_scenario_ids)
        return component_scenario_dataframe

    def setup_scenario(self):
        scenario_dataframe = self.generate_params_combination_dataframe(self.component_scenario_ids)
        scenario_ids = np.array(range(1, 1 + len(scenario_dataframe)))
        scenario_dataframe.insert(0, 'ID_Scenario', scenario_ids)
        data_types = {name: sqlalchemy.types.Integer for name in scenario_dataframe.columns}
        self.db.write_dataframe(self.scenario_table_name, scenario_dataframe,
                                data_types=data_types, if_exists='replace')

    def setup_project_database(self):
        self.setup_component_scenarios()
        self.setup_scenario()
