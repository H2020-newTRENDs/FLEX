import os
import itertools
import sqlalchemy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, ClassVar, TYPE_CHECKING, get_type_hints

from .db import create_db_conn

if TYPE_CHECKING:
    from .component import Component


def setup_folder_path(folder_path) -> str:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


@dataclass
class Config(ABC):
    project_name: str
    project_root: str = field(default=setup_folder_path(os.path.dirname(__file__)))
    database_folder: str = field(default=setup_folder_path(os.path.dirname('database/')))
    figure_folder: str = field(default=setup_folder_path(os.path.dirname('figure/')))
    component_scenario_ids: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.db = create_db_conn(self)
        self.setup_project_database()

    @staticmethod
    def convert_datatype_py2sql(datatypes: dict) -> dict:
        type_py2sql_dict = {
            int: sqlalchemy.types.BigInteger,
            str: sqlalchemy.types.Unicode,
            float: sqlalchemy.types.Float,
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
        table_name = "ScenarioComponent_" + component_cls.__name__
        data_types = self.convert_datatype_py2sql(get_type_hints(component_cls))
        assert data_types.keys() == component_scenarios.keys(), \
            print(f'ComponentAttributeNameError in config: {component_cls.__name__}')
        component_scenarios = self.generate_params_combination_dataframe(component_scenarios)
        component_scenario_ids = np.array(range(1, 1 + len(component_scenarios)))
        component_scenarios.insert(0, 'ID_' + component_cls.__name__, component_scenario_ids)
        self.component_scenario_ids['ID_' + component_cls.__name__] = component_scenario_ids
        self.db.write_dataframe(table_name, component_scenarios, data_types=data_types, if_exists='replace')

    @abstractmethod
    def setup_component_scenarios(self):
        pass

    def setup_scenarios(self):
        keys, values = zip(*self.component_scenario_ids.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        scenarios = pd.DataFrame(permutations_dicts)
        scenario_ids = np.array(range(1, 1 + len(scenarios)))
        scenarios.insert(0, 'ID_Scenario', scenario_ids)
        data_types = {name: sqlalchemy.types.Integer for name in scenarios.columns}
        self.db.write_dataframe('Scenarios', scenarios, data_types=data_types, if_exists='replace')

    def setup_project_database(self):
        self.setup_component_scenarios()
        self.setup_scenarios()
