import sys
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from flex.core.config import Config
from flex.core.db import create_db_conn
from flex.core.scenario import Scenario
from .component import Battery, Boiler
from dataclasses import dataclass


@dataclass
class OperationScenario(Scenario):
    scenario_id: int
    config: 'Config'
    battery: Optional['Battery'] = None
    boiler: Optional['Boiler'] = None

    def __post_init__(self):
        self.db = create_db_conn(self.config)
        self.component_scenario_ids = self.get_component_scenario_ids()
        self.setup_components()

    def get_component_scenario_ids(self):
        return self.get_scenario_data("OperationScenario", scenario_filter={"ID_Scenario": self.scenario_id})

    def setup_components(self):
        for component_key, component_scenario_id in self.component_scenario_ids.items():
            component_name = component_key.lower().replace("id_", "")
            if component_name in self.__dict__.keys():
                component_name_cap = component_name.capitalize()
                component_params = self.get_scenario_data("OperationScenarioComponent_" + component_name_cap,
                                                          scenario_filter={"ID_" + component_name_cap: self.scenario_id})
                instance = getattr(sys.modules[__name__], component_name_cap)()
                instance.set_params(component_params)
                setattr(self, component_name, instance)



