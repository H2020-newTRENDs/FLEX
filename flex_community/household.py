from typing import Dict

import numpy as np
import pandas as pd

from flex_operation.constants import OperationScenarioComponent


class Household:

    def __init__(self, operation_scenario_id):
        self.operation_scenario_id: int = operation_scenario_id
        self.id_region: int = None
        self.id_building: int = None
        self.id_boiler: int = None
        self.id_space_heating_tank: int = None
        self.id_hot_water_tank: int = None
        self.id_space_cooling_technology: int = None
        self.id_pv: int = None
        self.id_battery: int = None
        self.id_vehicle: int = None
        self.id_energy_price: int = None
        self.id_behavior: int = None
        self.id_heating_element: int = None
        # hour results from operation model
        self.PhotovoltaicProfile_hour: np.array = None
        self.Grid_hour: np.array = None
        self.Load_hour: np.array = None
        self.Feed2Grid_hour: np.array = None
        self.BatSoC_hour: np.array = None
        # year results from operation model
        self.TotalCost_year: float = None

    def setup_component_ids(self, operation_scenario: pd.DataFrame):
        component_scenario_ids: Dict[str, int] = operation_scenario.loc[operation_scenario["ID_Scenario"] ==
                                                                        self.operation_scenario_id].iloc[0].to_dict()
        del component_scenario_ids["ID_Scenario"]
        for id_component, component_scenario_id in component_scenario_ids.items():
            component_name_cap = id_component.replace("ID_", "")
            component_enum = OperationScenarioComponent.__dict__[component_name_cap]
            id_component_name = f'id_{component_enum.value}'
            if id_component_name in self.__dict__.keys():
                setattr(self, id_component_name, component_scenario_id)

    def setup_operation_result_hour(self, df: pd.DataFrame):
        operation_result_hour: pd.DataFrame = df.loc[df["ID_Scenario"] == self.operation_scenario_id]
        for key in self.__dict__.keys():
            if key.endswith("_hour"):
                key_remove_tail = key[:-5]
                if key_remove_tail in operation_result_hour.columns:
                    self.__setattr__(key, operation_result_hour[key_remove_tail].to_numpy())

    def setup_operation_result_year(self, df: pd.DataFrame):
        operation_result_year: Dict[str, float] = df.loc[df["ID_Scenario"] ==
                                                         self.operation_scenario_id].iloc[0].to_dict()
        for key in self.__dict__.keys():
            if key.endswith("_year"):
                key_remove_tail = key[:-5]
                if key_remove_tail in operation_result_year.keys():
                    self.__setattr__(key, operation_result_year[key_remove_tail])
