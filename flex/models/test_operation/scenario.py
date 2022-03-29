import sys
from dataclasses import dataclass
from typing import Optional

from flex.core.config import Config
from flex.core.db import create_db_conn
from flex.core.scenario import Scenario, ScenarioEnum
from .component import Region, Year, Building, Boiler, SpaceHeatingTank, HotWaterTank, \
    SpaceCoolingTechnology, PV, Battery, Vehicle, SEMS, OperationComponentEnum


@dataclass
class OperationScenario(Scenario):
    scenario_id: int
    config: 'Config'
    region: Optional['Region'] = None
    year: Optional['Year'] = None
    building: Optional['Building'] = None
    boiler: Optional['Boiler'] = None
    space_heating_tank: Optional['SpaceHeatingTank'] = None
    hot_water_tank: Optional['HotWaterTank'] = None
    space_cooling_technology: Optional['SpaceCoolingTechnology'] = None
    pv: Optional['PV'] = None
    battery: Optional['Battery'] = None
    vehicle: Optional['Vehicle'] = None
    sems: Optional['SEMS'] = None

    def __post_init__(self):
        self.db = create_db_conn(self.config)
        self.component_scenario_ids = self.get_component_scenario_ids()
        self.setup_components()

    def get_component_scenario_ids(self):
        component_scenario_ids = self.get_scenario_data_dict(
            OperationScenarioEnum.Scenario.table_name,
            scenario_filter={OperationScenarioEnum.Scenario.id: self.scenario_id}
        )
        return component_scenario_ids

    def setup_components(self):
        for id_component, component_scenario_id in self.component_scenario_ids.items():
            if id_component != OperationScenarioEnum.Scenario.id:
                component_name_cap = id_component.replace("ID_", "")
                component_enum = OperationComponentEnum.__dict__[component_name_cap]
                if component_enum.value in self.__dict__.keys():
                    component_params = self.get_scenario_data_dict(
                        component_enum.table_name,
                        scenario_filter={component_enum.id: component_scenario_id}
                    )
                    instance = getattr(sys.modules[__name__], component_name_cap)()
                    instance.set_params(component_params)
                    setattr(self, component_enum.value, instance)


class OperationScenarioEnum(ScenarioEnum):
    Scenario = "scenario"

    @property
    def table_name(self):
        return "Operation" + self.name
