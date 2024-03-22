import logging
from typing import List

import pandas as pd

from config import cfg
from flex.config import Config
from flex.db_init import DatabaseInitializer
from flex_behavior.constants import BehaviorTable
from flex_community.constants import CommunityTable
from flex_operation.constants import OperationComponentInfo
from flex_operation.constants import OperationScenarioComponent
from flex_operation.constants import OperationTable


class ProjectDatabaseInit(DatabaseInitializer):

    def __init__(self, config: "Config",
                 exclude_from_permutation: List[str] = None,
                 load_scenario_table: bool = True,
                 start_scenario_table: pd.DataFrame = None):
        super().__init__(config)
        self.exclude_from_permutation = exclude_from_permutation
        self.load_scenario_table = load_scenario_table
        self.start_scenario_table = start_scenario_table

    def load_behavior_tables(self):
        self.load_behavior_table(BehaviorTable.Scenarios)
        self.load_behavior_table(BehaviorTable.ID_HouseholdType)
        self.load_behavior_table(BehaviorTable.ID_PersonType)
        self.load_behavior_table(BehaviorTable.ID_Activity)
        self.load_behavior_table(BehaviorTable.ID_Location)
        self.load_behavior_table(BehaviorTable.ID_Technology)
        self.load_behavior_table(BehaviorTable.HouseholdComposition)
        self.load_behavior_table(BehaviorTable.ActivityChangeProb)
        self.load_behavior_table(BehaviorTable.ActivityDurationProb)
        self.load_behavior_table(BehaviorTable.ActivityLocation)
        self.load_behavior_table(BehaviorTable.TechnologyTriggerProbability)
        self.load_behavior_table(BehaviorTable.TechnologyPower)
        self.load_behavior_table(BehaviorTable.TechnologyDuration)

    def load_operation_component_tables(self):
        self.load_operation_table(OperationScenarioComponent.Region.table_name)
        self.load_operation_table(OperationScenarioComponent.Building.table_name)
        self.load_operation_table(OperationScenarioComponent.Boiler.table_name)
        self.load_operation_table(OperationScenarioComponent.HeatingElement.table_name)
        self.load_operation_table(OperationScenarioComponent.SpaceHeatingTank.table_name)
        self.load_operation_table(OperationScenarioComponent.HotWaterTank.table_name)
        self.load_operation_table(OperationScenarioComponent.SpaceCoolingTechnology.table_name)
        self.load_operation_table(OperationScenarioComponent.PV.table_name)
        self.load_operation_table(OperationScenarioComponent.Battery.table_name)
        self.load_operation_table(OperationScenarioComponent.Vehicle.table_name)
        self.load_operation_table(OperationScenarioComponent.EnergyPrice.table_name)
        self.load_operation_table(OperationScenarioComponent.Behavior.table_name)

    def setup_operation_scenario_table(self,):
        def generate(exclude_from_permutation: List[str] = None,
                     start_scenario_table: pd.DataFrame = None):
            component_table_names = {}
            for key, value in OperationScenarioComponent.__dict__.items():
                if isinstance(value, OperationComponentInfo):
                    component_table_names[value.id_name] = value.table_name
            self.generate_operation_scenario_table(
                scenario_table_name=OperationTable.Scenarios,
                component_table_names=component_table_names,
                exclude_from_permutation=exclude_from_permutation,
                start_scenario_table=start_scenario_table,
            )

        if self.load_scenario_table:
            self.load_operation_table(OperationTable.Scenarios)
        else:
            generate(exclude_from_permutation=self.exclude_from_permutation,
                     start_scenario_table=self.start_scenario_table)

    def load_operation_profile_tables(self):
        self.load_operation_table(OperationTable.BehaviorProfile)
        self.load_operation_table(OperationTable.DrivingProfile_ParkingHome)
        self.load_operation_table(OperationTable.DrivingProfile_Distance)
        self.load_operation_table(OperationTable.DrivingProfile_ChargingOutside)
        self.load_operation_table(OperationTable.EnergyPriceProfile)
        self.load_operation_table(OperationTable.RegionWeatherProfile)

    def load_community_tables(self):
        self.load_community_table(CommunityTable.Scenarios)

    def drop_tables(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        logger.info(f"Dropping Results for {self.config.project_name}")
        self.drop_table(OperationTable.ResultOptHour)
        self.drop_table(OperationTable.ResultOptYear)
        self.drop_table(OperationTable.ResultRefHour)
        self.drop_table(OperationTable.ResultRefYear)

    def run(self):
        self.load_behavior_tables()
        self.load_operation_component_tables()
        self.setup_operation_scenario_table()
        self.load_operation_profile_tables()
        self.load_community_tables()
        self.drop_tables()


if __name__ == "__main__":
    df_start = pd.read_excel(cfg.input_operation / 'Scenario_start_Murcia.xlsx')
    init = ProjectDatabaseInit(
        config=cfg,
        exclude_from_permutation=["ID_Building",
                                  "ID_PV",
                                  "ID_Battery",
                                  "ID_HotWaterTank",
                                  "ID_SpaceHeatingTank",
                                  "ID_HeatingElement",
                                  "ID_Boiler",
                                  "ID_Behavior",
                                  ],
        load_scenario_table=False,
        start_scenario_table=df_start)
    init.run()
