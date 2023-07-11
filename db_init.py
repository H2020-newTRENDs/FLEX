from config import cfg
from flex.db_init import DatabaseInitializer
from flex_behavior.constants import BehaviorTable
from flex_community.constants import CommunityTable
from flex_operation.constants import OperationComponentInfo
from flex_operation.constants import OperationScenarioComponent
from flex_operation.constants import OperationTable


class ProjectDatabaseInit(DatabaseInitializer):

    def load_behavior_tables(self):
        self.load_behavior_table(BehaviorTable.Scenarios)
        self.load_behavior_table(BehaviorTable.ID_Activity)
        self.load_behavior_table(BehaviorTable.ActivityProfile)

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

    def setup_operation_scenario_table(self):

        def generate():
            component_table_names = {}
            for key, value in OperationScenarioComponent.__dict__.items():
                if isinstance(value, OperationComponentInfo):
                    component_table_names[value.id_name] = value.table_name
            self.generate_operation_scenario_table(OperationTable.Scenarios, component_table_names)

        # generate()
        self.load_operation_table(OperationTable.Scenarios)

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
        # self.drop_tables()


if __name__ == "__main__":
    init = ProjectDatabaseInit(config=cfg)
    init.run()
