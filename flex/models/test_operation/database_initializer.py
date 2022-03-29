from flex.core.database_initializer import DatabaseInitializer
from flex.models.test_operation.component import OperationComponentEnum
from flex.models.test_operation.scenario import OperationScenarioEnum


class OperationDatabaseInitializer(DatabaseInitializer):

    def get_scenario_enum(self):
        return OperationScenarioEnum

    def get_component_enum(self):
        return OperationComponentEnum

    def setup_component_scenarios(self):
        pass
