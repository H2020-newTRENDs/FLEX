from flex.core.database_initializer import DatabaseInitializer
from flex.flex_operation.component import OperationComponentEnum
from flex.flex_operation.scenario import OperationScenarioEnum


class OperationDatabaseInitializer(DatabaseInitializer):

    def get_scenario_enum(self):
        return OperationScenarioEnum

    def get_component_enum(self):
        return OperationComponentEnum

    def setup_component_scenario(self):
        pass
