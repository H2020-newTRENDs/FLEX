from basics.database_initializer import DatabaseInitializer
from models.operation.enums import OperationScenarioComponent, OperationTable
from config import config


class ProjectOperationInit(DatabaseInitializer):

    def load_component_tables(self):
        self.load_component_table(OperationScenarioComponent.Region)
        self.load_component_table(OperationScenarioComponent.Building)
        self.load_component_table(OperationScenarioComponent.Boiler)
        self.load_component_table(OperationScenarioComponent.SpaceHeatingTank)
        self.load_component_table(OperationScenarioComponent.HotWaterTank)
        self.load_component_table(OperationScenarioComponent.SpaceCoolingTechnology)
        self.load_component_table(OperationScenarioComponent.PV)
        self.load_component_table(OperationScenarioComponent.Battery)
        self.load_component_table(OperationScenarioComponent.Vehicle)
        self.load_component_table(OperationScenarioComponent.EnergyPrice)
        self.load_component_table(OperationScenarioComponent.Behavior)

    def setup_scenario_table(self):
        # self.load_table(OperationTable.Scenarios)
        self.generate_scenario_table(table_name=OperationTable.Scenarios.value)

    def load_other_tables(self):
        self.load_table(OperationTable.BehaviorProfile)
        self.load_table(OperationTable.EnergyPriceProfile)
        self.load_table(OperationTable.RegionWeatherProfile)

    def drop_tables(self):
        self.drop_table(OperationTable.ResultOptHour)
        self.drop_table(OperationTable.ResultOptYear)
        self.drop_table(OperationTable.ResultRefHour)
        self.drop_table(OperationTable.ResultRefYear)

    def main(self):
        self.load_component_tables()
        self.setup_scenario_table()
        self.load_other_tables()
        self.drop_tables()


if __name__ == "__main__":
    init = ProjectOperationInit(
        config=config,
        input_folder=config.input_operation,
        scenario_components=OperationScenarioComponent,
    )
    init.main()
