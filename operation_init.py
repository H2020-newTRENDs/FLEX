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

    def load_scenario_table(self):
        self.load_table(OperationTable.Scenarios.value)

    def load_other_tables(self):
        self.load_table(OperationTable.BehaviorProfile.value)
        self.load_table(OperationTable.EnergyPriceProfile.value)
        self.load_table(OperationTable.RegionWeatherProfile.value)

    def drop_tables(self):
        self.db.drop_table(OperationTable.ResultOptHour.value)
        self.db.drop_table(OperationTable.ResultOptYear.value)
        self.db.drop_table(OperationTable.ResultRefHour.value)
        self.db.drop_table(OperationTable.ResultRefYear.value)

    def main(self):
        self.load_component_tables()
        self.load_scenario_table()
        self.load_other_tables()
        self.drop_tables()


if __name__ == "__main__":
    init = ProjectOperationInit(
        config=config,
        input_folder=config.input_operation,
        scenario_components=OperationScenarioComponent,
    )
    init.main()
