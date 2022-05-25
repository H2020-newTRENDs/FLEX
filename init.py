from basics.database_initializer import DatabaseInitializer
from models.operation.enums import ScenarioEnum, TableEnum
from config import config


class ProjectDatabaseInitializer(DatabaseInitializer):

    def load_component_scenario_tables(self):
        self.load_component_table(ScenarioEnum.Region)
        self.load_component_table(ScenarioEnum.Building)
        self.load_component_table(ScenarioEnum.Boiler)
        self.load_component_table(ScenarioEnum.SpaceHeatingTank)
        self.load_component_table(ScenarioEnum.HotWaterTank)
        self.load_component_table(ScenarioEnum.SpaceCoolingTechnology)
        self.load_component_table(ScenarioEnum.PV)
        self.load_component_table(ScenarioEnum.Battery)
        self.load_component_table(ScenarioEnum.Vehicle)
        self.load_component_table(ScenarioEnum.EnergyPrice)
        self.load_component_table(ScenarioEnum.Behavior)

    def load_other_source_tables(self):
        self.load_source_table(TableEnum.BehaviorProfile)
        self.load_source_table(TableEnum.EnergyPriceProfile)
        # self.load_source_table(TableEnum.RegionWeatherProfile)


if __name__ == "__main__":
    init = ProjectDatabaseInitializer(config=config, scenario_enums=ScenarioEnum)
    init.main()
