from models.operation.analyzer import OperationAnalyzer
from config import config
from basics import kit

logger = kit.get_logger(__name__)

COMPONENT_CHANGES = [
        ("ID_Building", 3, 2),
        ("ID_Building", 3, 1),
        ("ID_Building", 2, 1),
        ("ID_SEMS", 2, 1),
        ("ID_Boiler", 2, 1),
        ("ID_SpaceHeatingTank", 2, 1),
        ("ID_HotWaterTank", 2, 1),
        # ("ID_SpaceCoolingTechnology", 2, 1),
        ("ID_PV", 2, 1),
        ("ID_Battery", 2, 1),
    ]


class ProjectOperationAnalyzer(OperationAnalyzer):

    def plot_scenario_comparison(self):
        # self.compare_opt_ref(1)
        self.compare_opt(id1=119, id2=126)
        # self.compare_ref(id1=1, id2=16)

    def plot_scenario_electricity_balance(self):
        scenario_ids = [1]
        winter_hours = (25, 192)
        summer_hours = (5761, 5928)
        hour_ranges = [winter_hours, summer_hours]
        models = ["opt", "ref"]
        for scenario_id in scenario_ids:
            for model in models:
                for hour_range in hour_ranges:
                    self.plot_electricity_balance(
                        scenario_id=scenario_id,
                        model=model,
                        start_hour=hour_range[0],
                        end_hour=hour_range[1],
                    )

    def summarize_operation_energy_cost(self):
        self.create_operation_energy_cost_table()
        self.plot_operation_energy_cost_curve()

    def summarize_operation_energy_cost_change(self):
        self.create_component_energy_cost_change_tables(COMPONENT_CHANGES)
        self.plot_operation_energy_cost_change_curve(COMPONENT_CHANGES)

    def plot_building_pathway(self):
        for scenario_id in range(192, 193):
            logger.info(f'Scenario = {scenario_id}')
            self.get_building_pathway(scenario_id, COMPONENT_CHANGES)


if __name__ == "__main__":
    ana = ProjectOperationAnalyzer(config)
    ana.summarize_operation_energy_cost()
    ana.summarize_operation_energy_cost_change()