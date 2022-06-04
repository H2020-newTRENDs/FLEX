from models.operation.analyzer import OperationAnalyzer
from config import config


class ProjectOperationAnalyzer(OperationAnalyzer):

    def plot_scenario_comparison(self):
        self.compare_opt_ref(1)
        self.compare_opt(id1=1, id2=16)
        self.compare_ref(id1=1, id2=16)

    def plot_scenario_electricity_balance(self):
        scenario_ids = [1, 96]
        hour_ranges = [(25, 192), (5761, 5928)]
        models = ["opt", "ref"]
        for scenario_id in scenario_ids:
            for model in models:
                for hour_range in hour_ranges:
                    self.plot_electricity_balance(
                        scenario_id=scenario_id,
                        model=model,
                        start_hour=hour_range[0],
                        end_hour=hour_range[1]
                    )

    def summarize_operation_energy_cost(self):
        self.create_operation_energy_cost_table()
        self.plot_operation_energy_cost_curve()

    def summarize_operation_energy_cost_change(self):
        component_changes = [
            ("ID_Building", 3, 2),
            ("ID_Building", 2, 1),
            ("ID_Building", 3, 1),
            ("ID_SEMS", 2, 1),
            ("ID_Boiler", 2, 1),
            ("ID_SpaceHeatingTank", 2, 1),
            ("ID_HotWaterTank", 2, 1),
            ("ID_PV", 2, 1),
            ("ID_Battery", 2, 1)
        ]
        self.create_component_energy_cost_change_tables(component_changes)
        self.plot_operation_energy_cost_change_curve(component_changes)
        self.get_building_pathway(191, component_changes)


if __name__ == "__main__":
    ana = ProjectOperationAnalyzer(config)
    ana.plot_scenario_electricity_balance()
    ana.summarize_operation_energy_cost()
    ana.summarize_operation_energy_cost_change()





