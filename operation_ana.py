from models.operation.analyzer import OperationAnalyzer
from config import config


class ProjectOperationAnalyzer(OperationAnalyzer):
    pass


if __name__ == "__main__":
    ana = ProjectOperationAnalyzer(config)
    # ana.compare_opt_ref(1)
    # ana.compare_opt(id1=1, id2=16)
    # ana.compare_ref(id1=1, id2=16)
    ana.plot_operation_energy_cost_curve()
    # ana.plot_electricity_balance(scenario_id=1, model="opt", start_hour=25, end_hour=192)
    # ana.plot_electricity_balance(scenario_id=1, model="ref", start_hour=25, end_hour=192)
    # ana.plot_electricity_balance(scenario_id=1, model="opt", start_hour=5761, end_hour=5928)
    # ana.plot_electricity_balance(scenario_id=1, model="ref", start_hour=5761, end_hour=5928)

