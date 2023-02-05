from db_init import ProjectDatabaseInit
import logging
from flex.config import Config
from flex_operation.run import run_operation_model
logging.getLogger("pyomo.core").setLevel(logging.ERROR)


def run_init(config):
    init = ProjectDatabaseInit(config=config)
    init.run()


def run_scenarios(config):
    operation_scenario_ids = [id_scenario for id_scenario in range(1, 24001)]
    run_operation_model(operation_scenario_ids, config)


if __name__ == "__main__":
    cfg = Config(project_name="EV5000")
    # run_init(cfg)
    run_scenarios(cfg)
