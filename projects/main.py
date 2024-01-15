import os
import random
from models.operation.main import run_operation_model
from models.operation.main import run_operation_model_parallel
from plotters.operation import household_load_balance
from utils.config import Config
from utils.db import init_project_db
from projects.analysis import calculator as calc


def get_config(project_name: str):
    return Config(
        project_name=project_name,
        project_path=os.path.join(os.path.dirname(__file__), f"{project_name}")
    )


def run_flex_operation_model(config: "Config"):
    init_project_db(config)
    # run_operation_model(config=config, save_hour=True, scenario_ids=[1])
    run_operation_model_parallel(config=config, task_num=8, save_hour=True)


def run_flex_operation_plotter(config: "Config"):
    household_load_balance(config, scenario_ids=random.sample(range(1, 4375), 100))


if __name__ == "__main__":

    cfg = get_config("DEU_2020")
    run_flex_operation_model(cfg)
    # run_flex_operation_plotter(cfg)
    calc.merge_scenario(cfg)
    calc.calc_electricity_profiles(cfg)
