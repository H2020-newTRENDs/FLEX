import os

from models.operation.main import run_operation_model
from plotters.operation import household_load_balance
from utils.config import Config
from utils.db import init_project_db


def get_config(project_name: str):
    return Config(
        project_name=project_name,
        project_path=os.path.join(os.path.dirname(__file__), f"{project_name}")
    )


def run_flex_operation_model(config: "Config"):
    init_project_db(config)
    run_operation_model(config=config, save_hour=True)


def run_flex_operation_plotter(config: "Config"):
    household_load_balance(config, scenario_ids=[1])


if __name__ == "__main__":

    cfg = get_config("TEST")
    run_flex_operation_model(cfg)
    # run_flex_operation_plotter(cfg)
