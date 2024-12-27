from pathlib import Path
import random
from typing import List
from projects.analysis.main_operation import run_operation_model_parallel
from plotters.operation import household_load_balance
from utils.config import Config
from utils.db import init_project_db
from projects.analysis import generate_summary
from projects.analysis import process_summary


def get_config(project_name: str):
    return Config(
        project_name=project_name,
        project_path=Path(__file__).parent / f"{project_name}"
    )


def run_flex_operation_model(config: "Config", task_number: int):
    init_project_db(config)
    run_operation_model_parallel(config=config, task_num=task_number, save_hour=True, reset_task_dbs=True)


def run_flex_operation_plotter(config: "Config"):
    household_load_balance(config, scenario_ids=random.sample(range(1, 4375), 100))


if __name__ == "__main__":
    conf = get_config(
        project_name="electricity_price_check",
    )
    run_flex_operation_model(config=conf, task_number=2)
    


