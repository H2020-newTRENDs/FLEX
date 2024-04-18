import os
import random
from typing import List
from projects.analysis.main_operation import run_operation_model
from models.operation.main import run_operation_model_parallel
from plotters.operation import household_load_balance
from utils.config import Config
from utils.db import init_project_db
from projects.analysis import generate_summary
from projects.analysis import process_summary


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


def run_scenario_and_generate_summary_files(scenarios: List[str]):
    for scenario in scenarios:
        cfg = get_config(scenario)
        run_flex_operation_model(cfg)
        generate_summary.gen_summary_year(cfg)
        generate_summary.gen_summary_hour(cfg)


def process_summary_files():
    process_summary.concat_summary()
    process_summary.process_summary_year()
    process_summary.process_summary_hour()
    process_summary.compare_hwb_diff()


if __name__ == "__main__":

    SCENARIOS = [
        "FIN_2040",
        "GRC_2040",
        "LTU_2040",
        "LVA_2040",
        "ROU_2040",
        "SWE_2040",
    ]
    run_scenario_and_generate_summary_files(SCENARIOS)


