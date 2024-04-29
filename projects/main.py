import os
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
        project_path=os.path.join(os.path.dirname(__file__), f"{project_name}")
    )


def run_flex_operation_model(config: "Config"):
    init_project_db(config)
    run_operation_model_parallel(config=config, task_num=8, save_hour=True)


def run_flex_operation_plotter(config: "Config"):
    household_load_balance(config, scenario_ids=random.sample(range(1, 4375), 100))


def run_scenario_and_generate_summary_files():
    scenarios = []
    for scenario in scenarios:
        cfg = get_config(scenario)
        # run_flex_operation_model(cfg)
        generate_summary.gen_summary_year(cfg)
        # generate_summary.gen_summary_hour(cfg)


def process_summary_files():
    # process_summary.concat_summary()
    # process_summary.start_from_8760h_summary()
    # process_summary.harmonize_with_invert()
    # process_summary.implement_iamc_formatting_year()
    # process_summary.mix_scenarios_year()
    # process_summary.implement_iamc_formatting_hour()
    # process_summary.mix_scenarios_hour()
    # process_summary.insert_pv_2_grid_hour()
    process_summary.filter_aut()


if __name__ == "__main__":

    process_summary_files()


