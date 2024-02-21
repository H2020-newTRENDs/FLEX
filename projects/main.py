import os
import random
from projects.analysis.main_operation import run_operation_model
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
    # run_operation_model(config=config, save_hour=True, scenario_ids=[1])
    run_operation_model_parallel(config=config, task_num=8, save_hour=True)


def run_flex_operation_plotter(config: "Config"):
    household_load_balance(config, scenario_ids=random.sample(range(1, 4375), 100))


if __name__ == "__main__":

    cfg = get_config("DEU_2030")

    # run model and plotter
    # run_flex_operation_model(cfg)
    # run_flex_operation_plotter(cfg)

    # generate summary
    # generate_summary.gen_summary_year(cfg)
    # generate_summary.gen_summary_hour(cfg)

    # process summary
    process_summary.concat_summary()
    process_summary.process_summary_year()
    # process_summary.process_summary_hour()

