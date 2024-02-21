import math
import os
import shutil
from typing import List
from typing import Optional

import pandas as pd
import sqlalchemy
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm

from models.operation.data_collector import OptDataCollector
from models.operation.data_collector import RefDataCollector
from models.operation.model_opt import OptInstance
from models.operation.model_opt import OptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.scenario import OperationScenario
from utils.config import Config
from utils.db import create_db_conn
from utils.db import fetch_input_tables
from utils.tables import InputTables
from utils.tables import OutputTables

DB_RESULT_TABLES = [
        OutputTables.OperationResult_RefYear.name,
        OutputTables.OperationResult_OptYear.name,
        OutputTables.OperationResult_RefMonth.name,
        OutputTables.OperationResult_OptMonth.name
    ]


def run_ref_model(
    scenario: "OperationScenario",
    config: "Config",
    save_year: bool = True,
    save_month: bool = False,
    save_hour: bool = False,
    hour_vars: Optional[List[str]] = None
):
    ref_model = RefOperationModel(scenario).solve()
    RefDataCollector(model=ref_model,
                     scenario_id=scenario.scenario_id,
                     config=config,
                     save_year=save_year,
                     save_month=save_month,
                     save_hour=save_hour,
                     hour_vars=hour_vars).run()


def run_opt_model(
    opt_instance,
    scenario: "OperationScenario",
    config: "Config",
    save_year: bool = True,
    save_month: bool = False,
    save_hour: bool = False,
    hour_vars: Optional[List[str]] = None
):
    opt_model, solve_status = OptOperationModel(scenario).solve(opt_instance)
    if solve_status:
        OptDataCollector(model=opt_model,
                         scenario_id=scenario.scenario_id,
                         config=config,
                         save_year=save_year,
                         save_month=save_month,
                         save_hour=save_hour,
                         hour_vars=hour_vars).run()


def run_operation_model(config: "Config",
                        scenario_ids: Optional[List[int]] = None,
                        run_ref: bool = True,
                        run_opt: bool = True,
                        save_year: bool = True,
                        save_month: bool = False,
                        save_hour: bool = False,
                        hour_vars: List[str] = None):

    def align_progress(initial_scenario_ids):

        def get_latest_scenario_ids():
            latest_scenario_ids = []
            db_tables = db.get_table_names()
            for result_table in DB_RESULT_TABLES:
                if result_table in db_tables:
                    latest_scenario_ids.append(db.read_dataframe(result_table)["ID_Scenario"].to_list()[-1])
            return latest_scenario_ids

        def drop_until(lst, target_value):
            for i, value in enumerate(lst):
                if value == target_value:
                    return lst[i:]
            return []

        latest_scenario_ids = get_latest_scenario_ids()
        if len(latest_scenario_ids) > 0:
            latest_scenario_id = min(latest_scenario_ids)
            db_tables = db.get_table_names()
            for result_table in DB_RESULT_TABLES:
                if result_table in db_tables:
                    with db.connection.connect() as conn:
                        result = conn.execute(sqlalchemy.text(f"DELETE FROM {result_table} WHERE ID_Scenario >= '{latest_scenario_id}'"))
            updated_scenario_ids = drop_until(initial_scenario_ids, latest_scenario_id)
        else:
            updated_scenario_ids = initial_scenario_ids
        return updated_scenario_ids

    db = create_db_conn(config)
    input_tables = fetch_input_tables(config)
    if scenario_ids is None:
        scenario_ids = input_tables[InputTables.OperationScenario.name]["ID_Scenario"].to_list()
    scenario_ids = align_progress(scenario_ids)
    opt_instance = OptInstance().create_instance()
    for scenario_id in tqdm(scenario_ids, desc=f"{config.project_name}"):
        scenario = OperationScenario(config=config, scenario_id=scenario_id, input_tables=input_tables)
        if run_ref:
            run_ref_model(scenario=scenario, config=config, save_year=save_year, save_month=save_month,
                          save_hour=save_hour, hour_vars=hour_vars)
        if run_opt:
            if scenario.boiler.type == "Air_HP" or scenario.battery.capacity > 0:
                run_opt_model(opt_instance=opt_instance, scenario=scenario, config=config, save_year=save_year,
                              save_month=save_month, save_hour=save_hour, hour_vars=hour_vars)


def run_operation_model_parallel(
    config: "Config",
    task_num: int,
    run_ref: bool = True,
    run_opt: bool = True,
    save_year: bool = True,
    save_month: bool = False,
    save_hour: bool = False,
    hour_vars: List[str] = None,
    reset_task_dbs: bool = True
):

    def create_task_dbs():
        for task_id in range(1, task_num + 1):
            task_config = config.make_copy().set_task_id(task_id=task_id)
            shutil.copy(os.path.join(task_config.output, f'{config.project_name}.sqlite'),
                        os.path.join(task_config.task_output, f'{config.project_name}.sqlite'))

    def split_scenarios():
        total_scenario_num = len(create_db_conn(config).read_dataframe(InputTables.OperationScenario.name))
        task_scenario_num = math.ceil(total_scenario_num / task_num)
        for task_id in range(1, task_num + 1):
            db = create_db_conn(config.make_copy().set_task_id(task_id=task_id))
            df = db.read_dataframe(InputTables.OperationScenario.name)
            if task_id < task_num:
                lower = 1 + task_scenario_num * (task_id - 1)
                upper = task_scenario_num * task_id
                task_scenario_df = df.loc[(df["ID_Scenario"] >= lower) & (df["ID_Scenario"] <= upper)]
            else:
                lower = 1 + task_scenario_num * (task_id - 1)
                task_scenario_df = df.loc[df["ID_Scenario"] >= lower]
            db.write_dataframe(
                table_name=InputTables.OperationScenario.name,
                data_frame=task_scenario_df,
                if_exists="replace"
            )

    def run_tasks():
        tasks = [
            {
                "config": config.make_copy().set_task_id(task_id=task_id),
                "run_ref": run_ref,
                "run_opt": run_opt,
                "save_year": save_year,
                "save_month": save_month,
                "save_hour": save_hour,
                "hour_vars": hour_vars
            }
            for task_id in range(1, task_num + 1)
        ]
        Parallel(n_jobs=task_num)(delayed(run_operation_model)(**task) for task in tasks)

    def merge_task_results():

        def merge_year_month_tables():
            for table_name in DB_RESULT_TABLES:
                table_exists = False
                task_results = []
                for task_id in range(1, task_num + 1):
                    task_db = create_db_conn(config.make_copy().set_task_id(task_id=task_id))
                    if table_name in task_db.get_table_names():
                        table_exists = True
                        task_results.append(task_db.read_dataframe(table_name))
                    else:
                        break
                if table_exists:
                    create_db_conn(config).write_dataframe(
                        table_name=table_name,
                        data_frame=pd.concat(task_results, ignore_index=True)
                    )

        def move_hour_parquets():
            for task_id in range(1, task_num + 1):
                task_config = config.make_copy().set_task_id(task_id=task_id)
                for file_name in os.listdir(task_config.task_output):
                    if file_name.endswith(".parquet.gzip"):
                        shutil.move(os.path.join(task_config.task_output, file_name),
                                    os.path.join(task_config.output, file_name))

        merge_year_month_tables()
        move_hour_parquets()

    def remove_task_folders():
        for task_id in range(1, task_num + 1):
            task_config = config.make_copy().set_task_id(task_id=task_id)
            shutil.rmtree(task_config.task_output)

    if reset_task_dbs:
        create_task_dbs()
        split_scenarios()
    run_tasks()
    merge_task_results()
    # remove_task_folders()


