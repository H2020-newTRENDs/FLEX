import math
import os
import shutil
from typing import List
from typing import Optional
import time

import pandas as pd
import sqlalchemy
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
from pathlib import Path

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


def get_latest_scenario_ids(database) -> [int, int]:
    latest_scenario_ids = []
    db_tables = database.get_table_names()
    for result_table in DB_RESULT_TABLES:
        if result_table in db_tables:
            latest_scenario_ids.append(database.read_dataframe(result_table)["ID_Scenario"].to_list()[-1])
    return latest_scenario_ids


def drop_until(lst, target_value):
    for i, value in enumerate(lst):
        if value == target_value:
            return lst[i:]
    return []


def align_progress(initial_scenario_ids, database):
    latest_scenario_ids = get_latest_scenario_ids(database)
    if len(latest_scenario_ids) > 0 and len(set(latest_scenario_ids)) != 1:
        latest_scenario_id = min(latest_scenario_ids)
        db_tables = database.get_table_names()
        # in case the latest scenario id was saved as reference already, delete it so we dont have double entries:
        for result_table in DB_RESULT_TABLES:
            if result_table in db_tables:
                with database.engine.connect() as conn:
                    conn.execute(sqlalchemy.text(f"DELETE FROM {result_table} WHERE ID_Scenario >= '{latest_scenario_id}'"))
                    conn.commit()
        updated_scenario_ids = drop_until(initial_scenario_ids, latest_scenario_id)
    else:
        updated_scenario_ids = initial_scenario_ids
    return updated_scenario_ids


def run_operation_model(config: "Config",
                        scenario_ids: Optional[List[int]] = None,
                        run_ref: bool = True,
                        run_opt: bool = True,
                        save_year: bool = True,
                        save_month: bool = False,
                        save_hour: bool = False,
                        hour_vars: List[str] = None):

    db = create_db_conn(config)
    input_tables = fetch_input_tables(config)
    if scenario_ids is None:
        scenario_ids = input_tables[InputTables.OperationScenario.name]["ID_Scenario"].to_list()
    scenario_ids = align_progress(scenario_ids, db)
    opt_instance = OptInstance().create_instance()
    for scenario_id in tqdm(scenario_ids, desc=f"{config.project_name}"):
        scenario = OperationScenario(config=config, scenario_id=scenario_id, input_tables=input_tables)
        if run_ref:
            run_ref_model(scenario=scenario, config=config, save_year=save_year, save_month=save_month,
                          save_hour=save_hour, hour_vars=hour_vars)
        if run_opt:
            run_opt_model(opt_instance=opt_instance, scenario=scenario, config=config, save_year=save_year,
                          save_month=save_month, save_hour=save_hour, hour_vars=hour_vars)


def merge_year_month_tables(number_of_tasks, original_config):
    for table_name in DB_RESULT_TABLES:
        table_exists = False
        task_results = []
        for task_id in range(1, number_of_tasks + 1):
            task_db = create_db_conn(original_config.make_copy().set_task_id(task_id=task_id))
            if table_name in task_db.get_table_names():
                table_exists = True
                task_results.append(task_db.read_dataframe(table_name))
            else:
                break
        if table_exists:
            create_db_conn(original_config).write_dataframe(
                table_name=table_name,
                data_frame=pd.concat(task_results, ignore_index=True)
            )


def move_hour_parquets(number_of_tasks, original_config):
    for task_id in range(1, number_of_tasks + 1):
        task_config = original_config.make_copy().set_task_id(task_id=task_id)
        for file_name in os.listdir(task_config.task_output):
            if file_name.endswith(".parquet.gzip"):
                shutil.move(os.path.join(task_config.task_output, file_name),
                            os.path.join(task_config.output, file_name))


def split_scenarios(number_of_tasks, original_config):
    total_scenario_num = len(create_db_conn(original_config).read_dataframe(InputTables.OperationScenario.name))
    task_scenario_num = math.ceil(total_scenario_num / number_of_tasks)
    for task_id in range(1, number_of_tasks + 1):
        db = create_db_conn(original_config.make_copy().set_task_id(task_id=task_id))
        df = db.read_dataframe(InputTables.OperationScenario.name)
        if task_id < number_of_tasks:
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


def check_if_some_results_exist(number_of_tasks, original_config, multi: int) -> bool:
    results_exist = []
    for task_id in range(1, number_of_tasks + 1):
        task_config = original_config.make_copy().set_task_id(task_id=task_id)

        for file in Path(task_config.task_output).iterdir():
            if file.suffix == ".sqlite":
                db = create_db_conn(task_config)
                db_tables = db.get_table_names()
                for result_table in DB_RESULT_TABLES:
                    if result_table in db_tables:
                        results_exist.append(True)
    # result exists if all entries in list are "True", then the
    # calculation can be continued. Otherwise it will be restarted.
    if all(results_exist) and len(results_exist) > 0:
        return True
    else:
        return False


def create_task_dbs(number_of_tasks, original_config):
    for task_id in range(1, number_of_tasks + 1):
        task_config = original_config.make_copy().set_task_id(task_id=task_id)
        shutil.copy(os.path.join(task_config.output, f'{original_config.project_name}.sqlite'),
                    os.path.join(task_config.task_output, f'{original_config.project_name}.sqlite'))


def delete_file(file_path):
    """Attempt to delete a file with retries on PermissionError."""
    max_attempts = 5
    for attempt in range(max_attempts):
        file_deleted = False
        try:
            file_path.unlink()
            # print(f"File {file_path} deleted successfully.")
            file_deleted = True
            break
        except PermissionError as e:
            print(f"PermissionError on attempt {attempt+1}: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    return file_deleted


def delete_result_task_folders(conf):
    task_folder = Path(conf.output)  # Assuming you have a path attribute
    for item in task_folder.iterdir():
        if item.is_file():
            continue  # dont delete the result files in the real folder! only in sub-folders
        elif item.is_dir() and item.name != "figure":
            all_files_deleted = []
            for sub_item in item.iterdir():
                all_files_deleted.append(delete_file(sub_item))
            if all(all_files_deleted):
                item.rmdir()  # Remove the directory after it's emptied
            else:
                print(f"some files could not be deleted, skipping deletion of {item}")


def remove_task_folders(number_of_tasks, original_config):
    for task_id in range(1, number_of_tasks + 1):
        task_config = original_config.make_copy().set_task_id(task_id=task_id)
        # Ensure that the connection to the SQLite database is closed
        engine = create_db_conn(task_config)
        engine.dispose()  # dispose all connections
    delete_result_task_folders(original_config)


def run_operation_model_parallel(
    config: "Config",
    task_num: int,
    run_ref: bool = True,
    run_opt: bool = True,
    save_year: bool = True,
    save_month: bool = False,
    save_hour: bool = False,
    hour_vars: List[str] = None,
    reset_task_dbs: bool = False
):

    # if the optimization has not been started yet, initialise the different task dbs:
    # if results exist but they need to be re-calculated:
    multiplier = sum([save_year, save_month])
    if not check_if_some_results_exist(number_of_tasks=task_num, original_config=config, multi=multiplier) or reset_task_dbs:
        create_task_dbs(number_of_tasks=task_num, original_config=config)
        split_scenarios(number_of_tasks=task_num, original_config=config)

    # run the tasks
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

    # merge task results
    merge_year_month_tables(number_of_tasks=task_num, original_config=config)
    move_hour_parquets(number_of_tasks=task_num, original_config=config)
    remove_task_folders(number_of_tasks=task_num, original_config=config)



