
from typing import List
from typing import Optional


import sqlalchemy
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
                    with db.engine.connect() as conn:
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





