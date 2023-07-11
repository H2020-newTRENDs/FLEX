import logging
import os

import pandas as pd
import sqlalchemy

from flex.config import Config
from flex_operation.run import run_operation_model


def check_progress(conn):
    tables = sqlalchemy.engine.reflection.Inspector.from_engine(conn).get_table_names()
    opt_year = "OperationResult_OptimizationYear" in tables
    opt_month = "OperationResult_OptimizationMonth" in tables
    ref_year = "OperationResult_ReferenceYear" in tables
    ref_month = "OperationResult_ReferenceMonth" in tables
    if opt_year and opt_month and ref_year and ref_month:
        return True
    else:
        conn.execute(f"DROP TABLE IF EXISTS OperationResult_OptimizationYear")
        conn.execute(f"DROP TABLE IF EXISTS OperationResult_OptimizationMonth")
        conn.execute(f"DROP TABLE IF EXISTS OperationResult_ReferenceYear")
        conn.execute(f"DROP TABLE IF EXISTS OperationResult_ReferenceMonth")
        return False


def align_progress(conn):
    opt_year = pd.read_sql('select * from ' + "OperationResult_OptimizationYear", con=conn)["ID_Scenario"].tolist()[-1]
    opt_month = pd.read_sql('select * from ' + "OperationResult_OptimizationMonth", con=conn)["ID_Scenario"].tolist()[-12]
    ref_year = pd.read_sql('select * from ' + "OperationResult_ReferenceYear", con=conn)["ID_Scenario"].tolist()[-1]
    ref_month = pd.read_sql('select * from ' + "OperationResult_ReferenceMonth", con=conn)["ID_Scenario"].tolist()[-12]
    min_id = min(opt_year, opt_month, ref_year, ref_month)
    conn.execute(f"DELETE FROM OperationResult_OptimizationYear WHERE ID_Scenario >= '{min_id}'")
    conn.execute(f"DELETE FROM OperationResult_OptimizationMonth WHERE ID_Scenario >= '{min_id}'")
    conn.execute(f"DELETE FROM OperationResult_ReferenceYear WHERE ID_Scenario >= '{min_id}'")
    conn.execute(f"DELETE FROM OperationResult_ReferenceMonth WHERE ID_Scenario >= '{min_id}'")
    return min_id


def run_region(region: str):
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)
    cfg = Config(project_name=region)
    conn = sqlalchemy.create_engine(f'sqlite:///{os.path.join("data/output", region, region + ".sqlite")}')
    scenario_num = len(pd.read_sql('select * from ' + "OperationScenario", con=conn))
    if check_progress(conn):
        min_id = align_progress(conn)
    else:
        min_id = 1
    operation_scenario_ids = [id_scenario for id_scenario in range(min_id, scenario_num + 1)]
    run_operation_model(operation_scenario_ids, cfg)
