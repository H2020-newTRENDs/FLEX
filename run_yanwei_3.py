import logging
import os

import pandas as pd
import sqlalchemy

from db_init import ProjectDatabaseInit
from flex.config import Config
from flex_operation.run import run_operation_model


def init_region_db(region: str):
    cfg = Config(project_name=region)
    init = ProjectDatabaseInit(config=cfg)
    init.run()


def run_region(region: str):
    logging.getLogger("pyomo.core").setLevel(logging.ERROR)
    cfg = Config(project_name=region)
    conn = sqlalchemy.create_engine(f'sqlite:///{os.path.join("data/output", region, region + ".sqlite")}')
    scenario_num = len(pd.read_sql('select * from ' + "OperationScenario", con=conn))
    operation_scenario_ids = [id_scenario for id_scenario in range(1, scenario_num + 1)]
    run_operation_model(operation_scenario_ids, cfg)


if __name__ == "__main__":
    regions = [
        # 'SK', 'SE', 'PL', 'MT', 'BE', 'EE', 'EL',
        # 'LV', 'IT', 'CZ', 'RO', 'PT', 'SI', 'HR',
        'HU', 'NL', 'BG', 'AT', 'DE', 'DK', 'FI',
        # 'LU', 'FR', 'ES', 'IE', 'LT', 'CY'
    ]
    for index, region in enumerate(regions):
        print(f'{region} --> {index + 1}/{len(regions)}')
        # init_region_db(region)
        run_region(region)







