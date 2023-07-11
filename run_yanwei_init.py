import os

import pandas as pd
import sqlalchemy

from db_init import ProjectDatabaseInit
from flex.config import Config


def get_eu27_names():
    folder_path = 'data/output'
    return [
        name for name in os.listdir(folder_path) if
        os.path.isdir(os.path.join(folder_path, name)) and
        not name.startswith("EV") and
        not name.startswith("no")
    ]


def check_temperature():
    folder_path = 'data/output'
    eu27_names = get_eu27_names()
    for index, region in enumerate(eu27_names):
        conn = sqlalchemy.create_engine(f'sqlite:///{os.path.join(folder_path, region, region + ".sqlite")}')
        df = pd.read_sql('select * from ' + "OperationScenario_RegionWeather", con=conn)
        print(f'{region} --> {index + 1}/{len(eu27_names)}: [{min(df["temperature"])}, {max(df["temperature"])}]')


def gen_new_scenario_table():
    folder_path = 'data/output'
    eu27_names = get_eu27_names()

    for index, region in enumerate(eu27_names):
        print(f'{region} --> {index + 1}/{len(eu27_names)}')
        conn = sqlalchemy.create_engine(f'sqlite:///{os.path.join(folder_path, region, region + ".sqlite")}')
        df = pd.read_sql('select * from ' + "OperationScenario", con=conn)
        df1 = df.loc[(df["ID_SpaceHeatingTank"] == 1) & (df["ID_HotWaterTank"] == 1)]
        df2 = df.loc[(df["ID_SpaceHeatingTank"] == 2) & (df["ID_HotWaterTank"] == 2)]
        concatenated_df = pd.concat([df1, df2], axis=0)
        del concatenated_df["ID_Scenario"]
        concatenated_df.insert(0, 'ID_Scenario', range(1, len(concatenated_df) + 1))
        concatenated_df.to_excel(f'data/input_operation/{region}/OperationScenario.xlsx', index=False)


def init_region_db(region: str):
    cfg = Config(project_name=region)
    init = ProjectDatabaseInit(config=cfg)
    init.run()


def init_eu27():
    regions = [
        # 'SK', 'SE', 'PL', 'MT', 'BE', 'EE', 'EL',
        # 'LV', 'IT', 'CZ', 'RO', 'PT', 'SI', 'HR',
        # 'HU', 'NL', 'BG', 'AT', 'DE', 'DK', 'FI',
        # 'LU', 'FR', 'ES', 'IE', 'LT', 'CY'
    ]
    for index, region in enumerate(regions):
        print(f'{region} --> {index + 1}/{len(regions)}')
        init_region_db(region)


if __name__ == "__main__":
    # init_eu27()
    # gen_new_scenario_table()
    check_temperature()









