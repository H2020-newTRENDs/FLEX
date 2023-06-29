import pandas as pd
import sqlalchemy

import os


folder_path = 'data/output'  # Specify the path to the folder you want to explore
folder_names = [name for name in os.listdir(folder_path) if
                os.path.isdir(os.path.join(folder_path, name)) and
                not name.startswith("EV") and
                not name.startswith("no")]


for index, region in enumerate(folder_names):
    print(f'{region} --> {index + 1}/{len(folder_names)}')
    conn = sqlalchemy.create_engine(f'sqlite:///{os.path.join(folder_path, region, region + ".sqlite")}')
    df = pd.read_sql('select * from ' + "OperationScenario", con=conn)
    df1 = df.loc[(df["ID_SpaceHeatingTank"] == 1) & (df["ID_HotWaterTank"] == 1)]
    df2 = df.loc[(df["ID_SpaceHeatingTank"] == 2) & (df["ID_HotWaterTank"] == 2)]
    concatenated_df = pd.concat([df1, df2], axis=0)
    concatenated_df.to_excel(f'data/input_operation/{region}/OperationScenario.xlsx', index=False)

# df = pd.read_excel("data/input_operation/EV1000/OperationScenario_RegionWeather.xlsx")
# for index, region in enumerate(folder_names):
#     print(f'{region} --> {index + 1}/{len(folder_names)}')
#     df_region = df.loc[df["region"] == region]
#     df_region.to_excel(f"data/input_operation/{region}/OperationScenario_RegionWeather.xlsx", index=False)
