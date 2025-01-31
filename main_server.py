from projects.main import get_config
from projects.main import run_flex_operation_model
from projects.main import generate_summary, process_summary
import shutil
from pathlib import Path
from utils.db import init_project_db
from projects.analysis.main_operation import run_operation_model_parallel
import pandas as pd
from utils.db import DB
import seaborn as sns
import matplotlib.pyplot as plt
def delete_result_files(conf):
    # Iterate through each item in the directory
    for item in Path(conf.output).iterdir():
        # Check if the item is a file
        if item.is_file():
            # Check the file extension and delete if it's not .csv or .parquet
            if item.suffix in ['.gzip']:
                # print(f"Deleting: {item.name}")
                item.unlink()  # Delete the file


def delete_result_task_folders(conf):
    for item in Path(conf.output).iterdir():
        if item.is_dir():
            print(f"Deleting directory and all contents: {item}")
            for sub_item in item.iterdir():
                # Check if the sub_item is a file
                if sub_item.is_file():
                    sub_item.unlink()  # Delete the file
            shutil.rmtree(item)

def run_only_ref_model_and_change_indoor_set_temp_until_correct(config):
    # invert heating demand (hwb):
    invert = pd.read_csv(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/testing/Country_level_prosumaging") / "hwb_Invert_HP_heated_buildings.csv")
    target_hwb = invert.loc[(invert["year"] == int(config.project_name.split("_")[1])) & (invert["country"] == config.project_name.split("_")[0]) , "avg_hwb"].values[0]
    # change the indoor set temperature
    percentage_diff = 1
    indoor_set_temp = 20
    change = 0
    change_step = 0.5
    diff_list = [100]
    while abs(percentage_diff) > 0.03:
        df = pd.read_csv(config.input /  "OperationScenario_Component_Behavior.csv", sep=";")
        indoor_set_temp = indoor_set_temp + change
        df["target_temperature_at_home_min"] = indoor_set_temp
        df["target_temperature_not_at_home_min"] = indoor_set_temp
        df.to_csv(config.input /  "OperationScenario_Component_Behavior.csv", sep=";", index=False)

        # run the model
        init_project_db(config)
        run_operation_model_parallel(config=config, task_num=10, save_hour=True, reset_task_dbs=True, run_ref=True, run_opt=False)
        
        
        db = DB(path=config.sqlite_path)    
        scenario_table = db.read_dataframe(table_name="OperationScenario")
        scenario_table = scenario_table[scenario_table["ID_SpaceCoolingTechnology"] == 1]
        building_table = db.read_dataframe(table_name="OperationScenario_Component_Building")

        scen_df = pd.merge(left=scenario_table, right=building_table[["ID_Building", "Af"]], on="ID_Building")

        building_numbers = scen_df.loc[:, ["ID_Scenario", "number_of_buildings", "ID_EnergyPrice", "Af"]].copy()

        ref = db.read_dataframe(table_name="OperationResult_RefYear")
        ref = ref[ref["ID_Scenario"].isin(scen_df["ID_Scenario"].to_list())].copy()
        ref = pd.merge(left=ref, right=building_numbers, on="ID_Scenario")
        ref["Q_RoomHeating_per_m2"] = ref["Q_RoomHeating"] / ref["Af"] / 1_000
        ref = ref[ref["ID_EnergyPrice"] == 1]
        hwb = (ref['Q_RoomHeating_per_m2'] * ref['number_of_buildings']).sum() / ref['number_of_buildings'].sum()
        
        # if the hwb is lower than the target hwb (diff is negative), increase the indoor set temperature
        diff = hwb - target_hwb
        percentage_diff = diff / target_hwb
        print(f"set temp: {indoor_set_temp}")
        print(f"Percentage diff: {percentage_diff} in {config.project_name}")
        if abs(diff)>=abs(diff_list[-1]):
            change_step = change_step / 2
            print(f"Change step decreased to {change_step}")

        diff_list.append(diff)
        if diff < 0:
            change = abs(change_step)
        else:
            change = -change_step



def summarize_indoor_set_temps(country_list):

    dfs = []
    for country in country_list:
        for year in [2020, 2030, 2040, 2050]:
            cfg = get_config(f"{country}_{year}")
            df = pd.read_csv(cfg.input /  "OperationScenario_Component_Behavior.csv", sep=";")
            indoor_temp = df["target_temperature_at_home_min"].values[0]
            df = pd.DataFrame(data={"country": [country], "indoor_temp": [indoor_temp]})
            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    
    sns.barplot(data=df, 
                x="country", 
                y="indoor_temp",
                hue="year"
                )
    plt.ylabel("Indoor set temperature corrected")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/testing/Country_level_prosumaging/figures") / "Corrected_indoor_set_temps.png")

    # copy Operation_component file into other input folders:
    # for country in country_list:
    #     cfg = get_config(f"{country}_{2020}")
    #     for year in [2030, 2040, 2050]:
    #         cfg_year = get_config(f"{country}_{year}")
    #         shutil.copy(cfg.input / "OperationScenario_Component_Behavior.csv", cfg_year.input / "OperationScenario_Component_Behavior.csv")






if __name__ == "__main__":
    country_list = [
            "AUT",  
            "BEL", 
            "POL",
            # # "CYP", 
            "PRT",
            "DNK", 
            "FRA", 
            "CZE",  
            "DEU", 
            "HRV",
            "HUN", 
            "ITA",  
            "LUX",
            "NLD",
            "SVK",
            "SVN",
            'IRL',
            'ESP',  
            'SWE',
            'GRC',
            'LVA',  
            'LTU',
            # 'MLT',
            'ROU',
            'BGR',  
            'FIN',
            'EST',
        ]

    years = [2020, 2030,  2040, 2050]#2030, 2040]
    for country in country_list:
        for year in years:
            cfg = get_config(f"{country}_{year}")
            run_only_ref_model_and_change_indoor_set_temp_until_correct(cfg)

            # run_flex_operation_model(cfg, task_number=20)

    summarize_indoor_set_temps(country_list)
    # cfg = get_config("AUT_2030")
    # run_flex_operation_model(cfg, task_number=1)

          
