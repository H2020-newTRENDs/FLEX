import os.path
import glob
import numpy as np
import pandas as pd
from utils.parquet import read_parquet
from utils.config import Config
from utils.db import create_db_conn
from utils.tables import InputTables, OutputTables
from tqdm import tqdm

PROJECT_SUMMARY_YEAR = "SummaryYear"
PROJECT_SUMMARY_HOUR = "SummaryHour"


def gen_summary_year(config: "Config"):

    def get_scenario_result(id_scenario, id_sems, id_boiler, id_battery, var_name: str):
        if id_sems == 1:
            value = ref.at[id_scenario, var_name]
        else:
            if id_boiler == 1 and id_battery == 1:
                value = ref.at[id_scenario, var_name]
            else:
                value = opt.at[id_scenario, var_name]
        return value

    conn = create_db_conn(config)
    scenarios = conn.read_dataframe(InputTables.OperationScenario.name)
    ref = conn.read_dataframe(OutputTables.OperationResult_RefYear.name).set_index("ID_Scenario")
    opt = conn.read_dataframe(OutputTables.OperationResult_OptYear.name).set_index("ID_Scenario")
    buildings = conn.read_dataframe(InputTables.OperationScenario_Component_Building.name).set_index("ID_Building")
    l = []
    id_merged_scenario = 1
    for index, row in scenarios.iterrows():
        for index_sems in [1, 2]:
            d = {}
            d["Country"] = config.project_name.split("_")[0]
            d["Year"] = int(config.project_name.split("_")[1])
            d["ID_MergedScenario"] = id_merged_scenario
            d["ID_Scenario"] = row["ID_Scenario"]
            d["ID_SEMS"] = index_sems  # 1: no prosumaging; 2: with prosumaging
            d["ID_Teleworking"] = buildings.at[row["ID_Building"], "id_demand_profile_type"]  # 1: no teleworking; 2: with teleworking
            d["ID_PV"] = row["ID_PV"]
            d["ID_Battery"] = row["ID_Battery"]
            d["ID_Boiler"] = row["ID_Boiler"]
            d["unit"] = "kWh"
            d["Useful_Appliance"] = get_scenario_result(row["ID_Scenario"], d["ID_SEMS"], row["ID_Boiler"], row["ID_Battery"], "BaseLoadProfile") / 1000
            d["Useful_HotWater"] = get_scenario_result(row["ID_Scenario"], d["ID_SEMS"], row["ID_Boiler"], row["ID_Battery"], "HotWaterProfile") / 1000
            d["Useful_SpaceHeating"] = get_scenario_result(row["ID_Scenario"], d["ID_SEMS"], row["ID_Boiler"], row["ID_Battery"], "Q_RoomHeating") / 1000
            d["Final_PVGeneration"] = get_scenario_result(row["ID_Scenario"], d["ID_SEMS"], row["ID_Boiler"], row["ID_Battery"], "PhotovoltaicProfile") / 1000
            d["Final_PVFeed"] = get_scenario_result(row["ID_Scenario"], d["ID_SEMS"], row["ID_Boiler"], row["ID_Battery"], "PV2Grid") / 1000
            d["Final_Electricity"] = get_scenario_result(row["ID_Scenario"], d["ID_SEMS"], row["ID_Boiler"], row["ID_Battery"], "Grid") / 1000
            d["Final_Fuel"] = get_scenario_result(row["ID_Scenario"], d["ID_SEMS"], row["ID_Boiler"], row["ID_Battery"], "Fuel") / 1000
            d["BuildingNumber"] = row["building_num"]
            d["Total_Useful_Appliance"] = d["Useful_Appliance"] * row["building_num"]
            d["Total_Useful_HotWater"] = d["Useful_HotWater"] * row["building_num"]
            d["Total_Useful_SpaceHeating"] = d["Useful_SpaceHeating"] * row["building_num"]
            d["Total_Final_PVGeneration"] = d["Final_PVGeneration"] * row["building_num"]
            d["Total_Final_PVFeed"] = d["Final_PVFeed"] * row["building_num"]
            d["Total_Final_Electricity"] = d["Final_Electricity"] * row["building_num"]
            d["Total_Final_Fuel"] = d["Final_Fuel"] * row["building_num"]
            d["Total_Final_HeatingSystem"] = d["Total_Final_Fuel"] + d["Total_Final_Electricity"] - d["Total_Useful_Appliance"]
            l.append(d)
            id_merged_scenario += 1
    pd.DataFrame(l).to_csv(os.path.join(config.output, f"{config.project_name}_{PROJECT_SUMMARY_YEAR}.csv"), index=False)


def gen_summary_hour(config: "Config"):
    vars_to_collect = [
        "Q_RoomHeating",
        "E_Heating_HP_out",
        "E_DHW_HP_out",
        "Q_HeatingElement_heat",
        "Q_HeatingElement_DHW",
        "Q_RoomCooling",
        "E_RoomCooling",
        "PhotovoltaicProfile",
        "PV2Load",
        "PV2Bat",
        "PV2Grid",
        "BatCharge",
        "BatDischarge",
        "BaseLoadProfile",
        "Load",
        "Grid2Load",
        "Grid2Bat",
        "Grid",
        "Fuel",
    ]
    conn = create_db_conn(config)
    scenarios = conn.read_dataframe(InputTables.OperationScenario.name).set_index("ID_Scenario")
    electricity_price = conn.read_dataframe(InputTables.OperationScenario_EnergyPrice.name)["electricity_1"]
    summary = pd.read_csv(os.path.join(config.output, f"{config.project_name}_{PROJECT_SUMMARY_YEAR}.csv"))
    l = []
    for id_sems in [1, 2]:
        for id_teleworking in [1, 2]:
            for id_boiler in [1, 2]:
                for id_pv in [1, 2]:
                    for id_battery in tqdm([1, 2]):
                        df = summary.loc[
                            (summary["ID_SEMS"] == id_sems) &
                            (summary["ID_Teleworking"] == id_teleworking) &
                            (summary["ID_Boiler"] == id_boiler) &
                            (summary["ID_PV"] == id_pv) &
                            (summary["ID_Battery"] == id_battery)
                        ]
                        scenario_ids = df["ID_Scenario"].to_list()

                        var_profiles = {}
                        for var_name in vars_to_collect:
                            var_profiles[var_name] = np.zeros(8760, )

                        for id_scenario in scenario_ids:
                            id_scenario = int(id_scenario)
                            building_num = scenarios.at[id_scenario, "building_num"]
                            if id_sems == 1:
                                file_name = f'OperationResult_RefHour_S{id_scenario}'
                            else:
                                if id_boiler == 2 or id_battery == 2:
                                    file_name = f'OperationResult_OptHour_S{id_scenario}'
                                else:
                                    file_name = f'OperationResult_RefHour_S{id_scenario}'

                            scenario_result = read_parquet(file_name, config.output)
                            for var_name in vars_to_collect:
                                var_profiles[var_name] += building_num * scenario_result[var_name].to_numpy()

                        for hour in range(1, 8761):
                            d = {
                                "Country": config.project_name.split("_")[0],
                                "Year": int(config.project_name.split("_")[1]),
                                "ID_SEMS": id_sems,
                                "ID_Teleworking": id_teleworking,
                                "ID_Boiler": id_boiler,
                                "ID_PV": id_pv,
                                "ID_Battery": id_battery,
                                "YearHour": hour,
                                "DayHour": hour % 24 if hour % 24 != 0 else 24,
                                "demand_unit": "GWh"
                            }
                            for var_name in vars_to_collect:
                                d[var_name] = var_profiles[var_name][hour - 1]/10**9
                            d["price_unit"] = "Euro/kWh"
                            d["ElectricityPrice"] = electricity_price[hour - 1] * 10
                            l.append(d)
    pd.DataFrame(l).to_csv(os.path.join(config.output, f"{config.project_name}_{PROJECT_SUMMARY_HOUR}.csv"), index=False)


def get_building_hour_profile(config: "Config", id_scenario: int, model_name: str, var_name: str):
    file_name = f'OperationResult_{model_name}Hour_S{id_scenario}'
    return read_parquet(file_name, config.output)[var_name].to_numpy()


