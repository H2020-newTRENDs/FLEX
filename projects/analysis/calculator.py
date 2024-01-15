import os.path

import numpy as np
import pandas as pd
from utils.parquet import read_parquet
from utils.config import Config
from utils.db import create_db_conn
from utils.tables import InputTables, OutputTables

PROJECT_SUMMARY_YEAR = "SummaryYear"
PROJECT_SUMMARY_HOUR = "SummaryHour"


def merge_scenario(config: "Config"):
    conn = create_db_conn(config)
    scenarios = conn.read_dataframe(InputTables.OperationScenario.name)
    ref = conn.read_dataframe(OutputTables.OperationResult_RefYear.name).set_index("ID_Scenario")
    opt = conn.read_dataframe(OutputTables.OperationResult_OptYear.name).set_index("ID_Scenario")
    buildings = conn.read_dataframe(InputTables.OperationScenario_Component_Building.name).set_index("ID_Building")
    l = []
    id_merged_scenario = 1
    for index, row in scenarios.iterrows():
        for result_index, result_df in enumerate([ref, opt]):
            d = {}
            d["Country"] = config.project_name.split("_")[0]
            d["Year"] = int(config.project_name.split("_")[1])
            d["ID_MergedScenario"] = id_merged_scenario
            d["ID_Scenario"] = row["ID_Scenario"]
            d["ID_SEMS"] = result_index + 1  # 1: no prosumaging; 2: with prosumaging
            d["ID_Teleworking"] = buildings.at[row["ID_Building"], "id_demand_profile_type"]  # 1: no teleworking; 2: with teleworking
            d["ID_PV"] = row["ID_PV"]
            d["ID_Battery"] = row["ID_Battery"]
            d["ID_Boiler"] = row["ID_Boiler"]
            d["unit"] = "kWh"
            d["Useful_Appliance"] = result_df.at[row["ID_Scenario"], "BaseLoadProfile"] / 1000
            d["Useful_HotWater"] = result_df.at[row["ID_Scenario"], "HotWaterProfile"] / 1000
            d["Useful_SpaceHeating"] = result_df.at[row["ID_Scenario"], "Q_RoomHeating"] / 1000
            d["Final_PVGeneration"] = result_df.at[row["ID_Scenario"], "PhotovoltaicProfile"] / 1000
            d["Final_PVFeed"] = result_df.at[row["ID_Scenario"], "PV2Grid"] / 1000
            d["Final_Electricity"] = result_df.at[row["ID_Scenario"], "Grid"] / 1000
            d["Final_Fuel"] = result_df.at[row["ID_Scenario"], "Fuel"] / 1000
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
    pd.DataFrame(l).to_excel(os.path.join(config.output, f"{PROJECT_SUMMARY_YEAR}.xlsx"), index=False)


def calc_electricity_profiles(config: "Config"):
    conn = create_db_conn(config)
    scenarios = conn.read_dataframe(InputTables.OperationScenario.name).set_index("ID_Scenario")
    electricity_price = conn.read_dataframe(InputTables.OperationScenario_EnergyPrice.name)["electricity_1"]
    summary = pd.read_excel(os.path.join(config.output, f"{PROJECT_SUMMARY_YEAR}.xlsx"))
    l = []
    for id_sems in [1, 2]:
        for id_teleworking in [1, 2]:
            for id_boiler in [1, 2]:
                for id_pv in [1, 2]:
                    for id_battery in [1, 2]:
                        df = summary.loc[
                            (summary["ID_SEMS"] == id_sems) &
                            (summary["ID_Teleworking"] == id_teleworking) &
                            (summary["ID_Boiler"] == id_boiler) &
                            (summary["ID_PV"] == id_pv) &
                            (summary["ID_Battery"] == id_battery)
                        ]
                        scenario_ids = df["ID_Scenario"].to_list()
                        electricity_profile = np.zeros(8760, )
                        bat_discharge = np.zeros(8760, )
                        for id_scenario in scenario_ids:
                            building_num = scenarios.at[id_scenario, "building_num"]
                            if id_sems == 1:
                                file_name = f'OperationResult_RefHour_S{id_scenario}'
                            else:
                                file_name = f'OperationResult_OptHour_S{id_scenario}'
                            electricity_profile += building_num * read_parquet(file_name, config.output)["Grid"].to_numpy()
                            bat_discharge += building_num * read_parquet(file_name, config.output)["BatDischarge"].to_numpy()
                        for hour in range(1, 8761):
                            l.append({
                                "Country": config.project_name.split("_")[0],
                                "Year": int(config.project_name.split("_")[1]),
                                "ID_SEMS": id_sems,
                                "ID_Teleworking": id_teleworking,
                                "ID_Boiler": id_boiler,
                                "ID_PV": id_pv,
                                "ID_Battery": id_battery,
                                "YearHour": hour,
                                "DayHour": hour % 24 if hour % 24 != 0 else 24,
                                "demand_unit": "GWh",
                                "ElectricityDemand": electricity_profile[hour - 1]/10**9,
                                "BatDischarge": bat_discharge[hour - 1]/10**9,
                                "price_unit": "Euro/kWh",
                                "ElectricityPrice": electricity_price[hour - 1] * 10
                            })
    pd.DataFrame(l).to_excel(os.path.join(config.output, f"{PROJECT_SUMMARY_HOUR}.xlsx"), index=False)


def get_building_hour_profile(config: "Config", id_scenario: int, model_name: str, var_name: str):
    file_name = f'OperationResult_{model_name}Hour_S{id_scenario}'
    return read_parquet(file_name, config.output)[var_name].to_numpy()


