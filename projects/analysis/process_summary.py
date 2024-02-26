import pandas as pd
import os
import glob
from tqdm import tqdm
import numpy as np


INPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "ECEMF_Summary",
    "input"
)

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "ECEMF_Summary",
    "output"
)

SUMMARY_YEAR = "SummaryYear.csv"
SUMMARY_YEAR_AGG = "SummaryYear_aggregated.xlsx"
SUMMARY_HOUR = "SummaryHour.csv"
SUMMARY_HOUR_AGG = "SummaryHour_aggregated.csv"
SUMMARY_HOUR_AGG2 = "SummaryHour_further_aggregated.csv"
INVERT_SUMMARY = "INVERT_Summary.xlsx"


def concat_summary():
    summary_year_l = []
    summary_hour_l = []
    for csv_file_path in glob.glob(f"{INPUT_FOLDER}/*.csv"):
        file_name, _ = os.path.splitext(os.path.basename(csv_file_path))
        if file_name.endswith("_SummaryHour"):
            summary_hour_l.append(pd.read_csv(csv_file_path))
        elif file_name.endswith("_SummaryYear"):
            summary_year_l.append(pd.read_csv(csv_file_path))
        else:
            pass
    pd.concat(summary_year_l, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_YEAR), index=False)
    # pd.concat(summary_hour_l, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR), index=False)


def process_summary_year():

    def get_template_d():
        template_d = {
            "Country": country,
            "Year": year,
            "ID_SEMS": id_sems,
            "ID_Teleworking": id_teleworking,
            "ID_PV": id_pv,
            "ID_Battery": id_battery,
            "ID_Boiler": id_boiler
        }
        return template_d

    vars = [
        "Total_Useful_Appliance",
        "Total_Useful_HotWater",
        "Total_Useful_SpaceHeating",
        "Total_Final_PVGeneration",
        "Total_Final_PV2Load",
        "Total_Final_PV2Bat",
        "Total_Final_PV2Grid",
        "Total_Final_Grid",
        "Total_Final_Load",
        "Total_Final_HP",
        "Total_HeatingElement",
        "Total_Final_Fuel",
        "Total_Final_HeatingSystem"
    ]
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_YEAR))
    countries = list(df["Country"].unique())
    years = list(df["Year"].unique())
    l = []
    for country in tqdm(countries):
        for year in years:
            for id_sems in [1, 2]:
                for id_teleworking in [1, 2]:
                    for id_pv in [1, 2]:
                        for id_battery in [1, 2]:
                            for id_boiler in [1, 2]:
                                df_filtered = df.loc[
                                    (df["Country"] == country) &
                                    (df["Year"] == year) &
                                    (df["ID_SEMS"] == id_sems) &
                                    (df["ID_Teleworking"] == id_teleworking) &
                                    (df["ID_PV"] == id_pv) &
                                    (df["ID_Battery"] == id_battery) &
                                    (df["ID_Boiler"] == id_boiler)
                                ]
                                for var in vars:
                                    d = get_template_d()
                                    d["unit"] = "GWh"
                                    d["variable"] = var
                                    d["value"] = df_filtered[var].sum()/10**6
                                    l.append(d)

    pd.DataFrame(l).to_excel(os.path.join(OUTPUT_FOLDER, SUMMARY_YEAR_AGG), index=False)


def process_summary_hour():

    indices_aggregated = [
        "Country",
        "Year",
        "ID_SEMS",
        "ID_Teleworking",
        "ID_PV",
        "ID_Battery",
        "ID_Boiler",
        "DayHour",
        "demand_unit",
        "price_unit"
    ]
    indices_further_aggregated = [
        "Country",
        "Year",
        "ID_SEMS",
        "ID_Teleworking",
        "DayHour",
        "demand_unit",
        "price_unit"
    ]
    vars = [
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
        "ElectricityPrice"
    ]
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR))
    countries = list(df["Country"].unique())
    years = list(df["Year"].unique())
    l_aggregated = []
    l_further_aggregated = []
    for country in tqdm(countries):
        for year in years:
            for id_sems in [1, 2]:
                for id_teleworking in [1, 2]:
                    df_filtered_further = df.loc[
                        (df["Country"] == country) &
                        (df["Year"] == year) &
                        (df["ID_SEMS"] == id_sems) &
                        (df["ID_Teleworking"] == id_teleworking)
                    ]
                    l_further_aggregated.append(df_filtered_further.groupby(indices_further_aggregated, as_index=False)[vars].mean())
                    for id_pv in [1, 2]:
                        for id_battery in [1, 2]:
                            for id_boiler in [1, 2]:
                                df_filtered = df.loc[
                                    (df["Country"] == country) &
                                    (df["Year"] == year) &
                                    (df["ID_SEMS"] == id_sems) &
                                    (df["ID_Teleworking"] == id_teleworking) &
                                    (df["ID_PV"] == id_pv) &
                                    (df["ID_Battery"] == id_battery) &
                                    (df["ID_Boiler"] == id_boiler)
                                ]
                                l_aggregated.append(df_filtered.groupby(indices_aggregated, as_index=False)[vars].mean())
    pd.concat(l_aggregated, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR_AGG), index=False)
    pd.concat(l_further_aggregated, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_HOUR_AGG2), index=False)


def get_invert_summary():
    df = pd.read_excel(os.path.join(OUTPUT_FOLDER, INVERT_SUMMARY))
    countries = df["country"].unique().tolist()
    years = df["year"].unique().tolist()
    invert_summary = {}
    for country in countries:
        for year in years:
            df_filter = df.loc[(df["country"] == country) & (df["year"] == year)]
            invert_summary[(country, year)] = {}
            for _, row in df_filter.iterrows():
                invert_summary[(country, year)][row["energy_carrier"]] = row["value"]
    return invert_summary


def compare_hwb_diff():
    l = []
    for csv_file_path in glob.glob(f"{INPUT_FOLDER}/*.csv"):
        file_name, _ = os.path.splitext(os.path.basename(csv_file_path))
        country, year = file_name.split("_")[0:2]
        if file_name.endswith("_SummaryYear"):
            df = pd.read_csv(csv_file_path)
            hwb_diff = np.sum(df["hwb_diff"].to_numpy() * df["BuildingNumber"].to_numpy()) / df["BuildingNumber"].sum()
            l.append({
                "Country": country,
                "Year": year,
                "hwb_diff": hwb_diff
            })
    pd.DataFrame(l).to_excel(os.path.join(OUTPUT_FOLDER, "hwb_diff.xlsx"), index=False)
