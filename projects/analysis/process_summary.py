import pandas as pd
import os
import glob


INPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "output_summary",
    "input"
)

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "output_summary"
)


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
    pd.concat(summary_year_l, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, "SummaryYear.csv"), index=False)
    pd.concat(summary_hour_l, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, "SummaryHour.csv"), index=False)


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
        "Total_Final_PVFeed",
        "Total_Final_Electricity",
        "Total_Final_Fuel",
        "Total_Final_HeatingSystem"
    ]
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, "SummaryYear.csv"))
    countries = list(df["Country"].unique())
    years = list(df["Year"].unique())
    l = []
    for country in countries:
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
                                # append total heat pump
                                d = get_template_d()
                                d["unit"] = "GWh"
                                d["variable"] = "Total_Final_HP"
                                d["value"] = (df_filtered["Total_Final_Electricity"].sum() - df_filtered["Total_Useful_Appliance"].sum())/10**6
                                l.append(d)

    pd.DataFrame(l).to_excel(os.path.join(OUTPUT_FOLDER, "SummaryYear_aggregated.xlsx"), index=False)


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
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, "SummaryHour.csv"))
    countries = list(df["Country"].unique())
    years = list(df["Year"].unique())
    l_aggregated = []
    l_further_aggregated = []
    for country in countries:
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
    pd.concat(l_aggregated, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, "SummaryHour_aggregated.csv"), index=False)
    pd.concat(l_further_aggregated, axis=0, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, "SummaryHour_further_aggregated.csv"), index=False)

