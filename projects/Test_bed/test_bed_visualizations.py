import os
import sys
from pathlib import Path
import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from utils.db import create_db_conn
from utils.config import Config
from utils.tables import OutputTables, InputTables
from matplotlib.ticker import PercentFormatter

matplotlib.rc("font", **{"size": 22})

BUILDING_ID_2_NAME = {
    4: "EZFH 1 B" ,
    5: "EZFH 1 S" ,
    1: "EZFH 5 B" ,
    2: "EZFH 5 S" ,
    3: "EZFH 9 B" ,
    8: "MFH 1 B",
    9: "MFH 1 S",
    6: "MFH 5 B",
    7: "MFH 5 S",
}

def return_config():
    return Config(project_name="Test_bed", project_path=r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects\Test_bed")

def return_db():
    config = return_config()
    db = create_db_conn(config)
    return db

def load_yearly_data():
    db = return_db()
    ref_yearly = db.read_dataframe(table_name=OutputTables.OperationResult_RefYear.name,)
    opt_yearly = db.read_dataframe(table_name=OutputTables.OperationResult_OptYear.name)
    scen_table = db.read_dataframe(table_name=InputTables.OperationScenario.name) 

    opt_rolling = opt_yearly.query(f"ID_Scenario in {[x for x in range(1, 10)]}").copy()
    opt_rolling.loc[:, "method"] = "rolling horizon 1 day"

    opt_rolling_3_day = opt_yearly.query(f"ID_Scenario in {[x for x in range(19, 28)]}").copy()
    opt_rolling_3_day.loc[:, "method"] = "rolling horizon 3 days"

    opt_foresight = opt_yearly.query(f"ID_Scenario in {[x for x in range(10, 19)]}").copy()
    opt_foresight.loc[:, "method"] = "perfect foresight"
    
    ref_yearly = ref_yearly.query(f"ID_Scenario in {[x for x in range(1, 10)]}").copy()  # dann wiederholen sie sich
    ref_yearly.loc[:, "method"] = "simulation"

    total_df = pd.concat([opt_rolling, opt_foresight, ref_yearly, opt_rolling_3_day])
    merged = total_df.merge(scen_table.loc[:, ["ID_Scenario", "ID_Building"]], on="ID_Scenario", how="left")
    merged.loc[:, "TotalCost"] = merged["TotalCost"] / 100  # €
    return merged

def load_hourly_data():
    db = return_db()
    scen_table = db.read_dataframe(table_name=InputTables.OperationScenario.name) 
    ref_dfs = []
    opt_dfs = []
    for i in range(1, 28):
        try:
            df = db.read_parquet(table_name=OutputTables.OperationResult_OptHour.name, scenario_ID=i, folder=return_config().output)
        except:
            df = pd.DataFrame()
        if not df.empty:
            opt_dfs.append(df)

        if i <= 9:
            ref_dfs.append(db.read_parquet(table_name=OutputTables.OperationResult_RefHour.name, scenario_ID=i, folder=return_config().output))

    opt_df = pd.concat(opt_dfs)
    opt_rolling = opt_df.query(f"ID_Scenario in {[x for x in range(1, 10)]}").copy()
    opt_rolling.loc[:, "method"] = "rolling horizon 1 day"

    opt_rolling_3_day = opt_df.query(f"ID_Scenario in {[x for x in range(19, 28)]}").copy()
    opt_rolling_3_day.loc[:, "method"] = "rolling horizon 3 days"

    opt_foresight = opt_df.query(f"ID_Scenario in {[x for x in range(10, 19)]}").copy()
    opt_foresight.loc[:, "method"] = "perfect foresight"

    ref_df = pd.concat(ref_dfs)
    ref_df.loc[:, "method"] = "simulation"

    total_df = pd.concat([opt_rolling, opt_foresight, ref_df, opt_rolling_3_day])
    merged = total_df.merge(scen_table.loc[:, ["ID_Scenario", "ID_Building"]], on="ID_Scenario", how="left")
    return merged

def plot_rolling_and_perfect_foresight_cost(scenario_name: str):
    if scenario_name == "thermal_mass":
        df = load_hourly_runs_thermal_mass()
    else:
        df = load_yearly_runs()

    # perfect = df.loc[df["method"]=="perfect foresight", :].copy()
    # perfect["xi_dhw_tank"] = (perfect.loc[:, "Q_DHWTank_in"] - perfect.loc[:, "Q_DHWTank_out"]) / perfect.loc[:, "Q_DHWTank_in"]
    # perfect["ID_Building"] = perfect["ID_Building"].map(BUILDING_ID_2_NAME)
    # perfect.loc[:, ["ID_Building", "xi_dhw_tank"]].to_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") /"DHW_tank_loss_values.csv", index=False, sep=";")

    fig, ax = plt.subplots(figsize=(28, 14))
    sns.barplot(
        data=df.query("year==2022"),
        x="ID_Building",
        y="TotalCost",
        hue="method"
        
    )
    ax.set_xlabel("")
    ax.set_ylabel("yearly electricity cost (€)")
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Cost_comparison_rolling_horizon_vs_perfect_foresight_{scenario_name}_2022.png")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Cost_comparison_rolling_horizon_vs_perfect_foresight_{scenario_name}_2022.svg")
    plt.show()

def load_relative_cost_decrease_without_monetization_of_thermal_mass(day_3: bool = False):
    if day_3:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "relative_cost_decrease_rolling_horizon_no_monetization_3_days.csv", sep=";")
        df["method"] = "rolling horizon without monetization 3 days"
    else:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "relative_cost_decrease_rolling_horizon_no_monetization.csv", sep=";")
        df["method"] = "rolling horizon without monetization 1 day"
    return df


def load_relative_cost_decrease_without_monetization_of_heating_tank():
    df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "relative_cost_decrease_rolling_horizon_no_monetization_heating_tank.csv", sep=";")
    df["method"] = "rolling horizon without monetization"
    return df

def load_relative_cost_decrease_without_monetization_of_battery():
    df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "relative_cost_decrease_rolling_horizon_no_monetization_battery.csv", sep=";")
    df["method"] = "rolling horizon without monetization"
    return df

def load_relative_cost_decrease_without_monetization_of_dhw_tank():
    df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "relative_cost_decrease_rolling_horizon_no_monetization_dhw_tank.csv", sep=";")
    df["method"] = "rolling horizon without monetization"
    return df

def load_relative_cost_decrease_without_monetization_all(day_3: bool = False):
    if day_3:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "relative_cost_decrease_rolling_horizon_no_monetization_all_3_days.csv", sep=";")
        df["method"] = "rolling horizon without monetization 3 days"
    else:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "relative_cost_decrease_rolling_horizon_no_monetization_all.csv", sep=";")
        df["method"] = "rolling horizon without monetization 1 day"
    return df

def plot_realative_cost_decrease(scenario_name: str = None):
      
    if scenario_name=="thermal_mass":
        df = load_yearly_runs_thermal_mass()
        df["method"] = df["method"].str.replace("monetization", "terminal cost")
        df['method_year'] = df['method'] + ' (' + df['year'].astype(str) + ')'
        year = 2019
    elif scenario_name == "all":
        df = load_yearly_runs()
        year = 2022
 
    df.sort_values(["ID_Building", "method"], inplace=True)

    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(
        data=df.query("year==2019"),
        x="ID_Building",
        y="relative cost decrease (%)",
        hue="method",
        dodge=True
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    ax.set_xlabel("")
    ax.set_ylabel("operation cost savings (%)")
    legend = ax.legend()
    legend.set_title(None)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Relative_cost_comparison_rolling_horizon_vs_perfect_foresight_{scenario_name}_2019.png")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Relative_cost_comparison_rolling_horizon_vs_perfect_foresight_{scenario_name}_2019.svg")
    plt.show()




def compare_load_profiles():
    all_profiles = load_hourly_runs_thermal_mass()
    fig = px.line(
        data_frame=all_profiles,
        x="Hour",
        y="Grid",
        color="ID_Building",
        symbol="method",
        line_dash="method"
    )
    fig.show()


def load_shifted_energy_without_monetization_of_thermal_mas(day_3: bool = False):
    if day_3:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_3_days.csv", sep=";")
        df["method"] = "rolling horizon without monetization 3 days"
    else:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization.csv", sep=";")
        df["method"] = "rolling horizon without monetization 1 day"
    return df

def load_shifted_energy_without_monetization_heating_tank():
    df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_heating_tank.csv", sep=";")
    df["method"] = "rolling horizon without monetization"
    return df

def load_shifted_energy_without_monetization_battery():
    df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_battery.csv", sep=";")
    df["method"] = "rolling horizon without monetization"
    return df

def load_shifted_energy_without_monetization_dhw_tank():
    df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_dhw_tank.csv", sep=";")
    df["method"] = "rolling horizon without monetization"
    return df

def load_shifted_energy_without_monetization_all(day_3: bool = False):
    if day_3:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_all_3_days.csv", sep=";")
        df["method"] = "rolling horizon without monetization 3 days"
    else:
        df = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_all.csv", sep=";")
        df["method"] = "rolling horizon without monetization 1 day"
    return df

def show_price_quantile_where_building_is_charged(scenario_name):
    all_profiles = load_hourly_data().sort_values(by=["ID_Building", "Hour"])
    # rolling = all_profiles.query("method == 'rolling horizon'").reset_index(drop=True) 
    simulation = all_profiles.query("method == 'simulation'").reset_index(drop=True) 
    foresight = all_profiles.query("method == 'perfect foresight'").reset_index(drop=True) 

    def create_df_price_quatile(simulation_df, perfect_foresight_df):
        quantiles_weighted = {}
        quantiles_unweighted = {}
        for scen_id, group in simulation_df.groupby("ID_Scenario"):
            foresight_df = perfect_foresight_df.loc[perfect_foresight_df.loc[:, "ID_Scenario"]==scen_id+9, :]
            pre_heat_indices = np.where((foresight_df.Grid - group.Grid)>0)[0]
            preheating_energy = (foresight_df.Grid - group.Grid)
            preheating_energy_cut = preheating_energy[preheating_energy>0]
            electricity_pre_heat_price = (group.ElectricityPrice.iloc[pre_heat_indices] * preheating_energy_cut).sum() / preheating_energy_cut.sum()
            electricity_pre_heat_price_unweighted = group.ElectricityPrice.iloc[pre_heat_indices].mean()
            sorted_price = sorted(group.ElectricityPrice)

            # Step 2: Calculate the quantile
            # Use numpy to find the quantile
            # This function finds the relative position of the value in the sorted data
            def find_quantile(value, sorted_data):
                pos = np.searchsorted(sorted_data, value, side='right')
                quantile = pos / len(sorted_data)
                return quantile

            # Calculate the quantile for the given value
            quantile = find_quantile(electricity_pre_heat_price_unweighted, sorted_price)
            quantile_weighted = find_quantile(electricity_pre_heat_price, sorted_price)

            quantiles_weighted[scen_id] = quantile_weighted
            quantiles_unweighted[scen_id] = quantile
        return quantiles_unweighted, quantiles_weighted

    unweighted, weighted = create_df_price_quatile(simulation, foresight)
    df = pd.DataFrame([unweighted, weighted], index=["average", "weighted average"]).T.reset_index().rename(columns={"index": "ID_Building"}).melt(id_vars="ID_Building", value_name="price quantile for charging", var_name="method")
    df["ID_Building"] = df["ID_Building"].map(BUILDING_ID_2_NAME)
    df.sort_values(by="ID_Building", inplace=True)

    matplotlib.rc("font", **{"size": 22})
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(
        data=df,
        x="ID_Building",
        y="price quantile for charging",
        hue="method",
    )
    plt.ylabel(r"price quantile ($\alpha$) " + "at which a building \n is pre-heated and storages charged")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Price_quantile_for_charging_{scenario_name}.png")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Price_quantile_for_charging_{scenario_name}.svg")
    plt.show()

def amount_of_energy_shifted_over_outside_temperature(scenario_name: str=None):
    # scenario name is used for the figure name
    # this is only supposed to provide valuable insights of the scenarios dont contain any storages. We focus on the building mass:
    if scenario_name == "thermal_mass":
        all_profiles = load_hourly_runs_thermal_mass()
    else:

        all_profiles = load_hourly_runs()


    all = all_profiles.query("year == 2019").copy()
    
    def create_df_shifted(simulation_df, other_df):
        shifted_energy = (simulation_df.Grid - other_df.Grid) / 1_000
        shifted_energy[shifted_energy<0] = 0
        df_shifted = pd.concat([shifted_energy, simulation_df.loc[:, ["T_outside", "ID_Building", "year"]]], axis=1).rename(columns={"Grid": "shifted energy"})
        df_shifted['outside temperature'] = df_shifted['T_outside'].round().astype(int)
        df = df_shifted.groupby(["ID_Building", "outside temperature", "year"]).agg({"shifted energy": "sum"})
        df.reset_index(inplace=True)
        return df
    

    rolling_1_day_with_t = all.query("method == 'rolling horizon 1 day forecast with terminal cost'").reset_index(drop=True)
    rolling_1_day_no_t = all.query("method == 'rolling horizon 1 day forecast without terminal cost'").reset_index(drop=True)
    rolling_3_day_with_t =  all.query("method == 'rolling horizon 3 day forecast with terminal cost'").reset_index(drop=True)
    rolling_3_day_no_t =  all.query("method == 'rolling horizon 3 day forecast without terminal cost'").reset_index(drop=True)
    simulation = all.query("method == 'simulation'").reset_index(drop=True) 
    foresight = all.query("method == 'perfect forecast'").reset_index(drop=True) 
    # rolling_3_day_perfect_price = all.query("method == 'rolling horizon 3 day perfect price forecast without terminal cost'").reset_index(drop=True)



    foresight_shifted = create_df_shifted(simulation_df=simulation, other_df=foresight)
    foresight_shifted["method"] = "perfect foresight"
    rolling_shifted_1_day_with_t = create_df_shifted(simulation_df=simulation, other_df=rolling_1_day_with_t)
    rolling_shifted_1_day_with_t["method"] = "rolling horizon 1 day with terminal cost"
    rolling_shifted_1_day_no_t = create_df_shifted(simulation_df=simulation, other_df=rolling_1_day_no_t)
    rolling_shifted_1_day_no_t["method"] = "rolling horizon 1 day without terminal cost"
    rolling_shifted_3_day_with_t = create_df_shifted(simulation_df=simulation, other_df=rolling_3_day_with_t)
    rolling_shifted_3_day_with_t["method"] = "rolling horizon 3 days with terminal cost"
    rolling_shifted_3_day_no_t = create_df_shifted(simulation_df=simulation, other_df=rolling_3_day_no_t)
    rolling_shifted_3_day_no_t["method"] = "rolling horizon 3 days without terminal cost"
    # rolling_shifted_3_day_perfect_price = create_df_shifted(simulation_df=simulation, other_df=rolling_3_day_perfect_price)
    # rolling_shifted_3_day_perfect_price["method"] = "rolling horizon 3 day perfect price forecast without terminal cost"

    # rolling_shifted.to_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_all.csv", sep=";", index=False)
    # rolling_shifted_3_days.to_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Shifted_energy_rolling_horizon_no_monetization_3_days.csv", sep=";", index=False)

    df = pd.concat([foresight_shifted, rolling_shifted_1_day_with_t, rolling_shifted_1_day_no_t, rolling_shifted_3_day_with_t, rolling_shifted_3_day_no_t,])# rolling_shifted_3_day_perfect_price])
    
    matplotlib.rc("font", **{"size": 22})
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.boxplot(
        data=df,
        x="ID_Building",
        y="shifted energy",
        hue="method",
    )
    plt.ylabel("shifted energy per hour in kWh")
    plt.xlabel("")
    plt.xticks(rotation=45)
    legend = plt.legend()
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Shifted_energy_per_hour_FLEX_{scenario_name}_2019.png")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Shifted_energy_per_hour_FLEX_{scenario_name}_2019.svg")
    plt.show()


    # plot the total amount of shifted energy per building:
    t_df = df.groupby(["ID_Building", "method", "year"]).sum().reset_index()
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(
        data=t_df,
        x="ID_Building",
        y="shifted energy",
        hue="method",
        hue_order=["perfect foresight", "rolling horizon 1 day with terminal cost", "rolling horizon 1 day without terminal cost", 
                   "rolling horizon 3 days with terminal cost", "rolling horizon 3 days without terminal cost", 
                #    "rolling horizon 3 day perfect price forecast without terminal cost"
                   ]
    )
    plt.ylabel("shifted electric energy over 1 year in kWh")
    plt.xlabel("")
    legend = plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Shifted_energy_per_year_FLEX_{scenario_name}_2019.png")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / f"Shifted_energy_per_year_FLEX_{scenario_name}_2019.svg")
    plt.show()
    # for building_id in df["ID_Building"].unique():
    #     fig, ax = plt.subplots(figsize=(28, 14))
    #     sns.lineplot(
    #         data=df.loc[df["ID_Building"]==building_id, :],
    #         x="outside temperature",
    #         y="shifted energy",
    #         hue="method",
            
    #     )
    #     plt.ylabel("shifted energy in kWh")
    #     plt.xlabel("outside temperature in °C")
    #     plt.xticks(rotation=90)
    #     plt.title(f"Building {building_id}")
    #     plt.show()

def dynamic_thermal_mass_loss_factor():
    all_profiles = load_hourly_data().sort_values(by=["ID_Building", "Hour"])
    rolling = all_profiles.query("method == 'rolling horizon'").reset_index(drop=True) 
    simulation = all_profiles.query("method == 'simulation'").reset_index(drop=True) 
    foresight = all_profiles.query("method == 'perfect foresight'").reset_index(drop=True) 

    Q_preheat = foresight.loc[:, "Q_RoomHeating"].values - simulation.loc[:, "Q_RoomHeating"].values
    Q_preheat[Q_preheat<0] = 0
    Q_shifted = simulation.loc[:, "Q_RoomHeating"].values - foresight.loc[:, "Q_RoomHeating"].values
    Q_shifted[Q_shifted<0] = 0

    df = pd.concat([pd.DataFrame(data=[Q_preheat, Q_shifted], index=["Q_preheat", "Q_shifted"]).T, simulation.loc[:, ["T_outside", "ID_Building"]]], axis=1)
    # .rename{columns= :"Q_preheat", :"Q_shifted"}
    dynamic_losses = df.groupby("ID_Building").agg({"Q_preheat":"sum", "Q_shifted": "sum"}).reset_index()
    dynamic_losses["loss"] = (dynamic_losses["Q_preheat"] - dynamic_losses["Q_shifted"]) / dynamic_losses["Q_preheat"]


def load_yearly_runs():
    df_yearly_3_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_3_day_forecast_no_terminal_cost_all.csv", sep=";") #
    df_yearly_3_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_3_day_forecast_with_terminal_cost_all.csv", sep=";") #
    
    df_yearly_3_day_perfect_forecast = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_3_day_perfect_price_forecast_no_terminal_cost_all.csv", sep=";") 

    df_yearly_perfect = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_perfect_forecast_no_terminal_cost_all.csv", sep=";") #
    df_yearly_1_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_1_day_forecast_no_terminal_cost_all.csv", sep=";") #
    df_yearly_1_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_1_day_forecast_with_terminal_cost_all.csv", sep=";") #

    df = pd.concat([df_yearly_perfect, df_yearly_1_day_no_t, df_yearly_1_day_with_t, df_yearly_3_day_no_t, df_yearly_3_day_with_t, df_yearly_3_day_perfect_forecast])
    return df

def load_yearly_runs_thermal_mass():
    df_yearly_3_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_3_day_forecast_no_terminal_cost_thermal_mass.csv", sep=";") #
    df_yearly_1_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_1_day_forecast_no_terminal_cost_thermal_mass.csv", sep=";") #
    df_yearly_1_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_1_day_forecast_with_terminal_cost_thermal_mass.csv", sep=";") #
    df_yearly_perfect_forecast = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_perfect_forecast_thermal_mass.csv", sep=";") #
    df_yearly_3_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_3_day_forecast_with_terminal_cost_thermal_mass.csv", sep=";") #

    df = pd.concat([df_yearly_perfect_forecast, df_yearly_1_day_with_t, df_yearly_1_day_no_t, df_yearly_3_day_no_t, df_yearly_3_day_with_t])
    return df


def load_hourly_runs():
    df_hourly_3_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_3_day_forecast_no_terminal_cost_all.csv", sep=";") #
    df_hourly_3_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_3_day_forecast_with_terminal_cost_all.csv", sep=";") #
    df_hourly_3_day_perfectforecast = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_3_day_perfect_price_forecast_no_terminal_cost_all.csv", sep=";") 
    df_hourly_perfect = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_perfect_forecast_no_terminal_cost_all.csv", sep=";") #
    df_hourly_1_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_1_day_forecast_no_terminal_cost_all.csv", sep=";") #
    df_hourly_1_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_1_day_forecast_with_terminal_cost_all.csv", sep=";") #
    return pd.concat([df_hourly_perfect, df_hourly_1_day_with_t, df_hourly_1_day_no_t, df_hourly_3_day_no_t, df_hourly_3_day_with_t, df_hourly_3_day_perfectforecast])

def load_hourly_runs_thermal_mass():
    df_hourly_3_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_3_day_forecast_no_terminal_cost_thermal_mass.csv", sep=";") #
    df_hourly_1_day_no_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_1_day_forecast_no_terminal_cost_thermal_mass.csv", sep=";") #
    df_hourly_1_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_1_day_forecast_with_terminal_cost_thermal_mass.csv", sep=";") #
    df_hourly_perfect_forecast = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_perfect_forecast_thermal_mass.csv", sep=";") #
    df_hourly_3_day_with_t = pd.read_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_3_day_forecast_with_terminal_cost_thermal_mass.csv", sep=";") #
    
    df = pd.concat([df_hourly_perfect_forecast, df_hourly_1_day_with_t, df_hourly_1_day_no_t, df_hourly_3_day_no_t, df_hourly_3_day_with_t])
    # df.reset_index(drop=True, inplace=True)
    # mask1 = df["ID_Scenario"] - 9
    # mask2 = df["ID_Scenario"] - 18
    # mask1 = mask1.map(BUILDING_ID_2_NAME).dropna()
    # mask2 = mask2.map(BUILDING_ID_2_NAME).dropna()
    # df.loc[mask1.index.to_list(), "ID_Building"] = mask1
    # df.loc[mask2.index.to_list(), "ID_Building"]  = mask2

    return df



def save_run():
    db = return_db()
    scen_table = db.read_dataframe(table_name=InputTables.OperationScenario.name) 
    ref_dfs = []
    opt_dfs = []
    for i in range(1, 28):
        try:
            df = db.read_parquet(table_name=OutputTables.OperationResult_OptHour.name, scenario_ID=i, folder=return_config().output)
        except:
            df = pd.DataFrame()
        if not df.empty:
            opt_dfs.append(df)
            ref_dfs.append(db.read_parquet(table_name=OutputTables.OperationResult_RefHour.name, scenario_ID=i, folder=return_config().output))
        
    
    desccription = "rolling horizon 3 day forecast with terminal cost"
    opt_df = pd.concat(opt_dfs)
    ref_df = pd.concat(ref_dfs)
    opt_df.loc[:, "method"] = desccription
    ref_df.loc[:, "method"] = "simulation"
    merged = pd.concat([opt_df, ref_df], axis=0)

    merged.loc[merged["ID_Scenario"].isin([x for x in range(1, 10)]), "year"] = "2019"
    merged.loc[merged["ID_Scenario"].isin([x for x in range(10, 19)]), "year"] = "2021"
    merged.loc[merged["ID_Scenario"].isin([x for x in range(19, 28)]), "year"] = "2022"
    
    merged = merged.merge(scen_table.loc[:, ["ID_Scenario", "ID_Building"]], on="ID_Scenario", how="left")
    merged["ID_Building"] = merged["ID_Building"].map(BUILDING_ID_2_NAME)

    merged.to_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "hourly_results_3_day_forecast_with_terminal_cost_thermal_mass.csv", sep=";", index=False)

    # yearly
    ref_yearly = db.read_dataframe(table_name=OutputTables.OperationResult_RefYear.name,)
    opt_yearly = db.read_dataframe(table_name=OutputTables.OperationResult_OptYear.name)
    opt_yearly.loc[:, "method"] = desccription
    opt_yearly.loc[:, "relative cost decrease (%)"] = 100 - (opt_yearly.loc[:, "TotalCost"].to_numpy() / ref_yearly.loc[:, "TotalCost"].to_numpy() * 100)

    opt_yearly.loc[opt_yearly.loc[:, "ID_Scenario"].isin([x for x in range(1, 10)]), "year"] = "2019"
    opt_yearly.loc[opt_yearly["ID_Scenario"].isin([x for x in range(10, 19)]), "year"] = "2021"
    opt_yearly.loc[opt_yearly["ID_Scenario"].isin([x for x in range(19, 28)]), "year"] = "2022"
    opt_yearly = opt_yearly.merge(scen_table.loc[:, ["ID_Scenario", "ID_Building"]], on="ID_Scenario", how="left")
    opt_yearly["ID_Building"] = opt_yearly["ID_Building"].map(BUILDING_ID_2_NAME)

    opt_yearly.to_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "yearly_results_3_day_forecast_with_terminal_cost_thermal_mass.csv", sep=";", index=False)
    




def year_comparison_of_3_day_forecast():
    db = return_db()
    ref_yearly = db.read_dataframe(table_name=OutputTables.OperationResult_RefYear.name,)
    opt_yearly = db.read_dataframe(table_name=OutputTables.OperationResult_OptYear.name)
    opt_yearly.loc[:, "method"] = "perfect forecast"
    ref_yearly.loc[:, "method"] = "simulation"

    merged_yearly = pd.concat([opt_yearly, ref_yearly], axis=1)

    merged.loc[merged["ID_Scenario"] in [x for x in range(1, 10)], "year"] = "2019"
    merged.loc[merged["ID_Scenario"] in [x for x in range(10, 19)], "year"] = "2021"
    merged.loc[merged["ID_Scenario"] in [x for x in range(19, 28)], "year"] = "2022"

    
    df_2019 = opt_yearly.query(f"ID_Scenario in {[x for x in range(19, 28)]}").copy()
    df_ref_2019 = ref_yearly.query(f"ID_Scenario in {[x for x in range(19, 28)]}").copy()
    df_2019_perfect = opt_yearly.query(f"ID_Scenario in {[x for x in range(10, 19)]}").copy()

    df_2021 = opt_yearly.query(f"ID_Scenario in {[x for x in range(28, 37)]}").copy()
    df_ref_2021 = ref_yearly.query(f"ID_Scenario in {[x for x in range(28, 37)]}").copy()
    df_2021_perfect = opt_yearly.query(f"ID_Scenario in {[x for x in range(46, 55)]}").copy()

    df_2022 = opt_yearly.query(f"ID_Scenario in {[x for x in range(37, 46)]}").copy()
    df_ref_2022 = ref_yearly.query(f"ID_Scenario in {[x for x in range(37, 46)]}").copy()
    df_2022_perfect = opt_yearly.query(f"ID_Scenario in {[x for x in range(55, 64)]}").copy()



    df_2019.loc[:, "year"] = "2019 3-day horizon"
    df_2021.loc[:, "year"] = "2021 3-day horizon"
    df_2022.loc[:, "year"] = "2022 3-day horizon"
    df_2019_perfect.loc[:, "year"] = "2019 perfect forecast"
    df_2021_perfect.loc[:, "year"] = "2021 perfect forecast"
    df_2022_perfect.loc[:, "year"] = "2022 perfect forecast"

    df_2019.loc[:, "relative cost decrease (%)"] = 100 - (df_2019.loc[:, "TotalCost"].to_numpy() / df_ref_2019.loc[:, "TotalCost"].to_numpy() * 100)
    df_2021.loc[:, "relative cost decrease (%)"] = 100 - (df_2021.loc[:, "TotalCost"].to_numpy() / df_ref_2021.loc[:, "TotalCost"].to_numpy() * 100)
    df_2022.loc[:, "relative cost decrease (%)"] = 100 - (df_2022.loc[:, "TotalCost"].to_numpy() / df_ref_2022.loc[:, "TotalCost"].to_numpy() * 100)
    df_2019_perfect.loc[:, "relative cost decrease (%)"] = 100 - (df_2019_perfect.loc[:, "TotalCost"].to_numpy() / df_ref_2019.loc[:, "TotalCost"].to_numpy() * 100)
    df_2021_perfect.loc[:, "relative cost decrease (%)"] = 100 - (df_2021_perfect.loc[:, "TotalCost"].to_numpy() / df_ref_2021.loc[:, "TotalCost"].to_numpy() * 100)
    df_2022_perfect.loc[:, "relative cost decrease (%)"] = 100 - (df_2022_perfect.loc[:, "TotalCost"].to_numpy() / df_ref_2022.loc[:, "TotalCost"].to_numpy() * 100)


    total_df = pd.concat([df_2019, df_2021, df_2022, df_2019_perfect, df_2021_perfect, df_2022_perfect])
    merged = total_df.merge(scen_table.loc[:, ["ID_Scenario", "ID_Building"]], on="ID_Scenario", how="left")
    merged["ID_Building"] = merged["ID_Building"].map(BUILDING_ID_2_NAME)


    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(
        data=merged.sort_values(by=["ID_Building","year"]),
        x="ID_Building",
        y="relative cost decrease (%)",
        hue="year",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    ax.set_xlabel("")
    ax.set_ylabel("operation cost savings (%)")
    plt.tight_layout()

    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Relative_cost_comparison_3_day_forecast_years.png")
    plt.savefig(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "Relative_cost_comparison_3_day_forecast_years.svg")
    plt.show()



def heat_demand_per_square_meter():
    df_year = load_yearly_data()
    db = return_db()
    building_table = db.read_dataframe(table_name=InputTables.OperationScenario_Component_Building.name) 
    simulation = df_year.query("method == 'simulation'").reset_index(drop=True) 
    demand_per_square_meter = pd.DataFrame(simulation.sort_values(["ID_Building"])["Q_RoomHeating"] / building_table.sort_values(["ID_Building"])["Af"] / 1_000, columns=["heat demand (kWh/m2)"])
    demand_per_square_meter["ID_Building"] = simulation["ID_Building"]
    df = pd.merge(demand_per_square_meter, building_table, on="ID_Building")
    df.to_csv(Path(r"C:\Users\mascherbauer\PycharmProjects\Z_Testing\Diss Graphiken") / "heat_demand_per_square_meter.csv", index=False, sep=";")

if __name__ == "__main__":
    scenario_name = "thermal_mass"
    # save_run()

    # amount_of_energy_shifted_over_outside_temperature(scenario_name)
    plot_realative_cost_decrease(scenario_name)
    # show_price_quantile_where_building_is_charged(scenario_name)
    plot_rolling_and_perfect_foresight_cost(scenario_name)

    # year_comparison_of_3_day_forecast()

    # compare_load_profiles()
    
    # only used it to compare with the loss factor calculated in Z_Testing
    # dynamic_thermal_mass_loss_factor()

    # recalculate the losses for heating tank from perfect foresight. Most of the losses happen in summer, this should not be included 



