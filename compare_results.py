import multiprocessing
from typing import List
import logging

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import math
import re
import os
import sys
import matplotlib
import pickle

# Get the absolute path of the directory two levels up
two_levels_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add this directory to sys.path
sys.path.insert(0, two_levels_up)
from basics.db import DB
from config import config, get_config
from models.operation.enums import OperationTable, OperationScenarioComponent


# log_file = Path("/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C") / "compare_6R2C_results.log"
# logging.basicConfig(
#    filename=log_file,
#    filemode='a',  # append mode
# )
# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)


class CompareModels:
    def __init__(self, db_name):
        self.project_name = db_name
        self.config = get_config(self.project_name)
        path_to_sqlite_file = config.output / f"{config.project_name}.sqlite"
        self.input_path = self.config.project_root / Path(f"projects/{self.project_name}/input")
        self.figure_path = self.config.fig
        self.db = DB(path_to_sqlite_file)
        self.scenario_table = self.db.read_dataframe(table_name=OperationTable.Scenarios.value)

        self.grey = "#95a5a6"
        self.orange = "orange"
        self.blue = "#3498db"
        self.red = "#e74c3c"
        self.green = "#2ecc71"

        self.building_names = {
            1: "EZFH_5_B",
            2: "EZFH_5_S",
            3: "EZFH_9_B",
            4: "EZFH_1_B",
            5: "EZFH_1_S",
            6: "MFH_5_B",
            7: "MFH_5_S",
            8: "MFH_1_B",
            9: "MFH_1_S"
        }
        self.cooling_status = {1: "no cooling",
                               2: "cooling"}
        self.heating_system = ["ideal", "floor_heating"]
        #self.Q_RoomHeating_6R2C_optimization, self.T_Room_6R2C_optimization, self.T_BuildingMass_6R2C_optimization, self.E_HP_out_6R2C_optimization = self.load_6R2C_results(reference=False)
        #self.Q_RoomHeating_6R2C_high_Qmax, self.T_Room_6R2C_high_Qmax, self.T_BuildingMass_6R2C_high_Qmax, self.E_HP_out_6R2C_high_Qmax = self.load_6R2C_results(high_Q_max=True)
        #self.Q_RoomHeating_6R2C_reference, self.T_Room_6R2C_reference, self.T_BuildingMass_6R2C_reference, self.E_HP_out_6R2C_reference = self.load_6R2C_results(reference=True)
    def load_6R2C_Q_max_resuts(self):
        path = Path("/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C")
        Q_RoomHeating_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        T_Room_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        T_BuildingMass_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        E_HP_out_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        for scenario_id in np.arange(1,10):
            df = pd.read_csv(path / f"Scenario_{scenario_id}_optimzation_high_Q_max.csv")
            Q_RoomHeating_dfs[scenario_id] = df["Q_RoomHeating"].to_numpy()
            T_Room_dfs[scenario_id] = df["T_Room"].to_numpy()
            T_BuildingMass_dfs[scenario_id] = df["T_BuildingMass"].to_numpy()
            E_HP_out_dfs[scenario_id] = df["E_Heating_HP_out"].to_numpy()
        self.Q_RoomHeating_optimization_6R2C_Q_max = Q_RoomHeating_dfs.copy()
        self.Q_RoomHeating_optimization_6R2C_Q_max.columns = self.Q_RoomHeating_optimization_6R2C_Q_max.columns.map(self.building_names)
        self.Q_RoomHeating_optimization_6R2C_Q_max = self.reorder_table(self.Q_RoomHeating_optimization_6R2C_Q_max)
        self.T_Room_optimization_6R2C_Q_max = T_Room_dfs.copy()
        self.T_Room_optimization_6R2C_Q_max.columns = self.T_Room_optimization_6R2C_Q_max.columns.map(self.building_names)
        self.T_Room_optimization_6R2C_Q_max = self.reorder_table(self.T_Room_optimization_6R2C_Q_max)
        self.T_BuildingMass_optimization_6R2C_Q_max = T_BuildingMass_dfs.copy()
        self.T_BuildingMass_optimization_6R2C_Q_max.columns = self.T_BuildingMass_optimization_6R2C_Q_max.columns.map(self.building_names)
        self.T_BuildingMass_optimization_6R2C_Q_max = self.reorder_table(self.T_BuildingMass_optimization_6R2C_Q_max)
        self.E_HP_out_optimization_6R2C_Q_max = E_HP_out_dfs.copy()
        self.E_HP_out_optimization_6R2C_Q_max.columns = self.E_HP_out_optimization_6R2C_Q_max.columns.map(self.building_names)
        self.E_HP_out_optimization_6R2C_Q_max = self.reorder_table(self.E_HP_out_optimization_6R2C_Q_max)


    def load_6R2C_results(self, reference: bool)->pd.DataFrame:
        path = Path("/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C")
        Q_RoomHeating_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        T_Room_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        T_BuildingMass_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        E_HP_out_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        for scenario_id in np.arange(1,10):
            if reference:
                df = pd.read_csv(path / f"Scenario_{scenario_id}_reference.csv")
            else:
                df = pd.read_csv(path / f"Scenario_{scenario_id}_optimzation.csv")
            Q_RoomHeating_dfs[scenario_id] = df["Q_RoomHeating"].to_numpy()
            T_Room_dfs[scenario_id] = df["T_Room"].to_numpy()
            T_BuildingMass_dfs[scenario_id] = df["T_BuildingMass"].to_numpy()
            E_HP_out_dfs[scenario_id] = df["E_Heating_HP_out"].to_numpy()
        
        if reference:
            self.Q_RoomHeating_reference_6R2C = Q_RoomHeating_dfs.copy()
            self.Q_RoomHeating_reference_6R2C.columns = self.Q_RoomHeating_reference_6R2C.columns.map(self.building_names)
            self.Q_RoomHeating_reference_6R2C = self.reorder_table(self.Q_RoomHeating_reference_6R2C)
            self.T_Room_reference_6R2C = T_Room_dfs.copy()
            self.T_Room_reference_6R2C.columns = self.T_Room_reference_6R2C.columns.map(self.building_names)
            self.T_Room_reference_6R2C = self.reorder_table(self.T_Room_reference_6R2C)
            self.T_BuildingMass_reference_6R2C = T_BuildingMass_dfs.copy()
            self.T_BuildingMass_reference_6R2C.columns = self.T_BuildingMass_reference_6R2C.columns.map(self.building_names)
            self.T_BuildingMass_reference_6R2C = self.reorder_table(self.T_BuildingMass_reference_6R2C)
            self.E_HP_out_reference_6R2C = E_HP_out_dfs.copy()
            self.E_HP_out_reference_6R2C.columns = self.E_HP_out_reference_6R2C.columns.map(self.building_names)
            self.E_HP_out_reference_6R2C = self.reorder_table(self.E_HP_out_reference_6R2C)


        else:
            self.Q_RoomHeating_optimization_6R2C = Q_RoomHeating_dfs.copy()
            self.Q_RoomHeating_optimization_6R2C.columns = self.Q_RoomHeating_optimization_6R2C.columns.map(self.building_names)
            self.Q_RoomHeating_optimization_6R2C = self.reorder_table(self.Q_RoomHeating_optimization_6R2C)
            self.T_Room_optimization_6R2C = T_Room_dfs.copy()
            self.T_Room_optimization_6R2C.columns = self.T_Room_optimization_6R2C.columns.map(self.building_names)
            self.T_Room_optimization_6R2C = self.reorder_table(self.T_Room_optimization_6R2C)
            self.T_BuildingMass_optimization_6R2C = T_BuildingMass_dfs.copy()
            self.T_BuildingMass_optimization_6R2C.columns = self.T_BuildingMass_optimization_6R2C.columns.map(self.building_names)
            self.T_BuildingMass_optimization_6R2C = self.reorder_table(self.T_BuildingMass_optimization_6R2C)
            self.E_HP_out_optimization_6R2C = E_HP_out_dfs.copy()
            self.E_HP_out_optimization_6R2C.columns = self.E_HP_out_optimization_6R2C.columns.map(self.building_names)
            self.E_HP_out_optimization_6R2C = self.reorder_table(self.E_HP_out_optimization_6R2C)
        

        Q_RoomHeating_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        T_Room_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        T_BuildingMass_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        E_HP_out_dfs = pd.DataFrame(columns=list(np.arange(1,10)))
        for scenario_id in np.arange(1,10):
            df = pd.read_csv(path / f"Scenario_{scenario_id}_optimization_fixed_indoor_temp.csv")
            Q_RoomHeating_dfs[scenario_id] = df["Q_RoomHeating"].to_numpy()
            T_Room_dfs[scenario_id] = df["T_Room"].to_numpy()
            T_BuildingMass_dfs[scenario_id] = df["T_BuildingMass"].to_numpy()
            E_HP_out_dfs[scenario_id] = df["E_Heating_HP_out"].to_numpy()
        self.Q_RoomHeating_optimization_6R2C_target_temp = Q_RoomHeating_dfs.copy()
        self.Q_RoomHeating_optimization_6R2C_target_temp.columns = self.Q_RoomHeating_optimization_6R2C_target_temp.columns.map(self.building_names)
        self.Q_RoomHeating_optimization_6R2C_target_temp = self.reorder_table(self.Q_RoomHeating_optimization_6R2C_target_temp)
        self.T_Room_optimization_6R2C_target_temp = T_Room_dfs.copy()
        self.T_Room_optimization_6R2C_target_temp.columns = self.T_Room_optimization_6R2C_target_temp.columns.map(self.building_names)
        self.T_Room_optimization_6R2C_target_temp = self.reorder_table(self.T_Room_optimization_6R2C_target_temp)
        self.T_BuildingMass_optimization_6R2C_target_temp = T_BuildingMass_dfs.copy()
        self.T_BuildingMass_optimization_6R2C_target_temp.columns = self.T_BuildingMass_optimization_6R2C_target_temp.columns.map(self.building_names)
        self.T_BuildingMass_optimization_6R2C_target_temp = self.reorder_table(self.T_BuildingMass_optimization_6R2C_target_temp)
        self.E_HP_out_optimization_6R2C_target_temp = E_HP_out_dfs.copy()
        self.E_HP_out_optimization_6R2C_target_temp.columns = self.E_HP_out_optimization_6R2C_target_temp.columns.map(self.building_names)
        self.E_HP_out_optimization_6R2C_target_temp = self.reorder_table(self.E_HP_out_optimization_6R2C_target_temp)
        

    def load_6R2C_parameters_dict(self):
        cf_hf_dict_path = Path("/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C") / "Cf_Hf_dict.pkl"
        with open(cf_hf_dict_path, 'rb') as f:
            Cf_Hf_dict = pickle.load(f)
        return Cf_Hf_dict

    def check_price_scenario_name(self, prize_scenario: str):
        if "cooling" in prize_scenario:
            return prize_scenario.replace("_cooling", "")

    def scenario_id_2_building_name(self, id_scenario: int) -> str:
        building_id = int(
            self.scenario_table.loc[self.scenario_table.loc[:, "ID_Scenario"] == id_scenario, "ID_Building"].iloc[0])
        return f"{self.building_names[building_id]}"

    def reorder_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """ reorder the buildings in the table for the plots to start with light weight structures and end with
        heavy weight. Makes interpreting the results easier"""
        order = ["EZFH_1_B",
                 "EZFH_1_S",
                 "EZFH_5_B",
                 "EZFH_5_S",
                 "EZFH_9_B",
                 "MFH_1_B",
                 "MFH_1_S",
                 "MFH_5_B",
                 "MFH_5_S"]
        return_df = df[order]
        return return_df

    def grab_scenario_ids_for_price(self, id_price: int, cooling: bool) -> list:
        if cooling:
            cooling_id = 1
        else:
            cooling_id = 2  # TODO achtung später hab ich nur noch ohne kühung
        ids = self.scenario_table.loc[(self.scenario_table.loc[:, "ID_EnergyPrice"] == id_price) &
                                      (self.scenario_table.loc[:, "ID_SpaceCoolingTechnology"] == cooling_id),
                                      "ID_Scenario"]
        return list(ids)

    def show_elec_prices(self):
        """plot the electricity prices that are used in this paper and print their mean and std"""
        # plot the prices in one figure:
        price_table = self.db.read_dataframe(table_name=OperationTable.EnergyPriceProfile.value)
        price_ids = [2,]# 3, 4]
        fig = plt.figure()
        ax = plt.gca()
        colors = [self.blue, self.orange, self.green, self.red]
        for price_id in price_ids:
            column_name = f"electricity_{price_id}"
            price = price_table.loc[:, column_name].to_numpy() * 1000  # cent/kWh
            ax.plot(np.arange(8760), price, label=f"hourly price 2019",
                    linewidth=0.3,
                    alpha=0.9,
                    color=colors[price_id - 1])
            # calculate and print the mean and standard deviation:
            print(f"price {price_id}: \n "
                  f"mean: {price.mean()} \n"
                  f"standard deviation: {price.std()}")
        ax.set_xlabel("hours")
        ax.set_ylabel("cent/kWh")
        ax.grid()
        plt.legend()
        # fig.savefig(self.path_to_figures / Path("electricity_prices.png"))
        fig.savefig(self.figure_path / Path("electricity_prices.svg"))
        plt.close()

    def read_daniel_heat_demand(self, price: str, cooling: bool, floor_heating: bool) -> pd.DataFrame:
        if floor_heating:
            system = "floor_heating"
        else:
            system = "ideal"
        if cooling:
            ac = "_cooling"
        else:
            ac = ""
        filename = f"heating_demand_daniel_{system}_{price}{ac}.csv"
        demand_df = pd.read_csv(self.input_path / Path(filename), sep=";")
        # rearrange the df:
        df = self.reorder_table(demand_df)
        df["Hour"] = np.arange(1, 8761)
        df.set_index("Hour", inplace=True)
        return df

    def read_daniel_cooling_demand(self, price: str):
        system = "ideal"
        filename = f"cooling_demand_daniel_{system}_{price}_cooling.csv"
        demand_df = pd.read_csv(self.input_path / Path(filename), sep=";")
        # rearrange the df:
        df = self.reorder_table(demand_df)
        return df

    def read_heat_demand(self, table_name: str, prize_scenario: str, cooling: bool) -> pd.DataFrame:
        """
        Args:
            table_name: str
            prize_scenario: str
            cooling: bool, True if cooling is included

        Returns: returns the heat demand in columns for each building on hours level for the given price scenario
        and cooling. Floor heating is not modelled in 5R1C therefore it is not an input parameter.

        """
        if prize_scenario == "basic":
            p = "price1"
        else:
            p = prize_scenario
        # scenario_id = self.grab_scenario_ids_for_price(int(re.findall(r"\d", p)[0]), cooling)
        demand = self.db.read_dataframe(table_name=table_name,
                                        column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"],
                                        )
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        # df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [col[1] for col in df.columns]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in df.columns]
        return self.reorder_table(df)

    def read_cooling_demand(self, table_name: str, prize_scenario: str, cooling: bool) -> pd.DataFrame:
        if prize_scenario == "basic":
            p = "price1"
        else:
            p = prize_scenario
        scenario_id = self.grab_scenario_ids_for_price(int(re.findall(r"\d", p)[0]), cooling)
        demand = self.db.read_dataframe(table_name=table_name,
                                        column_names=["ID_Scenario", "Q_RoomCooling", "Hour"],
                                        )
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
        return self.reorder_table(df)

    def read_indoor_temp(self, table_name: str, prize_scenario: str, cooling: int) -> pd.DataFrame:
        if prize_scenario == "basic":
            p = "price1"
        else:
            p = prize_scenario
        # scenario_id = self.grab_scenario_ids_for_price(int(re.findall(r"\d", p)[0]), cooling)
        temp = self.db.read_dataframe(table_name=table_name,
                                      column_names=["ID_Scenario", "T_Room", "Hour"])
        df = temp.pivot_table(index="Hour", columns="ID_Scenario")
        # df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [col[1] for col in df.columns]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in df.columns]
        return self.reorder_table(df)

    def read_indoor_temp_daniel(self, price, cooling: bool, floor_heating: bool) -> pd.DataFrame:
        if floor_heating:
            system = "floor_heating"
        else:
            system = "ideal"
        if cooling:
            ac = "_cooling"
        else:
            ac = ""
        filename = f"indoor_temp_daniel_{system}_{price}{ac}.csv"
        temp_df = pd.read_csv(self.input_path / Path(filename), sep=";")
        # rearrange the df:
        df = self.reorder_table(temp_df)
        return df

    def read_costs(self, table_name: str) -> pd.DataFrame:
        cost = self.db.read_dataframe(table_name=table_name,
                                      column_names=["TotalCost"])
        cost.loc[:, "index"] = np.arange(1, len(cost) + 1)
        df = cost.pivot_table(columns="index")
        df.columns = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S", "MFH_5_B", "MFH_5_S", "MFH_1_B",
                      "MFH_1_S"]
        return self.reorder_table(df)

    def read_electricity_price(self, price_id: str) -> np.array:
        if price_id == "basic":
            p = "price1"
        else:
            p = price_id
        number = int(p[-1])
        price = self.db.read_dataframe(table_name=OperationTable.EnergyPriceProfile.value,
                                       column_names=[f"electricity_{number}"]).to_numpy()
        return price

    def read_COP(self, table_name: str) -> pd.DataFrame:
        # cop is the same in alle price scenarios
        # scenario_id = self.grab_scenario_ids_for_price(id_price=1, cooling=False)  # cooling not needed for heating COP
        cop = self.db.read_dataframe(table_name=table_name,
                                     column_names=["ID_Scenario", "SpaceHeatingHourlyCOP",
                                                   "Hour"])
        # split the demand into columns for every Scenario ID:
        df = cop.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        # df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [col[1] for col in df.columns]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in df.columns]
        return self.reorder_table(df)

    def read_COP_AC(self, table_name: str) -> pd.DataFrame:
        # cop is the same in alle price scenarios
        scenario_id = self.grab_scenario_ids_for_price(id_price=1, cooling=True)
        cop = self.db.read_dataframe(table_name=table_name,
                                     column_names=["ID_Scenario", "CoolingHourlyCOP",
                                                   "Hour"])
        # split the demand into columns for every Scenario ID:
        df = cop.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
        return self.reorder_table(df)

    def calculate_costs(self, table_name: str, price_id: str, cooling: bool) -> pd.DataFrame:
        """ floor heatig does not have an impact because it is not explicitly modeled by the 5R1C"""
        price_profile = self.read_electricity_price(price_id).squeeze()
        heat_demand = self.read_heat_demand(table_name, price_id, cooling)
        COP_HP = self.read_COP(table_name)
        electricity_heat = (heat_demand / COP_HP).reset_index(drop=True)
        hourly_cost_heating = electricity_heat.mul(pd.Series(price_profile), axis=0)

        if cooling:
            cool_demand = self.read_cooling_demand(table_name, price_id, cooling)
            COP_AC = self.read_COP_AC(table_name)
            electricity_cool = (cool_demand / COP_AC).reset_index(drop=True)
            hourly_cost_cooling = electricity_cool.mul(pd.Series(price_profile), axis=0)
            hourly_cost = hourly_cost_heating + hourly_cost_cooling
        else:
            hourly_cost = hourly_cost_heating

        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_cooling_costs(self, table_name: str, price_id: str, cooling: bool) -> pd.DataFrame:
        """ floor heatig does not have an impact because it is not explicitly modeled by the 5R1C"""
        price_profile = self.read_electricity_price(price_id).squeeze()
        cool_demand = self.read_cooling_demand(table_name, price_id, cooling)
        COP_AC = self.read_COP_AC(table_name)
        electricity_cool = (cool_demand / COP_AC).reset_index(drop=True)

        hourly_cost_cooling = electricity_cool.mul(pd.Series(price_profile), axis=0)

        total_cost = hourly_cost_cooling.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_optimization_costs_6R2C(self, price_id, reference: bool):
        price_profile = self.read_electricity_price(price_id).squeeze()
        COP_HP = self.read_COP("OperationResult_OptimizationHour")
        if reference:
            electricity_heat = (self.Q_RoomHeating_reference_6R2C / COP_HP.reset_index(drop=True)).reset_index(drop=True)
        else:
            electricity_heat = (self.Q_RoomHeating_optimization_6R2C / COP_HP.reset_index(drop=True)).reset_index(drop=True)
        hourly_cost_heating = electricity_heat.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost_heating.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()


    def calculate_costs_daniel_opt(self, table_name: str, price_id: str, cooling: bool,
                                   floor_heating: bool) -> pd.DataFrame:
        price_profile = self.read_electricity_price(price_id).squeeze()
        COP_HP = self.read_COP(table_name)
        heat_demand = self.read_daniel_heat_demand(price=price_id, floor_heating=floor_heating, cooling=cooling)
        electricity_heat = (heat_demand / COP_HP).reset_index(drop=True)
        hourly_cost_heating = electricity_heat.mul(pd.Series(price_profile), axis=0)

        if cooling:
            COP_AC = self.read_COP_AC(table_name)
            cooling_demand = self.read_daniel_cooling_demand(price=price_id)
            electricity_cooling = (cooling_demand / COP_AC).reset_index(drop=True)
            hourly_cost_cooling = electricity_cooling.mul(pd.Series(price_profile), axis=0)
            hourly_cost = hourly_cost_heating + hourly_cost_cooling
        else:
            hourly_cost = hourly_cost_heating 
        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_cooling_costs_daniel_opt(self, table_name: str, price_id: str) -> pd.DataFrame:
        price_profile = self.read_electricity_price(price_id).squeeze()

        COP_AC = self.read_COP_AC(table_name)
        cooling_demand = self.read_daniel_cooling_demand(price=price_id)
        electricity_cooling = (cooling_demand / COP_AC).reset_index(drop=True)

        hourly_cost_cooling = electricity_cooling.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost_cooling.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_costs_daniel_ref(self, table_name: str, price_id: str, cooling: bool,
                                   floor_heating: bool) -> pd.DataFrame:
        price_profile = self.read_electricity_price(price_id).squeeze()
        COP_HP = self.read_COP(table_name)
        heat_demand = self.read_daniel_heat_demand(price="basic", cooling=cooling,
                                                   floor_heating=floor_heating)  # because here the indoor set temp is steady
        electricity_heat = (heat_demand / COP_HP).reset_index(drop=True)
        hourly_cost_heating = electricity_heat.mul(pd.Series(price_profile), axis=0)

        if cooling:
            COP_AC = self.read_COP_AC(table_name)
            cooling_demand = self.read_daniel_cooling_demand(price="basic")
            electricity_cooling = (cooling_demand / COP_AC).reset_index(drop=True)
            hourly_cost_cooling = electricity_cooling.mul(pd.Series(price_profile), axis=0)
            hourly_cost = hourly_cost_heating + hourly_cost_cooling
        else:
            hourly_cost = hourly_cost_heating 

        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_cooling_costs_daniel_ref(self, table_name: str, price_id: str) -> pd.DataFrame:
        price_profile = self.read_electricity_price(price_id).squeeze()

        COP_AC = self.read_COP_AC(table_name)
        cooling_demand = self.read_daniel_cooling_demand(price="basic")
        electricity_cooling = (cooling_demand / COP_AC).reset_index(drop=True)

        hourly_cost_cooling = electricity_cooling.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost_cooling.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def compare_hourly_profile(self, demand_list: list, demand_names: list, title: str):
        # compare hourly demand
        column_number = len(list(demand_list[0].columns))
        x_axis = np.arange(8760)
        colors = ["blue", "orange", "green", "red"]

        fig = make_subplots(
            rows=column_number,
            cols=1,
            subplot_titles=sorted(list(demand_list[0].columns)),
            shared_xaxes=True,
        )
        for j, demand in enumerate(demand_list):
            for i, column_name in enumerate(sorted(list(demand.columns))):
                fig.add_trace(
                    go.Scatter(x=x_axis, y=demand[column_name],
                               name=demand_names[j],
                               line=dict(color=colors[j])),

                    row=i + 1,
                    col=1,
                )
        fig.update_layout(title_text=title, height=400 * column_number, width=1600)
        fig.show()

    def compare_yearly_value(self, profile_list: List[pd.DataFrame], profile_names: list, title: str):
        # compare total demand
        total_df = pd.concat([profile.sum() for profile in profile_list], axis=1) / 1_000  # kW
        total_df.columns = [name for name in profile_names]
        if "cost" in title:
            unit = "€"
        else:
            unit = "kWh"
        fig2 = px.bar(
            data_frame=total_df,
            barmode="group",
            labels={
                "index": "Building",
                "value": f"{title} {unit}",
            }
        )
        fig2.update_layout(title_text=title)
        fig_name = f"{title}.svg"
        fig2.write_image(self.figure_path.as_posix() + f"/{fig_name.replace(' ', '_')}")
        fig2.show()

    def yearly_comparison(self,
                          figure, 
                          axes,
                          profile_list: List[pd.DataFrame],
                          profile_names: list,
                          y_label: str,
                          ax_number: int) -> dict:
        if "kW" in y_label:
            denominator = 1_000  # kW
        elif "€" in y_label:
            denominator = 100  # €
        else:
            print("neither kW nor € in y_label")
        # compare total demand
        total_df = pd.concat([profile.sum() for profile in profile_list], axis=1) / denominator  # kW
        total_df.columns = [name for name in profile_names]
        ax = axes.flatten()[ax_number]

        plot = sns.barplot(
            data=total_df.reset_index().melt(id_vars="index").rename(
                columns={"variable": "model", "value": y_label, "index": "Building"}),
            x="Building",
            y=y_label,
            hue="model",
            ax=ax)
        # save legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # remove legend from subplot
        plot.legend().remove()
        ax.set_title(f"electricity price scenario {ax_number + 1}")

        return by_label

    def relative_yearly_comparison(self,
                                   figure, 
                                   axes,
                                   profile_list: List[pd.DataFrame],
                                   profile_names: list,
                                   y_label: str,
                                   ax_number: int) -> dict:
        if "demand" in y_label:
            denominator = 1_000  # kW
        elif "cost" in y_label:
            denominator = 100  # €
        else:
            print("neither kW nor € in y_label")
        # compare total demand
        total_df = pd.concat([profile.sum() for profile in profile_list], axis=1) / denominator  # kW
        total_df.columns = [name for name in profile_names]

        total_df.loc[:, "5R1C"] = (total_df.loc[:, "5R1C optimized"] - total_df.loc[:, "5R1C"]) / total_df.loc[:, "5R1C"] * 100
        total_df.loc[:, "IDA ICE"] = (total_df.loc[:, "IDA ICE optimized"] - total_df.loc[:, "IDA ICE"]) / total_df.loc[ :, "IDA ICE"] * 100
        total_df = total_df.drop(columns=["5R1C optimized", "IDA ICE optimized"])

        ax = axes.flatten()[ax_number]
        # optimization is always measured against the reference case (reference = 100%)
        plot = sns.barplot(
            data=total_df.reset_index().melt(id_vars="index").rename(
                columns={"variable": "model", "value": y_label, "index": "Building"}),
            x="Building",
            y=y_label,
            hue="model",
            ax=ax,
            palette=["blue", "green"])
        # save legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # remove legend from subplot
        plot.legend().remove()
        ax.set_title(f"electricity price: {ax_number + 1}")

        return by_label

    def plot_yearly_heat_demand(self, prices: list, cooling: bool, floor_heating: bool):
        plt.style.use("seaborn-paper")

        # heat demand:
        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        # ref heat demand is always the same
        heat_demand_ref = self.read_heat_demand(table_name=OperationTable.ResultRefHour.value,
                                                prize_scenario="basic",
                                                cooling=cooling)
        # ref IDA ICE is the one where the indoor set temp is not changed (basic)
        heat_demand_IDA_ref = self.read_daniel_heat_demand(price="basic",
                                                           cooling=cooling,
                                                           floor_heating=floor_heating)
        ax_number = 0
        for price in prices:
            heat_demand_opt = self.read_heat_demand(table_name=OperationTable.ResultOptHour.value,
                                                    prize_scenario=price,
                                                    cooling=cooling)
            heat_demand_daniel = self.read_daniel_heat_demand(price=price,
                                                              cooling=cooling,
                                                              floor_heating=floor_heating)
            by_label = self.yearly_comparison(fig, axes,
                                              [heat_demand_opt, heat_demand_ref, heat_demand_daniel,
                                               heat_demand_IDA_ref],
                                              ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                              y_label="heat demand (kWh)", ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                   borderaxespad=0, ncol=4, bbox_transform=fig.transFigure)
        if floor_heating:
            system = "floor heating"
        else:
            system = "ideal heating"
        if cooling:
            ac = "cooling"
        else:
            ac = "no cooling"
        fig.suptitle(f"Total heat demand with {system} and {ac}")
        for ax in [fig.axes[2], fig.axes[3]]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig.savefig(self.figure_path / Path(f"Total heat demand with {system} and {ac}.svg"))
        plt.close()

    def plot_yearly_cooling_demand(self, prices: list):
        plt.style.use("seaborn-paper")

        # cooling demand:
        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        # ref heat demand is always the same
        cooling_demand_ref = self.read_cooling_demand(table_name=OperationTable.ResultRefHour.value,
                                                      prize_scenario="basic",
                                                      cooling=True)
        # ref IDA ICE is the one where the indoor set temp is not changed (basic)
        cooling_demand_IDA_ref = self.read_daniel_cooling_demand(price="basic")
        ax_number = 0
        for price in prices:
            cooling_demand_opt = self.read_cooling_demand(table_name=OperationTable.ResultOptHour.value,
                                                          prize_scenario=price,
                                                          cooling=True)
            cooling_demand_daniel = self.read_daniel_cooling_demand(price=price)
            by_label = self.yearly_comparison(fig, axes,
                                              [cooling_demand_opt, cooling_demand_ref, cooling_demand_daniel,
                                               cooling_demand_IDA_ref],
                                              ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                              y_label="cooling demand (kWh)", ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                   borderaxespad=0, ncol=4, bbox_transform=fig.transFigure)

        fig.suptitle(f"Total cooling demand")
        for ax in [fig.axes[2], fig.axes[3]]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig.savefig(self.figure_path / Path(f"Total cooling demand.svg"))
        plt.close()

    def plot_yearly_costs(self, prices: list, cooling: bool, floor_heating: bool):
        # cost
        plt.style.use("seaborn-paper")
        # create figure
        fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        ax_number = 0
        for price in prices:
            costs_ref = self.calculate_costs(table_name=OperationTable.ResultRefHour.value,
                                             price_id=price,
                                             cooling=cooling)
            costs_opt = self.calculate_costs(table_name=OperationTable.ResultOptHour.value,
                                             price_id=price,
                                             cooling=cooling)
            cost_daniel_ref = self.calculate_costs_daniel_ref(table_name=OperationTable.ResultRefHour.value,
                                                              price_id=price,
                                                              cooling=cooling,
                                                              floor_heating=floor_heating)
            cost_daniel_opt = self.calculate_costs_daniel_opt(table_name=OperationTable.ResultOptHour.value,
                                                              price_id=price,
                                                              cooling=cooling,
                                                              floor_heating=floor_heating)

            if cooling:
                ac = " and cooling"
            else:
                ac = ""
            by_label = self.yearly_comparison(fig2, axes2,
                                              [costs_opt, costs_ref, cost_daniel_opt, cost_daniel_ref],
                                              ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                              y_label=f"heating{ac} cost (€)", ax_number=ax_number)
            ax_number += 1

        if floor_heating:
            system = "with_floor_heating"
        else:
            system = "ideal"
        # plot legend in the top middle of the figure
        fig2.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                    borderaxespad=0, ncol=4, bbox_transform=fig2.transFigure)
        fig2.suptitle(f"total_heating{ac}_cost_{system}".replace("_", " ").replace("ideal", ""))
        for ax in [fig2.axes[2], fig2.axes[3]]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig2.savefig(self.figure_path / Path(f"total_heating{ac}_cost_{system}.svg"))
        plt.close()

    def subplots_yearly(self, prices: list, cooling: bool, floor_heating: bool):
        """ plot the total heat and cooling demand over the year comparing 5R1C to IDA ICE"""
        self.plot_yearly_heat_demand(prices=prices, cooling=cooling, floor_heating=floor_heating)

        if cooling:
            self.plot_yearly_cooling_demand(prices=prices)

        self.plot_yearly_costs(prices=prices, cooling=cooling, floor_heating=floor_heating)

    def plot_relative_heat_demand(self, prices: list, floor_heating: bool, cooling: bool):
        plt.style.use("seaborn-paper")

        # heat demand:
        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        # ref heat demand is always the same
        heat_demand_ref = self.read_heat_demand(table_name=OperationTable.ResultRefHour.value,
                                                prize_scenario="basic",
                                                cooling=cooling)
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand(price="basic",
                                                           cooling=cooling,
                                                           floor_heating=floor_heating)
        ax_number = 0
        for price in prices:
            heat_demand_opt = self.read_heat_demand(table_name=OperationTable.ResultOptHour.value,
                                                    prize_scenario=price,
                                                    cooling=cooling)
            heat_demand_daniel = self.read_daniel_heat_demand(price=price,
                                                              cooling=cooling,
                                                              floor_heating=floor_heating)
            by_label = self.relative_yearly_comparison(fig, axes,
                                                       [heat_demand_opt, heat_demand_ref, heat_demand_daniel,
                                                        heat_demand_IDA_ref],
                                                       ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                                       y_label="relative change in heat demand (%)",
                                                       ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                   borderaxespad=0, ncol=4, bbox_transform=fig.transFigure)
        if cooling:
            ac = "cooling"
        else:
            ac = "no cooling"
        if floor_heating:
            system = "floor heating"
        else:
            system = "ideal"
        fig.suptitle(f"Heating system: {system}, {ac}")
        for ax in [fig.axes[2], fig.axes[3]]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig.savefig(self.figure_path / Path(f"relative_change_in_heat_demand_{system}_{ac.replace(' ', '_')}.svg"))
        plt.close()

    def plot_relative_cooling_demand(self, prices: list):
        plt.style.use("seaborn-paper")

        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        # ref heat demand is always the same
        cooling_demand_ref = self.read_cooling_demand(table_name=OperationTable.ResultRefHour.value,
                                                      prize_scenario="basic",
                                                      cooling=True)
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        cooling_demand_IDA_ref = self.read_daniel_cooling_demand(price="basic")
        ax_number = 0
        for price in prices:
            cooling_demand_opt = self.read_cooling_demand(table_name=OperationTable.ResultOptHour.value,
                                                          prize_scenario=price,
                                                          cooling=True)
            cooling_demand_daniel = self.read_daniel_cooling_demand(price=price)
            by_label = self.relative_yearly_comparison(fig, axes,
                                                       [cooling_demand_opt, cooling_demand_ref, cooling_demand_daniel,
                                                        cooling_demand_IDA_ref],
                                                       ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                                       y_label="relative change in cooling demand (%)",
                                                       ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                   borderaxespad=0, ncol=4, bbox_transform=fig.transFigure)

        fig.suptitle(f"Cooling demand")
        for ax in [fig.axes[2], fig.axes[3]]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig.savefig(self.figure_path / Path(f"relative_change_in_cooling_demand.svg"))
        plt.close()

    def plot_yearly_heat_demand_floor_ideal_not_optimized(self):
        """ plot the yearly heat demand for 5R1C, IDA ICE and IDA ICE Floor heating"""
        plt.style.use("seaborn-paper")
        heat_5R1C = self.read_heat_demand(table_name=OperationTable.ResultRefHour.value,
                                          prize_scenario="basic",
                                          cooling=False).sum() / 1_000  # kWh
        heat_ida = self.read_daniel_heat_demand(price="basic", cooling=False, floor_heating=False).sum() / 1_000
        heat_ida_floor = self.read_daniel_heat_demand(price="basic", cooling=False, floor_heating=True).sum() / 1_000
        plot_df = pd.concat([heat_5R1C, heat_ida, heat_ida_floor], axis=1)  # kWh
        plot_df.columns = ["5R1C", "IDA ICE ideal heating", "IDA ICE floor heating"]
        plot_df = plot_df.reset_index().rename(columns={"index": "Building"})
        melted_df = plot_df.melt(id_vars="Building").rename(columns={"value": "heat demand (kWh)",
                                                                     "variable": "model"})

        sns.barplot(data=melted_df, x="Building", y="heat demand (kWh)", hue="model",
                    palette=["blue", self.orange, "green"])
        ax = plt.gca()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.get_yaxis().set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
        plt.title("heat demand")
        plt.savefig(self.figure_path / f"Yearly_heat_demand_comparison.svg")
        plt.close()

    def plot_normalized_yearly_heat_demand_floor_ideal_not_optimized(self):
        """ plot the yearly heat demand for 5R1C, IDA ICE and IDA ICE Floor heating"""
        heat_5R1C = self.read_heat_demand(table_name=OperationTable.ResultRefHour.value,
                                          prize_scenario="basic",
                                          cooling=False).sum() / 1_000  # kWh
        heat_ida = self.read_daniel_heat_demand(price="basic", cooling=False, floor_heating=False).sum() / 1_000
        heat_ida_floor = self.read_daniel_heat_demand(price="basic", cooling=False, floor_heating=True).sum() / 1_000
        heat_6R2C = self.Q_RoomHeating_reference_6R2C.sum() / 1_000
        plot_df = pd.concat([heat_5R1C, heat_ida, heat_6R2C, heat_ida_floor], axis=1)  # kWh
        columns = ["5R1C", "IDA ICE ideal heating", "6R2C", "IDA ICE floor heating"]
        plot_df.columns = columns
        building_df = self.db.read_dataframe(table_name=OperationScenarioComponent.Building.table_name, column_names=["ID_Building", "type", "Af"])
        building_df["index"] = building_df.loc[:, "ID_Building"].map(self.building_names)
        plot_df_merged = plot_df.reset_index().merge(building_df.loc[:, ["index", "Af"]], how="left")
        plot_df_normalized = plot_df_merged.loc[:, columns].div(plot_df_merged.loc[:, "Af"], axis=0) 
        plot_df_normalized["Building"] = plot_df_merged["index"]
        
        melted_df = plot_df_normalized.melt(id_vars="Building").rename(columns={"value": "heat demand (kWh/m$^2$)",
                                                                                "variable": "model"})

        sns.barplot(data=melted_df, 
                    x="Building", 
                    y="heat demand (kWh/m$^2$)", 
                    hue="model",
                    palette=sns.color_palette()#["blue", self.orange, "green", "red"]
                )
        ax = plt.gca()
        ax.set_xticklabels([label.get_text().replace("_", " ") for label in ax.get_xticklabels()], rotation=45, ha='center')
        ax.get_yaxis().set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
        legend = ax.legend()
        legend.set_title(None)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(self.figure_path / f"Yearly_heat_demand_comparison_normalized.svg")
        plt.close()

    def calculate_relative_cost_change(self, opt_df, ref_df) -> pd.DataFrame:
        # compare total demand
        relative = (opt_df - ref_df) / ref_df * 100
        return relative

    def plot_relative_cost_reduction_floor_ideal(self):
        """ plot the relative cost reduction of ideal heating together with floor heating in comparison"""
        # create figure

       
        costs_ref = self.calculate_costs(table_name=OperationTable.ResultRefHour.value,
                                            price_id="price2",
                                            cooling=False)
        costs_opt = self.calculate_costs(table_name=OperationTable.ResultOptHour.value,
                                            price_id="price2",
                                            cooling=False)
        relative_5R1C = self.calculate_relative_cost_change(opt_df=costs_opt, ref_df=costs_ref)

        cost_daniel_ref = self.calculate_costs_daniel_ref(table_name=OperationTable.ResultRefHour.value,
                                                            price_id="price2",
                                                            cooling=False,
                                                            floor_heating=False)
        cost_daniel_opt = self.calculate_costs_daniel_opt(table_name=OperationTable.ResultOptHour.value,
                                                            price_id="price2",
                                                            cooling=False,
                                                            floor_heating=False)
        relative_ida = self.calculate_relative_cost_change(opt_df=cost_daniel_opt, ref_df=cost_daniel_ref)

        cost_daniel_ref_floor = self.calculate_costs_daniel_ref(table_name=OperationTable.ResultRefHour.value,
                                                                price_id="price2",
                                                                cooling=False,
                                                                floor_heating=True)
        cost_daniel_opt_floor = self.calculate_costs_daniel_opt(table_name=OperationTable.ResultOptHour.value,
                                                                price_id="price2",
                                                                cooling=False,
                                                                floor_heating=True)
        relative_ida_floor = self.calculate_relative_cost_change(opt_df=cost_daniel_opt_floor,
                                                                    ref_df=cost_daniel_ref_floor)
        
        cost_6R2C_opt = self.calculate_optimization_costs_6R2C(price_id="price2", reference=False)
        cost_6R2C_ref = self.calculate_optimization_costs_6R2C(price_id="price2", reference=True)
        relative_6R2C = self.calculate_relative_cost_change(opt_df=cost_6R2C_opt, ref_df=cost_6R2C_ref)
        
        # plot the relative bars:
        # create one pandas dataframe
        plot_df = pd.concat([relative_5R1C.T, relative_ida.T,relative_6R2C.T, relative_ida_floor.T], axis=1).reset_index()
        plot_df.columns = ["Building", "5R1C", "IDA ICE ideal heating", "6R2C", "IDA ICE floor heating"]
        melted_df = plot_df.melt(id_vars="Building").rename(columns={"variable": "model",
                                                                        "value": "change in cost (%)"})

        fontsize = 12
        fig = plt.figure()
        plot = sns.barplot(data=melted_df, 
                           x="Building", 
                           y="change in cost (%)", 
                           hue="model",
                            palette=sns.color_palette()
                        )
        ax = fig.gca()
        ax.set_xlabel("")
        # save legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # remove legend from subplot
        plot.legend().remove()
        # plot legend in the top middle of the figure
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 1.01), loc="lower center",
                    borderaxespad=0, ncol=4, fontsize=8)

        ax.set_xticklabels([label.get_text().replace("_", " ") for label in ax.get_xticklabels()], rotation=45, ha='center', fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.set_ylabel("change in cost (%)", fontsize=fontsize)

        plt.savefig(
            self.figure_path / Path(f"cost_change_IDA_ICE.svg"))
        plt.close()

    def plot_relative_cost_reduction(self, prices: list, floor_heating: bool, cooling: bool):

        # cost
        plt.style.use("seaborn-paper")
        # create figure
        fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(5, 8.5), sharex=True, sharey=True)
        ax_number = 0
        for price in prices:
            if price == "basic":
                continue
            costs_ref = self.calculate_costs(table_name=OperationTable.ResultRefHour.value,
                                             price_id=price,
                                             cooling=cooling)
            costs_opt = self.calculate_costs(table_name=OperationTable.ResultOptHour.value,
                                             price_id=price,
                                             cooling=cooling)

            cost_daniel_ref = self.calculate_costs_daniel_ref(table_name=OperationTable.ResultRefHour.value,
                                                              price_id=price,
                                                              cooling=cooling,
                                                              floor_heating=floor_heating)
            cost_daniel_opt = self.calculate_costs_daniel_opt(table_name=OperationTable.ResultOptHour.value,
                                                              price_id=price,
                                                              cooling=cooling,
                                                              floor_heating=floor_heating)
            if cooling:
                ac = " and cooling"
            else:
                ac = ""
            if floor_heating:
                system = "floor heating"
            else:
                system = "ideal heating"
            by_label = self.relative_yearly_comparison(fig2, axes2,
                                                       [costs_opt, costs_ref, cost_daniel_opt, cost_daniel_ref],
                                                       ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                                       y_label=f"relative change in heating{ac} cost (%)",
                                                       ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig2.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                    borderaxespad=0, ncol=4, bbox_transform=fig2.transFigure)
        ax = fig2.axes[2]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig2.suptitle(f"relative_change_in_heating{ac} cost with {system}".replace('_', ' '))
        fig2.savefig(
            self.figure_path / Path(f"relative_change_in_heating{ac}_cost_with_{system.replace(' ', '_')}.svg"))
        plt.close()

    def plot_relative_cooling_cost_reduction(self, prices: list, cooling: bool):
        # cost
        plt.style.use("seaborn-paper")
        # create figure
        fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        ax_number = 0
        for price in prices:
            cooling_costs_ref = self.calculate_cooling_costs(table_name=OperationTable.ResultRefHour.value,
                                                             price_id=price,
                                                             cooling=cooling)
            cooling_costs_opt = self.calculate_cooling_costs(table_name=OperationTable.ResultOptHour.value,
                                                             price_id=price,
                                                             cooling=cooling)

            cooling_cost_daniel_ref = self.calculate_cooling_costs_daniel_ref(
                table_name=OperationTable.ResultRefHour.value,
                price_id=price)
            cooling_cost_daniel_opt = self.calculate_cooling_costs_daniel_opt(
                table_name=OperationTable.ResultOptHour.value,
                price_id=price)

            by_label = self.relative_yearly_comparison(fig2, axes2,
                                                       [cooling_costs_opt, cooling_costs_ref, cooling_cost_daniel_opt,
                                                        cooling_cost_daniel_ref],
                                                       ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                                       y_label=f"relative change in cooling cost (%)",
                                                       ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig2.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                    borderaxespad=0, ncol=4, bbox_transform=fig2.transFigure)
        for ax in [fig2.axes[2], fig2.axes[3]]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig2.suptitle(f"relative change in cooling cost")
        fig2.savefig(self.figure_path / Path(f"relative_change_in_cooling_cost.svg"))
        plt.close()

    def subplots_relative(self, prizes: list, floor_heating: bool, cooling: bool):
        """ compares the relative change in heat demand between the reference case where no load is shifted
        and the optimized case with load shifting in the 5R1C model and the IDA ICE model.
        The relative change in heat and cooling demand as well as price is compared"""
        self.plot_relative_heat_demand(prices=prizes, floor_heating=floor_heating, cooling=cooling)
        if cooling:
            self.plot_relative_cooling_demand(prices=prizes)  # cooling scenario is only without floor heating

        self.plot_relative_cost_reduction(prices=prizes, floor_heating=floor_heating, cooling=cooling)

    def find_cooling_time_for_ida_ice(self):
        """ find the time window where IDA ICE is supposed to follow the temperature in 5R1C model in summer"""
        # only consider prices 2,3 ,4
        for price in ["price2", "price3", "price4"]:
            # read cooling demand from 5R1C
            cooling_5R1C = self.read_cooling_demand(table_name=OperationTable.ResultOptHour.value,
                                                    prize_scenario=price,
                                                    cooling=True)
            result = {}
            for column in cooling_5R1C.columns:
                non_zero = cooling_5R1C[column][cooling_5R1C[column] != 0]
                if non_zero.empty:
                    result[column] = (0, 8759)
                else:
                    result[column] = (non_zero.index.min(), non_zero.index.max())

            pd.DataFrame(result).to_csv(self.input_path.parent / Path(f"output/cooling_interval_{price}.csv"),
                                        sep=";",
                                        index=False)

    def indoor_temp_to_csv(self, cooling: bool):

        for price_id in [1, 2, 3, 4]:  # price 1 is linear
            scenario_ids = self.grab_scenario_ids_for_price(id_price=price_id, cooling=cooling)
            temp_df = self.db.read_dataframe(table_name=OperationTable.ResultOptHour.value,
                                             column_names=["ID_Scenario", "Hour"],
                                             filter={"ID_Scenario": 1})
            for scen_id in scenario_ids:
                indoor_temp_opt = self.db.read_dataframe(
                    table_name=OperationTable.ResultOptHour.value,
                    column_names=["T_Room"],
                    filter={"ID_Scenario": scen_id}).rename(
                    columns={"T_Room": self.scenario_id_2_building_name(scen_id)})

                temp_df = pd.concat([temp_df, indoor_temp_opt], axis=1)

            if cooling:
                # save indoor temp opt to csv for daniel:
                temp_df.to_csv(self.input_path.parent / Path(f"output/indoor_set_temp_price{price_id}_cooling_new.csv"),
                               sep=";",
                               index=False)
            else:
                temp_df.to_csv(self.input_path.parent / Path(f"output/indoor_set_temp_price{price_id}_new.csv"),
                               sep=";", index=False)

            del temp_df

        if cooling:
            self.find_cooling_time_for_ida_ice()

    def show_plotly_comparison(self, prices: list, cooling: bool, floor_heating: bool):
        # ref heat demand is always the same
        heat_demand_ref = self.read_heat_demand(table_name=OperationTable.ResultRefHour.value,
                                                prize_scenario=prices[0],
                                                cooling=cooling)
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand(price="basic", cooling=cooling, floor_heating=floor_heating)
        temperature_daniel_ref = self.read_indoor_temp_daniel(price="basic", cooling=cooling,
                                                              floor_heating=floor_heating)
        cooling_demand_daniel_ref = self.read_daniel_cooling_demand(price="basic")
        for price in prices:
            # heat demand
            heat_demand_opt = self.read_heat_demand(table_name=OperationTable.ResultOptHour.value,
                                                    prize_scenario=price,
                                                    cooling=cooling)
            heat_demand_daniel = self.read_daniel_heat_demand(price=price, cooling=cooling, floor_heating=floor_heating)

            # temperature

            indoor_temp_daniel_opt = self.read_indoor_temp_daniel(price=price, cooling=cooling,
                                                                  floor_heating=floor_heating)
            indoor_temp_opt = self.read_indoor_temp(table_name=OperationTable.ResultOptHour.value,
                                                    prize_scenario=price,
                                                    cooling=cooling)
            indoor_temp_ref = self.read_indoor_temp(table_name=OperationTable.ResultRefHour.value,
                                                    prize_scenario=price,
                                                    cooling=cooling)

            if cooling:
                ac = "with cooling"
            else:
                ac = "no cooling"
            if floor_heating:
                system = "floor heating"
            else:
                system = "ideal"
            self.compare_hourly_profile([heat_demand_IDA_ref, heat_demand_ref, heat_demand_daniel, heat_demand_opt],
                                        ["IDA ICE", "5R1C", "IDA ICE optimized", "5R1C optimized"],
                                        f"heat demand {price}, {ac}, Heating system: {system}")

            self.compare_hourly_profile(
                [indoor_temp_ref, temperature_daniel_ref, indoor_temp_daniel_opt, indoor_temp_opt],
                ["5R1C", "IDA ICE", "IDA ICE optimized", "5R1C optimized"],
                f"indoor temperature {price}, {ac}, Heating system: {system}")

            if cooling:
                cooling_demand_opt = self.read_cooling_demand(table_name=OperationTable.ResultOptHour.value,
                                                              prize_scenario=price,
                                                              cooling=cooling)
                cooling_demand_ref = self.read_cooling_demand(table_name=OperationTable.ResultRefHour.value,
                                                              prize_scenario=price,
                                                              cooling=cooling)
                cooling_demand_daniel_opt = self.read_daniel_cooling_demand(price=price)

                self.compare_hourly_profile(
                    [cooling_demand_ref, cooling_demand_daniel_ref, cooling_demand_daniel_opt, cooling_demand_opt],
                    ["5R1C", "IDA ICE", "IDA ICE optimized", "5R1C optimized"],
                    f"cooling demand {price}")

    def calculate_RMSE(self, profile_IDA_ICE: np.array, profile_5R1C: np.array) -> float:
        MSE = np.square(np.subtract(profile_IDA_ICE, profile_5R1C)).mean()
        RMSE = math.sqrt(MSE)
        return RMSE

    def show_rmse(self, prizes: list, floor_heating: bool, cooling: bool):
        """ plots the RMSE for all scenarios. In one plot all buildings """
        plt.style.use("seaborn-paper")
        fig_temp, axes_temp = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True, sharey=True)
        fig_heat, axes_heat = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True, sharey=True)
        for i, price in enumerate(prizes):
            # ref heat demand is always the same
            heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value, price, cooling)
            indoor_temp_ref = self.read_indoor_temp(OperationTable.ResultRefHour.value, price, cooling)
            # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
            heat_demand_IDA_ref = self.read_daniel_heat_demand(price=price, cooling=cooling,
                                                               floor_heating=floor_heating)
            temperature_daniel_ref = self.read_indoor_temp_daniel(price=price, cooling=cooling,
                                                                  floor_heating=floor_heating)
            building_names = list(heat_demand_IDA_ref.columns)
            rmse_temp_list = []
            rmse_heat_list = []
            for building in building_names:
                rmse_temp = self.calculate_RMSE(temperature_daniel_ref.loc[:, building].to_numpy(),
                                                indoor_temp_ref.loc[:, building].to_numpy())
                rmse_temp_list.append(rmse_temp)
                rmse_heat = self.calculate_RMSE(heat_demand_IDA_ref.loc[:, building].to_numpy(),
                                                heat_demand_ref.loc[:, building].to_numpy())
                rmse_heat_list.append(rmse_heat)

            ax_temp = axes_temp.flatten()[i]
            ax_heat = axes_heat.flatten()[i]

            # plot the RMSE:
            ax_temp.bar(building_names, rmse_temp_list)
            ax_temp.set_ylabel("RMSE indoor temperature")
            ax_temp.set_title(f"Price ID: {price[-1]}")

            ax_heat.bar(building_names, rmse_heat_list)
            ax_heat.set_ylabel("RMSE heat demand")
            ax_heat.set_title(f"Price ID: {price[-1]}")

        if cooling:
            ac = "cooling"
        else:
            ac = "no cooling"
        if floor_heating:
            system = "floor heating"
        else:
            system = "ideal"
        fig_temp.suptitle(f"Heating system: {system}, {ac}")
        fig_heat.suptitle(f"Heating system: {system}, {ac}")
        fig_temp.tight_layout()
        fig_heat.tight_layout()

        fig_temp.savefig(self.figure_path / f"RMSE_temperature_{ac.replace(' ', '_')}_{system}.svg")
        fig_heat.savefig(self.figure_path / f"RMSE_heat_demand_{ac.replace(' ', '_')}_{system}.svg")
        fig_temp.show()
        fig_heat.show()

    def show_heat_demand_for_one_building_in_multiple_scenarios(self, price_id: str, building: str = "EZFH_5_B"):
        self.load_6R2C_Q_max_resuts()
        floor_heating_df = self.read_daniel_heat_demand(price=price_id, cooling=False, floor_heating=True)
        floor_demand_ida = floor_heating_df[building].to_numpy() / 1_000  # kWh

        ideal_demand_ida = self.read_daniel_heat_demand(price=price_id, cooling=False, floor_heating=False)[
                               building].to_numpy() / 1_000  # kWh

        opt_demand_5R1C = self.read_heat_demand(table_name=OperationTable.ResultOptHour.value,
                                                prize_scenario=price_id,
                                                cooling=False)[building].to_numpy() / 1_000  # kWh
        
        opt_demand_6R2C = self.Q_RoomHeating_optimization_6R2C[building].to_numpy() / 1_000
        opt_demand_6R2C_set_temp = self.Q_RoomHeating_optimization_6R2C_target_temp[building].to_numpy() / 1_000
        opt_demand_6R2C_Q_max = self.Q_RoomHeating_optimization_6R2C_Q_max[building].to_numpy() / 1_000

        elec_price = self.read_electricity_price(price_id=price_id).flatten() * 1000  # cent/kWh
        outside_temp = self.db.read_dataframe(table_name=OperationTable.RegionWeatherProfile.value,
                                       column_names=[f"temperature"]).to_numpy().flatten()
        solar_radiation = self.db.read_dataframe(table_name=OperationTable.RegionWeatherProfile.value,
                                       column_names=[f"radiation_south","radiation_north","radiation_west","radiation_east"]).sum(axis=1).to_numpy()


        temp_floor_ida = self.read_indoor_temp_daniel(price=price_id, cooling=False, floor_heating=True)[
            building].to_numpy()
        temp_ideal_ida = self.read_indoor_temp_daniel(price=price_id, cooling=False, floor_heating=False)[
            building].to_numpy()
        temp_5R1C = self.read_indoor_temp(table_name=OperationTable.ResultOptHour.value,
                                          prize_scenario=price_id,
                                          cooling=False)[building].to_numpy()
        temp_6R2C = self.T_Room_optimization_6R2C[building].to_numpy()
        temp_6R2C_set_temp = self.T_Room_optimization_6R2C_target_temp[building].to_numpy()
        temp_6R2C_Q_max = self.T_Room_optimization_6R2C_Q_max[building].to_numpy()
        
        font_size = 14        
        # week plot for heating
        week_number = 5
        font_size = 14
        x_axis = np.arange(24 * 7 * week_number, 24 * 7 * (week_number + 1))
        
        fig2, (ax3,ax4,) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 11))
        ax3.plot(x_axis, ideal_demand_ida[x_axis], label="IDA ICE with ideal heating", color=sns.color_palette()[1], linestyle="-", marker="x", linewidth=1)
        ax3.plot(x_axis, opt_demand_5R1C[x_axis], label="5R1C", color=sns.color_palette()[0], linewidth=2)

        ax3.set_ylabel("heat demand (kWh)", fontsize=font_size)
        ax3.legend(fontsize=font_size)        
        ax4.plot(x_axis, temp_ideal_ida[x_axis], label="IDA ICE with ideal heating", color=sns.color_palette()[1], linestyle="-", marker="x", linewidth=1)
        ax4.plot(x_axis, temp_5R1C[x_axis], label="5R1C = set temperature", color=sns.color_palette()[0], linewidth=2)

        ax4.set_ylabel("indoor temperature (°C)", fontsize=font_size)
        ax4.set_xlim(x_axis[0], x_axis[-1])
        ax4.set_xticks(np.arange(x_axis[0], x_axis[-1], 24) + 12)
        ax4.set_ylim(20,23)
        ticks = np.array(["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
        ax4.set_xticklabels(ticks)
        ax3.tick_params(axis='y', labelsize=font_size)
        ax4.tick_params(axis='y', labelsize=font_size)
        ax4.tick_params(axis='x', labelsize=font_size)
        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: round(x,1)))

        plt.tight_layout()
        fig2.savefig(self.figure_path / f"Single_building_week_{building}_radiators.svg")
        plt.close()

        # week plot for floor heating
        week_number = 5
        font_size = 14
        x_axis = np.arange(24 * 7 * week_number, 24 * 7 * (week_number + 1))
        
        fig2, (ax3,ax4,) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 11))
        ax3.plot(x_axis, floor_demand_ida[x_axis], label="IDA ICE with floor heating", color=sns.color_palette()[3], linestyle="-", marker="x", linewidth=1)
        ax3.plot(x_axis, opt_demand_5R1C[x_axis], label="5R1C", color=sns.color_palette()[0], linewidth=2)
        ax3.plot(x_axis, opt_demand_6R2C[x_axis], label="6R2C", color=sns.color_palette()[2], linestyle="--")
        ax3.plot(x_axis, opt_demand_6R2C_set_temp[x_axis], label="6R2C with set temperature", color=sns.color_palette()[4], linestyle="--", marker="o", markersize=4, linewidth=1)
        # ax3.plot(x_axis, opt_demand_6R2C_Q_max[x_axis], label="6R2C Q_max", color=sns.color_palette()[5], linestyle="--")


        ax3.set_ylabel("heat demand (kWh)", fontsize=font_size)
        ax3.legend(fontsize=font_size)    
        ax4.plot(x_axis, temp_5R1C[x_axis], label="5R1C = set temperature", color=sns.color_palette()[0], linewidth=2)
        ax4.plot(x_axis, temp_floor_ida[x_axis], label="IDA ICE with floor heating", color=sns.color_palette()[3], linestyle="-", marker="x", linewidth=0.5)
        ax4.plot(x_axis, temp_6R2C[x_axis], label="6R2C with set temperature", color=sns.color_palette()[2],  linestyle="--")
        ax4.plot(x_axis, temp_6R2C_set_temp[x_axis], label="6R2C", color=sns.color_palette()[4],  linestyle="--", marker="o", markersize=4, linewidth=1)

        # ax4.plot(x_axis, temp_6R2C_Q_max[x_axis], label="6R2C Q_max", color=sns.color_palette()[5],  linestyle="--")

        ax4.set_ylabel("indoor temperature (°C)", fontsize=font_size)
        ax4.set_xlim(x_axis[0], x_axis[-1])
        ax4.set_xticks(np.arange(x_axis[0], x_axis[-1], 24) + 12)
        ax4.set_ylim(20,23)
        ticks = np.array(["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
        ax4.set_xticklabels(ticks)
        ax3.tick_params(axis='y', labelsize=font_size)
        ax4.tick_params(axis='y', labelsize=font_size)
        ax4.tick_params(axis='x', labelsize=font_size)
        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: round(x,1)))

        plt.tight_layout()
        fig2.savefig(self.figure_path / f"Single_building_week_{building}_floor_heating.svg")
        plt.close()

    def shifted_electrity_demand(self):
        cop = self.read_COP(table_name=OperationTable.ResultRefHour.value)

        ref_5R1C = self.read_heat_demand(table_name=OperationTable.ResultRefHour.value, prize_scenario="basic", cooling=False) / cop
        ref_IDA_floor = self.read_daniel_heat_demand(price="basic", cooling=False, floor_heating=True) / cop
        ref_IDA_ideal = self.read_daniel_heat_demand(price="basic", cooling=False, floor_heating=False) / cop

        total_5R1C = ref_5R1C.sum(axis=0)
        total_IDA_floor = ref_IDA_floor.sum(axis=0)
        total_IDA_ideal = ref_IDA_ideal.sum(axis=0)
        total_6R2C = self.Q_RoomHeating_reference_6R2C.sum(axis=0)
        total_6R2C_forced = self.Q_RoomHeating_optimization_6R2C_target_temp.sum(axis=0)

        opt_5R1C = self.read_heat_demand(table_name=OperationTable.ResultOptHour.value, prize_scenario="price2", cooling=False) / cop
        opt_IDA_floor = self.read_daniel_heat_demand(price="price2", cooling=False, floor_heating=True) / cop
        opt_IDA_ideal = self.read_daniel_heat_demand(price="price2", cooling=False, floor_heating=False) / cop

        RC_shifted = ref_5R1C - opt_5R1C
        IDA_shifted_floor = ref_IDA_floor - opt_IDA_floor
        IDA_shifted_ideal = ref_IDA_ideal - opt_IDA_ideal
        R62C_shifted = self.Q_RoomHeating_reference_6R2C - self.Q_RoomHeating_optimization_6R2C
        R62C_shifted_forced = self.Q_RoomHeating_optimization_6R2C_target_temp - self.Q_RoomHeating_reference_6R2C

        RC_shifted[RC_shifted<0] = 0
        IDA_shifted_floor[IDA_shifted_floor<0] = 0
        IDA_shifted_ideal[IDA_shifted_ideal<0] = 0
        R62C_shifted[R62C_shifted<0] = 0
        R62C_shifted_forced[R62C_shifted_forced<0] = 0

        RC_shifted_sum = RC_shifted.sum()
        IDA_shifted_floor_sum = IDA_shifted_floor.sum()
        IDA_shifted_ideal_sum = IDA_shifted_ideal.sum()
        R6C2_shifted_sum = R62C_shifted.sum()
        R62C_shifted_forced_sum = R62C_shifted_forced.sum()

        RC_shifted_perc = RC_shifted_sum / total_5R1C
        IDA_shifted_perc_floor = IDA_shifted_floor_sum / total_IDA_floor
        IDA_shifted_perc_ideal = IDA_shifted_ideal_sum / total_IDA_ideal
        R62C_shifted_perc = R6C2_shifted_sum / total_6R2C
        R62C_shifted_forced_perc = R62C_shifted_forced_sum / total_6R2C_forced

        df = pd.concat([RC_shifted_perc, IDA_shifted_perc_ideal, R62C_shifted_perc, IDA_shifted_perc_floor], axis=1).rename(
            columns={0: "5R1C", 3: "IDA ICE floor heating", 2: "6R2C", 1: "IDA ICE ideal heating", }
            ).reset_index()


        final_df = df.melt(id_vars=["index"], var_name="model", value_name="shifted electricity")

        font_size = 14
        fig = plt.figure(figsize=(12,8))
        ax = plt.gca()
        sns.barplot(
            data=final_df,
            x="index",
            y="shifted electricity",
            hue="model"
        )
        ax.tick_params(axis='y', labelsize=font_size)
        ax.tick_params(axis='x', labelsize=font_size)
        ax.legend(fontsize=font_size)
        ax.set_xlabel("")
        ax.set_ylabel("shifted electricity in (%)", fontsize=font_size)
        ax.set_xticklabels([label.get_text().replace("_", " ") for label in ax.get_xticklabels()], rotation=45, ha='center', fontsize=font_size)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        plt.tight_layout()
        fig.savefig(self.figure_path / f"Shifted_electricity_demand_IDA_ICE.svg")
        plt.show()


    def run(self, price_scenarios: list, floor_heating: bool, cooling: bool):
        # self.show_rmse(prizes=price_scenarios, floor_heating=floor_heating, cooling=cooling)
        self.subplots_relative(prizes=price_scenarios, floor_heating=floor_heating, cooling=cooling)
        self.subplots_yearly(prices=price_scenarios, cooling=cooling, floor_heating=floor_heating)
        # if cooling:
        self.plot_relative_cooling_cost_reduction(prices=price_scenarios,
                                                  cooling=cooling)  # no floor heating with cooling

    def compare_daily_heat_demand_peaks(self):
        floor_heating = True
        cooling = False
        cop = self.read_COP(table_name=OperationTable.ResultRefHour.value)

        dfs = []

        opt_5R1C = self.read_heat_demand(table_name=OperationTable.ResultOptHour.value, prize_scenario="price2", cooling=False) / cop
        opt_IDA_floor = self.read_daniel_heat_demand(price="price2", cooling=False, floor_heating=True) / cop
        opt_IDA_ideal = self.read_daniel_heat_demand(price="price2", cooling=False, floor_heating=False) / cop
        #opt_6R2C = self.load_6R2C_results(scenario_id=)
            
        opt_5R1C["type"] = "5R1C"
        opt_IDA_floor["type"] = "IDA ICE floor heating"
        opt_IDA_ideal["type"] = "IDA ICE ideal heating"
        df = pd.concat([opt_5R1C, opt_IDA_floor, opt_IDA_ideal], axis=0)
        #df["price scenario"] = price[-1]
        dfs.append(df)
        big_df = pd.concat(dfs)

        big_df["days"] = (big_df.index-1) // 24 + 1
        maxes = big_df.groupby(["days", "price scenario", "type"]).max().reset_index()
        plot_df = maxes.melt(id_vars=["days", "price scenario", "type"], value_name="peak demand", var_name="building")

        sns.boxplot(
            data=plot_df,
            x="building",
            y="peak demand",
            hue="type"
        )
        plt.suptitle("daily peaks over all price scenario")
        plt.tight_layout()
        plt.savefig(self.figure_path / "daily_peaks_over_all_price_scenarios.png")
        plt.show()

    def show_chosen_Cf_and_Hf_for_6R2C_model(self):
        with open(Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/FLEX/data/output/5R1C_6R2C/")/"Cf_Hf_dict.pkl", 'rb') as f:
            Cf_Hf_dict = pickle.load(f)
        new_dict = {self.building_names[i]: value for i,value in Cf_Hf_dict.items()}
        df = pd.DataFrame.from_dict(
            new_dict,
            orient='index',
            columns=['Cf', 'Hf']
        )
        df["Cf"] = df["Cf"] / 4200
        building_df = self.db.read_dataframe(table_name=OperationScenarioComponent.Building.table_name, column_names=["ID_Building", "type", "Af"])
        building_df["index"] = building_df.loc[:, "ID_Building"].map(self.building_names)
        merged = pd.merge(right=df.reset_index(), left=building_df.loc[:,["index", "Af"]], on="index")
        merged["Cf_liter"] = merged["Cf"] / 4200 / merged["Af"]
        merged["Hf W/m2"] = merged["Hf"] / merged["Af"]


    def main(self):
        # laod 6R2C results
        self.load_6R2C_results(reference=True)
        self.load_6R2C_results(reference=False)

        price_scenarios = ["basic", "price2",] #"price3", "price4"]  # only look at price 2 and basic which is without optim
        # self.show_elec_prices()
        # self.show_heat_demand_for_one_building_in_multiple_scenarios(price_id="price2", building="EZFH_9_B")
        # self.shifted_electrity_demand()
        self.plot_normalized_yearly_heat_demand_floor_ideal_not_optimized()
        # self.plot_relative_cost_reduction_floor_ideal()
        # self.show_plotly_comparison(prices=price_scenarios, cooling=False, floor_heating=True)
        



        # self.indoor_temp_to_csv(cooling=False)
        # self.indoor_temp_to_csv(cooling=True)# was only relevant for Daniel

        # self.compare_daily_heat_demand_peaks()
        # self.show_rmse(prizes=price_scenarios)
        # self.plot_yearly_heat_demand_floor_ideal_not_optimized()

        # run through results: run takes scenarios, floor_heating, cooling as input
        # run it with floor heating and without cooling (not included in floor heating)
        # run it with ideal heating system including end excluding cooling:
        # self.run(price_scenarios, floor_heating=True, cooling=False)
        # self.run(price_scenarios, floor_heating=False, cooling=False)
        # self.run(price_scenarios, floor_heating=False, cooling=True)




if __name__ == "__main__":

    CompareModels("5R1C_validation").main()

    # TODO Cooling demand plot passt nicht
