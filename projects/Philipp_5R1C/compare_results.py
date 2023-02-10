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
import seaborn as sns
import math
import re

from basics.db import DB
from config import config, get_config
from models.operation.enums import OperationTable

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CompareModels:
    def __init__(self, db_name):
        self.project_name = db_name
        self.input_path = get_config(self.project_name).project_root / Path(f"projects/{self.project_name}/input")
        self.figure_path = get_config(self.project_name).fig
        self.db = DB(get_config(self.project_name))
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

    def check_price_scenario_name(self, prize_scenario: str):
        if "cooling" in prize_scenario:
            return prize_scenario.replace("_cooling", "")

    def scenario_id_2_building_name(self, id_scenario: int) -> str:
        building_id = int(
            self.scenario_table.loc[self.scenario_table.loc[:, "ID_Scenario"] == id_scenario, "ID_Building"])
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
            cooling_id = 2
        else:
            cooling_id = 1
        ids = self.scenario_table.loc[(self.scenario_table.loc[:, "ID_EnergyPrice"] == id_price) &
                                      (self.scenario_table.loc[:, "ID_SpaceCoolingTechnology"] == cooling_id),
                                      "ID_Scenario"]
        return list(ids)

    def show_elec_prices(self):
        """plot the electricity prices that are used in this paper and print their mean and std"""
        # plot the prices in one figure:
        price_table = self.db.read_dataframe(table_name=OperationTable.EnergyPriceProfile.value)
        price_ids = [2, 3, 4]
        fig = plt.figure()
        ax = plt.gca()
        colors = [self.blue, self.orange, self.green, self.red]
        for price_id in price_ids:
            column_name = f"electricity_{price_id}"
            price = price_table.loc[:, column_name].to_numpy() * 1000  # cent/kWh
            ax.plot(np.arange(8760), price, label=f"Price scenario {price_id - 1}",
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
        fig.show()

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
        scenario_id = self.grab_scenario_ids_for_price(int(re.findall(r"\d", prize_scenario)[0]), cooling)
        demand = self.db.read_dataframe(table_name=table_name,
                                        column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"],
                                        )
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
        return self.reorder_table(df)

    def read_cooling_demand(self, table_name: str, prize_scenario: str, cooling: bool) -> pd.DataFrame:
        scenario_id = self.grab_scenario_ids_for_price(int(re.findall(r"\d", prize_scenario)[0]), cooling)
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
        scenario_id = self.grab_scenario_ids_for_price(int(re.findall(r"\d", prize_scenario)[0]), cooling)
        temp = self.db.read_dataframe(table_name=table_name,
                                      column_names=["ID_Scenario", "T_Room", "Hour"])
        df = temp.pivot_table(index="Hour", columns="ID_Scenario")
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
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
        number = int(price_id[-1])
        price = self.db.read_dataframe(table_name=OperationTable.EnergyPriceProfile.value,
                                       column_names=[f"electricity_{number}"]).to_numpy()
        return price

    def read_COP(self, table_name: str) -> pd.DataFrame:
        # cop is the same in alle price scenarios
        scenario_id = self.grab_scenario_ids_for_price(id_price=1, cooling=False)  # cooling not needed for heating COP
        cop = self.db.read_dataframe(table_name=table_name,
                                     column_names=["ID_Scenario", "SpaceHeatingHourlyCOP",
                                                   "Hour"])
        # split the demand into columns for every Scenario ID:
        df = cop.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
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
        cool_demand = self.read_cooling_demand(table_name, price_id, cooling)
        COP_HP = self.read_COP(table_name)
        COP_AC = self.read_COP_AC(table_name)
        electricity_heat = (heat_demand / COP_HP).reset_index(drop=True)
        electricity_cool = (cool_demand / COP_AC).reset_index(drop=True)

        hourly_cost_heating = electricity_heat.mul(pd.Series(price_profile), axis=0)
        hourly_cost_cooling = electricity_cool.mul(pd.Series(price_profile), axis=0)

        hourly_cost = hourly_cost_heating + hourly_cost_cooling
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

    def calculate_costs_daniel_opt(self, table_name: str, price_id: str, cooling: bool,
                                   floor_heating: bool) -> pd.DataFrame:
        price_profile = self.read_electricity_price(price_id).squeeze()
        COP_HP = self.read_COP(table_name)
        heat_demand = self.read_daniel_heat_demand(price=price_id, floor_heating=floor_heating, cooling=cooling)
        electricity_heat = (heat_demand / COP_HP).reset_index(drop=True)

        COP_AC = self.read_COP_AC(table_name)
        cooling_demand = self.read_daniel_cooling_demand(price=price_id)
        electricity_cooling = (cooling_demand / COP_AC).reset_index(drop=True)

        hourly_cost_heating = electricity_heat.mul(pd.Series(price_profile), axis=0)
        hourly_cost_cooling = electricity_cooling.mul(pd.Series(price_profile), axis=0)
        hourly_cost = hourly_cost_heating + hourly_cost_cooling
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
        heat_demand = self.read_daniel_heat_demand(price="price1", cooling=cooling,
                                                   floor_heating=floor_heating)  # because here the indoor set temp is steady
        electricity_heat = (heat_demand / COP_HP).reset_index(drop=True)

        COP_AC = self.read_COP_AC(table_name)
        cooling_demand = self.read_daniel_cooling_demand(price="price1")
        electricity_cooling = (cooling_demand / COP_AC).reset_index(drop=True)

        hourly_cost_heating = electricity_heat.mul(pd.Series(price_profile), axis=0)
        hourly_cost_cooling = electricity_cooling.mul(pd.Series(price_profile), axis=0)
        hourly_cost = hourly_cost_heating + hourly_cost_cooling
        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_cooling_costs_daniel_ref(self, table_name: str, price_id: str) -> pd.DataFrame:
        price_profile = self.read_electricity_price(price_id).squeeze()

        COP_AC = self.read_COP_AC(table_name)
        cooling_demand = self.read_daniel_cooling_demand(price="price1")
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
                          figure, axes,
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

    def relative_yearly_comparison_floor_ideal(self,
                                               figure, axes,
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

        total_df.loc[:, "5R1C"] = (total_df.loc[:, "5R1C optimized"] - total_df.loc[:, "5R1C"]) / total_df.loc[:,
                                                                                                  "5R1C"] * 100
        total_df.loc[:, "IDA ICE"] = (total_df.loc[:, "IDA ICE optimized"] - total_df.loc[:, "IDA ICE"]) / total_df.loc[
                                                                                                           :,
                                                                                                           "IDA ICE"] * 100
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
        price_numbers = [1, 1, 2, 2, 3, 3]

        price_number = price_numbers[ax_number]
        ax.set_title(f"price {price_number}")

        return by_label

    def relative_yearly_comparison(self,
                                   figure, axes,
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

        total_df.loc[:, "5R1C"] = (total_df.loc[:, "5R1C optimized"] - total_df.loc[:, "5R1C"]) / total_df.loc[:,
                                                                                                  "5R1C"] * 100
        total_df.loc[:, "IDA ICE"] = (total_df.loc[:, "IDA ICE optimized"] - total_df.loc[:, "IDA ICE"]) / total_df.loc[
                                                                                                           :,
                                                                                                           "IDA ICE"] * 100
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
                                                prize_scenario="price1",
                                                cooling=cooling)
        # ref IDA ICE is the one where the indoor set temp is not changed (price1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand(price="price1",
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
        plt.show()

    def plot_yearly_cooling_demand(self, prices: list):
        plt.style.use("seaborn-paper")

        # cooling demand:
        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        # ref heat demand is always the same
        cooling_demand_ref = self.read_cooling_demand(table_name=OperationTable.ResultRefHour.value,
                                                      prize_scenario="price1",
                                                      cooling=True)
        # ref IDA ICE is the one where the indoor set temp is not changed (price1)
        cooling_demand_IDA_ref = self.read_daniel_cooling_demand(price="price1")
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
        plt.show()

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
        plt.show()

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
                                                prize_scenario="price1",
                                                cooling=cooling)
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand(price="price1",
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
        plt.show()

    def plot_relative_cooling_demand(self, prices: list):
        plt.style.use("seaborn-paper")

        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 7), sharex=True)
        # ref heat demand is always the same
        cooling_demand_ref = self.read_cooling_demand(table_name=OperationTable.ResultRefHour.value,
                                                      prize_scenario="price1",
                                                      cooling=True)
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        cooling_demand_IDA_ref = self.read_daniel_cooling_demand(price="price1")
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
        plt.show()

    def plot_relative_cost_reduction_floor_ideal(self, prices):
        """ plot the relative cost reduction of ideal heating together with floor heating in comparison"""
        plt.style.use("seaborn-paper")
        # create figure
        fig2, axes2 = plt.subplots(nrows=3, ncols=2, figsize=(12, 8.5), sharex=True, sharey=True)
        counter = 0
        for column, floor_heating in enumerate([False, True]):
            for price in prices:
                if price == "price1":
                    continue
                ax_numbers = [0, 2, 4, 1, 3, 5]
                ax_number = ax_numbers[counter]

                costs_ref = self.calculate_costs(table_name=OperationTable.ResultRefHour.value,
                                                 price_id=price,
                                                 cooling=False)
                costs_opt = self.calculate_costs(table_name=OperationTable.ResultOptHour.value,
                                                 price_id=price,
                                                 cooling=False)

                cost_daniel_ref = self.calculate_costs_daniel_ref(table_name=OperationTable.ResultRefHour.value,
                                                                  price_id=price,
                                                                  cooling=False,
                                                                  floor_heating=floor_heating)
                cost_daniel_opt = self.calculate_costs_daniel_opt(table_name=OperationTable.ResultOptHour.value,
                                                                  price_id=price,
                                                                  cooling=False,
                                                                  floor_heating=floor_heating)

                by_label = self.relative_yearly_comparison_floor_ideal(fig2, axes2,
                                                                       [costs_opt, costs_ref, cost_daniel_opt,
                                                                        cost_daniel_ref],
                                                                       ["5R1C optimized", "5R1C", "IDA ICE optimized",
                                                                        "IDA ICE"],
                                                                       y_label=f"change in heating cost (%)",
                                                                       ax_number=ax_number)
                counter += 1

        # plot legend in the top middle of the figure
        fig2.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.92), loc="lower center",
                    borderaxespad=0, ncol=4, bbox_transform=fig2.transFigure)
        for ax in [fig2.axes[4], fig2.axes[5]]:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
        fig2.suptitle(f"Left: ideal heating, Right: floor heating")
        fig2.savefig(
            self.figure_path / Path(f"cost_change.svg"))
        plt.show()

    def plot_relative_cost_reduction(self, prices: list, floor_heating: bool, cooling: bool):

        # cost
        plt.style.use("seaborn-paper")
        # create figure
        fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(5, 8.5), sharex=True, sharey=True)
        ax_number = 0
        for price in prices:
            if price == "price1":
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
        fig2.suptitle(f"relative_change_in_heating{ac} with {system}_cost".replace('_', ' '))
        fig2.savefig(
            self.figure_path / Path(f"relative_change_in_heating{ac}_with_{system.replace(' ', '_')}_cost.svg"))
        plt.show()

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
        plt.show()

    def subplots_relative(self, prizes: list, floor_heating: bool, cooling: bool):
        """ compares the relative change in heat demand between the reference case where no load is shifted
        and the optimized case with load shifting in the 5R1C model and the IDA ICE model.
        The relative change in heat and cooling demand as well as price is compared"""
        # self.plot_relative_heat_demand(prices=prizes, floor_heating=floor_heating, cooling=cooling)
        if cooling:
            self.plot_relative_cooling_demand(prices=prizes)  # cooling scenario is only without floor heating

        self.plot_relative_cost_reduction(prices=prizes, floor_heating=floor_heating, cooling=cooling)

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

    def show_plotly_comparison(self, prices: list, cooling: bool, floor_heating: bool):
        # ref heat demand is always the same
        heat_demand_ref = self.read_heat_demand(table_name=OperationTable.ResultRefHour.value,
                                                prize_scenario="price1",
                                                cooling=cooling)
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand(price="price1", cooling=cooling, floor_heating=floor_heating)
        temperature_daniel_ref = self.read_indoor_temp_daniel(price="price1", cooling=cooling,
                                                              floor_heating=floor_heating)
        cooling_demand_daniel_ref = self.read_daniel_cooling_demand(price="price1")
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

        plt.style.use("seaborn-paper")
        floor_heating_df = self.read_daniel_heat_demand(price=price_id, cooling=False, floor_heating=True)
        floor_demand_ida = floor_heating_df[building].to_numpy() / 1_000  # kWh

        ideal_demand_ida = self.read_daniel_heat_demand(price=price_id, cooling=False, floor_heating=False)[
                               building].to_numpy() / 1_000  # kWh

        opt_demand_5R1C = self.read_heat_demand(table_name=OperationTable.ResultOptHour.value,
                                                prize_scenario=price_id,
                                                cooling=False)[building].to_numpy() / 1_000  # kWh
        elec_price = self.read_electricity_price(price_id=price_id) * 1000  # cent/kWh

        temp_floor_ida = self.read_indoor_temp_daniel(price=price_id, cooling=False, floor_heating=True)[
            building].to_numpy()
        temp_ideal_ida = self.read_indoor_temp_daniel(price=price_id, cooling=False, floor_heating=False)[
            building].to_numpy()
        temp_5R1C = self.read_indoor_temp(table_name=OperationTable.ResultOptHour.value,
                                          prize_scenario=price_id,
                                          cooling=False)[building].to_numpy()
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 5),
                                       gridspec_kw={'height_ratios': [2, 1]})

        x_axis = np.arange(8760)
        ax1.plot(x_axis, floor_demand_ida, label="IDA ICE with floor heating optimized", color="lawngreen")
        ax1.plot(x_axis, ideal_demand_ida, label="IDA ICE with ideal heating optimized", color="green")
        ax1.plot(x_axis, opt_demand_5R1C, label="5R1C optimized", color="firebrick")
        ax1.set_title(f"heat demand {building} for price {price_id[-1]}")
        ax1.set_ylabel("heat demand (kWh)")
        ax1.legend()

        ax2.plot(x_axis, elec_price, label="electricity price", color="gold")
        ax2.set_ylabel("electricity price (cent/kWh)")
        ax2.set_xlabel("hours")
        ax2.legend()

        plt.tight_layout()
        fig.savefig(self.figure_path / f"Single_building_year_{building}.svg")
        plt.show()

        # week plot
        week_number = 5
        x_axis = np.arange(24 * 7 * week_number, 24 * 7 * (week_number + 1))
        fig2, (ax3, ax4, ax5) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 8))
        ax3.plot(x_axis, floor_demand_ida[x_axis], label="IDA ICE with floor heating optimized", color="lawngreen")
        ax3.plot(x_axis, ideal_demand_ida[x_axis], label="IDA ICE with ideal heating optimized", color="green")
        ax3.plot(x_axis, opt_demand_5R1C[x_axis], label="5R1C optimized", color="firebrick")
        ax3.set_title(f"heat demand {building} for price {price_id[-1]}")
        ax3.set_ylabel("heat demand (kWh)")
        ax3.legend()

        ax4.plot(x_axis, elec_price[x_axis], label="electricity price", color="gold")
        ax4.legend()
        ax5.plot(x_axis, temp_floor_ida[x_axis], label="IDA ICE with floor heating optimized", color="lawngreen")
        ax5.plot(x_axis, temp_ideal_ida[x_axis], label="IDA ICE with ideal heating optimized", color="green")
        ax5.plot(x_axis, temp_5R1C[x_axis], label="5R1C optimized", color="firebrick")
        ax5.set_title(f"indoor temperature {building} for price {price_id[-1]}")
        ax5.set_ylabel("temperature (°C)")
        ax5.legend()

        ax5.set_xlim(x_axis[0], x_axis[-1])
        ax5.set_xticks(np.arange(x_axis[0], x_axis[-1], 24) + 12)
        ticks = np.array(["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
        ax5.set_xticklabels(ticks)


        plt.tight_layout()
        fig2.savefig(self.figure_path / f"Single_building_week_{building}.svg")
        plt.show()

    def run(self, price_scenarios: list, floor_heating: bool, cooling: bool):
        # self.show_rmse(prizes=price_scenarios, floor_heating=floor_heating, cooling=cooling)
        self.subplots_relative(prizes=price_scenarios, floor_heating=floor_heating, cooling=cooling)
        # self.subplots_yearly(prices=price_scenarios, cooling=cooling, floor_heating=floor_heating)
        # self.show_plotly_comparison(prices=price_scenarios, cooling=cooling, floor_heating=floor_heating)
        # if cooling:
        # self.plot_relative_cooling_cost_reduction(prices=price_scenarios,
        #                                           cooling=cooling)  # no floor heating with cooling

    def main(self):
        price_scenarios = ["price1", "price2", "price3", "price4"]
        # self.indoor_temp_to_csv(cooling=False)
        # self.indoor_temp_to_csv(cooling=True)

        self.plot_relative_cost_reduction_floor_ideal(prices=price_scenarios)

        floor_heating = True
        cooling = True
        # run through results: run takes scenarios, floor_heating, cooling as input
        # run it with floor heating and without cooling (not included in floor heating)
        # run it with ideal heating system including end excluding cooling:
        # self.run(price_scenarios, floor_heating=True, cooling=False)
        # self.run(price_scenarios, floor_heating=False, cooling=False)
        # self.run(price_scenarios, floor_heating=False, cooling=True)


if __name__ == "__main__":
    # CompareModels("5R1C_validation").show_elec_prices()
    # CompareModels("5R1C_validation").show_heat_demand_for_one_building_in_multiple_scenarios(price_id="price4",
    #                                                                                          building="EZFH_5_B")
    CompareModels("5R1C_validation").main()
