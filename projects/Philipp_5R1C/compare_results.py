from typing import List

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

from basics.db import DB
from config import config, get_config
from models.operation.enums import OperationTable


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

    def scenario_id_2_building_name(self, id_scenario: int) -> str:
        building_id = int(
            self.scenario_table.loc[self.scenario_table.loc[:, "ID_Scenario"] == id_scenario, "ID_Building"])
        return f"{self.building_names[building_id]}"

    def grab_scenario_ids_for_price(self, id_price: int) -> list:
        ids = self.scenario_table.loc[self.scenario_table.loc[:, "ID_EnergyPrice"] == id_price, "ID_Scenario"]
        return list(ids)

    def show_elec_prices(self):
        """plot the electricity prices that are used in this paper and print their mean and std"""
        # plot the prices in one figure:
        price_table = self.db.read_dataframe(table_name=OperationTable.EnergyPriceProfile.value)
        price_ids = [1, 2, 3, 4]
        fig = plt.figure()
        ax = plt.gca()
        colors = [self.blue, self.orange, self.green, self.red]
        for price_id in price_ids:
            column_name = f"electricity_{price_id}"
            price = price_table.loc[:, column_name].to_numpy() * 1000  # cent/kWh
            ax.plot(np.arange(8760), price, label=f"Price scenario {price_id}",
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

    def read_daniel_heat_demand(self, strat: str):
        filename = f"heating_demand_daniel_{strat}.csv"
        demand_df = pd.read_csv(self.input_path / Path(filename), sep=";")
        column_list = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        # rearrange the df:
        df = demand_df[column_list]
        return df

    def read_heat_demand(self, table_name: str, prize_scenario: str):
        scenario_id = self.grab_scenario_ids_for_price(int(prize_scenario[-1]))
        demand = self.db.read_dataframe(table_name=table_name,
                                        column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"],
                                        )
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
        return df

    def read_indoor_temp(self, table_name: str, prize_scenario: str):
        scenario_id = self.grab_scenario_ids_for_price(int(prize_scenario[-1]))
        temp = self.db.read_dataframe(table_name=table_name,
                                      column_names=["ID_Scenario", "T_Room", "Hour"])
        df = temp.pivot_table(index="Hour", columns="ID_Scenario")
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
        return df

    def read_indoor_temp_daniel(self, strat):
        filename = f"indoor_temp_daniel_{strat}.csv"
        temp_df = pd.read_csv(self.input_path / Path(filename), sep=";")
        column_list = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        # rearrange the df:
        df = temp_df[column_list]
        return df

    def read_costs(self, table_name: str):
        cost = self.db.read_dataframe(table_name=table_name,
                                      column_names=["TotalCost"])
        cost.loc[:, "index"] = np.arange(1, len(cost) + 1)
        df = cost.pivot_table(columns="index")
        df.columns = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        return df

    def read_electricity_price(self, price_id: str) -> np.array:
        number = int(price_id[-1])
        price = self.db.read_dataframe(table_name=OperationTable.EnergyPriceProfile.value,
                                       column_names=[f"electricity_{number}"]).to_numpy()
        return price

    def read_COP(self, table_name: str):
        # cop is the same in alle price scenarios
        scenario_id = self.grab_scenario_ids_for_price(1)
        cop = self.db.read_dataframe(table_name=table_name,
                                     column_names=["ID_Scenario", "SpaceHeatingHourlyCOP",
                                                   "Hour"])
        # split the demand into columns for every Scenario ID:
        df = cop.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
        return df

    def calculate_costs(self, table_name: str, price_id: str):
        price_profile = self.read_electricity_price(price_id).squeeze()
        heat_demand = self.read_heat_demand(table_name, price_id)
        COP_HP = self.read_COP(table_name)
        electricity = (heat_demand / COP_HP).reset_index(drop=True)

        hourly_cost = electricity.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_costs_daniel_opt(self, table_name: str, price_id: str):
        price_profile = self.read_electricity_price(price_id).squeeze()
        COP_HP = self.read_COP(table_name)
        heat_demand = self.read_daniel_heat_demand(price_id)
        electricity = (heat_demand / COP_HP).reset_index(drop=True)

        hourly_cost = electricity.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_costs_daniel_ref(self, table_name: str, price_id: str):
        price_profile = self.read_electricity_price(price_id).squeeze()
        COP_HP = self.read_COP(table_name)
        heat_demand = self.read_daniel_heat_demand("price1")  # because here the indoor set temp is steady
        electricity = (heat_demand / COP_HP).reset_index(drop=True)

        hourly_cost = electricity.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost.sum(axis=0)
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
        fig.update_layout(title_text=title)
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

        total_df.loc[:, "5R1C"] = (total_df.loc[:, "5R1C optimized"] - total_df.loc[:, "5R1C"]) / total_df.loc[:, "5R1C"] * 100
        total_df.loc[:, "IDA ICE"] = (total_df.loc[:, "IDA ICE optimized"] - total_df.loc[:, "IDA ICE"]) / total_df.loc[:, "IDA ICE"] * 100
        total_df = total_df.drop(columns=["5R1C optimized", "IDA ICE optimized"])

        ax = axes.flatten()[ax_number]
        # optimization is always measured against the reference case (reference = 100%)
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



    def subplots_yearly(self):
        plt.style.use("seaborn-paper")
        prizes = ["price1", "price2", "price3", "price4"]

        # heat demand:
        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
        # ref heat demand is always the same
        heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value, "price1")
        # ref IDA ICE is the one where the indoor set temp is not changed (price1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand("price1")
        ax_number = 0
        for price in prizes:
            heat_demand_opt = self.read_heat_demand(OperationTable.ResultOptHour.value, price)
            heat_demand_daniel = self.read_daniel_heat_demand(price)
            by_label = self.yearly_comparison(fig, axes,
                                              [heat_demand_opt, heat_demand_ref, heat_demand_daniel,
                                               heat_demand_IDA_ref],
                                              ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                              y_label="heat demand (kWh)", ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.97), loc="lower center",
                   borderaxespad=0, ncol=4, bbox_transform=fig.transFigure)
        fig.savefig(self.figure_path / Path("total_heat_demand.svg"))
        plt.show()

        # cost
        plt.style.use("seaborn-paper")
        # create figure
        fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
        ax_number = 0
        for price in prizes:
            costs_ref = self.calculate_costs(table_name=OperationTable.ResultRefHour.value, price_id=price)
            costs_opt = self.calculate_costs(table_name=OperationTable.ResultOptHour.value, price_id=price)
            cost_daniel_ref = self.calculate_costs_daniel_ref(OperationTable.ResultRefHour.value, price)
            cost_daniel_opt = self.calculate_costs_daniel_opt(OperationTable.ResultOptHour.value, price)

            by_label = self.yearly_comparison(fig2, axes2,
                                              [costs_opt, costs_ref, cost_daniel_opt, cost_daniel_ref],
                                              ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                              y_label="heating cost (€)", ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig2.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.97), loc="lower center",
                   borderaxespad=0, ncol=4, bbox_transform=fig2.transFigure)
        fig2.savefig(self.figure_path / Path("total_heating_cost.svg"))
        plt.show()

    def subplots_relative(self):
        plt.style.use("seaborn-paper")
        prizes = ["price1", "price2", "price3", "price4"]

        # heat demand:
        # create figure
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
        # ref heat demand is always the same
        heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value, "price1")
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand("price1")
        ax_number = 0
        for price in prizes:
            heat_demand_opt = self.read_heat_demand(OperationTable.ResultOptHour.value, price)
            heat_demand_daniel = self.read_daniel_heat_demand(price)
            by_label = self.relative_yearly_comparison(fig, axes,
                                              [heat_demand_opt, heat_demand_ref, heat_demand_daniel,
                                               heat_demand_IDA_ref],
                                              ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                              y_label="relative change in heat demand (%)", ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.97), loc="lower center",
                   borderaxespad=0, ncol=4, bbox_transform=fig.transFigure)
        fig.savefig(self.figure_path / Path("relative_change_in_heat_demand.svg"))
        plt.show()

        # cost
        plt.style.use("seaborn-paper")
        # create figure
        fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)
        ax_number = 0
        for price in prizes:
            costs_ref = self.calculate_costs(table_name=OperationTable.ResultRefHour.value, price_id=price)
            costs_opt = self.calculate_costs(table_name=OperationTable.ResultOptHour.value, price_id=price)
            cost_daniel_ref = self.calculate_costs_daniel_ref(OperationTable.ResultRefHour.value, price)
            cost_daniel_opt = self.calculate_costs_daniel_opt(OperationTable.ResultOptHour.value, price)

            by_label = self.relative_yearly_comparison(fig2, axes2,
                                              [costs_opt, costs_ref, cost_daniel_opt, cost_daniel_ref],
                                              ["5R1C optimized", "5R1C", "IDA ICE optimized", "IDA ICE"],
                                              y_label="relative change in heating cost (%)", ax_number=ax_number)
            ax_number += 1

        # plot legend in the top middle of the figure
        fig2.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 0.97), loc="lower center",
                    borderaxespad=0, ncol=4, bbox_transform=fig2.transFigure)
        fig2.savefig(self.figure_path / Path("relative_change_in_heating_cost.svg"))
        plt.show()

    def indoor_temp_to_csv(self):
        for price_id in [1, 2, 3, 4]:  # price 1 is linear
            scenario_ids = self.grab_scenario_ids_for_price(price_id)
            temp_df = self.db.read_dataframe(table_name=OperationTable.ResultOptHour.value,
                                             column_names=["ID_Scenario", "Hour"], filter={"ID_Scenario": 1})
            temp_df_cooling = temp_df.copy()
            for scen_id in scenario_ids:
                indoor_temp_opt = self.db.read_dataframe(
                    table_name=OperationTable.ResultOptHour.value,
                    column_names=["T_Room"], filter={"ID_Scenario": scen_id}).rename(
                    columns={"T_Room": self.scenario_id_2_building_name(scen_id)})
                cooling_id = int(self.scenario_table.loc[self.scenario_table.loc[:, "ID_Scenario"] == scen_id,
                                                         "ID_SpaceCoolingTechnology"])
                if cooling_id == 1:
                    temp_df = pd.concat([temp_df, indoor_temp_opt], axis=1)
                else:
                    temp_df_cooling = pd.concat([temp_df_cooling, indoor_temp_opt], axis=1)

            # save indoor temp opt to csv for daniel:
            temp_df.to_csv(self.input_path.parent / Path(f"output/indoor_set_temp_price{price_id}.csv"), sep=";",
                           index=False)
            temp_df_cooling.to_csv(self.input_path.parent / Path(f"output/indoor_set_temp_price{price_id}_cooling.csv"),
                                   sep=";", index=False)

            del temp_df

    def show_plotly_comparison(self):
        # ref heat demand is always the same
        heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value, "price1")
        # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
        heat_demand_IDA_ref = self.read_daniel_heat_demand("price1")
        temperature_daniel_ref = self.read_indoor_temp_daniel("price1")
        prizes = ["price1", "price2", "price3", "price4"]
        for price in prizes:
            # heat demand
            heat_demand_opt = self.read_heat_demand(OperationTable.ResultOptHour.value, price)
            heat_demand_daniel = self.read_daniel_heat_demand(price)

            # temperature

            indoor_temp_daniel_opt = self.read_indoor_temp_daniel(price)
            indoor_temp_opt = self.read_indoor_temp(OperationTable.ResultOptHour.value, price)
            indoor_temp_ref = self.read_indoor_temp(OperationTable.ResultRefHour.value, price)

            self.compare_hourly_profile([heat_demand_IDA_ref, heat_demand_ref, heat_demand_daniel, heat_demand_opt],
                                        ["IDA ICE", "5R1C", "IDA ICE optimized", "5R1C optimized"],
                                        f"heat demand {price} ")

            self.compare_hourly_profile(
                [indoor_temp_ref, temperature_daniel_ref, indoor_temp_daniel_opt, indoor_temp_opt],
                ["5R1C", "IDA ICE", "IDA ICE optimized", "5R1C optimized"],
                f"indoor temperature {price}")

    def calculate_RMSE(self, profile_IDA_ICE: np.array, profile_5R1C: np.array) -> float:
        MSE = np.square(np.subtract(profile_IDA_ICE, profile_5R1C)).mean()
        RMSE = math.sqrt(MSE)
        return RMSE

    def show_rmse(self):
        plt.style.use("seaborn-paper")
        fig_temp, axes_temp = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True, sharey=True)
        fig_heat, axes_heat = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True, sharey=True)
        prizes = ["price1", "price2", "price3", "price4"]
        for i, price in enumerate(prizes):
            # ref heat demand is always the same
            heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value, price)
            indoor_temp_ref = self.read_indoor_temp(OperationTable.ResultRefHour.value, price)
            # ref IDA ICE is the one where the indoor set temp is not changed (price_1)
            heat_demand_IDA_ref = self.read_daniel_heat_demand(price)
            temperature_daniel_ref = self.read_indoor_temp_daniel(price)
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
            ax_temp.set_title(f"price {price[-1]}")

            ax_heat.bar(building_names, rmse_heat_list)
            ax_heat.set_ylabel("RMSE heat demand")
            ax_heat.set_title(f"price {price[-1]}")

        fig_temp.tight_layout()
        fig_heat.tight_layout()
        fig_temp.savefig(self.figure_path / "RMSE_temperature.svg")
        fig_heat.savefig(self.figure_path / "RMSE_heat_demand.svg")
        fig_temp.show()
        fig_heat.show()

    def main(self):
        self.indoor_temp_to_csv()
        # self.show_rmse()
        # self.subplots_relative()
        # self.subplots_yearly()
        # self.show_plotly_comparison()



if __name__ == "__main__":
    CompareModels("5R1C_validation").show_elec_prices()
    CompareModels("5R1C_validation").main()
