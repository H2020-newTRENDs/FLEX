from typing import List

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

from basics.db import DB
from config import config, get_config
from models.operation.enums import OperationTable


class CompareModels:
    def __init__(self):
        self.input_path = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\Philipp_5R1C\input_data")


    def read_daniel_heat_demand(self, strat: str):
        filename = f"heating_demand_daniel_{strat}.csv"
        demand_df = pd.read_csv(self.input_path / Path(filename), sep=";").drop(columns=["Unnamed: 0"])
        column_list = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        # rearrange the df:
        df = demand_df[column_list]
        return df

    def read_heat_demand(self, table_name: str, project: str):
        demand = DB(get_config(project)).read_dataframe(table_name=table_name,
                                           column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"])
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        df.columns = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        return df

    def read_indoor_temp(self, table_name: str, project: str):
        temp = DB(get_config(project)).read_dataframe(table_name=table_name,
                                           column_names=["ID_Scenario", "T_Room", "Hour"])
        df = temp.pivot_table(index="Hour", columns="ID_Scenario")
        df.columns = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        return df

    def read_indoor_temp_daniel(self, strat):
        filename = f"indoor_temp_daniel_{strat}.csv"
        temp_df = pd.read_csv(self.input_path / Path(filename), sep=";").drop(columns=["Unnamed: 0"])
        column_list = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        # rearrange the df:
        df = temp_df[column_list]
        return temp_df

    def read_costs(self, table_name: str, project: str):
        cost = DB(get_config(project)).read_dataframe(table_name=table_name,
                                                                column_names=["TotalCost"])
        cost.loc[:, "index"] = np.arange(1, len(cost)+1)
        df = cost.pivot_table(columns="index")
        df.columns = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        return df

    def read_electricity_price(self, project: str) -> np.array:
        price = DB(get_config(project)).read_dataframe(table_name=OperationTable.EnergyPriceProfile.value,
                                       column_names=["electricity_1"]). to_numpy()
        return price

    def read_COP(self, table_name: str, project: str):
        cop = DB(get_config(project)).read_dataframe(table_name=table_name,
                                                                  column_names=["ID_Scenario", "SpaceHeatingHourlyCOP",
                                                                                "Hour"])
        # split the demand into columns for every Scenario ID:
        df = cop.pivot_table(index="Hour", columns="ID_Scenario")
        df.columns = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        return df

    def calculate_costs(self, table_name: str, project: str):
        price_profile = self.read_electricity_price("Flex_optimized").squeeze()
        heat_demand = self.read_heat_demand(table_name, project)
        COP_HP = self.read_COP(table_name, project)
        electricity = (heat_demand / COP_HP).reset_index(drop=True)

        hourly_cost = electricity.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def calculate_costs_daniel(self, table_name: str, strat: str):
        price_profile = self.read_electricity_price("Flex_optimized").squeeze()
        COP_HP = self.read_COP(table_name, "Flex_optimized")
        heat_demand = self.read_daniel_heat_demand(strat)
        electricity = (heat_demand / COP_HP).reset_index(drop=True)

        hourly_cost = electricity.mul(pd.Series(price_profile), axis=0)
        total_cost = hourly_cost.sum(axis=0)
        return pd.DataFrame(total_cost).transpose()

    def compare_hourly_profile(self, demand_list: list, demand_names: list, title: str):
        # compare hourly demand
        column_number = len(list(demand_list[0].columns))
        x_axis = np.arange(8760)
        colors = ["blue", "red", "green"]

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
        fig2 = px.bar(
            data_frame=total_df,
            barmode="group",
        )
        fig2.update_layout(title_text=title)
        fig2.show()

    def df_to_csv(self, df, path):
        column_names = ["EZFH_5_B", "EZFH_5_S", "EZFH_9_B", "EZFH_1_B", "EZFH_1_S"]
        df.columns = column_names
        df.to_csv(path, sep=";")

    def main(self):
        # heat demand
        heat_demand_daniel_steady = self.read_daniel_heat_demand("steady")
        heat_demand_daniel_var = self.read_daniel_heat_demand("optimized")
        heat_demand_opt = self.read_heat_demand(OperationTable.ResultOptHour.value, "Flex_optimized")
        heat_demand_ref = self.read_heat_demand(OperationTable.ResultRefHour.value, "Flex_steady")
        # temperature
        temperature_daniel_steady = self.read_indoor_temp_daniel("steady")
        indoor_temp_daniel_var = self.read_indoor_temp_daniel("optimized")
        indoor_temp_opt = self.read_indoor_temp(OperationTable.ResultOptHour.value, "Flex_optimized")
        indoor_temp_ref = self.read_indoor_temp(OperationTable.ResultRefHour.value, "Flex_steady")
        # cost
        costs_ref = self.calculate_costs(table_name=OperationTable.ResultRefHour.value, project="Flex_optimized")
        cost_daniel_ref = self.calculate_costs_daniel(OperationTable.ResultOptHour.value, "steady")
        costs_opt = self.calculate_costs(table_name=OperationTable.ResultOptHour.value, project="Flex_optimized")
        cost_daniel_opt = self.calculate_costs_daniel(OperationTable.ResultOptHour.value, "optimized")

        self.compare_hourly_profile([heat_demand_daniel_steady, heat_demand_ref], ["IDA ICE", "5R1C"], "heat demand steady price")

        self.compare_hourly_profile([
            indoor_temp_ref,
            temperature_daniel_steady
        ], ["5R1C", "IDA ICE"], "indoor temperature steady price")

        # save indoor temp opt to csv for daniel:
        self.df_to_csv(indoor_temp_opt.copy(), self.input_path.parent / Path("ouput_data/indoor_set_temp.csv"))

        self.compare_hourly_profile([heat_demand_opt, heat_demand_daniel_var],
                                    ["5R1C optimized", "IDA ICE"],
                                    "heat demand variable price")
        self.compare_hourly_profile([indoor_temp_opt, indoor_temp_daniel_var],
                                    ["5R1C optimized", "IDA ICE"],
                                    "indoor temperature variable price")

        self.compare_yearly_value([heat_demand_opt, heat_demand_daniel_var, heat_demand_ref, heat_demand_daniel_steady],
                                  ["5R1C optimized", "IDA ICE optimized", "5R1C", "IDA ICE"],
                                  title="total heat demand")
        self.compare_yearly_value([costs_opt, cost_daniel_opt, costs_ref, cost_daniel_ref],
                                  ["5R1C optimized", "IDA ICE optimized", "5R1C", "IDA ICE"],
                                  "total heating cost")



if __name__ == "__main__":
    CompareModels().main()

    # project names:
    # Flex_steady, Flex_optimized


